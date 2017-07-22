#include <cstdlib>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <chrono>
#include <ctime>

#include <unordered_map>
#include <unordered_set>

#include <execinfo.h>
#include <unistd.h>
#include <signal.h>

#include "dynet/training.h"
#include "dynet/dynet.h"
#include "dynet/expr.h"
#include "dynet/nodes.h"
#include "dynet/lstm.h"
#include "dynet/rnn.h"
#include "dynet/io.h"
#include "dynet/dict.h"

#include "impl/oracle.h"
#include "impl/cl-args.h"

dynet::Dict termdict, arcdict, adict, posdict;

volatile bool requested_stop = false;
unsigned ACTION_SIZE = 0;
unsigned VOCAB_SIZE = 0;
unsigned ARC_SIZE = 0;
unsigned POS_SIZE = 0;

using namespace dynet;
using namespace std;

Params params;

unordered_map<unsigned, vector<float>> pretrained;
vector<bool> singletons; // used during training
vector<unsigned> possible_actions;

struct ParserBuilder {

  LSTMBuilder state_lstm;
  LSTMBuilder l2rbuilder;
  LSTMBuilder r2lbuilder;
  LookupParameter p_w; // word embeddings
  LookupParameter p_t; // pretrained word embeddings (not updated)
  LookupParameter p_a; // input action embeddings
  LookupParameter p_r; // relation embeddings
  LookupParameter p_p; // pos tag embeddings
  Parameter p_w2l; // word to LSTM input
  Parameter p_p2l; // POS to LSTM input
  Parameter p_t2l; // pretrained word embeddings to LSTM input
  Parameter p_lb; // LSTM input bias

  Parameter p_sent_start;
  Parameter p_sent_end;

  Parameter p_s_input2att;
  Parameter p_s_h2att;
  Parameter p_s_attbias;
  Parameter p_s_att2attexp;
  Parameter p_s_att2combo;
  
  Parameter p_b_input2att;
  Parameter p_b_h2att;
  Parameter p_b_attbias;
  Parameter p_b_att2attexp;
  Parameter p_b_att2combo;

  Parameter p_h2combo;
  Parameter p_combobias;
  Parameter p_combo2rt;
  Parameter p_rtbias;
  explicit ParserBuilder(Model* model, const unordered_map<unsigned, vector<float>>& pretrained) :
      state_lstm(1, params.state_input_dim ,params.state_hidden_dim, *model),
      l2rbuilder(params.layers, params.bilstm_input_dim, params.bilstm_hidden_dim, *model),
      r2lbuilder(params.layers, params.bilstm_input_dim, params.bilstm_hidden_dim, *model),
      p_w(model->add_lookup_parameters(VOCAB_SIZE, {params.input_dim})),
      p_a(model->add_lookup_parameters(ACTION_SIZE, {params.action_dim})),
      p_r(model->add_lookup_parameters(ACTION_SIZE, {params.rel_dim})),
      p_w2l(model->add_parameters({params.bilstm_input_dim, params.input_dim})),
      p_lb(model->add_parameters({params.bilstm_input_dim})),
      p_sent_start(model->add_parameters({params.bilstm_input_dim})),
      p_sent_end(model->add_parameters({params.bilstm_input_dim})),
      p_s_input2att(model->add_parameters({params.attention_hidden_dim, params.bilstm_hidden_dim*2})),
      p_s_h2att(model->add_parameters({params.attention_hidden_dim, params.state_hidden_dim})),
      p_s_attbias(model->add_parameters({params.attention_hidden_dim})),
      p_s_att2attexp(model->add_parameters({params.attention_hidden_dim})),
      p_s_att2combo(model->add_parameters({params.state_hidden_dim, params.bilstm_hidden_dim*2})),
      p_b_input2att(model->add_parameters({params.attention_hidden_dim, params.bilstm_hidden_dim*2})),
      p_b_h2att(model->add_parameters({params.attention_hidden_dim, params.state_hidden_dim})),
      p_b_attbias(model->add_parameters({params.attention_hidden_dim})),
      p_b_att2attexp(model->add_parameters({params.attention_hidden_dim})),
      p_b_att2combo(model->add_parameters({params.state_hidden_dim, params.bilstm_hidden_dim*2})),
      p_h2combo(model->add_parameters({params.state_hidden_dim, params.state_hidden_dim})),
      p_combobias(model->add_parameters({params.state_hidden_dim})),
      p_combo2rt(model->add_parameters({ACTION_SIZE, params.state_hidden_dim})),
      p_rtbias(model->add_parameters({ACTION_SIZE})){
    if (params.use_pos) {
      p_p = model->add_lookup_parameters(POS_SIZE, {params.pos_dim});
      p_p2l = model->add_parameters({params.bilstm_input_dim, params.pos_dim});
    }
    if (pretrained.size() > 0) {
      p_t = model->add_lookup_parameters(VOCAB_SIZE, {params.pretrained_dim});
      for (auto it : pretrained)
        p_t.initialize(it.first, it.second);
      p_t2l = model->add_parameters({params.bilstm_input_dim, params.pretrained_dim});
    }
  }

static bool IsActionForbidden(const string& a, unsigned bsize, unsigned ssize, const vector<int>& stacki) {
  if (a[1]=='W' && ssize<3) return true;
  if (a[1]=='W') {
        int top=stacki[stacki.size()-1];
        int sec=stacki[stacki.size()-2];
        if (sec>top) return true;
  }

  bool is_shift = (a[0] == 'S' && a[1]=='H');
  bool is_reduce = !is_shift;
  if (is_shift && bsize == 1) return true;
  if (is_reduce && ssize < 3) return true;
  if (bsize == 2 && // ROOT is the only thing remaining on buffer
      ssize > 2 && // there is more than a single element on the stack
      is_shift) return true;
  // only attach left to ROOT
  if (bsize == 1 && ssize == 3 && a[0] == 'R') return true;
  return false;
}

// take a vector of actions and return a parse tree (labeling of every
// word position with its head's position)
static map<int,int> compute_heads(unsigned sent_len, const vector<int>& actions,  map<int,string>* pr = nullptr) {
  map<int,int> heads;
  map<int,string> r;
  map<int,string>& rels = (pr ? *pr : r);
  for(unsigned i=0;i<sent_len;i++) { heads[i]=-1; rels[i]="ERROR"; }
  vector<int> bufferi(sent_len + 1, 0), stacki(1, -999);
  for (unsigned i = 0; i < sent_len; ++i)
    bufferi[sent_len - i] = i;
  bufferi[0] = -999;
  for (auto action: actions) { // loop over transitions for sentence
    const string& actionString=adict.convert(action);
    const char ac = actionString[0];
    const char ac2 = actionString[1];
    if (ac =='S' && ac2=='H') {  // SHIFT
      assert(bufferi.size() > 1); // dummy symbol means > 1 (not >= 1)
      stacki.push_back(bufferi.back());
      bufferi.pop_back();
    } else if (ac=='S' && ac2=='W') { // SWAP
      assert(stacki.size() > 2);
      unsigned ii = 0, jj = 0;
      jj = stacki.back();
      stacki.pop_back();
      ii = stacki.back();
      stacki.pop_back();
      bufferi.push_back(ii);
      stacki.push_back(jj);
    } else { // LEFT or RIGHT
      assert(stacki.size() > 2); // dummy symbol means > 2 (not >= 2)
      assert(ac == 'L' || ac == 'R');
      unsigned depi = 0, headi = 0;
      (ac == 'R' ? depi : headi) = stacki.back();
      stacki.pop_back();
      (ac == 'R' ? headi : depi) = stacki.back();
      stacki.pop_back();
      stacki.push_back(headi);
      heads[depi] = headi;
      rels[depi] = actionString;
    }
  }
  assert(bufferi.size() == 1);
  //assert(stacki.size() == 2);
  return heads;
}

// *** if correct_actions is empty, this runs greedy decoding ***
// returns parse actions for input sentence (in training just returns the reference)
// OOV handling: raw_sent will have the actual words
//               sent will have words replaced by appropriate UNK tokens
// this lets us use pretrained embeddings, when available, for words that were OOV in the
// parser training data
Expression log_prob_parser(ComputationGraph* hg,
		     const parser::Sentence& sent,
                     const vector<int>& correct_actions,
                     double *right,
		     vector<int>* results,
		     bool train,
		     bool sample) {
if(params.debug) cerr<<"sent size: "<<sent.size()<<"\n";
    l2rbuilder.new_graph(*hg);
    r2lbuilder.new_graph(*hg);
    l2rbuilder.start_new_sequence();
    r2lbuilder.start_new_sequence();
    // variables in the computation graph representing the parameters
    Expression lb = parameter(*hg, p_lb);
    Expression w2l = parameter(*hg, p_w2l);
    Expression p2l;
    if (params.use_pos)
      p2l = parameter(*hg, p_p2l);
    Expression t2l;
    if (pretrained.size()>0)
      t2l = parameter(*hg, p_t2l); 
    state_lstm.new_graph(*hg);
    state_lstm.start_new_sequence();
    //state_lstm.start_new_sequence({zeroes(*hg, {params.state_hidden_dim}), state_start});
    
    Expression sent_start = parameter(*hg, p_sent_start);
    Expression sent_end = parameter(*hg, p_sent_end);
    //stack attention
    Expression s_input2att = parameter(*hg, p_s_input2att);
    Expression s_h2att = parameter(*hg, p_s_h2att);
    Expression s_attbias = parameter(*hg, p_s_attbias);
    Expression s_att2attexp = parameter(*hg, p_s_att2attexp);
    Expression s_att2combo = parameter(*hg, p_s_att2combo);

    //buffer attention
    Expression b_input2att = parameter(*hg, p_b_input2att);
    Expression b_h2att = parameter(*hg, p_b_h2att);
    Expression b_attbias = parameter(*hg, p_b_attbias);
    Expression b_att2attexp = parameter(*hg, p_b_att2attexp);
    Expression b_att2combo = parameter(*hg, p_b_att2combo);

    Expression h2combo = parameter(*hg, p_h2combo);
    Expression combobias = parameter(*hg, p_combobias);
    Expression combo2rt = parameter(*hg, p_combo2rt);
    Expression rtbias = parameter(*hg, p_rtbias);
    vector<Expression> input_expr;
    for (unsigned i = 0; i < sent.size(); ++i) {
      int wordid = sent.raw[i];
      if (train && singletons.size() > wordid && singletons[wordid] && rand01() > params.unk_prob)
          wordid = sent.unk[i];
      if (!train)
          wordid = sent.unk[i];

if(params.debug) cerr<<termdict.convert(wordid)<<" "<<posdict.convert(sent.pos[i]) << " " << termdict.convert(sent.lc[i])<<"\n";

      Expression w =lookup(*hg, p_w, wordid);
      if(train) w = dropout(w,params.pdrop);
      vector<Expression> args = {lb, w2l, w}; // learn embeddings
      if (params.use_pos) { // learn POS tag?
        Expression p = lookup(*hg, p_p, sent.pos[i]);
        if(train) p = dropout(p, params.pdrop);
        args.push_back(p2l);
        args.push_back(p);
      }
      if (pretrained.size() > 0 &&  pretrained.count(sent.lc[i])) {  // include fixed pretrained vectors?
        Expression t = const_lookup(*hg, p_t, sent.lc[i]);
        if(train) t = dropout(t, params.pdrop);
        args.push_back(t2l);
        args.push_back(t);
      }
      else{
        args.push_back(t2l);
        args.push_back(zeroes(*hg,{params.pretrained_dim}));
      }
      input_expr.push_back(rectify(affine_transform(args)));
    }
if(params.debug)	std::cerr<<"lookup table ok\n";
    vector<Expression> l2r(sent.size());
    vector<Expression> r2l(sent.size());
    Expression l2r_s = l2rbuilder.add_input(sent_start);
    Expression r2l_e = r2lbuilder.add_input(sent_end);
    for (unsigned i = 0; i < sent.size(); ++i) {
      l2r[i] = l2rbuilder.add_input(input_expr[i]);
      r2l[sent.size() - 1 - i] = r2lbuilder.add_input(input_expr[sent.size()-1-i]);
    }
    Expression l2r_e = l2rbuilder.add_input(sent_end);
    Expression r2l_s = r2lbuilder.add_input(sent_start);
    vector<Expression> input(sent.size());
    for (unsigned i = 0; i < sent.size(); ++i) {
      input[i] = concatenate({l2r[i],r2l[i]});
    }
    Expression sent_start_expr = concatenate({l2r_s, r2l_s});
    Expression sent_end_expr = concatenate({l2r_e, r2l_e});
if(params.debug)	std::cerr<<"bilstm ok\n";
    // dummy symbol to represent the empty buffer
    vector<int> bufferi(sent.size() + 1);
    bufferi[0] = -999;
    vector<int> stacki;
    stacki.push_back(-999);
 
    unsigned stack_buffer_split = 0;
    vector<Expression> log_probs;
    unsigned action_count = 0;  // incremented at each prediction
    
    vector<Expression> l2rhc = l2rbuilder.final_s();
    vector<Expression> r2lhc = r2lbuilder.final_s();

    vector<Expression> initc;
    //for(unsigned i = 0; i < params.layers; i ++){
      initc.push_back(concatenate({l2rhc.back(),r2lhc.back()}));
    //}

    //for(unsigned i = 0; i < params.layers; i ++){
      initc.push_back(zeroes(*hg, {params.bilstm_hidden_dim*2}));
    //}
    state_lstm.start_new_sequence(initc);

    while(stacki.size() > 2 || bufferi.size() > 1) {
if(params.debug)	std::cerr<<"action index " << action_count<<"\n";
     // get list of possible actions for the current parser state
      vector<unsigned> current_valid_actions;
      for (auto a : possible_actions) {
        if (IsActionForbidden(adict.convert((int)a), bufferi.size(), stacki.size(), stacki))
          continue;
        current_valid_actions.push_back(a);
      }
if(params.debug)	std::cerr<<"possible action " << current_valid_actions.size()<<"\n";
      //stack attention

if(params.debug) {
	for(unsigned i = 0; i < current_valid_actions.size(); i ++){
		std::cerr<<current_valid_actions[i]<<":"<<adict.convert(current_valid_actions[i])<<" ";
	}
	std::cerr<<"\n";
}
      Expression prev_h = state_lstm.final_h()[0];
      vector<Expression> s_att;
      vector<Expression> s_input;
      s_att.push_back(tanh(affine_transform({s_attbias, s_input2att, sent_start_expr, s_h2att, prev_h})));
      s_input.push_back(sent_start_expr);
      for(unsigned i = 0; i < stack_buffer_split; i ++){
        s_att.push_back(tanh(affine_transform({s_attbias, s_input2att, input[i], s_h2att, prev_h})));
        s_input.push_back(input[i]);
      }
      Expression s_att_col = transpose(concatenate_cols(s_att));
      Expression s_attexp = softmax(s_att_col * s_att2attexp);

/*if(params.debug){
	vector<float> s_see = as_vector(hg->incremental_forward(s_attexp));
	for(unsigned i = 0; i < s_see.size(); i ++){
		cerr<<s_see[i]<<" ";
	}
	cerr<<"\n";
}*/
      Expression s_input_col = concatenate_cols(s_input);
      Expression s_att_pool = s_input_col * s_attexp;

      vector<Expression> b_att;
      vector<Expression> b_input;
      for(unsigned i = stack_buffer_split; i < sent.size(); i ++){
        b_att.push_back(tanh(affine_transform({b_attbias, b_input2att, input[i], b_h2att, prev_h})));
        b_input.push_back(input[i]);
      }
      b_att.push_back(tanh(affine_transform({b_attbias, b_input2att, sent_end_expr, b_h2att, prev_h})));
      b_input.push_back(sent_end_expr);
      Expression b_att_col = transpose(concatenate_cols(b_att));
      Expression b_attexp = softmax(b_att_col * b_att2attexp);

/*if(params.debug){
        vector<float> b_see = as_vector(hg->incremental_forward(b_attexp));
        for(unsigned i = 0; i < b_see.size(); i ++){
                cerr<<b_see[i]<<" ";
        }
        cerr<<"\n";
}
*/
      Expression b_input_col = concatenate_cols(b_input);
      Expression b_att_pool = b_input_col * b_attexp;

      //
if(params.debug)	std::cerr<<"attention ok\n";
      Expression combo = affine_transform({combobias, h2combo, prev_h, s_att2combo, s_att_pool, b_att2combo, b_att_pool});
      Expression n_combo = rectify(combo);
      Expression rt = affine_transform({rtbias, combo2rt, n_combo});
      if (sample) rt = rt * params.alpha;
if(params.debug)	std::cerr<<"to action layer ok\n";
      Expression rt_s = select_rows(rt, current_valid_actions);
      Expression adiste = log_softmax(rt_s);
      vector<float> adist = as_vector(hg->incremental_forward(adiste));
      double best_score = adist[0];
      unsigned best_a = current_valid_actions[0];
      
      if (sample) {
        double p = rand01();
        assert(current_valid_actions.size() > 0);
        unsigned w = 0;
        for (; w < current_valid_actions.size(); ++w) {
          p -= exp(adist[w]);
          if (p < 0.0) { break; }
        }
        if (w == current_valid_actions.size()) w--;
        best_a = current_valid_actions[w];
      } else { // max
         for (unsigned i = 1; i < current_valid_actions.size(); ++i) {
           if (adist[i] > best_score) {
           	best_score = adist[i];
           	best_a = current_valid_actions[i];
           }
         }
      }

if(params.debug)	std::cerr<<"best action "<<best_a<<" " << adict.convert(best_a)<<"\n";
      unsigned action = best_a;
      if (train) {  // if we have reference actions (for training) use the reference action
        action = correct_actions[action_count];
        if (best_a == action) { (*right)++; }
      }
      ++action_count;
      unsigned w = 0;
      for(; w< current_valid_actions.size(); w++){
        if(current_valid_actions[w] == action) break;
      }
      assert(w!=current_valid_actions.size());
      log_probs.push_back(pick(adiste, w));
      if(results) results->push_back(action);

      // add current action to action LSTM
      Expression actione = lookup(*hg, p_a, action);
      state_lstm.add_input(concatenate({actione, s_att_pool, b_att_pool}));
      // do action
      const string& actionString=adict.convert(action);
      const char ac = actionString[0];
      const char ac2 = actionString[1];

if(params.debug)	std::cerr<<"action lookup ok\n";
      if (ac =='S' && ac2=='H') {  // SHIFT
        stacki.push_back(bufferi.back());
        bufferi.pop_back();
	stack_buffer_split += 1;
      } else if (ac=='S' && ac2=='W'){ //SWAP --- Miguel
        Expression toki, tokj;
        unsigned ii = 0, jj = 0;
        jj=stacki.back();
        stacki.pop_back();
        ii=stacki.back();
        stacki.pop_back();
        bufferi.push_back(ii);
        stacki.push_back(jj);
      } else { // LEFT or RIGHT
        assert(ac == 'L' || ac == 'R');
        unsigned depi = 0, headi = 0;
        (ac == 'R' ? depi : headi) = stacki.back();
        stacki.pop_back();
        (ac == 'R' ? headi : depi) = stacki.back();
        stacki.pop_back();
        // composed = cbias + H * head + D * dep + R * relation
        stacki.push_back(headi);
      }
if(params.debug)	std::cerr<<"state transit ok\n";
    }
    assert(stacki.size() == 2);
    assert(bufferi.size() == 1);
    Expression tot_neglogprob = -sum(log_probs);
    assert(tot_neglogprob.pg != nullptr);
    return tot_neglogprob;
  }
};

void signal_callback_handler(int /* signum */) {
  if (requested_stop) {
    cerr << "\nReceived SIGINT again, quitting.\n";
    _exit(1);
  }
  cerr << "\nReceived SIGINT terminating optimization early...\n";
  requested_stop = true;
}

unsigned compute_correct(const map<int,int>& ref, const map<int,int>& hyp, unsigned len) {
  unsigned res = 0;
  for (unsigned i = 0; i < len; ++i) {
    auto ri = ref.find(i);
    auto hi = hyp.find(i);
    assert(ri != ref.end());
    assert(hi != hyp.end());
    if (ri->second == hi->second) ++res;
  }
  return res;
}
void output_conll(const parser::Sentence& sentence,
                  const map<int,int>& hyp, const map<int,string>& rel_hyp, ofstream *out) {
  for (unsigned i = 0; i < (sentence.size()-1); ++i) {
    auto index = i + 1;
    string wit = sentence.surfaces[i];
    auto pit = posdict.convert(sentence.pos[i]);
    assert(hyp.find(i) != hyp.end());
    auto hyp_head = hyp.find(i)->second + 1;
    if (hyp_head == (int)sentence.size()) hyp_head = 0;
    auto hyp_rel_it = rel_hyp.find(i);
    assert(hyp_rel_it != rel_hyp.end());
    auto hyp_rel = hyp_rel_it->second;
    size_t first_char_in_rel = hyp_rel.find('(') + 1;
    size_t last_char_in_rel = hyp_rel.rfind(')') - 1;
    hyp_rel = hyp_rel.substr(first_char_in_rel, last_char_in_rel - first_char_in_rel + 1);
    if(out) {(*out) << index << '\t'       // 1. ID 
         << wit << '\t'         // 2. FORM
         << "_" << '\t'         // 3. LEMMA 
         << "_" << '\t'         // 4. CPOSTAG 
         << pit<< '\t' // 5. POSTAG
         << "_" << '\t'         // 6. FEATS
         << hyp_head << '\t'    // 7. HEAD
         << hyp_rel << '\t'     // 8. DEPREL
         << "_" << '\t'         // 9. PHEAD
         << "_" << endl;        // 10. PDEPREL
    }
    else{
       cout<< index << '\t'       // 1. ID 
         << wit << '\t'         // 2. FORM
         << "_" << '\t'         // 3. LEMMA 
         << "_" << '\t'         // 4. CPOSTAG 
         << pit<< '\t' // 5. POSTAG
         << "_" << '\t'         // 6. FEATS
         << hyp_head << '\t'    // 7. HEAD
         << hyp_rel << '\t'     // 8. DEPREL
         << "_" << '\t'         // 9. PHEAD
         << "_" << endl;        // 10. PDEPREL
    }
  }
  if(out) (*out) << endl;
  else cout << endl;
}

int main(int argc, char** argv) {
  DynetParams dynet_params = extract_dynet_params(argc, argv);
  dynet_params.random_seed = 1989121013;
  dynet::initialize(dynet_params);
  cerr << "COMMAND:"; 
  for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i) cerr << ' ' << argv[i];
  cerr << endl;
  unsigned status_every_i_iterations = 100;

  get_args(argc, argv, params);

  params.state_input_dim = params.action_dim + params.bilstm_hidden_dim*4;
  params.state_hidden_dim = params.bilstm_hidden_dim * 2;

  cerr << "Unknown word strategy: ";
  if (params.unk_strategy == 1) {
    cerr << "STOCHASTIC REPLACEMENT\n";
  } else {
    abort();
  }
  assert(params.unk_prob >= 0.); assert(params.unk_prob <= 1.);
  ostringstream os;
  os << "parser_" << (params.use_pos ? "pos" : "nopos")
     << '_' << params.layers
     << '_' << params.input_dim
     << '_' << params.action_dim
     << '_' << params.pos_dim
     << '_' << params.rel_dim
     << '_' << params.bilstm_input_dim
     << '_' << params.bilstm_hidden_dim
     << '_' << params.attention_hidden_dim
     << '_' << params.state_hidden_dim
     << "-pid" << getpid() << ".params";

  int best_correct_heads = 0;
  const string fname = os.str();
  cerr << "Writing parameters to file: " << fname << endl;

//=====================================================================================================================

  parser::StandardOracle corpus(&termdict, &adict, &posdict, &arcdict);
  corpus.load_oracle(params.train_file, true);

  if (params.words_file != "") {
    cerr << "Loading from " << params.words_file << " with" << params.pretrained_dim << " dimensions\n";
    ifstream in(params.words_file.c_str());
    string line;
    getline(in, line);
    vector<float> v(params.pretrained_dim, 0);
    string word;
    while (getline(in, line)) {
      istringstream lin(line);
      lin >> word;
      for (unsigned i = 0; i < params.pretrained_dim; ++i) lin >> v[i];
      unsigned id = termdict.convert(word);
      pretrained[id] = v;
    }
  }

  termdict.freeze();
  termdict.set_unk("UNK");
  adict.convert("RIGHT-ARC(preconj)"); // dev data has new possible action
  adict.freeze();
  arcdict.freeze();
  posdict.freeze();

  {  // compute the singletons in the parser's training data
    unordered_map<unsigned, unsigned> counts;
    for (auto& sent : corpus.sents)
      for (auto word : sent.raw) counts[word]++;
    singletons.resize(termdict.size(), false);
    for (auto wc : counts)
      if (wc.second == 1) singletons[wc.first] = true;
  }

  ARC_SIZE = arcdict.size();
  POS_SIZE = posdict.size();
  VOCAB_SIZE = termdict.size();
  ACTION_SIZE = adict.size();

  
  for(unsigned i = 0; i < adict.size(); ++i) possible_actions.push_back(i);

  cerr<<"action:\n";
  for(unsigned i = 0; i < adict.size(); i ++){
    cerr<<i<<":"<<adict.convert(i)<<"\n";
  }

  cerr<<"postag:\n";
  for(unsigned i = 0; i < posdict.size(); i ++){
    cerr<<i<<":"<<posdict.convert(i)<<"\n";
  }

  cerr<<"arc-label:\n";
  for(unsigned i = 0; i < arcdict.size(); i ++){
    cerr<<i<<":"<<arcdict.convert(i)<<"\n";
  }

  parser::StandardOracle dev_corpus(&termdict, &adict, &posdict, &arcdict);
  parser::StandardOracle test_corpus(&termdict, &adict, &posdict, &arcdict);
  if(params.dev_file != "") dev_corpus.load_oracle(params.dev_file, false);
  if(params.test_file != "") test_corpus.load_oracle(params.test_file, false);

//==========================================================================================================================
  
  Model model;
  ParserBuilder parser(&model, pretrained);
  if (params.model_file != "") {
    TextFileLoader loader(params.model_file);
    loader.populate(model);
  }
  
  //TRAINING
  if (params.train) {
    signal(SIGINT, signal_callback_handler);

    Trainer* sgd = NULL;
    unsigned method = params.train_methods;
    if(method == 0)
        sgd = new SimpleSGDTrainer(model,0.1, 0.1);
    else if(method == 1)
        sgd = new MomentumSGDTrainer(model,0.01, 0.9, 0.1);
    else if(method == 2){
        sgd = new AdagradTrainer(model);
        sgd->clipping_enabled = false;
    }
    else if(method == 3){
        sgd = new AdamTrainer(model);
        sgd->clipping_enabled = false;
    }

    vector<unsigned> order(corpus.sents.size());
    for (unsigned i = 0; i < corpus.sents.size(); ++i)
      order[i] = i;
    double tot_seen = 0;
    status_every_i_iterations = min(status_every_i_iterations, corpus.size());
    unsigned si = corpus.size();
    cerr << "NUMBER OF TRAINING SENTENCES: " << corpus.size() << endl;
    unsigned trs = 0;
    unsigned words = 0;
    double right = 0;
    double llh = 0;
    bool first = true;
    int iter = -1;
    while(!requested_stop) {
      ++iter;
      auto time_start = chrono::system_clock::now();
      for (unsigned sii = 0; sii < status_every_i_iterations; ++sii) {
           if (si == corpus.size()) {
             si = 0;
             if (first) { first = false; } else { sgd->update_epoch(); }
             cerr << "**SHUFFLE\n";
             random_shuffle(order.begin(), order.end());
           }
           tot_seen += 1;
   	   auto& sentence = corpus.sents[order[si]];
           const vector<int>& actions=corpus.actions[order[si]];

	   ComputationGraph hg;
           Expression nll = parser.log_prob_parser(&hg, sentence, actions, &right, NULL, true, false);

           double lp = as_scalar(hg.incremental_forward(nll));
           if (lp < 0) {
             cerr << "Log prob < 0 on sentence " << order[si] << ": lp=" << lp << endl;
             assert(lp >= 0.0);
           }
           hg.backward(nll);
           sgd->update(1.0);
           llh += lp;
           ++si;
           trs += actions.size();
	   words += sentence.size();
      }
      sgd->status();
      
      auto time_now = chrono::system_clock::now();
      auto dur = chrono::duration_cast<chrono::milliseconds>(time_now - time_start);

      cerr << "update #" << iter << " (epoch " << (tot_seen / corpus.sents.size()) <<")"
           << " per-action-ppl: " << exp(llh / trs)
           << " per-input-ppl: " << exp(llh / words)
           << " per-sent-ppl: " << exp(llh / status_every_i_iterations)
           << " err: " << (trs - right) / trs
           << " [" << dur.count() / (double)status_every_i_iterations << "ms per instance]" << endl;

      llh = trs = right = words = 0;
      static int logc = 0;
      ++logc;

      if (logc % 25 == 1) { // report on dev set
        unsigned dev_size = dev_corpus.size();
	double llh = 0;
        double trs = 0;
        double right = 0;
	double dwords = 0;
        double correct_heads = 0;
        double total_heads = 0;
        auto t_start = chrono::high_resolution_clock::now();
        for (unsigned sii = 0; sii < dev_size; ++sii) {
	   const auto& sentence=dev_corpus.sents[sii];
           const vector<int>& actions=dev_corpus.actions[sii];

           ComputationGraph hg;
	   vector<int> pred;
	   Expression nll = parser.log_prob_parser(&hg, sentence, actions, &right, &pred, false, false);
           double lp = as_scalar(hg.incremental_forward(nll));
           llh += lp;

           map<int,int> ref = parser.compute_heads(sentence.size(), actions, NULL);
           map<int,int> hyp = parser.compute_heads(sentence.size(), pred, NULL);

           correct_heads += compute_correct(ref, hyp, sentence.size() - 1);
           total_heads += sentence.size() - 1;
           trs += actions.size();
	   dwords += sentence.size();
	}
	double err = (trs - right) / trs;
        auto t_end = std::chrono::high_resolution_clock::now();
        cerr << "  **dev (iter=" << iter << " epoch=" << (tot_seen / corpus.size()) << ")\t"
                <<" llh= " << llh
                <<" ppl: " << exp(llh / dwords)
                <<" uas: " << correct_heads / total_heads
                <<" err: " << err
                <<"\t[" << dev_size << " sents in " << chrono::duration<double, milli>(t_end-t_start).count() << " ms]" << endl;

        if (correct_heads > best_correct_heads) {
          best_correct_heads = correct_heads;

      	  ostringstream part_os;
	  part_os << "parser"
     		<< '_' << params.layers
     		<< '_' << params.input_dim
     		<< '_' << params.action_dim
     		<< '_' << params.pos_dim
     		<< '_' << params.rel_dim
     		<< '_' << params.bilstm_input_dim
     		<< '_' << params.bilstm_hidden_dim
     		<< '_' << params.attention_hidden_dim
     		<< "-pid" << getpid()
		<< "-part" << (tot_seen/corpus.size()) << ".params";
	  const string part = part_os.str();
 
	  TextFileSaver saver("model/"+part);
	  saver.save(model);  
        }
      }
    }
    delete sgd;
  } // should do training?
  else{ // do test evaluation
	ofstream out("test.out");
        unsigned test_size = test_corpus.size();

        double llh = 0;
        double trs = 0;
        double right = 0;
        double dwords = 0;
        double correct_heads = 0;
        double total_heads = 0;
        if(params.samples !=0){
                for (unsigned sii = 0; sii < test_size; ++sii) {
                        const auto& sentence=test_corpus.sents[sii];
                        const vector<int>& actions=test_corpus.actions[sii];
                        for (unsigned z = 0; z < params.samples; ++z) {
                                ComputationGraph hg;
                                vector<int> pred;

                                parser.log_prob_parser(&hg,sentence,actions,&right,&pred,false,true);
                                map<int, string> rel_ref, rel_hyp;
                                map<int,int> ref = parser.compute_heads(sentence.size(), actions, &rel_ref);
                                map<int,int> hyp = parser.compute_heads(sentence.size(), pred, &rel_hyp);
                                output_conll(sentence, hyp, rel_hyp, NULL);
                        }
                }
        }
        auto t_start = std::chrono::high_resolution_clock::now();
        for (unsigned sii = 0; sii < test_size; ++sii) {
                const auto& sentence=test_corpus.sents[sii];
                const vector<int>& actions=test_corpus.actions[sii];
                ComputationGraph hg;
                vector<int> pred;
                Expression nll = parser.log_prob_parser(&hg, sentence, actions, &right, &pred, false, false);
                double lp = as_scalar(hg.incremental_forward(nll));
                llh += lp;

                map<int, string> rel_ref, rel_hyp;
                map<int,int> ref = parser.compute_heads(sentence.size(), actions, &rel_ref);
                map<int,int> hyp = parser.compute_heads(sentence.size(), pred, &rel_hyp);
                output_conll(sentence, hyp, rel_hyp, &out);

                correct_heads += compute_correct(ref, hyp, sentence.size() - 1);
                total_heads += sentence.size() - 1;
                trs += actions.size();
                dwords += sentence.size();
        }
        double err = (trs - right) / trs;
        auto t_end = std::chrono::high_resolution_clock::now();
        cerr << "  TEST llh= " << llh
                <<" ppl: " << exp(llh / dwords)
                <<" uas: " << correct_heads / total_heads
                <<" err: " << err
                <<"\t[" << test_size << " sents in " << chrono::duration<double, milli>(t_end-t_start).count() << " ms]" << endl;
  }
}

