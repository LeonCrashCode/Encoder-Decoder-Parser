#include <cstdlib>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <chrono>
#include <ctime>
#include <unordered_set>
#include <unordered_map>

#include <execinfo.h>
#include <unistd.h>
#include <signal.h>

#include "dynet/training.h"
#include "dynet/dynet.h"
#include "dynet/expr.h"
#include "dynet/nodes.h"
#include "dynet/lstm.h"
#include "dynet/rnn.h"
#include "dynet/dict.h"
#include "dynet/cfsm-builder.h"
#include "dynet/io.h"

#include "impl/oracle.h"
#include "impl/cl-args.h"

// dictionaries
dynet::Dict termdict, ntermdict, adict, posdict;

volatile bool requested_stop =false;
unsigned ACTION_SIZE = 0;
unsigned VOCAB_SIZE = 0;
unsigned NT_SIZE = 0;
unsigned POS_SIZE = 0;

std::map<int,int> action2NTindex;  // pass in index of action NT(X), return index of X
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
//    buffer_lstm = new LSTMBuilder(params.layers, LSTM_params.input_dim, HIDDEN_DIM, model);
    if (pretrained.size() > 0) {
      p_t = model->add_lookup_parameters(VOCAB_SIZE, {params.pretrained_dim});
      for (auto it : pretrained)
        p_t.initialize(it.first, it.second);
      p_t2l = model->add_parameters({params.bilstm_input_dim, params.pretrained_dim});
    }
  }

// checks to see if a proposed action is valid in discriminative models
static bool IsActionForbidden_Discriminative(const string& a, char prev_a, unsigned bsize, unsigned ssize, unsigned nopen_parens) {
  bool is_shift = (a[0] == 'S' && a[1]=='H');
  bool is_reduce = (a[0] == 'R' && a[1]=='E');
  bool is_nt = (a[0] == 'N');
  assert(is_shift || is_reduce || is_nt);
  static const unsigned MAX_OPEN_NTS = 100;
  if (is_nt && nopen_parens > MAX_OPEN_NTS) return true;
  if (ssize == 1) {
    if (!is_nt) return true;
    return false;
  }

  // be careful with top-level parens- you can only close them if you
  // have fully processed the buffer
  if (nopen_parens == 1 && bsize > 1) {
    if (is_reduce) return true;
  }

  // you can't reduce after an NT action
  if (is_reduce && prev_a == 'N') return true;
  if (is_nt && bsize == 1) return true;
  if (is_shift && bsize == 1) return true;
  if (is_reduce && ssize < 3) return true;

  // TODO should we control the depth of the parse in some way? i.e., as long as there
  // are items in the buffer, we can do an NT operation, which could cause trouble
  return false;
}


// *** if correct_actions is empty, this runs greedy decoding ***
// returns parse actions for input sentence (in training just returns the reference)
// this lets us use pretrained embeddings, when available, for words that were OOV in the
// parser training data
// set sample=true to sample rather than max
Expression log_prob_parser(ComputationGraph* hg,
                     const parser::Sentence& sent,
                     const vector<int>& correct_actions,
                     double *right,
		     vector<unsigned>* results,
		     bool train,
                     bool sample = false) {
    
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
    
	
/*    if (train) {
      l2rbuilder.set_dropout(params.pdrop);
      r2lbuilder.set_dropout(params.pdrop);
      state_lstm.set_dropout(params.pdrop);
    } else {
      l2rbuilder.disable_dropout();
      r2lbuilder.disable_dropout();
      state_lstm.disable_dropout();
    }
*/
    for (unsigned i = 0; i < sent.size(); ++i) {
      int wordid = sent.raw[i]; // this will be equal to unk at dev/test
      if (train && singletons.size() > wordid && singletons[wordid] && rand01() > params.unk_prob)
          wordid = sent.unk[i];
      if (!train)
	  wordid = sent.unk[i];

      Expression w =lookup(*hg, p_w, wordid);
      if(train) w = dropout(w, params.pdrop);
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

    vector<int> bufferi(sent.size() + 1);  // position of the words in the sentence
    // precompute buffer representation from left to right

    // in the discriminative model, here we set up the buffer contents
    // dummy symbol to represent the empty buffer
    bufferi[0] = -999;
    vector<int> stacki; // position of words in the sentence of head of subtree
    stacki.push_back(-999); // not used for anything
    // drive dummy symbol on stack through LSTM
    vector<int> is_open_paren; // -1 if no nonterminal has a parenthesis open, otherwise index of NT
    is_open_paren.push_back(-1); // corresponds to dummy symbol
    vector<Expression> log_probs;
    string rootword;
    unsigned stack_buffer_split = 0;
    unsigned action_count = 0;  // incremented at each prediction
    unsigned nt_count = 0; // number of times an NT has been introduced
    vector<unsigned> current_valid_actions;
    int nopen_parens = 0;
    char prev_a = '0';

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
      // get list of possible actions for the current parser state
if(params.debug) cerr<< "action_count " << action_count <<"\n";
	current_valid_actions.clear();
if(params.debug) cerr<< "nopen_parens: "<<nopen_parens<<"\n";
      for (auto a : possible_actions) {
        if (IsActionForbidden_Discriminative(adict.convert(a), prev_a, bufferi.size(), stacki.size(), nopen_parens))
          continue;
        current_valid_actions.push_back(a);
      }
      //cerr << "valid actions = " << current_valid_actions.size() << endl;
if(params.debug){
        cerr <<"current_valid_actions: "<<current_valid_actions.size()<<" :";
        for(unsigned i = 0; i < current_valid_actions.size(); i ++){
                cerr<<adict.convert(current_valid_actions[i])<<" ";
        }
        cerr <<"\n";
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
if(params.debug){
	auto s_see = as_vector(hg->incremental_forward(s_attexp));
	for(unsigned i = 0; i < s_see.size(); i ++){
		cerr<<s_see[i]<<" ";
	}
	cerr<<"\n";
}
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
if(params.debug){
	auto b_see = as_vector(hg->incremental_forward(b_attexp));
	for(unsigned i = 0; i < b_see.size(); i ++){
		cerr<<b_see[i]<<" ";
	}
	cerr<<"\n";
}
      Expression b_input_col = concatenate_cols(b_input);
      Expression b_att_pool = b_input_col * b_attexp;

if(params.debug)	std::cerr<<"attention ok\n";
      Expression combo = affine_transform({combobias, h2combo, prev_h, s_att2combo, s_att_pool, b_att2combo, b_att_pool});
      Expression n_combo = rectify(combo);
      Expression r_t = affine_transform({rtbias, combo2rt, n_combo});
if(params.debug)	std::cerr<<"to action layer ok\n";
 
      if (sample) r_t = r_t * params.alpha;
      // adist = log_softmax(r_t, current_valid_actions)
      Expression r_t_s = select_rows(r_t, current_valid_actions);
      Expression adiste = log_softmax(r_t_s);
      vector<float> adist = as_vector(hg->incremental_forward(adiste));
      double best_score = adist[0];
      unsigned model_action = current_valid_actions[0];
      if (sample) {
        double p = rand01();
        assert(current_valid_actions.size() > 0);
        unsigned w = 0;
        for (; w < current_valid_actions.size(); ++w) {
          p -= exp(adist[w]);
          if (p < 0.0) { break; }
        }
        if (w == current_valid_actions.size()) w--;
        model_action = current_valid_actions[w];
      } else { // max
        for (unsigned i = 1; i < current_valid_actions.size(); ++i) {
          if (adist[i] > best_score) {
            best_score = adist[i];
            model_action = current_valid_actions[i];
          }
        }
      }
      unsigned action = model_action;
      if (train) {  // if we have reference actions (for training) use the reference action
        if (action_count >= correct_actions.size()) {
          cerr << "Correct action list exhausted, but not in final parser state.\n";
          abort();
        }
        action = correct_actions[action_count];
        if (model_action == action) { (*right)++; }
      } else {
        //cerr << "Chosen action: " << adict.convert(action) << endl;
      }
      //cerr << "prob ="; for (unsigned i = 0; i < adist.size(); ++i) { cerr << ' ' << adict.convert(i) << ':' << adist[i]; }
      //cerr << endl;
      ++action_count;
      unsigned w = 0;
      for(;w< current_valid_actions.size(); w++){
        if(current_valid_actions[w] == action) break;
      } 
      assert(w != current_valid_actions.size());
    
      log_probs.push_back(pick(adiste, w));
      if(results) results->push_back(action);

      // add current action to action LSTM
      Expression actione = lookup(*hg, p_a, action);
      state_lstm.add_input(concatenate({actione,s_att_pool,b_att_pool}));

      // do action
      const string& actionString=adict.convert(action);
      //cerr << "ACT: " << actionString << endl;
      const char ac = actionString[0];
      const char ac2 = actionString[1];
      prev_a = ac;
if(params.debug){

      cerr << "MODEL_ACT: " << adict.convert(model_action)<<" ";
      cerr <<"GOLD_ACT: " << actionString<<"\n";
}

if(params.debug) {
	cerr <<"stack_buffer_split:" << stack_buffer_split <<"\n";
        cerr <<"stacki: ";
        for(unsigned i = 0; i < stacki.size(); i ++){
                cerr<<stacki[i]<<" ";
        }
        cerr<<"\n";

        cerr<<"is_open_paren: ";
        for(unsigned i = 0; i < is_open_paren.size(); i ++){
                cerr<<is_open_paren[i]<<" ";
        }
        cerr<<"\n";

}

      if (ac =='S' && ac2=='H') {  // SHIFT
        assert(bufferi.size() > 1); // dummy symbol means > 1 (not >= 1)
        stacki.push_back(bufferi.back());
        bufferi.pop_back();
        is_open_paren.push_back(-1);
	stack_buffer_split += 1;
      } else if (ac == 'N') { // NT
        ++nopen_parens;
        assert(bufferi.size() > 1);
        auto it = action2NTindex.find(action);
        assert(it != action2NTindex.end());
        int nt_index = it->second;
        nt_count++;
        stacki.push_back(-1);
        is_open_paren.push_back(nt_index);
      } else { // REDUCE
        --nopen_parens;
        assert(stacki.size() > 2); // dummy symbol means > 2 (not >= 2)
        // find what paren we are closing
        int i = is_open_paren.size() - 1;
        while(is_open_paren[i] < 0) { --i; assert(i >= 0); }
        int nchildren = is_open_paren.size() - i - 1;
        assert(nchildren > 0);
        //cerr << "  number of children to reduce: " << nchildren << endl;

        // REMOVE EVERYTHING FROM THE STACK THAT IS GOING
        // TO BE COMPOSED INTO A TREE EMBEDDING
        for (i = 0; i < nchildren; ++i) {
          assert (stacki.back() != -1);
          stacki.pop_back();
          is_open_paren.pop_back();
        }
	is_open_paren.pop_back(); // nt symbol
        stacki.pop_back(); // nonterminal dummy

        // BUILD TREE EMBEDDING USING BIDIR LSTM
        stacki.push_back(999); // who knows, should get rid of this
        is_open_paren.push_back(-1); // we just closed a paren at this position
      }
    }
    if (train && action_count != correct_actions.size()) {
      cerr << "Unexecuted actions remain but final state reached!\n";
      abort();
    }
    assert(stacki.size() == 2);
    assert(bufferi.size() == 1);
    Expression tot_neglogprob = -sum(log_probs);
    assert(tot_neglogprob.pg != nullptr);
    return tot_neglogprob;
  }




struct ParserState {
  LSTMBuilder stack_lstm;
  LSTMBuilder *buffer_lstm;
  LSTMBuilder action_lstm;
  vector<Expression> buffer;
  vector<int> bufferi;
  LSTMBuilder const_lstm_fwd;
  LSTMBuilder const_lstm_rev;

  vector<Expression> stack;
  vector<int> stacki;
  vector<unsigned> results;  // sequence of predicted actions
  bool complete;
  vector<Expression> log_probs;
  double score;
  int action_count;
  int nopen_parens;
  char prev_a;
};


struct ParserStateCompare {
  bool operator()(const ParserState& a, const ParserState& b) const {
    return a.score > b.score;
  }
};

static void prune(vector<ParserState>& pq, unsigned k) {
  if (pq.size() == 1) return;
  if (k > pq.size()) k = pq.size();
  partial_sort(pq.begin(), pq.begin() + k, pq.end(), ParserStateCompare());
  pq.resize(k);
  reverse(pq.begin(), pq.end());
  //cerr << "PRUNE\n";
  //for (unsigned i = 0; i < pq.size(); ++i) {
  //  cerr << pq[i].score << endl;
  //}
}

static bool all_complete(const vector<ParserState>& pq) {
  for (auto& ps : pq) if (!ps.complete) return false;
  return true;
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
  if (params.unk_strategy) {
    cerr << "STOCHASTIC REPLACEMENT\n";
  } else {
    abort();
  }
  assert(params.unk_prob >= 0.); assert(params.unk_prob <= 1.);
  ostringstream os;
  os << "parser"
     << '_' << params.layers
     << '_' << params.input_dim
     << '_' << params.action_dim
     << '_' << params.pos_dim
     << '_' << params.rel_dim
     << '_' << params.bilstm_input_dim
     << '_' << params.bilstm_hidden_dim
     << '_' << params.attention_hidden_dim
     << "-pid" << getpid() << ".params";
  const string fname = os.str();
  cerr << "Writing parameters to file: " << fname << endl;

//=====================================================================================================
  
  parser::TopDownOracle corpus(&termdict, &adict, &posdict, &ntermdict);
  corpus.load_oracle(params.train_file, true);
  corpus.load_bdata(params.bracketed_file);

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

  // freeze dictionaries so we don't accidentaly load OOVs
  termdict.freeze();
  termdict.set_unk("UNK"); // we don't actually expect to use this often
  adict.freeze();
  ntermdict.freeze();
  posdict.freeze();

  {  // compute the singletons in the parser's training data
    unordered_map<unsigned, unsigned> counts;
    for (auto& sent : corpus.sents)
      for (auto word : sent.raw) counts[word]++;
    singletons.resize(termdict.size(), false);
    for (auto wc : counts)
      if (wc.second == 1) singletons[wc.first] = true;
  }

  for (unsigned i = 0; i < adict.size(); ++i) {
    const string& a = adict.convert(i);
    if (a[0] != 'N') continue;
    size_t start = a.find('(') + 1;
    size_t end = a.rfind(')');
    int nt = ntermdict.convert(a.substr(start, end - start));
    action2NTindex[i] = nt;
  }

  NT_SIZE = ntermdict.size()+10;
  POS_SIZE = posdict.size()+10;
  VOCAB_SIZE = termdict.size()+10;
  ACTION_SIZE = adict.size()+10;

  for(unsigned i = 0; i < adict.size(); ++i) possible_actions.push_back(i);

  parser::TopDownOracle dev_corpus(&termdict, &adict, &posdict, &ntermdict);
  parser::TopDownOracle test_corpus(&termdict, &adict, &posdict, &ntermdict);
  if(params.dev_file != "") dev_corpus.load_oracle(params.dev_file, false);
  if(params.test_file != "") test_corpus.load_oracle(params.test_file, false);
  
//============================================================================================================

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
    if(method == 0){
        sgd = new SimpleSGDTrainer(model,0.1, 0.1);
	sgd->eta_decay = 0.05;
    }
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
    status_every_i_iterations = min((int)status_every_i_iterations, (int)corpus.sents.size());
    unsigned si = corpus.sents.size();
    cerr << "NUMBER OF TRAINING SENTENCES: " << corpus.sents.size() << endl;
    unsigned trs = 0;
    unsigned words = 0;
    double right = 0;
    double llh = 0;
    bool first = true;
    int iter = -1;
    double best_dev_err = 9e99;
    double bestf1=0.0;
    //cerr << "TRAINING STARTED AT: " << put_time(localtime(&time_start), "%c %Z") << endl;
    while(!requested_stop) {
      ++iter;
      auto time_start = chrono::system_clock::now();
      for (unsigned sii = 0; sii < status_every_i_iterations; ++sii) {
           if (si == corpus.sents.size()) {
             si = 0;
             if (first) { first = false; } else { sgd->update_epoch(); }
             cerr << "**SHUFFLE\n";
             random_shuffle(order.begin(), order.end());
           }
           tot_seen += 1;
           auto& sentence = corpus.sents[order[si]];
	   const vector<int>& actions=corpus.actions[order[si]];
	   
           ComputationGraph hg;
           Expression nll = parser.log_prob_parser(&hg,sentence,actions,&right,NULL,true,false);

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
        ofstream out("dev.out");
        auto t_start = chrono::high_resolution_clock::now();
        for (unsigned sii = 0; sii < dev_size; ++sii) {
           const auto& sentence=dev_corpus.sents[sii];
	   const vector<int>& actions=dev_corpus.actions[sii];
           
	   ComputationGraph hg;
	   vector<unsigned> pred;
           Expression nll = parser.log_prob_parser(&hg, sentence, actions, &right, &pred, false, false);
           double lp = as_scalar(hg.incremental_forward(nll));
           llh += lp;

	   int ti = 0;
           for (auto a : pred) {
             if (adict.convert(a)[0] == 'N') {
               out << '(' << ntermdict.convert(action2NTindex.find(a)->second) << ' ';
             } else if (adict.convert(a)[0] == 'S') {
               out << '(' << posdict.convert(sentence.pos[ti]) << ' ' << sentence.surfaces[ti] << ") ";
		ti ++;
             } else out << ") ";
           }
           out << endl;
           trs += actions.size();
	   dwords += sentence.size();
        }
        auto t_end = chrono::high_resolution_clock::now();
        out.close();
        double err = (trs - right) / trs;

        std::string command2="EVALB/evalb -p EVALB/COLLINS.prm "+corpus.devdata+" dev.out > evalbout.txt";
        const char* cmd2=command2.c_str();
        system(cmd2);
        
        std::ifstream evalfile("evalbout.txt");
        std::string lineS;
        std::string brackstr="Bracketing FMeasure";
        double newfmeasure=0.0;
        std::string strfmeasure="";
        bool found=0;
        while (getline(evalfile, lineS) && !newfmeasure){
		if (lineS.compare(0, brackstr.length(), brackstr) == 0) {
			//std::cout<<lineS<<"\n";
			strfmeasure=lineS.substr(lineS.size()-5, lineS.size());
                        std::string::size_type sz;     // alias of size_t

		        newfmeasure = std::stod (strfmeasure,&sz);
			//std::cout<<strfmeasure<<"\n";
		}
        }
        
        cerr << "  **dev (iter=" << iter << " epoch=" << (tot_seen / corpus.size()) << ")\t"
		<<" llh= " << llh
		<<" ppl: " << exp(llh / dwords)
		<<" f1: " << newfmeasure
		<<" err: " << err
		<<"\t[" << dev_size << " sents in " << chrono::duration<double, milli>(t_end-t_start).count() << " ms]" << endl;
       if (newfmeasure>bestf1) {
          cerr << "  new best...writing model to " << fname << " ...\n";
          best_dev_err = err;
	  bestf1=newfmeasure;

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
  } // should do training?
  else{ // do test evaluation
	ofstream out("test.out");
        unsigned test_size = test_corpus.size();
        double llh = 0;
        double trs = 0;
        double right = 0;
        double dwords = 0;
	if(params.samples !=0){
        	auto t_start = chrono::high_resolution_clock::now();
        	for (unsigned sii = 0; sii < test_size; ++sii) {
           		const auto& sentence=test_corpus.sents[sii];
			const vector<int>& actions=test_corpus.actions[sii];
           		for (unsigned z = 0; z < params.samples; ++z) {
             			ComputationGraph hg;
             			vector<unsigned> pred;
	     			parser.log_prob_parser(&hg,sentence,actions,&right,&pred,false,true);
             			int ti = 0;
             			for (auto a : pred) {
               				if (adict.convert(a)[0] == 'N') {
                 				cout << " (" << ntermdict.convert(action2NTindex.find(a)->second);
               				} else if (adict.convert(a)[0] == 'S') {
                     				cout << " (" << posdict.convert(sentence.pos[ti]) << " " << sentence.surfaces[ti] << ")";
						ti ++;
               				} else cout << ')';
             			}
             			cout << endl;
           		}
       		}
	}
        	auto t_start = chrono::high_resolution_clock::now();
        	for (unsigned sii = 0; sii < test_size; ++sii) {
           		const auto& sentence=test_corpus.sents[sii];
           		const vector<int>& actions=test_corpus.actions[sii];
           		dwords += sentence.size();
           		ComputationGraph hg;
           		vector<unsigned> pred;
	   		Expression nll = parser.log_prob_parser(&hg,sentence,actions,&right,&pred,false,false);
	   		double lp = as_scalar(hg.incremental_forward(nll));
           		llh += lp;

           		int ti = 0;
           		for (auto a : pred) {
             			if (adict.convert(a)[0] == 'N') {
               				out << '(' << ntermdict.convert(action2NTindex.find(a)->second) << ' ';
             			} else if (adict.convert(a)[0] == 'S') {
					out << " (" << posdict.convert(sentence.pos[ti]) << " " << sentence.surfaces[ti] << ")";
					ti ++;
             			} else out << ") ";
           		}
           		out << endl;
           		trs += actions.size();
        	}
        	auto t_end = chrono::high_resolution_clock::now();
        	out.close();
        	double err = (trs - right) / trs;

        	std::string command2="EVALB/evalb -p EVALB/COLLINS.prm "+corpus.devdata+" test.out > evalbout.txt";
        	const char* cmd2=command2.c_str();
		system(cmd2);

        	std::ifstream evalfile("evalbout.txt");
        	std::string lineS;
        	std::string brackstr="Bracketing FMeasure";
        	double newfmeasure=0.0;
        	std::string strfmeasure="";
        	bool found=0;
        	while (getline(evalfile, lineS) && !newfmeasure){
                	if (lineS.compare(0, brackstr.length(), brackstr) == 0) {
                        //std::cout<<lineS<<"\n";
                        strfmeasure=lineS.substr(lineS.size()-5, lineS.size());
                        std::string::size_type sz;
                        newfmeasure = std::stod (strfmeasure,&sz);
                        //std::cout<<strfmeasure<<"\n";
                	}
        	}

       		cerr<<"F1score: "<<newfmeasure<<"\n";
    
  }
}
