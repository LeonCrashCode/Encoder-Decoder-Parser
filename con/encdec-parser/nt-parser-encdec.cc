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

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/program_options.hpp>

#include "cnn/training.h"
#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "cnn/nodes.h"
#include "cnn/lstm.h"
#include "cnn/rnn.h"
#include "cnn/dict.h"
#include "cnn/cfsm-builder.h"

#include "encdec-parser/oracle.h"
#include "encdec-parser/pretrained.h"
#include "encdec-parser/compressed-fstream.h"
#include "encdec-parser/eval.h"

// dictionaries
cnn::Dict termdict, ntermdict, adict, posdict;
bool DEBUG = false;
volatile bool requested_stop = false;
unsigned LAYERS = 2;
unsigned INPUT_DIM = 40;
unsigned PRETRAINED_DIM = 50;
unsigned ACTION_DIM = 36;
unsigned POS_DIM = 10;
unsigned REL_DIM = 8;

unsigned BILSTM_INPUT_DIM = 64;
unsigned BILSTM_HIDDEN_DIM = 64;
unsigned ATTENTION_HIDDEN_DIM = 64;

unsigned STATE_INPUT_DIM = ACTION_DIM + ATTENTION_HIDDEN_DIM;
unsigned STATE_HIDDEN_DIM = 64; 

float ALPHA = 1.f;
unsigned N_SAMPLES = 1;
unsigned ACTION_SIZE = 0;
unsigned VOCAB_SIZE = 0;
unsigned NT_SIZE = 0;
float DROPOUT = 0.0f;
unsigned POS_SIZE = 0;
std::map<int,int> action2NTindex;  // pass in index of action NT(X), return index of X
bool USE_POS = false;  // in discriminative parser, incorporate POS information in token embedding

using namespace cnn::expr;
using namespace cnn;
using namespace std;
namespace po = boost::program_options;


vector<unsigned> possible_actions;
unordered_map<unsigned, vector<float>> pretrained;
vector<bool> singletons; // used during training

void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
        ("training_data,T", po::value<string>(), "List of Transitions - Training corpus")
        ("dev_data,d", po::value<string>(), "Development corpus")
        ("bracketing_dev_data,C", po::value<string>(), "Development bracketed corpus")
        ("test_data,p", po::value<string>(), "Test corpus")
        ("dropout,D", po::value<float>(), "Dropout rate")
        ("samples,s", po::value<unsigned>(), "Sample N trees for each test sentence instead of greedy max decoding")
        ("alpha,a", po::value<float>(), "Flatten (0 < alpha < 1) or sharpen (1 < alpha) sampling distribution")
        ("model,m", po::value<string>(), "Load saved model from this file")
        ("use_pos_tags,P", "make POS tags visible to parser")
        ("layers", po::value<unsigned>()->default_value(1), "number of LSTM layers")
        ("action_dim", po::value<unsigned>()->default_value(16), "action embedding size")
        ("pos_dim", po::value<unsigned>()->default_value(12), "POS dimension")
	("input_dim", po::value<unsigned>()->default_value(32), "input embedding size")
	("bilstm_input_dim", po::value<unsigned>()->default_value(64), "bilstm input dimension")
        ("bilstm_hidden_dim", po::value<unsigned>()->default_value(64), "bilstm hidden dimension")
        ("attention_hidden_dim", po::value<unsigned>()->default_value(64), "attention hidden dimension")
        ("state_hidden_dim", po::value<unsigned>()->default_value(64), "state hidden dimension")
        ("pretrained_dim", po::value<unsigned>()->default_value(50), "pretrained input dimension")
        ("train,t", "Should training be run?")
        ("words,w", po::value<string>(), "Pretrained word embeddings")
        ("debug", "debug")
	("help,h", "Help");
      
  po::options_description dcmdline_options;
  dcmdline_options.add(opts);
  po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
  if (conf->count("help")) {
    cerr << dcmdline_options << endl;
    exit(1);
  }
  if (conf->count("training_data") == 0) {
    cerr << "Please specify --traing_data (-T): this is required to determine the vocabulary mapping, even if the parser is used in prediction mode.\n";
    exit(1);
  }
}

struct ParserBuilder {

  LSTMBuilder state_lstm;
  LSTMBuilder l2rbuilder;
  LSTMBuilder r2lbuilder;
  LookupParameters* p_w; // word embeddings
  LookupParameters* p_t; // pretrained word embeddings (not updated)
  LookupParameters* p_a; // input action embeddings
  LookupParameters* p_r; // relation embeddings
  LookupParameters* p_p; // pos tag embeddings
  Parameters* p_w2l; // word to LSTM input
  Parameters* p_p2l; // POS to LSTM input
  Parameters* p_t2l; // pretrained word embeddings to LSTM input
  Parameters* p_lb; // LSTM input bias

  Parameters* p_sent_start;
  Parameters* p_sent_end;

  Parameters* p_s_input2att;
  Parameters* p_s_h2att;
  Parameters* p_s_attbias;
  Parameters* p_s_att2attexp;
  Parameters* p_s_att2combo;
  
  Parameters* p_b_input2att;
  Parameters* p_b_h2att;
  Parameters* p_b_attbias;
  Parameters* p_b_att2attexp;
  Parameters* p_b_att2combo;

  Parameters* p_h2combo;
  Parameters* p_combobias;
  Parameters* p_combo2rt;
  Parameters* p_rtbias;

  explicit ParserBuilder(Model* model, const unordered_map<unsigned, vector<float>>& pretrained) :
      state_lstm(1, STATE_INPUT_DIM ,STATE_HIDDEN_DIM, model),
      l2rbuilder(LAYERS, BILSTM_INPUT_DIM, BILSTM_HIDDEN_DIM, model),
      r2lbuilder(LAYERS, BILSTM_INPUT_DIM, BILSTM_HIDDEN_DIM, model),
      p_w(model->add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM})),
      p_a(model->add_lookup_parameters(ACTION_SIZE, {ACTION_DIM})),
      p_r(model->add_lookup_parameters(ACTION_SIZE, {REL_DIM})),
      p_w2l(model->add_parameters({BILSTM_INPUT_DIM, INPUT_DIM})),
      p_lb(model->add_parameters({BILSTM_INPUT_DIM})),
      p_sent_start(model->add_parameters({BILSTM_INPUT_DIM})),
      p_sent_end(model->add_parameters({BILSTM_INPUT_DIM})),
      p_s_input2att(model->add_parameters({ATTENTION_HIDDEN_DIM, BILSTM_HIDDEN_DIM*2})),
      p_s_h2att(model->add_parameters({ATTENTION_HIDDEN_DIM, STATE_HIDDEN_DIM})),
      p_s_attbias(model->add_parameters({ATTENTION_HIDDEN_DIM})),
      p_s_att2attexp(model->add_parameters({ATTENTION_HIDDEN_DIM})),
      p_s_att2combo(model->add_parameters({STATE_HIDDEN_DIM, BILSTM_HIDDEN_DIM*2})),
      p_b_input2att(model->add_parameters({ATTENTION_HIDDEN_DIM, BILSTM_HIDDEN_DIM*2})),
      p_b_h2att(model->add_parameters({ATTENTION_HIDDEN_DIM, STATE_HIDDEN_DIM})),
      p_b_attbias(model->add_parameters({ATTENTION_HIDDEN_DIM})),
      p_b_att2attexp(model->add_parameters({ATTENTION_HIDDEN_DIM})),
      p_b_att2combo(model->add_parameters({STATE_HIDDEN_DIM, BILSTM_HIDDEN_DIM*2})),
      p_h2combo(model->add_parameters({STATE_HIDDEN_DIM, STATE_HIDDEN_DIM})),
      p_combobias(model->add_parameters({STATE_HIDDEN_DIM})),
      p_combo2rt(model->add_parameters({ACTION_SIZE, STATE_HIDDEN_DIM})),
      p_rtbias(model->add_parameters({ACTION_SIZE})){

    if (USE_POS) {
      p_p = model->add_lookup_parameters(POS_SIZE, {POS_DIM});
      p_p2l = model->add_parameters({BILSTM_INPUT_DIM, POS_DIM});
    }
//    buffer_lstm = new LSTMBuilder(LAYERS, LSTM_INPUT_DIM, HIDDEN_DIM, model);
    if (pretrained.size() > 0) {
      p_t = model->add_lookup_parameters(VOCAB_SIZE, {PRETRAINED_DIM});
      for (auto it : pretrained)
        p_t->Initialize(it.first, it.second);
      p_t2l = model->add_parameters({BILSTM_INPUT_DIM, PRETRAINED_DIM});
    } else {
      p_t = nullptr;
      p_t2l = nullptr;
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
vector<unsigned> log_prob_parser(ComputationGraph* hg,
                     const parser::Sentence& sent,
                     const vector<int>& correct_actions,
                     double *right,
                     bool is_evaluation,
                     bool sample = false) {
    vector<unsigned> results;
    const bool build_training_graph = correct_actions.size() > 0;
    bool apply_dropout = (DROPOUT && !is_evaluation);
    
    l2rbuilder.new_graph(*hg);
    r2lbuilder.new_graph(*hg);
    l2rbuilder.start_new_sequence();
    r2lbuilder.start_new_sequence();
    // variables in the computation graph representing the parameters
    Expression lb = parameter(*hg, p_lb);
    Expression w2l = parameter(*hg, p_w2l);
    Expression p2l;
    if (USE_POS)
      p2l = parameter(*hg, p_p2l);
    Expression t2l;
    if (pretrained.size()>0)
      t2l = parameter(*hg, p_t2l); 
    state_lstm.new_graph(*hg);
    state_lstm.start_new_sequence();
    //state_lstm.start_new_sequence({zeroes(*hg, {STATE_HIDDEN_DIM}), state_start});
    
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
    
	
    if (apply_dropout) {
      l2rbuilder.set_dropout(DROPOUT);
      r2lbuilder.set_dropout(DROPOUT);
      state_lstm.set_dropout(DROPOUT);
    } else {
      l2rbuilder.disable_dropout();
      r2lbuilder.disable_dropout();
      state_lstm.disable_dropout();
    }

    for (unsigned i = 0; i < sent.size(); ++i) {
      int wordid = sent.raw[i]; // this will be equal to unk at dev/test
      if (build_training_graph && singletons.size() > wordid && singletons[wordid] && rand01() > 0.5)
          wordid = sent.unk[i];

      Expression w =lookup(*hg, p_w, wordid);
      vector<Expression> args = {lb, w2l, w}; // learn embeddings
      if (USE_POS) { // learn POS tag?
        Expression p = lookup(*hg, p_p, sent.pos[i]);
        args.push_back(p2l);
        args.push_back(p);
      }
      if (pretrained.size() > 0 &&  pretrained.count(sent.lc[i])) {  // include fixed pretrained vectors?
        Expression t = const_lookup(*hg, p_t, sent.lc[i]);
        args.push_back(t2l);
        args.push_back(t);
      }
      else{
        args.push_back(t2l);
        args.push_back(zeroes(*hg,{PRETRAINED_DIM}));
      }
      input_expr.push_back(rectify(affine_transform(args)));
    }
if(DEBUG)	std::cerr<<"lookup table ok\n";
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
if(DEBUG)	std::cerr<<"bilstm ok\n";
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
    //for(unsigned i = 0; i < LAYERS; i ++){
      initc.push_back(concatenate({l2rhc.back(),r2lhc.back()}));
    //}

    //for(unsigned i = 0; i < LAYERS; i ++){
      initc.push_back(zeroes(*hg, {BILSTM_HIDDEN_DIM*2}));
    //}
    state_lstm.start_new_sequence(initc);

    while(stacki.size() > 2 || bufferi.size() > 1) {
      // get list of possible actions for the current parser state
if(DEBUG) cerr<< "action_count " << action_count <<"\n";
	current_valid_actions.clear();
if(DEBUG) cerr<< "nopen_parens: "<<nopen_parens<<"\n";
      for (auto a: possible_actions) {
        if (IsActionForbidden_Discriminative(adict.Convert(a), prev_a, bufferi.size(), stacki.size(), nopen_parens))
          continue;
        current_valid_actions.push_back(a);
      }
      //cerr << "valid actions = " << current_valid_actions.size() << endl;
if(DEBUG){
        cerr <<"current_valid_actions: "<<current_valid_actions.size()<<" :";
        for(unsigned i = 0; i < current_valid_actions.size(); i ++){
                cerr<<adict.Convert(current_valid_actions[i])<<" ";
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
if(DEBUG){
	auto s_see = as_vector(hg->incremental_forward());
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
if(DEBUG){
	auto b_see = as_vector(hg->incremental_forward());
	for(unsigned i = 0; i < b_see.size(); i ++){
		cerr<<b_see[i]<<" ";
	}
	cerr<<"\n";
}
      Expression b_input_col = concatenate_cols(b_input);
      Expression b_att_pool = b_input_col * b_attexp;

if(DEBUG)	std::cerr<<"attention ok\n";
      Expression combo = affine_transform({combobias, h2combo, prev_h, s_att2combo, s_att_pool, b_att2combo, b_att_pool});
      Expression n_combo = rectify(combo);
      Expression r_t = affine_transform({rtbias, combo2rt, n_combo});
if(DEBUG)	std::cerr<<"to action layer ok\n";
 
      if (sample && ALPHA != 1.0f) r_t = r_t * ALPHA;
      // adist = log_softmax(r_t, current_valid_actions)
      Expression adiste = log_softmax(r_t, current_valid_actions);
      vector<float> adist = as_vector(hg->incremental_forward());
      double best_score = adist[current_valid_actions[0]];
      unsigned model_action = current_valid_actions[0];
      if (sample) {
        double p = rand01();
        assert(current_valid_actions.size() > 0);
        unsigned w = 0;
        for (; w < current_valid_actions.size(); ++w) {
          p -= exp(adist[current_valid_actions[w]]);
          if (p < 0.0) { break; }
        }
        if (w == current_valid_actions.size()) w--;
        model_action = current_valid_actions[w];
      } else { // max
        for (unsigned i = 1; i < current_valid_actions.size(); ++i) {
          if (adist[current_valid_actions[i]] > best_score) {
            best_score = adist[current_valid_actions[i]];
            model_action = current_valid_actions[i];
          }
        }
      }
      unsigned action = model_action;
      if (build_training_graph) {  // if we have reference actions (for training) use the reference action
        if (action_count >= correct_actions.size()) {
          cerr << "Correct action list exhausted, but not in final parser state.\n";
          abort();
        }
        action = correct_actions[action_count];
        if (model_action == action) { (*right)++; }
      } else {
        //cerr << "Chosen action: " << adict.Convert(action) << endl;
      }
      //cerr << "prob ="; for (unsigned i = 0; i < adist.size(); ++i) { cerr << ' ' << adict.Convert(i) << ':' << adist[i]; }
      //cerr << endl;
      ++action_count;
      log_probs.push_back(pick(adiste, action));
      results.push_back(action);

      // add current action to action LSTM
      Expression actione = lookup(*hg, p_a, action);
      state_lstm.add_input(concatenate({actione,s_att_pool,b_att_pool}));

      // do action
      const string& actionString=adict.Convert(action);
      //cerr << "ACT: " << actionString << endl;
      const char ac = actionString[0];
      const char ac2 = actionString[1];
      prev_a = ac;
if(DEBUG){

      cerr << "MODEL_ACT: " << adict.Convert(model_action)<<" ";
      cerr <<"GOLD_ACT: " << actionString<<"\n";
}

if(DEBUG) {
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
    if (build_training_graph && action_count != correct_actions.size()) {
      cerr << "Unexecuted actions remain but final state reached!\n";
      abort();
    }
    assert(stacki.size() == 2);
    assert(bufferi.size() == 1);
    Expression tot_neglogprob = -sum(log_probs);
    assert(tot_neglogprob.pg != nullptr);
    return results;
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
  cnn::Initialize(argc, argv, 1989121011);

  cerr << "COMMAND LINE:"; 
  for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i) cerr << ' ' << argv[i];
  cerr << endl;
  unsigned status_every_i_iterations = 100;

  po::variables_map conf;
  InitCommandLine(argc, argv, &conf);
  USE_POS = conf.count("use_pos_tags");
  if (conf.count("dropout"))
    DROPOUT = conf["dropout"].as<float>();
  LAYERS = conf["layers"].as<unsigned>();
  INPUT_DIM = conf["input_dim"].as<unsigned>();
  PRETRAINED_DIM = conf["pretrained_dim"].as<unsigned>();
  BILSTM_INPUT_DIM = conf["bilstm_input_dim"].as<unsigned>();
  BILSTM_HIDDEN_DIM = conf["bilstm_hidden_dim"].as<unsigned>();
  STATE_HIDDEN_DIM = BILSTM_HIDDEN_DIM * 2;
  ATTENTION_HIDDEN_DIM = conf["attention_hidden_dim"].as<unsigned>();

  ACTION_DIM = conf["action_dim"].as<unsigned>();
  POS_DIM = conf["pos_dim"].as<unsigned>();
  STATE_INPUT_DIM = ACTION_DIM + BILSTM_HIDDEN_DIM*2 + BILSTM_HIDDEN_DIM*2;
  DEBUG = conf.count("debug");
  if (conf.count("train") && conf.count("dev_data") == 0) {
    cerr << "You specified --train but did not specify --dev_data FILE\n";
    return 1;
  }
  if (conf.count("alpha")) {
    ALPHA = conf["alpha"].as<float>();
    if (ALPHA <= 0.f) { cerr << "--alpha must be between 0 and +infty\n"; abort(); }
  }
  if (conf.count("samples")) {
    N_SAMPLES = conf["samples"].as<unsigned>();
    if (N_SAMPLES == 0) { cerr << "Please specify N>0 samples\n"; abort(); }
  }
  
  ostringstream os;
  os << "ntparse"
     << (USE_POS ? "_pos" : "")
     << '_' << LAYERS
     << '_' << INPUT_DIM
     << '_' << ACTION_DIM
     << '_' << POS_DIM
     << '_' << REL_DIM
     << '_' << BILSTM_INPUT_DIM
     << '_' << BILSTM_HIDDEN_DIM
     << '_' << ATTENTION_HIDDEN_DIM
     << '_' << STATE_HIDDEN_DIM
     << "-pid" << getpid() << ".params";
  const string fname = os.str();
  cerr << "PARAMETER FILE: " << fname << endl;
  bool softlinkCreated = false;

  Model model;

  parser::TopDownOracle corpus(&termdict, &adict, &posdict, &ntermdict);
  parser::TopDownOracle dev_corpus(&termdict, &adict, &posdict, &ntermdict);
  parser::TopDownOracle test_corpus(&termdict, &adict, &posdict, &ntermdict);
  corpus.load_oracle(conf["training_data"].as<string>(), true);	
  corpus.load_bdata(conf["bracketing_dev_data"].as<string>());

  if (conf.count("words"))
    parser::ReadEmbeddings_word2vec(conf["words"].as<string>(), &termdict, &pretrained);

  // freeze dictionaries so we don't accidentaly load OOVs
  termdict.Freeze();
  termdict.SetUnk("UNK"); // we don't actually expect to use this often
     // since the Oracles are required to be "pre-UNKified", but this prevents
     // problems with UNKifying the lowercased data which needs to be loaded
  adict.Freeze();
  ntermdict.Freeze();
  posdict.Freeze();

  {  // compute the singletons in the parser's training data
    unordered_map<unsigned, unsigned> counts;
    for (auto& sent : corpus.sents)
      for (auto word : sent.raw) counts[word]++;
    singletons.resize(termdict.size(), false);
    for (auto wc : counts)
      if (wc.second == 1) singletons[wc.first] = true;
  }

  if (conf.count("dev_data")) {
    cerr << "Loading validation set\n";
    dev_corpus.load_oracle(conf["dev_data"].as<string>(), false);
  }
  if (conf.count("test_data")) {
    cerr << "Loading test set\n";
    test_corpus.load_oracle(conf["test_data"].as<string>(), false);
  }

  for (unsigned i = 0; i < adict.size(); ++i) {
    const string& a = adict.Convert(i);
    if (a[0] != 'N') continue;
    size_t start = a.find('(') + 1;
    size_t end = a.rfind(')');
    int nt = ntermdict.Convert(a.substr(start, end - start));
    action2NTindex[i] = nt;
  }

  NT_SIZE = ntermdict.size();
  POS_SIZE = posdict.size();
  VOCAB_SIZE = termdict.size();
  ACTION_SIZE = adict.size();
  possible_actions.resize(adict.size());
  for (unsigned i = 0; i < adict.size(); ++i)
    possible_actions[i] = i;

  ParserBuilder parser(&model, pretrained);
  if (conf.count("model")) {
    ifstream in(conf["model"].as<string>().c_str());
    boost::archive::text_iarchive ia(in);
    ia >> model;
  }

  //TRAINING
  if (conf.count("train")) {
    signal(SIGINT, signal_callback_handler);
	SimpleSGDTrainer sgd(&model);

	//AdamTrainer sgd(&model);
    //MomentumSGDTrainer sgd(&model);
    //sgd.eta_decay = 0.08;
    sgd.eta_decay = 0.05;
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
             if (first) { first = false; } else { sgd.update_epoch(); }
             cerr << "**SHUFFLE\n";
             random_shuffle(order.begin(), order.end());
           }
           tot_seen += 1;
           auto& sentence = corpus.sents[order[si]];
	   const vector<int>& actions=corpus.actions[order[si]];
           ComputationGraph hg;
           parser.log_prob_parser(&hg,sentence,actions,&right,false);
           double lp = as_scalar(hg.incremental_forward());
           if (lp < 0) {
             cerr << "Log prob < 0 on sentence " << order[si] << ": lp=" << lp << endl;
             assert(lp >= 0.0);
           }
           hg.backward();
           sgd.update(1.0);
           llh += lp;
           ++si;
           trs += actions.size();
           words += sentence.size();
      }
      sgd.status();
      auto time_now = chrono::system_clock::now();
      auto dur = chrono::duration_cast<chrono::milliseconds>(time_now - time_start);
      cerr << "update #" << iter << " (epoch " << (tot_seen / corpus.sents.size()) <<
         /*" |time=" << put_time(localtime(&time_now), "%c %Z") << ")\tllh: "<< */
        ") per-action-ppl: " << exp(llh / trs) << " per-input-ppl: " << exp(llh / words) << " per-sent-ppl: " << exp(llh / status_every_i_iterations) << " err: " << (trs - right) / trs << " [" << dur.count() / (double)status_every_i_iterations << "ms per instance]" << endl;
      llh = trs = right = words = 0;

      static int logc = 0;
      ++logc;
      if (logc % 25 == 1) { // report on dev set
        unsigned dev_size = dev_corpus.size();
        double llh = 0;
        double trs = 0;
        double right = 0;
        double dwords = 0;
        ostringstream os;
        os << "/tmp/parser_dev_eval." << getpid() << ".txt";
        const string pfx = os.str();
        ofstream out(pfx.c_str());
        auto t_start = chrono::high_resolution_clock::now();
        for (unsigned sii = 0; sii < dev_size; ++sii) {
           const auto& sentence=dev_corpus.sents[sii];
	   const vector<int>& actions=dev_corpus.actions[sii];
           dwords += sentence.size();
           {  ComputationGraph hg;
              parser.log_prob_parser(&hg,sentence,actions,&right,true);
              double lp = as_scalar(hg.incremental_forward());
              llh += lp;
           }
           ComputationGraph hg;
           vector<unsigned> pred = parser.log_prob_parser(&hg,sentence,vector<int>(),&right,true);
           int ti = 0;
           for (auto a : pred) {
             if (adict.Convert(a)[0] == 'N') {
               out << '(' << ntermdict.Convert(action2NTindex.find(a)->second) << ' ';
             } else if (adict.Convert(a)[0] == 'S') {
                 if (true) {
                   string preterminal = "XX";
                   out << '(' << preterminal << ' ' << termdict.Convert(sentence.raw[ti++]) << ") ";
                 } else { // use this branch to surpress preterminals
                   out << termdict.Convert(sentence.raw[ti++]) << ' ';
                 }
             } else out << ") ";
           }
           out << endl;
           double lp = 0;
           trs += actions.size();
        }
        auto t_end = chrono::high_resolution_clock::now();
        out.close();
        double err = (trs - right) / trs;
        cerr << "Dev output in " << pfx << endl;
        //parser::EvalBResults res = parser::Evaluate("foo", pfx);
	std::string command="python remove_dev_unk.py "+ corpus.devdata +" "+pfx+" > evaluable.txt";
	const char* cmd=command.c_str();
	system(cmd);

        std::string command2="EVALB/evalb -p EVALB/COLLINS.prm "+corpus.devdata+" evaluable.txt>evalbout.txt";
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
        
 
        
        cerr << "  **dev (iter=" << iter << " epoch=" << (tot_seen / corpus.size()) << ")\tllh=" << llh << " ppl: " << exp(llh / dwords) << " f1: " << newfmeasure << " err: " << err << "\t[" << dev_size << " sents in " << chrono::duration<double, milli>(t_end-t_start).count() << " ms]" << endl;
//        if (err < best_dev_err && (tot_seen / corpus.size()) > 1.0) {
       if (newfmeasure>bestf1) {
          cerr << "  new best...writing model to " << fname << " ...\n";
          best_dev_err = err;
	  bestf1=newfmeasure;
          ofstream out(fname);
          boost::archive::text_oarchive oa(out);
          oa << model;
          system((string("cp ") + pfx + string(" ") + pfx + string(".best")).c_str());
          // Create a soft link to the most recent model in order to make it
          // easier to refer to it in a shell script.
          if (!softlinkCreated) {
            string softlink = " latest_model";
            if (system((string("rm -f ") + softlink).c_str()) == 0 && 
                system((string("ln -s ") + fname + softlink).c_str()) == 0) {
              cerr << "Created " << softlink << " as a soft link to " << fname 
                   << " for convenience." << endl;
            }
            softlinkCreated = true;
          }
        }
      }
    }
  } // should do training?
  if (test_corpus.size() > 0) { // do test evaluation
        bool sample = conf.count("samples") > 0;
        unsigned test_size = test_corpus.size();
        double llh = 0;
        double trs = 0;
        double right = 0;
        double dwords = 0;
        auto t_start = chrono::high_resolution_clock::now();
	const vector<int> actions;
        for (unsigned sii = 0; sii < test_size; ++sii) {
           const auto& sentence=test_corpus.sents[sii];
           dwords += sentence.size();
           for (unsigned z = 0; z < N_SAMPLES; ++z) {
             ComputationGraph hg;
             vector<unsigned> pred = parser.log_prob_parser(&hg,sentence,actions,&right,sample,true);
             double lp = as_scalar(hg.incremental_forward());
             cout << sii << " ||| " << -lp << " |||";
             int ti = 0;
             for (auto a : pred) {
               if (adict.Convert(a)[0] == 'N') {
                 cout << " (" << ntermdict.Convert(action2NTindex.find(a)->second);
               } else if (adict.Convert(a)[0] == 'S') {
                   if (!sample) {
		     cout << ' ' << termdict.Convert(sentence.raw[ti++]);
                     //string preterminal = "XX";
                     //cout << " (" << preterminal << ' ' << termdict.Convert(sentence.raw[ti++]) << ")";
                   } else { // use this branch to surpress preterminals
                     //cout << ' ' << termdict.Convert(sentence.raw[ti++]);
		     //cout<< " (" << posdict.Convert(sentence.pos[ti]) << " " << termdict.Convert(sentence.raw[ti++]) << ")";
                     cout << " (" << "XX" << " " << termdict.Convert(sentence.raw[ti++]) << ")";
		   }
               } else cout << ')';
             }
             cout << endl;
           }
       }
       ostringstream os;
        os << "/tmp/parser_test_eval." << getpid() << ".txt";
        const string pfx = os.str();
        ofstream out(pfx.c_str());
        t_start = chrono::high_resolution_clock::now();
        for (unsigned sii = 0; sii < test_size; ++sii) {
           const auto& sentence=test_corpus.sents[sii];
           const vector<int>& actions=test_corpus.actions[sii];
           dwords += sentence.size();
           {  ComputationGraph hg;
              parser.log_prob_parser(&hg,sentence,actions,&right,true);
              double lp = as_scalar(hg.incremental_forward());
              llh += lp;
           }
           ComputationGraph hg;
           vector<unsigned> pred = parser.log_prob_parser(&hg,sentence,vector<int>(),&right,true);
           int ti = 0;
           for (auto a : pred) {
             if (adict.Convert(a)[0] == 'N') {
               out << '(' << ntermdict.Convert(action2NTindex.find(a)->second) << ' ';
             } else if (adict.Convert(a)[0] == 'S') {
                 if (true) {
                   string preterminal = "XX";
                   out << '(' << preterminal << ' ' << termdict.Convert(sentence.raw[ti++]) << ") ";
                 } else { // use this branch to surpress preterminals
                   out << termdict.Convert(sentence.raw[ti++]) << ' ';
                 }
             } else out << ") ";
           }
           out << endl;
           double lp = 0;
           trs += actions.size();
        }
        auto t_end = chrono::high_resolution_clock::now();
        out.close();
        double err = (trs - right) / trs;
        cerr << "Test output in " << pfx << endl;
        //parser::EvalBResults res = parser::Evaluate("foo", pfx);
        std::string command="python remove_dev_unk.py "+ corpus.devdata +" "+pfx+" > evaluable.txt";
        const char* cmd=command.c_str();
        system(cmd);

        std::string command2="EVALB/evalb -p EVALB/COLLINS.prm "+corpus.devdata+" evaluable.txt>evalbout.txt";
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
