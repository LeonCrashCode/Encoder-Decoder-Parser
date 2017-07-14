/**
 * \file cl-args.h
 * \brief This is a **very** minimal command line argument parser
 */
#include <iostream>
#include <stdlib.h>
#include <string>
#include <sstream>

/**
 * Values used to specify the task at hand, and incidentally the required command line arguments
 */
enum Task {
  TRAIN, /**< Self-supervised learning : Only requires train and dev file */
  TRAIN_SUP, /**< Supervised learning : Requires train and dev data as well as labels */
  TEST
};

using namespace std;
/**
 * \brief Structure holding any possible command line argument
 *
 */
struct Params {
  string exp_name = "encdec";
  string train_file = "";
  string dev_file = "";
  string test_file = "";
  string words_file = "";
  string model_file = "";

  unsigned layers = 1;
  unsigned action_dim = 16;
  unsigned input_dim = 32;
  unsigned pretrained_dim = 50;
  unsigned pos_dim = 12;
  unsigned rel_dim = 10;
  unsigned bilstm_input_dim = 64;
  unsigned bilstm_hidden_dim = 64;
  unsigned attention_hidden_dim = 64;
  unsigned state_input_dim = 64;
  unsigned state_hidden_dim = 64;
  unsigned train_methods = 0;

  bool unk_strategy = false;
  bool use_pos = false;
  bool debug = false;
  bool train = false;
  double unk_prob = 0.2;
  double pdrop = 0.3;
};

/**
 * \brief Get parameters from command line arguments
 * \details Parses parameters from `argv` and check for required fields depending on the task
 * 
 * \param argc Number of arguments
 * \param argv Arguments strings
 * \param params Params structure
 * \param task Task
 */
void get_args(int argc,
              char** argv,
              Params& params) {
  int i = 0;
  while (i < argc) {
    string arg = argv[i];
    if (arg == "--name" || arg == "-n") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.exp_name;
      i++;
    } else if (arg == "--train_file") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.train_file;
      i++;
    } else if (arg == "--dev_file") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.dev_file;
      i++;
    } else if (arg == "--test_file") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.test_file;
      i++;
    } else if (arg == "--words_file") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.words_file;
      i++;
    } else if (arg == "--model_file") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.model_file;
      i++;
    } else if (arg == "--layers") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.layers;
      i++;
    } else if (arg == "--action_dim") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.action_dim;
      i++;
    } else if (arg == "--input_dim") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.input_dim;
      i++;
    } else if (arg == "--pretrained_dim") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.pretrained_dim;
      i++;
    } else if (arg == "--pos_dim") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.pos_dim;
      i++;
    } else if (arg == "--rel_dim") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.rel_dim;
      i++;
    } else if (arg == "--bilstm_input_dim") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.bilstm_input_dim;
      i++;
    } else if (arg == "--bilstm_hidden_dim") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.bilstm_hidden_dim;
      i++;
    } else if (arg == "--attention_hidden_dim") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.attention_hidden_dim;
      i++;
    } else if (arg == "--state_input_dim") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.state_input_dim;
      i++;
    } else if (arg == "--state_hidden_dim") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.state_hidden_dim;
      i++;
    } else if (arg == "--train_methods") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.train_methods;
      i++;
    } else if (arg == "--unk_strategy") {
      params.unk_strategy = true;
    } else if (arg == "--use_pos") {
      params.use_pos = true;
    } else if (arg == "--train") {
      params.train = true;
    } else if (arg == "--debug") {
      params.debug = true;
    } else  if (arg == "--unk_prob") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.unk_prob;
      i++;
    } else  if (arg == "--pdrop") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.pdrop;
      i++;
    }
    i++;
  }
}
