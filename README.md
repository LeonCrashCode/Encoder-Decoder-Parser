This implementation is based on the [cnn](https://github.com/clab/dynet) for this software to function. The paper is "Encoder-Decoder Shift-Reduce Syntactic Parsing".

#### Building

    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen
    make    

### Data

The preparation of training, development and test data are the same as the work of [stack-LSTM](https://github.com/clab/lstm-parser) on dependency parsing, and the work of [recurrent neural network grammar](https://github.com/clab/rnng) on constituent parsing. After that, the training, development and test oracle files could be got.

### Training

Dependency parsing

    ./depparser --dynet-mem 2400 --train_file [training oracle] --dev_file [development oracle] --words_file [pretrained word embeddings] --layers 2 --action_dim 40 --input_dim 64 --pos_dim 6 --pretrained_dim 100 --rel_dim 20 --bilstm_input_dim 100 --bilstm_hidden_dim 200 --attention_hidden_dim 50 --train_methods 3 --train --use_pos --unk_strategy --unk_prob 0.2 --pdrop 0.3 

Constituent parsing

    ./conparser -T [training oracle] -d [development oracle] -C [development data in bracketed format]--layers 2 --input_dim 64 --pos_dim 6 --bilstm_input_dim 100 --bilstm_hidden_dim 200 --attention_hidden_dim 50 -w [pretrained word embeddings] --pretrained_dim 100 --action_dim 40 -t -P -D 0.5 --cnn-mem 1700

We provide the trained model [dependency model](https://drive.google.com/open?id=0B1VhP65vISjockpyZTZPci1tcVk) and [constituent model](https://drive.google.com/open?id=0B1VhP65vISjoZ3c5Z2toVkYxaEU).

### Decoding

Dependency parsing

    ./depparser --dynet-mem 2400 --train_file [training oracle] --dev_file [test oracle] --words_file [pretrained word embeddings] --layers 2 --action_dim 40 --input_dim 64 --pos_dim 6 --pretrained_dim 100 --rel_dim 20 --bilstm_input_dim 100 --bilstm_hidden_dim 200 --attention_hidden_dim 50 --train_methods 3 --use_pos

Constituent parsing

    ./conparser -T [training oracle] -p [test oracle] -C [test data in bracketed format] --layers 2 --input_dim 64 --pos_dim 6 --bilstm_input_dim 100 --bilstm_hidden_dim 200 --attention_hidden_dim 50 -w [pretrained word embeddings] --pretrained_dim 100 --action_dim 40 -P --cnn-mem 1700 -m [model]

### Cite

