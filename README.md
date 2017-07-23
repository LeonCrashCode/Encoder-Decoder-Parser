This implementation is based on the [dynet2.0 library](https://github.com/clab/dynet) for this software to function. The paper is "Encoder-Decoder Shift-Reduce Syntactic Parsing".

### Building

    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen
    make    

Add -DBACKEND=cuda to cmake for GPU compile. 

### Data

Dependency oracle

    python ./scripts/get_dependency_standard_oracle.py [training data in CoNLL format] [training data in CoNLL format] > [training dep oracle]
    python ./scripts/get_dependency_standard_oracle.py [training data in CoNLL format] [depelopment data in CoNLL format] > [depelopment dep oracle]
    python ./scripts/get_dependency_standard_oracle.py [training data in CoNLL format] [test data in CoNLL format] > [test dep oracle]

Constituent oracle

    python ./scripts/get_constituent_topdown_oracle.py [training data in bracketed format] [training data in bracketed format] > [training con oracle]
    python ./scripts/get_constituent_topdown_oracle.py [training data in bracketed format] [development data in bracketed format] > [development con oracle]
    python ./scripts/get_constituent_topdown_oracle.py [training data in bracketed format] [test data in bracketed format] > [test con oracle]
    
### Training

Dependency parsing

    ./dependency-parser --dynet-mem 2400 --train_file [training dep oracle] --dev_file [development dep oracle] --words_file [pretrained word embeddings] --layers 2 --action_dim 40 --input_dim 64 --pos_dim 6 --pretrained_dim 100 --rel_dim 20 --bilstm_input_dim 100 --bilstm_hidden_dim 200 --attention_hidden_dim 50 --train_methods 3 --train --use_pos --unk_strategy --unk_prob 0.2 --pdrop 0.3 

Constituent parsing

    ./constituent-parser -T [training con oracle] -d [development con oracle] -C [development data in bracketed format]--layers 2 --input_dim 64 --pos_dim 6 --bilstm_input_dim 100 --bilstm_hidden_dim 200 --attention_hidden_dim 50 -w [pretrained word embeddings] --pretrained_dim 100 --action_dim 40 -t -P -D 0.5 --dynet-mem 1700

### Decoding

Dependency parsing

    ./dependency-parser --dynet-mem 2400 --train_file [training dep oracle] --test_file [test dep oracle] --words_file [pretrained word embeddings] --layers 2 --action_dim 40 --input_dim 64 --pos_dim 6 --pretrained_dim 100 --rel_dim 20 --bilstm_input_dim 100 --bilstm_hidden_dim 200 --attention_hidden_dim 50 --train_methods 3 --use_pos

Constituent parsing

    ./constituent-parser -T [training con oracle] -p [test con oracle] -C [test data in bracketed format] --layers 2 --input_dim 64 --pos_dim 6 --bilstm_input_dim 100 --bilstm_hidden_dim 200 --attention_hidden_dim 50 -w [pretrained word embeddings] --pretrained_dim 100 --action_dim 40 -P --dybet-mem 1700 -m [model]

### Cite

    @article{liu2017encoder,
        title={Encoder-Decoder Shift-Reduce Syntactic Parsing},
        author={Liu, Jiangming and Zhang, Yue},
        journal={arXiv preprint arXiv:1706.07905},
        year={2017}
    


