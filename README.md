# BiLSTM-CRF Sequence Labeling with Adversarial Training

Implementation of BiLSTM-CRF model with adversarial training.   
Paper: [Robust Multilingual Part-of-Speech Tagging via Adversarial Training](http://arxiv.org/abs/1711.04903) (NAACL 2018).

## Requirements

* Python 2.7
* Theano 1.0
* Lasagne

## Data

* Penn Treebank - Wall Street Journal
* [Universal Dependencies (UD) v1.2](http://universaldependencies.org/)
* [GloVe word embeddings](https://nlp.stanford.edu/projects/glove/)
* [Polyglot word embeddings](https://sites.google.com/site/rmyeid/projects/polyglot)

## Run

Configure and run ``multi_lingual_run_blstm-blstm-crf_pos.sh``.

## Notes

If you use this tool for your work, please consider citing:
```
@InProceedings{Yasunaga&al.18.naacl,
  author =  {Michihiro Yasunaga and Jungo Kasai and Dragomir R. Radev},
  title =   {Robust Multilingual Part-of-Speech Tagging via Adversarial Training},
  year =    {2018},  
  booktitle =   {Proceedings of NAACL},  
  publisher =   {Association for Computational Linguistics},
}
```


## Acknowledgements

This tool uses the following open source component (big thank you to the developers). You can find its source code and license information below.
* LasagneNLP: [https://github.com/XuezheMax/LasagneNLP](https://github.com/XuezheMax/LasagneNLP)
    ([Apache License 2.0](https://github.com/XuezheMax/LasagneNLP/blob/master/LICENSE))
