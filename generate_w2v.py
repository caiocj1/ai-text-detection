import ast

import pandas as pd
import gensim, logging
import os
import yaml
import json
from torchtext.data import get_tokenizer

if __name__ == '__main__':
    train_file = open('data/train_set.json')
    train_json = json.load(train_file)

    config_path = os.path.join(os.getcwd(), 'config.yaml')
    with open(config_path) as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    word2vec_params = params['Word2VecParams']

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    tokenizer = get_tokenizer("spacy", language="en_core_web_sm")

    class Corpus(object):
        def __init__(self, json):
            self.json = json

        def __iter__(self):
            for sample in self.json:
                yield tokenizer(sample['text'])

    corpus = Corpus(train_json)
    model = gensim.models.Word2Vec(corpus,
                                   min_count=1,
                                   vector_size=word2vec_params['vector_size'],
                                   workers=4,
                                   epochs=word2vec_params['num_epochs'])

    model.save('models/word2vec.model')
    
    print('Saved w2v model')