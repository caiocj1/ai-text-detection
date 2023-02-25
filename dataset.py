import pandas as pd
import numpy as np
import os
import yaml
import ast
import datetime
from yaml import SafeLoader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import sklearn
import sklearn.cluster
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from typing import Optional
from collections import defaultdict
import json
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch
import torch.nn as nn

class TextDataset(Dataset):
    def __init__(self, X, y, tokenizer, vocab):
        super(TextDataset, self).__init__()
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __getitem__(self, item):
        return {'text': self.vocab.lookup_indices(self.tokenizer(self.X[item]['text'])), 'labels': self.y[item]}

    def __len__(self):
        return len(self.X)

class TextDataModule(LightningDataModule):
    def __init__(self,
            batch_size: int = 32,
            num_workers: int = 0,
            max_samples: int = None):
        super().__init__()

        train_file = open('data/train_set.json')
        test_file = open('data/test_set.json')
        self.train_json = json.load(train_file)
        self.test_json = json.load(test_file)

        # Save hyperparemeters
        self.save_hyperparameters(logger=False)

        # Read config file
        self.read_config()

        # Get tokenizer
        self.tokenizer = get_tokenizer("spacy", language="en_core_web_sm")

        # Get train and test data
        X, y = self.parse_json(self.train_json)
        self.X_train, self.X_val, self.y_train, self.y_val = sklearn.model_selection.train_test_split(X, y, test_size=0.1,
                                                                                  random_state=0)

        self.X_test, self.y_test = self.parse_json(self.test_json)

        # Build vocabulary
        def yield_tokens(text_list):
            for sample in text_list:
                yield self.tokenizer(sample['text'])
        self.vocab = build_vocab_from_iterator(yield_tokens(self.X_train), specials=["<bos>", "<eos>", "<unk>", "<pad>"])
        self.vocab.set_default_index(2)

    def read_config(self):
        """
        Read configuration file with hyperparameters.
        :return: None
        """
        config_path = os.path.join(os.getcwd(), 'config.yaml')
        with open(config_path) as f:
            params = yaml.load(f, Loader=SafeLoader)
        dataset_params = params['DatasetParams']

    def setup(self, stage: str = None):
        """
        Build data sets for training or prediction.
        :param stage: 'fit' for training, 'predict' for prediction
        :return: None
        """
        if stage == 'fit':
            self.data_train = TextDataset(self.X_train, self.y_train, self.tokenizer, self.vocab)
            self.data_val = TextDataset(self.X_val, self.y_val, self.tokenizer, self.vocab)

        elif stage == 'predict':
            self.data_predict = TextDataset(X, y, self.tokenizer, self.vocab)

    def train_dataloader(self):
        """
        Uses train dictionary (output of format_X) to return train DataLoader, that will be fed to pytorch lightning's
        Trainer.
        :return: train DataLoader
        """
        return DataLoader(dataset=self.data_train,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          collate_fn=self.collate_fn,
                          shuffle=True)

    def val_dataloader(self):
        """
        Uses val dictionary (output of format_X) to return val DataLoader, that will be fed to pytorch lightning's
        Trainer.
        :return: train DataLoader
        """
        return DataLoader(dataset=self.data_val,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          collate_fn=self.collate_fn,
                          shuffle=False)

    def predict_dataloader(self):
        """
        Uses predict dictionary (output of format_X) to return predict DataLoader, that will be fed to pytorch
        lightning's Trainer.
        :return: predict DataLoader
        """
        return DataLoader(dataset=self.data_predict,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          collate_fn=self.collate_fn,
                          shuffle=False)

    def parse_json(self, json):
        y = []
        X = []
        for sample in json:
            X.append({'id': sample['id'], 'text': sample['text']})
            y.append(sample['label'] if 'label' in sample else -1)

        return X, y

    def collate_fn(self, batch):
        batch_size = len(batch)
        texts = [torch.tensor(sample['text']) for sample in batch]
        sequences = nn.utils.rnn.pad_sequence(texts, padding_value=3)
        labels = torch.tensor([sample['labels'] for sample in batch])
        return sequences.long(), labels.long()