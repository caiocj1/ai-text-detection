import torch.optim
from pytorch_lightning import LightningModule
import torch.nn as nn
import os
import yaml
from yaml import SafeLoader
from collections import OrderedDict
import math

from models.blocks import PositionalEncoding

class TransformerModel(LightningModule):

    def __init__(self,
                 vocab_size,
                 d_model,
                 nhead=8,
                 dim_feedforward=2048,
                 num_layers=6,
                 dropout=0.1,
                 ):
        super(TransformerModel, self).__init__()
        self.read_config()

        self.emb = nn.Embedding(vocab_size, d_model)

        self.pe = PositionalEncoding(d_model=d_model, dropout=dropout, vocab_size=vocab_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Linear(d_model, 2)

        self.d_model = d_model

    def read_config(self):
        """
        Read configuration file with hyperparameters.
        :return: None
        """
        config_path = os.path.join(os.getcwd(), './config.yaml')
        with open(config_path) as f:
            params = yaml.load(f, Loader=SafeLoader)
        model_params = params['ModelParams']


    def training_step(self, batch, batch_idx):
        """
        Perform train step.
        :param batch: tuple (X, y), where the shape of X is (batch_size, 23) and of y is (batch_size)
        :param batch_idx: index of current batch, non applicable here
        :return: mean loss
        """
        loss, metrics = self._shared_step(batch)

        loss = loss.mean()
        self.log_metrics(metrics, 'train')
        self.log('loss_train', loss, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform validation step.
        :param batch: tuple (X, y), where the shape of X is (batch_size, 23) and of y is (batch_size)
        :param batch_idx: index of current batch, non applicable here
        :return: mean loss
        """
        loss, metrics = self._shared_step(batch)

        loss = loss.mean()
        self.log_metrics(metrics, 'val')
        self.log('loss_val', loss, on_step=False, on_epoch=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        """
        Perform test step.
        :param batch: tuple (X, y), where the shape of X is (batch_size, 23) and of y is (batch_size)
        :param batch_idx: index of current batch, non applicable here
        :return: mean loss
        """
        loss, metrics = self._shared_step(batch)

        loss = loss.mean()
        return loss

    def _shared_step(self, batch):
        """
        Get predictions, calculate loss and eventually useful metrics (here the only metric is MAE which is the same as
        the loss function).
        :param batch: tuple (X, y), where the shape of X is (batch_size, 23) and of y is (batch_size)
        :return: loss: tensor of shape (batch_size), metrics: dictionary with metrics
        """
        prediction = self.forward(batch)

        loss = self.calc_loss(prediction, batch[1])

        metrics = self.calc_metrics(prediction, batch[1])

        return loss, metrics

    def forward(self, batch):
        """
        Pass text embedding through convolutional layers. Concatenate result with base features and pass through final
        MLP to get predictions of a batch.
        :param batch: tuple (X, y), where the shape of X is (batch_size, 23) and of y is (batch_size)
        :return: predictions: tensor of shape (batch_size)
        """
        x = batch[0].transpose(0, 1)
        mask = (x != 3)[..., None]

        x = self.emb(x) * math.sqrt(self.d_model)
        x = self.pe(x)
        x = self.transformer_encoder(x)
        x = (x * mask.float()).sum(0) / mask.float().sum(0)
        x = self.classifier(x)

        return x

    def calc_loss(self, prediction, target):
        """
        Calculate L1 loss.
        :param prediction: tensor of predictions (batch_size)
        :param target: tensor of ground truths (batch_size)
        :return: tensor of losses (batch_size)
        """
        loss_func = nn.CrossEntropyLoss(reduction='none')

        loss = loss_func(prediction.float(), target)

        return loss

    def configure_optimizers(self):
        """
        Selection of gradient descent algorithm and learning rate scheduler.
        :return: optimizer algorithm, learning rate scheduler
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=72, eta_min=5e-5)

        return [optimizer], []

    def calc_metrics(self, prediction, target):
        """
        Calculate useful metrics. Not applicable here (MAE is already loss).
        :param prediction: tensor of predictions (batch_size)
        :param target: tensor of ground truths (batch_size)
        :return: metric dictionary
        """
        metrics = {}

        prediction = torch.argmax(prediction, dim=-1)
        metrics['accuracy'] = (prediction == target).float().mean()

        return metrics

    def log_metrics(self, metrics: dict, type: str):
        """
        Log metrics on Tensorboard.
        :param metrics: metric dictionary
        :param type: check if training or validation metrics
        :return: None
        """
        on_step = True if type == 'train' else False

        for key in metrics:
            self.log(key + '_' + type, metrics[key], on_step=on_step, on_epoch=True, logger=True)
