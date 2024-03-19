import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from functools import partial
from logging import getLogger
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from myencoder import MyEncoder


class MyModel(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.st_start_id0, self.st_start_id1, self.st_start_id2, self.st_end_id1, self.st_end_id2 = -1
        self.st_tower = MyEncoder(dim_in=2, dim_out=2)

    def forward(self, batch):
        return None


    def calculate_loss(self, batch, batches_seen=None, lap_mx=None):
        y_true = batch['y']
        y_predicted = self.predict(batch, lap_mx)
        return self.calculate_loss_without_predict(y_true, y_predicted, batches_seen)

    def predict(self, batch, lap_mx=None):
        return self.forward(batch, lap_mx)
