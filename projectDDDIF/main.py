import os
import torch
import torch.nn as nn
# import numpy as np
from util.backbone.Backbone import Backbone
from util.loader.LUMC_A4C.loader_vid import create_dataloader
from util.model.initialize_load_r21d import initialize_load_model
from analyze import analyze


def forward(batch, model, device, return_one_batch, criterion=nn.CrossEntropyLoss(), **kwargs):

    X, Y, ID = batch
    X = X.to(device)
    Y = Y.to(device).view(-1).long()
    out_logit = model(X)
    loss = criterion(out_logit, Y)

    if return_one_batch:
        sf = nn.Softmax(1)
        output = sf(out_logit)
        y_pred = output.argmax(1).item()
        y_true = Y.item()
        one_batch = [ID[0], loss.item(), y_true, y_pred, output[:, 1].item()]

        # analyze(X, [y_true], model, 1, 'model/DeepLIFT/', ID[0])

        return loss, one_batch
    else:
        return loss, []


if __name__ == '__main__':

    root = './model/regurg/'
    kwargs = {'in_channel': 3, 'out_channel': 3}

    b = Backbone(root, range(200), forward,
                 create_dataloader=create_dataloader, initialize_load_model=initialize_load_model)
    b.run(**kwargs)
