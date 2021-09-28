import torch
import torch.nn as nn
import torchvision


def initialize_load_model(mode, model_path='scratch', in_channel=3, out_channel=3, device="cuda", **kwargs):
    def r21d(in_channel, out_channel, pretrain=False, echo_pretrain=False):
        model = torchvision.models.video.__dict__["r2plus1d_18"](pretrained=pretrain)
        if in_channel == 1: model.stem[0] = nn.Conv3d(1, 45, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        model.fc = nn.Linear(model.fc.in_features, out_channel)
        return model

    if model_path == 'pretrain':
        model = r21d(in_channel, out_channel, pretrain=True)
    elif model_path == 'scratch':
        model = r21d(in_channel, out_channel)
    else:
        model = r21d(in_channel, out_channel)
        model.load_state_dict(torch.load(model_path))

    model.to(device)
    param = model.parameters()
    if mode == 'train':
        model.train()
    else:
        model.eval()

    return model, param