import os
import torch
import torch.nn as nn
from util.model.initialize_load_r21d import initialize_load_model
from util.loader.mtview.loader import create_dataloader
from util.checkpoint.checkpoint_train import checkpoint_train
from util.checkpoint.checkpoint_test  import checkpoint_test
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import sys, traceback
from shutil import copyfile
from util.checkpoint.minLoss import find_epo_test       #TODO: This is for finding the epo for calling whole_eval in the end. Remove to make it happen only in checkpoint_test


def whole_eval(forward, model, device, dataloader, **kwargs):
    model.eval()
    one_epoch = []
    for i, batch in enumerate(dataloader):
        loss, one_batch = forward(batch, model, device, return_one_batch=True, **kwargs)
        one_epoch.append(one_batch)

    return one_epoch


def batch_train(forward, model, batch, i, device, optimizer, loss_accu, loss_accu_period, **kwargs):

    loss, _ = forward(batch, model, device, return_one_batch=False, **kwargs)

    loss.backward()
    loss_accu += loss.item()
    one_batch = 0
    if (i+1) % loss_accu_period == 0:
        loss_accu = loss_accu / loss_accu_period
        optimizer.step()
        optimizer.zero_grad()
        one_batch = loss_accu
        loss_accu = 0

    return model, one_batch, loss_accu


class Backbone():
    def __init__(self, save_folder, epo_iter, forward, task='clas',
                 create_dataloader=create_dataloader, initialize_load_model=initialize_load_model,
                 batch_train=batch_train, whole_eval=whole_eval,
                 checkpoint_train=checkpoint_train, checkpoint_test = checkpoint_test,
                 optimizer=torch.optim.Adam, lr=1e-3, wd=0, loss_accu_period=1,
                 log_val_only=True, eval_per_iter=False):

        assert task in ['clas', 'seg', 'regres']

        # Variables
        self.save_folder = save_folder
        self.epo_iter = epo_iter
        self.task = task
        self.optimizer = optimizer
        self.lr = lr
        self.wd = wd
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loss_accu_period = loss_accu_period
        self.log_val_only = log_val_only
        self.eval_per_iter = eval_per_iter

        # Functions
        self.forward = forward
        self.batch_train = batch_train
        self.whole_eval = whole_eval
        self.create_dataloader = create_dataloader
        self.initialize_load_model = initialize_load_model
        self.checkpoint_train = checkpoint_train
        self.checkpoint_test = checkpoint_test

    def run(self, workflow='complete', subfolder='default', **kwargs):
        try:
            n = datetime.now()
            assert workflow in ['complete', 'train', 'test']
            dataloader_train = self.create_dataloader(mode='train', **kwargs)
            dataloader_val   = self.create_dataloader(mode='val', **kwargs)
            dataloader_test  = self.create_dataloader(mode='test', **kwargs)
            print('Dataloader initialization time: ', datetime.now() - n, '=' * 100)
            n = datetime.now()

            # [Training]
            if (workflow == 'complete') | (workflow == 'train'):
                mode = 'train'
                num_batch, num_epo = len(dataloader_train), self.epo_iter.stop
                assert num_batch % self.loss_accu_period == 0

                model, param = self.initialize_load_model(mode=mode, device=self.device, **kwargs)
                opt = self.optimizer(param, lr=self.lr, weight_decay=self.wd)
                opt.zero_grad() # Do zero_grad() here because of the gradient accumulation feature
                writer = SummaryWriter(self.save_folder)
                print('Training initialization time: ', datetime.now() - n,'='*100)
                n = datetime.now()

                for epo in tqdm(self.epo_iter, ncols=0):
                    one_epoch_train, loss_accu = [], 0
                    for i, batch in enumerate(dataloader_train):
                        wt = (datetime.now() - n).total_seconds()
                        if wt>2: print('Waiting time ', wt)
                        # itr counts the number of updates. When loss accumulation is used, itr would be different to i.
                        itr = i//self.loss_accu_period + epo * (num_batch//self.loss_accu_period)
                        model, one_batch_train, loss_accu = self.batch_train(self.forward, model, batch, i, self.device,
                                                                             opt, loss_accu, self.loss_accu_period, **kwargs)
                        # Log training loss of one batch for calculation of averaged training loss later on.
                        if (i+1)%self.loss_accu_period == 0:
                            one_epoch_train.append(one_batch_train)
                        # [Validation]
                        if (  (self.eval_per_iter & ((i+1)%self.loss_accu_period == 0)) or
                              ((not self.eval_per_iter) & ((i+1)/self.loss_accu_period == num_batch//self.loss_accu_period))  ):
                            # print('time per each training epoch ', datetime.now() - n)
                            # n = datetime.now()
                            one_epoch_val = self.whole_eval(self.forward, model, self.device, dataloader_val, **kwargs)
                            if self.log_val_only:
                                one_epoch_test = []
                            else:
                                one_epoch_test = self.whole_eval(self.forward, model, self.device, dataloader_test, **kwargs)

                            is_first_update = (self.eval_per_iter & (itr==0)) | ((not self.eval_per_iter) & (epo==0))
                            is_last_update = itr == ((num_batch//self.loss_accu_period)*num_epo)-1
                            self.checkpoint_train(itr, one_epoch_train, one_epoch_val, one_epoch_test, model,
                                                  self.save_folder, subfolder, epo, is_first_update, is_last_update,
                                                  writer, self.log_val_only, self.task, **kwargs)
                            one_epoch_train = []
                            # print('time per each training checkpoint ', datetime.now() - n)
                            # n = datetime.now()
                        n = datetime.now()

                writer.flush()
                writer.close()

            # [Testing]
            if (workflow == 'complete') | (workflow == 'test'):
                mode = 'test'
                model_subpath = 'train/model_val_min.pth'
                epo = find_epo_test(self.save_folder, self.epo_iter)
                epo=0
                model, _ = self.initialize_load_model(mode=mode, model_path=self.save_folder + model_subpath,
                                                      device=self.device, **kwargs)
                one_epoch = self.whole_eval(self.forward, model, self.device, dataloader_test, **kwargs)
                self.checkpoint_test(one_epoch, model, self.save_folder, subfolder, self.epo_iter, self.task, **kwargs)


        except KeyboardInterrupt:
            print('Interrupted at epo ', epo, )
            save_subfolder = self.save_folder + 'train'
            copyfile(save_subfolder + '/log_tmp.csv', save_subfolder + '/log_' + str(epo) + '.csv')
            epo_iter = range(epo+1)     #TODO: hide all these in checkpoint_test? the epo for last line and this line are either (epo, epo+1) or (epo-1,epo)
            mode = 'test'
            model_subpath = 'train/model_val_min.pth'
            epo = find_epo_test(self.save_folder, epo_iter)
            model, _ = self.initialize_load_model(mode=mode, model_path=self.save_folder + model_subpath,
                                                  device=self.device, **kwargs)
            one_epoch = self.whole_eval(self.forward, model, self.device, dataloader_test, **kwargs)
            self.checkpoint_test(one_epoch, model, self.save_folder, subfolder, epo_iter, self.task, **kwargs)

            sys.exit(0)