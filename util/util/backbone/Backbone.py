import os
import torch
import torch.nn as nn
from util.model.initialize_load_r21d import initialize_load_model
from util.loader.LUMC_A4C.loader_vid import create_dataloader
from util.checkpoint.checkpoint_train import checkpoint_train
from util.checkpoint.checkpoint_test  import checkpoint_test
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import sys, traceback
from shutil import copyfile


class Backbone():
    def __init__(self, save_folder, epo_iter, forward, task='clas',
                 create_dataloader=create_dataloader, initialize_load_model=initialize_load_model,
                 checkpoint_train=checkpoint_train, checkpoint_test = checkpoint_test,
                 optimizer=torch.optim.Adam, lr=1e-4, wd=1e-8, loss_accu_period=1,
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
        self.create_dataloader = create_dataloader
        self.initialize_load_model = initialize_load_model
        self.checkpoint_train = checkpoint_train
        self.checkpoint_test = checkpoint_test


    def batch_train(self, model, batch, i, opt, loss_running_accu, loss_accu_period, **kwargs):
        loss, _ = self.forward(batch, model, self.device, return_one_batch=False, **kwargs)
        loss.backward()
        loss_running_accu += loss.item()
        one_batch = 0
        if (i + 1) % loss_accu_period == 0:
            loss_running_accu = loss_running_accu / loss_accu_period
            opt.step()
            opt.zero_grad()

            one_batch = loss_running_accu
            loss_running_accu = 0

        return model, one_batch, loss_running_accu, opt


    def whole_eval(self, model, dataloader, **kwargs):
        model.eval()
        one_epoch = []
        for i, batch in enumerate(dataloader):
            loss, one_batch = self.forward(batch, model, self.device, return_one_batch=True, **kwargs)
            one_epoch.append(one_batch)

        return one_epoch


    def run_val(self, i, itr, epo, num_batch, num_epo, subfolder, one_epoch_train,
                model, dataloader_val, dataloader_test, **kwargs):
        one_epoch_val = self.whole_eval(model, dataloader_val, **kwargs)
        one_epoch_test = [] if self.log_val_only else self.whole_eval(model, dataloader_test, **kwargs)

        is_first_update = (self.eval_per_iter & (itr == 0)) | ((not self.eval_per_iter) & (epo == 0))
        is_last_update = itr == ((num_batch // self.loss_accu_period) * num_epo) - 1
        self.checkpoint_train(itr, one_epoch_train, one_epoch_val, one_epoch_test, model,
                              self.save_folder, subfolder, epo, is_first_update, is_last_update,
                              self.writer, self.log_val_only, self.task, **kwargs)


    def run_test(self, dataloader_test, subfolder, **kwargs):
        model, _ = self.initialize_load_model(mode='test', model_path=self.save_folder+'train/model_val_min.pth', device=self.device, **kwargs)
        one_epoch = self.whole_eval(model, dataloader_test, **kwargs)
        self.checkpoint_test(one_epoch, model, self.save_folder, subfolder, self.task, **kwargs)


    def run(self, workflow='complete', subfolder='default', verbose=False, **kwargs):
        try:
            n = datetime.now()
            assert workflow in ['complete', 'train', 'test']
            dataloader_train = self.create_dataloader(mode='train', **kwargs) #Have to be initialized here since kwargs are needed
            dataloader_val   = self.create_dataloader(mode='val', **kwargs)
            dataloader_test  = self.create_dataloader(mode='test', **kwargs)

            ## [Training]
            if (workflow == 'complete') | (workflow == 'train'):
                num_batch, num_epo = len(dataloader_train), self.epo_iter.stop
                assert num_batch % self.loss_accu_period == 0

                model, param = self.initialize_load_model(mode='train', device=self.device, **kwargs)
                opt = self.optimizer(param, lr=self.lr, weight_decay=self.wd)
                opt.zero_grad() # Do zero_grad() here because of the gradient accumulation feature
                self.writer = SummaryWriter(self.save_folder)
                if verbose: print('Training initialization time: ', datetime.now() - n,'='*100)
                n = datetime.now()

                for epo in tqdm(self.epo_iter, ncols=0):
                    one_epoch_train, loss_running_accu = [], 0
                    for i, batch in enumerate(dataloader_train):
                        wt = (datetime.now() - n).total_seconds()
                        if verbose&(wt>2): print('\n Batch loading waiting time ', wt)
                        # itr counts the number of updates. When loss accumulation is used, itr would be different to i.
                        itr = i//self.loss_accu_period + epo * (num_batch//self.loss_accu_period)
                        model, one_batch_train, loss_running_accu, opt = self.batch_train(model, batch, i, opt,
                                                                    loss_running_accu, self.loss_accu_period, **kwargs)
                        # Log training loss of one (full) batch for calculation of averaged training loss later on.
                        if (i+1)%self.loss_accu_period == 0:
                            one_epoch_train.append(one_batch_train)
                            ## [Validation]:
                            # Run validation if eval_per_iter & end of a batch; Or NOT eval_per_iter & end of a epoch
                            if (
                                (self.eval_per_iter & ((i+1)%self.loss_accu_period == 0)) or
                                ((not self.eval_per_iter) & ((i+1)/self.loss_accu_period == num_batch//self.loss_accu_period))
                                ):
                                self.run_val(i, itr, epo, num_batch, num_epo, subfolder, one_epoch_train,
                                             model, dataloader_val, dataloader_test, **kwargs)
                                one_epoch_train = []
                        n = datetime.now()
                self.writer.flush()
                self.writer.close()

            ## [Testing]
            if (workflow == 'complete') | (workflow == 'test'):
                self.run_test(dataloader_test, subfolder, **kwargs)


        except KeyboardInterrupt:
            ## [Test on current best if interrupted]
            print('Interrupted at epo ', epo, )
            # copyfile(self.save_folder + 'train/log_tmp.csv', self.save_folder + 'train/log_' + str(epo) + '.csv')
            # epo_iter = range(epo+1)
            self.run_test(dataloader_test, subfolder, **kwargs)
            sys.exit(0)