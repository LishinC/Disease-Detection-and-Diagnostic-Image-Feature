import os
import numpy as np
import torch
from util.checkpoint.create_header import create_header_clas, create_header_seg, create_header_regres
from util.eval.eval import one_epoch_avg_clas, one_epoch_avg_seg, one_epoch_avg_regres
from util.checkpoint.minLoss import find_epo_test


def checkpoint_test(one_epoch, model, save_folder, subfolder, epo_iter,
               task, find_epo_test=find_epo_test, header_train=None, header_eval=None, **kwargs):

    mode = 'test'
    create_header = globals()['create_header_'+task]
    one_epoch_avg = globals()['one_epoch_avg_'+task]
    if subfolder == 'default': subfolder = mode
    save_subfolder = save_folder + subfolder
    os.makedirs(save_subfolder, exist_ok=True)

    epo = find_epo_test(save_folder, epo_iter, **kwargs)  # Here the epo might actually be itr since the log might be per every update
    one_epoch_avg = one_epoch_avg(one_epoch)
    multi_epo = create_header(mode, None, header_train, header_eval)
    multi_epo = np.concatenate([multi_epo, one_epoch_avg], axis=0)
    np.savetxt(save_subfolder + '/prediction_' + str(epo) + '.csv', np.asarray(one_epoch), fmt='%s', delimiter=',')
    np.savetxt(save_subfolder + '/performance_' + str(epo) + '.csv', np.asarray(multi_epo), fmt='%s', delimiter=',')

    print('Epoch: ', epo, '| ', mode, ' | performance: ', one_epoch_avg, '\n')