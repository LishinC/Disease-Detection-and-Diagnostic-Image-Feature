import os
import numpy as np
import torch
from util.checkpoint.create_header import create_header_clas, create_header_seg, create_header_regres
from util.eval.eval import one_epoch_avg_clas, one_epoch_avg_seg, one_epoch_avg_regres


def checkpoint_train(itr, one_epoch_train, one_epoch_val, one_epoch_test, model, save_folder, subfolder, epo, is_first_update, is_last_update, w, log_val_only,
               task, header_train=None, header_eval=None, **kwargs):

    mode = 'train'
    create_header = globals()['create_header_'+task]
    one_epoch_avg = globals()['one_epoch_avg_'+task]
    if subfolder == 'default': subfolder = mode
    save_subfolder = save_folder + subfolder
    os.makedirs(save_subfolder, exist_ok=True)

    # TODO: make this optional. The default setting of log_individual_prediction is for clas, modify. Also uncomment the rename at line 45
    # log_individual_prediction(one_epoch_val, save_subfolder+'/individual_pred_tmp.csv', is_first_update)
    train_avg = np.mean(one_epoch_train)

    header = create_header(mode, log_val_only, header_train, header_eval)
    NUM_METRICS_TO_LOG = len(header[0]) - 2

    if log_val_only:
        # For clas, one_epoch contains appended [ID[0], loss.item(), y_true, y_pred]
        # one_epoch_avg returns numpy array of shape (1,-1) containing [loss, acc, prec, rec, f1]
        # Sor seg, one_epoch contains appended [ID[0], loss, 'dice', 'iou', 'precision', 'recall']
        # one_epoch_avg returns its average, with shape (1, -1)
        val_avg = one_epoch_avg(one_epoch_val)
        one_epoch_avg = [itr, train_avg] + list(val_avg.reshape(-1,))[:NUM_METRICS_TO_LOG]
    else:
        val_avg = one_epoch_avg(one_epoch_val)
        test_avg = one_epoch_avg(one_epoch_test)
        one_epoch_avg = [itr, train_avg] + list(val_avg.reshape(-1,))[:NUM_METRICS_TO_LOG/2] + list(test_avg.reshape(-1,))[:NUM_METRICS_TO_LOG/2]

    # The usual dir to log per itr train/val/test loss csv csv_name=root + 'train/log_iter.csv'
    has_min_val = logging(one_epoch_avg, header, w, save_subfolder+'/log_tmp.csv', is_first_update, log_val_only,
                          **kwargs)

    if has_min_val:
        torch.save(model.state_dict(), save_subfolder + '/model_val_min.pth')

    if is_last_update:
        os.rename(save_subfolder + '/log_tmp.csv', save_subfolder + '/log_' + str(epo) + '.csv')
        # os.rename(save_subfolder + '/individual_pred_tmp.csv', save_subfolder + '/individual_pred_' + str(epo) + '.csv')

    print('Epoch: ', epo, '| ', mode, ' | performance: ', one_epoch_avg, '\n')


def logging(one_epoch_avg, header, w, csv_name, is_first_update, log_val_only, **kwargs):
    def compare(one_epoch_avg, multi_epo):
        current = one_epoch_avg[2]
        history = min(multi_epo[1:,2].astype('float'))
        if current < history:
            has_min_val = True
        else:
            has_min_val = False
        return has_min_val

    # Write to tensorboard
    itr = one_epoch_avg[0]
    assert len(header[0]) == len(one_epoch_avg)

    for i in range(1,len(header[0])):
        w.add_scalar(header[0,i], one_epoch_avg[i], itr)

    # Write to csv file
    if is_first_update:
        multi_epo = header
        has_min_val = True
    else:
        multi_epo = np.genfromtxt(csv_name, dtype='str', delimiter=',')
        has_min_val = compare(one_epoch_avg, multi_epo)
    one_epoch_avg = np.asarray(one_epoch_avg).reshape(1, -1)
    multi_epo = np.concatenate([multi_epo, one_epoch_avg], axis=0)
    np.savetxt(csv_name, np.asarray(multi_epo), fmt='%s', delimiter=',')
    return has_min_val


def log_individual_prediction(one_epoch, csv_name, is_first_update):
    if is_first_update:
        multi_epo = np.asarray(one_epoch)[:,2].reshape(1, -1)
        # multi_epo = np.asarray(one_epoch)[:,0].reshape(1, -1)     # For seg
    else:
        multi_epo = np.genfromtxt(csv_name, dtype='str', delimiter=',')
    if multi_epo.ndim == 1: multi_epo = multi_epo.reshape(1, -1)
    one_epoch = np.asarray(one_epoch)[:,4].reshape(1, -1)
    # one_epoch = np.asarray(one_epoch)[:,1].reshape(1, -1)         # Fpr seg
    multi_epo = np.concatenate([multi_epo, one_epoch],axis=0)
    np.savetxt(csv_name, np.asarray(multi_epo), fmt='%s', delimiter=',')
