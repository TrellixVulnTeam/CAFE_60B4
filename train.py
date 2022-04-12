import numpy as np
import os
from torch.utils.data import DataLoader
from vad_datasets import unified_dataset_interface, TrainForegroundDataset3, img_tensor2numpy, img_batch_tensor2numpy, frame_info
from fore_det.inference import init_detector
from fore_det.obj_det_with_motion import imshow_bboxes, getObBboxes, delCoverBboxes, getOfBboxes
from model.resnet import resnet34, student_resnet9
from helper.misc import AverageMeter
import torch
import torch.optim as optim
import torch.nn as nn
from configparser import ConfigParser
from utils import calc_block_idx, varname
import shutil


#  /*-------------------------------------------------Overall parameter setting-----------------------------------------------------*/
cp = ConfigParser()
cp.read("config.cfg")  # To get parameter settings from config.cfg.

dataset_name = cp.get('shared_parameters', 'dataset_name')  # The name of dataset: UCSDped2/avenue/ShanghaiTech.
raw_dataset_dir = cp.get('shared_parameters', 'raw_dataset_dir')  # Fixed: The name of the folder that stores raw VAD datasets.
foreground_extraction_mode = cp.get('shared_parameters', 'foreground_extraction_mode')  # Foreground extraction method: obj_det_with_of.
data_root_dir = cp.get('shared_parameters', 'data_root_dir')  # Fixed: A folder that stores the data such as foreground produced by the program.
mode = cp.get('train_parameters', 'mode')  # Fixed: train.
method = cp.get('shared_parameters', 'method')  # Fixed: the name of method.
try:
    patch_size = cp.getint(dataset_name, 'patch_size')  # Resize the foreground bounding boxes.
    train_block_mode = cp.getint(dataset_name, 'train_block_mode')  # Fixed to 1: not really used.
    motionThr = cp.getfloat(dataset_name, 'motionThr')

    # Define (h_block * w_block) sub-regions of video frames for localized training: train a separated DNN to process each region for avoiding foreground depth variation.
    # Typically set to (1 * 1) (i.e., not really used)
    h_block = cp.getint(dataset_name, 'h_block')  # Typically set to 1.
    w_block = cp.getint(dataset_name, 'w_block')  # Typically set to 1.

    # Set 'train/test_bbox_saved = False' and 'train/test_foreground_saved = False' at first to calculate and store the bounding boxes and foreground,
    # and then set them to True to load the stored bboxes and foreground directly, if the foreground parameters remain unchanged.
    bbox_saved = cp.getboolean(dataset_name, '{}_bbox_saved'.format(mode))
    foreground_saved = cp.getboolean(dataset_name, '{}_foreground_saved'.format(mode))
except:
    raise NotImplementedError

#  /*--------------------------------------------------Foreground localization-----------------------------------------------------*/
config_file = 'fore_det/obj_det_config/yolov3_d53_mstrain-608_273e_coco.py'
checkpoint_file = 'fore_det/obj_det_checkpoints/yolov3_d53_mstrain-608_273e_coco-139f5633.pth'

# set dataset for foreground extraction
# raw dataset
dataset = unified_dataset_interface(dataset_name=dataset_name, dir=os.path.join('raw_datasets', dataset_name), context_frame_num=1, mode=mode, border_mode='hard')
# optical flow dataset
dataset2 = unified_dataset_interface(dataset_name=dataset_name, dir=os.path.join('optical_flow', dataset_name), context_frame_num=1, mode=mode, border_mode='hard', file_format='.npy')
if not bbox_saved:
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    all_bboxes = list()
    for idx in range(dataset.__len__()):
        batch, _ = dataset.__getitem__(idx)
        batch2, _ = dataset2.__getitem__(idx)
        print('Extracting bboxes of {}-th frame in total {} frame'.format(idx + 1, dataset.__len__()))
        cur_img = img_tensor2numpy(batch[1])
        cur_of = img_tensor2numpy(batch2[1])

        if foreground_extraction_mode == 'obj_det_with_of':
            # A coarse detection of bboxes by pretrained object detector
            ob_bboxes = getObBboxes(cur_img, model, dataset_name, verbose=False)
            ob_bboxes = delCoverBboxes(ob_bboxes, dataset_name)

            # visual object detection bounding boxes with covering filter
            # imshow_bboxes(cur_img, ob_bboxes, win_name='del_cover_bboxes')

            # further foreground detection by optical flow
            of_bboxes = getOfBboxes(cur_of, cur_img, ob_bboxes, dataset_name, verbose=False)

            if of_bboxes.shape[0] > 0:
                cur_bboxes = np.concatenate((ob_bboxes, of_bboxes), axis=0)
            else:
                cur_bboxes = ob_bboxes
        else:
            raise NotImplementedError

        # imshow_bboxes(cur_img, cur_bboxes)
        all_bboxes.append(cur_bboxes)
    np.save(os.path.join(dataset.dir, 'bboxes_{}_{}.npy'.format(mode, foreground_extraction_mode)), all_bboxes)
    print('bboxes for training data saved!')
elif not foreground_saved:
    all_bboxes = np.load(os.path.join(dataset.dir, 'bboxes_{}_{}.npy'.format(mode, foreground_extraction_mode)), allow_pickle=True)
    print('bboxes for training data loaded!')

# /*--------------------------------------------------Foreground extraction--------------------------------------------------------*/
border_mode = cp.get(method, 'border_mode')
if not foreground_saved:
    context_frame_num = cp.getint(method, 'context_frame_num')
    context_of_num = cp.getint(method, 'context_of_num')

    file_format1 = frame_info[dataset_name][2]
    file_format2 = '.npy'

    dataset = unified_dataset_interface(dataset_name=dataset_name, dir=os.path.join('raw_datasets', dataset_name),
                                        context_frame_num=context_frame_num, mode=mode,
                                        border_mode=border_mode, all_bboxes=all_bboxes, patch_size=patch_size,
                                        file_format=file_format1)
    dataset2 = unified_dataset_interface(dataset_name=dataset_name, dir=os.path.join('optical_flow', dataset_name),
                                         context_frame_num=context_of_num, mode=mode,
                                         border_mode=border_mode, all_bboxes=all_bboxes, patch_size=patch_size,
                                         file_format=file_format2)

    h_step, w_step = frame_info[dataset_name][0] / h_block, frame_info[dataset_name][1] / w_block

    # Create folders to store foreground.
    fore_folder_name = '{}_{}_foreground_{}'.format(dataset_name, mode, foreground_extraction_mode)
    if os.path.exists(os.path.join(data_root_dir, fore_folder_name)):
        shutil.rmtree(os.path.join(data_root_dir, fore_folder_name))
    for h in range(h_block):
        for w in range(w_block):
            raw_fore_dir = os.path.join(data_root_dir, fore_folder_name, 'block_{}_{}'.format(h, w), 'raw')
            of_fore_dir = os.path.join(data_root_dir, fore_folder_name, 'block_{}_{}'.format(h, w), 'of')
            os.makedirs(raw_fore_dir)
            os.makedirs(of_fore_dir)

    fore_idx = 0
    for idx in range(dataset.__len__()):
        batch, _ = dataset.__getitem__(idx)
        batch2, _ = dataset2.__getitem__(idx)

        print('Extracting foreground in {}-th batch, {} in total'.format(idx + 1, dataset.__len__() // 1))

        cur_bboxes = all_bboxes[idx]
        if len(cur_bboxes) > 0:
            batch = img_batch_tensor2numpy(batch)
            batch2 = img_batch_tensor2numpy(batch2)

            if len(batch2.shape) == 4:
                mag = np.sum(np.sum(np.sum(batch2 ** 2, axis=3), axis=2), axis=1)
            else:
                mag = np.mean(np.sum(np.sum(np.sum(batch2 ** 2, axis=4), axis=3), axis=2), axis=1)

            for idx_bbox in range(cur_bboxes.shape[0]):

                if mag[idx_bbox] > motionThr:
                    all_blocks = calc_block_idx(cur_bboxes[idx_bbox, 0], cur_bboxes[idx_bbox, 2], cur_bboxes[idx_bbox, 1], cur_bboxes[idx_bbox, 3], h_step, w_step, mode=train_block_mode)
                    for (h_block_idx, w_block_idx) in all_blocks:
                        np.save(os.path.join(data_root_dir, fore_folder_name, 'block_{}_{}'.format(h_block_idx, w_block_idx), 'raw', '{}.npy'.format(fore_idx)), batch[idx_bbox])
                        np.save(os.path.join(data_root_dir, fore_folder_name, 'block_{}_{}'.format(h_block_idx, w_block_idx), 'of', '{}.npy'.format(fore_idx)), batch2[idx_bbox])
                        fore_idx += 1

    print('foreground for training data saved!')

#  /*-----------------------------------------------Normal video event modeling---------------------------------------------------*/
if method == 'CAFE':
    loss_func = nn.MSELoss()  # Training loss function: MSE loss.
    score_func = nn.MSELoss(reduce=False)  # Anomaly scoring function: MSE score.
    epochs = cp.getint(method, 'epochs')  # Training epochs.
    batch_size = cp.getint(method, 'batch_size')
    lambda_consistency = cp.getfloat(method, 'lambda_consistency')
    border_mode = cp.get(method, 'border_mode')
    if border_mode == 'predict':
        tot_frame_num = cp.getint(method, 'context_frame_num') + 1
        cur_frame_idx = tot_frame_num - 1
    else:
        tot_frame_num = 2 * cp.getint(method, 'context_frame_num') + 1
        cur_frame_idx = tot_frame_num // 2

    model_set = [[[] for ww in range(w_block)] for hh in range(h_block)]  # To save the models of each block.
    raw_training_scores_set = [[[] for ww in range(w_block)] for hh in range(h_block)]  # To save the training scores of each block.
    consistency_training_scores_set = [[[] for ww in range(w_block)] for hh in range(h_block)]

    teacher = resnet34(num_classes=100).cuda()
    teacher_weights_path = 'pre_training/pytorch-cifar100/checkpoint/resnet34/resnet34-200-regular.pth'
    teacher.load_state_dict(torch.load(teacher_weights_path))
    teacher.fc = nn.Identity()
    teacher.eval()

    fore_folder_name = '{}_{}_foreground_{}'.format(dataset_name, mode, foreground_extraction_mode)
    model_name = varname(student_resnet9)
    for h_idx in range(h_block):
        for w_idx in range(w_block):
            raw_losses = AverageMeter()
            consistency_losses = AverageMeter()

            # Set the dataset to load the saved training foreground data for the model training of the current block.
            cur_dataset = TrainForegroundDataset3(fore_dir=os.path.join(data_root_dir, fore_folder_name), h_idx=h_idx, w_idx=w_idx)
            cur_dataloader = DataLoader(dataset=cur_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

            cur_model = torch.nn.DataParallel(student_resnet9()).cuda()  # Initialize the model.
            optimizer = optim.Adam(cur_model.parameters(), weight_decay=0.0, lr=0.001)

            cur_model.train()
            for epoch in range(epochs):
                for idx, (raw, _) in enumerate(cur_dataloader):
                    raw = raw.cuda().type(torch.cuda.FloatTensor)

                    t_output = teacher(raw[:, cur_frame_idx * 3:(cur_frame_idx + 1) * 3, :, :])
                    if epoch < 30:
                        s_output = cur_model(raw[:, cur_frame_idx * 3:(cur_frame_idx + 1) * 3, :, :])
                        loss_raw = 0.0
                        for i in range(len(s_output)):
                            loss_temp = loss_func(s_output[i], t_output[i])
                            loss_raw += loss_temp
                        raw_losses.update(loss_raw.item(), raw.size(0))

                        loss = loss_raw

                        if idx % 5 == 0:
                            print('Block: ({}, {}), epoch {}, batch {} of {}, raw loss: {}'.format(
                                h_idx, w_idx, epoch, idx, cur_dataset.__len__() // batch_size, raw_losses.avg))
                    else:
                        all_s_output = []
                        for i in range(tot_frame_num):
                            s_output = cur_model(raw[:, i*3:(i+1)*3, :, :])
                            all_s_output.append(s_output)

                        loss_raw = 0.0
                        for i in range(len(all_s_output[cur_frame_idx])):
                            loss_temp = loss_func(all_s_output[cur_frame_idx][i], t_output[i])
                            loss_raw += loss_temp
                        raw_losses.update(loss_raw.item(), raw.size(0))

                        all_loss_consistency = 0.0
                        for i in range(len(all_s_output)):
                            if i != cur_frame_idx:
                                loss_consistency = 0.0
                                for j in range(len(all_s_output[cur_frame_idx])):
                                    loss_temp = loss_func(all_s_output[cur_frame_idx][j], all_s_output[i][j])
                                    loss_consistency += loss_temp
                                all_loss_consistency += loss_consistency
                        all_loss_consistency = all_loss_consistency / (tot_frame_num - 1)
                        consistency_losses.update(all_loss_consistency.item(), raw.size(0))

                        loss = loss_raw + all_loss_consistency * lambda_consistency

                        if idx % 5 == 0:
                            print('Block: ({}, {}), epoch {}, batch {} of {}, raw loss: {}, consistency loss: {}'.format(
                                h_idx, w_idx, epoch, idx, cur_dataset.__len__() // batch_size, raw_losses.avg, consistency_losses.avg))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            model_set[h_idx][w_idx].append(cur_model.module.state_dict())
            #  /*--  A forward pass to store the training scores of each STC (foreground) --*/
            forward_dataloader = DataLoader(dataset=cur_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
            cur_model.eval()
            for idx, (raw, _) in enumerate(forward_dataloader):
                raw = raw.cuda().type(torch.cuda.FloatTensor)

                t_output = teacher(raw[:, cur_frame_idx * 3:(cur_frame_idx + 1) * 3, :, :])
                all_s_output = []
                for i in range(tot_frame_num):
                    s_output = cur_model(raw[:, i * 3:(i + 1) * 3, :, :])
                    all_s_output.append(s_output)

                raw_scores = 0.0
                for i in range(len(all_s_output[cur_frame_idx])):
                    temp_scores = score_func(all_s_output[cur_frame_idx][i], t_output[i]).cpu().data.numpy()
                    temp_scores = np.sum(np.sum(np.sum(temp_scores, axis=3), axis=2), axis=1)  # MSE score for each STC.
                    raw_scores = raw_scores + temp_scores
                raw_training_scores_set[h_idx][w_idx].append(raw_scores)

                all_consistency_scores = 0.0
                for i in range(len(all_s_output)):
                    if i != cur_frame_idx:
                        consistency_scores = 0.0
                        for j in range(len(all_s_output[cur_frame_idx])):
                            temp_scores = score_func(all_s_output[cur_frame_idx][j], all_s_output[i][j]).cpu().data.numpy()
                            temp_scores = np.sum(np.sum(np.sum(temp_scores, axis=3), axis=2), axis=1)
                            consistency_scores += temp_scores
                        all_consistency_scores += consistency_scores
                all_consistency_scores = all_consistency_scores / (tot_frame_num - 1)
                consistency_training_scores_set[h_idx][w_idx].append(all_consistency_scores)

            raw_training_scores_set[h_idx][w_idx] = np.concatenate(raw_training_scores_set[h_idx][w_idx], axis=0)
            consistency_training_scores_set[h_idx][w_idx] = np.concatenate(consistency_training_scores_set[h_idx][w_idx], axis=0)

    torch.save(raw_training_scores_set, os.path.join(data_root_dir, '{}_raw_training_scores_{}_{}.npy'.format(dataset_name, model_name, method)))
    torch.save(consistency_training_scores_set, os.path.join(data_root_dir, '{}_consistency_training_scores_{}_{}.npy'.format(dataset_name, model_name, method)))
    print('training scores saved')
    torch.save(model_set, os.path.join(data_root_dir, '{}_model_{}_{}.npy'.format(dataset_name, model_name, method)))
    print('Training of {} for dataset: {} has completed!'.format(method, dataset_name))

else:
    raise NotImplementedError
