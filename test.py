import numpy as np
import os
from torch.utils.data import DataLoader
from vad_datasets import unified_dataset_interface, FrameForegroundDataset2, TestForegroundDataset3, img_tensor2numpy, img_batch_tensor2numpy, frame_info
from fore_det.inference import init_detector
from fore_det.obj_det_with_motion import imshow_bboxes, getObBboxes, delCoverBboxes, getOfBboxes
import torch
from model.resnet import resnet34, student_resnet9
import torch.nn as nn
from utils import save_roc_pr_curve_data, calc_block_idx, moving_average, varname
from configparser import ConfigParser
from helper.visualization_helper import visualize_pair, visualize_batch, visualize_pair_map, visualize_score, visualize_img

#  /*-------------------------------------------------Overall parameter setting-----------------------------------------------------*/
cp = ConfigParser()
cp.read("config.cfg")  # To get parameter settings from config.cfg.

# The same parameter settings as train.py, see detailed comments in train.py.
dataset_name = cp.get('shared_parameters', 'dataset_name')
raw_dataset_dir = cp.get('shared_parameters', 'raw_dataset_dir')
foreground_extraction_mode = cp.get('shared_parameters', 'foreground_extraction_mode')
data_root_dir = cp.get('shared_parameters', 'data_root_dir')
mode = cp.get('test_parameters', 'mode')
method = cp.get('shared_parameters', 'method')
try:
    patch_size = cp.getint(dataset_name, 'patch_size')
    h_block = cp.getint(dataset_name, 'h_block')
    w_block = cp.getint(dataset_name, 'w_block')
    test_block_mode = cp.getint(dataset_name, 'test_block_mode')
    bbox_saved = cp.getboolean(dataset_name, mode + '_bbox_saved')
    foreground_saved = cp.getboolean(dataset_name, mode + '_foreground_saved')
    motionThr = cp.getfloat(dataset_name, 'motionThr')
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

#  /*--------------------------------------------------Foreground extraction--------------------------------------------------------*/
if not foreground_saved:
    border_mode = cp.get(method, 'border_mode')  # See details in vad_datasets.py.
    context_frame_num = cp.getint(method, 'context_frame_num')  # See details in vad_datasets.py.
    context_of_num = cp.getint(method, 'context_of_num')

    file_format1 = frame_info[dataset_name][2]
    file_format2 = '.npy'
    # Set dataset for foreground extraction.
    dataset = unified_dataset_interface(dataset_name=dataset_name, dir=os.path.join('raw_datasets', dataset_name),
                                        context_frame_num=context_frame_num, mode=mode,
                                        border_mode=border_mode, all_bboxes=all_bboxes, patch_size=patch_size,
                                        file_format=file_format1)
    dataset2 = unified_dataset_interface(dataset_name=dataset_name, dir=os.path.join('optical_flow', dataset_name),
                                         context_frame_num=context_of_num, mode=mode,
                                         border_mode=border_mode, all_bboxes=all_bboxes, patch_size=patch_size,
                                         file_format=file_format2)

    # To store bboxes corresponding to foreground in each frame.
    foreground_bbox_set = [[[[] for ww in range(w_block)] for hh in range(h_block)] for ii in range(dataset.__len__())]

    # Create folders to store foreground of each frame.
    fore_folder_name = '{}_{}_foreground_{}'.format(dataset_name, mode, foreground_extraction_mode)
    raw_fore_dir = os.path.join(data_root_dir, fore_folder_name, 'raw')
    os.makedirs(raw_fore_dir, exist_ok=True)
    of_fore_dir = os.path.join(data_root_dir, fore_folder_name, 'of')
    os.makedirs(of_fore_dir, exist_ok=True)

    h_step, w_step = frame_info[dataset_name][0] / h_block, frame_info[dataset_name][1] / w_block
    for idx in range(dataset.__len__()):
        print('Extracting foreground in {}-th batch, {} in total'.format(idx + 1, dataset.__len__() // 1))

        batch, _ = dataset.__getitem__(idx)
        batch2, _ = dataset2.__getitem__(idx)
        cur_bboxes = all_bboxes[idx]

        frame_foreground = [[[] for ww in range(w_block)] for hh in range(h_block)]
        frame_foreground2 = [[[] for ww in range(w_block)] for hh in range(h_block)]
        if len(cur_bboxes) > 0:
            batch = img_batch_tensor2numpy(batch)
            batch2 = img_batch_tensor2numpy(batch2)

            if len(batch2.shape) == 4:
                mag = np.sum(np.sum(np.sum(batch2 ** 2, axis=3), axis=2), axis=1)
            else:
                mag = np.mean(np.sum(np.sum(np.sum(batch2 ** 2, axis=4), axis=3), axis=2), axis=1)

            for idx_bbox in range(cur_bboxes.shape[0]):
                if mag[idx_bbox] > motionThr:
                    # Store the foreground into the frame block where it belongs to.
                    all_blocks = calc_block_idx(cur_bboxes[idx_bbox, 0], cur_bboxes[idx_bbox, 2], cur_bboxes[idx_bbox, 1], cur_bboxes[idx_bbox, 3], h_step, w_step, mode=test_block_mode)
                    for (h_block_idx, w_block_idx) in all_blocks:
                        frame_foreground[h_block_idx][w_block_idx].append(batch[idx_bbox])
                        frame_foreground2[h_block_idx][w_block_idx].append(batch2[idx_bbox])
                        foreground_bbox_set[idx][h_block_idx][w_block_idx].append(cur_bboxes[idx_bbox])

        frame_foreground = [[np.array(frame_foreground[hh][ww]) for ww in range(w_block)] for hh in range(h_block)]
        frame_foreground2 = [[np.array(frame_foreground2[hh][ww]) for ww in range(w_block)] for hh in range(h_block)]
        np.save(os.path.join(data_root_dir, fore_folder_name, 'raw', '{}.npy'.format(idx)), frame_foreground)
        np.save(os.path.join(data_root_dir, fore_folder_name, 'of', '{}.npy'.format(idx)), frame_foreground2)

    foreground_bbox_set = [[[np.array(foreground_bbox_set[ii][hh][ww]) for ww in range(w_block)] for hh in range(h_block)] for ii in range(dataset.__len__())]
    np.save(os.path.join(data_root_dir, fore_folder_name, 'foreground_bbox.npy'), foreground_bbox_set)
    print('foreground for testing data saved!')
else:
    fore_folder_name = '{}_{}_foreground_{}'.format(dataset_name, mode, foreground_extraction_mode)
    foreground_bbox_set = np.load(os.path.join(data_root_dir, fore_folder_name, 'foreground_bbox.npy'), allow_pickle=True)
    print('foreground bboxes for testing data loaded! Below will load saved testing foreground with dataset interface.')

#  /*-------------------------------------------------Abnormal event detection-----------------------------------------------------*/
scores_saved = cp.getboolean(dataset_name, 'scores_saved')
results_dir = 'results'
if scores_saved is False:
    if method == 'CAFE':  # End-to-end VAD.
        h, w, _, sn = frame_info[dataset_name]
        border_mode = cp.get(method, 'border_mode')
        w_raw = cp.getfloat(method, 'w_raw')
        w_consistency = cp.getfloat(method, 'w_consistency')
        if border_mode == 'predict':
            tot_frame_num = cp.getint(method, 'context_frame_num') + 1
            cur_frame_idx = tot_frame_num - 1
        else:
            tot_frame_num = 2 * cp.getint(method, 'context_frame_num') + 1
            cur_frame_idx = tot_frame_num // 2

        score_func = nn.MSELoss(reduce=False)
        big_number = 20  # To initialize the score mask of each frame.

        # Create folders to store all frame score masks or frame scores.
        frame_result_dir = os.path.join(results_dir, dataset_name)
        os.makedirs(frame_result_dir, exist_ok=True)
        pixel_result_dir = os.path.join(results_dir, dataset_name, 'score_mask')
        os.makedirs(pixel_result_dir, exist_ok=True)

        # Load saved models for testing.
        teacher = resnet34(num_classes=100).cuda()
        teacher_weights_path = 'pre_training/pytorch-cifar100/checkpoint/resnet34/resnet34-200-regular.pth'
        teacher.load_state_dict(torch.load(teacher_weights_path))
        teacher.fc = nn.Identity()
        teacher.eval()

        model_name = varname(student_resnet9)
        model_weights = torch.load(os.path.join(data_root_dir, '{}_model_{}_{}.npy'.format(dataset_name, model_name, method)))  # Load the saved model weights.
        model_set = [[[] for ww in range(len(model_weights[hh]))] for hh in range(len(model_weights))]
        for hh in range(len(model_weights)):
            for ww in range(len(model_weights[hh])):
                if len(model_weights[hh][ww]) > 0:
                    cur_model = student_resnet9().cuda()
                    cur_model.load_state_dict(model_weights[hh][ww][0])
                    model_set[hh][ww].append(cur_model.eval())

        # Load training scores of each STC (i.e., foreground).
        raw_training_scores_set = torch.load(os.path.join(data_root_dir, '{}_raw_training_scores_{}_{}.npy'.format(dataset_name, model_name, method)))
        consistency_training_scores_set = torch.load(os.path.join(data_root_dir, '{}_consistency_training_scores_{}_{}.npy'.format(dataset_name, model_name, method)))
        # Get the mean and std of training scores.
        raw_stats_set = [[(np.mean(raw_training_scores_set[hh][ww]), np.std(raw_training_scores_set[hh][ww])) for ww in range(w_block)] for hh in range(h_block)]
        consistency_stats_set = [[(np.mean(consistency_training_scores_set[hh][ww]), np.std(consistency_training_scores_set[hh][ww])) for ww in range(w_block)] for hh in range(h_block)]

        fore_folder_name = '{}_{}_foreground_{}'.format(dataset_name, mode, foreground_extraction_mode)
        # Set dataset to load the saved testing foreground data of each frame.
        frame_fore_dataset = FrameForegroundDataset2(frame_fore_dir=os.path.join(data_root_dir, fore_folder_name))
        # To get scores of each frame.
        frame_scores = []
        for frame_idx in range(frame_fore_dataset.__len__()):
            print('Calculating scores for {}-th frame in total {} frames'.format(frame_idx, frame_fore_dataset.__len__()))
            frame_raw_fore, frame_of_fore = frame_fore_dataset.__getitem__(frame_idx)  # Get the foreground data of the current frame.
            cur_bboxes = foreground_bbox_set[frame_idx]
            # normal: no objects in test set
            cur_pixel_results = -1 * np.ones(shape=(h, w)) * big_number  # The initial score mask of the current frame.

            for h_idx in range(h_block):
                for w_idx in range(w_block):
                    if len(frame_raw_fore[h_idx][w_idx]) > 0:
                        if len(model_set[h_idx][w_idx]) > 0:
                            cur_model = model_set[h_idx][w_idx][0]
                            # Set the dataset to load the foreground data of the current block in this frame.
                            cur_dataset = TestForegroundDataset3(raw_fore=frame_raw_fore[h_idx][w_idx], of_fore=frame_of_fore[h_idx][w_idx])
                            cur_dataloader = DataLoader(dataset=cur_dataset, batch_size=frame_raw_fore[h_idx][w_idx].shape[0], shuffle=False, num_workers=0)

                            for idx, (raw, _) in enumerate(cur_dataloader):
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

                                # Normalize scores with training scores.
                                raw_scores = (raw_scores - raw_stats_set[h_idx][w_idx][0]) / raw_stats_set[h_idx][w_idx][1]
                                all_consistency_scores = (all_consistency_scores - consistency_stats_set[h_idx][w_idx][0]) / consistency_stats_set[h_idx][w_idx][1]

                                scores = raw_scores * w_raw + all_consistency_scores * w_consistency

                        else:
                            # anomaly: no objects in training set but objects occur in this block
                            scores = np.ones(frame_raw_fore[h_idx][w_idx].shape[0], ) * big_number

                        for m in range(scores.shape[0]):
                            cur_score_mask = -1 * np.ones(shape=(h, w)) * big_number
                            cur_score = scores[m]
                            bbox = cur_bboxes[h_idx][w_idx][m]
                            x_min, x_max = np.int(np.ceil(bbox[0])), np.int(np.ceil(bbox[2]))
                            y_min, y_max = np.int(np.ceil(bbox[1])), np.int(np.ceil(bbox[3]))
                            cur_score_mask[y_min:y_max, x_min:x_max] = cur_score
                            cur_pixel_results = np.max(np.concatenate([cur_pixel_results[:, :, np.newaxis], cur_score_mask[:, :, np.newaxis]], axis=2), axis=2)
            frame_scores.append(cur_pixel_results.max())
        torch.save(frame_scores, os.path.join(frame_result_dir, 'frame_scores.npy'))

    else:
        raise NotImplementedError

#  /*-------------------------------------------------------Evaluation-----------------------------------------------------------*/
criterion = 'frame'
batch_size = 1
# Set dataset for evaluation.
dataset = unified_dataset_interface(dataset_name=dataset_name, dir=os.path.join(raw_dataset_dir, dataset_name), context_frame_num=0, mode=mode, border_mode='hard')
dataset_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=8)

print('Evaluating {} by {}-criterion:'.format(dataset_name, criterion))
if criterion == 'frame':
    all_frame_scores = torch.load(os.path.join(results_dir, dataset_name, 'frame_scores.npy'))
    all_targets = list()
    for idx, (_, target) in enumerate(dataset_loader):
        print('Processing {}-th frame'.format(idx))
        all_targets.append(target[0].numpy().max())  # Get the label of the current frame.
    all_frame_scores = np.array(all_frame_scores)
    all_targets = np.array(all_targets)
    all_targets = all_targets > 0
    results_path = os.path.join(results_dir, dataset_name, '{}_{}_frame_results.npz'.format(foreground_extraction_mode, method))
    print('Results written to {}:'.format(results_path))

    # Use a sliding window to smooth frame anomaly scores, which can filter score noises and boost performance; when window_size=1, it is not really used.
    all_frame_scores = moving_average(all_frame_scores, window_size=10, decay=1)
    auc = save_roc_pr_curve_data(all_frame_scores, all_targets, results_path)
else:
    raise NotImplementedError
