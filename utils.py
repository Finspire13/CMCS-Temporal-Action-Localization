import os
import matlab
import json
import subprocess
import numpy as np
import pandas as pd
import matlab.engine
from PIL import Image
from scipy.io import loadmat
from collections import defaultdict
from scipy.interpolate import interp1d
from skimage import measure
from skimage.morphology import dilation
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.signal import medfilt

import pdb

################ Config ##########################


def load_config_file(config_file):
    '''
    -- Doc for parameters in the json file --

    feature_oversample:   Whether data augmentation is used (five crop and filp).
    sample_rate:          How many frames between adjacent feature snippet.

    with_bg:              Whether hard negative mining is used.
    diversity_reg:        Whether diversity loss and norm regularization are used.
    diversity_weight:     The weight of both diversity loss and norm regularization.

    train_run_num:        How many times the experiment is repeated.
    training_max_len:     Crop the feature sequence when training if it exceeds this length.

    learning_rate_decay:  Whether to reduce the learning rate at half of training steps.
    max_step_num:         Number of training steps.
    check_points:         Check points to test and save models.
    log_freq:             How many training steps the log is added to tensorboard.

    model_params:
    cls_branch_num:       Branch number in the multibranch network.
    base_layer_params:    Filter number and size in each layer of the embedding module.
    cls_layer_params:     Filter number and size in each layer of the classification module.
    att_layer_params:     Filter number and size in each layer of the attention module.

    detect_params:        Parameters for action localization on the CAS. 
                          See detect.py for details.

    base_sample_rate:     'sample_rate' when feature extraction.
    base_snippet_size:    The size of each feature snippet.

    bg_mask_dir:          The folder of masks of static clips.

    < Others are easy to guess >

    '''

    all_params = json.load(open(config_file))

    dataset_name = all_params['dataset_name']
    feature_type = all_params['feature_type']

    all_params['file_paths'] = all_params['file_paths'][dataset_name]
    all_params['action_class_num'] = all_params['action_class_num'][
        dataset_name]
    all_params['base_sample_rate'] = all_params['base_sample_rate'][
        dataset_name][feature_type]
    all_params['base_snippet_size'] = all_params['base_snippet_size'][
        feature_type]

    assert (all_params['sample_rate'] % all_params['base_sample_rate'] == 0)

    all_params['model_class_num'] = all_params['action_class_num']
    if all_params['with_bg']:
        all_params['model_class_num'] += 1

    all_params['model_params']['class_num'] = all_params['model_class_num']

    # Convert second to frames
    all_params['detect_params']['proc_value'] = int(
        all_params['detect_params']['proc_value'] * all_params['sample_rate'])

    print(all_params)
    return all_params


################ Matlab #####################

matlab_eng = matlab.engine.start_matlab()

################ Class Name Mapping #####################

thumos14_old_cls_names = {
    7: 'BaseballPitch',
    9: 'BasketballDunk',
    12: 'Billiards',
    21: 'CleanAndJerk',
    22: 'CliffDiving',
    23: 'CricketBowling',
    24: 'CricketShot',
    26: 'Diving',
    31: 'FrisbeeCatch',
    33: 'GolfSwing',
    36: 'HammerThrow',
    40: 'HighJump',
    45: 'JavelinThrow',
    51: 'LongJump',
    68: 'PoleVault',
    79: 'Shotput',
    85: 'SoccerPenalty',
    92: 'TennisSwing',
    93: 'ThrowDiscus',
    97: 'VolleyballSpiking',
}

thumos14_old_cls_indices = {v: k for k, v in thumos14_old_cls_names.items()}

thumos14_new_cls_names = {
    0: 'BaseballPitch',
    1: 'BasketballDunk',
    2: 'Billiards',
    3: 'CleanAndJerk',
    4: 'CliffDiving',
    5: 'CricketBowling',
    6: 'CricketShot',
    7: 'Diving',
    8: 'FrisbeeCatch',
    9: 'GolfSwing',
    10: 'HammerThrow',
    11: 'HighJump',
    12: 'JavelinThrow',
    13: 'LongJump',
    14: 'PoleVault',
    15: 'Shotput',
    16: 'SoccerPenalty',
    17: 'TennisSwing',
    18: 'ThrowDiscus',
    19: 'VolleyballSpiking',
    20: 'Background',
}

thumos14_new_cls_indices = {v: k for k, v in thumos14_new_cls_names.items()}

old_cls_names = {
    'thumos14': thumos14_old_cls_names,
    'ActivityNet12': np.load('misc/old_cls_names_anet12.npy').item(),
    'ActivityNet13': np.load('misc/old_cls_names_anet13.npy').item(),
}

old_cls_indices = {
    'thumos14': thumos14_old_cls_indices,
    'ActivityNet12': np.load('misc/old_cls_indices_anet12.npy').item(),
    'ActivityNet13': np.load('misc/old_cls_indices_anet13.npy').item(),
}

new_cls_names = {
    'thumos14': thumos14_new_cls_names,
    'ActivityNet12': np.load('misc/new_cls_names_anet12.npy').item(),
    'ActivityNet13': np.load('misc/new_cls_names_anet13.npy').item(),
}

new_cls_indices = {
    'thumos14': thumos14_new_cls_indices,
    'ActivityNet12': np.load('misc/new_cls_indices_anet12.npy').item(),
    'ActivityNet13': np.load('misc/new_cls_indices_anet13.npy').item(),
}

################ Load dataset #####################


def load_meta(meta_file):
    '''Load video metas from the mat file (Only for thumos14).'''
    meta_data = loadmat(meta_file)
    meta_mat_name = [i for i in meta_data.keys() if 'videos' in i][0]
    meta_data = meta_data[meta_mat_name][0]
    return meta_data


def load_annotation_file(anno_file):
    '''Load action instaces from a single file (Only for thumos14).'''
    anno_data = pd.read_csv(anno_file, header=None, delimiter=' ')
    anno_data = np.array(anno_data)
    return anno_data


def __get_thumos14_meta(meta_file, anno_dir):

    meta_data = load_meta(meta_file)

    dataset_dict = {}

    anno_files = [i for i in os.listdir(anno_dir) if 'Ambiguous' not in i]
    anno_files.remove('detclasslist.txt')
    anno_files.sort()

    for anno_file in anno_files:

        action_label = anno_file.split('_')[0]
        action_label = new_cls_indices['thumos14'][action_label]

        anno_file = os.path.join(anno_dir, anno_file)
        anno_data = load_annotation_file(anno_file)

        for entry in anno_data:
            video_name = entry[0]
            start = entry[2]
            end = entry[3]

            ### Initializatiton ###
            if video_name not in dataset_dict.keys():

                video_meta = [i for i in meta_data if i[0][0] == video_name][0]

                duration = video_meta[meta_data.dtype.names.index(
                    'video_duration_seconds')][0, 0]
                frame_rate = video_meta[meta_data.dtype.names.index(
                    'frame_rate_FPS')][0, 0]

                dataset_dict[video_name] = {
                    'duration': duration,
                    'frame_rate': frame_rate,
                    'labels': [],
                    'annotations': {},
                }

            if action_label not in dataset_dict[video_name]['labels']:
                dataset_dict[video_name]['labels'].append(action_label)
                dataset_dict[video_name]['annotations'][action_label] = []
            ###

            dataset_dict[video_name]['annotations'][action_label].append(
                [start, end])

    return dataset_dict


def __get_anet_meta(anno_json_file, dataset_name, subset):

    data = json.load(open(anno_json_file, 'r'))
    taxonomy_data = data['taxonomy']
    database_data = data['database']
    missing_videos = np.load('misc/anet_missing_videos.npy')

    if subset == 'train':
        subset_data = {
            k: v for k, v in database_data.items() if v['subset'] == 'training'
        }
    elif subset == 'val':
        subset_data = {
            k: v
            for k, v in database_data.items()
            if v['subset'] == 'validation'
        }
    elif subset == 'train_and_val':
        subset_data = {
            k: v
            for k, v in database_data.items()
            if v['subset'] in ['training', 'validation']
        }
    elif subset == 'test':
        subset_data = {
            k: v for k, v in database_data.items() if v['subset'] == 'testing'
        }

    dataset_dict = {}

    for video_name, v in subset_data.items():

        if video_name in missing_videos:
            print('Missing video: {}'.format(video_name))
            continue

        dataset_dict[video_name] = {
            'duration': v['duration'],
            'frame_rate': 25,  # ActivityNet should be formatted to 25 fps first
            'labels': [],
            'annotations': {},
        }

        for entry in v['annotations']:

            action_label = entry['label']
            action_label = new_cls_indices[dataset_name][action_label]

            if action_label not in dataset_dict[video_name]['labels']:
                dataset_dict[video_name]['labels'].append(action_label)
                dataset_dict[video_name]['annotations'][action_label] = []

            dataset_dict[video_name]['annotations'][action_label].append(
                entry['segment'])

    return dataset_dict


def __load_features(
        dataset_dict,  # dataset_dict will be modified
        dataset_name,
        feature_type,
        sample_rate,
        base_sample_rate,
        temporal_aug,
        rgb_feature_dir,
        flow_feature_dir):

    assert (feature_type in ['i3d', 'untri'])

    assert (sample_rate % base_sample_rate == 0)
    f_sample_rate = int(sample_rate / base_sample_rate)

    # sample_rate of feature sequences, not original video

    ###############
    def __process_feature_file(filename):
        ''' Load features from a single file. '''

        feature_data = np.load(filename)

        frame_cnt = feature_data['frame_cnt'].item()

        if feature_type == 'untri':
            feature = np.swapaxes(feature_data['feature'][:, :, :, 0, 0], 0, 1)
        elif feature_type == 'i3d':
            feature = feature_data['feature']

        # Feature: (B, T, F)
        # Example: (1, 249, 1024) or (10, 249, 1024) (Oversample)

        if temporal_aug:  # Data augmentation with temporal offsets
            feature = [
                feature[:, offset::f_sample_rate, :]
                for offset in range(f_sample_rate)
            ]
            # Cut to same length, OK when training
            min_len = int(min([i.shape[1] for i in feature]))
            feature = [i[:, :min_len, :] for i in feature]

            assert (len(set([i.shape[1] for i in feature])) == 1)
            feature = np.concatenate(feature, axis=0)

        else:
            feature = feature[:, ::f_sample_rate, :]

        return feature, frame_cnt

        # Feature: (B x f_sample_rate, T, F)

    ###############

    # Load all features
    for k in dataset_dict.keys():

        print('Loading: {}'.format(k))

        # Init empty
        dataset_dict[k]['frame_cnt'] = -1
        dataset_dict[k]['rgb_feature'] = -1
        dataset_dict[k]['flow_feature'] = -1

        if rgb_feature_dir:

            if dataset_name == 'thumos14':
                rgb_feature_file = os.path.join(rgb_feature_dir, k + '-rgb.npz')
            else:
                rgb_feature_file = os.path.join(rgb_feature_dir,
                                                'v_' + k + '-rgb.npz')

            rgb_feature, rgb_frame_cnt = __process_feature_file(
                rgb_feature_file)

            dataset_dict[k]['frame_cnt'] = rgb_frame_cnt
            dataset_dict[k]['rgb_feature'] = rgb_feature

        if flow_feature_dir:

            if dataset_name == 'thumos14':
                flow_feature_file = os.path.join(flow_feature_dir,
                                                 k + '-flow.npz')
            else:
                flow_feature_file = os.path.join(flow_feature_dir,
                                                 'v_' + k + '-flow.npz')

            flow_feature, flow_frame_cnt = __process_feature_file(
                flow_feature_file)

            dataset_dict[k]['frame_cnt'] = flow_frame_cnt
            dataset_dict[k]['flow_feature'] = flow_feature

        if rgb_feature_dir and flow_feature_dir:
            assert (rgb_frame_cnt == flow_frame_cnt)
            assert (dataset_dict[k]['rgb_feature'].shape[1] == dataset_dict[k]
                    ['flow_feature'].shape[1])
            assert (dataset_dict[k]['rgb_feature'].mean() !=
                    dataset_dict[k]['flow_feature'].mean())

    return dataset_dict


def __load_background(
        dataset_dict,  # dataset_dict will be modified
        dataset_name,
        bg_mask_dir,
        sample_rate,
        action_class_num):

    bg_mask_files = os.listdir(bg_mask_dir)
    bg_mask_files.sort()

    for bg_mask_file in bg_mask_files:

        if dataset_name == 'thumos14':
            video_name = bg_mask_file[:-4]
        else:
            video_name = bg_mask_file[2:-4]

        new_key = video_name + '_bg'

        if video_name not in dataset_dict.keys():
            continue

        bg_mask = np.load(os.path.join(bg_mask_dir, bg_mask_file))
        bg_mask = bg_mask['mask']

        assert (dataset_dict[video_name]['frame_cnt'] == bg_mask.shape[0])

        # Remove if static clips are too long or too short
        bg_ratio = bg_mask.sum() / bg_mask.shape[0]
        if bg_ratio < 0.05 or bg_ratio > 0.30:
            print('Bad bg {}: {}'.format(bg_ratio, video_name))
            continue

        bg_mask = bg_mask[::sample_rate]  # sample rate of original videos

        dataset_dict[new_key] = {}

        if type(dataset_dict[video_name]['rgb_feature']) != int:

            rgb = np.array(dataset_dict[video_name]['rgb_feature'])
            bg_mask = bg_mask[:rgb.shape[1]]  # same length
            bg_rgb = rgb[:, bg_mask.astype(bool), :]
            dataset_dict[new_key]['rgb_feature'] = bg_rgb

            frame_cnt = bg_rgb.shape[
                1]  # Psuedo frame count of a virtual bg video

        if type(dataset_dict[video_name]['flow_feature']) != int:

            flow = np.array(dataset_dict[video_name]['flow_feature'])
            bg_mask = bg_mask[:flow.shape[1]]
            bg_flow = flow[:, bg_mask.astype(bool), :]
            dataset_dict[new_key]['flow_feature'] = bg_flow

            frame_cnt = bg_flow.shape[
                1]  # Psuedo frame count of a virtual bg video

        dataset_dict[new_key]['annotations'] = {action_class_num: []}
        dataset_dict[new_key]['labels'] = [action_class_num]  # background class

        fps = dataset_dict[video_name]['frame_rate']
        dataset_dict[new_key]['frame_rate'] = fps
        dataset_dict[new_key]['frame_cnt'] = frame_cnt  # Psuedo
        dataset_dict[new_key]['duration'] = frame_cnt / fps  # Psuedo

    return dataset_dict


def get_dataset(dataset_name,
                subset,
                file_paths,
                sample_rate,
                base_sample_rate,
                action_class_num,
                modality='both',
                feature_type=None,
                feature_oversample=True,
                temporal_aug=False,
                load_background=False):

    assert (dataset_name in ['thumos14', 'ActivityNet12', 'ActivityNet13'])

    if dataset_name == 'thumos14':
        if load_background:
            assert (subset in ['val'])
        else:
            assert (subset in ['val', 'test'])
    else:
        assert (subset in ['train', 'val', 'train_and_val', 'test'])

    assert (modality in ['both', 'rgb', 'flow', None])
    assert (feature_type in ['i3d', 'untri'])

    if dataset_name == 'thumos14':
        dataset_dict = __get_thumos14_meta(
            meta_file=file_paths[subset]['meta_file'],
            anno_dir=file_paths[subset]['anno_dir'])
    else:
        dataset_dict = __get_anet_meta(file_paths[subset]['anno_json_file'],
                                       dataset_name, subset)

    _temp_f_type = (feature_type +
                    '-oversample' if feature_oversample else feature_type +
                    '-resize')

    if modality == 'both':
        rgb_dir = file_paths[subset]['feature_dir'][_temp_f_type]['rgb']
        flow_dir = file_paths[subset]['feature_dir'][_temp_f_type]['flow']
    elif modality == 'rgb':
        rgb_dir = file_paths[subset]['feature_dir'][_temp_f_type]['rgb']
        flow_dir = None
    elif modality == 'flow':
        rgb_dir = None
        flow_dir = file_paths[subset]['feature_dir'][_temp_f_type]['flow']
    else:
        rgb_dir = None
        flow_dir = None

    dataset_dict = __load_features(dataset_dict, dataset_name, feature_type,
                                   sample_rate, base_sample_rate, temporal_aug,
                                   rgb_dir, flow_dir)

    if load_background:
        dataset_dict = __load_background(dataset_dict, dataset_name,
                                         file_paths[subset]['bg_mask_dir'],
                                         sample_rate, action_class_num)

    return dataset_dict


def get_single_label_dict(dataset_dict):
    '''
    If a video has multiple action classes, we treat it as multiple videos with
    single class. And the weight of each of them is reduced.
    '''
    new_dict = {}  # Create a new dict

    for k, v in dataset_dict.items():
        for label in v['labels']:

            new_key = '{}-{}'.format(k, label)

            new_dict[new_key] = dict(v)

            new_dict[new_key]['label_single'] = label
            new_dict[new_key]['annotations'] = v['annotations'][label]
            new_dict[new_key]['weight'] = (1 / len(v['labels']))

            new_dict[new_key]['old_key'] = k

    return new_dict  # This dict should be read only


def get_videos_each_class(dataset_dict):

    videos_each_class = defaultdict(list)

    for k, v in dataset_dict.items():

        if 'label_single' in v.keys():
            label = v['label_single']
            videos_each_class[label].append(k)

        else:
            for label in v['labels']:
                videos_each_class[label].append(k)

    return videos_each_class


################ Post-Processing #####################


def normalize(x):
    x -= x.min()
    x /= x.max()
    return x


def smooth(x):  # Two Dim nparray, On 1st dim
    temp = np.array(x)

    temp[1:, :] = temp[1:, :] + x[:-1, :]
    temp[:-1, :] = temp[:-1, :] + x[1:, :]

    temp[1:-1, :] /= 3
    temp[0, :] /= 2
    temp[-1, :] /= 2

    return temp


def __get_frame_ticks(feature_type, frame_cnt, sample_rate, snippet_size=None):
    '''Get the frames of each feature snippet location.'''

    assert (feature_type in ['i3d', 'untri'])

    if feature_type == 'i3d':
        assert (snippet_size is not None)

        clipped_length = frame_cnt - snippet_size
        clipped_length = (clipped_length // sample_rate) * sample_rate
        # the start of the last chunk

        frame_ticks = np.arange(0, clipped_length + 1, sample_rate)
        # From 0, the start of chunks, clipped_length included

    elif feature_type == 'untri':
        frame_ticks = np.arange(0, frame_cnt, sample_rate)
        # From 0, image files are from 1

    return frame_ticks


def interpolate(x,
                feature_type,
                frame_cnt,
                sample_rate,
                snippet_size=None,
                kind='linear'):
    '''Upsample the sequence the original video fps.'''

    frame_ticks = __get_frame_ticks(feature_type, frame_cnt, sample_rate,
                                    snippet_size)

    full_ticks = np.arange(frame_ticks[0], frame_ticks[-1] + 1)
    # frame_ticks[-1] included

    interp_func = interp1d(frame_ticks, x, kind=kind)
    out = interp_func(full_ticks)

    return out


################ THUMOS Evaluation #####################


def eval_thumos_detect(detfilename, gtpath, subset, threshold):
    assert (subset in ['test', 'val'])

    matlab_eng.addpath('THUMOS14_evalkit_20150930')
    aps = matlab_eng.TH14evalDet(detfilename, gtpath, subset, threshold)

    aps = np.array(aps)
    mean_ap = aps.mean()

    return aps, mean_ap


def eval_thumos_recog(scores, labels, action_class_num):
    matlab_eng.addpath('THUMOS14_evalkit_20150930')

    aps = []

    for i in range(action_class_num):  # Without bg
        aps.append(
            matlab_eng.TH14evalRecog_clspr(
                matlab.double(scores[:, i:i + 1].tolist()),
                matlab.double(labels[:, i:i + 1].tolist())))

    aps = np.array(aps)
    mean_ap = aps.mean()

    return aps, mean_ap


################ Action Localization #####################


def detections_to_mask(length, detections):

    mask = np.zeros((length, 1))
    for entry in detections:
        mask[entry[0]:entry[1]] = 1

    return mask


def mask_to_detections(mask, metric, weight_inner, weight_outter):

    out_detections = []
    detection_map = measure.label(mask, background=0)
    detection_num = detection_map.max()

    for detection_id in range(1, detection_num + 1):

        start = np.where(detection_map == detection_id)[0].min()
        end = np.where(detection_map == detection_id)[0].max() + 1

        length = end - start

        inner_area = metric[detection_map == detection_id]

        left_start = min(int(start - length * 0.25),
                         start - 1)  # Context size 0.25
        right_end = max(int(end + length * 0.25), end + 1)

        outter_area_left = metric[left_start:start, :]
        outter_area_right = metric[end:right_end, :]

        outter_area = np.concatenate((outter_area_left, outter_area_right),
                                     axis=0)

        if outter_area.shape[0] == 0:
            detection_score = inner_area.mean() * weight_inner
        else:
            detection_score = (inner_area.mean() * weight_inner +
                               outter_area.mean() * weight_outter)

        out_detections.append([start, end, None, detection_score])

    return out_detections


def detect_with_thresholding(metric,
                             thrh_type,
                             thrh_value,
                             proc_type,
                             proc_value,
                             debug_file=None):

    assert (thrh_type in ['max', 'mean'])
    assert (proc_type in ['dilation', 'median'])

    out_detections = []

    if thrh_type == 'max':
        mask = metric > thrh_value

    elif thrh_type == 'mean':
        mask = metric > (thrh_value * metric.mean())

    if proc_type == 'dilation':
        mask = dilation(mask, np.array([[1] for _ in range(proc_value)]))
    elif proc_type == 'median':
        mask = medfilt(mask[:, 0], kernel_size=proc_value)
        # kernel_size should be odd
        mask = np.expand_dims(mask, axis=1)

    return mask


################ Output Detection To Files ################


def output_detections_thumos14(out_detections, out_file_name):

    for entry in out_detections:
        class_id = entry[3]
        class_name = new_cls_names['thumos14'][class_id]
        old_class_id = int(old_cls_indices['thumos14'][class_name])
        entry[3] = old_class_id

    out_file = open(out_file_name, 'w')

    for entry in out_detections:
        out_file.write('{} {:.2f} {:.2f} {} {:.4f}\n'.format(
            entry[0], entry[1], entry[2], int(entry[3]), entry[4]))

    out_file.close()


def output_detections_anet(out_detections, out_file_name, dataset_name,
                           feature_type):

    assert (dataset_name in ['ActivityNet12', 'ActivityNet13'])
    assert (feature_type in ['untri', 'i3d'])

    for entry in out_detections:
        class_id = entry[3]
        class_name = new_cls_names[dataset_name][class_id]
        entry[3] = class_name

    output_dict = {}

    if dataset_name == 'ActivityNet12':
        output_dict['version'] = 'VERSION 1.2'
    else:
        output_dict['version'] = 'VERSION 1.3'

    if feature_type == 'untri':
        output_dict['external_data'] = {
            'used': False,
            'details': 'Untri feature'
        }
    else:
        output_dict['external_data'] = {'used': True, 'details': 'I3D feature'}

    output_dict['results'] = {}

    for entry in out_detections:

        if entry[0] not in output_dict['results']:
            output_dict['results'][entry[0]] = []

        output_dict['results'][entry[0]].append({
            'label': entry[3],
            'score': entry[4],
            'segment': [entry[1], entry[2]],
        })

    with open(out_file_name, 'w') as f:
        f.write(json.dumps(output_dict))


################ Visualization #####################


def get_snippet_gt(annos, fps, sample_rate, snippet_num):

    gt = np.zeros((snippet_num,))

    for i in annos:
        start = int(i[0] * fps // sample_rate)
        end = int(i[1] * fps // sample_rate)

        gt[start:start + 1] = 0.5
        gt[end:end + 1] = 0.5
        gt[start + 1:end] = 1

    return gt


def visualize_scores_barcodes(score_titles,
                              scores,
                              ylim=None,
                              out_file=None,
                              show=False):

    lens = [i.shape[0] for i in scores]
    assert (len(set(lens)) == 1)
    frame_cnt = lens[0]  # Not all frame are visualized, clipped at end

    subplot_sum = len(score_titles)

    fig = plt.figure(figsize=(20, 10))

    height_ratios = [1 for _ in range(subplot_sum)]

    gs = gridspec.GridSpec(subplot_sum, 1, height_ratios=height_ratios)

    for j in range(len(score_titles)):

        fig.add_subplot(gs[j])

        plt.xticks([])
        plt.yticks([])

        plt.title(score_titles[j], position=(-0.1, 0))

        axes = plt.gca()

        if j == 0:
            barprops = dict(aspect='auto',
                            cmap=plt.cm.PiYG,
                            interpolation='nearest',
                            vmin=-1,
                            vmax=1)
        elif j == 1:
            barprops = dict(aspect='auto',
                            cmap=plt.cm.seismic,
                            interpolation='nearest',
                            vmin=-1,
                            vmax=1)
        elif j == 2 or j == 3:

            if ylim:
                barprops = dict(aspect='auto',
                                cmap=plt.cm.Purples,
                                interpolation='nearest',
                                vmin=ylim[0],
                                vmax=ylim[1])
            else:
                barprops = dict(
                    aspect='auto',
                    cmap=plt.cm.Purples,  #BrBG
                    interpolation='nearest')

        else:
            if ylim:
                barprops = dict(aspect='auto',
                                cmap=plt.cm.Blues,
                                interpolation='nearest',
                                vmin=ylim[0],
                                vmax=ylim[1])
            else:
                barprops = dict(aspect='auto',
                                cmap=plt.cm.Blues,
                                interpolation='nearest')

        axes.imshow(scores[j].reshape((1, -1)), **barprops)

    if out_file:
        plt.savefig(out_file)

    if show:
        plt.show()

    plt.close()


def visualize_video_with_scores_barcodes(images_dir,
                                         images_prefix,
                                         score_titles,
                                         scores,
                                         out_file,
                                         fps,
                                         ylim=None):  # Fps: original video fps

    images_paths = [
        os.path.join(images_dir, i)
        for i in os.listdir(images_dir)
        if i.startswith(images_prefix)
    ]

    images_paths.sort()

    lens = [i.shape[0] for i in scores]
    assert (len(set(lens)) == 1)
    frame_cnt = lens[0]  # Not all frame are visualized, clipped at end

    subplot_sum = len(score_titles) + 1

    temp_dir = './temp_plots'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    for i in range(frame_cnt):

        fig = plt.figure(figsize=(15, 10))

        height_ratios = [1 for _ in range(subplot_sum)]
        height_ratios[0] = 12

        gs = gridspec.GridSpec(subplot_sum, 1, height_ratios=height_ratios)

        fig.add_subplot(gs[0])

        plt.axis('off')
        plt.title('Video')
        plt.imshow(Image.open(images_paths[i]).convert('RGB'))

        for j in range(len(score_titles)):

            fig.add_subplot(gs[j + 1])

            plt.xticks([])
            plt.yticks([])

            plt.title(score_titles[j], position=(-0.1, 0))

            axes = plt.gca()

            if j == 0:
                barprops = dict(aspect='auto',
                                cmap=plt.cm.PiYG,
                                interpolation='nearest',
                                vmin=-1,
                                vmax=1)
            elif j == 1:
                barprops = dict(aspect='auto',
                                cmap=plt.cm.seismic,
                                interpolation='nearest',
                                vmin=-1,
                                vmax=1)
            elif j == 2 or j == 3:

                if ylim:
                    barprops = dict(aspect='auto',
                                    cmap=plt.cm.Purples,
                                    interpolation='nearest',
                                    vmin=ylim[0],
                                    vmax=ylim[1])
                else:
                    barprops = dict(aspect='auto',
                                    cmap=plt.cm.Purples,
                                    interpolation='nearest')

            else:
                if ylim:
                    barprops = dict(aspect='auto',
                                    cmap=plt.cm.Blues,
                                    interpolation='nearest',
                                    vmin=ylim[0],
                                    vmax=ylim[1])
                else:
                    barprops = dict(aspect='auto',
                                    cmap=plt.cm.Blues,
                                    interpolation='nearest')

            axes.imshow(scores[j].reshape((1, -1)), **barprops)

            axes.axvline(x=i, color='darkorange')

        plt.savefig(os.path.join(temp_dir, '{:0>6}.png'.format(i)))
        plt.close()

    subprocess.call([
        'ffmpeg', '-framerate',
        str(fps), '-i',
        os.path.join(temp_dir, '%06d.png'), '-pix_fmt', 'yuv420p', out_file
    ])

    os.system('rm -r {}'.format(temp_dir))
