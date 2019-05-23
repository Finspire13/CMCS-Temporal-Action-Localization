import matlab.engine  # Must import matlab.engine first

import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from utils import smooth
from utils import eval_thumos_detect, detect_with_thresholding
from utils import get_dataset, normalize, interpolate
from utils import mask_to_detections, load_config_file
from utils import output_detections_thumos14, output_detections_anet

import pdb


def softmax(x, dim):
    x = F.softmax(torch.from_numpy(x), dim=dim)
    return x.numpy()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config-file', type=str)
    parser.add_argument('--train-subset-name', type=str)
    parser.add_argument('--test-subset-name', type=str)

    parser.add_argument('--include-train',
                        dest='include_train',
                        action='store_true')
    parser.add_argument('--no-include-train',
                        dest='include_train',
                        action='store_false')
    parser.set_defaults(include_train=True)

    args = parser.parse_args()

    print(args.config_file)
    print(args.train_subset_name)
    print(args.test_subset_name)
    print(args.include_train)

    all_params = load_config_file(args.config_file)
    locals().update(all_params)

    if args.include_train:
        train_dataset_dict = get_dataset(
            dataset_name=dataset_name,
            subset=args.train_subset_name,
            file_paths=file_paths,
            sample_rate=sample_rate,
            base_sample_rate=base_sample_rate,
            action_class_num=action_class_num,
            modality='both',
            feature_type=feature_type,
            feature_oversample=False,
            temporal_aug=False,
        )

    else:
        train_dataset_dict = None

    test_dataset_dict = get_dataset(
        dataset_name=dataset_name,
        subset=args.test_subset_name,
        file_paths=file_paths,
        sample_rate=sample_rate,
        base_sample_rate=base_sample_rate,
        action_class_num=action_class_num,
        modality='both',
        feature_type=feature_type,
        feature_oversample=False,
        temporal_aug=False,
    )

    dataset_dicts = {'train': train_dataset_dict, 'test': test_dataset_dict}

    def detect(
            cas_dir,
            subset,
            out_file_name,
            global_score_thrh,
            metric_type,
            thrh_type,
            thrh_value,
            interpolate_type,
            proc_type,
            proc_value,
            sample_offset,
            weight_inner,
            weight_outter,
            weight_global,
            att_filtering_value=None,
    ):

        assert (metric_type in ['score', 'multiply', 'att-filtering'])
        assert (thrh_type in ['mean', 'max'])
        assert (interpolate_type in ['quadratic', 'linear', 'nearest'])
        assert (proc_type in ['dilation', 'median'])

        out_detections = []

        dataset_dict = dataset_dicts[subset]

        for video_name in dataset_dict.keys():

            cas_file = video_name + '.npz'
            cas_data = np.load(os.path.join(cas_dir, cas_file))

            avg_score = cas_data['avg_score']
            att_weight = cas_data['weight']
            branch_scores = cas_data['branch_scores']
            global_score = cas_data['global_score']

            duration = dataset_dict[video_name]['duration']
            fps = dataset_dict[video_name]['frame_rate']
            frame_cnt = dataset_dict[video_name]['frame_cnt']

            global_score = softmax(global_score, dim=0)

            ################ Threshoding ################
            for class_id in range(action_class_num):

                if global_score[class_id] <= global_score_thrh:
                    continue

                if metric_type == 'score':

                    metric = softmax(avg_score, dim=1)[:, class_id:class_id + 1]
                    #metric = smooth(metric)
                    metric = normalize(metric)

                elif metric_type == 'multiply':

                    _score = softmax(avg_score, dim=1)[:, class_id:class_id + 1]
                    metric = att_weight * _score
                    #metric = smooth(metric)
                    metric = normalize(metric)

                elif metric_type == 'att-filtering':
                    assert (att_filtering_value is not None)

                    metric = softmax(avg_score, dim=1)[:, class_id:class_id + 1]
                    #metric = smooth(metric)
                    metric = normalize(metric)
                    metric[att_weight < att_filtering_value] = 0
                    metric = normalize(metric)

                #########################################

                metric = interpolate(metric[:, 0],
                                     feature_type,
                                     frame_cnt,
                                     sample_rate,
                                     snippet_size=base_snippet_size,
                                     kind=interpolate_type)

                metric = np.expand_dims(metric, axis=1)

                mask = detect_with_thresholding(metric, thrh_type, thrh_value,
                                                proc_type, proc_value)

                temp_out = mask_to_detections(mask, metric, weight_inner,
                                              weight_outter)

                #########################################

                for entry in temp_out:

                    entry[2] = class_id

                    entry[3] += global_score[class_id] * weight_global

                    entry[0] = (entry[0] + sample_offset) / fps
                    entry[1] = (entry[1] + sample_offset) / fps

                    entry[0] = max(0, entry[0])
                    entry[1] = max(0, entry[1])
                    entry[0] = min(duration, entry[0])
                    entry[1] = min(duration, entry[1])

                #########################################

                for entry_id in range(len(temp_out)):
                    temp_out[entry_id] = [video_name] + temp_out[entry_id]

                out_detections += temp_out

        if dataset_name == 'thumos14':
            output_detections_thumos14(out_detections, out_file_name)
        elif dataset_name in ['ActivityNet12', 'ActivityNet13']:
            output_detections_anet(out_detections, out_file_name, dataset_name,
                                   feature_type)

        return out_detections

    if dataset_name == 'thumos14':
        summary_file = './outputs/summary-{}.npz'.format(experiment_naming)

        all_train_map = np.zeros((train_run_num, len(check_points), 4, 9, 1))
        all_train_aps = np.zeros(
            (train_run_num, len(check_points), 4, 9, action_class_num))
        all_test_map = np.zeros((train_run_num, len(check_points), 4, 9, 1))
        all_test_aps = np.zeros(
            (train_run_num, len(check_points), 4, 9, action_class_num))

    for run_idx in range(train_run_num):

        for cp_idx, check_point in enumerate(check_points):

            for mod_idx, modality in enumerate(
                ['both', 'rgb', 'flow', 'late-fusion']):

                cas_dir = os.path.join(
                    'cas-features',
                    '{}-run-{}-{}-{}'.format(experiment_naming, run_idx,
                                             check_point, modality))

                pred_dir = os.path.join('outputs', 'predictions')

                if not os.path.exists(pred_dir):
                    os.makedirs(pred_dir)

                if args.include_train:
                    train_pred_file = os.path.join(
                        pred_dir,
                        '{}-run-{}-{}-{}-train'.format(experiment_naming,
                                                       run_idx, check_point,
                                                       modality))

                    train_outs = detect(cas_dir, 'train', train_pred_file,
                                        **detect_params)

                test_pred_file = os.path.join(
                    pred_dir,
                    '{}-run-{}-{}-{}-test'.format(experiment_naming, run_idx,
                                                  check_point, modality))

                test_outs = detect(cas_dir, 'test', test_pred_file,
                                   **detect_params)

                if dataset_name == 'thumos14':

                    for IoU_idx, IoU in enumerate(
                        [.1, .2, .3, .4, .5, .6, .7, .8, .9]):

                        if args.include_train:
                            if len(train_outs) != 0:
                                temp_aps, temp_map = eval_thumos_detect(
                                    train_pred_file, file_paths[
                                        args.train_subset_name]['anno_dir'],
                                    args.train_subset_name, IoU)

                                all_train_aps[run_idx, cp_idx, mod_idx,
                                              IoU_idx, :] = temp_aps
                                all_train_map[run_idx, cp_idx, mod_idx, IoU_idx,
                                              0] = temp_map
                            else:
                                print('Empty Detections')
                                all_train_aps[run_idx, cp_idx, mod_idx,
                                              IoU_idx, :] = 0
                                all_train_map[run_idx, cp_idx, mod_idx, IoU_idx,
                                              0] = 0

                        if len(test_outs) != 0:
                            temp_aps, temp_map = eval_thumos_detect(
                                test_pred_file,
                                file_paths[args.test_subset_name]['anno_dir'],
                                args.test_subset_name, IoU)

                            all_test_aps[run_idx, cp_idx, mod_idx,
                                         IoU_idx, :] = temp_aps
                            all_test_map[run_idx, cp_idx, mod_idx, IoU_idx,
                                         0] = temp_map
                        else:
                            print('Empty Detections')
                            all_test_aps[run_idx, cp_idx, mod_idx,
                                         IoU_idx, :] = 0
                            all_test_map[run_idx, cp_idx, mod_idx, IoU_idx,
                                         0] = 0

                        print('{}{}{}{}'.format(run_idx, cp_idx, mod_idx,
                                                IoU_idx))

    if dataset_name == 'thumos14':

        np.savez(summary_file,
                 all_train_aps=all_train_aps,
                 all_train_map=all_train_map,
                 all_test_aps=all_test_aps,
                 all_test_map=all_test_map)
