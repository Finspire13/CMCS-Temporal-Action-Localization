import matlab.engine  # Must import matlab.engine first

import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from model import BackboneNet
from dataset import SingleVideoDataset
from utils import get_dataset, load_config_file

import pdb

device = torch.device('cuda')

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

    def get_features(loader, model_rgb, model_flow, model_both, modality,
                     save_dir):

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        assert (modality in ['both', 'rgb', 'flow', 'late-fusion'])

        model_both.eval()
        model_rgb.eval()
        model_flow.eval()

        for _, data in enumerate(loader):

            video_name = data['video_name'][0]

            print('Forwarding: {}'.format(video_name))

            assert (data['rgb'].shape[0] == 1)
            assert (data['flow'].shape[0] == 1)

            rgb = data['rgb'].to(device).squeeze(0)  # 1 at dim0
            flow = data['flow'].to(device).squeeze(0)
            rgb = rgb.transpose(2, 1)
            flow = flow.transpose(2, 1)
            cat = torch.cat([rgb, flow], dim=1)

            with torch.no_grad():

                if modality == 'both':
                    avg_score, weight, global_score, branch_scores, _ = model_both.forward(
                        cat)  # Add softmax
                elif modality == 'rgb':
                    avg_score, weight, global_score, branch_scores, _ = model_rgb.forward(
                        rgb)
                elif modality == 'flow':
                    avg_score, weight, global_score, branch_scores, _ = model_flow.forward(
                        flow)
                else:
                    avg_score1, weight1, global_score1, branch_scores1, _ = model_rgb.forward(
                        rgb)
                    avg_score2, weight2, global_score2, branch_scores2, _ = model_flow.forward(
                        flow)
                    avg_score = (avg_score1 + avg_score2) / 2

                    if (weight1 is None) or (weight2 is None):
                        weight = None
                    else:
                        weight = (weight1 + weight2) / 2

                    global_score = (global_score1 + global_score2) / 2

                    branch_scores = []

                    for branch in range(model_params['cls_branch_num']):

                        branch_scores.append(
                            (branch_scores1[branch] + branch_scores2[branch]) /
                            2)

            branch_scores = torch.stack(branch_scores).cpu().numpy()

            np.savez(os.path.join(save_dir, video_name + '.npz'),
                     avg_score=avg_score.mean(0).cpu().numpy(),
                     weight=weight.mean(0).cpu().numpy()
                     if weight is not None else None,
                     global_score=global_score.mean(0).cpu().numpy(),
                     branch_scores=branch_scores)

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
            feature_oversample=feature_oversample,
            temporal_aug=False,
        )

        train_detect_dataset = SingleVideoDataset(
            train_dataset_dict, single_label=False,
            random_select=False)  # SIngle label false!!!

        train_detect_loader = torch.utils.data.DataLoader(train_detect_dataset,
                                                          batch_size=1,
                                                          pin_memory=True,
                                                          shuffle=False)

    else:
        train_detect_loader = None

    test_dataset_dict = get_dataset(
        dataset_name=dataset_name,
        subset=args.test_subset_name,
        file_paths=file_paths,
        sample_rate=sample_rate,
        base_sample_rate=base_sample_rate,
        action_class_num=action_class_num,
        modality='both',
        feature_type=feature_type,
        feature_oversample=feature_oversample,
        temporal_aug=False,
    )

    test_detect_dataset = SingleVideoDataset(test_dataset_dict,
                                             single_label=False,
                                             random_select=False)

    test_detect_loader = torch.utils.data.DataLoader(test_detect_dataset,
                                                     batch_size=1,
                                                     pin_memory=True,
                                                     shuffle=False)

    for run_idx in range(train_run_num):

        naming = '{}-run-{}'.format(experiment_naming, run_idx)

        for cp_idx, check_point in enumerate(check_points):

            model_both = BackboneNet(in_features=feature_dim * 2,
                                     **model_params).to(device)
            model_rgb = BackboneNet(in_features=feature_dim,
                                    **model_params).to(device)
            model_flow = BackboneNet(in_features=feature_dim,
                                     **model_params).to(device)

            model_both.load_state_dict(
                torch.load(
                    os.path.join('models', naming,
                                 'model-both-{}'.format(check_point))))

            model_rgb.load_state_dict(
                torch.load(
                    os.path.join('models', naming,
                                 'model-rgb-{}'.format(check_point))))

            model_flow.load_state_dict(
                torch.load(
                    os.path.join('models', naming,
                                 'model-flow-{}'.format(check_point))))

            for mod_idx, modality in enumerate(
                ['both', 'rgb', 'flow', 'late-fusion']):
                # Both: Early fusion

                save_dir = os.path.join(
                    'cas-features',
                    '{}-run-{}-{}-{}'.format(experiment_naming, run_idx,
                                             check_point, modality))

                if args.include_train:
                    get_features(train_detect_loader, model_rgb, model_flow,
                                 model_both, modality, save_dir)

                get_features(test_detect_loader, model_rgb, model_flow,
                             model_both, modality, save_dir)
