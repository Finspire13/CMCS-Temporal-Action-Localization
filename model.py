import matlab.engine  # Must import matlab.engine first

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict

import pdb


class BackboneNet(nn.Module):

    def __init__(self, in_features, class_num, dropout_rate, cls_branch_num,
                 base_layer_params, cls_layer_params, att_layer_params):
        '''
        Layer_params: 
        [[kerel_num_1, kernel_size_1],[kerel_num_2, kernel_size_2], ...]
        '''
        super(BackboneNet, self).__init__()

        assert (dropout_rate > 0)

        self.cls_branch_num = cls_branch_num
        self.att_layer_params = att_layer_params

        self.dropout = nn.Dropout2d(
            p=dropout_rate)  # Drop same channels for untri features

        base_module_list = self._get_module_list(in_features, base_layer_params,
                                                 'base')

        self.base = nn.Sequential(OrderedDict(base_module_list))

        cls_module_lists = []
        for branch_idx in range(cls_branch_num):
            cls_module_lists.append(
                self._get_module_list(base_layer_params[-1][0],
                                      cls_layer_params,
                                      'cls_b{}'.format(branch_idx)))

        self.cls_bottoms = nn.ModuleList(
            [nn.Sequential(OrderedDict(i)) for i in cls_module_lists])

        self.cls_heads = nn.ModuleList([
            nn.Linear(cls_layer_params[-1][0], class_num)
            for i in range(cls_branch_num)
        ])

        if self.att_layer_params:
            att_module_list = self._get_module_list(base_layer_params[-1][0],
                                                    att_layer_params, 'att')

            self.att_bottom = nn.Sequential(OrderedDict(att_module_list))

            self.att_head = nn.Linear(att_layer_params[-1][0], 1)
        else:
            self.gap = nn.AdaptiveMaxPool1d(1)

    def _get_module_list(self, in_features, layer_params, naming):

        module_list = []

        for layer_idx in range(len(layer_params)):

            if layer_idx == 0:
                in_chl = in_features
            else:
                in_chl = layer_params[layer_idx - 1][0]

            out_chl = layer_params[layer_idx][0]
            kernel_size = layer_params[layer_idx][1]
            conv_pad = kernel_size // 2

            module_list.append(('{}_conv_{}'.format(naming, layer_idx),
                                nn.Conv1d(in_chl,
                                          out_chl,
                                          kernel_size,
                                          padding=conv_pad)))

            module_list.append(('{}_relu_{}'.format(naming,
                                                    layer_idx), nn.ReLU()))

        return module_list

    def forward(self, x):  # In: B x F x T

        x_drop = self.dropout(x.unsqueeze(3)).squeeze(3)
        base_feature = self.base(x_drop)

        cls_features = []
        branch_scores = []

        for branch_idx in range(self.cls_branch_num):

            cls_feature = self.cls_bottoms[branch_idx](base_feature)
            cls_score = self.cls_heads[branch_idx](cls_feature.transpose(1, 2))

            cls_features.append(cls_feature)
            branch_scores.append(cls_score)

        avg_score = torch.stack(branch_scores).mean(0)

        if self.att_layer_params:
            att_feature = self.att_bottom(base_feature)
            att_weight = self.att_head(att_feature.transpose(1, 2))
            att_weight = F.softmax(att_weight, dim=1)
            global_score = (avg_score * att_weight).sum(1)

        else:
            att_feature = None
            att_weight = None
            global_score = self.gap(avg_score.transpose(1, 2)).squeeze(2)

        # for debug and future work
        feature_dict = {
            'base_feature': base_feature,
            'cls_features': cls_features,
            #'att_feature': att_feature,
        }

        return avg_score, att_weight, global_score, branch_scores, feature_dict
