import matlab.engine  # Must import matlab.engine first

import random
import numpy as np
from torch.utils.data import Dataset

from utils import get_single_label_dict
import pdb


def _random_select(rgb=-1, flow=-1):
    ''' Randomly select one augmented feature sequence. '''

    if type(rgb) != int and type(flow) != int:

        assert (rgb.shape[0] == flow.shape[0])
        random_idx = random.randint(0, rgb.shape[0] - 1)
        rgb = np.array(rgb[random_idx, :, :])
        flow = np.array(flow[random_idx, :, :])

    elif type(rgb) != int:
        random_idx = random.randint(0, rgb.shape[0] - 1)
        rgb = np.array(rgb[random_idx, :, :])

    elif type(flow) != int:
        random_idx = random.randint(0, flow.shape[0] - 1)
        flow = np.array(flow[random_idx, :, :])
    else:
        pass

    return rgb, flow


def _check_length(rgb, flow, max_len):

    if type(rgb) != int and type(flow) != int:

        assert (rgb.shape[1] == flow.shape[1])
        if rgb.shape[1] > max_len:
            print('Crop Both!')
            start = random.randint(0, rgb.shape[1] - max_len)
            rgb = np.array(rgb[:, start:start + max_len, :])
            flow = np.array(flow[:, start:start + max_len, :])

    elif type(rgb) != int:

        if rgb.shape[1] > max_len:
            print('Crop RGB!')
            start = random.randint(0, rgb.shape[1] - max_len)
            rgb = np.array(rgb[:, start:start + max_len, :])

    elif type(flow) != int:

        if flow.shape[1] > max_len:
            print('Crop FLOW!')
            start = random.randint(0, flow.shape[1] - max_len)
            flow = np.array(flow[:, start:start + max_len, :])
    else:
        pass

    return rgb, flow


class SingleVideoDataset(Dataset):

    def __init__(self,
                 dataset_dict,
                 single_label=False,
                 random_select=False,
                 max_len=None):

        self.dataset_dict = dataset_dict
        self.single_label = single_label
        self.random_select = random_select
        self.max_len = max_len

        if self.single_label:
            self.dataset_dict = get_single_label_dict(self.dataset_dict)

        self.video_list = list(self.dataset_dict.keys())
        self.video_list.sort()

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video = self.video_list[idx]

        rgb, flow = (self.dataset_dict[video]['rgb_feature'],
                     self.dataset_dict[video]['flow_feature'])

        if self.max_len:
            rgb, flow = _check_length(rgb, flow, self.max_len)

        if self.random_select:
            rgb, flow = _random_select(rgb, flow)

        return_dict = {
            'video_name': video,
            'rgb': rgb,
            'flow': flow,
            'frame_rate':
            self.dataset_dict[video]['frame_rate'],  # frame_rate == fps
            'frame_cnt': self.dataset_dict[video]['frame_cnt'],
            'anno': self.dataset_dict[video]['annotations']
        }

        if self.single_label:
            return_dict['label'] = self.dataset_dict[video]['label_single']
            return_dict['weight'] = self.dataset_dict[video]['weight']

        return return_dict
