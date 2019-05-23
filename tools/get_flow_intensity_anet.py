import os
import io
import torch
import zipfile
import argparse
import numpy as np
from PIL import Image
from scipy.signal import medfilt

import pdb

def remove_short_clips(xx, min_length):

    x = np.array(xx)

    start = -1
    end = -1
    flag = False

    for i in range(len(x)):

        if not flag and x[i] == 1:
            start = i
            flag = True
        
        if flag and x[i] == 0:
            end = i
            flag = False

            if end - start < min_length:
                x[start:end] = 0

            start = -1
            end = -1

    if flag:
        end = i
        if end - start < min_length:
            x[start:end] = 0

    return x


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--in_dir', type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--percentile_thrh', type=int)  # 25
    args = parser.parse_args()

    print(args.in_dir)
    print(args.out_dir)
    print(args.percentile_thrh)

    fps = 25

    video_names = [i for i in os.listdir(args.in_dir) if i.startswith('v_')]

    for video_name in video_names:

        print(video_name)

        save_file = os.path.join(args.out_dir, video_name + '.npz')

        if save_file in os.listdir(args.out_dir):
            continue

        video_dir = os.path.join(args.in_dir, video_name)

        flow_x_zipdata = zipfile.ZipFile(os.path.join(video_dir, 'flow_x.zip'), 'r')
        flow_y_zipdata = zipfile.ZipFile(os.path.join(video_dir, 'flow_y.zip'), 'r')

        flow_x_files = [i for i in flow_x_zipdata.namelist() if i.startswith('x_')]
        flow_y_files = [i for i in flow_y_zipdata.namelist() if i.startswith('y_')]

        flow_x_files.sort()
        flow_y_files.sort()

        assert(len(flow_x_files) == len(flow_y_files))

        intensity_xy_mean = np.zeros((len(flow_x_files),))

        for frame_id in range(len(flow_x_files)):

            flow_x = Image.open(io.BytesIO(flow_x_zipdata.read(flow_x_files[frame_id])))
            flow_y = Image.open(io.BytesIO(flow_y_zipdata.read(flow_y_files[frame_id])))

            flow_x = np.array(flow_x).astype(float)
            flow_y = np.array(flow_y).astype(float)

            flow_x = (flow_x * 2 / 255) - 1  # -1, 1
            flow_y = (flow_y * 2 / 255) - 1

            flow_xy = np.sqrt(flow_x * flow_x + flow_y * flow_y)

            intensity_xy_mean[frame_id] = flow_xy.mean()



        intensity_xy_mean = np.log(intensity_xy_mean+0.0000001)

        threshold = np.percentile(intensity_xy_mean, args.percentile_thrh)
        thresholded = intensity_xy_mean < threshold

        filtered = medfilt(thresholded, kernel_size=(2*int(fps)-1))  # maybe shorter

        removed = remove_short_clips(filtered, 2*int(fps))  # maybe shorter

        np.savez(save_file,
                intensity=intensity_xy_mean,
                mask=removed)
