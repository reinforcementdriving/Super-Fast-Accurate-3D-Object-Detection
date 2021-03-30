"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Meng Liu
# DoC: 2020.09.27
# email:
-----------------------------------------------------------------------------------
# Description: Eval script for kitti evaluation tool
# Refer from: https://github.com/maudzung/SFA3D/blob/master/sfa/test.py
"""

import argparse
import sys
import os
import time
import warnings

warnings.filterwarnings("ignore", category = UserWarning)

from   easydict import EasyDict as edict
import cv2
import torch
import numpy as np

sys.path.append('../')

from   sfa.data_process.kitti_dataloader import create_val_dataloader
from   sfa.models.model_utils            import create_model
from   sfa.utils.misc                    import make_folder, time_synchronized
from   sfa.utils.evaluation_utils        import decode, post_processing, draw_predictions, convert_det_to_real_values
from   sfa.utils.torch_utils             import _sigmoid
import sfa.config.kitti_config           as cnf
from   sfa.data_process.transformation   import lidar_to_camera_box
from   sfa.utils.visualization_utils     import merge_rgb_to_bev, show_rgb_image_with_boxes, show_rgb_image_with_boxes_gt
from   sfa.data_process.kitti_data_utils import Calibration
from   sfa.utils.voxel_utils             import box3d_to_label


def parse_test_configs():
    parser = argparse.ArgumentParser(description = 'Testing config for the Implementation')
    parser.add_argument('--saved_fn',   type=str, default='fpn_resnet_18', metavar='FN',   help='The name using for saving logs, models,...')
    parser.add_argument('-a', '--arch', type=str, default='fpn_resnet_18', metavar='ARCH', help='The name of the model architecture')
    parser.add_argument('--pretrained_path', type=str,
                        default='/data/SFA3DOD/checkpoints/fpn_resnet_18/Model_fpn_resnet_18_epoch_300.pth',
                        metavar='PATH', help='the path of the pretrained checkpoint')
    parser.add_argument('--K',       default=50, type=int, help='the number of top K')
    parser.add_argument('--gpu_idx', default=0,  type=int, help='GPU index to use.')
    parser.add_argument('--num_samples', type=int, default=None, help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of threads for loading data')
    parser.add_argument('--batch_size',  type=int, default=1, help='mini-batch size (default: 4)')
    parser.add_argument('--peak_thresh', type=float, default=0.2)
    parser.add_argument('--save_test_output', action='store_true', help='If true, the output image of the testing phase will be saved')
    parser.add_argument('--output_format',   type=str, default='image', metavar='PATH', help='the type of the test output (support image or video)')
    parser.add_argument('--output_video_fn', type=str, default='out_fpn_resnet_18', metavar='PATH', help='the video filename if the output format is video')
    parser.add_argument('--output-width',    type=int, default=608, help='the width of showing output, the height maybe vary')

    configs             = edict(vars(parser.parse_args()))
    configs.pin_memory  = True
    configs.distributed = False  # For testing on 1 GPU only

    configs.input_size  = (608, 608)
    configs.hm_size     = (152, 152)
    configs.down_ratio  = 4
    configs.max_objects = 50

    configs.imagenet_pretrained = False
    configs.head_conv           = 64
    configs.num_classes         = 3
    configs.num_center_offset   = 2
    configs.num_z               = 1
    configs.num_dim             = 3
    configs.num_direction       = 2  # sin, cos

    configs.heads = {
        'hm_cen':     configs.num_classes,
        'cen_offset': configs.num_center_offset,
        'direction':  configs.num_direction,
        'z_coor':     configs.num_z,
        'dim':        configs.num_dim
    }
    configs.num_input_features = 4


    configs.root_dir    = '/data/kitti'
    configs.dataset_dir = os.path.join(configs.root_dir) #dataset path

    return configs



if __name__ == '__main__':
    configs = parse_test_configs()

    model = create_model(configs)
    print('\n\n' + '-*=' * 30 + '\n\n')
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)

    # load the well-trained weight
    model.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
    print('Loaded weights from {}\n'.format(configs.pretrained_path))

    configs.device = torch.device('cuda')
    model = model.to(device = configs.device)

    #
    pred_dir = "/data/SFA3DOD/predict_dir/"

    model.eval()


    # load the val split of dataset
    val_dataloader, val_dataset = create_val_dataloader(configs)
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_dataloader):

            t1 = time_synchronized()
            metadatas, imgs, targets, img_rgbs = batch_data  #imgs: bev_map, targets: regression targets
            batch_size = imgs.size(0)

            _, labels, _, _  = val_dataset.draw_img_with_label(batch_idx)

            #inference
            imgs    = imgs.to(configs.device, non_blocking = True).float()
            outputs = model(imgs)

            outputs['hm_cen']     = _sigmoid(outputs['hm_cen'])
            outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
            detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], outputs['z_coor'], outputs['dim'], K=configs.K)
            detections = detections.cpu().numpy().astype(np.float32)
            # post processing
            detections = post_processing(detections, configs.num_classes, configs.down_ratio, configs.peak_thresh)
            #detections format [n, score, cx, cy, cz, dim-z, dim-y, dim-x, yaw]

            detections = detections[0]  # only first batch

            # visualization for analysing the predictions
            img_path = metadatas['img_path'][0]
            img_rgb  = img_rgbs[0].numpy()
            img_rgb  = cv2.resize(img_rgb, (img_rgb.shape[1], img_rgb.shape[0]))
            img_bgr  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            calib    = Calibration(img_path.replace(".png", ".txt").replace("image_2", "calib"))


            # predictions to real values
            kitti_dets        = convert_det_to_real_values(detections)
            print("length of detections:", len(kitti_dets))


            # write the predictions to files
            with open(pred_dir + "%06d.txt"%(batch_idx + 6000), "w") as fp:
                if len(kitti_dets) == 0:
                    fp.close()
                    continue

                # lidar coordinate to camera coordinate
                kitti_dets[:, 1:] = lidar_to_camera_box(kitti_dets[:, 1:], calib.V2C, calib.R0, calib.P2)


                # class 0 0 0 x-min y_min x_max y_max dim-z dim-w dim-l x y z yaw conf
                result       = kitti_dets

                # to kitti GT format
                kitti_lables = box3d_to_label([result[:, 1:8]], [result[:, 0]], [result[:, -1]], coordinate='camera')[0]

                # write to files
                for line in kitti_lables:
                    fp.write(line)


                ## for visualization
                ####-----------------------------------------------------------------
                #### Coordinate transformation
                ###labels[:, 1:] = lidar_to_camera_box(labels[:, 1:], calib.V2C, calib.R0, calib.P2)

                #### draw boxes on pictures
                ###img_bgr = show_rgb_image_with_boxes(img_bgr, kitti_dets, calib) #kitti_dets: predictions
                ###img_bgr = show_rgb_image_with_boxes_gt(img_bgr, labels,     calib) #labels: GT labels

                #### show the predictions
                ###cv2.imshow('test-img', img_bgr)
                ###print('\n[INFO] Press n to see the next sample >>> Press Esc to quit...\n')
                ###if cv2.waitKey(0) & 0xFF == 27:
                ###    break

            t2 = time_synchronized()

            print('Done testing the {}th sample <--> {}th , time: {:.1f}ms, speed {:.2f}FPS'.format(batch_idx, batch_idx + 6000, (t2 - t1) * 1000,  1 / (t2 - t1)))

