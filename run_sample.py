import argparse
from misc import pyutils
import torch
import numpy as np
import random

import os




def fix_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    # torch.set_deterministic(True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=5, type=int)

    # Environment
    parser.add_argument("--num_workers", default=os.cpu_count() // 2, type=int)
    parser.add_argument("--voc12_root", default='/home/cly/datasets/VOC2012', type=str,
                        help="Path to VOC 2012 Devkit, must contain ./JPEGImages as subdirectory.")

    # Dataset
    parser.add_argument("--train_aug_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--train_list", default="voc12/train.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--test_list", default="voc12/test.txt", type=str)
    parser.add_argument("--infer_list", default="voc12/train.txt", type=str,
                        help="voc12/train_aug.txt to train a fully supervised model, "
                             "voc12/train.txt or voc12/val.txt to quickly check the quality of the labels.")
    parser.add_argument("--chainer_eval_set", default="train", type=str)

    # Class Activation Map
    parser.add_argument("--cam_network", default="net.resnet50_cam", type=str)
    parser.add_argument("--cam_crop_size", default=512, type=int)
    parser.add_argument("--cam_batch_size", default=16, type=int)
    parser.add_argument("--cam_num_epoches", default=5, type=int)
    parser.add_argument("--cam_learning_rate", default=0.1, type=float)
    parser.add_argument("--cam_weight_decay", default=1e-4, type=float)
    parser.add_argument("--cam_eval_thres", default=0.11, type=float)
    parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5, 2.0),
                        help="Multi-scale inferences")

    parser.add_argument("--conf_fg_thres", default=0.30, type=float)
    parser.add_argument("--conf_bg_thres", default=0.07, type=float)

    # parameters for 'make_boundary_label'
    parser.add_argument("--window_size", default=13, type=int)
    parser.add_argument("--theta_scale", default=0.30, type=float)
    parser.add_argument("--theta_diff", default=0.10, type=float)
    parser.add_argument("--boundary_labels",
                        default={'BG': 0, 'FG': 50, 'BOUNDARY_FG_FG': 100, 'BOUNDARY_FG_BG': 150, 'IGNORE': 200},
                        type=dict)

    parser.add_argument("--bes_network", default="net.resnet50_bes", type=str)
    parser.add_argument("--bes_crop_size", default=512, type=int)
    parser.add_argument("--bes_batch_size", default=16, type=int)
    parser.add_argument("--bes_num_epoches", default=3, type=int)
    parser.add_argument("--bes_learning_rate", default=0.1, type=float)
    parser.add_argument("--bes_weight_decay", default=1e-4, type=float)

    # Random Walk Params
    parser.add_argument("--beta", default=5)
    parser.add_argument("--exp_times", default=8,
                        help="Hyper-parameter that controls the number of random walk iterations,"
                             "The random walk is performed 2^{exp_times}.")
    parser.add_argument("--sem_seg_bg_thres", default=0.25)

    # Output Path
    parser.add_argument("--log_name", default="record", type=str)
    parser.add_argument("--cam_weights_name", default="sess/res50_cam_attention.pth", type=str)
    parser.add_argument("--bes_weights_name", default="sess/res50_bes.pth", type=str)
    parser.add_argument("--cam_out_dir", default="result/cam", type=str)
    parser.add_argument("--cam_vis_dir", default="result/cam_vis", type=str)
    parser.add_argument("--boundary_label_dir", default="result/boundary_label", type=str)
    parser.add_argument("--sem_seg_out_dir", default="result/sem_seg", type=str)

    # Step
    parser.add_argument("--train_cam_pass", default=True)
    parser.add_argument("--make_cam_pass", default=True)
    parser.add_argument("--eval_cam_pass", default=True)
    parser.add_argument("--make_boundary_label_pass", default=True)
    parser.add_argument("--train_bes_pass", default=True)
    parser.add_argument("--make_sem_seg_pass", default=True)
    parser.add_argument("--eval_sem_seg_pass", default=True)

    args = parser.parse_args()

    fix_seed(args.seed)
    os.makedirs("sess", exist_ok=True)
    os.makedirs(args.cam_out_dir, exist_ok=True)
    os.makedirs(args.cam_vis_dir, exist_ok=True)
    os.makedirs(args.boundary_label_dir, exist_ok=True)
    os.makedirs(args.sem_seg_out_dir, exist_ok=True)

    pyutils.Logger(args.log_name + '.log')
    print(vars(args))

    if args.train_cam_pass is True:
        import step.train_cam

        timer = pyutils.Timer('step.train_cam:')
        step.train_cam.run(args)

    if args.make_cam_pass is True:
        import step.make_cam

        timer = pyutils.Timer('step.make_cam:')
        step.make_cam.run(args)

    if args.eval_cam_pass is True:
        import step.eval_cam

        timer = pyutils.Timer('step.eval_cam:')
        step.eval_cam.run(args)

    if args.make_boundary_label_pass is True:
        import step.make_boundary_label

        timer = pyutils.Timer('step.cam_to_boundary_label:')
        step.make_boundary_label.run(args)

    if args.train_bes_pass is True:
        import step.train_bes

        timer = pyutils.Timer('step.train_bes:')
        step.train_bes.run(args)

    if args.make_sem_seg_pass is True:
        import step.make_sem_seg

        timer = pyutils.Timer('step.make_sem_seg:')
        step.make_sem_seg.run(args)

    if args.eval_sem_seg_pass is True:
        import step.eval_sem_seg

        timer = pyutils.Timer('step.eval_sem_seg:')
        step.eval_sem_seg.run(args)
