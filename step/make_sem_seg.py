import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn
import numpy as np
import importlib
import os
import imageio
from PIL import Image
from misc import imutils
import voc12.dataloader
from misc import torchutils, indexing

cudnn.enabled = True
from net.resnet50_bes import Boundary

palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0, 0,
           192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64,
           0, 0, 192, 0, 128, 192, 0, 0, 64, 128]


def _work(process_id, model, dataset, args):
    n_gpus = torch.cuda.device_count()
    databin = dataset[process_id]
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):
        model.cuda()
        for iter, pack in enumerate(data_loader):
            img_name = voc12.dataloader.decode_int_filename(pack['name'][0])

            orig_img = imageio.imread(os.path.join(args.voc12_root, 'JPEGImages/{}.jpg'.format(img_name)))
            orig_img_size = np.asarray(pack['size'])

            edge = model.infer(pack['img'][0].cuda(non_blocking=True))

            cam_dict = np.load(args.cam_out_dir + '/' + img_name + '.npy', allow_pickle=True).item()

            cams = cam_dict['cam'].cuda()
            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')

            rw = indexing.propagate_to_edge(cams, edge, beta=args.beta, exp_times=args.exp_times, radius=5)

            rw_up = F.interpolate(rw, scale_factor=4, mode='bilinear', align_corners=False)[..., 0, :orig_img_size[0],
                    :orig_img_size[1]]
            rw_up = rw_up / torch.max(rw_up)
            rw_up_bg = F.pad(rw_up, (0, 0, 0, 0, 1, 0), value=args.sem_seg_bg_thres)
            rw_pred = torch.argmax(rw_up_bg, dim=0).cpu().numpy()

            rw_pred_raw = keys[rw_pred]
            image = Image.fromarray(rw_pred_raw.astype(np.uint8)).convert('P')
            image.putpalette(palette)
            image.save(os.path.join(args.sem_seg_out_dir, img_name + '.png'))

            rw_pred_crf = imutils.crf_inference_label(orig_img, rw_pred, sxy=20, n_labels=keys.shape[0])
            rw_pred_crf = keys[rw_pred_crf]
            image_crf = Image.fromarray(rw_pred_crf.astype(np.uint8)).convert('P')
            image_crf.putpalette(palette)
            image_crf.save(os.path.join(args.sem_seg_out_dir, img_name + '_crf.png'))

            imageio.imsave(os.path.join(args.sem_seg_out_dir, img_name + '_edge.png'),
                           (edge[0] * 255).detach().cpu().numpy().astype(np.uint8))
            imageio.imsave(os.path.join(args.sem_seg_out_dir, img_name + '_orig.png'), orig_img.astype(np.uint8))

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5 * iter + 1) // (len(databin) // 20)), end='')


def run(args):
    model = Boundary()
    model.load_state_dict(torch.load(args.bes_weights_name), strict=False)
    model.eval()

    n_gpus = torch.cuda.device_count()

    dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.infer_list,
                                                             voc12_root=args.voc12_root,
                                                             scales=(1.0,))
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print("[", end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print("]")

    torch.cuda.empty_cache()
