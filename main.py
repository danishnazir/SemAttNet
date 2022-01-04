import argparse
import os

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import time

from dataloaders.kitti_loader import  input_options, KittiDepth
from metrics import AverageMeter, Result

import helper
import vis_utils

from model import three_branch_bb
from model import A_CSPN_plus_plus
from torch.optim.lr_scheduler import ReduceLROnPlateau


parser = argparse.ArgumentParser(description='Sparse-to-Dense')
parser.add_argument('-n',
                    '--network-model',
                    type=str,
                    default="bb",
                    choices=["bb", "sem_att"],
                    help='choose a model: enet or penet'
                    )
parser.add_argument('--workers',
                    default=4,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs',
                    default=100,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run (default: 100)')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--start-epoch-bias',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number bias(useful on restarts)')

parser.add_argument('-b',
                    '--batch-size',
                    default=1,
                    type=int,
                    help='mini-batch size (default: 1)')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=1.2727e-3,
                    type=float,
                    metavar='LR',
                    help='initial learning rate (default 1e-5)')
parser.add_argument('--weight-decay',
                    '--wd',
                    default=1e-6,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 0)')
parser.add_argument('--print-freq',
                    '-p',
                    default=10,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')

parser.add_argument('--resume',
                    default="",
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--data-folder',
                    default="",
                    type=str,
                    metavar='PATH',
                    help='data folder (default: none)')
parser.add_argument('--data-folder-rgb',
                    default="",
                    type=str,
                    metavar='PATH',
                    help='data folder rgb (default: none)')

parser.add_argument('--data-semantic',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='data folder test results(default: none)')

#data_folder_save

parser.add_argument('--test-save',
                    default="test_results/",
                    type=str,
                    metavar='PATH',
                    help='data folder test results(default: none)')
parser.add_argument('-i',
                    '--input',
                    type=str,
                    default='rgbd',
                    choices=input_options,
                    help='input: | '.join(input_options))
parser.add_argument('--val',
                    type=str,
                    default="select",
                    choices=["select", "full"],
                    help='full or select validation set')
parser.add_argument('--jitter',
                    type=float,
                    default=0.1,
                    help='color jitter for images')
parser.add_argument('--rank-metric',
                    type=str,
                    default='rmse',
                    choices=[m for m in dir(Result()) if not m.startswith('_')],
                    help='metrics for which best result is saved')

#default=''
parser.add_argument('-e', '--evaluate', default='', type=str, metavar='PATH')

parser.add_argument('-f', '--freeze-backbone', action="store_true", default=False,
                    help='freeze parameters in backbone')


parser.add_argument('--s_p',   default="",type=str, metavar='PATH')

parser.add_argument('--val_results',   default="val_results/",type=str, metavar='PATH')



#defalut=False
parser.add_argument('--test', action="store_true", default=False,
                    help='save result kitti test dataset for submission')
parser.add_argument('--cpu', action="store_true", default=False, help='run on cpu')

#random cropping
parser.add_argument('--not-random-crop', action="store_true", default=False,
                    help='prohibit random cropping')
parser.add_argument('-he', '--random-crop-height', default=320, type=int, metavar='N',
                    help='random crop height')
parser.add_argument('-w', '--random-crop-width', default=1216, type=int, metavar='N',
                    help='random crop height')

parser.add_argument('-d', '--dilation-rate', default="2", type=int,
                    choices=[1, 2, 4],
                    help='CSPN++ dilation rate')

args = parser.parse_args()
args.result = args.s_p
args.use_rgb = ('rgb' in args.input)
args.use_d = 'd' in args.input
args.use_g = 'g' in args.input
args.val_h = 352
args.val_w = 1216
print(args)

cuda = torch.cuda.is_available() and not args.cpu
if cuda:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("=> using '{}' for computation.".format(device))





def iterate(mode, args, loader, model, optimizer, logger, epoch):
    actual_epoch = epoch - args.start_epoch + args.start_epoch_bias

    block_average_meter = AverageMeter()
    block_average_meter.reset(False)
    average_meter = AverageMeter()
    meters = [block_average_meter, average_meter]

    # switch to appropriate mode
    assert mode in ["train", "val", "eval", "test_prediction", "test_completion"], \
        "unsupported mode: {}".format(mode)

    model.eval()
    lr = 0

    torch.cuda.empty_cache()
    for i, batch_data in enumerate(loader):
        dstart = time.time()
        batch_data = {
            key: val.to(device)
            for key, val in batch_data.items() if val is not None
        }

        gt = batch_data[
            'gt'] if mode != 'test_prediction' and mode != 'test_completion' else None
        data_time = time.time() - dstart

        pred = None
        start = None
        gpu_time = 0

        if(args.network_model == 'bb'):
            start = time.time()
            st1_pred, st2_pred, st3_pred, pred = model(batch_data)
        else:
            start = time.time()
            rgb_conf, semantic_conf, d_conf, rgb_depth, semantic_depth, d_depth,coarse_depth,pred = model(batch_data)


        if(args.evaluate):
            gpu_time = time.time() - start


        if mode == "val":
            str_i = str(i)
            path_i = str_i.zfill(10) + '.png'
            path = os.path.join(args.val_results, path_i)
            vis_utils.save_depth_as_uint8colored(batch_data['d'], gt, pred, path)

        if mode == "test_completion":
            str_i = str(i)
            path_i = str_i.zfill(10) + '.png'
            path = os.path.join(args.test_save, path_i)
            vis_utils.save_depth_as_uint16png_upload(pred, path)

        if(not args.evaluate):
            gpu_time = time.time() - start

        with torch.no_grad():
            mini_batch_size = next(iter(batch_data.values())).size(0)
            result = Result()
            if mode != 'test_prediction' and mode != 'test_completion':
                result.evaluate(pred.data, gt.data)
                [
                    m.update(result, gpu_time, data_time, mini_batch_size)
                    for m in meters
                ]


                logger.conditional_print(mode, i, epoch, lr, len(loader),
                                 block_average_meter, average_meter)

    avg = average_meter.average()
    is_best = logger.rank_conditional_save_best(mode, avg, epoch)
    logger.conditional_summarize(mode, avg, is_best)

    return avg, is_best

def main():
    global args
    checkpoint = None
    is_eval = False
    if args.evaluate:
        args_new = args
        if os.path.isfile(args.evaluate):
            print("=> loading checkpoint '{}' ... ".format(args.evaluate),
                  end='')
            checkpoint = torch.load(args.evaluate, map_location=device)
            args.start_epoch = checkpoint['epoch'] + 1
            args.data_folder = args_new.data_folder
            args.val = args_new.val
            is_eval = True

            print("Completed.")
        else:
            is_eval = True
            print("No model found at '{}'".format(args.evaluate))
            print("PLEASE  PROVIDE CORRECT PATH.")
            return


    print("=> creating model and optimizer ... ", end='')
    model = None
    if (args.network_model == 'bb'):
        model = three_branch_bb(args).to(device)
    else:
        model = A_CSPN_plus_plus(args).to(device)


    if checkpoint is not None:
      
        if (args.freeze_backbone == True):
            model.backbone.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint['model'], strict=False)
        print("=> checkpoint state loaded.")

    logger = helper.logger(args)
    if checkpoint is not None:
        logger.best_result = checkpoint['best_result']
    print("=> logger created.")

    test_dataset = None
    test_loader = None
    if (args.test):
        test_dataset = KittiDepth('test_completion', args)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True)
        iterate("test_completion", args, test_loader, model, None, logger, 0)
        return

    val_dataset = KittiDepth('val', args)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True)  # set batch size to be 1 for validation
    print("\t==> val_loader size:{}".format(len(val_loader)))

    if is_eval == True:
        for p in model.parameters():
            p.requires_grad = False

        result, is_best = iterate("val", args, val_loader, model, None, logger,
                              args.start_epoch - 1)
        return

if __name__ == '__main__':
    main()