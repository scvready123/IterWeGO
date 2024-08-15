import os
import torch
import numpy as np
import random
import argparse
import time
import json
import pickle
#from build_vocab import Vocabulary
from data import get_loader
from model import train
from pathlib import Path
from PIL import Image
from torchvision import transforms

def parse_args():
    parser = argparse.ArgumentParser(description='order')
    parser.add_argument('--model', type=str, default=f'wego')  
    parser.add_argument('--main_path', type=str, default="./")
    parser.add_argument('--model_path', type=str, default="model")
    parser.add_argument('--ref', type=str, help='references, word unit')

    parser.add_argument('--train_image_dir', type=str, default= '', help='directory for resized train images')
    parser.add_argument('--val_image_dir', type=str, default='', help='directory for resized val images')
    parser.add_argument('--train_sis_path', type=str, default='', help='path for train sis json file')
    parser.add_argument('--val_sis_path', type=str, default='', help='path for val sis json file')
    parser.add_argument('--test_image_dir', type=str, default='', help='directory for resized test images')
    parser.add_argument('--test_sis_path', type=str, default='', help='path for test sis json file')

    parser.add_argument('--image_size', type=int, default=256, help='size for input images')
    parser.add_argument('--img_feature_size', type=int, default=512, help='dimension of image feature')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--thr_thi', type=float, default=0.8)
    parser.add_argument('--thr_iht', type=float, default=0.9)  
    parser.add_argument('--n_heads', type=int, default=4, help='number of heads')
    parser.add_argument('--embed_size', type=int, default=300, help='dimension of word embedding vectors')
    parser.add_argument('--d_rnn', type=int, default=768, help='hidden dimention size')
    parser.add_argument('--d_mlp', type=int, default=768, help='dimention size for FFN')
    parser.add_argument('--gnnl', default=2, type=int, help='stacked layer number')
    parser.add_argument('--attdp', default=0.2, type=float, help='self-att dropout')
    parser.add_argument('--initnn', default='standard', help='parameter init')
    parser.add_argument('--early_stop', type=int, default=5)
    parser.add_argument('--num_classes', type=int, default=2)

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--seed', type=int, default=34, help='seed for randomness')  
    parser.add_argument('--keep_cpts', type=int, default=1, help='save n checkpoints, when 1 save best model only')

    parser.add_argument('--eval_every', type=int, default=100, help='validate every * step')
    parser.add_argument('--save_every', type=int, default=2, help='save model every * step (5000)')
    parser.add_argument('--delay', type=int, default=1)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--margin', type=float, default=0.2, help='loss2')
    parser.add_argument('--lamda_t', type=float, default=1)
    parser.add_argument('--lamda_i', type=float, default=0.25)

    parser.add_argument('--lrdecay', type=float, default=0, help='learning rate decay')
    parser.add_argument('--patience', type=int, default=0, help='learning rate decay 0.5')
    parser.add_argument('--maximum_steps', type=int, default=5, help='maximum steps you take to train a model')
    parser.add_argument('--drop_ratio', type=float, default=0.2, help='dropout ratio')
    parser.add_argument('--input_drop_ratio', type=float, default=0.2, help='dropout ratio only for inputs')
    parser.add_argument('--grad_clip', type=float, default=0.0, help='gradient clipping')
    parser.add_argument('--do_test', type=float, default=False, help='')
    parser.add_argument('--load_from',default='', help='load from 1.modelname, 2.lastnumber, 3.number')
    parser.add_argument('--cp',default='', help='pretrained clip model') 
    parser.add_argument('--load_img_model',default='', help='pretrained img model')
    parser.add_argument('--load_txt_model',default='', help='pretrained txt model')
    parser.add_argument('--resume', action='store_true', help='when resume, need other things besides parameters')

    return parser.parse_args()


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def curtime():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def override(args, load_dict, except_name):
    for k in args.__dict__:
        if k not in except_name:
            args.__dict__[k] = load_dict[k]


if __name__ == '__main__':
    args = parse_args()

    if args.mode == 'train':
        load_txt_from = '{}'.format(args.load_txt_model)
        checkpoint_txt = torch.load(load_txt_from, map_location='cpu')

        load_img_from = '{}'.format(args.load_img_model)
        checkpoint_img = torch.load(load_img_from, map_location='cpu')

        main_path = Path(args.main_path)
        model_path = main_path / args.model_path
        args.model_path = str(model_path)


        if args.model == '[time]':
            args.model = time.strftime("%m.%d_%H.%M.", time.gmtime())

        set_seeds(args.seed)

        train_data_loader = get_loader(args.train_image_dir, args.train_sis_path, args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_data_loader = get_loader(args.val_image_dir, args.val_sis_path, 1, shuffle=False, num_workers=args.num_workers)
        test_data_loader = get_loader(args.test_image_dir, args.test_sis_path, 1, shuffle=False, num_workers=args.num_workers)
        print('{} Start training'.format(curtime()))
        train(args, train_data_loader, val_data_loader, test_data_loader, checkpoint_txt, checkpoint_img)
