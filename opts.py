import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='loc_const')
    # FIXED SET STARTING...
    parser.add_argument('--bs', type=int, default=8, help='mini-batch size')
    parser.add_argument('--pf', type=int, default=100, help='print frequency every batch')
    parser.add_argument('--epochs', type=int, default=1, help='number of total epochs to run')
    parser.add_argument('--model_name', type=str, default='i3d', help='model name')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--sup_loc_loss', type=str, default='dice', help='dice or iou loss')
    parser.add_argument('--exp_id', type=str, default='debug', help='experiment name')
    
    parser.add_argument('--pkl_file_label', type=str, default='train_annots_10_labeled_random.pkl', help='label subset')
    parser.add_argument('--pkl_file_unlabel', type=str, default='train_annots_90_unlabeled_random.pkl',
                        help='unlabel subset')
    
    parser.add_argument('--const_loss', type=str, default='l2', help='consistency loss type')

    parser.add_argument('--wt_loc', type=float, default=1, help='sup loc loss weight')
    parser.add_argument('--wt_cls', type=float, default=1, help='sup class loss weight')
    parser.add_argument('--wt_cons', type=float, default=0.1, help='consistency loss weight')
    # FIXED SET END...
    
    # VARIABLES STARTING...
    parser.add_argument('-at', '--aug_type', type=int, help="0-spatial, 1- temporal, 2 - both")
    parser.add_argument('-ema', '--ema_val', type=float, help="0.5-0.99")
    parser.add_argument('-t', '--temp', type=float, help="0.5-0.99")
    
    # Thresholds
    parser.add_argument('--thresh_epoch', type=int, default=11, help='thresh epoch to introduce pseudo labels')
    parser.add_argument('--ramp_thresh', type=int, default=0, help='ramp up consistency loss till which epoch')

    parser.add_argument('-ut', '--upper_thresh', type=float, default=None, help="0.5/0.7")
    parser.add_argument('-lt', '--lower_thresh', type=float, default=None, help="0.3/0.5")
    parser.add_argument('-lts', '--lower_thresh_st', type=float, default=0.1, help="0.1/0.2")

    # Burn-in params
    parser.add_argument('-burn', '--burn_in', action='store_true', help='use burn in weights')
    parser.add_argument('-bw', '--burn_wts', type=str, default='debug', help='experiment name')
    parser.add_argument('-baw', '--burn_aux_wts', type=str, default='debug', help='experiment name')

    # AUXILIARY PARAMS
    # parser.add_argument('--aux_depth', type=int, default=3, help='depth of auxiliary n/w')
    parser.add_argument('-ac', '--aux_channels', type=int, default=1, help='unet i/p channels')
    parser.add_argument('--aux_4d', action='store_true', help='use sigmoid on models raw input')
    parser.add_argument('-an', '--aux_nw', default="3d", type=str, metavar='TYPE',
                        choices=['2d', '2d+', '3d', '2d3d'],
                        help='dataset to use')
    
    parser.add_argument('--dice', action='store_true', help='L2+dice')
    parser.add_argument('--opt1', action='store_true', help='L2+dice')
    parser.add_argument('--opt2', action='store_true', help='L2+dice')
    parser.add_argument('--opt3', action='store_true', help='L2+dice')
    
    # loss add types
    parser.add_argument('--opt4', action='store_true', help='L2+dice')
    parser.add_argument('--opt5', action='store_true', help='L2+dice')
    
    parser.add_argument('--scheduler', action='store_true', help='lr scheduler')

    parser.add_argument('--wt_diff', type=float, default=10, help="dop weight")
    parser.add_argument('--wt_l2', type=float, default=1, help="normal l2 weight")

    # define seed params
    parser.add_argument('--seed', type=int, default=47, help='seed for initializing training.')
    parser.add_argument('--seed_data', type=int, default=47, help='seed variation pickle files')
    args = parser.parse_args()
    return args
