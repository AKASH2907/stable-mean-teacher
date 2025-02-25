# from networks.efficientunet import Effi_UNet
# from networks.enet import ENet
# from networks.pnet import PNet2D
from models.aux_networks.unet import UNet
from models.aux_networks.unet_less_filters import UNet
# import argparse
# from networks.vision_transformer import SwinUnet as ViT_seg
# from networks.config import get_config
# from networks.nnunet import initialize_network


def net_factory(net_type="unet", in_chns=1, class_num=3):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    if net_type == "unet+":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    
    # elif net_type == "enet":
    #     net = ENet(in_channels=in_chns, num_classes=class_num).cuda()
    # elif net_type == "unet_ds":
    #     net = UNet_DS(in_chns=in_chns, class_num=class_num).cuda()
    # # elif net_type == "unet_cct":
    # #     net = UNet_CCT(in_chns=in_chns, class_num=class_num).cuda()
    # # elif net_type == "unet_urpc":
    # #     net = UNet_URPC(in_chns=in_chns, class_num=class_num).cuda()
    # elif net_type == "efficient_unet":
    #     net = Effi_UNet('efficientnet-b3', encoder_weights='imagenet',
    #                     in_channels=in_chns, classes=class_num).cuda()
    # elif net_type == "ViT_Seg":
    #     net = ViT_seg(config, img_size=args.patch_size,
    #                   num_classes=args.num_classes).cuda()
    # # elif net_type == "pnet":
    # #     net = PNet2D(in_chns, class_num, 64, [1, 2, 4, 8, 16]).cuda()
    # elif net_type == "nnUNet":
    #     net = initialize_network(num_classes=class_num).cuda()
    else:
        net = None
    return net

