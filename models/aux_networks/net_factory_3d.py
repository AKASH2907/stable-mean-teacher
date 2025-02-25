# from models.aux_networks.unet_3D import unet_3D
from models.aux_networks.unet_3D_mod import unet_3D

# from aux_networks.vnet import VNet
# from aux_networks.VoxResNet import VoxResNet
# from aux_networks.attention_unet import Attention_UNet
# from networks.nnunet import initialize_network


def net_factory_3d(net_type="unet_3D", in_chns=1, class_num=1):
    if net_type == "unet_3D":
        net = unet_3D(n_classes=class_num, in_channels=in_chns)
        # .cuda()
    elif net_type == "attention_unet":
        net = Attention_UNet(n_classes=class_num, in_channels=in_chns)
    elif net_type == "voxresnet":
        net = VoxResNet(in_chns=in_chns, feature_chns=64,
                        class_num=class_num)
    elif net_type == "vnet":
        net = VNet(n_channels=in_chns, n_classes=class_num,
                   normalization='batchnorm', has_dropout=True)
    # elif net_type == "nnUNet":
    #     net = initialize_network(num_classes=class_num).cuda()
    else:
        net = None
    return net
