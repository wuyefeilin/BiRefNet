import os
import math


class Config():
    def __init__(self) -> None:
        self.cxt_num = [0, 3][1]    # multi-scale skip connections from encoder
        self.mul_scl_ipt = ['', 'add', 'cat'][2]
        self.refine = ['', 'itself', 'RefUNet', 'Refiner', 'RefinerPVTInChannels4'][0]
        self.progressive_ref = self.refine and True
        self.ender = self.progressive_ref and False
        self.scale = self.progressive_ref and 2
        self.dec_att = ['', 'ASPP', 'ASPPDeformable'][1]
        self.squeeze_block = ['', 'BasicDecBlk_x1', 'ResBlk_x4', 'ASPP_x3', 'ASPPDeformable_x3'][1]
        self.dec_blk = ['BasicDecBlk', 'ResBlk', 'HierarAttDecBlk'][0]
        self.auxiliary_classification = False
        self.refine_iteration = 1
        self.freeze_bb = False
        self.compile_and_precisionHigh = True
        self.load_all = True

        self.size = 1024
        self.batch_size = 5
        self.IoU_finetune_last_epochs = [-20, 0][0]     # choose 0 to skip
        self.ms_supervision = False
        if self.dec_blk == 'HierarAttDecBlk':
            self.batch_size = 2 ** [0, 1, 2, 3, 4][2]
        self.model = [
            'BSL',
            # 'PVTVP',
        ][0]

        # Components
        self.lat_blk = ['BasicLatBlk'][0]
        self.dec_channels_inter = ['fixed', 'adap'][0]

        # Backbone
        self.bb = [
            'vgg16', 'vgg16bn', 'resnet50',         # 0, 1, 2
            'pvt_v2_b2', 'pvt_v2_b5',               # 3-bs10, 4-bs5
            'swin_v1_b', 'swin_v1_l'                # 5-bs9, 6-bs6
        ][5]
        self.lateral_channels_in_collection = {
            'vgg16': [512, 256, 128, 64], 'vgg16bn': [512, 256, 128, 64], 'resnet50': [1024, 512, 256, 64],
            'pvt_v2_b2': [512, 320, 128, 64], 'pvt_v2_b5': [512, 320, 128, 64],
            'swin_v1_b': [1024, 512, 256, 128], 'swin_v1_l': [1536, 768, 384, 192],
        }[self.bb]
        if self.mul_scl_ipt == 'cat':
            self.lateral_channels_in_collection = [channel * 2 for channel in self.lateral_channels_in_collection]
        self.cxt = self.lateral_channels_in_collection[1:][::-1][-self.cxt_num:] if self.cxt_num else []
        self.sys_home_dir = '/home/user2'
        self.weights_root_dir = os.path.join(self.sys_home_dir, 'weights')
        self.weights = {
            'pvt_v2_b2': os.path.join(self.weights_root_dir, 'pvt_v2_b2.pth'),
            'pvt_v2_b5': os.path.join(self.weights_root_dir, ['pvt_v2_b5.pth', 'pvt_v2_b5_22k.pth'][0]),
            'swin_v1_b': os.path.join(self.weights_root_dir, ['swin_base_patch4_window12_384_22kto1k.pth', 'swin_base_patch4_window12_384_22k.pth'][0]),
            'swin_v1_l': os.path.join(self.weights_root_dir, ['swin_large_patch4_window12_384_22kto1k.pth', 'swin_large_patch4_window12_384_22k.pth'][0]),
        }

        # Training
        self.num_workers = 5        # will be decrease to min(it, batch_size) at the initialization of the data_loader 
        self.optimizer = ['Adam', 'AdamW'][0]
        self.lr = 1e-5 * math.sqrt(self.batch_size / 5)  # adapt the lr linearly
        self.lr_decay_epochs = [1e4]    # Set to negative N to decay the lr in the last N-th epoch.
        self.only_S_MAE = False

        # Data
        self.data_root_dir = os.path.join(self.sys_home_dir, 'datasets/dis')
        self.dataset = 'DIS5K'
        self.preproc_methods = ['flip', 'enhance', 'rotate', 'crop', 'pepper'][:1]

        # Loss
        self.lambdas_pix_last = {
            # not 0 means opening this loss
            # original rate -- 1 : 30 : 1.5 : 0.2, bce x 30
            'bce': 30 * 1,          # high performance
            'iou': 0.5 * 1,         # 0 / 255
            'iou_patch': 0.5 * 0,   # 0 / 255, win_size = (64, 64)
            'mse': 150 * 0,         # can smooth the saliency map
            'triplet': 3 * 0,
            'reg': 100 * 0,
            'ssim': 5 * 0,          # help contours,
            'cnt': 5 * 0,          # help contours
        }
        self.lambdas_cls = {
            'ce': 5.0
        }
        # Adv
        self.lambda_adv_g = 10. * 0        # turn to 0 to avoid adv training
        self.lambda_adv_d = 3. * (self.lambda_adv_g > 0)

        # others
        self.device = [0, 'cpu'][0]     # .to(0) = .to('cuda:0')

        self.batch_size_valid = 1
        self.rand_seed = 7
        run_sh_file = [f for f in os.listdir('.') if 'go.sh' == f] + [os.path.join('..', f) for f in os.listdir('..') if 'go.sh' == f]
        with open(run_sh_file[0], 'r') as f:
            lines = f.readlines()
            self.val_last = int([l.strip() for l in lines if 'val_last=' in l][0].split('=')[-1])
            self.save_step = int([l.strip() for l in lines if 'step=' in l][0].split('=')[-1])
