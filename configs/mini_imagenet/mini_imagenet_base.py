_base_ = [
    '../_base_/models/vmamba_etf.py',
    '../_base_/datasets/mini_imagenet_fscil.py',
    '../_base_/schedules/mini_imagenet_500e.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(backbone=dict(type='VMambaBackbone',
                           model_name='vmamba_base_s2l15',  # 모델 변경
                           pretrained_path='./vssm_base_0229_ckpt_epoch_237.pth',
                           out_indices=(0, 1, 2, 3),  # Extract features from all 4 stages
                           frozen_stages=0,  # Freeze patch embedding and first stage
                           channel_first=True),
             neck=dict(type='SS2DNeck',
                       in_channels=1024,  # VMamba base stage4 output channels
                       out_channels=1024,
                       feat_size=3,  # 224 / (4*8) = 7 (patch_size=4, 4 downsample stages with 2x each)
                       num_layers=2,
                       # SS2D-specific parameters
                       d_state=256,  # SS2D hidden state dimension
                       dt_rank=256,  # SS2D delta rank
                       ssm_expand_ratio=1.0,  # SS2D expansion ratio
                       # Enhanced skip connection settings (MASC-M) for VMamba features
                       use_multi_scale_skip=False,
                       multi_scale_channels=[128, 256, 512]),
             head=dict(type='ETFHead',
                       in_channels=1024,
                       with_len=True,
                       loss=dict(type='CombinedLoss', dr_weight=0.0, ce_weight=1.0)),
             mixup=0,
             mixup_prob=0)

optimizer = dict(
    type='SGD',
    lr=0.25,
    momentum=0.9,
    weight_decay=0.0005,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'neck.mlp_proj': dict(lr_mult=1.2),
            'neck.pos_embed': dict(lr_mult=10.0),
            'neck.ss2d_fscil.processor': dict(lr_mult=10.0),  # SS2D 프로세서
        }
    ))

find_unused_parameters=True