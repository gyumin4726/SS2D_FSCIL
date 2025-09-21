_base_ = [
    '../_base_/models/vmamba_etf.py', '../_base_/datasets/cub_fscil.py',
    '../_base_/schedules/cub_80e.py', '../_base_/default_runtime.py'
]

# CUB requires different inc settings
inc_start = 100
inc_end = 200
inc_step = 10

model = dict(backbone=dict(type='VMambaBackbone',
                           model_name='vmamba_base_s2l15', 
                           pretrained_path='./vssm_base_0229_ckpt_epoch_237.pth',
                           out_indices=(0, 1, 2, 3), 
                           frozen_stages=0, 
                           channel_first=True),
             neck=dict(type='SS2DNeck',
                       in_channels=1024,
                       out_channels=1024,
                       feat_size=7,
                       num_layers=3,
                       use_multi_scale_skip=False,
                       multi_scale_channels=[128, 256, 512],
                       d_state=256,
                       dt_rank=256,
                       ssm_expand_ratio=1.0),
             head=dict(type='ETFHead',
                       in_channels=1024,
                       num_classes=200,
                       eval_classes=100,
                       with_len=False,
                       cal_acc=True,
                       loss=dict(type='CombinedLoss', dr_weight=1.0, ce_weight=0.0)),
             mixup=0,
             mixup_prob=0)

optimizer = dict(
    type='SGD',
    lr=0.1,
    momentum=0.9,
    weight_decay=0.0005,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'neck.mlp_proj.': dict(lr_mult=1.2),
            'neck.pos_embed': dict(lr_mult=1.2),
            'neck.ss2d_fscil.processor': dict(lr_mult=5.0),  # SS2D 프로세서
            # Multi-Scale Skip Connection 관련
            'neck.multi_scale_adapters.': dict(lr_mult=1.5),  # MultiScaleAdapter들
            'neck.cross_attention.': dict(lr_mult=1.0),       # Cross-attention
            'neck.query_proj.': dict(lr_mult=1.0),            # Query projection
            'neck.key_proj.': dict(lr_mult=1.0),              # Key projection  
            'neck.value_proj.': dict(lr_mult=1.0),            # Value projection
        }
    ))

optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='CosineAnnealingCooldown',
    min_lr=None,
    min_lr_ratio=0.1,
    cool_down_ratio=0.1,
    cool_down_time=10,
    by_epoch=False,
    # warmup
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.1,
    warmup_by_epoch=False)

runner = dict(type='EpochBasedRunner', max_epochs=20)

find_unused_parameters = True