from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule
from rope import *
from timm.models.layers import trunc_normal_

from mmcls.models.builder import NECKS
from mmcls.utils import get_root_logger

from .mamba_ssm.modules.mamba_simple import Mamba
from .ss2d import SS2D


class MultiScaleAdapter(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels=512,
                 feat_size=7,
                 num_layers=2,
                 mid_channels=None):
        super(MultiScaleAdapter, self).__init__(init_cfg=None)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feat_size = feat_size
        self.num_layers = num_layers
        self.mid_channels = in_channels * 2 if mid_channels is None else mid_channels
        # 1. SS2D와 동일한 크기로 맞춤
        self.spatial_adapter = nn.AdaptiveAvgPool2d((feat_size, feat_size))
        
        # 2. 간단한 MLP 프로젝션
        self.mlp_proj = self._build_mlp(in_channels, out_channels, self.mid_channels, num_layers, feat_size)
        
    def _build_mlp(self, in_channels, out_channels, mid_channels, num_layers, feat_size):
        """Build MLP projection layers."""
        layers = []
        layers.append(nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0, bias=True))
        layers.append(build_norm_layer(dict(type='LN'), [mid_channels, feat_size, feat_size])[1])
        layers.append(nn.LeakyReLU(0.1))
        
        if num_layers == 3:
            layers.append(nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=True))
            layers.append(build_norm_layer(dict(type='LN'), [mid_channels, feat_size, feat_size])[1])
            layers.append(nn.LeakyReLU(0.1))
        
        layers.append(nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Step 1: 공간 크기 통일
        x = self.spatial_adapter(x)  # (B, C, feat_size, feat_size)
        
        # Step 2: MLP 프로젝션
        x = self.mlp_proj(x)         # (B, out_channels, feat_size, feat_size)
        
        return x  # (B, out_channels, feat_size, feat_size) - 공간 정보 유지


class SS2DProcessor(nn.Module):
    def __init__(self,
                 dim,
                 d_state=256,
                 dt_rank=256,
                 ssm_expand_ratio=1.0):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.dt_rank = dt_rank
        
        directions = ('h', 'h_flip', 'v', 'v_flip')
        self.ss2d_block = SS2D(
            dim,
            ssm_ratio=ssm_expand_ratio,
            d_state=d_state,
            dt_rank=dt_rank,
            directions=directions,
            use_out_proj=False,
            use_out_norm=True
        )

        self.norm = nn.LayerNorm(dim)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        B, H, W, dim = x.shape

        x_processed, _ = self.ss2d_block(x)  # [B, H, W, dim]

        x_processed = x_processed.permute(0, 3, 1, 2)  # [B, dim, H, W]
        x_processed = self.avg_pool(x_processed).view(B, -1)  # [B, dim]

        output = self.norm(x_processed)
        
        return output


class SS2DFSCIL(nn.Module):
    """단순화된 SS2D FSCIL 모듈"""

    def __init__(self,
                 dim,
                 d_state=256,
                 dt_rank=256,
                 ssm_expand_ratio=1.0):
        super().__init__()
        self.dim = dim
        
        # SS2D 프로세서 1개
        self.processor = SS2DProcessor(
            dim=dim,
            d_state=d_state,
            dt_rank=dt_rank,
            ssm_expand_ratio=ssm_expand_ratio
        )
        
    def forward(self, x):
        """단순화된 forward - SS2D 프로세서 직접 사용"""
        B, H, W, dim = x.shape
        
        # SS2D 프로세서로 직접 처리
        output = self.processor(x)  # [B, dim]
        
        return output


@NECKS.register_module()
class SS2DNeck(BaseModule):
    """SS2D Neck - 단순화된 FSCIL Neck"""

    def __init__(self,
                 in_channels=512,
                 out_channels=512,
                 d_state=256,
                 dt_rank=None,
                 ssm_expand_ratio=1.0,
                 feat_size=2,
                 mid_channels=None,
                 num_layers=2,
                 loss_weight_supp=0.0,
                 loss_weight_supp_novel=0.0,
                 loss_weight_sep=0.0,
                 loss_weight_sep_new=0.0,
                 param_avg_dim='0-1-3',
                 use_multi_scale_skip=False,
                 multi_scale_channels=[128, 256, 512]):
        super(SS2DNeck, self).__init__(init_cfg=None)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feat_size = feat_size
        self.mid_channels = in_channels * 2 if mid_channels is None else mid_channels
        self.num_layers = num_layers

        self.use_multi_scale_skip = use_multi_scale_skip
        self.multi_scale_channels = multi_scale_channels

        self.loss_weight_supp = loss_weight_supp
        self.loss_weight_supp_novel = loss_weight_supp_novel
        self.loss_weight_sep = loss_weight_sep
        self.loss_weight_sep_new = loss_weight_sep_new
        
        self.param_avg_dim = [int(item) for item in param_avg_dim.split('-')]
        
        self.logger = get_root_logger()
        self.logger.info(f"SS2D Neck initialized: single SS2D processor, no aux_loss")
        
        if self.use_multi_scale_skip:
            self.logger.info(f"Enhanced SS2D with Multi-Scale Skip Connections: {len(self.multi_scale_channels)} layers")
            self.logger.info(f"Multi-Scale Adapters: {self.multi_scale_channels} → {out_channels} channels")
            self.logger.info(f"Shared SS2D for skip connection processing after weighted combination")
        
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

        self.mlp_proj = self._build_mlp(
            in_channels, out_channels, self.mid_channels, num_layers, feat_size
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, feat_size * feat_size, out_channels)
        )
        trunc_normal_(self.pos_embed, std=.02)

        if self.use_multi_scale_skip:
            self.multi_scale_adapters = nn.ModuleList()
            for ch in self.multi_scale_channels:

                adapter = MultiScaleAdapter(
                    in_channels=ch,
                    out_channels=out_channels,
                    feat_size=self.feat_size,
                    num_layers=self.num_layers,
                    mid_channels=ch * 2 
                )
                self.multi_scale_adapters.append(adapter)
            
            directions = ('h', 'h_flip', 'v', 'v_flip')
            self.skip_ss2d = SS2D(
                out_channels,
                ssm_ratio=ssm_expand_ratio,
                d_state=d_state,
                dt_rank=dt_rank if dt_rank is not None else d_state,
                directions=directions,
                use_out_proj=False,
                use_out_norm=True
            )

        if self.use_multi_scale_skip:
            num_skip_sources = 1  # identity
            num_skip_sources += len(self.multi_scale_channels)
            
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=out_channels,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            
            self.query_proj = nn.Linear(out_channels, out_channels)
            self.key_proj = nn.Linear(out_channels, out_channels)
            self.value_proj = nn.Linear(out_channels, out_channels)

        self.ss2d_fscil = SS2DFSCIL(
            dim=out_channels,
            d_state=d_state,
            dt_rank=dt_rank if dt_rank is not None else d_state,
            ssm_expand_ratio=ssm_expand_ratio
        )
        
        self.init_weights()
        
    def _build_mlp(self, in_channels, out_channels, mid_channels, num_layers, feat_size):
        """Build MLP projection layers."""
        layers = []
        layers.append(nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0, bias=True))
        layers.append(build_norm_layer(dict(type='LN'), [mid_channels, feat_size, feat_size])[1])
        layers.append(nn.LeakyReLU(0.1))
        
        if num_layers == 3:
            layers.append(nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=True))
            layers.append(build_norm_layer(dict(type='LN'), [mid_channels, feat_size, feat_size])[1])
            layers.append(nn.LeakyReLU(0.1))
        
        layers.append(nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False))
        return nn.Sequential(*layers)
    
    def init_weights(self):
        """Initialize weights with proper scaling for SS2D and multi-scale adapters."""
        self.logger.info("🔧 Initializing SS2D Neck weights...")
        
        if self.use_multi_scale_skip:
            for i, adapter in enumerate(self.multi_scale_adapters):

                if hasattr(adapter, 'mlp_proj'):
                    first_layer = adapter.mlp_proj[0]  
                    if isinstance(first_layer, nn.Conv2d):
                        nn.init.kaiming_normal_(first_layer.weight, mode='fan_out', nonlinearity='relu')
                
                self.logger.info(f'Initialized MultiScaleAdapter {i} for channel {self.multi_scale_channels[i]} with {adapter.num_layers}-layer MLP')
            
            if hasattr(self, 'skip_ss2d'):
                with torch.no_grad():
                    if hasattr(self.skip_ss2d, 'in_proj'):
                        self.skip_ss2d.in_proj.weight.data *= 0.1
                self.logger.info('Initialized shared skip SS2D block for weighted feature processing')
    
    def forward(self, x, multi_scale_features=None):

        if isinstance(x, tuple):
            x = x[-1]  # layer4 as main input
            if self.use_multi_scale_skip and multi_scale_features is None and len(x) > 1:
                multi_scale_features = x[:-1]  # [layer1, layer2, layer3]
        
        # multi_scale_features가 없으면 오류 발생 (multi-scale 사용시에만)
        if self.use_multi_scale_skip and (multi_scale_features is None or len(multi_scale_features) == 0):
            raise ValueError('use_multi_scale_skip=True 인데 multi_scale_features가 없습니다. backbone에서 여러 레이어 출력을 반환하도록 설정하세요.')

        B, C, H, W = x.shape
        identity = x
        outputs = {}
        
        x_proj = self.mlp_proj(identity)  # [B, out_channels, H, W]
        x_proj = x_proj.permute(0, 2, 3, 1).view(B, H * W, -1)  # [B, H*W, out_channels]
        
        x_proj = x_proj + self.pos_embed  # [B, H*W, out_channels]
        
        x_spatial = x_proj.view(B, H, W, -1)  # [B, H, W, out_channels]

        ss2d_output = self.ss2d_fscil(x_spatial)

        final_output = ss2d_output

        skip_features_spatial = [identity] if self.use_multi_scale_skip else None  # 공간 정보 유지

        if self.use_multi_scale_skip and skip_features_spatial is not None:
            if multi_scale_features is not None:
                for i, feat in enumerate(multi_scale_features):
                    if i < len(self.multi_scale_adapters):
                        adapted_feat = self.multi_scale_adapters[i](feat)  # (B, out_channels, H, W)
                        skip_features_spatial.append(adapted_feat)

                        if hasattr(self, 'logger') and torch.rand(1).item() < 0.01:  # 1% 확률로 로그
                            self.logger.info(f"MultiScaleAdapter {i}: {feat.shape} → {adapted_feat.shape}")

        if self.use_multi_scale_skip and skip_features_spatial is not None and len(skip_features_spatial) > 1:
            # 모든 skip features는 이미 동일한 공간 크기 (MultiScaleAdapter에서 맞춤)
            B, C, H, W = skip_features_spatial[0].shape  # 기준 크기 (identity)
            
            skip_stack = torch.stack(skip_features_spatial, dim=1)  # [B, N, C, H, W]
            
            # 공간 차원을 유지하면서 가중치 계산을 위해 평균 풀링
            skip_flat = skip_stack.mean(dim=(-2, -1))  # [B, N, C] - 공간 정보 압축
            
            # Prepare Query (from SS2D output), Key, Value (from skip features)
            query = self.query_proj(ss2d_output).unsqueeze(1)  # [B, 1, C]
            keys = self.key_proj(skip_flat)         # [B, N, C]
            values = self.value_proj(skip_flat)     # [B, N, C]
            
            # Multi-head cross-attention
            attended_features, attention_weights = self.cross_attention(query, keys, values)
            # attention_weights: [B, 1, N]
            
            # softmax 정규화 (안정성 보강)
            weights = torch.softmax(attention_weights.squeeze(1), dim=-1)  # [B, N]
            
            # Skip features에 가중치 적용 (공간 정보 유지)
            weighted_skip = (weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * skip_stack).sum(dim=1)  # [B, C, H, W]
            
            # Apply shared SS2D to weighted skip features (공간 정보 활용)
            weighted_skip_spatial = weighted_skip.permute(0, 2, 3, 1)  # [B, H, W, C]
            skip_ss2d_output, _ = self.skip_ss2d(weighted_skip_spatial)  # [B, H, W, C]
            
            # Convert back to vector format
            skip_ss2d_output = skip_ss2d_output.permute(0, 3, 1, 2)  # [B, C, H, W]
            skip_ss2d_output = F.adaptive_avg_pool2d(skip_ss2d_output, (1, 1)).view(B, -1)  # [B, C]
            
            # 최종 출력: SS2D + SS2D-processed skip connections
            final_output = ss2d_output + 0.1 * skip_ss2d_output
            
            # 디버깅: cross-attention weights 출력
            if hasattr(self, 'logger') and torch.rand(1).item() < 0.01:  # 1% 확률로 로그
                weight_values = weights[0].detach().cpu().numpy()
                # Generate dynamic feature names based on actual skip features
                feature_names = ['layer4']
                if self.use_multi_scale_skip:
                    feature_names.extend([f'layer{i+1}' for i in range(len(self.multi_scale_channels))])
                feature_names = feature_names[:len(skip_features_spatial)]
                weight_info = ', '.join([f"{name}: {val:.3f}" for name, val in zip(feature_names, weight_values)])
                self.logger.info(f"Cross-attention weights: {weight_info}")
        else:
            final_output = ss2d_output
        
        # Prepare outputs
        outputs.update({
            'out': final_output,
            'main': ss2d_output,
        })

        if self.use_multi_scale_skip and skip_features_spatial is not None:
            outputs['skip_features'] = skip_features_spatial
        
        return outputs