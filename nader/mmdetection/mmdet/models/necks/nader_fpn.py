import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.registry import MODELS
from mmengine.model import BaseModule
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig

import sys
import os
sys.path.append('/home/jeongyoon/neural_architecture/NADER/nader')
from ModelFactory.register import Registers


@MODELS.register_module() #mmdetection에 이 클래스를 등록 
class NADERFPNAdapter(BaseModule):
    """ 
FPN 구현:
    1. Lateral (고정): 1x1 conv (Cx → 256)
    2. Top-down + Merge (고정): upsample + add
    3. Output (LLM 생성): base_block(256→256) 3개(P3, P4, P5에 동일 블록 복사)
    4. Extra (고정): stride=2 conv (P6, P7)
    
    Args:
        in_channels (List[int]): 입력 채널 수 (e.g., [512, 1024, 2048] for C3,C4,C5)
        out_channels (int): 출력 채널 수 (모든 레벨 동일, 256)
        num_outs (int): 출력 레벨 수 (e.g., 5 for P3~P7)
        start_level (int): 시작 레벨 (0: C2, 1: C3, ...)
        end_level (int): 끝 레벨 (-1: 마지막)
        add_extra_convs (bool | str): extra level 추가 여부 ('on_input', 'on_lateral', 'on_output')
        block_prefix (str): NADER 블록 이름 prefix (default: 'fpn_nasbench')
        blocks_dir (str): 블록 파일 디렉토리
    """

    def __init__(
        self,
        in_channels: list, # 입력 채널 수 (e.g., [512, 1024, 2048] for C3,C4,C5)
        out_channels: int, # 출력 채널 수 (모든 레벨 동일, 256)
        num_outs: int, # 출력 레벨 수 (e.g., 5 for P3~P7)
        start_level: int = 0, # 시작 레벨 (0: C2, 1: C3, ...)
        end_level: int = -1, # 끝 레벨 (-1: 마지막)
        add_extra_convs: bool | str = False, # P6, P7을 어떻게 만들지 추가 conv를 입력 feature map 쪽에서 먼저 수행 on_input
        relu_before_extra_convs: bool = False,
        no_norm_on_lateral: bool = False,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        act_cfg: OptConfigType = None,
        upsample_cfg: dict = dict(mode='nearest'),
        init_cfg: OptMultiConfig = None,
        block_prefix: str = 'fpn_nasbench', # 블록 이름 prefix (default: 'fpn_nasbench')
        blocks_dir: str = None, # 블록 파일 디렉토리
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = len(in_channels)
            assert num_outs >= len(in_channels) - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        
        if isinstance(add_extra_convs, str):
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:
            self.add_extra_convs = 'on_input'

        # Load NADER base block (NAS-Bench style)
        self._load_base_block(block_prefix, blocks_dir) # 베이스블록 가져옴 
        
        # Build network layers
        self._build_lateral_convs()  # Stem: 고정
        self._build_fpn_convs()       # Output: base block 복사
        self._build_extra_convs()     # Extra (P6, P7): 고정

    def _load_base_block(self, block_prefix: str, blocks_dir: str):
        """NAS-Bench 스타일로 base block 1개만 로드"""
        import json
        
        if blocks_dir is None:
            # 기본 경로: data/detection-bench/blocks/txts/fpn_nasbench.txt
            blocks_dir = os.path.join(
                os.path.dirname(__file__), 
                '../../../..',  # mmdet/models/necks/ -> nader/
                'data/detection-bench/blocks/txts'
            )
        
        # Register blocks
        from ModelFactory.block_factory import BlockFactory # dag를 코드로 변환
        
        # Find fpn_nasbench.txt
        src_txt = os.path.join(blocks_dir, 'fpn_nasbench.txt')
        if not os.path.exists(src_txt):
            # Fallback to relative path
            src_txt = '/home/jeongyoon/neural_architecture/NADER/nader/data/detection-bench/blocks/txts/fpn_nasbench.txt'
        
        if os.path.exists(src_txt):
            print(f"Loading NAS-Bench style FPN blocks from: {src_txt}")
            
            # Use detection mode for single base block
            blocks_out = os.path.join(os.path.dirname(src_txt), '..', 'blocks')
            os.makedirs(blocks_out, exist_ok=True)
            
            bf = BlockFactory(
                blocks_dir=blocks_out,
                type='base',
                register_path='ModelFactory.register',
                mode='detection'
            )
            bf.add_blocks_from_sections_path(src_txt, id_prefix=block_prefix)
            
            # Load base block class
            block_id = f'{block_prefix}__FPN_base_base'
            if block_id in Registers.block:
                self.base_block_class = Registers.block[block_id]
                print(f" Loaded base block: {block_id}")
            else:
                print(f" Base block not found: {block_id}, using ConvModule")
                self.base_block_class = None
        else:
            print(f" fpn_nasbench.txt not found: {src_txt}")
            self.base_block_class = None

    def _build_lateral_convs(self):
        """Stem: 고정된 1x1 conv (Cx → 256)"""
        self.lateral_convs = nn.ModuleList()
        
        for i in range(self.start_level, self.backbone_end_level):
            lateral_conv = ConvModule(
                self.in_channels[i],
                self.out_channels,
                1,
                conv_cfg=None,
                norm_cfg=dict(type='BN'),
                act_cfg=None,
                inplace=False
            )
            self.lateral_convs.append(lateral_conv)
            print(f"Lateral Conv {i}: {self.in_channels[i]} → {self.out_channels}")

    def _build_fpn_convs(self):
        """Output: base block 1개 생성해서 복사 (NAS-Bench 스타일)"""
        self.fpn_convs = nn.ModuleList()
        
        num_levels = self.backbone_end_level - self.start_level
        
        if self.base_block_class is not None:
            # base block 1개 생성해서 num_levels번 복사
            print(f"✓ Using NADER base block (NAS-Bench style): 1 block → {num_levels} copies")
            for i in range(num_levels):
                fpn_conv = self.base_block_class(
                    in_channels=self.out_channels,
                    out_channels=self.out_channels
                )
                self.fpn_convs.append(fpn_conv)
                print(f"  P{self.start_level + 2 + i}: base_block({self.out_channels} → {self.out_channels})")
        else:
            # base block 없으면 표준 3x3 conv 사용
            print(f" Base block not found, using standard ConvModule")
            for i in range(num_levels):
                fpn_conv = ConvModule(
                    self.out_channels,
                    self.out_channels,
                    3,
                    padding=1,
                    conv_cfg=None,
                    norm_cfg=dict(type='BN'),
                    act_cfg=None,
                    inplace=False
                )
                self.fpn_convs.append(fpn_conv)
                print(f"  P{self.start_level + 2 + i}: ConvModule({self.out_channels} → {self.out_channels})")

    def _build_extra_convs(self):
        """Extra levels (P6, P7): 고정 stride=2 conv"""
        self.extra_convs = nn.ModuleList()
        
        if not self.add_extra_convs or self.num_outs <= (self.backbone_end_level - self.start_level):
            return
        
        extra_levels = self.num_outs - (self.backbone_end_level - self.start_level)
        
        for i in range(extra_levels):
            if i == 0:
                # P6 from C5 (on_input) or P5 (on_lateral/on_output)
                if self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = self.out_channels
            else:
                # P7 from P6
                in_channels = self.out_channels
            
            extra_conv = ConvModule(
                in_channels,
                self.out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=None,
                norm_cfg=dict(type='BN'),
                act_cfg=None,
                inplace=False
            )
            self.extra_convs.append(extra_conv)
            print(f"Extra Conv P{self.backbone_end_level + i + 1}: {in_channels} → {self.out_channels} (stride=2)")

    def forward(self, inputs: tuple) -> tuple:
        """Forward function (FPN)
        
        Args:
            inputs (tuple): Backbone features (C3, C4, C5)
            
        Returns:
            tuple: FPN features (P3, P4, P5, P6, P7)
        """
        assert len(inputs) == len(self.in_channels) # 백본 피쳐 수와 입력 채널 수가 같아야 함

        # Step 1: Lateral connections 
        # Cx → 256
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # 가장 높은 레벨(P5)부터 시작해서 아래로 merge
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # Upsample
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + nn.functional.interpolate(
                laterals[i], size=prev_shape, **self.upsample_cfg)

        # Base block 또는 ConvModule 사용
        outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels)
        ]

        # Step 4: Extra levels (P6, P7)
        if self.add_extra_convs and self.num_outs > len(outs):
            # Determine source for extra convs
            if self.add_extra_convs == 'on_input':
                # Use C5 (raw backbone feature) - 2048 channels
                extra_source = inputs[self.backbone_end_level - 1]
            elif self.add_extra_convs == 'on_lateral':
                # Use lateral P5 - 256 channels
                extra_source = laterals[-1]
            elif self.add_extra_convs == 'on_output':
                # Use output P5 - 256 channels
                extra_source = outs[-1]
            else:
                extra_source = inputs[self.backbone_end_level - 1]
            
            # Apply extra convs sequentially
            for i, extra_conv in enumerate(self.extra_convs):
                if i == 0:
                    # P6 from source
                    if self.relu_before_extra_convs:
                        extra_source = nn.functional.relu(extra_source)
                    outs.append(extra_conv(extra_source))
                else:
                    # P7 from P6
                    if self.relu_before_extra_convs:
                        outs.append(extra_conv(nn.functional.relu(outs[-1])))
                    else:
                        outs.append(extra_conv(outs[-1]))

        return tuple(outs)
