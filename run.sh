#!/usr/bin/env bash
# Example launch script for S4T.
# `device_num` selects the GPU; `port_num` is a free port for torch.distributed.
device_num=0
port_num=29500

# ---------------------------------------------------------------------------
# Stage 1: Source-domain pre-training
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=$device_num python -m torch.distributed.launch --nproc_per_node=1 --master_port $port_num \
    main.py --config_exp ./configs/S4T/pretrain/taskonomy_nyud_resnet50_S4T.yml --run_mode train --seed 1

# ---------------------------------------------------------------------------
# Stage 2: Test-time training (TTT) on the target domain.
# Loads the pre-trained checkpoint referenced by the `pretrained_ckpt:` field of the config.
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=$device_num python -m torch.distributed.launch --nproc_per_node=1 --master_port $port_num \
    main.py --config_exp ./configs/S4T/ttt/taskonomy_nyud_resnet50_S4T.yml --run_mode train --seed 1
