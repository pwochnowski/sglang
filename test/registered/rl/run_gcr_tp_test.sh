#!/bin/bash
cd "$(git rev-parse --show-toplevel)"

export GCR_PRELOAD_PATH=/root/GCR/GCR/libpreload.so:/root/GCR/GCR/libcuda.so

export CUDA_VISIBLE_DEVICES="2,3"

export NCCL_SHM_DISABLE=1
export NCCL_NET_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_CUMEM_ENABLE=1
export NCCL_MAX_P2P_NCHANNELS=2
# export NCCL_TIMEOUT=3600
# export NCCL_DEBUG=INFO

# export NCCL_NET=Socket
python -m pytest test/registered/rl/test_gcr_tp.py -v -s 2>&1
