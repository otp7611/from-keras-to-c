#!/usr/bin/env bash
#set -x

basepath=$(cd `dirname $0`; pwd)
cd ${basepath}

cp data/mnist.npz ~/.keras/datasets
pip install -i https://pypi.mirrors.ustc.edu.cn/simple/  bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow-2.18.1-cp310-cp310-linux_x86_64.whl
LD_LIBRARY_PATH=/media/workspace/clang17/instdir/lib/x86_64-unknown-linux-gnu:/media/workspace/cuda1204/instdir/lib64:/media/workspace/cudnn0910/instdir/lib/  XLA_FLAGS=--xla_gpu_cuda_data_dir=/media/workspace/github/tensorflow/OutputUserRoot//458e09755a092c4c605421b63672f1ff/external/cuda_nvcc TF_ENABLE_ONEDNN_OPTS=0 python3 ../train_and_export.py
LD_LIBRARY_PATH=/media/workspace/clang17/instdir/lib/x86_64-unknown-linux-gnu:/media/workspace/cuda1204/instdir/lib64:/media/workspace/cudnn0910/instdir/lib/  XLA_FLAGS=--xla_gpu_cuda_data_dir=/media/workspace/github/tensorflow/OutputUserRoot//458e09755a092c4c605421b63672f1ff/external/cuda_nvcc TF_ENABLE_ONEDNN_OPTS=0 ../builddir/model_run
