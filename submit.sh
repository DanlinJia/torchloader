
#!/bin/bash

appid=$1
arch=$2
depth=$3
batch=$4
workers=$5
folder=$6
master=$7
port=$8
cuda_device=$9

# Print help text and exit.
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  echo "Usage: "
  echo "- submit.sh <appid> <arch> <depth> <batch> <workers> <folder> <port> <cuda_device>"
  echo 
  echo "Examples:"
  echo "- bash submit.sh 0 resnet 18 128 16 ./out_put_folder 127.0.0.1 5663 '0 1 2 3' "
  exit 1
fi

#/tmp/opt/nvidia/nsight-systems/2021.1.1/target-linux-x64/nsys profile
# nsys profile
CUDA_VISIBLE_DEVICES=$cuda_device \
num_gpus=$(awk -F '[0-9]' '{print NF-1}' <<<"$CUDA_VISIBLE_DEVICES")
echo "num_gpus="$num_gpus
python3 pretrain.py \
        /work/yanzhi_group/datasets/imagenet/ \
        --app-id $appid \
        --arch $arch \
        --depth $depth \
        --batch-size $batch \
        --iteration 500 \
        --epochs 1 \
        --workers $workers \
        --dist-url tcp://$master:$port \
        --dist-backend 'nccl' \
        --multiprocessing-distributed \
        --world-size 1 \
        --rank 0 \
        --trace  ./trace/$folder \
        --print-freq 10 \
        # --profile --profile-name test_dist --profile-epochs 1 --profile-batches 4 --record-shapes --profile-memory --use-cuda
# > ./trace/$folder/trace_test.txt


# --profile --profile-name dist_test --profile-epochs 2 --profile-batches 100 --record-shapes --profile-memory --use-cuda