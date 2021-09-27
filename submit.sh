
#!/bin/bash

appid=$1
arch=$2
depth=$3
batch=$4
workers=$5
folder=$6
port=$7
cuda_device=$8

# Print help text and exit.
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  echo "Usage: "
  echo "- submit.sh <appid> <arch> <depth> <batch> <workers> <folder> <port> <cuda_device>"
  echo 
  echo "Examples:"
  echo "- bash submit.sh 0 resnet 18 128 16 ./out_put_folder 5663 '0 1 2 3' "
  exit 1
fi

#/tmp/opt/nvidia/nsight-systems/2021.1.1/target-linux-x64/nsys profile
CUDA_VISIBLE_DEVICES=$cuda_device \
python3 pretrain.py \
        ../../datasets/ImageNet/ \
        --app-id $appid \
        --arch $arch \
        --depth $depth \
        --batch-size $batch \
        --iteration 500 \
        --epochs 1 \
        --workers $workers \
        --dist-url tcp://127.0.0.1:$port \
        --dist-backend 'nccl' \
        --multiprocessing-distributed \
        --world-size 1 \
        --rank 0 \
        --trace  ./trace/$folder \
        --print-freq 100 \
        #--profile --profile-name test_dist --profile-epochs 1 --profile-batches 20 --record-shapes --profile-memory --use-cuda
        #> /tmp/home/geyuan/pytorch_test/logs/trace_3_9/${TEST_NAME}.txt


# --profile --profile-name dist_test --profile-epochs 2 --profile-batches 100 --record-shapes --profile-memory --use-cuda