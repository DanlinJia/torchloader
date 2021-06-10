

#CUDA_VISIBLE_DEVICES=0,1,2,3 python pretrain_test.py --arch resnet --depth 18 --dist-url 'tcp://127.0.0.1:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 [imagenet-folder with train and val folders]


#CUDA_VISIBLE_DEVICES=0,1,2,3 
#python3 danj-pretrain.py \


#/tmp/opt/nvidia/nsight-systems/2021.1.1/target-linux-x64/nsys profile 
python3 pretrain.py \
                                    ../../datasets/ImageNet/ \
                                    --app-id 0 \
                                    --arch resnet \
                                    --depth 35 \
                                    --batch-size 128 \
                                    --iteration 500 \
                                    --epochs 1 \
                                    --workers 16 \
                                    --dist-url 'tcp://127.0.0.1:5488' \
                                    --dist-backend 'nccl' \
                                    --multiprocessing-distributed \
                                    --world-size 1 \
                                    --rank 0 \
                                    --trace  ./trace/test \
                                    --print-freq 100 \
                                    #--profile --profile-name test_dist --profile-epochs 1 --profile-batches 20 --record-shapes --profile-memory --use-cuda
                                    #> /tmp/home/geyuan/pytorch_test/logs/trace_3_9/${TEST_NAME}.txt


# --profile --profile-name dist_test --profile-epochs 2 --profile-batches 100 --record-shapes --profile-memory --use-cuda