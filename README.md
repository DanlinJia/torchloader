# torchloader

## Architecture
Torchloader is deployed on a GPU cluster of multiple GPU nodes. Such a cluster contains a coordinator (Master) and multiple worker (Worker) processes. The Master process is Singleton and can be launched on a GPU or CPU node. Each GPU node joins the cluster by launching a Worker process on it, which manages the resources on the GPU node. Worker processes join the cluster by registering to the Master process. On each worker node (where the Worker processes are launched), an Application Master (AM) is spawned to manage all applications on it.

## Scheduler
The scheduler is responsible for launching, pausing, and resuming applications by sending signals to the Master process accordingly. The scheduler activates data-loading worker reallocation with the help of two modules, the Throughput Prediction Model (TPM) and Worker Allocation Algorithm (WAA). When an arrival or departure event occurs, the scheduler uses TPM to predict the number of data-loading workers applications required to achieve the maximum training throughput and uses WAA to allocate data-loading workers to optimize the overall training throughput (i.e., aggregated throughput of all applications).

## Submitter
The submitter reads a workload configuration file to generate DDL applications. A workload configuration file has the following columns:
- arch: the architecture of the DNN model.
- depth: the layer number of the DNN model.
- batch: the overall batch size across all devices on all nodes.
- workers: the overall workers across all devices on all nodes.
- output_folder: the folder to save experimental results with the prefix "trace"
- master: the IP of the pytorch master process, which can be any IP of the Worker nodes. 
- port: the port of the pytorch master process's URL.
- arrival_time: the arrival time of an application in a unit of seconds.
- cuda_device: the list of devices used for training the DNN model. Each colon pair indicates the device indexes on one node.
- start_iter: the start iteration of training a DNN model, usually is 0.
- start_epoch: the start epoch of training a DNN model, usually is 1.
- end_iter: the end iteration of training a DNN model.
- end_epoch: the end epoch of training a DNN model. The total number of iterations will be the number of epochs times the number of iterations of each epoch.
- node_size: how many nodes this application will be distributed across.

E.g.,
```
arch,depth,batch,workers,output_folder,master,port,arrival_time,cuda_device,start_iter,start_epoch,end_iter,end_epoch,node_size
resnet, 10, 512, 16, "test-48worker-2node-mod3",d3093, 12448, 1, ["0 1" "0 1"], 0, 1, 500, 1, 2
vgg, 11, 512, 16, "test-48worker-2node-mod3",d3093, 23448, 1, ["1 2" "1 2"], 0, 1, 500, 1, 2
googlenet, 1, 512, 16, "test-48worker-2node-mod3",d3093, 58448,  1, ["0 2" "0 2"], 0, 1, 500, 1, 2
```

## Quick Start
1. From the discovery gateway, allocate two nodes.
```
salloc -N 2 -p ce-mri --gres=gpu:v100:4 --exclusive
```
2. SSH to all nodes. Do the following on each node.
```
cd torchloader
conda activate pytorch_env
```
3. SSH to the node as master. Do the followings:
'''
cd torchloader
conda activate pytorch_env
'''
4. Modify cluster config file in dl_cluster_config.xml. Set master_ip to the master ip/hostname.
5. On master:
```
python3 -i dl_scheduler_test.py
```
6. On each Worker:
```
python3 -i dl_worker.py
# Important: execute above command on pytorch application master first. 
# The pytorch application master can be found in "dl_scheduler-test.conf.csv" under "master" column.
```

7. On master Python terminal:
```
run()
```
