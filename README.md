# torchloader

The goal of this project is to maximize overall throughput by tuning the number of workers (i.e., data loaders) for submitted applications.

### Project Roadmap:
1. running deep learning applications with different combinations of parameters.
2. modeling throughput of CPU-GPU clusters.
3. use throughput model to automatically tune the number of workers.

### Tools:
- `pretain.py` is the pytorch application code used to traine models, which receives a set of arguments. Check the usage of `pretain.py` with the following command.
```sh
python3 pretain.py --help
```

- `submit.sh` is a bash script wrapper on `pretain.py`, which exposes a subset of arguments to users, e.g., 

```sh
Usage: 
- submit.sh <appid> <arch> <depth> <batch> <workers> <folder> <port> <cuda_device>

Example:
- bash submit.sh 0 resnet 18 128 16 ./out_put_folder 5663 '0 1 2 3' 
```

- `dl_submit.py'.` is an application dispatcher, which reads applications' info from `dl_submit.conf.csv` and calls `submit.sh` to lauch applications. Please edit `dl_submit.conf.csv` before running `dl_submit.py'.
```sh
python3 dl_submit.sh
```
- `dl_submit.conf.csv` is a csv file containing applications' info, where each row is an application, and each column is a parameter of the application. `output_folder` specifies which folder the stats should save. `submit_interval` gives the interval between current application with the last one. `cuda_device` sets the GPUs to use. e.g.,

```sh
arch,depth,batch,workers,output_folder,port,submit_interval,cuda_device
resnet,12,128,1,single_1workers_1device,5336,0,"0"
resnet,14,128,1,single_1workers_1device,5336,0,"0"
```
The calling order is `dl_submit.py` to `submit.sh`, then to `pretain.py`.

`dl_submit.py` can run applications simutaneously or sequentially, based on the paramerter set in code `worker_scheduler.run_apps_from_path(background=False)`. If `backgoud` is set to `True`, all applications in `dl_submit.conf.csv` should run simutaneously, othervise applications are running in sequential.

Examples:
1. submit three sequential applications:
set `dl_submit.conf.csv`
```sh
arch,depth,batch,workers,output_folder,port,submit_interval,cuda_device
resnet,12,128,1,single_1workers_1device,5336,0,"0"
resnet,14,128,1,single_1workers_1device,5336,0,"0"
resnet,16,128,1,single_1workers_1device,5336,0,"0"
```
change the code in `dl_submit.py` to  `worker_scheduler.run_apps_from_path(background=False)`

2. submit three simutaneous applications:
set `dl_submit.conf.csv`
```sh
arch,depth,batch,workers,output_folder,port,submit_interval,cuda_device
resnet,12,128,1,single_1workers_1device,5336,0,"0"
resnet,14,128,1,single_1workers_1device,5337,0,"0"
resnet,16,128,1,single_1workers_1device,5338,0,"0"
```
change the code in `dl_submit.py` to  `worker_scheduler.run_apps_from_path(background=True)`. Note that the port number of three applications should be different.

3. submit applications in 10 second interval simutaneously/sequentially:
set `dl_submit.conf.csv`
```sh
arch,depth,batch,workers,output_folder,port,submit_interval,cuda_device
resnet,12,128,1,single_1workers_1device,5336,10,"0"
resnet,14,128,1,single_1workers_1device,5337,10,"0"
resnet,16,128,1,single_1workers_1device,5338,10,"0"
```

3. submit applications in different intervals simutaneously/sequentially:
set `dl_submit.conf.csv`
```sh
arch,depth,batch,workers,output_folder,port,submit_interval,cuda_device
resnet,12,128,1,single_1workers_1device,5336,10,"0"
resnet,14,128,1,single_1workers_1device,5337,20,"0"
resnet,16,128,1,single_1workers_1device,5338,30,"0"
```

4. submit applications to different GPU devices simutaneously/sequentially:
set `dl_submit.conf.csv`
```sh
arch,depth,batch,workers,output_folder,port,submit_interval,cuda_device
resnet,12,128,1,single_1workers_1device,5336,0,"0 1"
resnet,14,128,1,single_1workers_1device,5337,0,"0 2"
resnet,16,128,1,single_1workers_1device,5338,0,"3 4"
```
