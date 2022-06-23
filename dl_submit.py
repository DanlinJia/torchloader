import os
import subprocess
import time
import pickle
import pandas as pd
import numpy as np

class application():
    def __init__(self, appid, arch, depth, batch, workers, output_folder, port, wait_time=0, cuda_device="0, 1"):
        self.appid = appid
        self.arch = arch
        self.depth = depth
        self.batch = batch
        self.workers = workers
        self.output_folder = output_folder
        self.port = port
        self.wait_time = wait_time
        self.cuda_device = cuda_device

    def print(self):
        print("appid: {}, model: {}{}, batch: {}, workers: {}, output: {}, port: {}, cuda_device: {}" \
            .format(self.appid, self.arch, self.depth, self.batch, self.workers, self.output_folder, self.port, self.cuda_device))
    
class worker_scheduler():
    def __init__(self, cpu_cores, gpu_devices, submit_path, model_path, pretrained=True, cpu_model_path="", gpu_model_path=""):
        self.cpu_cores = cpu_cores
        self.gpu_devices = gpu_devices
        # path of a csv file where users submit application requests
        self.submit_path = submit_path
        # path of a csv file including information of all models (i.e., flops, weight size)
        self.model_path = model_path
        # initializing cpu and gpu throughput model
        if pretrained:
            assert(len(cpu_model_path)>0)
            with open(cpu_model_path, 'rb') as cpu_model_obj:
                self.cpu_model = pickle.load(cpu_model_obj)
            assert(len(gpu_model_path)>0)
            with open(gpu_model_path, 'rb') as gpu_model_obj:
                self.gpu_model = pickle.load(gpu_model_obj)
        else:
            self.gpu_model = None
            self.cpu_model = None
            # training at run-time

    def read_app_submissions(self):
        # from the path of submission requests, generate a set of corresponding applications
        df = pd.read_csv(self.submit_path, header=0, skipinitialspace=True)
        df.loc[:, "cuda_device"] = df["cuda_device"].apply(lambda x: x.replace(" ", ",") if (type(x)==str) else x)
        apps = []
        if len(df)==0:
            print("Warning: No application submitted")
        for i in range(len(df)):
            app = application(0, *df.loc[i, :])
            apps.append(app)
        return apps
    
    def tpt_predictor(self):
        # The throughput predictor reads submssion and model info, and generates a table including the max throughput can be achieved (bounded by gpu) and the throuput of one dataloader per device
        df = pd.read_csv(self.submit_path, header=0, skipinitialspace=True)
        model_df = pd.read_csv(self.model_path)
        df["num_device"]=df.cuda_device.apply(lambda x: len(x.split()) if (type(x)==str) else 1)
        df["model"] = df["arch"]+df["depth"].astype(str)
        df = df.join(model_df.set_index('model'), on="model")
        df["gpu_x0"] = df.batch/df.num_device*df.ops
        df["gpu_x1"]=(df.num_device-1)*df.params
        df["gpu_x2"]=df.params
        # calculate the throughput of a singe dataloader per device
        df["cpu_x0"] = 1/(df.num_device*1)
        df["gpu_y_"] = df[["gpu_x0", "gpu_x1", "gpu_x2"]].apply(lambda x: self.gpu_model.predict(np.array(x).reshape(1, -1))[0], axis=1)
        df["gpu_tpt"] = df.batch/df.gpu_y_
        df["cpu_tpt"] = df[["cpu_x0"]].apply(lambda x: 1/self.cpu_model.predict(np.array(x).reshape(1, -1))[0], axis=1)
        return df
    
    def worker_allocator(self):
        apps = self.read_app_submissions()
        tpt_df = self.tpt_predictor()
        tpt_df["worker_per_device"] = tpt_df.gpu_tpt/tpt_df.cpu_tpt
        tpt_df["allocated_per_device"] = tpt_df["worker_per_device"].astype(int) + 1
        tpt_df["overloaded_worker_per_device"] = tpt_df["allocated_per_device"] - tpt_df["worker_per_device"]
        tpt_df["tmp_id"] = tpt_df.index
        if (tpt_df["allocated_per_device"]*tpt_df["num_device"]).sum() > self.cpu_cores:
            tpt_df = tpt_df.sort_values(by=["overloaded_worker_per_device", "worker_per_device"], ascending=False)
            index = 0
            while (tpt_df["allocated_per_device"]*tpt_df["num_device"]).sum() > self.cpu_cores and tpt_df["allocated_per_device"].sum()!=len(tpt_df) :
                tpt_df["allocated_per_device"].iloc[index] -= 1
                index = (index + 1)%len(tpt_df)
            tpt_df = tpt_df.sort_values(by=["tmp_id"])
        for app_index in range(len(apps)):
            apps[app_index].workers = tpt_df["allocated_per_device"].iloc[app_index] * tpt_df["num_device"].iloc[app_index]
        return apps

    def submit_single_app(self, app, background=True):
        command = "bash ./submit.sh {} {} {} {} {} {} {} {}" \
                .format(app.appid, app.arch, app.depth, app.batch, app.workers, app.output_folder, app.port, app.cuda_device)
        try:
            if background:
                proc = subprocess.Popen(command, shell=True)
                print("application {} starts at PID {}".format(app.appid, proc.pid))
                return proc
            else:
                # subprocess.run(command, shell=True)
                os.system(command)
                return 0
        except subprocess.CalledProcessError as e:
            print(e.output)

    def run_apps(self, apps, background):
        for app in apps:
            time.sleep(app.wait_time)
            app.appid = time.time_ns()
            self.submit_single_app(app, background=background)

    def run_apps_from_path(self, background):
        # run applications submitted in a csv file
        apps = self.read_app_submissions()
        self.run_apps(apps, background)


submit_path = "dl_submit.conf.csv"
model_path="models.csv"
cpu_model_path="cpu_model"
gpu_model_path='gpu_model'
# names=["arch", "depth", "batch", "workers", "output_folder", "port", "submit_interval"]
# df = pd.read_csv(submit_path, header=0, names=names, skipinitialspace=True)
# apps = {}
# if len(df)==0:
#     print("Warning: No application submitted")
# runs = df["output_folder"].drop_duplicates().values
# for run in runs:
#     run_df = df[df["output_folder"]==run]
#     apps_in_run = []
#     for i in run_df.index:
#         app = application(0, *df.loc[i, :])
#         apps_in_run.append(app)
#     apps[run] = apps_in_run
#     run_apps(apps_in_run)
    #time.sleep(1200)
ws = worker_scheduler(cpu_cores=12, gpu_devices=4, \
                        submit_path=submit_path, \
                        model_path=model_path, \
                        pretrained=False, \
                        cpu_model_path=cpu_model_path, \
                        gpu_model_path=gpu_model_path)
ws.run_apps_from_path(background=False)