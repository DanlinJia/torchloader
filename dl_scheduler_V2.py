from builtins import enumerate
from typing import overload
import pandas as pd
import numpy as np
import os
import subprocess
import time, datetime
import pickle
import logging
import threading
import pickle
from multiprocessing import Process, Pipe, Lock
from dateutil import parser
from pandas.io.clipboards import read_clipboard
from dl_app import application, app_main


def conn_message(mesg_type, mesg_data=None):
    return [str(datetime.datetime.now()), mesg_type, mesg_data]

def worker_test_fn(appid, conn, delay=1, start_iter=0):
    for i in range(start_iter, 10):
        time.sleep(delay)
        print("{}: worker: {}, iteration {}".format(datetime.datetime.now(), appid, i))
        if conn.poll(timeout=0.01):
            if conn.recv()[0] == "Pause":
                time_stamp = datetime.datetime.now()
                conn.send(conn_message("Paused", i))
                print("{}: worker: {} paused at iter {}".format(time_stamp, appid, i))
                return 
    conn.send(conn_message("Finished"))
    return

class submitter():
    def __init__(self, submit_path, conn, time_window=10):
        self.submit_path = submit_path
        # a list of submitted applications
        # self.apps = collections.defaultdict(list)
        self.app_infos = []
        # application list
        self.apps = []
        # communication pipe for submitting apps
        self.conn = conn
        # cumulate app submissions within buffer_size
        self.buffer_size = time_window
        self.buffer = []
        self.submisson_num = 0

    def print_apps_info(self):
        for app in self.apps:
            app.print_info()
        return 0

    def get_app(self):
        return self.apps

    def read_app_submissions(self):
        # generate a set of applications by reading data from a csv file
        df = pd.read_csv(self.submit_path, header=0, skipinitialspace=True)
        df.loc[:, "cuda_device"] = df["cuda_device"].apply(lambda x: [int(d) for d in x.split(" ")] if (type(x)==str) else [int(x)])
        if len(df)==0:
            print("Warning: No application submitted")
        for i in range(len(df)):
            app = application(i, *df.loc[i, :])
            # self.apps[app.arrival_time].append(app)
            self.apps.append(app)
        self.submisson_num = len(self.apps)
        return 0

    def user_submiter_fn(self):
        def _submit():
            if len(self.buffer)>0:
                print("{}: submitter submits app {}".format(datetime.datetime.now(), [app.appid for app in self.buffer]))
                self.conn.send(conn_message("Arrival", self.buffer))
                self.buffer = []
        def _inwindow(app_ari, start, end):
            if app_ari >= start and app_ari <= end:
                return True
            return False
        self.read_app_submissions()
        cur_time = 0
        while len(self.apps) > 0:
            app = self.apps[0]
            if _inwindow(app.arrival_time, cur_time, cur_time+self.buffer_size):
                app = self.apps.pop(0)
                app.appid = int((app.arrival_time - cur_time)*1e9 + time.time_ns())
                expriment_name = "app{}_{}{}_{}_{}_{}".format(
                                app.appid, app.arch, app.depth, app.batch, app.workers, len(app.cuda_device))
                app.work_space = os.path.join("trace", app.work_space, expriment_name)
                os.system("mkdir -p {}".format(app.work_space))
                self.buffer.append(app)
                continue
            _submit()
            time.sleep(self.buffer_size)
            cur_time += self.buffer_size
        _submit()
        self.conn.send(conn_message("End", self.submisson_num))

    def main_fn(self):
        submiter_j = threading.Thread(target=self.user_submiter_fn, args=(), name="user_submitter", daemon=True)
        submiter_j.start()
        # submiter_j.join()


class dl_tpt_pridictor():
    def __init__(self, cpu_cores, gpu_devices, cpu_model_path='', gpu_model_path='', model_info_path=''):
        self.cpu_cores = cpu_cores
        self.gpu_devices = gpu_devices
        self.cpu_model = None
        self.gpu_model = None
        self.model_info = None
        try:
            if cpu_model_path:
                with open(cpu_model_path, 'rb') as cpu_model_obj:
                    self.cpu_model = pickle.load(cpu_model_obj)
            if gpu_model_path:
                with open(gpu_model_path, 'rb') as gpu_model_obj:
                    self.gpu_model = pickle.load(gpu_model_obj)
            if model_info_path:
                self.model_info = pd.read_csv(model_info_path)
        except Exception as e:
            print(e)
            raise

    def predict_cpu_tpt(self, app: application):
        num_device = len(app.cuda_device)
        cpu_x0 = 1/(num_device)
        return (1/self.cpu_model.predict(np.array([cpu_x0]).reshape(1, -1))[0])/num_device

    def predict_gpu_tpt(self, app: application):
        """
        return the maximum throughput an application can achieve on a single device.
        """
        num_device = len(app.cuda_device)
        model_name = "{}{}".format(app.arch, app.depth)
        flops = self.model_info[self.model_info["model"]==model_name]["flops"].drop_duplicates().item()
        params = self.model_info[self.model_info["model"]==model_name]["params"].drop_duplicates().item()
        c_flops = self.model_info[self.model_info["model"]==model_name]["C_flops"].drop_duplicates().item()*(app.batch/num_device)
        pc_flops = self.model_info[self.model_info["model"]==model_name]["PC_flops"].drop_duplicates().item()*(app.batch/num_device)
        dc_flops = self.model_info[self.model_info["model"]==model_name]["DC_flops"].drop_duplicates().item()*(app.batch/num_device)
        o_flops = self.model_info[self.model_info["model"]==model_name]["O_flops"].drop_duplicates().item()*(app.batch/num_device)
        gpu_x0, gpu_x1, gpu_x2, gpu_x3 = app.batch/num_device*flops, (num_device-1)*params, params, app.batch/num_device*params
        return (app.batch/len(app.cuda_device))/self.gpu_model.predict(np.array([gpu_x1, gpu_x2, gpu_x3, c_flops,pc_flops,dc_flops,o_flops ]).reshape(1, -1))[0]

    def count_app_per_device(self, apps):
        apps_count_device = {}
        for app in apps:
            for d in app.cuda_device:
                if d in apps_count_device:
                    apps_count_device[d] += 1
                else:
                    apps_count_device[d] = 1
        return apps_count_device

    def worker_allocator(self, apps, debug=False):
        diff_apps = []
        intact_apps = []
        apps_count_device = self.count_app_per_device(apps)
        
        tpt_df =  pd.DataFrame(columns=["appid", "gpu_tpt", "cpu_tpt", "cal_w", "over_w", "workers", "num_device"])
        for app in apps:
            num_device = len(app.cuda_device)
            cpu_tpt_per_device = self.predict_cpu_tpt(app)
            gpu_max_tpt = self.predict_gpu_tpt(app) / np.array([ apps_count_device[d] for d in app.cuda_device]).max()
            worker_per_device_float = gpu_max_tpt/cpu_tpt_per_device
            worker_per_device_int = int(worker_per_device_float) + 1
            overload_worker_per_device = worker_per_device_int - worker_per_device_float
            tpt_df.loc[len(tpt_df), :] = [app.appid, gpu_max_tpt, \
                                            cpu_tpt_per_device, worker_per_device_float, \
                                            overload_worker_per_device, worker_per_device_int, \
                                            num_device]
        if debug:
            print(tpt_df)
        if (tpt_df["workers"]*tpt_df["num_device"]).sum() > self.cpu_cores:
            tpt_df = tpt_df.sort_values(by=["over_w", "cal_w"], ascending=False)
            index = 0
            while (tpt_df["workers"]*tpt_df["num_device"]).sum() > self.cpu_cores and tpt_df["workers"].sum()!=len(tpt_df):
                if tpt_df["workers"].iloc[index] > 1:
                    tpt_df["workers"].iloc[index] -= 1
                index = (index + 1)%len(tpt_df)
        if debug:
            print(tpt_df)
        for app in apps:
            new_workers = tpt_df.loc[tpt_df["appid"]==app.appid, "workers"].item() * tpt_df.loc[tpt_df["appid"]==app.appid, "num_device"].item()
            if new_workers != app.workers:
                diff_apps.append(app)
                app.workers = new_workers
            else:
                intact_apps.append(app)
        return diff_apps, intact_apps
        

class dl_scheduler():
    def __init__(self, sub_conn, through_predictor, mode=3):
        # channels is a appid:[parent_conn, child_conn] mapping directory.
        self.channels = {}
        # a list of ready applications
        self.app_ready = []
        # a list of running applications
        self.app_running = []
        # a list of paused applications
        self.app_paused = []
        # a list of finished applications
        self.app_finished = []
        # a list of applications to finish
        self.finish_buffer = []
        # a record of lisenters created for each app
        self.lisenters = {}
        # communication pipe for receiving apps submitions
        self.sub_conn = sub_conn
        # count how many pause happend
        self.paused_counter = 0
        # count how many pause signal was sent
        self.paused_signal = 0
        self.stop_flag = -1
        # count how many apps finished
        self.finished_count = 0
        # throughput predictor initialization
        self.tp = through_predictor
        # global lock to protect spawning application
        self.lock = Lock()
        # a flag representing the scheduler status
        # if schduler is in pausing, hold on the following pausing.
        self.in_pausing = False
        # mode:
        # 1. no worker reallocation
        # 2. reallocate workers for arrival signal
        # 3. reallocate workers for both arrival and finish signals
        self.mode = mode
        self.logger = logging.getLogger("dl_scheduler.log")

    def init_logger(self):
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler("dl_scheduler.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def main_loop_fn(self):
        self.create_listener()

    def exec_single_app(self, app: application):
        try:
            _, child_conn = self.channels[app.appid]
            p = Process(target=app_main, args=(child_conn, app,), name="app-{}".format(app.appid), daemon=False)
            # p = Process(target=worker_test_fn, args=(app.appid, child_conn, 2, app.start_iter ), name="app-{}".format(app.appid), daemon=True)
            app.process = p
            app.process.start()
        except Exception as e:
            print("exec app: {}, ".format(app.appid), e)

    def exec_ready_apps(self):
        for app in self.app_ready:
            self.exec_single_app(app)
        
    def pause_world(self, to_pause_apps):
        if self.mode==1:
            return
        try:
            self.in_pausing = True
            for app in to_pause_apps:
                # get the parent_conn of app
                parent_conn = self.channels[app.appid][0]
                # send a pause signal to the child_conn
                parent_conn.send(conn_message("Pause"))
            # set the number of pause signal in this round 
            self.paused_signal = len(to_pause_apps)
        except Exception as e:
            print("Error in pause_world for app {}".format(app.appid), e)

    def reallocate_workers(self):
        if self.mode==1:
            return [], []
        # no app should be paused at this stage, app_paused should be thread safe (TODO?)
        assert(len(self.app_paused)==0)
        apps = self.app_running + self.app_ready
        print("old_apps")
        for app in apps:
            print("app: {}, app_workers: {}".format(app.appid, app.workers))
        new_apps, intact_apps = self.tp.worker_allocator(apps, debug=False)
        print("new_apps")
        for app in new_apps:
            print("app: {}, app_workers: {}".format(app.appid, app.workers))
        to_pause_apps = []
        to_launch_apps = []
        for i, app in enumerate(new_apps):
            if app in self.app_running:
                to_pause_apps.append(app)
            elif app in self.app_ready:
                to_launch_apps.append(app)
        self.print_apps( to_pause_apps)
        self.print_apps( to_launch_apps)
        return to_pause_apps, to_launch_apps

    def create_listener(self):
        t = threading.Thread(target=self.event_listener, args=( ), name="listener", daemon=False)
        t.start()

    def event_listener(self):
        event_type = ("Paused", "Finished", "Resume", "Arrival")
        # use a time window to wait for finished applications
        finish_window_start = time.time()
        # self.sub_conn.send(["Start"])
        while self.stop_flag != self.finished_count:
            # check the conn channel to submitter
            if self.sub_conn.poll():
                submit_event = self.sub_conn.recv()
                # if new app buffer arrivals
                if submit_event[1] == "Arrival":
                    apps = submit_event[2]
                    print("{}: master: {}, receives submition.".format(datetime.datetime.now(), [app.appid for app in submit_event[-1]]))
                    self.arrival_handler(apps)
                # if no app will come, save the total number of apps submitted
                elif submit_event[1] == "End":
                    self.stop_flag = submit_event[2]
                    print("stop at {}".format(self.stop_flag))

            # check communication channel of each app
            for app in self.app_running + self.app_paused:
                if app.appid in self.channels:
                    parent_conn, _ = self.channels[app.appid]
                    if parent_conn.poll():
                        app_event = parent_conn.recv()
                        # if an app is paused, record the app's next start iteration
                        if app_event[1]=="Paused":
                            print("{}: master: {}, receives pause echo: {}".format(datetime.datetime.now(), app.appid, app_event))
                            app.start_iter = app_event[2]["iter"] if \
                                                (app.start_iter==0 or app_event[2]["iter"]<app.start_iter) \
                                                else app.start_iter
                            self.pause_handler(app)
                        # if an app is finished
                        elif app_event[1]=="Finished":
                            print("{}: master: {}, receives finish echo: {}".format(datetime.datetime.now(), app.appid, app_event))
                            app.finish_time = parser.parse(app_event[0])
                            # trigger finish_handler in 5 seconds after the current app finished
                            if len(self.finish_buffer)==0:
                                finish_window_start = time.time()
                                self.finish_buffer.append(app)
                            elif time.time() - finish_window_start < 5:
                                self.finish_buffer.append(app)
                            else:
                                self.finish_handler()
                            print("finish_buffer :", self.finish_buffer)
                        elif app_event[1]=="HeartBeat":
                            print("{}: master: {}, receives heartbeat: {}".format(datetime.datetime.now(), app.appid, app_event))
            if  len(self.finish_buffer)>0 and time.time() - finish_window_start > 5:
                self.finish_handler()

    def arrival_handler(self, apps):
        # all new app needs to initialize conn channels
        for app in apps:
            if not app.appid in self.channels:
                parent_conn, child_conn = Pipe()
                self.channels[app.appid] = [parent_conn, child_conn]
        # if no apps submitted before, reallocate workers and run apps in app buffer
        if len(self.app_ready+self.app_paused+self.app_running) == 0:
            self.app_ready.extend(apps)
            _, to_launch_apps = self.reallocate_workers()
            for app in self.app_ready:
                if app in to_launch_apps:
                    self.app_ready.remove(app)
                    self.app_ready.append(app)
            self.print_queues()
            self.resume_handler()
        # if there exists apps in running queue:
        # 1. recalculate worker number
        # 2. pause apps that need to reallocate worker number
        # 3. append app buffer to ready queue, waiting for resume
        else:
            try:
                self.app_ready.extend(apps)
                to_pause_apps, to_launch_apps = self.reallocate_workers()
                self.pause_world(to_pause_apps)
                # update ready queue
                for app in self.app_ready:
                    if app in to_launch_apps:
                        self.app_ready.remove(app)
                        self.app_ready.append(app)
                self.print_queues()
                if len(to_pause_apps)==0:
                    self.resume_handler()
            except Exception as e:
                print("arrival_handler", e)
        
    def pause_handler(self, app):
        def wait_process_stop(ps):
            while len(ps)>0:
                for i, p in enumerate(ps):
                    # if (not psutil.pid_exists(p.pid)) or (not p.is_alive()):
                    if not p.is_alive():
                        ps.pop(i)
        # try:
        self.app_running.remove(app)
        app.checkpoint = True
        app.port += 1
        self.app_paused.append(app)
        self.paused_counter += 1
        self.print_queues()
        # if all apps are paused, resume the world
        if (self.paused_signal == self.paused_counter): 
            start = time.time()
            # with self.lock:
            wait_process_stop([app.process for app in self.app_paused])
            print("pause waiting time: {}".format(time.time() - start))
            self.resume_handler()
            self.paused_counter = 0
            self.in_pausing = False
        # except Exception as e:
        #     print("pause_handler", e)

    def finish_handler(self):
        try:
            for app in self.finish_buffer:
                # join app process once it reaches to the end
                app.process.join()
                # TODO: check if receives paused before finished signal for any app 
                if app in self.app_running:
                    self.app_running.remove(app)
                if app in self.app_paused:
                    self.app_paused.remove(app)
                self.app_finished.append(app)
                # close conn channels for finished app
                # child_conn, parent_conn = self.channels[app.appid]
                # child_conn.close()
                # parent_conn.close()
                self.finished_count += 1
            self.finish_buffer = []
            self.print_queues()
            if self.mode == 3:
                to_pause_apps, _ = self.reallocate_workers()
                self.pause_world(to_pause_apps)
        except Exception as e:
            print("finish_handler", e)

    def resume_handler(self):
        try:
            while len(self.app_paused)>0:
                app = self.app_paused.pop()
                self.app_ready.append(app)
            self.exec_ready_apps()
            self.app_running.extend(self.app_ready)
            self.app_ready = []
            self.paused_counter=0
            self.print_queues()
        except Exception as e:
            print("resume_handler", e)

    def print_apps(self, apps):
        print([app.appid for app in apps])

    def print_queues(self):
        print("ready: {}, running: {}, paused: {}, finished: {}".format([app.appid for app in self.app_ready], [app.appid for app in self.app_running], [app.appid for app in self.app_paused], [app.appid for app in self.app_finished]))

