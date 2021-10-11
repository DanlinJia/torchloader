from builtins import enumerate
import pandas as pd
import numpy as np
import os
import subprocess
import time, datetime
import pickle
import collections
import threading
import psutil
from multiprocessing import Process, Pipe, Lock
from dateutil import parser
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
        # self.app_infos = collections.defaultdict(list)
        self.app_infos = []
        # communication pipe for submitting apps
        self.conn = conn
        # cumulate app submissions within buffer_size
        self.buffer_size = time_window
        self.buffer = []
        self.submisson_num = 0

    def read_app_submissions(self):
        # generate a set of applications by reading data from a csv file
        df = pd.read_csv(self.submit_path, header=0, skipinitialspace=True)
        df.loc[:, "cuda_device"] = df["cuda_device"].apply(lambda x: [int(d) for d in x.split(" ")] if (type(x)==str) else [int(x)])
        if len(df)==0:
            print("Warning: No application submitted")
        for i in range(len(df)):
            app = application(i, *df.loc[i, :])
            # self.app_infos[app.arrival_time].append(app)
            self.app_infos.append(app)
        self.submisson_num = len(self.app_infos)

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
        while len(self.app_infos) > 0:
            app = self.app_infos[0]
            if _inwindow(app.arrival_time, cur_time, cur_time+self.buffer_size):
                app = self.app_infos.pop(0)
                app.appid = int((app.arrival_time - cur_time)*1e9 + time.time_ns())
                expriment_name = "{}{}_{}_{}_{}_app{}".format(
                                app.arch, app.depth, app.batch, app.workers, len(app.cuda_device) ,app.appid)
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

class dl_scheduler():
    def __init__(self, sub_conn):
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

    def main_loop_fn(self):
        # get submitted app info
        # self.read_app_submissions()
        # scheduler_conn, listener_conn = Pipe()
        # self.create_listener(listener_conn)
        # for arrival_time in self.app_infos.keys():
        #     time.sleep(arrival_time)
        #     # self.app_health_tracker()
        #     self.reallocate_workers()
        #     self.pause_world()
        #     self.resume_world(scheduler_conn)
        #     self.app_ready.extend(self.app_infos[arrival_time])
        #     self.exec_ready_apps()
        # # while len(self.app_ready)!=0:
        # #     self.app_health_tracker()
        # for app in self.app_finished:
        #     app.process.join()
        #     app.listener.join()
        # scheduler_conn, listener_conn = Pipe()
        self.create_listener()
        # t = threading.Thread(target=self.resume_world, args=(scheduler_conn, ), name="scheduler", daemon=False)
        # t.start()
        # p.join()

    def exec_ready_apps(self):
        for app in self.app_ready:
            self.exec_single_app(app)
        
    def exec_single_app(self, app: application):
        try:
            _, child_conn = self.channels[app.appid]
            p = Process(target=app_main, args=(child_conn, app,), name="app-{}".format(app.appid), daemon=False)
            # p = Process(target=worker_test_fn, args=(app.appid, child_conn, 2, app.start_iter ), name="app-{}".format(app.appid), daemon=True)
            app.process = p
            app.process.start()
        except Exception as e:
            print("exec app: {}, ".format(app.appid), e)


    def pause_world(self):
        try:
            for app in self.app_running:
                # get the parent_conn of app
                parent_conn = self.channels[app.appid][0]
                # send a pause signal to the child_conn
                parent_conn.send(conn_message("Pause"))
            # set the number of pause signal in this round 
            self.paused_signal = len(self.app_running)
        except Exception as e:
            print("Error in pause_world for app {}".format(app.appid), e)

    def reallocate_workers(self):
        pass 

    def create_listener(self):
        t = threading.Thread(target=self.event_listener, args=( ), name="listener", daemon=False)
        t.start()

    def event_listener(self):
        event_type = ("Paused", "Finished", "Resume", "Arrival")
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
                            app.start_iter = app_event[2]["iter"] + 1 if \
                                                (app.start_iter==0 or app_event[2]["iter"]<app.start_iter) \
                                                else app.start_iter
                            self.pause_handler(app)
                        # if an app is finished
                        elif app_event[1]=="Finished":
                            print("{}: master: {}, receives finish echo: {}".format(datetime.datetime.now(), app.appid, app_event))
                            app.finish_time = parser.parse(app_event[0])
                            self.finish_handler(app)
                        elif app_event[1]=="HeartBeat":
                            print("{}: master: {}, receives heartbeat: {}".format(datetime.datetime.now(), app.appid, app_event))

    def arrival_handler(self, apps):
        # all new app needs to initialize conn channels
        for app in apps:
            if not app.appid in self.channels:
                parent_conn, child_conn = Pipe()
                self.channels[app.appid] = [parent_conn, child_conn]
        # if no apps submitted before, directly run apps in app buffer
        if len(self.app_ready+self.app_paused+self.app_running) == 0:
            self.app_ready.extend(apps)
            self.print_queues()
            self.resume_handler()
        # if there exists apps in running queue:
        # 1. recalculate worker number
        # 2. pause apps that need to reallocate worker number
        # 3. append app buffer to ready queue, waiting for resume
        else:
            try:
                self.reallocate_workers()
                self.pause_world()
                # only add app into ready queue, waiting for execution
                self.app_ready.extend(apps)
                self.print_queues()
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
            wait_process_stop([app.process for app in self.app_paused])
            print("pause waiting time: {}".format(time.time() - start))
            self.resume_handler()
            self.paused_counter = 0
        # except Exception as e:
        #     print("pause_handler", e)

    def finish_handler(self, app):
        try:
            # join app process once it reaches to the end
            app.process.join()
            # TODO: check if receives paused before finished signal for any app 
            if app in self.app_running:
                self.app_running.remove(app)
            if app in self.app_paused:
                self.app_paused.remove(app)
            self.app_finished.append(app)
            # close conn channels for finished app
            child_conn, parent_conn = self.channels[app.appid]
            child_conn.close()
            parent_conn.close()
            self.print_queues()
            self.finished_count += 1
            self.reallocate_workers()
            self.pause_world()
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

submit_path = "/tmp/home/danlinjia/torchloader/torchloader/dl_scheduler-test.conf.csv"
ws_conn, sb_conn = Pipe()
ws = dl_scheduler(ws_conn)
sb = submitter(submit_path, sb_conn, time_window=5)
sb.main_fn()
ws.main_loop_fn()
# ws.main_loop_fn()
# ws.exec_single_app(ws.app_infos[0][0])