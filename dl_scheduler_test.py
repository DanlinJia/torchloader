from dl_app import application, app_main
from multiprocessing import Process, Pipe, Lock
from dl_scheduler import *
from dl_master import *
from dl_worker import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode", "-m")

opts = parser.parse_args()

# submit_path = "dl_submit_rpc.conf.csv"
submit_path = "dl_scheduler-test.conf.csv"

# create communication pipe between scheduler(ws_conn) and submitter(sb_conn)
ws_conn, sb_conn = Pipe()
# create communication pipe between scheduler(ws_conn1) and master(ms_conn)
ws_conn_1, ms_conn = Pipe()

sb = submitter(submit_path, sb_conn, time_window=10)
tp = dl_tpt_pridictor(cpu_cores=48, gpu_devices=4, model_info_path="models.csv", \
                        cpu_model_path="cpu_model", \
                        gpu_model_path='gpu_model')
master = dl_master("dl_cluster_config.xml", ms_conn)

# mode:
# 1. no worker reallocation
# 2. reallocate workers for arrival signal
# 3. reallocate workers for both arrival and finish signals
print(opts.mode)
ws = dl_scheduler(ws_conn, tp, (master, ws_conn_1), 3)
ws.master.launch_listener()

# sb.read_app_submissions()
# tp.worker_allocator(sb.app_infos, True)

# worker = dl_worker("dl_cluster_config.xml")
# app_master = application_master()
# app_master.register_worker(worker)
# worker.launch_listener()
# worker.app_master = app_master

def run():
    sb.main_fn()
    ws.main_loop_fn()