from dl_app import application, app_main
from multiprocessing import Process, Pipe, Lock
from dl_scheduler import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode", "-m")

opts = parser.parse_args()

submit_path = "/tmp/home/danlinjia/torchloader/torchloader/dl_scheduler-8_corun-test.conf.csv"
ws_conn, sb_conn = Pipe()

sb = submitter(submit_path, sb_conn, time_window=10)
tp = dl_tpt_pridictor(cpu_cores=18, gpu_devices=4, model_info_path="models.csv", \
                        cpu_model_path="/tmp/home/danlinjia/torchloader/torchloader/cpu_model", \
                        gpu_model_path='/tmp/home/danlinjia/torchloader/torchloader/gpu_model')
# mode:
# 1. no worker reallocation
# 2. reallocate workers for arrival signal
# 3. reallocate workers for both arrival and finish signals
print(opts.mode)
ws = dl_scheduler(ws_conn, tp, 1)

# sb.read_app_submissions()
# tp.worker_allocator(sb.app_infos, True)

sb.main_fn()
ws.main_loop_fn()