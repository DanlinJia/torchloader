import socket
import datetime
import xml.etree.ElementTree as et
from contextlib import closing
from multiprocessing import Process, Pipe
import threading
import json

#pdb
def pdb_bp():
    import pdb; pdb.set_trace()

# network utils
def is_socket_open(host, port):
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        if sock.connect_ex((host, port)) == 0:
            print("Port {} is open on {}".format(port, host))
            return True
        else:
            print("Port {} is not open on {}".format(port, host))
            return False

def get_current_host():
    return socket.gethostname()

mesg_types = ('Register', 'Registered')

def conn_message(mesg_type, mesg_data=None):
    return {'time': str(datetime.datetime.now()), 'type': mesg_type, 'msg': mesg_data}

# xml config utils
class dl_config():
    def __init__(self, config_path):
        self.config_path = config_path
        self.master_ip = None
        self.master_port = None
        self.worker_ip = None
        self.worker_rpc_port = None
        self.master_auth = None
        self.parse_config()

    def parse_config(self):
        tree = et.parse(self.config_path)
        root = tree.getroot()
        self.master_ip = root.find('master_config').find('master_ip').text
        self.master_port = root.find('master_config').find('master_port').text
        self.master_auth = root.find('master_config').find('authkey').text

        self.worker_ip = root.find('worker_config').find('worker_ip').text
        self.worker_rpc_port = root.find('worker_config').find('worker_rpc_port').text
        self.worker_socket_port = root.find('worker_config').find('worker_socket_port').text
        

# rpc utils
def ping():
    return 0

def wait_processes_to_stop(ps):
    if not isinstance(ps, list):
        ps = [ps]
    while len(ps)>0:
        for i, p in enumerate(ps):
            if not p.is_alive():
                ps.pop(i)
# states
SUCCESS = 1
FAILED = 0