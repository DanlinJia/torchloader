import zmq

import xmlrpc.client
from util import *
from copy import copy
from dl_scheduler import submitter

class dl_master():
    def __init__(self, config_path, ms_conn=None):
        self.config_path = config_path
        self.ms_conn = ms_conn
        self.master_ip = None
        self.master_port = None
        self.worker_socket_port = None
        # create a communication socket between the master and workers
        self.listener = None
        # the worker's info, each element is a {'worker_ip':x, 'worker_rpc_port': x} pair
        self.workers = []
        # the rpc clients, each element is a {worker_info: rpc_client} pair
        self.rpc_clients = {}
        self.worker_messager = {}
        self.app_event_count = {}
        self._init_master()

    def _init_master(self):
        self.read_config()
        self.create_master_listener()

    def read_config(self):
        config = dl_config(self.config_path)
        self.master_ip = config.master_ip
        self.master_port = config.master_port
        self.authkey = config.master_auth
        self.worker_socket_port = config.worker_socket_port

    def create_master_listener(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        address = "tcp://*:{}".format(self.master_port)
        socket.bind(address)
        print("listener is created on {}".format(address))
        self.listener = socket

    def listener_thread(self):
        while True:
            try:
                # Wait for next request from client
                msg = self.listener.recv_json()
                # print(f"Received request: {msg}")
                if msg['type'] == 'Register':
                    worker_info = msg['msg']
                    self.register_worker(worker_info)
                elif msg['type'] == 'Paused':
                    self.listener.send_json(conn_message("Paused_ack"))
                    app_id = msg['msg']['app_id']
                    worker_id = msg['msg']['worker_id']
                    if self.app_event_count[app_id] + 1 == len(self.workers):
                        self.app_event_count[app_id] = 0
                        if self.ms_conn:
                            self.ms_conn.send(msg)
                    else:
                        self.app_event_count[app_id] += 1

                elif msg['type'] == 'Finished':
                    self.listener.send_json(conn_message("Finished_ack"))
                    app_id = msg['msg']['app_id']
                    worker_id = msg['msg']['worker_id']
                    if self.app_event_count[app_id] + 1 == len(self.workers):
                        self.app_event_count[app_id] = 0
                        if self.ms_conn:
                            self.ms_conn.send(msg)
                    else:
                        self.app_event_count[app_id] += 1

                elif msg['type'] == 'HeartBeat':
                    self.listener.send_json(conn_message("hb_ack"))
                    if self.ms_conn:
                        self.ms_conn.send(msg)  

            except Exception as e:
                print(msg, e) 
                raise()

    def launch_listener(self):
        t = threading.Thread(target=self.listener_thread, args=( ), name="listener", daemon=False)
        t.start()

    def register_worker(self, worker_info):
        if worker_info not in self.workers:
            self.workers.append(worker_info)
            worker_id = len(self.workers) - 1
            self.listener.send_json(conn_message('Registered', {'worker_id': worker_id}))
            self.create_worker_messager(worker_info)
            # self.create_rpc_client(worker_info)
    
    def worker_id2ip(self, worker_id):
        return self.workers[worker_id]["worker_ip"]

    def create_worker_messager(self, worker_info):
        context = zmq.Context()
        #  Socket to talk to server
        socket = context.socket(zmq.REQ)
        ip, port = worker_info['worker_ip'], self.worker_socket_port 
        address = "tcp://{}:{}".format(ip, port)
        socket.connect(address)
        print("Create worker messager on {}".format(address))
        self.worker_messager[(ip, port)] = socket

    def create_rpc_client(self, worker_info):
        ip, port = worker_info['worker_ip'], worker_info['worker_rpc_port']
        url = "http://{}:{}".format(ip, port)
        s = xmlrpc.client.ServerProxy(url)
        self.rpc_clients[(ip, port )] = s
        # test if rpc_server is created on workers
        if s.ping()==0:
            print(f"rpc_client created for {worker_info}")
            print(f"the following function calls are available:\n {s.system.listMethods()}")

    def get_app_for_worker(self, worker_id, app, base_rank):
        """
        determine app's arguments for worker_id
        """
        app.attach_node(worker_id)
        node_name = self.worker_id2ip(worker_id)
        app.cuda_device = app.node_info[node_name]
        app.base_rank = base_rank
        base_rank += len(app.cuda_device)
        app.batch = int(app.batch * (len(app.cuda_device)/app.world_size))
        app.workers = int(app.workers * (len(app.cuda_device)/app.world_size))
        return app, base_rank

    def send_app(self, app):
        # worker_id represents the index of node in the cluster
        base_rank = 0
        for worker_id, messager in enumerate(self.worker_messager.values()):
            if self.worker_id2ip(worker_id) not in app.node_info:
                continue
            # save app infomation
            self.app_event_count[app.appid] = 0
            # an app will be coming
            messager.send_json(conn_message("app"))
            # you know it, right?
            msg = messager.recv_json()
            if(msg['type']=='app_ack'):
                worker_app, base_rank = self.get_app_for_worker(worker_id, copy(app), base_rank)
                print(f"send app to worker {worker_id}, {worker_app.print_info()}")
                # send the app object
                messager.send_pyobj(worker_app)
                # good! you received it
                msg = messager.recv_json()
                if msg['type']!="app_data_ack":
                    raise Exception("app obj transfer failed!")
            else:
                raise Exception("app header transfer failed!")

    def launch_app(self, app):
        for worker_id, messager in enumerate(self.worker_messager.values()):
            if self.worker_id2ip(worker_id) not in app.node_info:
                continue
            # I need to launch an app
            messager.send_json(conn_message("Launch"))
            # you know the app will be coming
            msg = messager.recv_json()
            if msg['type']!='Launch_ack':
                raise Exception("Launch app failed!")
        # send app to master
        self.send_app(app)
    
    def pause_app(self, app):
        app_id = app.appid
        for messager in self.worker_messager.values():
            if self.worker_id2ip(worker_id) not in app.node_info:
                continue
            # pause app with app_id
            messager.send_json(conn_message("Pause", {"app_id":app_id}))
            # receive pause ack
            msg = messager.recv_json()
            if msg['type']!='Pause_ack':
                raise Exception("Pause app instruction failed!")


if __name__=="__main__":
    ws_conn, sb_conn = Pipe()
    master = dl_master("dl_cluster_config.xml")
    master.launch_listener()

    submit_path = "dl_scheduler-test.conf.csv"

    sb = submitter(submit_path, sb_conn, time_window=10)
    sb.read_app_submissions()
    apps = sb.apps

    app = apps[0]
