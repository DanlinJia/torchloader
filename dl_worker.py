import os
import sys
import zmq
import time

from xmlrpc.server import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler

from util import *
from dl_app import app_main

class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)

class dl_worker():
    def __init__(self, config_path):
        self.config_path = config_path
        self.master_ip = None
        self.master_port = None
        self.worker_ip = None
        self.worker_rpc_port = None
        self.rpc_server = None
        self.messager = None
        self.app_master = None
        self._init_worker()

    def _init_worker(self):
        self.read_config()
        # self.create_rpc_server()
        self.create_worker_listener()
        self.create_master_messager()
        self.register_server()

    def read_config(self):
        config = dl_config(self.config_path)
        self.master_ip = config.master_ip
        self.master_port = config.master_port
        self.worker_ip = config.worker_ip
        self.worker_rpc_port = config.worker_rpc_port
        self.worker_socket_port = config.worker_socket_port
        if not self.worker_ip :
            self.worker_ip = get_current_host()
    
    def create_rpc_server(self):
        if not self.worker_rpc_port:
            self.worker_rpc_port = 6321
        while is_socket_open(self.worker_ip, int(self.worker_rpc_port)):
            self.worker_rpc_port = int(self.worker_rpc_port) + 100
        try:
            self.rpc_server = SimpleXMLRPCServer((self.worker_ip, int(self.worker_rpc_port)),
                                requestHandler=RequestHandler, 
                                logRequests=True, 
                                allow_none=True, 
                                bind_and_activate=True)
            print("rpc server is created on {}".format(self.worker_ip))
        except OSError as e:    
            print("PID: {}".format(os.getpid()), e)

    # called by main or other instances
    def register_rpc_server(self, funcs: list, instances: list):
        # register inrto functions
        self.rpc_server.register_introspection_functions()
        # register functions 
        for f in funcs:
            self.rpc_server.register_function(f)
        # register instances
        for ins in instances:
            self.rpc_server.register_instance(ins)
    
    # called by main or other instances
    def launch_rpc_server(self):
        # Run the server's main loop
        try:
            p = Process(target=self.rpc_server.serve_forever)
            p.start()
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received, exiting.")
            sys.exit(0) 
    
    def create_worker_listener(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        address = "tcp://*:{}".format(self.worker_socket_port)
        socket.bind(address)
        print("worker listener is created on {}".format(address))
        self.listener = socket

    def create_master_messager(self):
        context = zmq.Context()
        #  Socket to talk to server
        socket = context.socket(zmq.REQ)
        address = "tcp://{}:{}".format(self.master_ip, self.master_port)
        socket.connect(address)
        print("Create messager on {}".format(address))
        self.messager = socket

    def listener_thread(self):
        while True:
            # try:
            # Wait for next request from server 
            msg = self.listener.recv_json()
            print(f"Received request: {msg}")
            if msg['type']=='app':
                app = self.receive_app_handler()
            if msg['type']=='Launch':
                app = self.launch_app_handler()
            if msg['type']=='Pause':
                app_id = msg['msg']['app_id']
                self.pause_handler(app_id)

            # except Exception as e:
            #     print(msg, e) 

    def launch_listener(self):
        t = threading.Thread(target=self.listener_thread, args=( ), name="worker_listener", daemon=False)
        t.start()

    def register_server(self):
        self.messager.send_json(conn_message('Register', {'worker_ip':self.worker_ip, 'worker_rpc_port': self.worker_rpc_port}))
        mesg = self.messager.recv_json()
        if mesg['type'] != 'Registered':
            print("Registration failed")
        else:
            self.worker_id = mesg['msg']['worker_id']
            print("worker {} registered as {}.".format(self.worker_ip, self.worker_id))

    def receive_app_handler(self):
        # I know an app obj will come
        self.listener.send_json(conn_message("app_ack"))
        # receive the app obj
        app = self.listener.recv_pyobj()
        app.print_info()
        # the app obj is received
        self.listener.send_json(conn_message("app_data_ack"))
        return app

    def launch_app_handler(self):
        self.listener.send_json(conn_message("Launch_ack"))
        # waiting for app obj
        msg = self.listener.recv_json()
        if msg['type']=='app':
            app = self.receive_app_handler()
            self.app_master.launch_app(app)
        else:
            raise Exception("app launch failed! Did not receive app head!")

    def pause_handler(self, app_id):
        self.listener.send_json(conn_message("Pause_ack"))
        m_conn = self.app_master.app_warehouse[app_id]['conn']
        app_pid = self.app_master.app_warehouse[app_id]['process'].pid
        print("worker{} sends pause to app{} ".format(self.worker_ip, app_id))
        m_conn.send(conn_message("Pause"))

    def send_paused_signal(self, app_id):
        self.messager.send_json(conn_message("Paused", {'app_id': app_id, 'worker_id': self.worker_id}))
        msg = self.messager.recv_json()
        if msg['type']!='Paused_ack':
                raise Exception("Sending paused signal failed!")
    
    def send_finished_signal(self, app_id):
        self.messager.send_json(conn_message("Finished", {'app_id': app_id, 'worker_id': self.worker_id}))
        print("app {} sends finishsed signal to master".format(app_id))
        msg = self.messager.recv_json()
        if msg['type']!='Finished_ack':
                raise Exception("Sending finished signal failed!") 

    def send_heartbeat(self, hb_msg):
        self.messager.send_json(hb_msg)
        msg = self.messager.recv_json()
        if msg['type']!='hb_ack':
                raise Exception("Sending heartbeat failed!") 

class application_master():
    def __init__(self):
        # maintain imformation about app id, process, pipe
        self.app_warehouse = {}
        self.app_checkpoint_path = {}
        self.running_app = 0
        self.worker = None
        self.lock = threading.Lock()
    
    def create_listener(self):
        t = threading.Thread(target=self.event_listener, args=( ), name="app master listener", daemon=False)
        t.start()

    def event_listener(self):
        event_type = ("Paused", "Finished", "Resume", "Arrival")
        # use a time window to wait for finished applications
        finish_window_start = time.time()
        # self.sub_conn.send(["Start"])
        while 1:
            # check communication channel of each app
            try:
                app_ids = [app_id for app_id in self.app_warehouse.keys() if self.app_warehouse[app_id]["status"]=="R" ]
                for app_id in app_ids:
                    if self.app_warehouse[app_id]["status"] == "P" or self.app_warehouse[app_id]["status"] == "F":
                        continue
                    m_conn = self.app_warehouse[app_id]['conn']
                    if m_conn.poll():
                        self.lock.acquire()
                        app_event = m_conn.recv()
                        self.lock.release()
                        # if an app is paused, record the app's next start iteration
                        if app_event['type']=="Paused":
                            print("{}: worker{}: {}, receives pause echo: {}".format(datetime.datetime.now(), self.worker.worker_ip ,app_id, app_event))
                            paused_iter = app_event['msg']["iter"]
                            wait_processes_to_stop(self.app_warehouse[app_id]["process"])
                            self.app_checkpoint_path[app_id] = app_event['msg']['checkpoint']
                            print("all subprocess paused")
                            self.running_app -= 1
                            self.app_warehouse[app_id]["status"] = "P"
                            del self.app_warehouse[app_id]
                            self.worker.send_paused_signal(app_id)
                        #if an app is finished
                        elif app_event['type']=="Finished":
                            print("{}: worker{}: {}, receives finish echo: {}".format(datetime.datetime.now(), self.worker.worker_ip , app_id, app_event))
                            self.running_app -= 1
                            self.app_warehouse[app_id]["status"] = "F"
                            del self.app_warehouse[app_id]
                            self.worker.send_finished_signal(app_id)
                        elif app_event['type']=="HeartBeat":
                            # print(app_event)
                            self.worker.send_heartbeat(app_event)
            except Exception as e:
                print("errors!")
                raise Exception(e)

            if(self.running_app==0):
                break
        # print("event listener inished because no app running") 

    def spawn_app(self, app, conn):
        p = Process(target=app_main, args=(conn, app, ), name="app-{}".format(app.appid), daemon=False)
        p.start()
        return p
    
    def launch_app(self, app):
        m_conn, app_conn = Pipe()
        if app.checkpoint and (app.checkpoint_path==""):
            app.checkpoint_path = self.app_checkpoint_path[app.appid]
        app_proc = self.spawn_app(app, app_conn)
        self.lock.acquire()
        self.app_warehouse[app.appid] = {"conn": m_conn, "process": app_proc, "status": "R"}
        self.lock.release()
        self.running_app += 1
        if(self.running_app==1):
            self.create_listener()

    def register_worker(self, worker):
        self.worker = worker

if __name__=="__main__":
    worker = dl_worker("dl_cluster_config.xml")
    app_master = application_master()
    app_master.register_worker(worker)
    
    # func_list = [ping]
    # instances_list = [sb, tp, ws]
    # instances_list = [app_master]

    # worker.register_rpc_server(func_list, instances_list)
    # worker.launch_rpc_server()
    worker.launch_listener()
    worker.app_master = app_master
