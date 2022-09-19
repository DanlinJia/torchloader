import os
import re
import ast
import pandas as pd
import numpy as np
import argparse
from datetime import datetime

from pandas.core.frame import DataFrame
from pandas.io.pytables import dropna_doc

parser = argparse.ArgumentParser(description="a trace parser to analyze events in dl_schduler.")
parser.add_argument("--input", "-i", type=str, help="the even trace path.")
parser.add_argument("--output", "-o", type=str, help="the output folder (csv).")

args = parser.parse_args()


########################## parse event trace  ##########################
with open(args.input, "r") as event_trace:
    lines = event_trace.read().splitlines()

event_type = ("receives submition", "receives heartbeat", 
                "receives pause echo", "receives finish echo")

app_events = pd.DataFrame(columns=["appid", "time", "event", "message"])
app_workers = pd.DataFrame(columns=["appid", "time", "pre_workers", "new_workers", "reason"])

def clean_split(s, regex=","):
    sl = re.split(regex, s)
    return [s for s in sl if (s!='' and s!='\n')]

def extract_info(s: str):
    word1, event_msg = clean_split(s, regex='\[|\]')
    words = clean_split(word1, ' |,')
    t1 = words[0]
    t2 = words[1]
    app = words[3]
    t = "{} {}".format(t1, t2[:-1])
    if "heartbeat" in s:
        event_msg = re.split("\{", event_msg)
        event_msg = ast.literal_eval("{" + event_msg[-1])
    return t, app, event_msg

track_old, track_new = False, False
for l in lines:
    event = ''
    if ">>>" in l:
        l = l[4:]
    if "receives submission" in l:
        t, apps, _ = clean_split(l, regex='\[|\]')
        t1, t2, _ = clean_split(t, ' ')
        t = "{} {}".format(t1, t2[:-1])
        event = "arrival"
        for app in clean_split(apps, regex=' |,'):
            app_events.loc[len(app_events), :] = [app, t, event, None]
        continue
    elif "receives heartbeat" in l:
        event = "heartbeat"
    elif "receives pause echo" in l:
        event = "pause"
    elif  "receives finish echo" in l:
        event = "finish"
    elif "old_apps" in l:
        track_old = True
        continue
    elif "new_apps" in l:
        track_old = False
        track_new = True
        continue

    if "iteration" in l:
        continue
    if track_old:
        app, worker = clean_split(l, ",")
        reason = "{} {}".format(app_events.appid.iloc[len(app_events)-1], app_events.event.iloc[len(app_events)-1])
        app_workers.loc[len(app_workers), :] = [clean_split(app, ":")[-1], t ,clean_split(worker, ":")[-1], 0, reason]
    if track_new:
        if "[" in l:
            track_new = False
            continue
        app, worker = clean_split(l, ",")
        app_workers.loc[(app_workers.appid==clean_split(app, ":")[-1])&(app_workers.time==t), "new_workers"] = clean_split(worker, ":")[-1]

    if event!='':
        t, app, event_msg = extract_info(l)
        app_events.loc[len(app_events), :] = [app, t, event, event_msg]

# post-process
app_workers.loc[app_workers.new_workers==0, "new_workers"] = app_workers.loc[app_workers.new_workers==0, "pre_workers"]
app_events.loc[:, "time"] = app_events.loc[:, "time"].apply(lambda t: datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f'))

################################### generate stats #########################
def calculate_app_latency(appid: str, app_events: pd.DataFrame):
    # return the latency of an application in a df
    # appid = "1634654752345101056"
    latency_df = pd.DataFrame(columns=["appid", "latency", "start", "finish"])
    app_df = app_events[app_events.appid==appid]
    status_df = app_df[~(app_df.event=="heartbeat")]
    arrival_time = status_df.loc[status_df.event=="arrival", "time"].iloc[0] 
    if (status_df.event=="finish").any()==True:
        finish_time = status_df.loc[status_df.event=="finish", "time"].iloc[0]
    else:
        finish_time = arrival_time
    latency_df.loc[len(latency_df), :] = [appid, finish_time - arrival_time, arrival_time, finish_time]
    return latency_df

def calculate_pause_cost(appid: str, app_events: pd.DataFrame):
    # return a df recording all pause costs
    # appid = "1634654752345101056"
    app_df = app_events[app_events.appid==appid]
    app_df = app_df.reset_index(drop=True)
    pause_df = app_df[app_df.event=="pause"].copy(deep=True)
    pause_df["cost"] = [0]*len(pause_df)
    for pause_idx in pause_df.index:
        heartbeat_idx = pause_idx + 1
        while app_df.loc[app_df.index==heartbeat_idx, "event"].iloc[0] != "heartbeat":
            heartbeat_idx += 1 
        next_heartbeat = app_df.loc[app_df.index==heartbeat_idx, "time"].iloc[0]
        pause_time = app_df[app_df.index==pause_idx]["time"].iloc[0] 
        pause_df.loc[pause_df.index == pause_idx, "cost"] = next_heartbeat - pause_time
    return pause_df

def get_heartbeat_info(appid:str, app_events: pd.DataFrame):
    app_df = app_events[app_events.appid==appid]
    app_df = app_df.reset_index(drop=True)
    heartbeat_df = app_df.loc[app_df.event=="heartbeat", "message"]
    heartbeat_df = heartbeat_df.apply(pd.Series)
    heartbeat_df.epoch = heartbeat_df.epoch.astype(int)
    heartbeat_df.iter = heartbeat_df.iter.astype(int)
    heartbeat_df.worker = heartbeat_df.worker.astype(int)
    heartbeat_df["appid"] = [appid]*len(heartbeat_df)
    return heartbeat_df


lats = []
p_costs = []
heartbeat = []
for appid in app_events.appid.drop_duplicates().values:
    try:
        lats.append(calculate_app_latency(appid, app_events))
        p_costs.append(calculate_pause_cost(appid, app_events))
        heartbeat.append(get_heartbeat_info(appid, app_events))
    except Exception as e:
        print(e)
        

lat_df = pd.concat(lats)
p_cost_df = pd.concat(p_costs)
cost_df = p_cost_df[["appid", "cost"]].groupby(by=["appid"], as_index=False).sum()
heartbeat_df = pd.concat(heartbeat)

batch_size = 512
sum_tpt = 0
tpt_df = pd.DataFrame(columns=["appid", "worker", "mini_batch", "iter_time", "data_time" ,"tpt"])
for appid in heartbeat_df.appid.drop_duplicates().values:
    # filter out iteration time larger than 1 second, as such iterations involve initialziation costs
    app_df = heartbeat_df.loc[(heartbeat_df.appid == appid) & (heartbeat_df.iter_time<1), :]
    workers = app_df.worker.drop_duplicates().values
    mini_batch = int(batch_size/len(workers))
    for worker in workers:
        worker_df = app_df[app_df.worker==worker]
        ave_iter_time = worker_df.iter_time.mean()
        ave_data_time = worker_df.dataloading_time.mean()
        tpt_df.loc[len(tpt_df), :] = [appid, worker, mini_batch , ave_iter_time, ave_data_time , mini_batch/ave_iter_time]
        # print("worker: {}, ave_iter_time: {}".format(worker, ave_iter_time))
aggreagted_tpt_df = tpt_df.groupby(by=["appid"]).sum()
lat_df = lat_df.join(aggreagted_tpt_df["tpt"], on="appid")

tpt_df.to_csv(path_or_buf=os.path.join( args.output, "tpt_df.csv"), index=False)
lat_df.to_csv(path_or_buf=os.path.join( args.output, "lat_df.csv"), index=False)
p_cost_df.to_csv(path_or_buf=os.path.join(args.output, "p_cost_df.csv"), index=False)
app_workers.to_csv(path_or_buf=os.path.join(args.output, "workers.csv"), index=False)
heartbeat_df.to_csv(path_or_buf=os.path.join(args.output, "heartbeat.csv"), index=False)

print("ave latency: {}, sum tpt: {}, makespan: {}".format(
    lat_df.latency.apply(lambda x:x.total_seconds()).mean(), 
    lat_df.tpt.sum(),
    (lat_df.finish.max() - lat_df.start.min()).total_seconds()
    )
)
