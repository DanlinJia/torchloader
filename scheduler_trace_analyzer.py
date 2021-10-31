import os
import re
import pandas as pd
import argparse
from datetime import datetime

from pandas.core.frame import DataFrame

parser = argparse.ArgumentParser(description="a trace parser to analyze events in dl_schduler.")
parser.add_argument("--input", "-i", type=str, help="the even trace path.")
parser.add_argument("--output", "-o", type=str, help="the output file (csv).")

args = parser.parse_args()


########################## parse event trace  ##########################
with open(args.input, "r") as event_trace:
    lines = event_trace.read().splitlines()

event_type = ("receives submition", "receives heartbeat", 
                "receives pause echo", "receives finish echo")

app_events = pd.DataFrame(columns=["appid", "time", "event"])
app_workers = pd.DataFrame(columns=["appid", "time", "pre_workers", "new_workers", "reason"])

def clean_split(s: str, regex=","):
    sl = re.split(regex, s)
    return [s for s in sl if (s!='' and s!='\n')]

def extract_info(s: str):
    word1, event_msg = clean_split(s, regex='\[|\]')
    words = clean_split(word1, ' |,')
    t1 = words[0]
    t2 = words[1]
    app = words[3]
    t = "{} {}".format(t1, t2[:-1])
    return t, app

track_old, track_new = False, False
for l in lines:
    event = ''
    if ">>>" in l:
        l = l[4:]
    if "receives submition" in l:
        t, apps, _ = clean_split(l, regex='\[|\]')
        t1, t2, _ = clean_split(t, ' ')
        t = "{} {}".format(t1, t2[:-1])
        event = "arrival"
        for app in clean_split(apps, regex=' |,'):
            app_events.loc[len(app_events), :] = [app, t, event]
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
        t, app = extract_info(l)
        app_events.loc[len(app_events), :] = [app, t, event]

# post-process
app_workers.loc[app_workers.new_workers==0, "new_workers"] = app_workers.loc[app_workers.new_workers==0, "pre_workers"]
app_events.loc[:, "time"] = app_events.loc[:, "time"].apply(lambda t: datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f'))

################################### generate stats #########################
def calculate_app_latency(appid: str, app_events: pd.DataFrame):
    # return the latency of an application in a df
    # appid = "1634654752345101056"
    latency_df = pd.DataFrame(columns=["appid", "latency"])
    app_df = app_events[app_events.appid==appid]
    status_df = app_df[~(app_df.event=="heartbeat")]
    arrival_time = status_df.loc[status_df.event=="arrival", "time"].item() 
    finish_time = status_df.loc[status_df.event=="finish", "time"].item()
    latency_df.loc[len(latency_df), :] = [appid, finish_time - arrival_time]
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
        while app_df.loc[app_df.index==heartbeat_idx, "event"].item() != "heartbeat":
            heartbeat_idx += 1 
        next_heartbeat = app_df.loc[app_df.index==heartbeat_idx, "time"].item()
        pause_time = app_df[app_df.index==pause_idx]["time"].item() 
        pause_df.loc[pause_df.index == pause_idx, "cost"] = next_heartbeat - pause_time
    return pause_df

lats = []
p_costs = []
for appid in app_events.appid.drop_duplicates().values:
    lats.append(calculate_app_latency(appid, app_events))
    p_costs.append(calculate_pause_cost(appid, app_events))

lat_df = pd.concat(lats)
lat_df["tpt"] = 2*500*128/lat_df.latency.apply(lambda x:x.total_seconds())
print("ave latency: {}, sum tpt: {}".format(lat_df.latency.apply(lambda x:x.total_seconds()).mean(), lat_df.tpt.sum() ))
p_cost_df = pd.concat(p_costs)
cost_df = p_cost_df[["appid", "cost"]].groupby(by=["appid"], as_index=False).sum()
lat_df.join(cost_df.set_index("appid"), on="appid").to_csv(path_or_buf=os.path.join( args.output, "lat_df.csv") )
p_cost_df.to_csv(path_or_buf=os.path.join(args.output, "p_cost_df.csv"))
app_workers.to_csv(path_or_buf=os.path.join(args.output, "workers.csv"))
