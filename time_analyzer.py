import os
import re
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

def parse_path(path):
    # $root_folder/$experiment_folder/$app_folder/$device_path
    app_path, device_path = os.path.split(path)
    # $root_folder/$experiment_folder/$app_folder
    experiment_path, app_folder = os.path.split(app_path)
    # $root_folder/$experiment_folder
    root_folder, experiment_folder = os.path.split(experiment_path)
    # $root_folder
    return root_folder, experiment_folder, app_folder ,device_path.split(".")[0]

def plot_single_device(device_df, fig_path):
    fig, axs = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    x = device_df.index
    for key in ["dataloading", "forward", "backward", "update"]:
        axs.plot(x, device_df[key]/device_df["iteration"], label=key)
    axs.legend()
    axs.set_ylabel("percentage of iteration time")
    axs.set_xlabel("iteration")

    axs1 = axs.twinx()
    axs1.plot(x, device_df["iteration"], linestyle='dashed', color="black")
    axs1.set_ylabel("iteration time(s)")
    plt.savefig(fig_path)
    plt.clf()

parser = argparse.ArgumentParser(description="Plot DL runtime")
parser.add_argument("--folder", default='', type=str)
args = parser.parse_args()

# $root_folder/$experiment_folder/$app_folder
root_folder = "/tmp/home/danlinjia/torchloader/torchloader/trace"
experiment_folder = "."
app_folder = "." 
walk_folder = root_folder

if len(args.folder)>0:
    walk_folder = os.path.join(walk_folder, args.folder)
    experiment_folder = args.folder
device_traces = []
for (dirpath, dirnames, filenames) in os.walk(walk_folder):
    device_traces.extend([os.path.join(dirpath, file) for file in filenames if re.search("device_..csv", file)])

statistic_df = pd.DataFrame(columns=["model", "batch_size", "num_worker", "num_device", "experiment_name" ,"deviceID" ,"ave_iteration", "ave_dataloading", "ave_datatrans" ,"ave_forward", "ave_lossupdate", "ave_backward","ave_update"])

for trace in device_traces:
    root_folder, experiment_folder, app_folder, device_id = parse_path(trace)
    device_df = pd.read_csv(trace)
    device_df = device_df.loc[2:,:].reset_index()
    names = ["iteration", "dataloading", "datatransfer", "forward","lossupdate", "backward", "update"]
    ave_list = []
    for name in names:
        ave_list.append(device_df[name].mean())
    if len(app_folder.split("_"))==5:
        statistic_df.loc[len(statistic_df),:] = np.array(app_folder.split("_")[:-1] + [experiment_folder, device_id] + ave_list)
    else:
        num_worker = int(experiment_folder.split("_")[1].split("workers")[0])
        num_device = int(experiment_folder.split("_")[2].split("device")[0])
        statistic_df.loc[len(statistic_df),:] = np.array([app_folder.split("_")[0], app_folder.split("_")[1], num_worker, num_device, experiment_folder, device_id] + ave_list)
    plot_single_device(device_df=device_df.loc[2:, names], fig_path=os.path.join(root_folder, experiment_folder, app_folder, "{}_{}_time_breakdown.png".format(app_folder, device_id)))

statistic_df = statistic_df.sort_values(by=["model","batch_size", "experiment_name" ,"deviceID"])
statistic_df["throughput"] = statistic_df["num_worker"].astype(float)/statistic_df["ave_iteration"].astype(float)
if len(args.folder)>0:
    statistic_df.to_csv(os.path.join(root_folder, experiment_folder, "statistics.csv"), index=False)
else:
    statistic_df.to_csv(os.path.join(root_folder, "statistics.csv"), index=False)

