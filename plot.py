from os import path
from pandas.io.parsers import read_csv
from tpt_predict import read_trace
import matplotlib.pyplot as plt 
import numpy as np

path="/tmp/home/danlinjia/pytorch_test/trace/test-contention/statistics.csv"
single_df = read_trace(path)
single_df = read_csv(path)
device_0_df = single_df[(single_df.deviceID=="device_0")]

def plot_tpt(input_df, batch, device, model_set):
    fig, axs = plt.subplots()
    input_df = input_df[(input_df.batch_size==batch)&(input_df.num_device==device)]
    if len(input_df)==0:
        raise Exception("no df for batch {}, device{}".format(batch, device))
    models = input_df.model.drop_duplicates().values
    for model in models:
        model_df = input_df[input_df.model==model]
        model_df = model_df.sort_values(by=["num_worker"])
        x = model_df.num_worker
        y = model_df.throughput
        axs.plot(x, y, "o-", label=model)

    plt.ylabel("throughput (fig/s)")
    plt.xlabel("num_dataloader")
    plt.title("{}_batch_{}_device{}".format(model_set, batch, device))
    step = device
    plt.xticks(np.arange(input_df.num_worker.min(), input_df.num_worker.max()+step, step=step))
    plt.legend()
    figure = plt.gcf()
    figure.set_size_inches(10, 6)
    plt.savefig("figs/{}_batch_{}_device{}_plot.png".format(model_set, batch, device), dpi=100)
    plt.clf()

# for model_set in ["smresnet", "vgg", "small", "cifar10"]:
for model_set in ["mdresnet"]:
    model_df = device_0_df[ device_0_df.model.str.startswith(model_set)]
    model_df["throughput"] = model_df.batch_size/model_df.ave_iteration
    if len(model_df)!=0:
        # plot_tpt(model_df, 256, 1, model_set)
        plot_tpt(model_df, 128, 1, model_set)
    # plot_tpt(model_df, 64, 4, model_set)
    # plot_tpt(model_df, 128, 4, model_set)
    # plot_tpt(model_df, 256, 4, model_set)

    # plot_tpt(model_df, 64, 3, model_set)
    # plot_tpt(model_df, 128, 3, model_set)
    # plot_tpt(model_df, 256, 3, model_set)

    # plot_tpt(model_df, 64, 2, model_set)
    # plot_tpt(model_df, 128, 2, model_set)
    # plot_tpt(model_df, 256, 2, model_set)

    # plot_tpt(model_df, 64, 1, model_set)
    # plot_tpt(model_df, 256, 1, model_set)
    # plot_tpt(model_df, 256, 1, model_set)