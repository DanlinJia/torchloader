import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import svm

sklearn_model = LinearRegression

def split_df(dataset_df, train_ratio):
    #train_df = dataset_df.sort_values(by=["model","batch_size", "num_worker", "num_device"]).head(int(len(dataset_df)*train_ratio))
    train_df = dataset_df.sample(n=int(len(dataset_df)*train_ratio))
    val_df = dataset_df[~dataset_df.index.isin(train_df.index)]
    # val_df = dataset_df[dataset_df.model=="vgg7"]
    # train_df = dataset_df[~dataset_df.index.isin(val_df.index)]
    return train_df, val_df

def preprocess_df(dataset_df, train_ratio, gpu=True):
    if gpu:
        dataset_df["gpu_x0"] = (dataset_df.batch_size/dataset_df.num_device)*dataset_df.flops
        # dataset_df["gpu_x0"] = (dataset_df.batch_size/dataset_df.num_device)
        dataset_df["gpu_x1"] = (dataset_df.num_device - 1)*dataset_df.params
        dataset_df["gpu_x2"] = dataset_df.params
        #dataset_df["y"] = dataset_df.batch_size/dataset_df.throughput
        dataset_df["y"] = dataset_df.ave_iteration
        train_df, val_df = split_df(dataset_df, train_ratio)
        train_x = train_df[["gpu_x0", "gpu_x1", "gpu_x2"]].values
        train_y = train_df["y"].values
        val_x = val_df[["gpu_x0", "gpu_x1", "gpu_x2"]].values
        val_y = val_df["y"].values
    else:
        # dataset_df["x0"] = dataset_df.num_worker * dataset_df.num_device
        # dataset_df["y"] = dataset_df.throughput
        # train_df, val_df = split_df(dataset_df, train_ratio)
        # train_x = train_df[["x0"]].values
        # train_y = train_df["y"].values
        # val_x = val_df[["x0"]].values
        # val_y = val_df["y"].values
        dataset_df["cpu_x0"] = 1/(dataset_df.num_worker)
        dataset_df["y"] = 1/dataset_df.throughput
        train_df, val_df = split_df(dataset_df, train_ratio)
        train_x = train_df[["cpu_x0"]].values
        train_y = train_df["y"].values
        val_x = val_df[["cpu_x0"]].values
        val_y = val_df["y"].values
    return train_df, val_df, train_x, train_y, val_x, val_y

def read_trace():
    df = pd.read_csv("/tmp/home/danlinjia/pytorch_test/trace/statistics.csv")
    single_df = df[df.experiment_name.str.contains("single")]
    models_df = pd.read_csv("/tmp/home/danlinjia/pytorch_test/models.csv")
    single_df = single_df.join(models_df.set_index('model'), on="model")
    #single_df["num_device"] = np.array([4]*len(single_df))
    #single_df["num_worker"] = single_df.colocate_dataloader.apply(lambda x: x.split("_")[1].split("workers")[0]).astype(int)
    #single_df.loc[:, "params"] = single_df.params/1e6
    #single_df.loc[:, "flops"] = single_df.flops/1e6
    return single_df

def worker_allocator(df, gpu_model, cpu_model):
    model_df = pd.read_csv("/tmp/home/danlinjia/pytorch_test/models.csv")
    #df["num_device"]=df.cuda_device.apply(lambda x: len(x.split()) if (type(x)==str) else 1)
    #df["model"] = df["arch"]+df["depth"].astype(str)
    df = df.join(model_df.set_index('model'), on="model")
    df["gpu_x0"] = df.batch/df.num_device*df.flops
    # df["gpu_x0"] = df.batch/df.num_device
    df["gpu_x1"]=(df.num_device-1)*df.params
    df["gpu_x2"]=df.params
    df["cpu_x0"] = 1/df.num_device
    df["gpu_y_"] = df[["gpu_x0", "gpu_x1", "gpu_x2"]].apply(lambda x: gpu_model.predict(np.array(x).reshape(1, -1))[0], axis=1)
    df["gpu_tpt"] = df.batch/df.gpu_y_
    df["cpu_tpt"] = df[["cpu_x0"]].apply(lambda x: 1/cpu_model.predict(np.array(x).reshape(1, -1))[0], axis=1)
    return df

def gpu_predictor(df, gpu_model):
    df["gpu_y_"] = df[["gpu_x0", "gpu_x1", "gpu_x2"]].apply(lambda x: gpu_knn_model.predict(np.array(x).reshape(1, -1))[0], axis=1)
    df["gpu_tpt"] = df.batch_size/df.gpu_y_
    df["distance"] = (df.gpu_tpt - df.throughput)/ df.throughput
    return df[["model",  "batch_size" , "num_worker",  "num_device","experiment_name","throughput","gpu_tpt","distance"]]

if __name__=="__main__":
    single_df = read_trace()
    single_df = single_df[single_df.deviceID=="device_0"]
    single_df = single_df[single_df.model!="resnet20"]
    # single_df["flops"] = single_df.flops.apply(lambda x: x/1e6)
    # single_df["params"] = single_df.params.apply(lambda x: x/1e6)
    # single_df = single_df[single_df.model.str.contains("vgg")]
    gpu_df = single_df.loc[single_df.ave_dataloading/single_df.ave_iteration<0.05, :]
    cpu_df = single_df.loc[~single_df.index.isin(gpu_df.index), :]
    gpu_df = gpu_df.drop_duplicates(subset=["model", "batch_size","num_device"])
    

    # GPU modeling
    g_train_df, g_val_df, train_x, train_y, val_x, val_y = preprocess_df(gpu_df, 0.6, gpu=True)
    gpu_model = sklearn_model(fit_intercept=True, normalize=True).fit(train_x, train_y)
    gpu_score = gpu_model.score(val_x, val_y)
    print("Linear Regression: {}".format(gpu_score))

    transformer = PolynomialFeatures(degree=2, include_bias=True)
    x_ = transformer.fit_transform(train_x.tolist())
    gpu_PLR_model = sklearn_model(fit_intercept=True, normalize=True).fit(x_, train_y)
    x_ = transformer.fit_transform(val_x.tolist())
    gpu_score = gpu_PLR_model.score(x_, val_y)
    print("Polynomial Regression: {}".format(gpu_score))

    # gpu_svr_model = svm.SVR(kernel='linear')
    # gpu_svr_model.fit(train_x, train_y)
    # gpu_score = gpu_svr_model.score(val_x, val_y)
    # print("Support Vector Regression: {}".format(gpu_score))

    gpu_knn_model = KNeighborsRegressor(n_neighbors=2)
    gpu_knn_model.fit(train_x, train_y)
    gpu_score = gpu_knn_model.score(val_x, val_y)
    print("K-Nearest Neighbor: {}".format(gpu_score))

    gpu_rfr_model = RandomForestRegressor(max_depth=2, random_state=0)
    gpu_rfr_model.fit(train_x, train_y)
    gpu_score = gpu_rfr_model.score(val_x, val_y)
    print("Random Forest Regressor: {}".format(gpu_score))

    gpu_mlp_model = MLPRegressor(random_state=3, max_iter=500).fit(train_x, train_y)
    gpu_score = gpu_mlp_model.score(val_x, val_y)
    print("MLP: {}".format(gpu_score))

    # CPU modeling
    c_train_df, c_val_df, train_x, train_y, val_x, val_y = preprocess_df(cpu_df, 0.6, gpu=False)
    #val_x, val_y = train_x , train_y
    cpu_model = sklearn_model(fit_intercept=True, normalize=True).fit(train_x, train_y)
    cpu_score = cpu_model.score(val_x, val_y)
    print("Linear Regression: {}".format(cpu_score))

    transformer = PolynomialFeatures(degree=2, include_bias=True)
    x_ = transformer.fit_transform(train_x.tolist())
    cpu_PLR_model = sklearn_model(fit_intercept=True, normalize=True).fit(x_, train_y)
    x_ = transformer.fit_transform(val_x.tolist())
    cpu_score = cpu_PLR_model.score(x_, val_y)
    print("Polynomial Regression: {}".format(cpu_score))

    cpu_svr_model = svm.SVR(kernel='linear')
    cpu_svr_model.fit(train_x, train_y)
    cpu_score = cpu_svr_model.score(val_x, val_y)
    print("Support Vector Regression: {}".format(cpu_score))

    cpu_knn_model = KNeighborsRegressor(n_neighbors=2)
    cpu_knn_model.fit(train_x, train_y)
    cpu_score = cpu_knn_model.score(val_x, val_y)
    print("K-Nearest Neighbor: {}".format(cpu_score))

    cpu_rfr_model = RandomForestRegressor(max_depth=2, random_state=0)
    cpu_rfr_model.fit(train_x, train_y)
    cpu_score = cpu_rfr_model.score(val_x, val_y)
    print("Random Forest Regressor: {}".format(cpu_score))

    cpu_mlp_model = MLPRegressor(random_state=3, max_iter=500).fit(train_x, train_y)
    cpu_score = cpu_mlp_model.score(val_x, val_y)
    print("MLP: {}".format(cpu_score))

    # import pickle
    # with open('cpu_model', 'wb') as cpu_output:
    #     pickle.dump(cpu_model, cpu_output)
    # with open('gpu_model', 'wb') as gpu_output:
    #     pickle.dump(gpu_knn_model, gpu_output)

    submit_path = "/tmp/home/danlinjia/scripts/dl_submit.conf.csv"
    df = pd.read_csv(submit_path, header=0, skipinitialspace=True)

    a = gpu_predictor(g_val_df, gpu_model)
    b = gpu_predictor(g_train_df, gpu_model)
    c = gpu_predictor(g_val_df[g_val_df.model.str.contains("resnet")], gpu_model)
    d = gpu_predictor(g_val_df[g_val_df.model.str.contains("vgg")], gpu_model)
    #f = worker_allocator(df, gpu_model, cpu_knn_model)