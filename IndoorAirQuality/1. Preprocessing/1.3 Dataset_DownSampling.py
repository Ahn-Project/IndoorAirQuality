import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path3 = "C:/IndoorAirQuality Data/3.Dataset by weeks (add Features)/"

dataset = pd.read_csv(path3 + "3.Dataset by weeks (add Features).csv")

columns1 = ['TVOC', 'CO2', 'PM 2.5', 'PM 10', 'pm25Value', 'pm10Value']
columns2 = ['time_prime', 'delta_Pw', 'delta_Pt', 'window1_area', 'window2_area']
columns3 = ['temp_in_air', 'rainfall', 'wind_direction', 'wind_speed', 'air_pressure', 'humidity_in_air']
columns = columns1 + columns2 + columns3
dataset = dataset[columns]

# dataset.info()
n_step = 1
Field_Name = ['TVOC', 'CO2', 'PM 2.5', 'PM 10']


# ============================================
# 1. 이상치 처리
# ============================================
def Dealing_with_Outliers(data):
    dataset = data
    # 1.1 OpenWindowsDistance 변수 이상치 구간 time_prime=0으로 대체
    dataset_nonzero = dataset[dataset.time_prime != 0]
    dataset_nonzero = dataset_nonzero.reset_index(drop="True")
    # dataset_nonzero.info()

    idx = dataset[dataset.time_prime == 1].index
    idx2 = dataset_nonzero[dataset_nonzero.time_prime == 1].index

    temp = dataset.copy()
    for i in range(len(idx)):
        if i > 0:
            idx_diff = idx2[i] - idx2[i - 1]
            if idx_diff < 300:
                temp.time_prime[idx[i - 1]:idx[i - 1] + idx_diff] = 0

    temp = temp.reset_index(drop="True")
    dataset = temp


    # 1.2 이상치 "-" 대체
    temp2 = dataset.copy()

    if len(dataset.pm25Value[dataset.pm25Value=='-']) > 0:
        temp2.pm25Value[temp2.pm25Value=='-'] = None
    if len(dataset.pm10Value[dataset.pm10Value=='-']) > 0:
        temp2.pm10Value[temp2.pm10Value=='-'] = None

    temp2 = temp2.fillna(method="ffill")
    temp2 = temp2.fillna(method="bfill")
    temp2.pm25Value = temp2.pm25Value.astype("float")
    temp2.pm10Value = temp2.pm10Value.astype("float")

    dataset = temp2
    return dataset



# ============================================
### 다운샘플링
# ============================================
# n초 타임스텝으로 다운샘플링
def DownSampling(n_step):
    dataset2 = Dealing_with_Outliers(dataset)
    time_step = 60*n_step
    dataset2.index = pd.date_range("2020-1-1", periods=len(dataset), freq="s")
    t_mean = dataset2.resample('{}s'.format(time_step)).mean()
    dataset_DownSample = t_mean
    # dataset_DownSample.info()

    idx = dataset[dataset.time_prime == 1].index.values
    idx_DownSample = np.trunc(idx/time_step).astype('int')
    return dataset_DownSample, idx_DownSample, time_step



# ============================================
# time_prime 시작값 수정 (평균으로 된 값 다시 수정)
# ============================================
def time_prime_to_default(n_step):
    dataset_DownSample, idx_DownSample, time_step = DownSampling(n_step)
    DownSample = dataset_DownSample.copy()

    idx_nonzero = dataset[dataset.time_prime != 0].index.values
    idx_nonzero_DownSample = np.trunc(idx_nonzero/time_step).astype('int')
    idx_nonzero_DownSample = np.unique(idx_nonzero_DownSample)


    DownSample["time_prime2"]=0

    for i, val1 in enumerate(idx_nonzero_DownSample):
        for j, val2 in enumerate(idx_DownSample):
            if val1 == val2:
                first = val2
                DownSample["time_prime2"][val1] = 1
            else:
                diff = (val1-first) + 1
                DownSample["time_prime2"][val1] = diff

    DownSample.time_prime2 = DownSample.time_prime2.fillna(0)
    # plt.plot(test.time_prime2)

    DownSample = DownSample.drop("time_prime", axis=1)
    DownSample = DownSample.rename(columns = {"time_prime2":"time_prime"})

    return DownSample

dataset3 = time_prime_to_default(n_step)


path4 = "C:/IndoorAirQuality Data/4.Dataset by weeks (for Training)/"
dataset3.to_csv(path4 + "Dataset by week1 (DownSample_step={}min).csv".format(n_step))




