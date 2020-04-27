import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime


path = "C:/IndoorAirQuality Data/2.Dataset by weeks/"

dataset = pd.read_csv(path+"2.Dataset by weeks.csv")
# dataset.info()

#==============================================
# 1.환기시간 변수 만들기
#==============================================
def Ventilation_Time():
    # 각 레코드별 환기누적시간
    global dataset
    dataset2 = dataset.copy()
    dataset2["time_prime"] = None

    time_prime = 0
    for i in range(len(dataset2)-1):
        if dataset2['OpenWindowsDistance1'][i] > 20:
            t11 = datetime.datetime.strptime(dataset2["Time"][i].split()[0],'%H:%M:%S')
            t22 = datetime.datetime.strptime(dataset2["Time"][i+1].split()[0],'%H:%M:%S')
            delta_time = t22-t11
            time_prime += int(str(delta_time).split(":")[2])
            dataset2["time_prime"][i+1] = time_prime
        else:
            time_prime = 0
            dataset2["time_prime"][i+1] = time_prime

    # dataset2.head()
    dataset2.time_prime.isnull().sum()
    dataset2.time_prime = dataset2.time_prime.fillna(method="bfill")
    return dataset2


#==============================================
# 2. 환기 구동력 변수 만들기
#==============================================
def Ventilation_DrivingForce():
    dataset2 = Ventilation_Time()
    dataset2["delta_Pw"] = None
    dataset2["delta_Pt"] = None

    for i in range(len(dataset2)):
        # 창문개폐여부
        if dataset2['OpenWindowsDistance1'][i] > 20:
            open_or_close = 1
        else:
            open_or_close = 0

        # 풍력에 의한 환기 구동력 변수
        Cp = 0.25   # 압력계수
        P_air = dataset2.loc[i].at['air_pressure']  #대기압
        R = 287.058 #기체상수
        Temp_air =  dataset2.loc[i].at['temp_in_air'] #대기온도
        Density_out = P_air/(R*(Temp_air+273.15))   #실외의 공기밀도
        V = dataset2.loc[i].at['wind_speed']   #대기 풍속
        delta_Pw = Cp * (Density_out/2) * (V**2)
        delta_Pw = open_or_close * delta_Pw

        # 온도차에 의한 환기 구동력 변수
        h = 1   #높이
        P_in = P_air-0.5 #실내 압력 (*일반적 실내외 압력차(0.5hPa) 활용한 실내 압력)
        Temp_in = dataset2.loc[i].at['Temperature']
        Density_in = P_in/(R*(Temp_in+273.15))
        Density_Difference = Density_out - Density_in
        delta_Pt = Density_Difference * h
        delta_Pt = open_or_close * delta_Pt

        dataset2["delta_Pw"][i] = delta_Pw
        dataset2["delta_Pt"][i] = delta_Pt
    return dataset2


#==============================================
# 3. 유입단면적 변수 만들기
#==============================================
def Inflow_Area():
    dataset2 = Ventilation_DrivingForce()
# 창문 개폐 구간 별 평균으로 대체한 변수 만들기
    sum = 0
    count = 0
    means = []
    for i in range(len(dataset2['OpenWindowsDistance1'])):
        if dataset2['OpenWindowsDistance1'][i]>20:
            sum+=dataset2['OpenWindowsDistance1'][i]
            count+=1
        elif sum>0:
            means.append(sum/count)
            sum = 0
            count = 0

    # len(means)

    mean_column = []
    group = -1
    for i in range(len(dataset2['OpenWindowsDistance1'])):
        if i>0:
            last_value = dataset2['OpenWindowsDistance1'][i-1]
            current_value = dataset2['OpenWindowsDistance1'][i]
        else:
            last_value = 0
            current_value = dataset2['OpenWindowsDistance1'][i]

        if current_value>=20 and last_value<20:
            if group < len(means)-1:
                group += 1
        #     print("GROUP: ", str(group))
        # print(last_value, current_value)

        if dataset2['OpenWindowsDistance1'][i]<20:
            mean_column.append(0)
        else:
            mean_column.append(means[group])

    len(mean_column)
    len(dataset2)

    dataset3 = dataset2.copy()
    dataset3["OWD_mean"] = mean_column


    len(dataset3["OWD_mean"].unique())

    # plt.plot(dataset3["OWD_mean"])
    #-------------------

    window1_area_kitchen = (dataset3["OWD_mean"]/100) * (105/100) #미터 환산
    window2_area_TV = (dataset3["OWD_mean"]/100) * (160/100) #미터 환산

    dataset2["window1_area"] = window1_area_kitchen
    dataset2["window2_area"] = window2_area_TV
    return dataset2


dataset2 = Inflow_Area()

dataset2.info()


path2 = "C:/IndoorAirQuality Data/3.Dataset by weeks (add Features)/"
dataset2.to_csv(path2 + "3.Dataset by weeks (add Features).csv")











