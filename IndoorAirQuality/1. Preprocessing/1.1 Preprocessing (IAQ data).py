import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# ============================================
### 1. 데이터 로드
# ============================================

path = "C:/IndoorAirQuality Data/1.RawData/"

# 1-1 날짜 생성 (디렉토리 접근을 위한)
def make_date(startyear, endyear, startmonth, endmonth, startday, endday):
    import calendar
    date = []
    monthrange = []
    for year in range(startyear,endyear+1):
        for mon in range(startmonth,endmonth+1):
            monthlen = calendar.monthrange(year,mon)[1]
            if mon < 10:
                mon = "0" + str(mon)
            else:
                str(mon)
            for d in range(1,monthlen+1):
                if d < 10:
                    d="0"+str(d)
                else:
                    str(d)
                monthrange.append(d)
                concat_date = str(year) + str(mon) + str(d)
                date.append(concat_date)
    date2 = date[startday - 1: endday]
    return date2

date =  make_date(20,20,1,1,17,18)
# date = date + make_date(20,20,2,2,1,9)
# len(date)

#-----------------------
def load(path,date):
    #1-2 실내공기질 데이터 로드
    each_data1 = []
    for i,val in enumerate(date):
        data = pd.read_excel(path+val+"/"+"{}_IndoorAirQuality.xlsx".format(val))
        del data['Unnamed: 0']
        del data['touchSensor']
        Time_str = data.time.astype(str).tolist()
        Time_str = pd.DataFrame(Time_str, columns=["datetime"])
        Time_str["Date"] = Time_str.datetime.str.split().str[0]
        Time_str["Time"] = Time_str.datetime.str.split().str[1]
        data = pd.concat([data,Time_str],axis=1)
        del data['datetime']
        each_data1.append(data)
    # each_data1[0]

    #-----------------------
    #1-3 기상데이터 로드
    each_data2 = []
    for i,val in enumerate(date):
        data = pd.read_csv(path+val+"/"+"{}_weather.csv".format(val))
        Time_str = pd.to_datetime(data["일시"]).astype(str).tolist()  #0~9시까지 01~09시로 바꾼 후, str 변환
        Time_str = pd.DataFrame(Time_str, columns=["datetime"])
        Time_str["Time"] = Time_str.datetime.str.split().str[1] #time만 추출, 열 추가
        data = pd.concat([data, Time_str], axis=1)
        del data['datetime']
        each_data2.append(data)
    # each_data2[0]


    #-----------------------
    #1-4 대기질 데이터 로드
    each_data3 = []
    for i,val in enumerate(date):
        data = pd.read_csv(path+val+"/"+"{}_AQI.csv".format(val))
        data["Time"] = data["dataTime"].str.split().str[1] #time만 추출, 열 추가
        data["Time"][len(data)-1] = "23:59:59"
        each_data3.append(data)
    # each_data3[0]
    return each_data1, each_data2, each_data3


each_data1, each_data2, each_data3 = load(path,date)

# ============================================
### 2. 시계열 데이터 등간격
# ============================================
# temp = each_data1[6]
# temp = temp.set_index('Time', drop=True)
# each_data1[0].resample('2S', on='Time').ffill()
# temp.resample('2s').ffill().head(20)
# temp["Humidity"].resample('1S').ffill()


def same_interval(data_LabelIndex):
    # 2.1 측정구간 만들기
    match_time = pd.date_range("00:00:00", "23:59:59", freq="s")
    # time_list = each_data1[0]["Time"].values

    # 2.2 측정구간과 실제관측시간 교집합 인덱스 찾기
    match_time_list = []
    for i in range(len(match_time)):
        match_time_i = str(datetime.datetime.time(match_time[i]))
        match_time_list.append(match_time_i)

    df_time = pd.DataFrame(match_time_list, columns=["Time"])
    len(df_time)

    # 2.3 판다스 "vlookup" 사용하기
    # https://rfriend.tistory.com/258
    global each_data1
    global each_data2
    global each_data3
    indoorair_i = each_data1[data_LabelIndex]
    weather_i = each_data2[data_LabelIndex]
    AQI_i = each_data3[data_LabelIndex]
    #indoorair_i["Time"] = standard_datetime(data_LabelIndex)
    df_merge_how_left = pd.merge(df_time, indoorair_i, how='left', on="Time")

    # 2.4 Upsampling (back-filling)
    df_fillna_bfill = df_merge_how_left.fillna(method="bfill")

    # 2.5 merge (vlookup)
    df_merge_how_left2 = pd.merge(df_fillna_bfill, weather_i, how='left', on="Time")

    # 2.6 Upsampling (forward-filling)
    df_fillna_ffill = df_merge_how_left2.fillna(method="ffill")

    # 2.7 merge (vlookup)
    df_merge_how_left3 = pd.merge(df_fillna_ffill, AQI_i, how='left', on="Time")

    # 2.8 Upsampling (back-filling)
    df_fillna_bfill2 = df_merge_how_left3.fillna(method="bfill")

    len(df_fillna_bfill2)
    df_dropna = df_fillna_bfill2.dropna()
    df_dropna = df_dropna.reset_index(drop=True)
    len(df_dropna)
    # df_dropna.to_csv("merge4.csv")
    return df_dropna


# ============================================
### 3. 일별 데이터 병합 (rbind)
# ============================================
first_df = same_interval(0)
for i in range(1,len(date)):
    next_df = same_interval(i)
    first_df = pd.concat([first_df,next_df])

merge_df = first_df
merge_df = merge_df.reset_index(drop=True)
# len(merge_df)

# ============================================
### 4. 이상치 처리
# ============================================
def Dealing_with_Outliers():
    # 4.1 측정가능범위 밖 이상치 처리
    global merge_df
    merge_df.isnull().sum()

    for i in range(len(merge_df)):
        if merge_df["TVOC"][i]>=1187:
            merge_df["TVOC"][i] = None
        if merge_df["CO2"][i] >= 8192:
            merge_df["CO2"][i] = None
        if merge_df["PM 1.0"][i] >= 1000:
            merge_df["PM 1.0"][i] = None
        if merge_df["PM 2.5"][i] >= 1000:
            merge_df["PM 2.5"][i] = None
        if merge_df["PM 10"][i] >= 1000:
            merge_df["PM 10"][i] = None
        if merge_df["Temperature"][i] >= 80 or merge_df["Temperature"][i] <= -40:
            merge_df["Temperature"][i] = None

    merge_df.isnull().sum()

    merge_df = merge_df.fillna(method="ffill")
    merge_df = merge_df.rename(columns = {'time': 'datetime'})

    # 4.2 지수평균으로 이상치 처리
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np


    # 이상치 탐지를 위한 적절한 ema alpha값 찾기
    ema_dict = {}
    for i in range(6):
        plt.subplot(3,2,i+1)
        column_i = merge_df.columns[6:11+1][i]
        if i <2:
            window_size = 60
            column_window = merge_df[merge_df.columns[11]]
            ema = merge_df[column_i].ewm(span=window_size).mean()
            ema_dict[column_i] = ema
            plt.plot(merge_df[column_i])
            plt.plot(ema, c='r')
            plt.plot(column_window, c='g')
            plt.title(column_i + "  (window_size=60, alpha=0.03)")
        else:
            window_size = 15
            column_window = np.log(merge_df[merge_df.columns[11]])*5
            ema = merge_df[column_i].ewm(span=window_size).mean()
            ema_dict[column_i] = ema
            plt.plot(merge_df[column_i])
            plt.plot(ema, c='r')
            plt.plot(column_window, c='g')
            plt.title(column_i + "  (window_size=15)")
        # ema = merge_df[column_i].ewm(span=window_size).mean()

    # ema로 이상치 포함한 column 대체
    merge_df2 = merge_df.copy()
    for i in range(len(ema_dict)):
        ema_key = list(ema_dict.keys())[i]
        ema_value = ema_dict[ema_key]
        merge_df2[ema_key] = ema_value
    return merge_df2


def rename():
    preprocessed_data = Dealing_with_Outliers()
    preprocessed_data.info()
    columns = ['Time', 'Humidity', 'Temperature',
                'TVOC', 'CO2', 'PM 1.0', 'PM 2.5', 'PM 10',
                'OpenWindowsDistance1',
                'datetime', 'Date',
                '기온(°C)', '누적강수량(mm)',
                '풍향(deg)', '풍속(m/s)', '현지기압(hPa)', '습도(%)',
                'dataTime', 'pm10Value', 'pm25Value']
    dataset = preprocessed_data[columns]

    original_name = dataset.columns[11:17]
    rename = ['temp_in_air', 'rainfall', 'wind_direction', 'wind_speed', 'air_pressure', 'humidity_in_air']
    for i in range(len(original_name)):
        dataset = dataset.rename(columns={original_name[i]:rename[i]})
    return dataset


dataset = rename()
path2 = "C:/IndoorAirQuality Data/2.Dataset by weeks/"
dataset.to_csv(path2 + "2.Dataset by weeks.csv")




# # # 잘 대체되었는지 확인
# test = pd.read_csv(path2+"merge_week3(5rd week. Jan (27_02))_UTF-8.csv")
# test = test[test.columns[1:]]
#
# for i in range(6):
#     plt.subplot(3,2,i+1)
#     column_i = test.columns[6:11+1][i]
#     if i <2:
#         plt.plot(test[column_i])
#         plt.plot(test[column_i], c='r')
#         # plt.plot(column_window, c='g')
#         plt.title(column_i + "  (window_size=60, alpha=0.03)")
#     else:
#         plt.plot(test[column_i])
#         plt.plot(test[column_i], c='r')
#         # plt.plot(column_window, c='g')
#         plt.title(column_i + "  (alpha=0.1)")
#


