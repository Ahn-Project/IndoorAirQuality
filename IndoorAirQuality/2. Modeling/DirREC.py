# 0. 사용할 패키지 불러오기
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras import initializers
font = {'size': 18}
plt.rc('font', **font)

# 랜덤시드 고정시키기
seed_nb=0
np.random.seed(seed_nb)
import tensorflow as tf
tf.set_random_seed(seed_nb)

#=====================================
# 1. 데이터 준비하기
#=====================================
def load(num_of_week):
    path = "C:/IndoorAirQuality Data/4.Dataset by weeks (for Training)/"

    # 주간 데이터 로드
    each_data = []
    num_of_weeks = num_of_week
    for i in range(1,num_of_weeks+1):
        data = pd.read_csv(path+"Dataset by week1 (DownSample_step=1min).csv".format(i))
        each_data.append(data)

    # 데이터 행 병합
    dataset = each_data[0]
    for i in range(1,num_of_weeks):
        dataset = pd.concat([dataset,each_data[i]])

    dataset = dataset.reset_index(drop=True)
    dataset = dataset.rename(columns={'time_prime': "open-close time",
                                      'PM 2.5': 'PM 2.5_Indoor',
                                      'pm25Value': 'PM 2.5_Outdoor',
                                      'PM 10': 'PM 10_Indoor',
                                      'pm10Value': 'PM 10_Outdoor'
                                      })
    return dataset


# 랜덤시드 고정시키기
np.random.seed(5)

#-------------------
### 데이터셋 생성 함수
def seq2dataset(dataset, X_col_name, y_col_name, n_inputs, n_outputs):
    col_names = X_col_name + y_col_name
    dataset_X = []
    dataset_Y = []
    cols = []

    # 열벡터 행렬 만들기
    for i, col in enumerate(dataset[col_names].columns):
        cols.append(dataset[col].values)
    n_features = len(cols)
    n_samples = len(cols[0])
    cols = np.reshape(cols, (n_features, n_samples)).T

    # moving window format 만들기
    for i in range(len(dataset) - (n_inputs + n_outputs)):
        subset = cols[i:(i + n_inputs + n_outputs)]
        subset_X = subset[:n_inputs]
        subset_y = subset[n_inputs:len(subset), len(cols[0]) - 1]
        dataset_X.append(subset_X)
        dataset_Y.append(subset_y)

    return np.array(dataset_X), np.array(dataset_Y)

# dataset_X.shape
# dataset_Y.shape


#-------------------
# 정규화
def std(dataset_X):
    def Standardized(X):
        from sklearn.preprocessing import StandardScaler
        std_X = StandardScaler().fit_transform(X)
        return std_X
    temp = np.reshape(dataset_X, (len(dataset_X),dataset_X.shape[1]*dataset_X.shape[2]))
    std_dataset_X = np.reshape(Standardized(temp), (len(dataset_X),dataset_X.shape[1],dataset_X.shape[2]))
    return std_dataset_X

#-------------------
def split_data(dataset, dataset_X, dataset_Y):
    # 2.1 분할 기준 인덱스 만들기
    index_timeprime1 = dataset[dataset["open-close time"] == 1].index.values
    start_index = []
    end_index = []
    for i in range(len(index_timeprime1)):
         if i!=len(index_timeprime1)-1:
            start = index_timeprime1[i]
            end = index_timeprime1[i+1]-1
            start_index.append(start)
            end_index.append(end)
         else:
             start = index_timeprime1[i]
             end = len(dataset["open-close time"])
             start_index.append(start)
             end_index.append(end)

    def Hold_out(X_data, y_data, ratio1, ratio2):
        train_index = end_index[:int(len(start_index) * ratio1)]
        train_index = train_index[len(train_index)-1]
        val_index = end_index[int(len(start_index) * ratio1):int(len(start_index) * ratio2)]
        val_index = val_index[len(val_index) - 1]
        train_X = X_data[:train_index+1]
        train_y = y_data[:train_index+1]
        val_X = X_data[train_index + 1:val_index + 1]
        val_y = y_data[train_index + 1:val_index + 1]
        test_X = X_data[val_index + 1:]
        test_y = y_data[val_index + 1:]
        # test_X = X_data[train_index+1:val_index+1]
        # test_y = y_data[train_index+1:val_index+1]
        # val_X = X_data[val_index+1:]
        # val_y = y_data[val_index+1:]

        return train_X, train_y, val_X, val_y, test_X, test_y


    train_X, train_y, val_X, val_y, test_X, test_y = \
        Hold_out(dataset_X, dataset_Y, 0.8, 0.9)

    return train_X, train_y, val_X, val_y, test_X, test_y



#-----------------------------
def Modeling(train_X, train_y, val_X, val_y, test_X, test_y):
    global time_step
    global num_of_features
    global n_inputs
    global n_outputs
    global num_epochs
    global alpha
    # initializer = initializers.he_normal(seed=seed_nb)
    initializer = initializers.glorot_uniform(seed=seed_nb)

    # vanilla
    model = Sequential()
    model.add(LSTM(128, input_shape=(time_step, num_of_features), kernel_initializer=initializer))
    model.add(Dense(n_outputs))

    # # stacked
    # model = Sequential()
    # model.add(LSTM(32, return_sequences=True, input_shape=(time_step, num_of_features), kernel_initializer=initializer))
    # # model.add(LSTM(64, activation='relu', return_sequences=True))
    # model.add(LSTM(32, activation='relu'))
    # model.add(Dense(n_outputs))

    # 4. 모델 학습과정 설정하기
    from keras.optimizers import Adam
    model.compile(optimizer=Adam(alpha), loss='MSE')

    # 손실 이력 클래스 정의
    class LossHistory(keras.callbacks.Callback):
        def init(self):
            self.losses = []

        def on_epoch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))

    # 5. 모델 학습시키기
    history = LossHistory()  # 손실 이력 객체 생성
    history.init()

    import timeit
    start = timeit.default_timer()

    model.fit(train_X, train_y, epochs=num_epochs, batch_size=100, verbose=2, shuffle=False,
              validation_data=(val_X, val_y),
              callbacks=[history])  # 50 is X.shape[0]

    stop = timeit.default_timer()
    runtime=[]
    runtime.append(stop - start)
    # RMSE
    # predicted value에 대한 RMSE
    actual_test = test_y
    predicted_test = model.predict(test_X, batch_size=1)
    RMSE = np.sqrt(np.mean(np.square(actual_test - predicted_test)))
    print("RMSE : {}".format(RMSE))
    print(np.mean(runtime))

    return  model, history, RMSE, test_X, actual_test, predicted_test, runtime



#=================================
def dirREC_strategy():
    global n_inputs
    global time_step

    RMSE_byStep=[]
    hist_byStep = []
    runtime_byStep = []
    actual_tests = []
    predicted_tests = []

    #-----------------
    # dirREC strategy
    idx = 0
    for i in range(10):
        if idx == 0:
            dataset = load(1)
            dataset_X, dataset_Y = seq2dataset(dataset, X_col_name, y_col_name, n_inputs, n_outputs)
            std_X = std(dataset_X)
            train_X, train_y, val_X, val_y, test_X, test_y =\
                split_data(dataset, std_X, dataset_Y)
            model_TVOC, hist_TVOC, RMSE_TVOC, actual_test_X,\
            actual_test_TVOC, predicted_test_TVOC, runtime \
                = Modeling(train_X, train_y, val_X, val_y, test_X, test_y)
            actual_tests.append(actual_test_TVOC)
            predicted_tests.append(predicted_test_TVOC)
            RMSE_byStep.append(RMSE_TVOC)
            runtime_byStep.append(runtime[0])
            # hist_mean.append(hist_TVOC)
            idx += 1
            time_step += 1
            n_inputs += 1

        else:
            dataset = load(1)

            # X 만들기
            if idx ==1:
                temp = dataset_X.copy()
                predicted_X = model_TVOC.predict(dataset_X)
                predicted_X = predicted_X.reshape(dataset_X.shape[0], n_outputs, num_of_features)
                dirREC_X = np.concatenate([temp, predicted_X], axis=1)
                std_dirREC_X = std(dirREC_X)

            else:
                predicted_X = model_TVOC.predict(dirREC_X)
                predicted_X = predicted_X.reshape(dataset_X.shape[0], n_outputs, num_of_features)
                dirREC_X = np.concatenate([dirREC_X, predicted_X], axis=1)
                std_dirREC_X = std(dirREC_X)

            # Y 만들기
            temp_y = dataset_Y.copy()
            temp_y = temp_y[idx:]
            end_y = temp_y[len(dataset_Y) - 1 - idx][0]
            end_y = np.array([[end_y]])
            # temp_y = list(temp_y)
            for i in range(idx):
                temp_y = np.concatenate((temp_y, end_y), axis=0)
                # temp_y.extend(temp_y[len(dataset_Y) - 1 - idx])
            dirREC_y = np.reshape(np.array(temp_y), (dataset_Y.shape[0], dataset_Y.shape[1]))


            # 모델 학습
            train_X, train_y, val_X, val_y, test_X, test_y = \
                split_data(dataset, std_dirREC_X, dirREC_y)
            model_TVOC, hist_TVOC, RMSE_TVOC, actual_test_X, \
            actual_test_TVOC, predicted_test_TVOC, runtime \
                = Modeling(train_X, train_y, val_X, val_y, test_X, test_y)
            actual_tests.append(actual_test_TVOC)
            predicted_tests.append(predicted_test_TVOC)
            RMSE_byStep.append(RMSE_TVOC)
            runtime_byStep.append(runtime[0])
            # hist_mean.append(hist_TVOC)
            idx += 1
            time_step += 1
            n_inputs += 1

    RMSE = round(np.mean(RMSE_byStep), 2)

    p = np.array(predicted_tests)
    a = np.array(actual_tests)

    for i in range(len(p)):
        if i==0:
            actual_test = a[0]
            predicted_test = p[0]
        else:
            actual_test = np.concatenate((actual_test,a[i]), axis=1)
            predicted_test = np.concatenate((predicted_test,p[i]), axis=1)
    return actual_test, predicted_test, RMSE_byStep, runtime_byStep


#=================================
# 하이퍼파라미터 설정 및 학습
#=================================
# data_X_base = ['time_prime', 'delta_Pw', 'delta_Pt', 'window1_area', 'window2_area']
# X_col_name = ["EMA(2)_cumsum_TVOC", 'delta_Pw', 'delta_Pt', 'window1_area', 'window2_area']
X_col_name = []
y_col_names = ['TVOC', 'PM 2.5_Indoor', 'PM 10_Indoor']
y_col_name = [y_col_names[1]]
time_step = 2
num_of_features = len(X_col_name+y_col_name)
n_inputs, n_outputs = time_step ,1
num_epochs = 10
alpha = 0.01
# initializer = initializers.he_normal(seed=seed_nb)
initializer = initializers.glorot_uniform(seed=seed_nb)

actual_test, predicted_test, RMSE_byStep, runtime_byStep = \
    dirREC_strategy()




