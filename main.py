from asset import Asset
from constants import BTCUSD, BTC_CSV_FILE_NAME
from constants import DXY_CSV_FILE_NAME, DXY, ASSETS_START_DATE
from constants import MERGED_CSV_FILE_NAME, TARGET_SCALER_FILE
import matplotlib.pyplot as plt
from mergeDataProcessor import MergeDataProcessor
from prepareRNNData import PrepareRNNData
from lstmModel import RNNLSTMModel
from sklearn.preprocessing import StandardScaler
import joblib as jl
from tensorflow.python.client import device_lib
import os
import tensorflow as tf
from constants import TIME_INTERVAL
import time
# from keras.backend.tensorflow_backend import set_session

# Wypisac co mozemy zoptymalizowac, jakie wersje mozemy testowac
# Dodac zbior testowy
print("Wersja TensorFlow:", tf.__version__)
print("Czy TensorFlow widzi GPU?:", tf.config.list_physical_devices('GPU'))
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))

print(device_lib.list_local_devices())

btcusd = Asset(symbol=BTCUSD, interval=TIME_INTERVAL, start=ASSETS_START_DATE, period=None)
dxy = Asset(symbol=DXY, interval=TIME_INTERVAL, start=ASSETS_START_DATE, period=None)

btcusd.saveHistoryToCsv(fileName=BTC_CSV_FILE_NAME)
time.sleep(1)
dxy.saveHistoryToCsv(fileName=DXY_CSV_FILE_NAME)

dxy.deleteRow(name="Volume")

mergedFile = MergeDataProcessor(BTC_CSV_FILE_NAME, DXY_CSV_FILE_NAME)
mergedFile.saveToCsv(MERGED_CSV_FILE_NAME)

RNNData = PrepareRNNData()
RNNData.savePreparedDataToCsv("TrainData.csv")
RNNData.savePreparedDataToCsv("TestData.csv", False)

# Tworzenie i trenowanie modelu
model = RNNLSTMModel(RNNData.XTrain, RNNData.yTrain,
                     RNNData.XVal, RNNData.yVal,
                     RNNData.XTest, RNNData.yTest)
model.train()