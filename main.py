from asset import Asset
from constants import BTCUSD, BTC_CSV_FILE_NAME
from constants import DXY_CSV_FILE_NAME, DXY, DXY_START_DATE
from constants import MERGED_CSV_FILE_NAME, TARGET_SCALER_FILE
import matplotlib.pyplot as plt
from mergeDataProcessor import MergeDataProcessor
from prepareRNNData import PrepareRNNData
from lstmModel import RNNLSTMModel
from sklearn.preprocessing import StandardScaler
import joblib as jl
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

btcusd = Asset(symbol=BTCUSD, interval="1d", period="max")
dxy = Asset(symbol=DXY, interval="1d", start=DXY_START_DATE)

btcusd.saveHistoryToCsv(fileName=BTC_CSV_FILE_NAME)
dxy.saveHistoryToCsv(fileName=DXY_CSV_FILE_NAME)

dxy.deleteRow(name="Volume")

mergedFile = MergeDataProcessor(BTC_CSV_FILE_NAME, DXY_CSV_FILE_NAME)
mergedFile.saveToCsv(MERGED_CSV_FILE_NAME)

RNNData = PrepareRNNData()
RNNData.savePreparedDataToCsv("TrainData.csv")
RNNData.savePreparedDataToCsv("TestData.csv", False)

# Tworzenie i trenowanie modelu
model = RNNLSTMModel(RNNData.XTrain, RNNData.yTrain,
                     RNNData.XTest, RNNData.yTest)
model.train()

model.evaluate()

model.printPredictionsVsActual()
model.evaluateDirectionAccuracy()