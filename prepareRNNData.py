import pandas as pd
import numpy as np
import joblib as jl
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.signal import argrelextrema
from constants import (
    DAYS_PREDICTION_FORWARD,
    SCALER_TYPE,
    SPLIT_DATE,
    MERGED_CSV_FILE_NAME,
    ANALISYS_INDICATORS,
    SEQUENCE_LENGTH,
    INDICATOR_SCALER_FILE,
    TARGET_SCALER_FILE,
    FIBO_LEVELS
    )

class PrepareRNNData:
    def __init__(self, sequenceLength=SEQUENCE_LENGTH, splitDate=SPLIT_DATE,
                 csvDataFile=MERGED_CSV_FILE_NAME,
                 analisysIndicators=ANALISYS_INDICATORS,
                 scalerType=SCALER_TYPE,
                 daysPredictionForward=DAYS_PREDICTION_FORWARD):
        
        # Czytamy dane z podanego pliku *.csv
        self.df = pd.read_csv(csvDataFile)


        # Sprawdzamy jaki skaler wybrany
        if scalerType == "standard":
            # Scaler dla zmiennej docelowej i wskaznikow
            TargetScalerClass = StandardScaler
            IndicatorScalerClass = StandardScaler

        elif scalerType == "robust":
            # Scaler dla zmiennej docelowej i wskaznikow
            TargetScalerClass = RobustScaler
            IndicatorScalerClass = RobustScaler
        else:
            raise NameError(f"Scaler: {scalerType} not found!")
        

        # Skalowanie targetu
        self.targetScaler = TargetScalerClass()
        self.df["TargetScaled"] = self.targetScaler.fit_transform(self.df[["Daily%Change"]].shift(-DAYS_PREDICTION_FORWARD))

        # Usuwamy puste linie
        self.df.dropna(inplace=True)

        
        # Scaler dla wskaznikow
        self.indicatorStandardScaler = IndicatorScalerClass()
        self.dfScaled = pd.DataFrame(
            self.indicatorStandardScaler.fit_transform(self.df[analisysIndicators]),
            columns=analisysIndicators
            )

        # Oblicz i przechowaj lokalne ekstrema
        self.localMins, self.localMaxs = self.calculateLocalExtrema()

        # Tworzenie sekwencji czasowych
        X, y, dates = [], [], []
        for i in range(len(self.dfScaled) - sequenceLength- daysPredictionForward):
            seqScaled = self.dfScaled.iloc[i:i + sequenceLength].values
            seqEndDate = self.df.iloc[i + sequenceLength]["Date"]
            close = self.df.iloc[i + sequenceLength]["Close"]

            fiboArray = self.getFiboFeatures(seqEndDate, close, sequenceLength)
            if fiboArray is None:
                continue

            fullInput = np.concatenate((seqScaled, fiboArray), axis=1)
            X.append(fullInput)

            y.append(self.df["TargetScaled"].iloc[i + sequenceLength])
            dates.append(seqEndDate)

        # Konwersja na numpy array
        X, y = np.array(X), np.array(y)
        dates = pd.to_datetime(np.array(dates))

        # Podzial na zbior testowy i daty
        splitIndex = np.where(dates >= np.datetime64(splitDate))[0][0]

        self.XTrain, self.XTest = X[:splitIndex], X[splitIndex:]
        self.yTrain, self.yTest = y[:splitIndex], y[splitIndex:]

        print(f"Split date: {splitDate}")
        print(f"Training set: {self.XTrain.shape}\n\nTest set: {self.XTest.shape}")

        # Zapisz scalery na pozniej
        self.saveScalers()

    def saveScalers(self):
        jl.dump(self.indicatorStandardScaler, INDICATOR_SCALER_FILE)
        jl.dump(self.targetScaler, TARGET_SCALER_FILE)

    def calculateLocalExtrema(self, order=5):
        localMinIdx = argrelextrema(self.df["Close"].values, np.less_equal, order=order)[0]
        localMaxIdx = argrelextrema(self.df["Close"].values, np.greater_equal, order=order)[0]

        localMins = self.df.iloc[localMinIdx][["Date", "Close"]].reset_index(drop=True)
        localMaxs = self.df.iloc[localMaxIdx][["Date", "Close"]].reset_index(drop=True)

        return localMins, localMaxs
    
    def getFiboFeatures(self, seqEndDate, close, sequenceLength, _fiboLevels=FIBO_LEVELS):
        fiboLevels=_fiboLevels

        pastMax = self.localMaxs[self.localMaxs["Date"] < seqEndDate]
        pastMin = self.localMins[self.localMins["Date"] < seqEndDate]

        if pastMax.empty or pastMin.empty:
            return None
        
        high = pastMax.iloc[-1]["Close"]
        low = pastMin.iloc[-1]["Close"]
        diff = high - low if high != low else 1e-6

        fiboFeatures = [close - (high - level * diff) for level in fiboLevels]
        fiboArray = np.tile(fiboFeatures, (sequenceLength, 1))
        return fiboArray
    
    def savePreparedDataToCsv(self, fileName="prepared_sequences.csv", train=True):
        X = self.XTrain if train else self.XTest
        y = self.yTrain if train else self.yTest

        # Bierzemy ostatni dzień z każdej sekwencji
        last_days = X[:, -1, :]  # shape (samples, features)
        
        df = pd.DataFrame(last_days)
        df["TargetScaled"] = y

        df.to_csv(fileName, index=False)
        print(f"Saved prepared data to {fileName}")
