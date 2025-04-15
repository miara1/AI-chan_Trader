import pandas as pd
import numpy as np
import joblib as jl
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
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
    FIBO_LEVELS,
    TARGET_COLUMN,
    SPLIT_RATIO
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
        elif scalerType == "minmax":
            # Scaler dla zmiennej docelowej i wskaznikow
            TargetScalerClass = MinMaxScaler
            IndicatorScalerClass = MinMaxScaler
        else:
            raise NameError(f"Scaler: {scalerType} not found!")
        

        # Skalowanie targetu
        self.targetScaler = TargetScalerClass()
        # self.df["TargetScaled"] = self.targetScaler.fit_transform(self.df[["Daily%Change"]].shift(-DAYS_PREDICTION_FORWARD))
        self.df["TargetScaled"] = self.targetScaler.fit_transform(self.df[[TARGET_COLUMN]].shift(-DAYS_PREDICTION_FORWARD))

        # Usuwamy puste linie
        self.df.dropna(inplace=True)

        
        # Scaler dla wskaznikow
        self.indicatorScaler = IndicatorScalerClass()
        self.dfScaled = pd.DataFrame(
            self.indicatorScaler.fit_transform(self.df[analisysIndicators]),
            columns=analisysIndicators
            )

        # Tworzenie sekwencji czasowych
        X, y, dates = [], [], []
        for i in range(len(self.dfScaled) - sequenceLength):
            X.append(self.dfScaled.iloc[i:i+sequenceLength].values)
 
            y.append(self.df["TargetScaled"].iloc[i+sequenceLength])
 
            dates.append(self.df["Date"].iloc[i+sequenceLength])

        # Konwersja na numpy array
        X, y = np.array(X), np.array(y)

        self.splitData(X=X, y=y)

        # Zapisz scalery na pozniej
        self.saveScalers()

    def splitData(self, X, y, splitRatio=SPLIT_RATIO):
        # Przekształć wektor procentowy na liczby rzeczywiste
        splitRatio = np.array(splitRatio) / np.sum(splitRatio)

                # Oblicz indeksy dla każdego zbioru
        total_samples = len(X)
        train_end = int(splitRatio[0] * total_samples)
        val_end = train_end + int(splitRatio[1] * total_samples)

        # Tworzenie zbiorów danych
        self.XTrain, self.yTrain = X[:train_end], y[:train_end]
        self.XVal, self.yVal = X[train_end:val_end], y[train_end:val_end]
        self.XTest, self.yTest = X[val_end:], y[val_end:]

        print(f"Training set: {self.XTrain.shape}")
        print(f"Validation set: {self.XVal.shape}")
        print(f"Test set: {self.XTest.shape}")

    def saveScalers(self):
        jl.dump(self.indicatorScaler, INDICATOR_SCALER_FILE)
        jl.dump(self.targetScaler, TARGET_SCALER_FILE)
    
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
