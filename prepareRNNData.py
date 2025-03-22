import pandas as pd
import numpy as np
import joblib as jl
from sklearn.preprocessing import StandardScaler
from constants import SPLIT_DATE, MERGED_CSV_FILE_NAME, ANALISYS_INDICATORS, SEQUENCE_LENGTH

class PrepareRNNData:
    def __init__(self, sequenceLength=SEQUENCE_LENGTH, splitDate=SPLIT_DATE,
                 csvDataFile=MERGED_CSV_FILE_NAME,
                 analisysIndicators=ANALISYS_INDICATORS):
        
        # Czytamy dane z podanego pliku *.csv
        self.df = pd.read_csv(csvDataFile)

        # Scaler dla zmiennej docelowej
        self.targetScaler = StandardScaler()
        self.df["TargetScaled"] = self.targetScaler.fit_transform(self.df[["Daily%Change"]].shift(-1))
        
        # Usuwamy puste linie
        self.df.dropna(inplace=True)

                # Scaler dla wskaznikow
        self.indicatorStandardScaler = StandardScaler()
        self.dfScaled = pd.DataFrame(
            self.indicatorStandardScaler.fit_transform(self.df[analisysIndicators]),
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
        jl.dump(self.indicatorStandardScaler, "indicatorScaler.pkl")
        jl.dump(self.targetScaler, "targetScaler.pkl")