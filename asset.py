import yfinance as yf
import matplotlib.pyplot as plt
import os
import pandas as pd
from constants import EMAPeriodList, FIBO_LEVELS, USE_FIBO
from scipy.signal import argrelextrema
import numpy as np

# Klasa dla ustalonego aktywa
class Asset:
    def __init__(self, symbol, interval = "1d", period = "max", start = None ):
        self.symbol = symbol
        self.interval = interval
        self.period = period
        self.start = start
        self.data = self.preProcessHistory()
        self.csvFile = None


    # Pobieranie historii o aktywie z yahoo finance
    def getHistory(self):
        asset = yf.Ticker(self.symbol)
        history = asset.history(period = self.period, interval = self.interval,
                                start = self.start)
        return history
    

    # Sformatuj poprawnie dane
    # Usun zbedne kolumny oraz 
    # ustaw format day
    def preProcessHistory(self):
        rawHistory = self.getHistory()
        rawHistory.index = rawHistory.index.date  # Ustawiamy daty jako index
        rawHistory["Date"] = rawHistory.index
        rawHistory.index.name = "Date"  # Nadajemy nazwę kolumnie indeksu
        rawHistory = rawHistory.drop(columns = ["Dividends", "Stock Splits"])
    
        rawHistory["Date"] = pd.to_datetime(rawHistory["Date"]).dt.strftime("%Y-%m-%d")

        # Obliczanie dziennej zmiany procentowej na podstawie kolumny ,,Close''
        rawHistory['Daily%Change'] = ( (rawHistory['Close'] - rawHistory['Close'].shift(1)) / rawHistory['Close'].shift(1) ) * 100

        rawHistory['RSI'] = self.calculateRSI(_rawHistory=rawHistory)

        rawHistory = self.calculateEMA(EMAPeriodArray=EMAPeriodList, _rawHistory=rawHistory)

        if USE_FIBO is True:
            rawHistory = self.addFiboLevels(rawHistory)

        self.addMoveSigns(_rawHistory=rawHistory)

        return rawHistory
    
    # Oblicz dwu etapowe RSI z okreslonym okresem
    def calculateRSI(self, period=14, _rawHistory=None):
        if _rawHistory is None:
            raise ValueError("rawHistory is None or is not specified!")
        
        delta = _rawHistory['Close'].diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avgGain = gain.rolling(window=period, min_periods=period).mean()
        avgLoss = loss.rolling(window=period, min_periods=period).mean()

        avgGain = avgGain.ewm(span=period, adjust=False).mean()
        avgLoss = avgLoss.ewm(span=period, adjust=False).mean()

        rs = avgGain / avgLoss
        rsi = 100 - (100 / (1 + rs))

        _rawHistory['RSI'] = rsi

        return rsi
    

    # Oblicz okreslona przez EMAPeriodArray ilosc 
    # Exponential Moving Average'ow o okreslonych
    # tam wartosciach 
    def calculateEMA(self, EMAPeriodArray=None, _rawHistory=None):
        if EMAPeriodArray is None or len(EMAPeriodArray) == 0:
            print("EMA period array is empty!")
            return
        
        if _rawHistory is None:
            raise FileNotFoundError("rawHistory is None or is not specified!")

        if "Close" not in _rawHistory.columns:
            raise ValueError("Column 'Close' not found")

        for period in EMAPeriodArray:
            _rawHistory[f"EMA{period}"] = _rawHistory["Close"].ewm(span=period, adjust=False).mean()
            _rawHistory.iloc[:period-1, _rawHistory.columns.get_loc(f"EMA{period}")] = None

        return _rawHistory
    

    # Zapisz do CSV do wskazanego pliku
    def saveHistoryToCsv(self, fileName = None):
        if fileName is None:
            print("No file name specified!")
            return
        
        self.csvFile = fileName

        if os.path.exists(fileName):
            print(f"File '{fileName}' already exists")
            return
        
        if self.data is not None:
            self.data.to_csv(fileName, index=True, index_label="Date")

    # Usun wskazana po nazwie kolumne
    def deleteRow(self, name=None):
        if name is None or name not in self.data:
            raise KeyError(f"Row name '{name}' does not exist")

        data = pd.read_csv(self.csvFile, index_col=0)

        data.drop(columns=[name], inplace=True)

        data.to_csv(self.csvFile, index=True)


    # Zwroc serie z okreslonej kolumny
    def getRowValues(self, columnName = None):
        if columnName is None:
            return "No row name specified"
        
        if columnName in self.data.columns:
            return self.data[columnName]
        else:
            return f"Row '{columnName}' does not exist"
        

    # Narysuj wykres wybranej kolumny
    def plotColumn(self, columnName):
        if columnName in self.data.columns:
            plt.figure(figsize=(10,6))
            plt.plot(self.data.index, self.data[columnName], label = columnName)
            plt.xlabel('Date')
            plt.ylabel(columnName)
            plt.title(f"{self.symbol} - {columnName}")
            plt.legend()
            plt.grid(True)
            # plt.show()
        else:
            print(f"Column '{columnName}' does not exist")


    def calculateLocalExtrema(self, order=5, _rawHistory=None):
        if _rawHistory is None:
            raise ValueError("No data file in calculateExtrema!")
        

        localMinIdx = argrelextrema(_rawHistory["Close"].values, np.less_equal, order=order)[0]
        localMaxIdx = argrelextrema(_rawHistory["Close"].values, np.greater_equal, order=order)[0]

        localMins = _rawHistory.iloc[localMinIdx][["Date", "Close"]].reset_index(drop=True)
        localMaxs = _rawHistory.iloc[localMaxIdx][["Date", "Close"]].reset_index(drop=True)

        return localMins, localMaxs
    
    def getFiboFeatures(self, seqEndDate, close, _fiboLevels=FIBO_LEVELS, _rawHistory=None):
        if _rawHistory is None:
            raise ValueError("No data file in getFibFeatures!")
    
        fiboLevels=_fiboLevels

        # Oblicz i przechowaj lokalne ekstrema
        localMins, localMaxs = self.calculateLocalExtrema(_rawHistory=_rawHistory)

        # Upewniamy się, że seqEndDate jest pojedynczą datą typu datetime
        # seqEndDate = pd.to_datetime(seqEndDate).date()

        pastMax = localMaxs[localMaxs["Date"] < seqEndDate]
        pastMin = localMins[localMins["Date"] < seqEndDate]

        if pastMax.empty or pastMin.empty:
            return None
        
        high = pastMax.iloc[-1]["Close"]
        low = pastMin.iloc[-1]["Close"]
        diff = high - low if high != low else 1e-6

        fiboFeatures = {f"FIBO_{level}": close - (high - level * diff) for level in fiboLevels}
        return fiboFeatures

    def addFiboLevels(self, _rawHistory):
        fiboColumns = {f"FIBO_{level}": [] for level in FIBO_LEVELS}
        for idx, row in _rawHistory.iterrows():
            fiboFeatures = self.getFiboFeatures(seqEndDate=row["Date"], close=row["Close"], _rawHistory=_rawHistory)
            if fiboFeatures:
                for level, value in fiboFeatures.items():
                    fiboColumns[level] = fiboColumns.get(level, [])
                    fiboColumns[level].append(value)

            else:
                for level in FIBO_LEVELS:
                    fiboColumns[f"FIBO_{level}"].append(np.nan)

        for level, values in fiboColumns.items():
            _rawHistory[level] = values

        return _rawHistory
    
    def addMoveSigns(self, _rawHistory=None):
        if _rawHistory is None:
            raise ValueError("No data file in addMoveSigns!")
        
        _rawHistory["MoveDirection"] = _rawHistory["Daily%Change"].apply(lambda x: 1 if x > 0 else 0)