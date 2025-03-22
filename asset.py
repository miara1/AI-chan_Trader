import yfinance as yf
import matplotlib.pyplot as plt
import os
import pandas as pd
from constants import EMAPeriodList

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
        rawHistory.index = rawHistory.index.date
        rawHistory = rawHistory.drop(columns = ["Dividends", "Stock Splits"])
    
        # Obliczanie dziennej zmiany procentowej na podstawie kolumny ,,Close''
        rawHistory['Daily%Change'] = ( (rawHistory['Close'] - rawHistory['Close'].shift(1)) / rawHistory['Close'].shift(1) ) * 100

        rawHistory['RSI'] = self.calculateRSI(_rawHistory=rawHistory)

        rawHistory = self.calculateEMA(EMAPeriodArray=EMAPeriodList, _rawHistory=rawHistory)

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