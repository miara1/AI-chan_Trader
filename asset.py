import yfinance as yf
import matplotlib.pyplot as plt
import os

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
        rawHistory['Daily % Change'] = ( (rawHistory['Close'] - rawHistory['Close'].shift(1)) / rawHistory['Close'].shift(1) ) * 100

        return rawHistory
    

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
            self.data.to_csv(fileName)


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