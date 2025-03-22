import pandas as pd

class MergeDataProcessor:
    def __init__(self, btcFile, dxyFile):
        self.btcFile = btcFile
        self.dxyFile = dxyFile

        self.btcDf = pd.read_csv(btcFile)
        self.dxyDf = pd.read_csv(dxyFile)

        self.btcDf["Date"] = pd.to_datetime(self.btcDf["Date"]).dt.strftime("%Y-%m-%d")
        self.dxyDf["Date"] = pd.to_datetime(self.dxyDf["Date"]).dt.strftime("%Y-%m-%d")

        self.df = self.btcDf.merge(self.dxyDf, on="Date", how="left", suffixes=("","_DXY"))

        self.df.ffill(inplace=True)

    def saveToCsv(self, outputFile):
        self.df.to_csv(outputFile, index=False)
        print(f"File saved as '{outputFile}'")

    def getData(self):
        return self.df
