from asset import Asset
from constants import BTCUSD, BTC_CSV_FILE_NAME
from constants import DXY_CSV_FILE_NAME, DXY, DXY_START_DATE
import matplotlib.pyplot as plt
btcusd = Asset(symbol=BTCUSD, interval="1d", period="max")
dxy = Asset(symbol=DXY, interval="1d", start=DXY_START_DATE)

btcusd.saveHistoryToCsv(fileName=BTC_CSV_FILE_NAME)
dxy.saveHistoryToCsv(fileName=DXY_CSV_FILE_NAME)

pctChangeBTC = btcusd.getRowValues("Daily % Change")

print(pctChangeBTC)
print(dxy.getRowValues("Daily % Change"))

btcusd.plotColumn(columnName="Open")
btcusd.plotColumn(columnName="Close")
dxy.plotColumn("Close")

plt.show()

