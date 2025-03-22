# Nazwy aktyw
DXY = "DX-Y.NYB"
BTCUSD = "BTC-USD"

# Nazwy roznych plikow
BTC_CSV_FILE_NAME = "testBTCHistory.csv"
DXY_CSV_FILE_NAME = "testDXYHistory.csv"
MERGED_CSV_FILE_NAME = "testMergedFile.csv"

# Data rozpoczecia indeksu DXY
DXY_START_DATE = "2014-09-17"

# Wartosci kolejnych EMA
EMAPeriodList = [12, 26, 50, 200]

# Przygotowanie danych dla RNN
SPLIT_DATE = "2024-01-01"
SEQUENCE_LENGTH = 30
ANALISYS_INDICATORS = ["Open", "Close", "High", "Low", "Volume", "RSI",
                       "EMA12", "EMA26", "EMA50", "EMA200", "Daily%Change",
                      "Close_DXY", "RSI_DXY", "EMA12_DXY", "EMA26_DXY",
                      "EMA50_DXY", "EMA200_DXY", "Daily%Change_DXY"]
