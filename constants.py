# Nazwy aktyw
DXY = "DX-Y.NYB"
BTCUSD = "BTC-USD"


#================================================================


# Nazwy roznych plikow
BTC_CSV_FILE_NAME = "testBTCHistory.csv"
DXY_CSV_FILE_NAME = "testDXYHistory.csv"
MERGED_CSV_FILE_NAME = "testMergedFile.csv"


#================================================================


# Data rozpoczecia indeksu DXY
DXY_START_DATE = "2014-09-17"


#================================================================


# Wartosci kolejnych EMA
EMAPeriodList = [12, 26, 50, 200]


#================================================================


# Interwal czasowy
TIME_INTERVAL = "1d"


#================================================================


# Przygotowanie danych dla RNN
SPLIT_DATE = "2024-01-01" # aktualnie stare - nie dziala na split date
SPLIT_RATIO = [70, 15, 15]
SEQUENCE_LENGTH = 30
ANALISYS_INDICATORS_OLD = ["Open", "Close", "High", "Low", "Volume", "RSI",
                       "EMA12", "EMA26", "EMA50", "EMA200", "PriceChange",
                      "Close_DXY", "RSI_DXY", "EMA12_DXY", "EMA26_DXY",
                      "EMA50_DXY", "EMA200_DXY", "PriceChange_DXY"]

ANALISYS_INDICATORS = ["Open", "Close", "High", "Low", "Volume", "RSI",
                            "EMA12", "EMA26", "EMA50", "EMA200", "PriceChange", "Close_DXY"]
SCALER_TYPE = "standard"    # "robust" - robust scaler
                            # "standard" - standard scaler
                            # "minmax" - MinMaxScaler

INDICATOR_SCALER_FILE = "indicatorScaler.pkl"
TARGET_SCALER_FILE = "targetScaler.pkl"


USE_FIBO = False    # True - liczenie fibo wlaczone (ale nie koniecznie uwzgledniane)
                    # False - liczenie fibo wyłączone

FIBO_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786]



#================================================================



# Parametry modelu
NUMBER_OF_NEURONS = 512
DROPOUT = 0.15
DENSE = 1
RETURN_SEQUENCES = True # True - dodaje druga i trzecia warstwe,
                        # False - pozostaje jedna warstwa
NEGATIVE_SLOPE = 0.6
RE_LU = "Leaky" # "Leaky" - LeakyReLu
                # "P" - PReLu
                # "_ReLu" - standardowy ReLu
                # "tanh" - tanh
HUBER_DELTA = 0.2
LOSS = "Huber"  # "Huber" - Huber
                # "mse" - mse
                # "binary_crossentropy"



#================================================================



# Parametry uczenia
BATCH_SIZE = 32
EPOCHS = 65
INTERVALS_PREDICTION_FORWARD = 1  # Przewidywanie o jeden interwał do przodu



#================================================================



# Parametry wyniku
HOW_MANY_OUTPUTS = 100
TARGET_COLUMN = "PriceChange"
