# Nazwy aktyw
DXY = "DX-Y.NYB"
BTCUSD = "BTC-USD"


#================================================================


# Nazwy roznych plikow
BTC_CSV_FILE_NAME = "./DataAndScalerFiles/testBTCHistory.csv"
DXY_CSV_FILE_NAME = "./DataAndScalerFiles/testDXYHistory.csv"
MERGED_CSV_FILE_NAME = "./DataAndScalerFiles/testMergedFile.csv"
SAVE_PATH = "./FinalResults/Models/1.keras"


#================================================================


# Data rozpoczecia pobierania danych aktywow
ASSETS_START_DATE = "2015-01-01"


#================================================================


# Wartosci kolejnych EMA
EMAPeriodList = [12, 26, 50, 200]


#================================================================


# Interwal czasowy
TIME_INTERVAL = "1d"


#================================================================


# Przygotowanie danych dla RNN
SPLIT_DATE = "2022-01-01" # aktualnie stare - nie dziala na split date
SPLIT_RATIO = [80, 10, 10]
SEQUENCE_LENGTH = 30
ANALISYS_INDICATORS_OLD = ["Open", "Close", "High", "Low", "Volume", "RSI",
                       "EMA12", "EMA26", "EMA50", "EMA200", "PriceChange",
                      "Close_DXY", "RSI_DXY", "EMA12_DXY", "EMA26_DXY",
                      "EMA50_DXY", "EMA200_DXY", "PriceChange_DXY"]

ANALISYS_INDICATORS = ["Open", "Close", "High", "Low", "Volume", "RSI",
                            "EMA12", "EMA26", "EMA50", "EMA200", "PriceChange",
                            "Close_DXY", "MACD", "MACD_Signal", "MACD_Histogram",
                            "MoveDirection"]
ANALISYS_INDICATORS_ONLY_MOVE_DIRECTION = ["MoveDirection"] # tylko MoveDirection

SCALER_TYPE = "minmax"    # "robust" - robust scaler
                            # "standard" - standard scaler
                            # "minmax" - MinMaxScaler

INDICATOR_SCALER_FILE = "./DataAndScalerFiles/indicatorScaler.pkl"
TARGET_SCALER_FILE = "./DataAndScalerFiles/targetScaler.pkl"


USE_FIBO = False    # True - liczenie fibo wlaczone (ale nie koniecznie uwzgledniane)
                    # False - liczenie fibo wyłączone

FIBO_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786]



#================================================================



# Parametry modelu
NUMBER_OF_NEURONS = 512
DROPOUT = 0.2
DENSE = 1
RETURN_SEQUENCES = True # True - dodaje druga warstwe,
                        # False - pozostaje jedna warstwa
NEGATIVE_SLOPE = 0.1
RE_LU = "tanh" # "Leaky" - LeakyReLu
                # "P" - PReLu
                # "_ReLu" - standardowy ReLu
                # "tanh" - tanh
HUBER_DELTA = 0.75
LOSS = "binary_crossentropy"  # "Huber" - Huber
                # "mse" - mse
                # "binary_crossentropy" - dla binarnej klasifikacji
OPTIMIZER = 'adamax' # "adam" - adam
                    # "adamax" - adamax


#================================================================



# Parametry uczenia
BATCH_SIZE = 128
EPOCHS = 100
INTERVALS_PREDICTION_FORWARD = 1  # Przewidywanie o jeden interwał do przodu
EARLY_STOP_PATIENCE = 20 # None - wyłączone



#================================================================



# Parametry wyniku
HOW_MANY_OUTPUTS = 100
TARGET_COLUMN = "MoveDirection"
IS_BINARY_PREDICTION = True
# Zbadac czy dla Close jest lepsze niz PriceChange
# Sprawdzic czy przewiduje sam znak
# Przesuwanie okienka czasowego

# Wykreslic jak przebiega PriceChange dla trenigowego i walidacynego


#Zrobic testy i porownac wyniki

# Skrocic caly zakres danych, skrocic dane testowe i walidacyjne