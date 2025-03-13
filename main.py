import cryptocompare as cc
import datetime
import csv
import os

CRYPTOCOMPARE_API_KEY = 'fd5ae8d2c2fc4786bcf5fb8b3c1ac2deefc35f26a163cf4b4cfcaddbd5b7e6c7'
dayCsvFileName = "dayCrypto.csv"
cc.cryptocompare._set_api_key_parameter(CRYPTOCOMPARE_API_KEY)


# Read market daily data and write it into a file 
def readMarketDayData(csvFileName = "noName.csv", toTimeStamp = None,
                      assetName = 'BTC', currency = 'USDT', exchange = 'Binance', limit = 1):
    
    if toTimeStamp is None:
        toTimeStamp = int(datetime.datetime.now().timestamp())

    historicalDayData = cc.get_historical_price_day(coin = assetName, currency = currency, limit = limit,
                                                toTs = toTimeStamp, exchange = exchange )
    
    with open(csvFileName, "a", newline="", encoding = "utf-8") as file:
        writer = csv.writer(file)

        fileIsEmpty = os.path.getsize(csvFileName) == 0

        if fileIsEmpty:
            writer.writerow(["Date", "Open", "High", "Low", "Close", "Volume From", "Volume To"])

        for entry in historicalDayData:
            date = datetime.datetime.fromtimestamp(entry['time']).strftime('%d.%m.%Y')
            writer.writerow([date, entry['open'], entry['high'], entry['low'], entry['close'], entry['volumefrom'], entry['volumeto']])

# function for resetting the file
def resetFile(fileName):
    if os.path.exists(fileName):
        os.remove(fileName)

    open(fileName, "w").close()



# Test market reading
resetFile(dayCsvFileName)
readMarketDayData(csvFileName = dayCsvFileName, limit = 365)
readMarketDayData(csvFileName = dayCsvFileName, limit = 365, toTimeStamp = datetime.datetime(2024, 3, 13).timestamp())