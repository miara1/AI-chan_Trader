import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from constants import HOW_MANY_OUTPUTS, TARGET_SCALER_FILE, NUMBER_OF_NEURONS, DENSE, DROPOUT, BATCH_SIZE, EPOCHS
from joblib import load
import numpy as np

class RNNLSTMModel:
    def __init__(self, XTrain, yTrain, XTest, yTest):

        self.XTrain, self.yTrain = XTrain, yTrain
        self.XTest, self.yTest = XTest, yTest
        self.model = self.buildModel()


    def buildModel(self, numberOfNeurons=NUMBER_OF_NEURONS,
                   dropout=DROPOUT, dense=DENSE):
        model = Sequential()
        model.add(LSTM(NUMBER_OF_NEURONS, return_sequences=False, input_shape=(self.XTrain.shape[1], self.XTrain.shape[2])))
        model.add(Dropout(DROPOUT))
        model.add(Dense(DENSE))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def train(self, epochs=EPOCHS,
              batchSize=BATCH_SIZE):
        history = self.model.fit(self.XTrain, self.yTrain,
                                 epochs=epochs,
                                 batch_size=batchSize,
                                 validation_data=(self.XTest, self.yTest),
                                 verbose=1
                                 )
        self.plotLoss(history)

    def plotLoss(self, history):
        plt.plot(history.history['loss'], label='Loss')
        plt.plot(history.history['val_loss'], label='ValLoss')
        plt.legend()
        plt.title("Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.grid(True)
        plt.show()

    def evaluate(self):
        mse, mae = self.model.evaluate(self.XTest, self.yTest)
        print(f"Test MSE: {mse:.4f}\nTest MAE: {mae:0.4f}")

    def predict(self):
        return self.model.predict(self.XTest)
    
    def printPredictionsVsActual(self, howMany=HOW_MANY_OUTPUTS):
        predictions = self.model.predict(self.XTest)

        try:
            targetScaler = load(TARGET_SCALER_FILE)
            realPredictions = targetScaler.inverse_transform(predictions)
            realYTest = targetScaler.inverse_transform(self.yTest.reshape(-1, 1))
        except FileNotFoundError:
            print("Scaler file not found! Showing scaled values!")
            realPredictions = predictions
            realYTest = self.yTest.reshape(-1, 1)

        print(f"Predictions Vs Real ({howMany} first):")
        for pred, real in zip(realPredictions[:howMany], realYTest[:howMany]):
            print(f"Pred: {pred[0]:+6.3f}%, Real: {real[0]:+6.3f}%")

    def evaluateDirectionAccuracy(self):
        predictions = self.model.predict(self.XTest)

        try:
            targetScaler = load(TARGET_SCALER_FILE)
            realPredictions = targetScaler.inverse_transform(predictions)
            realYTest = targetScaler.inverse_transform(self.yTest.reshape(-1, 1))
        except FileNotFoundError:
            print("Scaler file not found! Showing scaled values!")
            realPredictions = predictions
            realYTest = self.yTest.reshape(-1, 1)

        # Oblicz trafnosc kierunku
        predSigns = np.sign(realPredictions)
        trueSigns = np.sign(realYTest)
        correct = np.sum(predSigns == trueSigns)
        total = len(realYTest)
        accuracy = correct / total

        print(f"Direction accuracy: {accuracy * 100:.2f}% ({correct}/{total})")
        return accuracy