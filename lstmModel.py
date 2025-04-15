import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.layers import LeakyReLU, PReLU
import matplotlib.pyplot as plt
from constants import (
    RETURN_SEQUENCES,
    HOW_MANY_OUTPUTS,
    TARGET_SCALER_FILE,
    NUMBER_OF_NEURONS,
    DENSE,
    DROPOUT,
    BATCH_SIZE,
    EPOCHS,
    ALPHA,
    RE_LU,
    SCALER_TYPE,
    DAYS_PREDICTION_FORWARD,
    LOSS,
    HUBER_DELTA
    )
from joblib import load
import numpy as np

class RNNLSTMModel:
    def __init__(self, XTrain, yTrain, XVal, yVal, XTest, yTest):

        self.XTrain, self.yTrain = XTrain, yTrain
        self.XVal, self.yVal = XVal, yVal
        self.XTest, self.yTest = XTest, yTest
        self.model = self.buildModel()


    def buildModel(self, numberOfNeurons=NUMBER_OF_NEURONS,
                   dropout=DROPOUT, dense=DENSE,
                   returnSequences = RETURN_SEQUENCES,
                   _alpha=ALPHA, reLu=RE_LU,
                   loss=LOSS, huber_delta=HUBER_DELTA):
        model = Sequential()

        # Pierwsza warstwa modelu
        model.add(LSTM(numberOfNeurons, return_sequences=returnSequences,
                       input_shape=(self.XTrain.shape[1], self.XTrain.shape[2])))
        model.add(Dropout(dropout))

        # Druga warstwa modelu w przypadku
        # wlaczenia return sequences
        if returnSequences is True:
            # Druga warstwa LSTM
            model.add(LSTM(numberOfNeurons // 2, return_sequences=True))
            model.add(Dropout(dropout))

            # Trzecia warstwa LSTM
            model.add(LSTM(numberOfNeurons // 2, return_sequences=False))
            model.add(Dropout(dropout))

        model.add(Dense(dense))
        # model.add(Dense(1, activation='sigmoid'))  # Dla danych binarnych 0/1

        # Wybierz ReLU
        if reLu == "_ReLu":
            pass
        elif reLu == "Leaky": 
            model.add(LeakyReLU(alpha=_alpha))

        elif reLu == "P":
            model.add(PReLU())
        
        # Wybierz jak obliczac loss
        if loss == "mse":
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        elif loss == "Huber":
            model.compile(optimizer='adam', loss=tf.keras.losses.Huber(delta=huber_delta), metrics=['mae'])
        elif loss == "binary_crossentropy":
            model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])
        else:
            raise NameError(f"Loss not recognized '{loss}'")
        
        return model
    
    def train(self, epochs=EPOCHS,
              batchSize=BATCH_SIZE):
        history = self.model.fit(self.XTrain, self.yTrain,
                                 epochs=epochs,
                                 batch_size=batchSize,
                                 validation_data=(self.XVal, self.yVal),
                                 verbose=1
                                 )
        self.plotLoss(history)

    def plotLoss(self, history):

        # Tworzymy 2 panele: wykres + parametry
        fig, (ax_loss, ax_params) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 1]})

        # Panel wykresu strat
        ax_loss.plot(history.history['loss'], label='Loss')
        ax_loss.plot(history.history['val_loss'], label='ValLoss')
        ax_loss.set_title("Training vs Validation Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend()
        ax_loss.grid(True)

        # Panel z parametrami modelu
        param_text = (
            f"Neurons: {NUMBER_OF_NEURONS}\n"
            f"Dropout: {DROPOUT}\n"
            f"Return Sequences: {RETURN_SEQUENCES}\n"
            f"ReLU type: {RE_LU}" + (f" (alpha={ALPHA})" if RE_LU == "Leaky" else "") + "\n"
            f"Batch Size: {BATCH_SIZE}\n"
            f"Epochs: {EPOCHS}\n"
            f"Loss: {LOSS}" + (f": (delta={HUBER_DELTA})" if LOSS == "Huber" else "") + "\n"
            f"Scaler: {SCALER_TYPE}\n"
            f"Prediction forward: +{DAYS_PREDICTION_FORWARD} day(s)"
        )
        ax_params.axis('off')  # Ukrywamy osie
        ax_params.text(0.01, 0.98, param_text, va='top', fontsize=10)

        plt.tight_layout()
        plt.show()

    def evaluate(self):
        mse, mae = self.model.evaluate(self.XTest, self.yTest)
        print(f"\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\nTest MSE: {mse:.4f}\nTest MAE: {mae:0.4f}\n!!!!!!!!!!!!!!!!!!!!!!!!!\n\n\n")

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