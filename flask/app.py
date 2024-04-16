from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GRU
import datetime

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict_stock():
    if request.method == 'POST':
        company = request.form['company']
        predicted_days = int(request.form['predicted_days'])
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        ticker = yf.Ticker(company)
        hist = ticker.history(start='2020-01-01', end=current_date)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(hist['Close'].values.reshape(-1, 1))
        x_train = []
        y_train = []
        for x in range(predicted_days, len(scaled_data)):
            x_train.append(scaled_data[x - predicted_days:x, 0])
            y_train.append(scaled_data[x, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        model = Sequential()
        model.add(GRU(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(GRU(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(units=50))
        model.add(Dropout(0.1))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=24, batch_size=64)
        x_test = []
        for x in range(predicted_days, len(scaled_data)):
            x_test.append(scaled_data[x - predicted_days:x, 0])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        next_month_start = datetime.datetime.now() + datetime.timedelta(days=1)
        next_month_end = datetime.datetime.now() + datetime.timedelta(days=predicted_days)
        future_dates = pd.date_range(start=next_month_start, end=next_month_end)
        predicted_prices = model.predict(x_test)
        predicted_prices = scaler.inverse_transform(predicted_prices)
        dates = pd.date_range(start=current_date, periods=len(predicted_prices))  # Variable agregada
        # Plot para los precios reales
        plt.plot(hist.index, hist['Close'], color="black", label="Precios reales")
        plt.xlabel("Fecha")
        plt.ylabel("Precio de cierre")
        plt.title("Precios reales de acciones hasta la fecha actual")
        plt.legend()
        plt.xticks(rotation=45, fontsize=8)  # Rotar las fechas y ajustar el tamaño de la fuente
        plt.tight_layout()  # Ajustar el espaciado automáticamente
        plt.savefig('static/precios_reales.png', transparent=True)
        plt.close()

        # Plot para los precios predichos
        predicted_prices = predicted_prices[-predicted_days:]
        plt.plot(future_dates, predicted_prices, color="blue")
        plt.xlabel("Fecha")
        plt.title(f"Precios predichos de acciones de {company} en los próximos {predicted_days} días")
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.xticks(rotation=45, fontsize=8)  # Rotar las fechas y ajustar el tamaño de la fuente
        plt.tight_layout()  # Ajustar el espaciado automáticamente
        plt.savefig('static/precios_predichos.png', transparent=True)
        plt.close()
        return redirect(url_for('show_result'))
    else:
        return render_template('index.html')

@app.route('/result')
def show_result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
