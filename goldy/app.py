from flask import Flask, request, jsonify
from datetime import datetime
import yfinance as yf
import pickle

app = Flask(__name__)

linear = pickle.load(open('regression_linear_model.pkl', 'rb'))
@app.route('/prediction', methods=['GET'])
def get_prediction():
    # Mendapatkan tanggal input dari query parameter
    date_str = request.args.get('date')
    date = datetime.strptime(date_str, '%Y-%m-%d').date()

    # Mengambil data menggunakan yfinance
    data = yf.download('GLD', '2008-06-01', date, auto_adjust=True)
    data['S_3'] = data['Close'].rolling(window=3).mean()
    data['S_9'] = data['Close'].rolling(window=9).mean()
    data = data.dropna()

    # Melakukan prediksi hanya untuk data terakhir
    last_data = data.tail(1)
    predicted_price = linear.predict([[last_data['S_3'].iloc[0], last_data['S_9'].iloc[0]]])[0]
    signal = 'Buy' if predicted_price > last_data['Close'].iloc[0] else 'No Position'

    response = {
        'predicted_price': predicted_price,
        'signal': signal
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run()