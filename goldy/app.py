from flask import Flask, render_template, request, redirect, url_for, flash, Response 
import sklearn
import pickle
app = Flask(__name__)

regressor = pickle.load(open('finalized_model.pkl', 'rb'))

@app.route('/prediction')
def index():
    SPX = request.args.get('SPX')
    USO = request.args.get('USO')
    SLV = request.args.get('SLV')
    EUR_USD = request.args.get('EUR_USD')

    SPX = float(SPX)
    USO = float(USO)
    SLV = float(SLV)
    EUR_USD = float(EUR_USD)

    result = regressor.predict([[SPX, USO, SLV, EUR_USD]])
    #return response as json
    res = {
        'result': result[0]
    }
    return res
