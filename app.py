import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
reliance = pickle.load(open('rel.pkl', 'rb'))
tcs = pickle.load(open('tcs.pkl','rb'))
maruti = pickle.load(open('mar.pkl','rb'))

# op = reliance.predict([[2085.00,2167.80,2081.45]])
# print(op)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/Reliance',methods=['POST','GET'])
def predict_reliance():
    if request.method =='POST':
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = reliance.predict(final_features)

        # output = round(prediction[0][0], 2) rel
        prediction = prediction[0][0]#tcs,maruti
        output = round(prediction, 2)
        return render_template('reliance.html', prediction_text='Predicted Closing Price is {}'.format(output))
    else:
        return render_template('reliance.html')

@app.route('/Maruti',methods=['POST','GET'])
def predict_maruti():
    if request.method =='POST':
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = maruti.predict(final_features)

        # output = round(prediction[0][0], 2) rel
        prediction = prediction[0]#tcs,maruti
        output = round(prediction, 2)
        return render_template('maruti.html', prediction_text='Predicted Closing Price is {}'.format(output))
    else:
        return render_template('maruti.html')

@app.route('/TCS',methods=['POST','GET'])
def predict_tcs():
    if request.method =='POST':
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = tcs.predict(final_features)

        # output = round(prediction[0][0], 2) rel
        prediction = prediction[0]#tcs,maruti
        output = round(prediction, 2)
        return render_template('tcs.html', prediction_text='Predicted Closing Price is {}'.format(output))
    else:
        return render_template('tcs.html')
if __name__ == "__main__":
    app.run(debug=True)