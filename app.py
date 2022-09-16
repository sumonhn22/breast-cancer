import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('modelDL.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [list(int_features)]
    prediction = model.predict(final_features)
    # prediction = model.predict([[.2, .9, .6,.2,2,.3,.3,.3,.6,.5,   .4,.1,.4,.1,.4,.1,.7,.7,.8,.5,   .5, .5,.4,.4,.8,.8,.9,.6,.5,.4]])
    # output = prediction
    # output = round(prediction[0], 2)
    if prediction > 0:
        output="positive"
    else:
        output="negative"
   # print('safds',prediction)
    return render_template('index.html', prediction_text=f'Cancer prediction is {output}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)