#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


# In[9]:


app = Flask(__name__)
pkl_model = pickle.load(open('housing_model.pkl', 'rb'))
pkl_scale = pickle.load(open('housing_scale.pkl', 'rb'))


# In[ ]:


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    #print('Response', final_features)
    final_features = pkl_scale.scale_[1:]*(final_features)[0]
    prediction = pkl_model.predict([final_features])
    prediction = prediction*1/pkl_scale.scale_[0]
    output = str(round(prediction[0], 2))

    return render_template('index.html', prediction_text='Housing price should be INR {}'.format(output))


if __name__ == "__main__":
    app.run(debug=False )


# In[ ]:




