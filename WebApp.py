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
    return render_template('layout.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    # Loads the inputs from UI
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    # Scaling the input features- MinMaxScaling
    final_features = pkl_scale.scale_[1:]*(final_features)[0]
    
    # Using the trained model, prediction is done
    prediction = pkl_model.predict([final_features])
    
    # Scaling the output back to input scale for interpretation
    prediction = prediction*1/pkl_scale.scale_[0]
    
    output = str(round(prediction[0], 2))

    return render_template('layout.html', prediction_text='Housing price should be INR {}'.format(output))


if __name__ == "__main__":
    app.run(debug=False )


# In[ ]:




