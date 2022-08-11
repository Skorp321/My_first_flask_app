from flask import Flask, render_template, request
import plotly.express as px
import plotly.graph_objects as go
from joblib import load
import numpy as np
import pandas as pd

import plotly.io as pio
pio.kaleido.scope.chromium_args = tuple([arg for arg in pio.kaleido.scope.chromium_args if arg != "--disable-dev-shm-usage"])

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('index.html', href='static/static.svg')
    else:
        text = request.form['text']
        train_data_file = "AgesAndHeights.pkl"
        output_file = 'static/pred_file.svg'
        model = load('model.joblib')
        make_picture(train_data_file, model, floats_string_to_input_arr(text), output_file)
        return render_template('index.html', href=output_file)
    

def make_picture(train_data_file, model, new_points, output_file):
    
    x_new = np.linspace(0, 18, 10)
    data = pd.read_pickle(train_data_file)   
    data = data[data['Age'] > 0]
    ages = data['Age']
    height = data.Height    

    preds = model.predict(x_new.reshape(-1, 1))
    new_preds = model.predict(new_points.reshape(-1, 1))

    fig = px.scatter(x=ages, y=height, title='Height and Age of People',
                        labels={'x': 'Age (years)', 'y': 'Height (inches)'})

    fig.add_trace(go.Scatter(x=x_new, y=preds, mode='lines', name='Model'))   
    
    fig.add_trace(go.Scatter(x=new_points.squeeze(), y=new_preds, name='New Outputs', mode='markers',
                        marker=dict(color='purple', size=15, line=dict(color='purple', width=2))))
                        
    fig.write_image(output_file, width=800, engine="kaleido")                    
    fig.show()

def floats_string_to_input_arr(floats_str):
    def is_float(s):
        try:
            float(s)
            return True
        except:
            return False

    floats = [float(x) for x in floats_str.split(',') if is_float(x)]
    as_np_arr = np.array(floats)
    return as_np_arr    