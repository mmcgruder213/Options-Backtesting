# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.


from dash import Dash, html, dcc, callback, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import scipy.stats as sc

app = Dash(__name__)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

app.layout = html.Section(children =[ 
    html.Div(id = 'top-header', 
            children = [
               html.H1('Options Backtracking Software'),
               html.Div('This software is designed to help you backtest options strategies', style = {'font-size' : '16px'})
            ] ,  style = {'text-align' : 'center', 'background-color' : 'lightgrey', 'padding' : '20px', 'margin' : '0 auto', 'width' : '100%'}),

    html.Div(id = 'main', children = 
             [
                dcc.Graph(id = 'options-graph', figure = fig)
             ]),

    html.Br(),

    html.Div(id = 'options-input', children = 
             [
                html.Div(['Call Strike: ', dcc.Input(id = 'call', type = 'number', placeholder = 'Call Strike', value = 1000)]),
                html.Br(),
                html.Div(['Put Strike: ', dcc.Input(id = 'put', type = 'number', placeholder = 'Put Strike' , value = 2000)]),
             ]),
], style = {'margin' : '0 auto', 'width' : '80%', 'height' : '80%', 'display' : 'flex', 'flex-direction' : 'column', 
            'flex-wrap' : 'wrap', 'justify-content' : 'center', 'background-color' : 'lightblue', 
            'font-family' : 'Arial, sans-serif', 'font-size' : '16px', 
            'color' : 'black', 'padding' : '20px', 'box-sizing' : 'border-box', 'text-align' : 'center',
            'align-items': 'center', 'align-content' : 'center'
            })


@callback(
    Output('options-graph', 'figure'),
    Input('call', 'value'),
    Input('put', 'value')

)
def update_graph(call, put):
    S = np.linspace(call-200, put+200, 300)
    r = 0.03
    sigma = 0.1

    ts = [0, 6, 12, 16.9999]
    portfolio = pd.DataFrame(S, columns = ['x'])
 
    for i, t in enumerate(ts):  
        portfolio[str(t)] = -10 * put_price(0.0402, 0.1158, 17/365, t/365, put, S) - 10 * call_price(0.0402, 0.0965, 17/365, t/365, call, S)

    fig = px.line(portfolio, x = 'x', y = portfolio.columns[1:], title = 'PnL vs Stock Price')
    return fig

def call_price(r, iv, T, t, K, St):
  d_plus = (np.log (St / K) + (r + iv ** 2 / 2) * (T - t)) / (iv * np.sqrt(T - t))
  d_minus = d_plus - iv * np.sqrt(T - t)
  return sc.norm.cdf(d_plus) * St - sc.norm.cdf(d_minus) * K * np.exp(-r * (T - t))

def put_price(r, iv, T, t, K, St):
  return K * np.exp(-r * (T - t)) - St + call_price(r, iv, T, t, K, St)

if __name__ == '__main__':
    app.run(debug=True)
