# Import libraries
from flask import Flask
from dash import Dash, dcc, html
import plotly.graph_objs as go
import pandas as pd

# Load or define DataFrames
combined_results = pd.read_csv("combined_results.csv")
future_results = pd.read_csv("future_results.csv")

# Initialize the Flask app
server = Flask(__name__)

# Initialize the Dash app with Flask server
app = Dash(__name__, server=server)

# Define the layout for the app
app.layout = html.Div([
    # Add the graph component to display the plot
    dcc.Graph(
        id='predictions-plot',
        figure={
            'data': [
                go.Scatter(x=combined_results['Date'], y=combined_results['Train Predictions'], mode='lines', name='Train Predictions'),
                go.Scatter(x=combined_results['Date'], y=combined_results['Actuals'], mode='lines', name='Actuals'),
                go.Scatter(x=future_results['Date'], y=future_results['Predictions'], mode='lines', name='Future Predictions')
            ],
            'layout': go.Layout(
                title='Train Predictions, Actuals, and Future Predictions',
                xaxis={'title': 'Date'},
                yaxis={'title': 'Value'},
                margin={'l': 60, 'r': 10, 't': 60, 'b': 60},
                legend={'x': 0, 'y': 1},
                hovermode='closest'
            )
        }
    )
])

# Run the Dash app
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=True)