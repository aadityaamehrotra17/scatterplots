import dash
from dash import dcc, html, Input, Output, State, ctx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import base64
import io

app = dash.Dash(__name__, assets_folder='assets')
app.title = "Scatterplot Manipulation Tool"

# Define the layout of the app
app.layout = html.Div([
    html.H1("SCATTERPLOT MANIPULATION TOOL", id='main-title', className='title'),
    html.Div([
        html.Label("Upload CSV Data:", id='csv-label', className='label'),
        dcc.Upload(id='upload-data', children=html.Button('Upload File', className='button')),
    ], id='csv-section', className='section'),
    html.Div([
        html.Label("Adjust Size Factor:", id='size-label', className='label'),
        dcc.Slider(id='size-slider', min=1, max=10, value=5, marks={i: str(i) for i in range(1, 11)}, className='slider'),
        dcc.Store(id='size-factor-store', data=5)
    ], id='size-section', className='section'),
    html.Div([
        html.Label("Mean For Random Data (Comma Separated): ", id='mean-label', className='label'),
        dcc.Input(id='mean-input', value='0,0', type='text', className='input-text'),
    ], id='mean-section', className='section'),
    html.Div([
        html.Label("Covariance Matrix For Random Data:", id='cov-label', className='label'),
        html.Div([
            dcc.Input(id='cov-input-00', type='number', value=1, step=0.1, className='input-number'),
            dcc.Input(id='cov-input-01', type='number', value=0.8, step=0.1, className='input-number'),
        ], className='matrix-row'),
        html.Div([
            dcc.Input(id='cov-input-10', type='number', value=0.8, step=0.1, className='input-number'),
            dcc.Input(id='cov-input-11', type='number', value=1, step=0.1, className='input-number'),
        ], className='matrix-row'),
    ], id='cov-section', className='section'),
    html.Div([
        html.Button("Generate Plot", id='generate-button', n_clicks=0, className='button'),
        html.Button("Export Plot As PNG", id='export-button', n_clicks=0, className='button'),
    ], id='buttons-section', className='section'),
    html.Div([
        html.Div([dcc.Graph(id='standard-plot', className='graph')], id='standard-plot-div', className='plot-div'),
        html.Div([dcc.Graph(id='scatterplot', className='graph')], id='scatterplot-div', className='plot-div'),
    ], id='plots-section', className='plots-section'),
    dcc.Download(id="download-image"),
], id='main-section', className='main-section')

@app.callback(
    Output('size-factor-store', 'data'),
    Input('size-slider', 'value')
)
def update_size_factor_store(slider_value):
    return slider_value

# Define the callback to update the plot
@app.callback(
    [Output('scatterplot', 'figure'),
     Output('standard-plot', 'figure')],
    [Input('generate-button', 'n_clicks'),
     Input('size-factor-store', 'data'),
     Input('upload-data', 'contents'),
     Input('mean-input', 'value'),
     Input('cov-input-00', 'value'),
     Input('cov-input-01', 'value'),
     Input('cov-input-10', 'value'),
     Input('cov-input-11', 'value')],
    prevent_initial_call=True
)
def update_plots(n_clicks, size_factor, uploaded_file, mean_input, cov_00, cov_01, cov_10, cov_11):
    if not ctx.triggered_id or ctx.triggered_id != 'generate-button':
        raise dash.exceptions.PreventUpdate

    # Load data from uploaded file or generate random data
    if uploaded_file:
        content_type, content_string = uploaded_file.split(',')
        decoded = base64.b64decode(content_string)
        data = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        if data.shape[1] < 2:
            raise ValueError("Uploaded CSV does not have at least two columns.")
        
        x = data.iloc[:, 0]  # Use the first column for x-axis
        y = data.iloc[:, 1]  # Use the second column for y-axis
    else:
        mean = list(map(float, mean_input.split(',')))
        cov = np.array([[cov_00, cov_01], [cov_10, cov_11]])
        np.random.seed(123)
        data = pd.DataFrame(np.random.multivariate_normal(mean, cov, 100), columns=['V1', 'V2'])
        x = data['V1']
        y = data['V2']
    
    # Calculate residuals
    slope, intercept, _, _, _ = linregress(x, y)
    predicted_y = slope * x + intercept
    residuals = y - predicted_y
    slope_inverted = (1 + (0.25)**np.abs(residuals)) - 1
    slope_inverted_floored = np.maximum(0.1, slope_inverted)
    sizes = size_factor * (slope_inverted_floored * 3)
    opacities = slope_inverted_floored

    # Create scatterplot with varying sizes and opacities, no border on points
    manipulated_fig = {
        'data': [{'x': x, 'y': y, 'mode': 'markers', 'marker': {'size': sizes, 'opacity': opacities, 'color': 'blue', 'line': {'width': 0}}}],
        'layout': {'title': 'Manipulated Scatterplot'}
    }

    # Standard scatterplot
    standard_fig = {
        'data': [{'x': x, 'y': y, 'mode': 'markers', 'marker': {'color': 'red', 'line': {'width': 0}}}],
        'layout': {'title': 'Standard Scatterplot'}
    }

    return [manipulated_fig, standard_fig]

# Export plot as PNG
@app.callback(
    Output("download-image", "data"),
    Input("export-button", "n_clicks"),
    prevent_initial_call=True
)
def export_plot_as_png(n_clicks):
    if not ctx.triggered_id or ctx.triggered_id != 'export-button':
        raise dash.exceptions.PreventUpdate

    buf = io.BytesIO()
    plt.figure(figsize=(8, 8))
    plt.scatter(np.random.randint(10, 100, 100), np.random.randint(10, 100, 100), alpha=0.5)
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    return dcc.send_bytes(buf.getvalue(), filename="scatterplot.png")

# Update CSV label when a file is uploaded
@app.callback(
    Output('csv-label', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_csv_label(contents, filename):
    if contents is not None:
        return f"Uploaded \"{filename}\""
    return "Upload CSV Data:"

# Update upload button text when a file is uploaded
@app.callback(
    Output('upload-data', 'children'),
    Input('upload-data', 'contents')
)
def update_upload_button(contents):
    if contents is not None:
        return html.Button('Re-Upload', className='button')
    return html.Button('Upload File', className='button')

if __name__ == '__main__':
    app.run_server(debug=True)
