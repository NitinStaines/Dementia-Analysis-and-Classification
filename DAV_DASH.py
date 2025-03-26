
# Import libraries
import os
import numpy as np
import pandas as pd
import cv2
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.figure_factory as ff
from scipy import stats
from sklearn.linear_model import LinearRegression

# Define the path to your main dataset folder
dataset_path = r'C:\Users\nitin\OneDrive\Desktop\dav\Data'  # Replace with your actual path

# Initialize the app
app = Dash(__name__)

# Load labels and pre-compute statistics without loading images
image_statistics = {}
for label in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, label)
    
    if os.path.isdir(folder_path):  # Ensure it's a directory
        pixel_means = []
        for image_name in os.listdir(folder_path)[:50]:  # Load only the first 50 images
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                image = cv2.resize(image, (128, 128))
                pixel_means.append(image.mean())
        
        pixel_means_array = np.array(pixel_means)
        if pixel_means_array.size > 0:
            mode_result = stats.mode(pixel_means_array)
            mode_val = mode_result.mode if mode_result.mode.size > 0 else None
        else:
            mode_val = None

        image_statistics[label] = {
            'mean': np.mean(pixel_means_array) if pixel_means_array.size > 0 else None,
            'median': np.median(pixel_means_array) if pixel_means_array.size > 0 else None,
            'mode': mode_val[0] if mode_val is not None and isinstance(mode_val, np.ndarray) and mode_val.size > 0 else mode_val,
            'std_dev': np.std(pixel_means_array) if pixel_means_array.size > 0 else None,
            'pixel_means': pixel_means_array.tolist()
        }

# Debugging: Print `image_statistics` to confirm data loading


app.layout = html.Div([
    html.H1("Dementia MRI Image Analysis Dashboard"),

    dcc.Dropdown(id='class_dropdown', options=[{'label': cls, 'value': cls} for cls in image_statistics.keys()],
                 value=list(image_statistics.keys())[0]),

    html.Div(id='stats_output'),
    dcc.Graph(id='intensity_histogram'),
    dcc.Graph(id='intensity_boxplot'),
    html.Div(id='anomaly_output'),
    dcc.Graph(id='regression_scatter'),
    dcc.Graph(id='correlation_matrix')
])

# Callback for descriptive statistics based on selected class
@app.callback(
    Output('stats_output', 'children'),
    Input('class_dropdown', 'value')
)
def update_stats(selected_class):
    stats = image_statistics[selected_class]
      # Debugging print
    return html.Div([
        html.H4(f"Descriptive Statistics for {selected_class}"),
        html.P(f"Mean: {stats['mean']:.2f}" if stats['mean'] is not None else "Mean: N/A"),
        html.P(f"Median: {stats['median']:.2f}" if stats['median'] is not None else "Median: N/A"),
        html.P(f"Mode: {stats['mode']:.2f}" if stats['mode'] is not None else "Mode: N/A"),
        html.P(f"Standard Deviation: {stats['std_dev']:.2f}" if stats['std_dev'] is not None else "Standard Deviation: N/A")
    ])

# Callback for interactive histogram
@app.callback(
    Output('intensity_histogram', 'figure'),
    Input('class_dropdown', 'value')
)
def update_histogram(selected_class):
    pixel_means = image_statistics[selected_class]['pixel_means']
     # Debugging print
    fig = px.histogram(x=pixel_means, nbins=30, title=f"Histogram of Pixel Intensities - {selected_class}")
    return fig

# Callback for interactive box plot
@app.callback(
    Output('intensity_boxplot', 'figure'),
    Input('class_dropdown', 'value')
)
def update_boxplot(selected_class):
    pixel_means = image_statistics[selected_class]['pixel_means']
    print("Updating boxplot for:", selected_class, pixel_means)  # Debugging print
    fig = px.box(y=pixel_means, title=f"Box Plot of Pixel Intensities - {selected_class}")
    return fig

# Callback for anomaly detection output
@app.callback(
    Output('anomaly_output', 'children'),
    Input('class_dropdown', 'value')
)
def update_anomalies(selected_class):
    pixel_means = image_statistics[selected_class]['pixel_means']
    print("Updating anomalies for:", selected_class, pixel_means)  # Debugging print
    z_scores = (np.array(pixel_means) - np.mean(pixel_means)) / np.std(pixel_means)
    anomalies = sum(abs(z_scores) > 3)
    return html.Div([
        html.H4(f"Anomalies for {selected_class}"),
        html.P(f"Number of anomalies detected: {anomalies}")
    ])

# Callback for regression analysis scatter plot
@app.callback(
    Output('regression_scatter', 'figure'),
    Input('class_dropdown', 'value')
)
def update_regression(selected_class):
    pixel_means = np.array(image_statistics[selected_class]['pixel_means'])
    X = pixel_means.reshape(-1, 1)
    y = np.random.rand(len(pixel_means)) * 100  # Random output
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    fig = px.scatter(x=X.flatten(), y=y, title="Linear Regression Analysis", labels={'x': 'Pixel Means', 'y': 'Output'})
    fig.add_scatter(x=X.flatten(), y=y_pred, mode='lines', name='Prediction', line=dict(color='red'))
    
    return fig

# Callback for correlation matrix heatmap
@app.callback(
    Output('correlation_matrix', 'figure'),
    Input('class_dropdown', 'value')
)
def update_correlation(selected_class):
    pixel_means = image_statistics[selected_class]['pixel_means']
    df = pd.DataFrame({'pixel_means': pixel_means, 'dummy_variable': np.random.rand(len(pixel_means))})
    correlation_matrix = df.corr()
    fig = ff.create_annotated_heatmap(z=correlation_matrix.values, 
                                       x=list(correlation_matrix.columns), 
                                       y=list(correlation_matrix.index),
                                       colorscale='Viridis')
     # Debugging print
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
