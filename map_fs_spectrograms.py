import os
import re
import base64
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# --- Configuration ---
dx = 3.19  # channel spacing in meters
total_length = 1920
image_dir = r"C:\Users\sjuber\results\decimator_2025-10-15_16.00.00_UTC_006816\decimator_2025-10-15_16.00.00_UTC_006816"

# Loop definitions
loop_starts = np.array([118, 147, 271, 400, 466, 510, 630, 1300, 1800])
loop_lengths = np.array([30, 60, 60, 30, 30, 30, 30, 60, 60])

x, y = [0], [0]
current_dist = 0
main_y = 0
loop_dir = 1

for next_loop_start, L in zip(loop_starts, loop_lengths):
    while current_dist + dx < next_loop_start:
        current_dist += dx
        x.append(current_dist)
        y.append(main_y)

    r = L / (2 * np.pi)
    theta = np.linspace(0, 2 * np.pi, int(L / dx))
    x_loop = current_dist + r * np.cos(theta)
    y_loop = main_y + loop_dir * r * np.sin(theta)
    x.extend(x_loop)
    y.extend(y_loop)
    current_dist += L
    loop_dir *= -1

while current_dist + dx <= total_length:
    current_dist += dx
    x.append(current_dist)
    y.append(main_y)

# --- Reverse direction ---
x = total_length - np.array(x)   # flip along x-axis
y = np.array(y)
coords = np.linspace(0, total_length, len(x))

# --- Load images ---
pattern = re.compile(r"spectrogram_ch([0-9_.]+)m\.png")
images, distances = [], []
for fname in sorted(os.listdir(image_dir)):
    match = pattern.search(fname)
    if match:
        dist = float(match.group(1).replace("_", "."))
        distances.append(dist)
        images.append(os.path.join(image_dir, fname))

distances = np.array(distances)

# --- Dash app ---
app = Dash(__name__)
app.layout = html.Div([
    html.H2("Interactive Cable Spectrogram Viewer"),
    html.Div([
        dcc.Graph(id='cable-map', style={'height': '70vh', 'width': '70vw'}),
        html.Div([
            html.H3(id='spec-title', style={'textAlign': 'center'}),
            html.Img(id='spectrogram', style={'height': '70vh', 'border': '1px solid #ccc'})
        ])
    ], style={'display': 'flex', 'gap': '1rem'}),
])

@app.callback(
    Output('spectrogram', 'src'),
    Output('spec-title', 'children'),
    Input('cable-map', 'clickData')
)
def show_image(clickData):
    if not clickData:
        return '', 'Click a point to view its spectrogram'
    idx = clickData['points'][0]['pointIndex']
    dist = coords[idx]
    nearest = np.abs(distances - dist).argmin()
    img_path = images[nearest]
    with open(img_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    title = f"Channel {idx} — {distances[nearest]:.2f} m"
    return f"data:image/png;base64,{encoded}", title

@app.callback(
    Output('cable-map', 'figure'),
    Input('spectrogram', 'src')
)
def update_map(_):
    fig = go.Figure(go.Scatter(
        x=x, y=y, mode='markers+lines',
        marker=dict(size=4, color='black'),
        hovertext=[f"Channel {i}, {c:.1f} m" for i, c in enumerate(coords)],
        hoverinfo='text'
    ))
    fig.update_layout(
        xaxis=dict(visible=False, autorange='reversed'),  # <— reverse axis here
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor='white',
    )
    return fig

if __name__ == "__main__":
    app.run(debug=True)
