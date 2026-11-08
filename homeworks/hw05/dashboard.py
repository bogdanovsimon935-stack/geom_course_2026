import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

current_file = Path(__file__).resolve()
repo_root = current_file.parents[2] 

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from homeworks.hw01.data_utils import ROOT

CSV_PATH = ROOT / "homeworks" / "hw05" / "pressure_long.csv"

# Загружаем таблицу
df = pd.read_csv(CSV_PATH)
unique_times = sorted(df["time_h"].unique())

app = Dash(__name__)

app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif", "padding": "20px", "maxWidth": "800px", "margin": "0 auto"},
    children=[
        html.H1("Мониторинг поля давления", style={"textAlign": "center", "color": "#2c3e50"}),
        html.Div(id="selected-time-output", style={"fontSize": "18px", "fontWeight": "bold", "marginBottom": "15px", "color": "#16a085", "textAlign": "center"}),
        dcc.Graph(id="pressure-graph"),
        html.Div(
            style={"marginTop": "25px", "padding": "20px", "backgroundColor": "#f8f9fa", "borderRadius": "5px"},
            children=[
                html.Label("Выберите момент времени (индекс кадра):", style={"fontWeight": "bold", "marginBottom": "10px", "display": "block"}),
                dcc.Slider(
                    id="time-slider",
                    min=0,
                    max=len(unique_times) - 1,
                    value=0,
                    marks={i: f"#{i}" for i in range(len(unique_times))},
                    step=1
                ),
            ]
        )
    ]
)


@app.callback(
    [Output("pressure-graph", "figure"),
     Output("selected-time-output", "children")],
    [Input("time-slider", "value")]
)
def update_dashboard(selected_index):
    current_time = unique_times[selected_index]
    filtered_df = df[df["time_h"] == current_time].sort_values(by="radius_m")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered_df["radius_m"],
        y=filtered_df["pressure_mpa"],
        mode="lines+markers",
        line=dict(color="#1f77b4", width=3),
        marker=dict(size=10, color="#d62728"),
        text=filtered_df["well_id"],
        hovertemplate="<b>Скважина:</b> %{text}<br><b>Радиус:</b> %{x} м<br><b>Давление:</b> %{y:.3f} МПа<extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(text=f"Профиль давления при t = {current_time:.1f} ч", x=0.5),
        xaxis=dict(title="Расстояние, м", type="log", gridcolor="rgba(0,0,0,0.1)", zeroline=False),
        yaxis=dict(title="Давление, МПа", range=[df["pressure_mpa"].min() - 0.1, df["pressure_mpa"].max() + 0.1], gridcolor="rgba(0,0,0,0.1)", zeroline=False),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    
    return fig, f"Текущий кадр: #{selected_index} | Время: {current_time:.2f} ч"


if __name__ == "__main__":
        app.run(debug=True)

