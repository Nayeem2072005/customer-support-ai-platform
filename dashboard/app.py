"""
dashboard/app.py - Plotly Dash dashboard
========================================
Brief: "Interactive Power BI / Plotly Dash dashboard: ticket volume by type, CSAT trends,
resolution time by channel, model performance KPIs."

It reads only the small files in dashboard/data/ (made by prepare_data.py), so it needs no raw
data and no model files, and it runs comfortably on a free web host.

Run locally:   python dashboard/app.py        then open http://127.0.0.1:8050
Production:    gunicorn --chdir dashboard app:server
"""
import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
with open(os.path.join(DATA, "summary.json"), encoding="utf-8") as handle:
    SUMMARY = json.load(handle)
RESULTS = pd.read_csv(os.path.join(DATA, "test_results_all_models.csv")).fillna("")

KPIS = SUMMARY["kpis"]
PRIORITY_ORDER = ["Critical", "High", "Medium", "Low"]
BASELINE = "Majority-class baseline"
BLUE, RED, GREY = "#2f6db5", "#c0392b", "#6b7280"

# K-Means segments (k = 7) with the business labels written in notebook 09.
SEGMENTS = pd.DataFrame(
    [
        (0, 260, "Satisfied chat & email regulars", "Chat 60%, Email 40%", 47, 4.2),
        (1, 282, "Dissatisfied phone & social-media customers", "Social 53%, Phone 42%", 46, 1.7),
        (2, 261, "Young phone & social-media users", "Social 52%, Phone 39%", 28, 3.1),
        (3, 302, "Most dissatisfied mixed-channel customers", "Email 38%, Chat 36%, Phone 20%", 48, 1.4),
        (4, 255, "Senior chat & email customers", "Chat 50%, Email 43%, Phone 7%", 58, 3.2),
        (5, 278, "Satisfied older phone & social-media customers", "Social 60%, Phone 40%", 56, 4.0),
        (6, 300, "Young multi-channel customers", "Email 41%, Chat 27%, Phone 25%", 29, 3.6),
    ],
    columns=["Segment", "Customers", "Business label", "Channel mix", "Avg age", "Avg CSAT"],
)


def frame(key):
    """A summary table from summary.json as a DataFrame."""
    return pd.DataFrame(SUMMARY[key])


def tidy(fig, title, height=380):
    fig.update_layout(title=title, template="plotly_white", height=height,
                      margin={"l": 40, "r": 20, "t": 60, "b": 40})
    return fig


def empty_figure(text):
    fig = go.Figure()
    fig.add_annotation(text=text, showarrow=False, font={"size": 14})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return tidy(fig, "")


# ---------------------------------------------------------------- figure builders
def fig_volume_by_type():
    fig = px.bar(frame("volume_by_type"), x="label", y="tickets", text="tickets",
                 color_discrete_sequence=[BLUE], labels={"label": "Ticket type", "tickets": "Tickets"})
    return tidy(fig, "Ticket volume by type")


def fig_volume_by_priority():
    df = frame("volume_by_priority")
    df["label"] = pd.Categorical(df["label"], PRIORITY_ORDER, ordered=True)
    fig = px.bar(df.sort_values("label"), x="label", y="tickets", text="tickets",
                 color_discrete_sequence=[BLUE], labels={"label": "Priority", "tickets": "Tickets"})
    return tidy(fig, "Ticket volume by priority")


def fig_status():
    fig = px.pie(frame("volume_by_status"), names="label", values="tickets", hole=0.45)
    return tidy(fig, "Ticket status")


def fig_channel_volume():
    fig = px.bar(frame("volume_by_channel"), x="label", y="tickets", text="tickets",
                 color_discrete_sequence=[BLUE], labels={"label": "Channel", "tickets": "Tickets"})
    return tidy(fig, "Ticket volume by channel")


def fig_trend(selected):
    df = frame("volume_trend")
    if df.empty:
        return empty_figure("No first-response dates in the data")
    if selected != "All":
        df = df[df["type"] == selected]
    df = df.copy()
    df["hour"] = pd.to_datetime(df["hour"])
    hourly = df.groupby("hour")["tickets"].sum()
    hourly = hourly.reindex(pd.date_range(hourly.index.min(), hourly.index.max(), freq="h"), fill_value=0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hourly.index, y=hourly.values, mode="lines+markers", name="tickets per hour",
                             line={"color": BLUE}, marker={"size": 3}))
    fig.add_trace(go.Scatter(x=hourly.index, y=hourly.rolling(12, min_periods=1).mean().values, mode="lines",
                             name="12-hour average", line={"color": RED, "dash": "dash"}))
    fig.update_layout(xaxis_title="Hour of first response", yaxis_title="Tickets")
    return tidy(fig, f"Tickets per hour: {selected}")


def fig_purchase_months():
    df = frame("tickets_by_purchase_month")
    fig = px.bar(df, x="month", y="tickets", color_discrete_sequence=[BLUE],
                 labels={"month": "Month the product was bought", "tickets": "Tickets"})
    return tidy(fig, "Tickets by month of product purchase")


def fig_csat_by(group):
    key = {"Ticket type": "csat_by_type", "Channel": "csat_by_channel", "Priority": "csat_by_priority"}[group]
    df = frame(key)
    fig = px.bar(df, x="label", y="mean", text="mean", hover_data=["tickets"],
                 color_discrete_sequence=[BLUE], labels={"label": group, "mean": "Average CSAT (1-5)"})
    fig.update_yaxes(range=[1, 5])
    return tidy(fig, f"Average satisfaction by {group.lower()}")


def fig_csat_distribution():
    fig = px.bar(frame("csat_distribution"), x="rating", y="tickets", text="tickets",
                 color_discrete_sequence=[BLUE], labels={"rating": "Rating (1-5)", "tickets": "Tickets"})
    return tidy(fig, "Satisfaction ratings given")


def fig_csat_heatmap():
    df = frame("csat_heatmap")
    matrix = df.pivot(index="type", columns="channel", values="mean_csat")
    fig = px.imshow(matrix, text_auto=".2f", color_continuous_scale="RdYlGn", zmin=1, zmax=5,
                    labels={"color": "Average CSAT"}, aspect="auto")
    return tidy(fig, "Average satisfaction: ticket type x channel (scale fixed at 1 to 5)")


def fig_csat_trend():
    df = frame("csat_trend")
    if df.empty:
        return empty_figure("No rated tickets with a resolution date")
    df["hour"] = pd.to_datetime(df["hour"])
    df = df.set_index("hour").reindex(pd.date_range(df["hour"].min(), df["hour"].max(), freq="h"))
    df["tickets"] = df["tickets"].fillna(0)
    weighted = (df["mean_csat"].fillna(0) * df["tickets"]).rolling(12, min_periods=1).sum()
    rolling = weighted / df["tickets"].rolling(12, min_periods=1).sum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["mean_csat"], mode="lines+markers", name="average CSAT per hour",
                             line={"color": BLUE}, marker={"size": 3}, connectgaps=True))
    fig.add_trace(go.Scatter(x=df.index, y=rolling, mode="lines", name="12-hour average",
                             line={"color": RED, "dash": "dash"}))
    fig.update_yaxes(range=[1, 5])
    fig.update_layout(xaxis_title="Hour of resolution", yaxis_title="Average CSAT")
    return tidy(fig, "Satisfaction trend (by hour of resolution)")


def fig_resolution(group, stat):
    key = {"Channel": "resolution_by_channel", "Ticket type": "resolution_by_type",
           "Priority": "resolution_by_priority"}[group]
    column = {"Average": "mean_hours", "Median": "median_hours"}[stat]
    df = frame(key)
    fig = px.bar(df, x="label", y=column, text=column, hover_data=["tickets", "pct_before_first"],
                 color_discrete_sequence=[BLUE], labels={"label": group, column: f"{stat} gap (hours)"})
    return tidy(fig, f"{stat} gap between first response and resolution, by {group.lower()} (hours)")


def model_rows(task):
    return RESULTS[RESULTS["task"] == task]


def fig_models(task, metric):
    rows = model_rows(task)
    models = rows[rows["model"] != BASELINE]
    baseline = rows[rows["model"] == BASELINE]
    fig = px.bar(models, x="model", y=metric, text=models[metric].round(3), color_discrete_sequence=[BLUE],
                 labels={"model": "Model", metric: metric.replace("_", " ")})
    if not baseline.empty:
        fig.add_hline(y=float(baseline[metric].iloc[0]), line_dash="dash", line_color=RED,
                      annotation_text="majority-class baseline", annotation_position="top left")
    return tidy(fig, f"{task}: test-set {metric.replace('_', ' ')}", height=420)


# ---------------------------------------------------------------- layout pieces
CARD = {"flex": "1", "minWidth": "170px", "background": "#f3f6fb", "borderRadius": "10px",
        "padding": "14px 18px", "margin": "6px"}
NOTE = {"background": "#fff6e5", "border": "1px solid #f0c36d", "borderRadius": "8px", "padding": "10px 14px",
        "margin": "10px 6px", "fontSize": "14px"}
ROW = {"display": "flex", "flexWrap": "wrap"}
HALF = {"flex": "1", "minWidth": "420px"}


def kpi(title, value, note=""):
    return html.Div([html.Div(title, style={"fontSize": "13px", "color": GREY}),
                     html.Div(value, style={"fontSize": "28px", "fontWeight": "700"}),
                     html.Div(note, style={"fontSize": "12px", "color": GREY})], style=CARD)


def note(text):
    return html.Div(text, style=NOTE)


def best_vs_baseline(task):
    rows = model_rows(task)
    best = rows[rows["model"] != BASELINE].sort_values("accuracy", ascending=False).iloc[0]
    base = float(rows[rows["model"] == BASELINE]["accuracy"].iloc[0])
    return best, base


type_best, type_base = best_vs_baseline("Ticket type")
prio_best, prio_base = best_vs_baseline("Ticket priority")
TYPES = [row["label"] for row in SUMMARY["volume_by_type"]]

overview_tab = [
    html.Div([
        kpi("Tickets", f"{KPIS['total_tickets']:,}", "Kaggle customer-support dataset"),
        kpi("Closed", f"{KPIS['closed_tickets']:,}", f"{KPIS['closed_tickets'] / KPIS['total_tickets']:.0%} of tickets"),
        kpi("Average CSAT", f"{KPIS['mean_csat']}", f"{KPIS['rated_tickets']:,} rated tickets (1 to 5)"),
        kpi("Ticket types", f"{KPIS['ticket_types']}", "the brief lists 4; the data has 5"),
        kpi("Channels", f"{KPIS['channels']}", "Chat, Email, Phone, Social media"),
    ], style=ROW),
    html.Div([html.Div(dcc.Graph(figure=fig_volume_by_type()), style=HALF),
              html.Div(dcc.Graph(figure=fig_volume_by_priority()), style=HALF)], style=ROW),
    html.Div([html.Div(dcc.Graph(figure=fig_status()), style=HALF),
              html.Div(dcc.Graph(figure=fig_channel_volume()), style=HALF)], style=ROW),
]

volume_tab = [
    note("The data has no 'ticket created' date. Tickets are placed on the date of their first response, "
         "so only tickets with a first-response timestamp appear in the trend "
         f"({KPIS['tickets_with_first_response']:,} of {KPIS['total_tickets']:,}). "
         f"Those timestamps cover only {KPIS['first_response_first_day']} to {KPIS['first_response_last_day']}, "
         "so the trend is shown per hour."),
    html.Label("Ticket type", style={"margin": "6px"}),
    dcc.Dropdown(id="trend-type", options=["All"] + TYPES, value="All", clearable=False,
                 style={"maxWidth": "320px", "margin": "6px"}),
    dcc.Graph(id="trend-chart", figure=fig_trend("All")),
    note("Date of purchase is when the customer bought the product, not when the ticket was raised, so this "
         "chart shows which purchase months the tickets come from."),
    dcc.Graph(figure=fig_purchase_months()),
]

_shares = frame("csat_distribution")["tickets"] / frame("csat_distribution")["tickets"].sum() * 100
csat_tab = [
    note(f"Each rating from 1 to 5 was given by {_shares.min():.1f}% to {_shares.max():.1f}% of the "
         f"{KPIS['rated_tickets']:,} rated tickets, an almost perfectly even spread. Real satisfaction scores are "
         "rarely this flat, which is one more sign that this dataset is synthetic, so differences between the "
         "groups below are small and should not be over-read."),
    html.Label("Average satisfaction by", style={"margin": "6px"}),
    dcc.Dropdown(id="csat-by", options=["Ticket type", "Channel", "Priority"], value="Ticket type",
                 clearable=False, style={"maxWidth": "320px", "margin": "6px"}),
    html.Div([html.Div(dcc.Graph(id="csat-by-chart", figure=fig_csat_by("Ticket type")), style=HALF),
              html.Div(dcc.Graph(figure=fig_csat_distribution()), style=HALF)], style=ROW),
    html.Div([html.Div(dcc.Graph(figure=fig_csat_heatmap()), style=HALF),
              html.Div(dcc.Graph(figure=fig_csat_trend()), style=HALF)], style=ROW),
]

resolution_tab = [
    note(f"Read with care: 'resolution time' here is the absolute gap between the first-response and resolution "
         f"timestamps. {KPIS['pct_resolved_before_first_response']}% of the {KPIS['closed_with_both_timestamps']:,} "
         "closed tickets have a resolution time BEFORE their first response, which is impossible in real "
         "operations, so these timestamps look synthetic and the differences below are not real-world performance."),
    html.Div([
        html.Div([html.Label("Break down by"),
                  dcc.Dropdown(id="res-by", options=["Channel", "Ticket type", "Priority"], value="Channel",
                               clearable=False)], style={"width": "260px", "margin": "6px"}),
        html.Div([html.Label("Statistic"),
                  dcc.RadioItems(id="res-stat", options=["Average", "Median"], value="Average", inline=True)],
                 style={"margin": "6px"}),
    ], style=ROW),
    dcc.Graph(id="resolution-chart", figure=fig_resolution("Channel", "Average")),
]

model_tab = [
    html.Div([
        kpi("Ticket type: best test accuracy", f"{type_best['accuracy']:.1%}",
            f"{type_best['model']} | majority-class baseline {type_base:.1%}"),
        kpi("Ticket priority: best test accuracy", f"{prio_best['accuracy']:.1%}",
            f"{prio_best['model']} | majority-class baseline {prio_base:.1%}"),
        kpi("Resolution time: test R squared", "-0.02", "MAE 4.86 h | no better than predicting the average"),
    ], style=ROW),
    note("Every model scores at about chance level on held-out test data (ROC-AUC between 0.49 and 0.50). The dataset "
         "looks synthetic, so the labels carry almost no signal. The dashed line is a model that always guesses the "
         "most common class."),
    html.Div([
        html.Div([html.Label("Task"),
                  dcc.Dropdown(id="model-task", options=["Ticket type", "Ticket priority"], value="Ticket type",
                               clearable=False)], style={"width": "260px", "margin": "6px"}),
        html.Div([html.Label("Metric"),
                  dcc.Dropdown(id="model-metric", options=["accuracy", "f1_macro", "f1_weighted", "roc_auc"],
                               value="accuracy", clearable=False)], style={"width": "260px", "margin": "6px"}),
    ], style=ROW),
    dcc.Graph(id="model-chart", figure=fig_models("Ticket type", "accuracy")),
    dash_table.DataTable(
        id="model-table",
        columns=[{"name": c, "id": c} for c in ["model", "accuracy", "f1_macro", "f1_weighted", "roc_auc", "note"]],
        data=model_rows("Ticket type").round(4).to_dict("records"),
        style_cell={"textAlign": "left", "padding": "6px", "fontFamily": "Segoe UI, Arial"},
        style_header={"fontWeight": "700"}, page_size=12),
]

segment_tab = [
    note("K-Means on four customer-profile features (age, channel, product, satisfaction), k = 7. Silhouette 0.214 "
         "and Davies-Bouldin 1.24 mean the segments overlap a lot, so they are descriptive groupings."),
    dcc.Graph(figure=tidy(px.bar(SEGMENTS, x="Segment", y="Customers", color="Avg CSAT", text="Customers",
                                 color_continuous_scale="RdYlGn", range_color=[1, 5],
                                 hover_data=["Business label"]), "Customers per segment, coloured by average CSAT")),
    dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in SEGMENTS.columns], data=SEGMENTS.to_dict("records"),
        style_cell={"textAlign": "left", "padding": "6px", "fontFamily": "Segoe UI, Arial", "whiteSpace": "normal"},
        style_header={"fontWeight": "700"}),
]

app = Dash(__name__, title="Customer Support Intelligence Dashboard")
server = app.server  # used by gunicorn on the web host
app.layout = html.Div([
    html.H2("AI-Powered Customer Support Intelligence Platform"),
    html.Div("GUVI x HCL capstone. Ticket analytics, satisfaction, resolution time and model performance.",
             style={"color": GREY, "marginBottom": "8px"}),
    dcc.Tabs([
        dcc.Tab(label="Overview", children=overview_tab),
        dcc.Tab(label="Ticket volume", children=volume_tab),
        dcc.Tab(label="Customer satisfaction", children=csat_tab),
        dcc.Tab(label="Resolution time", children=resolution_tab),
        dcc.Tab(label="Model performance", children=model_tab),
        dcc.Tab(label="Customer segments", children=segment_tab),
    ]),
], style={"maxWidth": "1250px", "margin": "0 auto", "padding": "16px", "fontFamily": "Segoe UI, Arial, sans-serif"})


# ---------------------------------------------------------------- interactivity
@app.callback(Output("trend-chart", "figure"), Input("trend-type", "value"))
def update_trend(selected):
    return fig_trend(selected)


@app.callback(Output("csat-by-chart", "figure"), Input("csat-by", "value"))
def update_csat_by(group):
    return fig_csat_by(group)


@app.callback(Output("resolution-chart", "figure"), Input("res-by", "value"), Input("res-stat", "value"))
def update_resolution(group, stat):
    return fig_resolution(group, stat)


@app.callback(Output("model-chart", "figure"), Output("model-table", "data"),
              Input("model-task", "value"), Input("model-metric", "value"))
def update_models(task, metric):
    return fig_models(task, metric), model_rows(task).round(4).to_dict("records")


if __name__ == "__main__":
    app.run(debug=False, port=8050)
