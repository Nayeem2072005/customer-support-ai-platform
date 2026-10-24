"""
dashboard/prepare_data.py
=========================
Builds the small summary file that the Plotly Dash dashboard reads, from the ORIGINAL ticket
CSV. Only counts and averages are written (no names, e-mails or ticket text), so the dashboard
needs neither the raw data nor any model file.

Run from the project root:
    python dashboard/prepare_data.py
"""
import json
import os
import shutil

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV = os.path.join(BASE, "data_raw", "customer_support_tickets.csv")
RESULTS_CSV = os.path.join(BASE, "reports", "test_results_all_models.csv")
OUT_DIR = os.path.join(BASE, "dashboard", "data")
CSAT = "Customer Satisfaction Rating"


def to_records(frame):
    """DataFrame -> list of plain dicts (JSON friendly; NaN becomes null)."""
    return json.loads(frame.to_json(orient="records"))


def count_table(df, column):
    """Number of tickets per value of `column`."""
    table = df[column].value_counts().rename_axis("label").reset_index(name="tickets")
    return to_records(table)


def mean_table(df, column, value):
    """Average of `value` and number of rated tickets per value of `column`."""
    grouped = df.dropna(subset=[value]).groupby(column)[value].agg(["mean", "count"]).reset_index()
    grouped.columns = ["label", "mean", "tickets"]
    grouped["mean"] = grouped["mean"].round(3)
    return to_records(grouped)


def resolution_table(closed, column):
    """Resolution-time gap (hours) per value of `column`, for tickets with both timestamps."""
    grouped = closed.groupby(column).agg(
        mean_hours=("gap_hours", "mean"),
        median_hours=("gap_hours", "median"),
        tickets=("gap_hours", "size"),
        pct_before_first=("before_first", "mean"),
    ).reset_index().rename(columns={column: "label"})
    grouped["pct_before_first"] = grouped["pct_before_first"] * 100
    return to_records(grouped.round(2))


def main():
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(f"Original ticket file not found: {RAW_CSV}")
    if not os.path.exists(RESULTS_CSV):
        raise FileNotFoundError(f"Run the final-results cell of notebook 10 first: {RESULTS_CSV}")

    df = pd.read_csv(RAW_CSV)
    first = pd.to_datetime(df["First Response Time"], errors="coerce")
    resolved = pd.to_datetime(df["Time to Resolution"], errors="coerce")
    df["first_hour"] = first.dt.strftime("%Y-%m-%d %H:00")
    df["resolved_hour"] = resolved.dt.strftime("%Y-%m-%d %H:00")
    df["purchase_month"] = pd.to_datetime(df["Date of Purchase"], errors="coerce").dt.strftime("%Y-%m")
    signed_hours = (resolved - first).dt.total_seconds() / 3600
    df["gap_hours"] = signed_hours.abs()          # same target definition as notebook 09
    df["before_first"] = signed_hours < 0         # resolved BEFORE the first response (impossible)
    closed = df[signed_hours.notna()]

    trend = (df.dropna(subset=["first_hour"]).groupby(["first_hour", "Ticket Type"]).size()
             .reset_index(name="tickets").rename(columns={"first_hour": "hour", "Ticket Type": "type"}))
    rated = df.dropna(subset=[CSAT, "resolved_hour"])
    csat_trend = rated.groupby("resolved_hour")[CSAT].agg(["mean", "count"]).reset_index()
    csat_trend.columns = ["hour", "mean_csat", "tickets"]
    purchase = (df.dropna(subset=["purchase_month"]).groupby("purchase_month").size()
                .reset_index(name="tickets").rename(columns={"purchase_month": "month"}))
    csat_trend["mean_csat"] = csat_trend["mean_csat"].round(3)
    heat = df.dropna(subset=[CSAT]).groupby(["Ticket Type", "Ticket Channel"])[CSAT].agg(
        ["mean", "count"]).reset_index()
    heat.columns = ["type", "channel", "mean_csat", "tickets"]
    heat["mean_csat"] = heat["mean_csat"].round(3)
    distribution = (df[CSAT].dropna().astype(int).value_counts().sort_index()
                    .rename_axis("rating").reset_index(name="tickets"))

    summary = {
        "kpis": {
            "total_tickets": int(len(df)),
            "closed_tickets": int((df["Ticket Status"] == "Closed").sum()),
            "rated_tickets": int(df[CSAT].notna().sum()),
            "mean_csat": round(float(df[CSAT].mean()), 2),
            "ticket_types": int(df["Ticket Type"].nunique()),
            "channels": int(df["Ticket Channel"].nunique()),
            "tickets_with_first_response": int(first.notna().sum()),
            "first_response_first_day": str(first.min().date()),
            "first_response_last_day": str(first.max().date()),
            "closed_with_both_timestamps": int(len(closed)),
            "pct_resolved_before_first_response": round(float(closed["before_first"].mean() * 100), 1),
        },
        "volume_by_type": count_table(df, "Ticket Type"),
        "volume_by_priority": count_table(df, "Ticket Priority"),
        "volume_by_status": count_table(df, "Ticket Status"),
        "volume_by_channel": count_table(df, "Ticket Channel"),
        "volume_trend": to_records(trend),
        "tickets_by_purchase_month": to_records(purchase),
        "csat_distribution": to_records(distribution),
        "csat_by_type": mean_table(df, "Ticket Type", CSAT),
        "csat_by_channel": mean_table(df, "Ticket Channel", CSAT),
        "csat_by_priority": mean_table(df, "Ticket Priority", CSAT),
        "csat_heatmap": to_records(heat),
        "csat_trend": to_records(csat_trend),
        "resolution_by_channel": resolution_table(closed, "Ticket Channel"),
        "resolution_by_type": resolution_table(closed, "Ticket Type"),
        "resolution_by_priority": resolution_table(closed, "Ticket Priority"),
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    out_json = os.path.join(OUT_DIR, "summary.json")
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1)
    shutil.copy(RESULTS_CSV, os.path.join(OUT_DIR, "test_results_all_models.csv"))

    k = summary["kpis"]
    print(f"Tickets: {k['total_tickets']} | closed: {k['closed_tickets']} | rated: {k['rated_tickets']} "
          f"| mean CSAT: {k['mean_csat']}")
    print(f"Tickets with a first-response timestamp: {k['tickets_with_first_response']}")
    print(f"Closed tickets with both timestamps: {k['closed_with_both_timestamps']}; "
          f"{k['pct_resolved_before_first_response']}% resolved BEFORE the first response")
    hours = sorted(trend["hour"].unique())
    print(f"Ticket-volume hours: {len(hours)} ({hours[0] if hours else '-'} to {hours[-1] if hours else '-'})")
    print(f"Purchase months: {len(purchase)} ({purchase['month'].min()} to {purchase['month'].max()})")
    counts = ", ".join("{}: {}".format(r["rating"], r["tickets"]) for r in summary["csat_distribution"])
    print("CSAT ratings -> " + counts)
    print(f"Wrote {out_json} ({os.path.getsize(out_json) / 1024:.0f} KB) and the results CSV into {OUT_DIR}")


if __name__ == "__main__":
    main()
