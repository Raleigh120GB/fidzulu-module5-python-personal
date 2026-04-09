import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional


def price_time_series_figure(
    train_df: pd.DataFrame,
    title: str = "Price by Product",
    width: int = 900,
    height: int = 500,
) -> go.Figure:
    """Create a Plotly line chart of `base_price` over `start_date` for each product.

    - Coerces `start_date` to datetime and `base_price` to numeric.
    - Adds one trace per `prod_id` with a unique color.
    - Returns a `go.Figure` ready to show or save.
    Prints brief verification information.
    """
    if train_df is None or train_df.empty:
        raise ValueError("train_df must be a non-empty DataFrame")

    df = train_df.copy()

    # Coerce types safely
    if "start_date" in df.columns:
        before_dtype = df["start_date"].dtype
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    else:
        raise KeyError("train_df must contain a 'start_date' column")

    if "base_price" in df.columns:
        before_price_dtype = df["base_price"].dtype
        df["base_price"] = pd.to_numeric(df["base_price"], errors="coerce")
    else:
        raise KeyError("train_df must contain a 'base_price' column")

    print(f"[plotting] start_date coerced from {before_dtype} to {df['start_date'].dtype}; base_price coerced from {before_price_dtype} to {df['base_price'].dtype}")

    colors = px.colors.qualitative.Plotly
    fig = go.Figure()

    prod_ids = sorted(df["prod_id"].dropna().unique())
    for i, prod in enumerate(prod_ids):
        grp = df[df["prod_id"] == prod].sort_values("start_date")
        fig.add_trace(
            go.Scatter(
                x=grp["start_date"],
                y=grp["base_price"],
                mode="lines+markers",
                name=str(prod),
                line=dict(color=colors[i % len(colors)]),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="start_date",
        yaxis_title="base_price",
        width=width,
        height=height,
    )

    print(f"[plotting] created figure with {len(prod_ids)} traces")
    return fig


if __name__ == "__main__":
    # quick demo to produce an HTML file for manual inspection
    print("plotting demo starting...")
    sample = pd.DataFrame(
        {
            "prod_id": [1, 1, 1, 2, 2, 2],
            "start_date": [
                "2020-01-01",
                "2020-02-01",
                "2020-03-01",
                "2020-01-15",
                "2020-02-15",
                "2020-03-15",
            ],
            "base_price": ["10", "11.5", "12", "20", "21", "22"],
        }
    )

    fig = price_time_series_figure(sample, title="Demo: base_price by prod_id")
    out = "src/fidzulu/utils/tmp_price_plot.html"
    fig.write_html(out)
    print(f"plotting demo wrote {out}")
