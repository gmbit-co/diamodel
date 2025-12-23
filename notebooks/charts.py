from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd
import altair as alt
import diamodel as dm


def plot_density(name, posterior: dm.Fit, prior: Optional[dm.Fit] = None, width=400, height=200):
    posterior = getattr(posterior, name)
    df = pd.DataFrame({name: posterior, "kind": "posterior"})

    if prior is not None:
        prior = getattr(prior, name)
        prior_df = pd.DataFrame({name: prior, "kind": "prior"})
        df = pd.concat([df, prior_df], ignore_index=True)

    chart = (
        alt.Chart(df)
        .transform_density(
            density=name,
            groupby=["kind"],
            counts=False,
            steps=50,
            as_=[name, "density"],
        )
        .mark_area(opacity=0.5)
        .encode(
            alt.X(f"{name}:Q"),
            alt.Y("density:Q").stack(None),
            alt.Color("kind:N"),
        )
        .properties(width=width, height=height, title=f"{name} density")
    )
    return chart


def predict(ticka: int, tickb: int, fit: dm.Fit, cfg: dm.Config, bg0=None, bolus_df=None):
    """Calculates BG predictions between ticka and tickb"""
    carbs = None if bolus_df is None else bolus_df["carbs"]
    insulin = None if bolus_df is None else bolus_df["insulin"]

    ticks = np.arange(ticka, tickb + 1)
    car = fit.predict_car(ticks, cfg, carbs)
    iar = fit.predict_iar(ticks, cfg, insulin)

    bg = fit.predict_bg(ticks, cfg, bg0, bolus_df)

    df = pd.DataFrame(
        {
            "tick": ticks,
            "car": list(car.T),
            "iar": list(iar.T),
            "bg": list(bg.T),
        }
    )
    df.set_index("tick", drop=False, inplace=True)
    return df


def plot_bg(pred_df, xaxis="timestamp", height=200, width=600):
    cgm = pred_df[["tick", "timestamp"]].copy()
    cgm["kind"] = "cgm"
    cgm["val"] = pred_df["cgm"]
    cgm["val10"] = np.nan
    cgm["val90"] = np.nan

    pbg = pred_df[["tick", "timestamp"]].copy()
    pbg["kind"] = "prediction"

    pbg["val"] = pred_df["bg"].map(np.mean)
    pbg["val10"] = pred_df["bg"].map(lambda a: np.percentile(a, 10))
    pbg["val90"] = pred_df["bg"].map(lambda a: np.percentile(a, 90))

    df = pd.concat([pbg, cgm], ignore_index=True)

    chart_val = (
        alt.Chart(df)
        .mark_line(strokeWidth=1, point={"size": 15})
        .encode(
            x=xaxis,
            y=alt.Y("val").scale(zero=False),
            color=alt.Color(
                "kind",
                scale=alt.Scale(domain=["cgm", "prediction"], range=["green", "#1f77b4"]),
            ),
        )
    )

    chart_band = (
        alt.Chart(df)
        .encode(x=xaxis)
        .mark_area(opacity=0.3)
        .encode(
            alt.Y("val90", axis=alt.Axis(title="mmol/L")),
            alt.Y2("val10"),
            color="kind",
        )
    )

    chart = (chart_val + chart_band).properties(
        title="Blood Glucose",
        height=height,
        width=width,
    )

    return chart


def plot_absorptions(pred_df, xaxis="timestamp", height=200, width=600):
    car = pred_df[["tick", "timestamp"]].copy()
    car["kind"] = "carbs"
    car["val"] = pred_df["car"].map(np.mean)
    car["val10"] = pred_df["car"].map(lambda a: np.percentile(a, 10))
    car["val90"] = pred_df["car"].map(lambda a: np.percentile(a, 90))

    iar = pred_df[["tick", "timestamp"]].copy()
    iar["kind"] = "insulin"
    iar["val"] = pred_df["iar"].map(np.mean)
    iar["val10"] = pred_df["iar"].map(lambda a: np.percentile(a, 10))
    iar["val90"] = pred_df["iar"].map(lambda a: np.percentile(a, 90))

    df = pd.concat([car, iar], ignore_index=True)

    chart_rate = (
        alt.Chart(df)
        .mark_line(strokeWidth=2, point={"size": 15})
        .encode(
            x=xaxis,
            y="val",
            color=alt.Color(
                "kind",
                scale=alt.Scale(domain=["insulin", "carbs"], range=["navy", "orange"]),
            ),
        )
    )

    chart_band = (
        alt.Chart(df)
        .encode(x=xaxis)
        .mark_area(opacity=0.3)
        .encode(
            alt.Y("val90", axis=alt.Axis(title="mmol/L / 5 mins")),
            alt.Y2("val10"),
            color="kind",
        )
    )

    chart = (chart_rate + chart_band).properties(
        title="Absorption rate", height=height, width=width
    )
    return chart


def plot_predictions(
    ticka: int,
    tickb: int,
    data_df: pd.DataFrame,
    fit: dm.Fit,
    cfg: dm.Config,
    bg0=None,
    bolus_df=None,
    height=200,
    width=600,
):
    tickb = min(tickb, data_df.iloc[-1].tick)

    pred_df = predict(ticka, tickb, fit, cfg, bg0, bolus_df)

    pred_df["cgm"] = data_df["cgm"].loc[pred_df["tick"]]
    # pred_df["cgm"] = data_df["bg"].loc[pred_df["tick"]]
    pred_df["timestamp"] = data_df["timestamp"].loc[pred_df["tick"]]

    xaxis = "timestamp"
    ch1 = plot_bg(pred_df, xaxis=xaxis, height=height, width=width)
    ch2 = plot_absorptions(pred_df, xaxis=xaxis, height=height, width=width)

    return (ch1 & ch2).resolve_scale(color="independent").resolve_legend(color="independent")


def plot_fits_density(name, ifits, height=200, width=600):
    get_samples = lambda name: [getattr(f.posterior, name) for f in ifits]

    if name == "icr":
        isens = get_samples("isens")
        csens = get_samples("csens")
        samples = [ic / cs for ic, cs in zip(isens, csens)]
    else:
        samples = get_samples(name)

    dates = [datetime.fromisoformat(f.date.isoformat()) for f in ifits]
    means = [ss.mean() for ss in samples]
    p5s = [np.percentile(ss, 5) for ss in samples]
    p95s = [np.percentile(ss, 95) for ss in samples]

    df = pd.DataFrame({"date": dates, "mean": means, "p5": p5s, "p95": p95s})

    ch_line = (
        alt.Chart(df)
        .mark_line(strokeWidth=2, point={"size": 20})
        .encode(
            x="date",
            y=alt.Y("mean").scale(zero=False),
        )
    )

    chart_band = (
        alt.Chart(df)
        .encode(x="date")
        .mark_area(opacity=0.3)
        .encode(
            alt.Y("p95"),
            alt.Y2("p5"),
        )
    )

    return (ch_line + chart_band).properties(
        title=f"{name} Posterior Distribution",
        height=height,
        width=width,
    )
