"""The plot of one element: every run of a quantity over time, their spread and the mean.

The chart is a Plotly figure as plain data (lists and dicts) for trame-plotly. Two panels share
the time axis: the values (each run thin and gray, their min–max band, the mean in bold) and
each run's difference from the mean, where a spread too small to see beside the values shows
(pressures of hundreds of Pa differ by hundredths of a Pa between the runs). A vertical line
marks the selected time step. ``add_run_chart`` puts the chart into the page.
"""

import json
import math
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import pyvista as pv
from plotly.subplots import make_subplots
from trame.widgets import html
from trame.widgets import plotly as plotly_widgets

from lungscope.analysis import Spread, run_spread

MEAN_COLOR = "#2a78d6"  # the accent: the mean is the series the chart is about
RUN_COLOR = "rgba(137, 135, 129, 0.55)"  # each run: muted gray, behind the mean
BAND_COLOR = "rgba(42, 120, 214, 0.12)"  # min–max: a wash of the mean's hue
TEXT_COLOR = "#52514e"
MUTED_COLOR = "#898781"
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
FONT = "Roboto, system-ui, sans-serif"  # the page's font (Vuetify)
PANEL_GAP = 0.17  # vertical space between the panels (fraction of the plot), for a panel title


@dataclass
class RunChart:
    """The chart of one element and quantity, and what it shows at each time step."""

    figure: dict  # Plotly figure (data and layout) without the time step marker
    times: np.ndarray
    readouts: list[dict]  # per time step: "time" and "items" (label, value) in words


def add_run_chart(state_name: str, height: int) -> None:
    """The chart's widget: the Plotly figure in state variable ``state_name`` (data and layout)."""
    with html.Div(style=f"height: {height}px"):
        plotly_widgets.Figure(
            state_variable_name=state_name,
            display_logo=("false",),
            mode_bar_buttons_to_remove=("['select2d', 'lasso2d']",),
        )


def element_title(mesh: pv.UnstructuredGrid, element: int, is_unit: bool) -> str:
    """E.g. "Airway 7 · generation 3" or "Terminal unit 18", numbered like the study's entities."""
    number = int(mesh.cell_data["entity_id"][element]) if "entity_id" in mesh.cell_data else element
    if is_unit:
        return f"Terminal unit {number}"
    title = f"Airway {number}"
    if "generation" in mesh.cell_data and int(mesh.cell_data["generation"][element]) >= 0:
        title += f" · generation {int(mesh.cell_data['generation'][element])}"
    return title


def run_chart(times, runs, label: str, unit: str, revision: str) -> RunChart | None:
    """Chart of one element's runs, an array (realization, time); None if no run has a value.

    ``label`` and ``unit`` name the values. ``revision`` names the element and quantity: while
    it stays the same, Plotly keeps the reader's zoom when the figure is replaced. Runs are
    numbered from 0, like the realizations (and the output folders) of the study.
    """
    times = np.asarray(times, dtype=float)
    runs = np.asarray(runs, dtype=float)
    if not np.isfinite(runs).any():
        return None
    spread = run_spread(runs)
    digits = _decimals(spread)
    suffix = f" {unit}" if unit else ""
    value = f"%{{y:.{digits}f}}{suffix}"  # (plotly.js 3.0 ignores a format with a "+" sign flag)
    x, x_runs = _data(times), _data(np.tile(np.append(times, np.nan), len(runs)))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=PANEL_GAP, row_heights=[0.56, 0.44])
    for row, base in ((1, 0.0), (2, spread.mean)):
        top = row == 1
        joined = np.column_stack([runs - base, np.full(len(runs), np.nan)])
        traces = [
            go.Scatter(
                x=x, y=_data(spread.lo - base, digits), customdata=[int(k) for k in spread.lowest],
                mode="lines", line={"width": 0}, name="lowest", legendgroup="band", showlegend=False,
                hovertemplate=f"{value} · lowest, run %{{customdata}}<extra></extra>",
            ),
            go.Scatter(
                x=x, y=_data(spread.hi - base, digits), customdata=[int(k) for k in spread.highest],
                mode="lines", line={"width": 0}, fill="tonexty", fillcolor=BAND_COLOR,
                name="min–max", legendgroup="band", showlegend=top,
                hovertemplate=f"{value} · highest, run %{{customdata}}<extra></extra>",
            ),
            go.Scatter(
                x=x_runs, y=_data(joined, digits), mode="lines", connectgaps=False,
                line={"color": RUN_COLOR, "width": 1}, name="each run", legendgroup="runs",
                showlegend=top, hoverinfo="skip",
            ),
            go.Scatter(
                x=x, y=_data(spread.mean - base, digits), mode="lines",
                line={"color": MEAN_COLOR, "width": 2}, name="mean", legendgroup="mean", showlegend=top,
                **({"hovertemplate": f"{value} · mean<extra></extra>"} if top else {"hoverinfo": "skip"}),
            ),
        ]
        for trace in traces:
            fig.add_trace(trace, row=row, col=1)
    axis = {
        "showgrid": True, "gridcolor": GRID_COLOR, "gridwidth": 1, "zeroline": False, "showline": True,
        "linecolor": AXIS_COLOR, "linewidth": 1, "ticks": "", "automargin": True,
        "tickfont": {"color": MUTED_COLOR, "size": 10},
    }
    # the unit on every time tick also puts it into the hover header ("0.50 s")
    fig.update_xaxes(**axis, hoverformat=".2f", ticksuffix=" s")
    fig.update_yaxes(**axis)
    fig.update_xaxes(title={"text": "time", "font": {"size": 11, "color": TEXT_COLOR}}, row=2, col=1)
    named = f"{label} [{unit}]" if unit else label
    difference = f"each run minus the mean [{unit}]" if unit else "each run minus the mean"
    fig.update_layout(
        annotations=[
            _panel_title(named, fig.layout.yaxis.domain[1]),
            _panel_title(difference, fig.layout.yaxis2.domain[1]),
        ],
        margin={"l": 8, "r": 8, "t": 22, "b": 8},
        font={"family": FONT, "size": 11, "color": TEXT_COLOR},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        hoverlabel={
            "bgcolor": "#ffffff", "bordercolor": GRID_COLOR,
            "font": {"family": FONT, "size": 11, "color": "#0b0b0b"},
        },
        legend={
            "orientation": "h", "x": 0, "xanchor": "left", "y": -0.16, "yanchor": "top",
            "traceorder": "reversed", "bgcolor": "rgba(0,0,0,0)", "font": {"size": 11},
        },
        uirevision=revision,
    )
    figure = json.loads(fig.to_json())
    figure["layout"].pop("template", None)  # plotly.py's default look: replaced by the settings above
    readouts = [_readout(spread, step, time, digits, suffix) for step, time in enumerate(times)]
    return RunChart(figure, times, readouts)


def figure_at(chart: RunChart, step: int) -> dict:
    """The chart's figure with a vertical line at time step ``step``, across both panels."""
    t = float(chart.times[min(max(int(step), 0), len(chart.times) - 1)])
    line = {
        "type": "line", "x0": t, "x1": t, "y0": 0, "y1": 1, "layer": "above",
        "line": {"color": TEXT_COLOR, "width": 1},
    }
    shapes = [{**line, "xref": "x", "yref": "y domain"}, {**line, "xref": "x2", "yref": "y2 domain"}]
    return {"data": chart.figure["data"], "layout": {**chart.figure["layout"], "shapes": shapes}}


def _panel_title(text: str, top: float) -> dict:
    """A panel's title, left-aligned just above it (``top``: the panel's upper edge, 0..1)."""
    return {
        "text": text, "xref": "paper", "yref": "paper", "x": 0, "y": top, "xanchor": "left",
        "yanchor": "bottom", "showarrow": False, "font": {"size": 11, "color": TEXT_COLOR},
    }


def _readout(spread: Spread, step: int, time: float, digits: int, suffix: str) -> dict:
    """What the chart shows at one time step, in words: mean, standard deviation, range."""
    mean = spread.mean[step]
    if not np.isfinite(mean):
        return {"time": f"t = {time:.2f} s", "items": [{"label": "value", "value": "none"}]}
    low, high = spread.lo[step], spread.hi[step]
    return {
        "time": f"t = {time:.2f} s",
        "items": [
            {"label": "mean", "value": f"{mean:.{digits}f}{suffix}"},
            {"label": "std", "value": f"{_significant(spread.std[step])}{suffix}"},
            {"label": "min–max", "value": f"{low:.{digits}f} – {high:.{digits}f}{suffix}"},
        ],
    }


def _decimals(spread: Spread) -> int:
    """Decimals that resolve about a hundredth of the widest spread between the runs."""
    width = np.nanmax(spread.hi - spread.lo)
    scale = width if np.isfinite(width) and width > 0 else 1e-3 * np.nanmax(np.abs(spread.mean))
    if not np.isfinite(scale) or scale <= 0:
        return 2
    return int(np.clip(math.ceil(-math.log10(scale)) + 2, 0, 8))


def _significant(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}g}" if np.isfinite(value) else "–"


def _data(values, digits: int | None = None) -> list:
    """A plain list for the figure: rounded, NaN as None (a gap in a line)."""
    values = np.asarray(values, dtype=float).ravel()
    if digits is not None:
        values = np.round(values, digits + 2)
    return [float(v) if math.isfinite(v) else None for v in values]
