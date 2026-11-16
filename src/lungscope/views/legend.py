"""The legend of the scene's coloring: a horizontal color bar with ticks, bottom left."""

from trame.widgets import html
from trame.widgets import vuetify3 as v3

INK = "#52514e"


def add_legend() -> None:
    """The legend in state variable ``legend`` (from ``LungScene.show``); hidden while it is None.

    A gradient or one block per value, with ticks, and a swatch for each listed item, such as
    the gray of elements for which the coloring is not defined.
    """
    tick_style = (
        "{ position: 'absolute', left: tick.pos + '%', top: 0, "
        "transform: 'translateX(-50%)', textAlign: 'center', whiteSpace: 'nowrap' }"
    )
    swatch_style = (
        "{ background: item.color, width: '12px', height: '12px', "
        "borderRadius: '2px', marginRight: '8px', display: 'inline-block' }"
    )
    with v3.VCard(
        v_if="legend",
        classes="px-5 pt-3 pb-2",
        elevation=2,
        style="position: absolute; bottom: 20px; left: 20px; z-index: 1; width: 300px",
    ):
        html.Div("{{ legend.title }}", classes="text-subtitle-2 text-center mb-1")
        html.Div(
            v_if="legend.gradient",
            style=("{ background: legend.gradient, height: '12px', borderRadius: '2px' }",),
        )
        with html.Div(v_if="legend.steps.length", style="display: flex; gap: 2px; height: 12px"):
            html.Div(
                v_for="(step, k) in legend.steps",
                key="k",
                style=("{ background: step.color, flex: 1, borderRadius: '2px' }",),
            )
        with (
            html.Div(style="position: relative; height: 22px", classes="text-caption"),
            html.Div(v_for="(tick, k) in legend.ticks", key="k", style=(tick_style,)),
        ):
            html.Div(style=f"width: 1px; height: 4px; background: {INK}; margin: 0 auto")
            html.Div("{{ tick.label }}", style="line-height: 16px")
        with html.Div(
            v_for="item in legend.items",
            key="item.label",
            classes="d-flex align-center text-caption",
        ):
            html.Span(style=(swatch_style,))
            html.Span("{{ item.label }}")
