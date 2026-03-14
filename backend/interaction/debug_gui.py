"""
Debug GUI - Dash-based dashboard for real-time monitoring.

Single-page dark dashboard with:
- Interactive map with H3 hexagon overlay and bus markers
- Stats cards for active buses, observers, hexagons
- Tabs: Map stats, Ledgers, Model, Vehicles

No application logic lives here. All data comes from StateInterface.
"""

import threading
from typing import TYPE_CHECKING

import dash
from dash import html, dcc, dash_table, Input, Output
import dash_leaflet as dl
import dash_bootstrap_components as dbc

if TYPE_CHECKING:
    from .state_interface import StateInterface

_state: "StateInterface" = None
_REFRESH_MS = 5000

# ============================================================
# Color Helpers
# ============================================================

_BG_DARK = "#1a1f2e"
_BG_CARD = "#232a3b"
_BG_CARD_LIGHT = "#2a3347"
_TEXT = "#e0e6ed"
_TEXT_DIM = "#8899aa"
_ACCENT = "#00d4aa"
_ACCENT_BLUE = "#4dc9f6"
_ORANGE = "#ff9f43"
_RED = "#ff6b6b"
_GREEN = "#2ed573"


def _hex_color(speed_ratio: float) -> str:
    """Map speed_ratio (0-1) to a traffic color."""
    if speed_ratio > 0.7:
        return _GREEN
    if speed_ratio > 0.3:
        return _ORANGE
    return _RED


def _format_timestamp(ts: float) -> str:
    """Format Unix timestamp to HH:MM:SS."""
    if not ts:
        return "N/A"
    from datetime import datetime
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


# ============================================================
# Layout Builders
# ============================================================


def _build_header():
    return dbc.Navbar(
        dbc.Container([
            dbc.Row([
                dbc.Col(html.Div([
                    html.Span("Backend", style={
                        "fontSize": "1.4rem", "fontWeight": "bold",
                        "color": _TEXT, "marginRight": "12px",
                    }),
                    dbc.Badge("Dashboard", color="info", className="align-middle"),
                ]), width="auto"),
                dbc.Col(html.Div(id="live-indicator", children=[
                    html.Span(
                        "\u25cf ", style={"color": _GREEN, "fontSize": "0.9rem"}
                    ),
                    html.Span("Live", style={"color": _TEXT_DIM, "fontSize": "0.85rem"}),
                ], style={"textAlign": "right"}), width="auto"),
            ], align="center", justify="between", className="w-100"),
        ], fluid=True),
        color=_BG_CARD, dark=True,
        style={"borderBottom": f"1px solid {_BG_CARD_LIGHT}"},
    )


def _build_stats_row():
    return html.Div(id="stats-cards", style={"marginBottom": "12px"})


def _stat_card(title, value, color=_ACCENT):
    return dbc.Card(
        dbc.CardBody([
            html.P(title, style={
                "color": _TEXT_DIM, "fontSize": "0.75rem",
                "marginBottom": "2px", "textTransform": "uppercase",
            }),
            html.H4(str(value), style={
                "color": color, "fontWeight": "bold", "marginBottom": "0",
            }),
        ], style={"padding": "10px 14px"}),
        style={
            "backgroundColor": _BG_CARD, "border": f"1px solid {_BG_CARD_LIGHT}",
            "borderRadius": "8px", "minWidth": "120px",
        },
    )


def _build_map_component():
    return html.Div([
        dl.Map([
            dl.TileLayer(
                url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
                attribution='&copy; <a href="https://carto.com/">CARTO</a>',
            ),
            dl.LayerGroup(id="hex-layer"),
            dl.LayerGroup(id="bus-layer"),
        ], id="main-map", center=[41.9028, 12.4964], zoom=12,
           style={
               "height": "72vh", "borderRadius": "8px",
               "border": f"1px solid {_BG_CARD_LIGHT}",
           }),
    ])


def _build_right_panel():
    return html.Div([
        _build_stats_row(),
        dbc.Tabs([
            dbc.Tab(label="Overview", tab_id="tab-overview"),
            dbc.Tab(label="Ledgers", tab_id="tab-ledgers"),
            dbc.Tab(label="Model", tab_id="tab-model"),
            dbc.Tab(label="Vehicles", tab_id="tab-vehicles"),
        ], id="right-tabs", active_tab="tab-overview",
           style={"marginBottom": "8px"}),
        html.Div(id="tab-content", style={
            "maxHeight": "50vh", "overflowY": "auto",
            "backgroundColor": _BG_CARD, "borderRadius": "8px",
            "padding": "12px", "border": f"1px solid {_BG_CARD_LIGHT}",
        }),
    ], style={"height": "100%"})


def _build_layout():
    return html.Div([
        _build_header(),
        dbc.Container([
            dbc.Row([
                dbc.Col(_build_map_component(), width=8, style={"paddingRight": "8px"}),
                dbc.Col(_build_right_panel(), width=4, style={"paddingLeft": "8px"}),
            ], style={"marginTop": "12px"}),
        ], fluid=True),
        dcc.Interval(id="refresh-interval", interval=_REFRESH_MS, n_intervals=0),
    ], style={
        "backgroundColor": _BG_DARK, "minHeight": "100vh",
        "fontFamily": "'Segoe UI', system-ui, sans-serif",
    })


# ============================================================
# Callback Registration
# ============================================================


def _register_callbacks(app):

    @app.callback(
        Output("hex-layer", "children"),
        Output("bus-layer", "children"),
        Input("refresh-interval", "n_intervals"),
    )
    def update_map(_n):
        if not _state:
            return [], []

        # Hexagons with traffic data
        hex_children = []
        traffic_hexes = _state.get_traffic_hexagons("Rome")
        for h in traffic_hexes:
            color = _hex_color(h["speed_ratio"])
            positions = [coord for coord in h["boundary"]]
            # Close the polygon
            if positions:
                positions.append(positions[0])
            hex_children.append(
                dl.Polygon(
                    positions=positions,
                    pathOptions={
                        "color": color, "weight": 1, "opacity": 0.6,
                        "fillColor": color, "fillOpacity": 0.25,
                    },
                    children=dl.Tooltip(
                        f"Hex: {h['hex_id'][:12]}...\n"
                        f"Speed ratio: {h['speed_ratio']:.2f}\n"
                        f"Buses: {h['bus_count']}\n"
                        f"Avg speed: {h['current_speed']} kph"
                    ),
                )
            )

        # Bus markers
        bus_children = []
        bus_positions = _state.get_bus_positions("Rome")
        for b in bus_positions:
            marker_color = _GREEN if b["status"] == "ACTIVE" else _ORANGE
            bus_children.append(
                dl.CircleMarker(
                    center=[b["lat"], b["lon"]],
                    radius=5, pathOptions={
                        "color": marker_color, "fillColor": marker_color,
                        "fillOpacity": 0.9, "weight": 1,
                    },
                    children=dl.Tooltip(
                        f"Route {b['route_id']} | {b['label']}\n"
                        f"{b['direction']}\n"
                        f"Speed: {b['speed']} kph"
                    ),
                )
            )

        return hex_children, bus_children

    @app.callback(
        Output("stats-cards", "children"),
        Input("refresh-interval", "n_intervals"),
    )
    def update_stats(_n):
        if not _state:
            return html.Div("Waiting for data...", style={"color": _TEXT_DIM})

        stats = _state.get_system_stats()
        feed_ts = _state.get_feed_timestamp()
        traffic = _state.get_traffic_stats("Rome")

        return dbc.Row([
            dbc.Col(_stat_card("Active", stats["active_buses"], _GREEN), width=3),
            dbc.Col(_stat_card("Deposit", stats["deposit_buses"], _ORANGE), width=3),
            dbc.Col(_stat_card("Observers", stats["observer_count"], _ACCENT_BLUE), width=3),
            dbc.Col(_stat_card("Hexagons", traffic["with_traffic"], _ACCENT), width=3),
        ], className="g-2")

    @app.callback(
        Output("tab-content", "children"),
        Input("right-tabs", "active_tab"),
        Input("refresh-interval", "n_intervals"),
    )
    def render_tab(active_tab, _n):
        if not _state:
            return html.Div("Waiting for data...", style={"color": _TEXT_DIM})

        if active_tab == "tab-overview":
            return _render_overview_tab()
        elif active_tab == "tab-ledgers":
            return _render_ledgers_tab()
        elif active_tab == "tab-model":
            return _render_model_tab()
        elif active_tab == "tab-vehicles":
            return _render_vehicles_tab()
        return html.Div()


def _render_overview_tab():
    """Render system alerts and service thread status."""
    thread_status = _state.get_service_thread_status()
    feed_ts = _state.get_feed_timestamp()
    geo_stats = _state.get_geocoding_stats()
    stats = _state.get_system_stats()

    items = []

    # Service status
    items.append(html.H6("System Services", style={
        "color": _ACCENT, "marginBottom": "10px", "fontWeight": "bold",
    }))

    for name, status in thread_status.items():
        if status == "running":
            color = _GREEN
            icon = "\u25cf"
        elif status == "stopped":
            color = _RED
            icon = "\u25cf"
        else:
            color = _TEXT_DIM
            icon = "\u25cb"

        items.append(html.Div([
            html.Span(f"{icon} ", style={"color": color}),
            html.Span(name, style={"color": _TEXT, "fontSize": "0.85rem"}),
            html.Span(
                f" {status}", style={
                    "color": _TEXT_DIM, "fontSize": "0.75rem", "float": "right",
                },
            ),
        ], style={"marginBottom": "4px"}))

    # Feed timestamp
    items.append(html.Hr(style={"borderColor": _BG_CARD_LIGHT, "margin": "12px 0"}))
    items.append(html.Div([
        html.Span("Last Feed: ", style={"color": _TEXT_DIM, "fontSize": "0.85rem"}),
        html.Span(
            _format_timestamp(feed_ts),
            style={"color": _ACCENT, "fontSize": "0.85rem"},
        ),
    ]))

    # Geocoding
    if geo_stats["enabled"]:
        items.append(html.Div([
            html.Span("Geo Queue: ", style={"color": _TEXT_DIM, "fontSize": "0.85rem"}),
            html.Span(
                str(geo_stats["pending"]),
                style={"color": _ACCENT_BLUE, "fontSize": "0.85rem"},
            ),
        ], style={"marginTop": "4px"}))

    # Network summary
    items.append(html.Hr(style={"borderColor": _BG_CARD_LIGHT, "margin": "12px 0"}))
    items.append(html.H6("Network Summary", style={
        "color": _ACCENT, "marginBottom": "8px", "fontWeight": "bold",
    }))
    items.append(html.Div([
        html.Span(f"Cities: {stats['city_count']}", style={
            "color": _TEXT, "fontSize": "0.85rem",
        }),
    ]))
    items.append(html.Div([
        html.Span(f"Total hexagons: {stats['hexagon_count']}", style={
            "color": _TEXT, "fontSize": "0.85rem",
        }),
    ]))

    return html.Div(items)


def _render_ledgers_tab():
    """Render ledger information."""
    ledger_info = _state.get_all_ledger_info()
    items = []

    # Topology
    topo = ledger_info["topology"]
    items.append(_ledger_card(
        "Topology Ledger",
        [
            ("Status", "Loaded" if topo["loaded"] else "Not loaded"),
            ("Routes", str(topo.get("routes", "-"))),
            ("Trips", str(topo.get("trips", "-"))),
            ("Stops", str(topo.get("stops", "-"))),
            ("Shapes", str(topo.get("shapes", "-"))),
            ("MD5", str(topo.get("md5", "-"))[:16]),
        ] if topo["loaded"] else [("Status", "Not loaded")],
        color=_GREEN if topo["loaded"] else _RED,
    ))

    # Schedule
    sched = ledger_info["schedule"]
    items.append(_ledger_card(
        "Schedule Ledger",
        [
            ("Status", "Loaded" if sched["loaded"] else "Not loaded"),
            ("Routes indexed", str(sched.get("routes_indexed", "-"))),
        ] if sched["loaded"] else [("Status", "Not loaded")],
        color=_GREEN if sched["loaded"] else _RED,
    ))

    # Database-backed ledgers
    for name, key in [
        ("Historical Ledger", "historical"),
        ("Predicted Ledger", "predicted"),
        ("Vehicle Ledger", "vehicle"),
    ]:
        info = ledger_info[key]
        items.append(_ledger_card(
            name,
            [
                ("Type", info["type"]),
                ("Table", info["table"]),
            ],
            color=_ACCENT_BLUE,
        ))

    return html.Div(items)


def _ledger_card(title, rows, color=_ACCENT):
    """Build a small card for a single ledger."""
    row_elements = []
    for label, value in rows:
        row_elements.append(html.Div([
            html.Span(f"{label}: ", style={
                "color": _TEXT_DIM, "fontSize": "0.8rem",
            }),
            html.Span(value, style={
                "color": _TEXT, "fontSize": "0.8rem",
            }),
        ], style={"marginBottom": "2px"}))

    return html.Div([
        html.Div([
            html.Span("\u25cf ", style={"color": color, "fontSize": "0.7rem"}),
            html.Span(title, style={
                "color": _TEXT, "fontSize": "0.9rem", "fontWeight": "bold",
            }),
        ], style={"marginBottom": "6px"}),
        html.Div(row_elements, style={"paddingLeft": "16px"}),
    ], style={
        "backgroundColor": _BG_CARD_LIGHT, "borderRadius": "6px",
        "padding": "10px", "marginBottom": "8px",
    })


def _render_model_tab():
    """Render model info and validation status."""
    model_info = _state.get_model_info()

    items = []
    loaded = model_info["loaded"]

    items.append(html.Div([
        html.Span("Model Status: ", style={
            "color": _TEXT_DIM, "fontSize": "0.9rem",
        }),
        dbc.Badge(
            "Loaded" if loaded else "Not Loaded",
            color="success" if loaded else "danger",
            className="ms-1",
        ),
    ], style={"marginBottom": "12px"}))

    items.append(html.Div([
        html.Span("Model Name: ", style={
            "color": _TEXT_DIM, "fontSize": "0.85rem",
        }),
        html.Span(model_info["model_name"], style={
            "color": _ACCENT if loaded else _TEXT_DIM, "fontSize": "0.85rem",
        }),
    ], style={"marginBottom": "8px"}))

    # Validation status from services
    thread_status = _state.get_service_thread_status()
    batch_status = thread_status.get("Batch Validation", "idle")
    live_status = thread_status.get("Live Validation", "idle")

    items.append(html.Hr(style={"borderColor": _BG_CARD_LIGHT, "margin": "12px 0"}))
    items.append(html.H6("Validation", style={
        "color": _ACCENT, "marginBottom": "8px", "fontWeight": "bold",
    }))

    for label, status in [("Batch", batch_status), ("Live", live_status)]:
        status_color = _GREEN if status not in ("idle", "not started") else _TEXT_DIM
        items.append(html.Div([
            html.Span(f"{label}: ", style={
                "color": _TEXT_DIM, "fontSize": "0.85rem",
            }),
            html.Span(status, style={
                "color": status_color, "fontSize": "0.85rem",
            }),
        ], style={"marginBottom": "4px"}))

    return html.Div(items)


def _render_vehicles_tab():
    """Render a table of active vehicles."""
    rows = _state.get_tracking_summary()

    if not rows:
        return html.Div(
            "No active vehicles",
            style={"color": _TEXT_DIM, "textAlign": "center", "padding": "20px"},
        )

    return dash_table.DataTable(
        id="vehicle-table-inner",
        columns=[
            {"name": "ID", "id": "bus_id"},
            {"name": "Route", "id": "route_id"},
            {"name": "Direction", "id": "headsign"},
            {"name": "Speed", "id": "speed"},
            {"name": "Status", "id": "status"},
            {"name": "Last Seen", "id": "last_seen"},
            {"name": "Samples", "id": "samples"},
        ],
        data=rows,
        page_size=20,
        sort_action="native",
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": _BG_CARD_LIGHT,
            "color": _ACCENT, "fontWeight": "bold",
            "fontSize": "0.8rem", "border": "none",
            "textAlign": "left",
        },
        style_cell={
            "backgroundColor": _BG_CARD,
            "color": _TEXT, "border": f"1px solid {_BG_CARD_LIGHT}",
            "fontSize": "0.8rem", "padding": "6px 10px",
            "textAlign": "left", "maxWidth": "150px",
            "overflow": "hidden", "textOverflow": "ellipsis",
        },
        style_data_conditional=[
            {
                "if": {"filter_query": '{status} = "ACTIVE"', "column_id": "status"},
                "color": _GREEN, "fontWeight": "bold",
            },
            {
                "if": {"filter_query": '{status} = "DEPOSIT"', "column_id": "status"},
                "color": _ORANGE, "fontWeight": "bold",
            },
        ],
    )


# ============================================================
# App Factory & Runner
# ============================================================


def create_app(state: "StateInterface") -> dash.Dash:
    """Create and configure the Dash application."""
    global _state
    _state = state

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        title="ATAC Backend Dashboard",
    )
    app.layout = _build_layout()
    _register_callbacks(app)
    return app


def start_gui(state: "StateInterface", port: int = 8050):
    """Start the Dash server in a background thread."""
    app = create_app(state)

    def run_server():
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    print(f" > Dashboard GUI started at http://localhost:{port}")
    return thread


def stop_gui():
    """Stop the Dash server (best-effort for threaded Flask)."""
    pass
