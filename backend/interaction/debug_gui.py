"""
Debug GUI - Dash-based dashboard for real-time monitoring.

Vertical sidebar navigation with full-width content:
- Map tab: interactive map with H3 hexagon overlay, bus markers, active vehicles table
- Ledger tab: prediction summaries + vehicle trips (full in-memory data, 50/page)
- Vehicles tab: all tracked vehicles
- Commands tab: console command execution
- Overview tab: service status, network summary

Click-to-expand detail popups for predictions (per-stop) and vehicle trips.

No application logic lives here. All data comes from StateInterface.
"""

import logging
import threading
from typing import TYPE_CHECKING

import dash
from dash import html, dcc, dash_table, Input, Output, State, no_update
import dash_leaflet as dl
import dash_bootstrap_components as dbc

_log = logging.getLogger(__name__)

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

_SIDEBAR_WIDTH = "80px"

_TABLE_STYLE_HEADER = {
    "backgroundColor": _BG_CARD_LIGHT,
    "color": _ACCENT, "fontWeight": "bold",
    "fontSize": "0.78rem", "border": "none",
    "textAlign": "left",
}
_TABLE_STYLE_CELL = {
    "backgroundColor": _BG_CARD,
    "color": _TEXT, "border": f"1px solid {_BG_CARD_LIGHT}",
    "fontSize": "0.78rem", "padding": "5px 10px",
    "textAlign": "left",
}


def _hex_color(speed_ratio: float) -> str:
    if speed_ratio > 0.7:
        return _GREEN
    if speed_ratio > 0.3:
        return _ORANGE
    return _RED


def _format_timestamp(ts: float) -> str:
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
            "borderRadius": "8px", "minWidth": "100px",
        },
    )


def _sidebar_button(label, icon, tab_id):
    return html.Div(
        html.Button(
            html.Div([
                html.Div(icon, style={"fontSize": "1.3rem", "lineHeight": "1"}),
                html.Div(label, style={"fontSize": "0.6rem", "marginTop": "2px"}),
            ], style={"textAlign": "center"}),
            id={"type": "sidebar-btn", "tab": tab_id},
            n_clicks=0,
            style={
                "width": "100%", "border": "none", "background": "transparent",
                "color": _TEXT_DIM, "padding": "12px 4px", "cursor": "pointer",
            },
        ),
        style={"borderBottom": f"1px solid {_BG_CARD_LIGHT}"},
    )


def _build_sidebar():
    return html.Div([
        _sidebar_button("Map", "\U0001f5fa", "tab-map"),
        _sidebar_button("Ledger", "\U0001f4ca", "tab-ledgers"),
        _sidebar_button("Vehicles", "\U0001f68c", "tab-vehicles"),
        _sidebar_button("Cmds", "\u2318", "tab-commands"),
        _sidebar_button("Status", "\u2139", "tab-overview"),
    ], style={
        "width": _SIDEBAR_WIDTH, "minWidth": _SIDEBAR_WIDTH,
        "backgroundColor": _BG_CARD, "height": "calc(100vh - 56px)",
        "borderRight": f"1px solid {_BG_CARD_LIGHT}",
        "overflowY": "auto", "flexShrink": "0",
    })


def _build_layout():
    return html.Div([
        _build_header(),
        html.Div([
            _build_sidebar(),
            html.Div([
                # Stats row at top of content
                html.Div(id="stats-cards", style={"marginBottom": "12px"}),
                # Tab content
                html.Div(id="tab-content"),
            ], style={
                "flex": "1", "padding": "12px", "overflowY": "auto",
                "height": "calc(100vh - 56px)",
            }),
        ], style={"display": "flex"}),
        # Hidden store for active tab
        dcc.Store(id="active-tab-store", data="tab-map"),
        dcc.Interval(id="refresh-interval", interval=_REFRESH_MS, n_intervals=0),
        # Detail modals
        _build_prediction_modal(),
        _build_vehicle_modal(),
    ], style={
        "backgroundColor": _BG_DARK, "minHeight": "100vh",
        "fontFamily": "'Segoe UI', system-ui, sans-serif",
    })


def _build_prediction_modal():
    return dbc.Modal([
        dbc.ModalHeader(
            dbc.ModalTitle(id="pred-modal-title"),
            close_button=True,
            style={"backgroundColor": _BG_CARD, "color": _TEXT,
                    "borderBottom": f"1px solid {_BG_CARD_LIGHT}"},
        ),
        dbc.ModalBody(
            id="pred-modal-body",
            style={"backgroundColor": _BG_CARD, "color": _TEXT, "padding": "16px"},
        ),
    ], id="pred-modal", is_open=False, size="lg",
       style={"color": _TEXT})


def _build_vehicle_modal():
    return dbc.Modal([
        dbc.ModalHeader(
            dbc.ModalTitle(id="veh-modal-title"),
            close_button=True,
            style={"backgroundColor": _BG_CARD, "color": _TEXT,
                    "borderBottom": f"1px solid {_BG_CARD_LIGHT}"},
        ),
        dbc.ModalBody(
            id="veh-modal-body",
            style={"backgroundColor": _BG_CARD, "color": _TEXT, "padding": "16px"},
        ),
    ], id="veh-modal", is_open=False, size="lg",
       style={"color": _TEXT})


# ============================================================
# Map Component
# ============================================================

def _build_map_component():
    return html.Div([
        dl.Map([
            dl.TileLayer(
                url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
                attribution='&copy; <a href="https://carto.com/">CARTO</a>',
            ),
            dl.LayerGroup(id="hex-layer"),
            dl.Pane(dl.LayerGroup(id="bus-layer"), name="buses", style={"zIndex": 650}),
        ], id="main-map", center=[41.9028, 12.4964], zoom=12,
           style={
               "height": "65vh", "borderRadius": "8px",
               "border": f"1px solid {_BG_CARD_LIGHT}",
           }),
    ])


# ============================================================
# Callback Registration
# ============================================================


def _register_callbacks(app):

    # Sidebar tab switching
    @app.callback(
        Output("active-tab-store", "data"),
        Input({"type": "sidebar-btn", "tab": dash.ALL}, "n_clicks"),
        State("active-tab-store", "data"),
        prevent_initial_call=True,
    )
    def switch_tab(n_clicks_list, current_tab):
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update
        trigger = ctx.triggered[0]
        if trigger["value"] is None or trigger["value"] == 0:
            return no_update
        import json
        prop_id = json.loads(trigger["prop_id"].rsplit(".", 1)[0])
        return prop_id["tab"]

    # Map layers update
    @app.callback(
        Output("hex-layer", "children"),
        Output("bus-layer", "children"),
        Input("refresh-interval", "n_intervals"),
        Input("active-tab-store", "data"),
    )
    def update_map(_n, active_tab):
        if not _state or active_tab != "tab-map":
            return no_update, no_update
        try:
            return _build_map_layers()
        except Exception as e:
            _log.error(f"Map update error: {e}", exc_info=True)
            return [], []

    def _build_map_layers():
        hex_children = []
        traffic_hexes = _state.get_traffic_hexagons("Rome")
        for h in traffic_hexes:
            color = _hex_color(h["speed_ratio"])
            positions = [coord for coord in h["boundary"]]
            if positions:
                positions.append(positions[0])
            hex_children.append(
                dl.Polygon(
                    positions=positions,
                    pathOptions={
                        "color": color, "weight": 1, "opacity": 0.35,
                        "fillColor": color, "fillOpacity": 0.15,
                    },
                    children=dl.Tooltip(
                        f"Hex: {h['hex_id'][:12]}...\n"
                        f"Speed ratio: {h['speed_ratio']:.2f}\n"
                        f"Buses: {h['bus_count']}\n"
                        f"Avg speed: {h['current_speed']} kph"
                    ),
                )
            )

        bus_children = []
        bus_positions = _state.get_bus_positions("Rome")
        for b in bus_positions:
            marker_color = _GREEN if b["status"] == "ACTIVE" else _ORANGE
            bus_children.append(
                dl.CircleMarker(
                    center=[b["lat"], b["lon"]],
                    radius=7,
                    bubblingMouseEvents=False,
                    pathOptions={
                        "color": "#ffffff", "fillColor": marker_color,
                        "fillOpacity": 0.95, "weight": 2,
                    },
                    children=dl.Tooltip(
                        f"Route {b['route_id']} | {b['label']}\n"
                        f"{b['direction']}\n"
                        f"Speed: {b['speed']} kph"
                    ),
                )
            )

        return hex_children, bus_children

    # Stats cards
    @app.callback(
        Output("stats-cards", "children"),
        Input("refresh-interval", "n_intervals"),
    )
    def update_stats(_n):
        if not _state:
            return html.Div("Waiting for data...", style={"color": _TEXT_DIM})
        try:
            stats = _state.get_system_stats()
            traffic = _state.get_traffic_stats("Rome")
            return dbc.Row([
                dbc.Col(_stat_card("Active", stats["active_buses"], _GREEN), width="auto"),
                dbc.Col(_stat_card("Deposit", stats["deposit_buses"], _ORANGE), width="auto"),
                dbc.Col(_stat_card("Observers", stats["observer_count"], _ACCENT_BLUE), width="auto"),
                dbc.Col(_stat_card("Hexagons", traffic["with_traffic"], _ACCENT), width="auto"),
            ], className="g-2")
        except Exception as e:
            _log.error(f"Stats update error: {e}", exc_info=True)
            return html.Div(f"Error: {e}", style={"color": _RED})

    # Tab content
    @app.callback(
        Output("tab-content", "children"),
        Input("active-tab-store", "data"),
        Input("refresh-interval", "n_intervals"),
    )
    def render_tab(active_tab, _n):
        if not _state:
            return html.Div("Waiting for data...", style={"color": _TEXT_DIM})

        if active_tab == "tab-commands":
            ctx = dash.callback_context
            trigger = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
            if "refresh-interval" in trigger:
                return no_update
            return _render_commands_tab()

        try:
            if active_tab == "tab-map":
                return _render_map_tab()
            elif active_tab == "tab-ledgers":
                return _render_ledgers_tab()
            elif active_tab == "tab-overview":
                return _render_overview_tab()
            elif active_tab == "tab-vehicles":
                return _render_vehicles_tab()
        except Exception as e:
            _log.error(f"Tab render error ({active_tab}): {e}", exc_info=True)
            return html.Div(f"Error loading tab: {e}", style={"color": _RED})
        return html.Div()

    # Command execution
    @app.callback(
        Output("cmd-output", "children"),
        Input({"type": "cmd-btn", "name": dash.ALL}, "n_clicks"),
        State({"type": "cmd-args", "name": dash.ALL}, "value"),
        prevent_initial_call=True,
    )
    def execute_command(n_clicks_list, args_list):
        if not _state:
            return no_update
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update
        trigger = ctx.triggered[0]
        if trigger["value"] is None:
            return no_update
        import json
        prop_id = json.loads(trigger["prop_id"].rsplit(".", 1)[0])
        cmd_name = prop_id["name"]
        args = ""
        for state_entry in (ctx.states_list[0] if ctx.states_list else []):
            if state_entry.get("id", {}).get("name") == cmd_name:
                args = state_entry.get("value") or ""
                break
        output = _state.execute_command(cmd_name, args)
        return html.Pre(
            output,
            style={
                "color": _TEXT, "fontSize": "0.75rem",
                "whiteSpace": "pre-wrap", "margin": "0",
                "fontFamily": "monospace",
            },
        )

    # Prediction detail modal
    @app.callback(
        Output("pred-modal", "is_open"),
        Output("pred-modal-title", "children"),
        Output("pred-modal-body", "children"),
        Input("predictions-table", "active_cell"),
        State("predictions-table", "data"),
        prevent_initial_call=True,
    )
    def show_prediction_detail(active_cell, table_data):
        if not active_cell or not _state:
            return False, "", ""
        row = table_data[active_cell["row"]]
        route_id = row.get("route_id", "")
        direction_id = int(row.get("direction_id", 0))
        scheduled_start = row.get("scheduled_start", "")

        stops = _state.get_prediction_stops(route_id, direction_id, scheduled_start)
        if not stops:
            return True, f"Route {route_id} | Dir {direction_id} | {scheduled_start}", \
                html.Div("No stop-level data available.", style={"color": _TEXT_DIM})

        title = f"Route {route_id} | Dir {direction_id} | {scheduled_start}"
        table = dash_table.DataTable(
            columns=[
                {"name": "Seq", "id": "stop_sequence"},
                {"name": "Stop ID", "id": "stop_id"},
                {"name": "Predicted Arrival", "id": "predicted_arrival"},
                {"name": "Delay (s)", "id": "predicted_delay_sec"},
                {"name": "Crowd", "id": "predicted_crowd_level"},
            ],
            data=stops,
            page_size=20,
            sort_action="native",
            style_table={"overflowX": "auto"},
            style_header=_TABLE_STYLE_HEADER,
            style_cell=_TABLE_STYLE_CELL,
        )
        return True, title, table

    # Vehicle trip detail modal
    @app.callback(
        Output("veh-modal", "is_open"),
        Output("veh-modal-title", "children"),
        Output("veh-modal-body", "children"),
        Input("vehicle-trips-table", "active_cell"),
        State("vehicle-trips-table", "data"),
        prevent_initial_call=True,
    )
    def show_vehicle_detail(active_cell, table_data):
        if not active_cell or not _state:
            return False, "", ""
        row = table_data[active_cell["row"]]

        title = f"Vehicle {row.get('vehicle_id', '?')} | Route {row.get('route_id', '?')} | {row.get('scheduled_start', '')}"

        detail_items = []
        display_fields = [
            ("Vehicle ID", "vehicle_id"), ("Trip ID", "trip_id"),
            ("Route", "route_id"), ("Direction", "direction_id"),
            ("Vehicle Type", "vehicle_type_name"),
            ("Trip Date", "trip_date"), ("Scheduled Start", "scheduled_start"),
            ("Duration (s)", "trip_duration_sec"),
            ("Mean Delay (s)", "mean_delay_sec"), ("Median Delay (s)", "median_delay_sec"),
            ("Max Delay (s)", "max_delay_sec"), ("Min Delay (s)", "min_delay_sec"),
            ("Std Delay (s)", "std_delay_sec"),
            ("Mean Occupancy", "mean_occupancy"), ("Max Occupancy", "max_occupancy"),
            ("Measurements", "measurement_count"),
            ("Preferential Ratio", "preferential_ratio"),
        ]
        for label, key in display_fields:
            val = row.get(key, "N/A")
            if isinstance(val, float):
                val = f"{val:.1f}"
            detail_items.append(html.Div([
                html.Span(f"{label}: ", style={"color": _TEXT_DIM, "fontSize": "0.85rem"}),
                html.Span(str(val), style={"color": _TEXT, "fontSize": "0.85rem"}),
            ], style={"marginBottom": "4px"}))

        return True, title, html.Div(detail_items)


# ============================================================
# Tab Renderers
# ============================================================


def _render_map_tab():
    """Map + active vehicles mini-table below."""
    rows = _state.get_tracking_summary()
    vehicle_table = html.Div()
    if rows:
        vehicle_table = html.Div([
            html.H6("Active Vehicles", style={
                "color": _ACCENT, "marginTop": "12px", "marginBottom": "8px",
                "fontWeight": "bold",
            }),
            dash_table.DataTable(
                columns=[
                    {"name": "ID", "id": "bus_id"},
                    {"name": "Route", "id": "route_id"},
                    {"name": "Direction", "id": "headsign"},
                    {"name": "Speed", "id": "speed"},
                    {"name": "Status", "id": "status"},
                    {"name": "Last Seen", "id": "last_seen"},
                ],
                data=rows,
                page_size=10,
                sort_action="native",
                style_table={"overflowX": "auto"},
                style_header=_TABLE_STYLE_HEADER,
                style_cell=_TABLE_STYLE_CELL,
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
            ),
        ])

    return html.Div([
        _build_map_component(),
        vehicle_table,
    ])


def _render_ledgers_tab():
    """Full-width prediction + vehicle trip tables with all in-memory data."""
    items = []

    # Predictions table
    predictions = _state.get_recent_predictions()
    items.append(html.H6(
        f"Predictions ({len(predictions)} trips)",
        style={"color": _ACCENT, "fontWeight": "bold", "marginBottom": "8px"},
    ))

    if predictions:
        items.append(dash_table.DataTable(
            id="predictions-table",
            columns=[
                {"name": "Route", "id": "route_id"},
                {"name": "Dir", "id": "direction_id"},
                {"name": "Date", "id": "trip_date"},
                {"name": "Start", "id": "scheduled_start"},
                {"name": "Stops", "id": "stop_count"},
                {"name": "Avg Delay", "id": "avg_delay"},
                {"name": "Max Delay", "id": "max_delay"},
            ],
            data=predictions,
            page_size=50,
            sort_action="native",
            style_table={"overflowX": "auto", "marginBottom": "24px"},
            style_header=_TABLE_STYLE_HEADER,
            style_cell={**_TABLE_STYLE_CELL, "cursor": "pointer"},
        ))
    else:
        items.append(html.Div(
            "No predictions recorded yet.",
            style={"color": _TEXT_DIM, "marginBottom": "24px", "padding": "12px"},
        ))

    # Vehicle trips table
    vehicle_trips = _state.get_recent_vehicle_trips()
    items.append(html.H6(
        f"Vehicle Trips ({len(vehicle_trips)} recorded)",
        style={"color": _ACCENT, "fontWeight": "bold", "marginBottom": "8px"},
    ))

    if vehicle_trips:
        items.append(dash_table.DataTable(
            id="vehicle-trips-table",
            columns=[
                {"name": "Vehicle", "id": "vehicle_id"},
                {"name": "Route", "id": "route_id"},
                {"name": "Trip ID", "id": "trip_id"},
                {"name": "Start", "id": "scheduled_start"},
                {"name": "Mean Delay (s)", "id": "mean_delay_sec"},
                {"name": "Measurements", "id": "measurement_count"},
                {"name": "Type", "id": "vehicle_type_name"},
            ],
            data=vehicle_trips,
            page_size=50,
            sort_action="native",
            style_table={"overflowX": "auto"},
            style_header=_TABLE_STYLE_HEADER,
            style_cell={**_TABLE_STYLE_CELL, "cursor": "pointer"},
        ))
    else:
        items.append(html.Div(
            "No vehicle trips recorded yet.",
            style={"color": _TEXT_DIM, "padding": "12px"},
        ))

    return html.Div(items)


def _render_overview_tab():
    """System services, feed timestamp, network summary, ledger info."""
    thread_status = _state.get_service_thread_status()
    feed_ts = _state.get_feed_timestamp()
    geo_stats = _state.get_geocoding_stats()
    stats = _state.get_system_stats()
    ledger_info = _state.get_all_ledger_info()

    items = []

    # Service status
    items.append(html.H6("System Services", style={
        "color": _ACCENT, "marginBottom": "10px", "fontWeight": "bold",
    }))
    for name, status in thread_status.items():
        if status == "running":
            color, icon = _GREEN, "\u25cf"
        elif status == "stopped":
            color, icon = _RED, "\u25cf"
        else:
            color, icon = _TEXT_DIM, "\u25cb"
        items.append(html.Div([
            html.Span(f"{icon} ", style={"color": color}),
            html.Span(name, style={"color": _TEXT, "fontSize": "0.85rem"}),
            html.Span(f" {status}", style={
                "color": _TEXT_DIM, "fontSize": "0.75rem", "float": "right",
            }),
        ], style={"marginBottom": "4px"}))

    items.append(html.Hr(style={"borderColor": _BG_CARD_LIGHT, "margin": "12px 0"}))
    items.append(html.Div([
        html.Span("Last Feed: ", style={"color": _TEXT_DIM, "fontSize": "0.85rem"}),
        html.Span(_format_timestamp(feed_ts), style={"color": _ACCENT, "fontSize": "0.85rem"}),
    ]))
    if geo_stats["enabled"]:
        items.append(html.Div([
            html.Span("Geo Queue: ", style={"color": _TEXT_DIM, "fontSize": "0.85rem"}),
            html.Span(str(geo_stats["pending"]), style={"color": _ACCENT_BLUE, "fontSize": "0.85rem"}),
        ], style={"marginTop": "4px"}))

    # Network summary
    items.append(html.Hr(style={"borderColor": _BG_CARD_LIGHT, "margin": "12px 0"}))
    items.append(html.H6("Network Summary", style={
        "color": _ACCENT, "marginBottom": "8px", "fontWeight": "bold",
    }))
    items.append(html.Div(f"Cities: {stats['city_count']}", style={"color": _TEXT, "fontSize": "0.85rem"}))
    items.append(html.Div(f"Total hexagons: {stats['hexagon_count']}", style={"color": _TEXT, "fontSize": "0.85rem"}))

    # Ledger info
    items.append(html.Hr(style={"borderColor": _BG_CARD_LIGHT, "margin": "12px 0"}))
    items.append(html.H6("Ledgers", style={
        "color": _ACCENT, "marginBottom": "8px", "fontWeight": "bold",
    }))

    topo = ledger_info["topology"]
    items.append(_ledger_card(
        "Topology",
        [("Routes", str(topo.get("routes", "-"))), ("Trips", str(topo.get("trips", "-"))),
         ("Stops", str(topo.get("stops", "-")))] if topo["loaded"] else [("Status", "Not loaded")],
        color=_GREEN if topo["loaded"] else _RED,
    ))

    sched = ledger_info["schedule"]
    items.append(_ledger_card(
        "Schedule",
        [("Routes indexed", str(sched.get("routes_indexed", "-")))] if sched["loaded"] else [("Status", "Not loaded")],
        color=_GREEN if sched["loaded"] else _RED,
    ))

    # Model info
    model_info = _state.get_model_info()
    items.append(html.Hr(style={"borderColor": _BG_CARD_LIGHT, "margin": "12px 0"}))
    items.append(html.H6("Model", style={
        "color": _ACCENT, "marginBottom": "8px", "fontWeight": "bold",
    }))
    items.append(html.Div([
        html.Span("Status: ", style={"color": _TEXT_DIM, "fontSize": "0.85rem"}),
        dbc.Badge(
            "Loaded" if model_info["loaded"] else "Not Loaded",
            color="success" if model_info["loaded"] else "danger",
            className="ms-1",
        ),
        html.Span(f"  {model_info['model_name']}", style={"color": _TEXT, "fontSize": "0.85rem", "marginLeft": "8px"}),
    ]))

    return html.Div(items, style={
        "backgroundColor": _BG_CARD, "borderRadius": "8px",
        "padding": "16px", "border": f"1px solid {_BG_CARD_LIGHT}",
    })


def _render_vehicles_tab():
    """Full-width vehicle tracking table."""
    rows = _state.get_tracking_summary()
    if not rows:
        return html.Div(
            "No active vehicles",
            style={"color": _TEXT_DIM, "textAlign": "center", "padding": "20px"},
        )

    return dash_table.DataTable(
        columns=[
            {"name": "ID", "id": "bus_id"},
            {"name": "Type", "id": "vehicle_type"},
            {"name": "Route", "id": "route_id"},
            {"name": "Direction", "id": "headsign"},
            {"name": "Speed", "id": "speed"},
            {"name": "Status", "id": "status"},
            {"name": "Last Seen", "id": "last_seen"},
            {"name": "Samples", "id": "samples"},
            {"name": "Weather", "id": "weather"},
        ],
        data=rows,
        page_size=50,
        sort_action="native",
        style_table={"overflowX": "auto"},
        style_header=_TABLE_STYLE_HEADER,
        style_cell={**_TABLE_STYLE_CELL, "maxWidth": "180px",
                    "overflow": "hidden", "textOverflow": "ellipsis"},
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


def _render_commands_tab():
    """Command buttons with optional argument inputs."""
    if not _state:
        return html.Div("Waiting for data...", style={"color": _TEXT_DIM})

    commands = _state.get_available_commands()
    if not commands:
        return html.Div("No commands registered", style={"color": _TEXT_DIM})

    skip = {"quit", "help"}
    items = []

    for cmd in commands:
        if cmd["name"] in skip:
            continue
        row_children = [
            dbc.Button(
                cmd["name"],
                id={"type": "cmd-btn", "name": cmd["name"]},
                size="sm", color="info", outline=True,
                style={"marginRight": "8px", "minWidth": "160px", "textAlign": "left"},
            ),
        ]
        if cmd["needs_args"]:
            row_children.append(
                dbc.Input(
                    id={"type": "cmd-args", "name": cmd["name"]},
                    placeholder="args...", size="sm",
                    style={
                        "backgroundColor": _BG_DARK, "color": _TEXT,
                        "border": f"1px solid {_BG_CARD_LIGHT}",
                        "maxWidth": "200px",
                    },
                ),
            )
        else:
            row_children.append(
                dcc.Input(
                    id={"type": "cmd-args", "name": cmd["name"]},
                    type="hidden", value="",
                ),
            )
        items.append(html.Div(
            row_children,
            style={"display": "flex", "alignItems": "center", "marginBottom": "6px"},
        ))

    return html.Div([
        html.H6("Console Commands", style={
            "color": _ACCENT, "marginBottom": "10px", "fontWeight": "bold",
        }),
        html.Div(items),
        html.Hr(style={"borderColor": _BG_CARD_LIGHT, "margin": "12px 0"}),
        html.H6("Output", style={
            "color": _ACCENT, "marginBottom": "6px", "fontWeight": "bold",
        }),
        html.Div(
            id="cmd-output",
            style={
                "backgroundColor": _BG_DARK, "borderRadius": "6px",
                "padding": "10px", "maxHeight": "300px", "overflowY": "auto",
                "border": f"1px solid {_BG_CARD_LIGHT}",
                "fontSize": "0.75rem", "color": _TEXT_DIM,
            },
            children="Click a command to see output here.",
        ),
    ], style={
        "backgroundColor": _BG_CARD, "borderRadius": "8px",
        "padding": "16px", "border": f"1px solid {_BG_CARD_LIGHT}",
    })


def _ledger_card(title, rows, color=_ACCENT):
    row_elements = []
    for label, value in rows:
        row_elements.append(html.Div([
            html.Span(f"{label}: ", style={"color": _TEXT_DIM, "fontSize": "0.8rem"}),
            html.Span(value, style={"color": _TEXT, "fontSize": "0.8rem"}),
        ], style={"marginBottom": "2px"}))

    return html.Div([
        html.Div([
            html.Span("\u25cf ", style={"color": color, "fontSize": "0.7rem"}),
            html.Span(title, style={"color": _TEXT, "fontSize": "0.85rem", "fontWeight": "bold"}),
        ], style={"marginBottom": "4px"}),
        html.Div(row_elements, style={"paddingLeft": "16px"}),
    ], style={
        "backgroundColor": _BG_CARD_LIGHT, "borderRadius": "6px",
        "padding": "8px", "marginBottom": "6px",
    })


# ============================================================
# App Factory & Runner
# ============================================================


def create_app(state: "StateInterface") -> dash.Dash:
    global _state
    _state = state

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        title="ATAC Backend Dashboard",
        suppress_callback_exceptions=True,
    )
    app.layout = _build_layout()
    _register_callbacks(app)
    return app


def start_gui(state: "StateInterface", port: int = 8050):
    app = create_app(state)

    def run_server():
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False, threaded=True)

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    print(f" > Dashboard GUI started at http://localhost:{port}")
    return thread


def stop_gui():
    pass
