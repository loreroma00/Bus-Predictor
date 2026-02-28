"""
Debug GUI - NiceGUI-based debugging and data inspection interface.

This module provides a web-based GUI that runs parallel to the main application,
allowing real-time inspection of data structures and ingestion metrics.

Three views:
1. Inspection View: Hierarchical data browser (Observatory → Cities → Buses → Diaries)
2. Ingestion View: Live metrics table (like print_tracking_summary but richer)
3. Raw Data View: JSON explorer for deep data structure inspection
"""

from nicegui import ui, app
import threading
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state_interface import StateInterface


# Global state reference (set by start_gui)
_state: "StateInterface" = None
_refresh_interval: float = 5.0  # Seconds between auto-refresh


def _format_timestamp(ts: float) -> str:
    """Format Unix timestamp to readable string."""
    if not ts:
        return "N/A"
    from datetime import datetime

    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def create_header():
    """Create the header with navigation tabs."""
    with ui.header().classes("bg-slate-800 text-white items-center"):
        ui.label("🚌 ATAC Data Inspector").classes("text-xl font-bold")
        ui.space()
        with ui.row().classes("gap-4"):
            ui.link("Inspection", "/").classes("text-white hover:text-blue-300")
            ui.link("Hexagons", "/hexagons").classes("text-white hover:text-blue-300")
            ui.link("Ingestion", "/ingestion").classes("text-white hover:text-blue-300")
            ui.link("Raw Data", "/raw").classes("text-white hover:text-blue-300")


def create_stats_card(title: str, value: str, icon: str = "📊"):
    """Create a statistics card."""
    with ui.card().classes("p-4 bg-slate-700 text-white min-w-[140px]"):
        ui.label(f"{icon} {title}").classes("text-sm text-slate-300")
        ui.label(value).classes("text-2xl font-bold")


# ============================================================
# Inspection View (Debugging)
# ============================================================


@ui.page("/")
def inspection_page():
    """Main inspection page with hierarchical data browser."""
    create_header()

    with ui.column().classes("w-full p-4 gap-4"):
        # Stats overview
        stats_container = ui.row().classes("gap-4 flex-wrap")

        # Main content
        with ui.row().classes("w-full gap-4"):
            # Left panel: Selection
            with ui.card().classes("w-1/3 min-w-[300px] p-4"):
                ui.label("📂 Bus Selector").classes("text-lg font-bold mb-2")

                # Bus selector dropdown - format: {value: label}
                bus_options = {}
                if _state:
                    buses = _state.get_all_buses("Rome")
                    for b in buses:  # Show ALL buses
                        bus_id = b["id"]
                        label = b.get("label", bus_id)
                        route = b.get("route_id", "?")
                        loc = (b.get("location_name") or "Unknown")[:15]
                        bus_options[bus_id] = f"{label} | {route} | {loc}"

                bus_select = ui.select(
                    label="Select a bus to inspect",
                    options=bus_options,
                    with_input=True,
                ).classes("w-full")

                ui.button(
                    "🔍 Inspect Selected Bus",
                    on_click=lambda: show_bus(bus_select.value),
                ).classes("w-full mt-2")

                ui.separator().classes("my-4")

                # Quick stats
                if _state:
                    stats = _state.get_system_stats()
                    ui.label(
                        f"Active: {stats['active_buses']} | Deposit: {stats['deposit_buses']} | Observers: {stats['observer_count']}"
                    )

                # Refresh button
                ui.button(
                    "🔄 Refresh Bus List", on_click=lambda: ui.navigate.reload()
                ).classes("w-full mt-2")

            # Right panel: Detail view
            with ui.card().classes("flex-1 p-4"):
                ui.label("🔍 Details").classes("text-lg font-bold mb-2")
                detail_container = ui.column().classes("w-full")

                with detail_container:
                    ui.label("Select a bus from the dropdown to view details").classes(
                        "text-slate-500"
                    )

    def show_bus(bus_id):
        """Show bus details."""
        detail_container.clear()

        if not bus_id or not _state:
            with detail_container:
                ui.label("Please select a bus").classes("text-amber-500")
            return

        bus = _state.get_bus("Rome", bus_id)
        if not bus:
            with detail_container:
                ui.label(f"Bus {bus_id} not found").classes("text-red-500")
            return

        with detail_container:
            ui.label(f"🚌 Bus {bus.get('label', bus['id'])}").classes("text-xl font-bold")

            # Status badge
            status = bus.get("status", "UNKNOWN")
            status_color = "green" if status == "ACTIVE" else "orange"
            ui.badge(status, color=status_color).classes("mb-2")

            # Key info in a grid
            with ui.grid(columns=2).classes("gap-2 mb-4"):
                ui.label("Route:").classes("font-bold")
                ui.label(str(bus.get("route_id", "N/A")))

                ui.label("Trip:").classes("font-bold")
                ui.label(str(bus.get("trip_id", "N/A")))

                ui.label("Direction:").classes("font-bold")
                ui.label(str(bus.get("direction", "N/A"))[:30])

                ui.label("Location:").classes("font-bold")
                ui.label(str(bus.get("location_name", "N/A"))[:30])

                ui.label("Last Seen:").classes("font-bold")
                ui.label(_format_timestamp(bus.get("last_seen", 0)))

            # GPS Data
            if "gps" in bus:
                with ui.expansion("📡 GPS Data", icon="gps_fixed").classes("w-full"):
                    ui.code(json.dumps(bus["gps"], indent=2, default=str))

            # Observer Data
            if "observer" in bus:
                with ui.expansion("👁️ Observer Data", icon="visibility").classes(
                    "w-full"
                ):
                    ui.code(json.dumps(bus["observer"], indent=2, default=str))

                    # View diary button
                    if bus["observer"].get("has_diary"):
                        ui.button(
                            "📖 View Diary", on_click=lambda: show_diary(bus_id)
                        ).classes("mt-2")

            # Full JSON
            with ui.expansion("📋 Full JSON Data", icon="code").classes("w-full"):
                ui.code(json.dumps(bus, indent=2, default=str))

    def show_diary(bus_id):
        """Show diary for a bus."""
        detail_container.clear()

        if not _state:
            return

        diary = _state.get_observer_diary(bus_id)
        if not diary:
            with detail_container:
                ui.label("Diary not found").classes("text-red-500")
            return

        with detail_container:
            ui.button("← Back", on_click=lambda: show_bus(bus_id)).classes("mb-2")
            ui.label(f"📖 Diary for {bus_id}").classes("text-xl font-bold")
            ui.label(
                f"Trip: {diary['trip_id']} | Measurements: {diary['total_measurements']}"
            )

            ui.code(json.dumps(diary, indent=2, default=str)).classes(
                "max-h-[500px] overflow-auto"
            )

    # Refresh stats
    def refresh_stats():
        stats_container.clear()
        if not _state:
            return

        stats = _state.get_system_stats()
        with stats_container:
            create_stats_card("Active Buses", str(stats["active_buses"]), "🟢")
            create_stats_card("In Deposit", str(stats["deposit_buses"]), "🟡")
            create_stats_card("Observers", str(stats["observer_count"]), "👁️")
            create_stats_card("Hexagons", str(stats["hexagon_count"]), "🔷")

            geo_stats = _state.get_geocoding_stats()
            if geo_stats["enabled"]:
                create_stats_card("Geo Queue", str(geo_stats["pending"]), "📍")

    refresh_stats()
    ui.timer(_refresh_interval, refresh_stats)


# ============================================================
# Hexagon View (Traffic & Weather)
# ============================================================


@ui.page("/hexagons")
def hexagon_page():
    """Hexagon inspection page."""
    create_header()

    with ui.column().classes("w-full p-4 gap-4"):
        ui.label("🌐 Hexagon Inspector").classes("text-2xl font-bold")

        # Traffic coverage stats
        if _state:
            traffic_stats = _state.get_traffic_stats("Rome")
            total = traffic_stats["total_hexagons"]
            with_data = traffic_stats["with_traffic"]
            pct = (with_data / total * 100) if total > 0 else 0
            ui.label(
                f"Traffic coverage: {with_data:,} / {total:,} hexagons ({pct:.1f}%)"
            ).classes("text-slate-500 text-sm")

        with ui.row().classes("w-full gap-4"):
            # Left: Selector
            with ui.card().classes("w-1/3 min-w-[300px] p-4"):
                ui.label("Select Hexagon").classes("font-bold")

                hex_options = {}
                if _state:
                    hexs = _state.get_hexagons("Rome")
                    # Filter to only hexagons WITH traffic data
                    hexs_with_data = [h for h in hexs if h.get("speed_ratio", 0) > 0]
                    # Sort by speed_ratio ascending (lower ratio = more congestion)
                    hexs_with_data.sort(key=lambda x: x.get("speed_ratio", 1))

                    for h in hexs_with_data[:500]:  # Show top 500 most congested
                        hid = h["hex_id"]
                        ratio = h.get("speed_ratio", 0)
                        jam_pct = (1 - ratio) * 100
                        hex_options[hid] = f"{hid} | Congestion: {jam_pct:.0f}%"

                hex_select = ui.select(
                    label="Hexagons with Traffic (sorted by congestion)",
                    options=hex_options,
                    with_input=True,
                ).classes("w-full")

                ui.button(
                    "Show Details", on_click=lambda: show_hex(hex_select.value)
                ).classes("mt-2")

            # Right: Details
            with ui.card().classes("flex-1 p-4"):
                ui.label("Details").classes("font-bold mb-2")
                detail_container = ui.column().classes("w-full")

    def show_diary(bus_id, from_hex_id):
        """Show diary for a bus within Hexagon view."""
        detail_container.clear()

        if not _state:
            return

        diary = _state.get_observer_diary(bus_id)
        if not diary:
            with detail_container:
                ui.label("Diary not found").classes("text-red-500")
                ui.button(
                    "← Back to Hexagon", on_click=lambda: show_hex(from_hex_id)
                ).classes("mt-2")
            return

        with detail_container:
            ui.button(
                "← Back to Hexagon", on_click=lambda: show_hex(from_hex_id)
            ).classes("mb-2")
            ui.label(f"📖 Diary for {bus_id}").classes("text-xl font-bold")
            ui.label(
                f"Trip: {diary['trip_id']} | Measurements: {diary['total_measurements']}"
            )

            ui.code(json.dumps(diary, indent=2, default=str)).classes(
                "max-h-[500px] overflow-auto"
            )

    def show_hex(hex_id):
        detail_container.clear()
        if not hex_id or not _state:
            return

        # Fetch detailed hex data
        hex_data = _state.get_hexagon_details("Rome", hex_id)

        if not hex_data:
            with detail_container:
                ui.label("Hexagon not found").classes("text-red-500")
            return

        with detail_container:
            ui.label(f"Hexagon {hex_id}").classes("text-xl font-bold")

            # Coordinates
            center = hex_data.get("center", {})
            ui.label(
                f"📍 {center.get('lat', 0):.6f}, {center.get('lon', 0):.6f}"
            ).classes("text-slate-500 text-sm mb-2")

            # Streets
            with ui.expansion("🛣️ Streets", icon="add_road").classes("w-full"):
                streets = hex_data.get("streets", [])
                if streets:
                    for s in streets:
                        ui.label(s).classes("ml-4")
                else:
                    ui.label("No streets mapped").classes("ml-4 italic text-slate-400")

            # Preferential Lanes
            with ui.expansion("🚌 Preferential Lanes", icon="edit_road").classes(
                "w-full"
            ):
                prefs = hex_data.get("preferentials", [])
                if prefs:
                    ui.label(f"Angles: {prefs}").classes("ml-4")
                else:
                    ui.label("No preferential lanes").classes(
                        "ml-4 italic text-slate-400"
                    )

            # Buses inside
            buses = hex_data.get("buses", [])
            with ui.expansion(
                f"🚌 Buses ({len(buses)})", icon="directions_bus"
            ).classes("w-full"):
                if buses:
                    for bus in buses:
                        try:
                            with ui.card().classes("w-full p-2 mb-2 bg-slate-50"):
                                with ui.row().classes(
                                    "w-full items-center justify-between"
                                ):
                                    # Safe string conversion before slicing
                                    direction = str(bus.get("direction") or "Unknown")
                                    route = str(bus.get("route_id") or "?")

                                    ui.label(f"{route} → {direction[:15]}...").classes(
                                        "font-bold"
                                    )

                                    status = str(bus.get("status", "UNKNOWN"))
                                    color = "green" if status == "ACTIVE" else "orange"
                                    ui.badge(status, color=color)

                                ui.label(f"ID: {bus.get('id')}").classes(
                                    "text-xs text-slate-500"
                                )

                                # Link to diary
                                ui.button(
                                    "View Diary",
                                    on_click=lambda b=bus.get("id"): show_diary(
                                        b, hex_id
                                    ),
                                ).props("size=sm").classes("mt-2 w-full")
                        except Exception as e:
                            ui.label(f"⚠ Error bus {bus.get('id')}: {e}").classes(
                                "text-red-500 text-xs"
                            )
                else:
                    ui.label("No buses in this hexagon").classes(
                        "ml-4 italic text-slate-400"
                    )

            # Traffic Card
            with ui.card().classes("w-full bg-slate-100 p-2 mt-2"):
                ui.label("🚗 Traffic").classes("font-bold")

                c_speed = hex_data.get("current_speed") or 0.0
                f_speed = hex_data.get("flow_speed") or 0.0
                ratio = hex_data.get("speed_ratio") or 0.0
                jam = (1 - ratio) if ratio > 0 else 0

                # Overall congestion bar
                color = "green"
                if jam > 0.3:
                    color = "orange"
                if jam > 0.6:
                    color = "red"

                ui.linear_progress(value=jam, color=color).classes("my-2")
                ui.label(f"Overall Congestion: {jam * 100:.1f}%").classes("mb-2")

                with ui.grid(columns=2).classes("gap-2 mb-3"):
                    ui.label(f"Avg Current: {c_speed:.1f} kph")
                    ui.label(f"Avg Free Flow: {f_speed:.1f} kph")

                # Per-direction traffic
                traffic_by_dir = hex_data.get("traffic_by_direction", {})
                if traffic_by_dir:
                    with ui.expansion(
                        "📍 Traffic by Direction", icon="explore"
                    ).classes("w-full"):
                        # Display in compass order
                        directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
                        with ui.grid(columns=4).classes("gap-1 text-sm"):
                            for d in directions:
                                data = traffic_by_dir.get(d, {})
                                spd = data.get("current_speed", 0)
                                r = data.get("speed_ratio", 0)
                                cong = (1 - r) * 100 if r > 0 else 0

                                # Color code based on congestion
                                bg = "bg-green-100"
                                if cong > 30:
                                    bg = "bg-orange-100"
                                if cong > 60:
                                    bg = "bg-red-100"
                                if spd == 0:
                                    bg = "bg-slate-50"

                                with ui.card().classes(f"p-1 {bg}"):
                                    ui.label(f"{d}").classes("font-bold text-center")
                                    if spd > 0:
                                        ui.label(f"{spd:.0f} kph").classes(
                                            "text-xs text-center"
                                        )
                                        ui.label(f"{cong:.0f}%").classes(
                                            "text-xs text-center text-slate-500"
                                        )
                                    else:
                                        ui.label("--").classes(
                                            "text-xs text-center text-slate-400"
                                        )

            # Weather Card
            weather = hex_data.get("weather", {})
            with ui.card().classes("w-full bg-blue-50 p-2 mt-2"):
                ui.label("Current Weather").classes("font-bold")
                temp = weather.get("temperature") or 0.0
                precip = weather.get("precip_intensity") or 0.0

                ui.label(f"Temp: {temp:.1f}°C")
                ui.label(f"Precip: {precip:.1f} mm/h")


# ============================================================
# Ingestion View (Metrics)
# ============================================================


@ui.page("/ingestion")
def ingestion_page():
    """Ingestion metrics page showing live tracking data."""
    create_header()

    # Define columns once
    columns = [
        {"name": "bus_id", "label": "Bus ID", "field": "bus_id", "sortable": True},
        {"name": "vehicle_type", "label": "Type", "field": "vehicle_type", "sortable": True},
        {"name": "route_id", "label": "Route", "field": "route_id", "sortable": True},
        {"name": "headsign", "label": "Direction", "field": "headsign"},
        {"name": "location", "label": "Location", "field": "location"},
        {"name": "speed", "label": "Speed", "field": "speed"},
        {"name": "status", "label": "Status", "field": "status", "sortable": True},
        {"name": "last_seen", "label": "Last Seen", "field": "last_seen"},
        {"name": "weather", "label": "Weather", "field": "weather"},
        {"name": "samples", "label": "Samples", "field": "samples", "sortable": True},
    ]

    with ui.column().classes("w-full p-4 gap-4"):
        # Stats row - these will be updated
        stats_container = ui.row().classes("gap-4 flex-wrap")

        # Tracking table - created ONCE, rows updated dynamically
        with ui.card().classes("w-full p-4"):
            ui.label("📊 Live Tracking Summary").classes("text-lg font-bold mb-2")

            # Get initial data
            initial_rows = []
            if _state:
                initial_rows = _state.get_tracking_summary()

            # Create table ONCE
            tracking_table = ui.table(
                columns=columns,
                rows=initial_rows,
                row_key="bus_id",
                pagination={"rowsPerPage": 50},  # More rows per page
            ).classes("w-full")

            # Apply status coloring via slot
            tracking_table.add_slot(
                "body-cell-status",
                """
                <q-td :props="props">
                    <q-badge :color="props.value === 'ACTIVE' ? 'green' : 'orange'">
                        {{ props.value }}
                    </q-badge>
                </q-td>
            """,
            )

    def refresh_stats():
        """Refresh only the stats (not the table)."""
        stats_container.clear()
        if not _state:
            return

        stats = _state.get_system_stats()
        geo_stats = _state.get_geocoding_stats()
        feed_ts = _state.get_feed_timestamp()

        with stats_container:
            create_stats_card("Active Buses", str(stats["active_buses"]), "🟢")
            create_stats_card("In Deposit", str(stats["deposit_buses"]), "🟡")
            create_stats_card("Observers", str(stats["observer_count"]), "👁️")
            create_stats_card("Feed Time", _format_timestamp(feed_ts), "📡")
            if geo_stats["enabled"]:
                create_stats_card("Geo Queue", str(geo_stats["pending"]), "📍")

    def refresh_table_rows():
        """Update table rows without recreating the table (preserves pagination)."""
        if not _state:
            return
        new_rows = _state.get_tracking_summary()
        tracking_table.rows = new_rows
        tracking_table.update()

    # Initial load
    refresh_stats()

    # Auto-refresh timers - separate for stats and table
    ui.timer(_refresh_interval, refresh_stats)
    ui.timer(_refresh_interval + 0.5, refresh_table_rows)  # Slight offset


# ============================================================
# Raw Data View (Deep Inspection)
# ============================================================


@ui.page("/raw")
def raw_data_page():
    """Raw data exploration page for deep debugging."""
    create_header()

    with ui.column().classes("w-full p-4 gap-4"):
        ui.label("🔬 Raw Data Explorer").classes("text-2xl font-bold")
        ui.label("Inspect the actual data structures in memory").classes(
            "text-slate-500"
        )

        # Data source selector
        with ui.row().classes("gap-4 items-center"):
            data_source = ui.select(
                label="Select Data Source",
                options=[
                    "Observatory Summary",
                    "All Buses (Rome)",
                    "All Observers",
                    "All Diaries",
                    "Hexagon List",
                    "Ledger Stats",
                ],
                value="Observatory Summary",
            ).classes("w-64")

            # Manual bus ID input
            bus_id_input = ui.input(label="Or enter specific Bus ID").classes("w-48")

        ui.button("🔄 Load Data", on_click=lambda: load_data()).classes("mt-2")

        # JSON display
        json_container = ui.column().classes("w-full")

    def load_data():
        """Load and display the selected data source."""
        json_container.clear()

        if not _state:
            with json_container:
                ui.label("State not available").classes("text-red-500")
            return

        # Check if specific bus ID was entered
        if bus_id_input.value:
            bus = _state.get_bus("Rome", bus_id_input.value)
            if bus:
                data = {"bus": bus}
                diary = _state.get_observer_diary(bus_id_input.value)
                if diary:
                    data["diary"] = diary
            else:
                data = {"error": f"Bus {bus_id_input.value} not found"}

            with json_container:
                ui.label(f"Data for Bus: {bus_id_input.value}").classes(
                    "font-bold text-lg"
                )
                ui.code(
                    json.dumps(data, indent=2, default=str), language="json"
                ).classes("w-full max-h-[600px] overflow-auto")
            return

        source = data_source.value
        data = {}

        try:
            if source == "Observatory Summary":
                data = {
                    "system_stats": _state.get_system_stats(),
                    "geocoding_stats": _state.get_geocoding_stats(),
                    "city_names": _state.get_city_names(),
                    "observer_count": len(_state.get_observers()),
                }
                for city in _state.get_city_names():
                    data[f"city_{city}"] = _state.get_city_summary(city)

            elif source == "All Buses (Rome)":
                data = {
                    "active_buses": _state.get_all_buses("Rome"),
                    "deposit_buses": _state.get_deposit_buses("Rome"),
                }

            elif source == "All Observers":
                data = {"observers": _state.get_observers()}

            elif source == "All Diaries":
                # Get diary info for each observer
                observers = _state.get_observers()
                diaries = {}
                for obs in observers:
                    bus_id = obs.get("bus_id")
                    if bus_id:
                        diary = _state.get_observer_diary(bus_id)
                        if diary:
                            diaries[bus_id] = diary
                data = {"diaries": diaries}

            elif source == "Hexagon List":
                data = {"hexagons": _state.get_hexagons("Rome")}

            elif source == "Ledger Stats":
                data = _state.get_ledger_stats()

        except Exception as e:
            data = {"error": str(e), "type": type(e).__name__}

        with json_container:
            ui.label(f"Data for: {source}").classes("font-bold text-lg")
            ui.label(f"Total items in response: {len(str(data))} chars").classes(
                "text-sm text-slate-500"
            )

            # Pretty JSON display
            json_str = json.dumps(data, indent=2, default=str)
            ui.code(json_str, language="json").classes(
                "w-full max-h-[600px] overflow-auto"
            )

    # Initial load
    load_data()


# ============================================================
# GUI Runner
# ============================================================


def start_gui(state: "StateInterface", port: int = 8080):
    """
    Start the NiceGUI server in a background thread.

    Args:
        state: StateInterface instance for data access.
        port: Port to run the web server on.
    """
    global _state
    _state = state

    def run_server():
        ui.run(
            port=port,
            title="ATAC Data Inspector",
            reload=False,
            show=False,  # Don't auto-open browser
        )

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    print(f" > Debug GUI started at http://localhost:{port}")
    return thread


def stop_gui():
    """Stop the NiceGUI server."""
    try:
        app.shutdown()
    except Exception:
        pass
