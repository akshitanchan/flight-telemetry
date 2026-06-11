"""Flight Telemetry Dashboard — Streamlit entrypoint.

CMD (fixed by infra/docker/dashboard.Dockerfile):
    streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0

Architecture — lazy view registry
----------------------------------
Each sidebar entry maps a human-readable label to a dotted module path inside
``dashboard.views``.  When the user selects a view, the module is imported on
demand and its ``render()`` function is called.

Extending the dashboard (ws5-02, ws5-03, …)
--------------------------------------------
Downstream tasks **only** need to overwrite their stub file
(``dashboard/views/ask_ai.py`` or ``dashboard/views/platform_health.py``) with
a real implementation.  This file never needs to be edited again — the registry
already contains all three entries and dispatches by module path.
"""

import importlib
import sys
from pathlib import Path

# Streamlit executes this file with dashboard/ as sys.path[0]. Add the
# repository root so lazy imports such as dashboard.views.operational resolve
# under the documented `streamlit run dashboard/app.py` entrypoint.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# ---------------------------------------------------------------------------
# View registry
# ---------------------------------------------------------------------------
# Each entry: (sidebar_label, dotted_module_path_within_dashboard_views)
# ws5-02 overwrites dashboard/views/ask_ai.py          — no app.py change needed
# ws5-03 overwrites dashboard/views/platform_health.py — no app.py change needed
_VIEW_REGISTRY: list[tuple[str, str]] = [
    ("Operational", "dashboard.views.operational"),
    ("Ask AI", "dashboard.views.ask_ai"),
    ("Platform Health", "dashboard.views.platform_health"),
]


def _render_view(module_path: str) -> None:
    """Import *module_path* lazily and call its ``render()`` function."""
    module = importlib.import_module(module_path)
    module.render()


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Flight Telemetry",
        page_icon="✈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("Flight Telemetry")
    st.sidebar.caption("Operational Intelligence Platform")

    labels = [label for label, _ in _VIEW_REGISTRY]
    selected_label = st.sidebar.radio("Navigation", labels, label_visibility="collapsed")

    # Dispatch to the selected view
    for label, module_path in _VIEW_REGISTRY:
        if label == selected_label:
            _render_view(module_path)
            break


if __name__ == "__main__":
    main()
