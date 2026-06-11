"""Dashboard views package.

Each view module exposes a single ``render()`` function that Streamlit calls.
The view registry in ``dashboard/app.py`` dispatches to each view by lazy
import, so adding or replacing a view never requires editing ``app.py``.
"""
