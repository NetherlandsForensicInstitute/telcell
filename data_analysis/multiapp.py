from typing import Callable

import streamlit as st

class MultiApp:
    """Framework for combining multiple streamlit applications.
    Usage:
        def foo():
            st.title("Hello Foo")
        def bar():
            st.title("Hello Bar")
        app = MultiApp()
        app.add_app("Foo", foo)
        app.add_app("Bar", bar)
        app.run()
    It is also possible keep each application as a class in separate files.
        import foo
        import bar
        app = MultiApp()
        app.add_app("Foo", foo.app)
        app.add_app("Bar", bar.app)
        app.run()
    """

    def __init__(self):
        self._apps = {}

    def add_app(self, title: str, render_func: Callable[[], None]):
        """
        Adds a new application.

        :param title: The title of the app. Appears in the dropdown menu in the
         sidebar.
        :param render_func: A python callable for rendering this app.
        """
        self._apps[title] = render_func

    def run(self):
        """
        Run the selected dashboard.
        """
        titles = list(self._apps.keys())

        selected_db = st.sidebar.selectbox(
            'Choose dashboard:', titles, key='selected_db')

        self._apps[selected_db]()