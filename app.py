"""
JCSS CodeCal 2.0 — the graphical interface.

    streamlit run app.py

Learning mode walks through the six lessons interactively; lab mode is the
research tool: edit the probabilistic model, run the status quo, calibrate,
compare, test the assumptions. Both use the same codecal functions as the
lessons and the command line.
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title='JCSS CodeCal 2.0', page_icon='📐', layout='wide')

from codecal.gui import lab, learning          # noqa: E402  (after set_page_config)

MODES = {
    'Learning mode': learning.render,
    'Lab mode': lab.render,
}

with st.sidebar:
    st.markdown('### JCSS CodeCal 2.0')
    mode = st.radio('Mode', list(MODES), key='app_mode',
                    help='Learning: the lessons, step by step. Lab: the full calibration tool.')
    if sys.platform == 'emscripten':
        st.info('Running in your browser — nothing is uploaded. A sweep takes a few seconds here, '
                'a calibration a few minutes; the installed version is several times faster.', icon='🌐')
    st.divider()

MODES[mode]()

with st.sidebar:
    st.divider()
    st.caption('Reliability-based calibration of partial factors, EN 1990:2023 · '
               'model: AHG "Reliability Background of the Eurocodes" (2021). '
               'Code: `codecal/`, lessons: `lessons/`.')
