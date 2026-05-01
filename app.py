import streamlit as st
import pandas as pd
import json
from datetime import datetime
import uuid
from google.oauth2.service_account import Credentials
import gspread

from utils import log_error, save_data, init_session_state
from questions import screening_questions, section_a_questions, section_b_questions, section_c_questions

# Initialize session state
init_session_state()

st.set_page_config(
    page_title="New Survey Bot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("New Survey Application")
st.write("Welcome to the new survey bot! This is a fresh starting point based on the reference project.")

if st.button("Start Survey"):
    st.success("Survey started! (Add your logic here)")
