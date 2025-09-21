#!/usr/bin/env python3
"""
Streamlit entry point for AI Legal Document Demystifier
This file serves as the main entry point for Streamlit Cloud deployment.
"""

import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the main application
try:
    from Data_Analyst_Agent import create_streamlit_ui
    
    if __name__ == "__main__":
        create_streamlit_ui()
        
except ImportError as e:
    print(f"Import error: {e}")
    import streamlit as st
    st.error(f"❌ Failed to import main application: {e}")
    st.error("Please check that all dependencies are installed correctly.")
    st.code("pip install -r requirements.txt", language="bash")
