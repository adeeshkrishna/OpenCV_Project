import streamlit as st
from streamlit_option_menu import option_menu
import base64

# Import the different pages
from introduction import show as show_introduction
from about_project import show as show_about_project
from smart_ad import show as show_smart_ad
from future_scope import show as show_future_scope

st.set_page_config(
    page_title="SmartAd",
    page_icon=":tv:"
)

# Sidebar setup for navigation
with st.sidebar:
    selected = option_menu(
        menu_title="SmartAd Menu",
        options=["Introduction", "About the Project", "SmartAd", "Future Scope"],
        icons=["house", "info", "camera-video", "lightbulb"],
        menu_icon="cast",
        default_index=0
    )

# Add background
def add_bg_from_local(image_file):
    try:
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(data:image/{"png"};base64,{encoded_string});
                background-size: cover;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.error(f"Background image file '{image_file}' not found.")

if __name__ == '__main__':
    add_bg_from_local('BG.png')

# Display the selected page
if selected == "Introduction":
    show_introduction()
elif selected == "About the Project":
    show_about_project()
elif selected == "SmartAd":
    show_smart_ad()
elif selected == "Future Scope":
    show_future_scope()
