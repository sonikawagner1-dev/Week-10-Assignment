import streamlit as st

st.set_page_config(page_title="Week 10 Assignment", page_icon=":sparkles:")

st.title("Week 10 Assignment")
st.write("Your Streamlit app is running.")

with st.container():
    st.subheader("Status")
    st.success("Deployment entrypoint loaded from app.py")
