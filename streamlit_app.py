import streamlit as st
import requests
import pandas as pd

API_URL_GET_LONGEST_REVIEW = 'http://localhost:5000/get_longest_review'
API_URL_GET_RESPONSE = 'http://localhost:5000/get_response'

# Load the dataset
df = pd.read_excel(r"your_path//data_533.xlsx")

st.title("E-Commerce Help Assitant")

# Dropdown list for product titles
product_titles = df['product_title'].unique()
product_name = st.selectbox("Select a Bluetooth headset:", product_titles)

# User selects feature
feature = st.radio(
    "Select feature:",
    ("Get Detailed Review and Summary", "Get Response to a Query")
)

if feature == "Get Detailed Review and Summary":
    if st.button("Fetch Detailed Review and Summary"):
        if product_name:
            with st.spinner("Fetching response..."):
                response = requests.post(API_URL_GET_LONGEST_REVIEW, json={'product_name': product_name})
                if response.status_code == 200:
                    result = response.json()
                    longest_review = result.get('longest_review')
                    summary = result.get('summary')
                    st.subheader("Detailed Review")
                    st.write(longest_review)
                    st.subheader("Summary")
                    st.write(summary)
                else:
                    st.error("Error fetching response")
        else:
            st.warning("Please select a product.")
else:
    query = st.text_input("Enter your query about Bluetooth headsets:")
    if st.button("Get Response"):
        if query:
            with st.spinner("Fetching response..."):
                response = requests.post(API_URL_GET_RESPONSE, json={'query': query})
                if response.status_code == 200:
                    result = response.json()
                    st.subheader("Response")
                    st.write(result['response'])
                else:
                    st.error("Error fetching response")
        else:
            st.warning("Please enter a query.")

