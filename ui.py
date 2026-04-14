import streamlit as st
import requests

st.set_page_config(page_title="Argo Ocean Agent", layout="wide")

st.title("🌊 Argo Ocean Data Agent")
st.write("Ask me anything about ocean temperature, salinity, and depth profiles!")

query = st.text_input("Enter your question:", placeholder="e.g., How does salinity change with depth?")

if st.button("Run Query"):
    if query.strip() != "":
        with st.spinner("Analyzing ocean data..."):
            try:
                
                resp = requests.post("http://127.0.0.1:8000/query", json={"query": query})
                
                if resp.status_code == 200:
                    data = resp.json()
                    
                    
                    if "summary" in data:
                        st.subheader("Analysis & Charts")
                        
                        
                        st.markdown(data["summary"], unsafe_allow_html=True)
                    else:
                        st.error("Something went wrong. The backend didn't send a summary.")
                else:
                    st.error(f"Server Error {resp.status_code}: Please check your FastAPI terminal.")
            
            except requests.exceptions.ConnectionError:
                st.error("🚨 Connection Error: Could not reach the backend. Are you sure `main.py` is running?")
    else:
        st.warning("Please enter a question first!")