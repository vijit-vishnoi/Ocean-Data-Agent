import streamlit as st
import requests
import base64

st.set_page_config(page_title="Argo Ocean Agent", layout="wide")

with st.sidebar:
    st.header("⚙️ Settings")
    st.write("Configure the pipeline behavior.")
    
    top_k = st.slider(
        "Context Window (FAISS Top-K)", 
        min_value=10, 
        max_value=50, 
        value=20, 
        step=5,
        help="Number of chunks to retrieve for semantic search context."
    )
    
    debug_mode = st.toggle(
        "Developer Debug Mode", 
        value=False,
        help="Show generated SQL, raw context, and execution latency."
    )

st.title("🌊 Argo Ocean Data Agent")
st.write("Ask me anything about ocean temperature, salinity, and depth profiles!")

query = st.text_input("Enter your question:", placeholder="e.g., How does salinity change with depth?")

if st.button("Run Query"):
    if query.strip() != "":
        with st.spinner("Analyzing ocean data..."):
            try:
                payload = {
                    "query": query,
                    "top_k": top_k
                }
                resp = requests.post("http://127.0.0.1:8000/query", json=payload)
                
                if resp.status_code == 200:
                    data = resp.json()
                    
                    if "error" in data and data["error"]:
                        st.error(f"Pipeline Error: {data['error']}")
                    
                    if "summary" in data:
                        st.subheader("Analysis & Findings")
                        st.write(data["summary"])
                        
                        col1, col2 = st.columns(2)
                        
                        def decode_and_render_image(b64_str, col, caption):
                            if b64_str:
                                image_bytes = base64.b64decode(b64_str)
                                col.image(image_bytes, caption=caption, use_container_width=True)
                                
                        decode_and_render_image(data.get("line_chart_b64"), col1, "Line Chart")
                        decode_and_render_image(data.get("step_chart_b64"), col2, "Step Chart")

                        if debug_mode:
                            st.divider()
                            st.subheader("🛠️ Debug Information")
                            
                            st.metric("Total Execution Latency", f"{data.get('debug_latency_ms', 0)} ms")
                            
                            with st.expander("Generated DuckDB SQL Query"):
                                sql = data.get("debug_sql")
                                if sql:
                                    st.code(sql, language="sql")
                                else:
                                    st.write("No SQL generated (Generic Query or Error).")
                                    
                            with st.expander("Retrieved Context (FAISS)"):
                                context = data.get("debug_context")
                                if context:
                                    st.text(context)
                                else:
                                    st.write("No context retrieved.")
                    else:
                        st.error("Something went wrong. The backend didn't return a summary.")
                else:
                    st.error(f"Server Error {resp.status_code}: Please check your FastAPI terminal.")
            
            except requests.exceptions.ConnectionError:
                st.error("🚨 Connection Error: Could not reach the backend. Ensure `uvicorn main:app` is running.")
    else:
        st.warning("Please enter a question first!")