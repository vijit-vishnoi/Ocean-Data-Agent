import base64
import os
import requests
import streamlit as st

st.set_page_config(
    page_title="Argo Ocean Data Agent",
    page_icon="🌊",
    layout="wide",
)

try:
    API_URL = st.secrets.get("API_URL", os.getenv("API_URL", "http://127.0.0.1:8000"))
except Exception:
    API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

API_URL = API_URL.rstrip("/")

with st.sidebar:
    st.header("⚙️ Settings")
    st.write("Configure the pipeline behavior.")

    top_k = st.slider(
        "Context Window (Top-K)",
        min_value=10,
        max_value=50,
        value=20,
        step=5,
        help="Number of profile summaries to retrieve via TF-IDF.",
    )

    debug_mode = st.toggle(
        "Developer Debug Mode",
        value=False,
        help="Show generated SQL, retrieved context, and execution latency.",
    )

st.title("🌊 Argo Ocean Data Agent")
st.write("Ask anything about ocean temperature, salinity, and depth profiles.")

with st.form("query_form"):
    query = st.text_input(
        "Enter your question:",
        placeholder="e.g., How does salinity change with depth?",
    )
    submitted = st.form_submit_button("Run Query", type="primary")


def render_chart(b64_str: str | None, container, caption: str) -> None:
    if b64_str:
        image_bytes = base64.b64decode(b64_str)
        container.image(image_bytes, caption=caption, width="stretch")


if submitted:
    if not query.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Analyzing ocean data..."):
            try:
                resp = requests.post(
                    f"{API_URL}/query",
                    json={"query": query.strip(), "top_k": top_k},
                    timeout=60,
                )

                if resp.status_code == 200:
                    data = resp.json()

                    if data.get("error"):
                        st.error(f"Pipeline Error: {data['error']}")

                    summary = data.get("summary")
                    if summary:
                        st.subheader("Analysis & Findings")
                        st.markdown(summary, unsafe_allow_html=True)

                        line_b64 = data.get("line_chart_b64")
                        step_b64 = data.get("step_chart_b64")
                        if line_b64 or step_b64:
                            st.subheader("Visualizations")
                            col1, col2 = st.columns(2)
                            render_chart(line_b64, col1, "Line Chart")
                            render_chart(step_b64, col2, "Step Chart")

                        if debug_mode:
                            st.divider()
                            st.subheader("🛠️ Debug Information")
                            st.metric(
                                "Total Execution Latency",
                                f"{data.get('debug_latency_ms', 0)} ms",
                            )

                            with st.expander("Generated DuckDB SQL Query"):
                                sql = data.get("debug_sql")
                                if sql:
                                    st.code(sql, language="sql")
                                else:
                                    st.write("No SQL generated (Generic Query or Error).")

                            with st.expander("Retrieved Context (TF-IDF)"):
                                context = data.get("debug_context")
                                if context:
                                    st.text(context)
                                else:
                                    st.write("No context retrieved.")
                    else:
                        st.error("The backend did not return a summary.")
                else:
                    st.error(f"Server Error {resp.status_code}: {resp.text}")

            except requests.exceptions.Timeout:
                st.error(
                    "⏱️ Request timed out. If the Render backend was asleep, wait a few seconds and try again."
                )
            except requests.exceptions.ConnectionError:
                st.error(
                    "🚨 Connection Error: Could not reach the backend. Ensure your FastAPI server is running."
                )