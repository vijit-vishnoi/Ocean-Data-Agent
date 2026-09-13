import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Any
from logger import get_logger

logger = get_logger(__name__)

def render_depth_profile_charts(df: Optional[pd.DataFrame], retrieved: List[Dict[str, Any]], user_query: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Renders the data as a line chart and step chart, returning them as base64 encoded strings.
    
    Args:
        df (Optional[pd.DataFrame]): The dataframe resulting from the SQL query.
        retrieved (List[Dict]): The retrieved vector search context (fallback for charting).
        user_query (str): The user's query, used to infer charting labels.
        
    Returns:
        Tuple[Optional[str], Optional[str]]: Base64 encoded strings for the line chart and step chart.
    """
    plt.style.use('ggplot')

    if df is not None and not df.empty and len(df.columns) >= 2:
        try:
            x_col = df.columns[0]
            y_col = df.columns[1]
            if not pd.api.types.is_numeric_dtype(df[y_col]):
                x_col, y_col = y_col, x_col
            df = df.sort_values(x_col) if pd.api.types.is_datetime64_any_dtype(df[x_col]) else df
            
            x = df[x_col]
            y = df[y_col]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(x, y, color='#007bff', marker='o', linestyle='-', linewidth=2)
            ax.set_xlabel(x_col.capitalize(), fontsize=12)
            ax.set_ylabel(y_col.capitalize(), fontsize=12)
            ax.set_title(f'{y_col.capitalize()} over {x_col.capitalize()} (Line Chart)', fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.7)
            if 'depth' in y_col.lower():
                ax.invert_yaxis()
                
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            line_plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.step(x, y, color='#ff4500', where='post', linewidth=2)
            ax.plot(x, y, 'o', color='#ff4500', markersize=5)
            ax.set_xlabel(x_col.capitalize(), fontsize=12)
            ax.set_ylabel(y_col.capitalize(), fontsize=12)
            ax.set_title(f'{y_col.capitalize()} over {x_col.capitalize()} (Step Chart)', fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.7)
            if 'depth' in y_col.lower():
                ax.invert_yaxis()
                
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            step_plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            logger.info("Successfully rendered charts from dataframe.")
            return line_plot_base64, step_plot_base64
        except Exception as e:
            logger.warning(f"Plotting from dataframe failed, falling back to context: {e}")

    depths, temps, sals, profile_ids = [], [], [], []
    import re
    pattern = r"Measurement in profile (\d+\.\d+) at depth ([\d\.]+)m: Temp ([\d\.]+) °C, Salinity ([\d\.]+) PSU"
    
    for entry in retrieved:
        if entry.get("source") == "measurements":
            summary = entry.get("summary", "")
            m = re.match(pattern, summary)
            if m:
                profile_ids.append(float(m.group(1)))
                depths.append(float(m.group(2)))
                temps.append(float(m.group(3)))
                sals.append(float(m.group(4)))
                
    if len(depths) < 2:
        logger.info("Not enough data points to generate charts.")
        return None, None
        
    sorted_indices = np.argsort(depths)
    depths_sorted = np.array(depths)[sorted_indices]
    
    is_salinity = "salinity" in user_query.lower()
    values_sorted = np.array(sals)[sorted_indices] if is_salinity else np.array(temps)[sorted_indices]
    label = 'Salinity (PSU)' if is_salinity else 'Temperature (°C)'
    color = '#007bff' if is_salinity else '#ff4500'
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(values_sorted, depths_sorted, color=color, marker='o', linestyle='-', linewidth=2)
    ax.set_xlabel(label, fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.invert_yaxis()
    ax.set_title('Measurements vs. Depth (Line Chart)', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    line_plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.step(values_sorted, depths_sorted, color=color, where='post', linewidth=2)
    ax.plot(values_sorted, depths_sorted, 'o', color=color, markersize=5)
    ax.set_xlabel(label, fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.invert_yaxis()
    ax.set_title('Measurements vs. Depth (Step Chart)', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    step_plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    logger.info("Successfully rendered charts from retrieved context fallback.")
    return line_plot_base64, step_plot_base64
