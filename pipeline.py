import time
from typing import Dict, Any
from dotenv import load_dotenv

from logger import get_logger
from exceptions import OceanBotError
from db_client import DuckDBClient
from vector_store import VectorStore
from llm_service import LLMService
from visualization import render_depth_profile_charts

load_dotenv()
logger = get_logger(__name__)

try:
    db_client = DuckDBClient()
    vector_store = VectorStore()
    llm_service = LLMService()
except Exception as e:
    logger.critical(f"Failed to initialize core services: {e}")
    db_client = None
    vector_store = None
    llm_service = None


def query_pipeline(user_query: str, top_k: int = 20) -> Dict[str, Any]:
    """
    Main orchestration function for the Text-to-SQL and RAG pipeline.
    
    Args:
        user_query (str): The user's input question.
        top_k (int): Number of context vectors to retrieve.
        
    Returns:
        Dict[str, Any]: A dictionary containing the summary, charts, and debug information.
    """
    start_time = time.time()
    result = {
        "summary": "",
        "line_chart_b64": None,
        "step_chart_b64": None,
        "debug_sql": None,
        "debug_context": None,
        "debug_latency_ms": 0,
        "error": None
    }
    
    try:
        if llm_service.is_generic_query(user_query):
            logger.info("Processing as a generic query.")
            result["summary"] = llm_service.generate_generic_response(user_query)
            result["debug_latency_ms"] = round((time.time() - start_time) * 1000, 2)
            return result

        logger.info(f"Retrieving context with top_k={top_k}")
        context_text, retrieved_data = vector_store.retrieve_relevant_context(user_query, top_k=top_k)
        result["debug_context"] = context_text

        logger.info("Generating SQL query via LLM")
        raw_llm, sql_text = llm_service.generate_sql_query(user_query, context_text)
        result["debug_sql"] = sql_text

        logger.info("Executing SQL against DuckDB")
        db_start_time = time.time()
        df = None
        try:
            df = db_client.execute_query(sql_text)
            logger.info(f"DuckDB execution took {(time.time() - db_start_time) * 1000:.2f} ms")
        except Exception as e:
            logger.warning(f"DuckDB execution failed, will rely on context. Error: {e}")

        logger.info("Rendering charts")
        line_chart, step_chart = render_depth_profile_charts(df, retrieved_data, user_query)
        result["line_chart_b64"] = line_chart
        result["step_chart_b64"] = step_chart

        logger.info("Generating final summary")
        summary = llm_service.generate_summary(user_query, df, sql_text, retrieved_data)
        result["summary"] = summary

    except OceanBotError as e:
        logger.error(f"Pipeline error: {e}")
        result["error"] = str(e)
        result["summary"] = f"An error occurred during processing: {e}"
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        result["error"] = "An unexpected system error occurred."
        result["summary"] = "An unexpected system error occurred while processing your request."
    finally:
        result["debug_latency_ms"] = round((time.time() - start_time) * 1000, 2)
        logger.info(f"Pipeline completed in {result['debug_latency_ms']} ms")
        return result