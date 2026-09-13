import os
import re
import pandas as pd
from typing import List, Dict, Any, Optional
from groq import Groq
from dotenv import load_dotenv

from exceptions import LLMGenerationError
from logger import get_logger

load_dotenv()
logger = get_logger(__name__)

class LLMService:
    """Service for handling interactions with the Groq LLM API."""

    def __init__(self, model_name: str = None):
        """
        Initializes the LLM Service.
        
        Args:
            model_name (str): The Groq model to use. Defaults to os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile").
        """
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Missing GROQ_API_KEY. Please set it in a .env file.")
        
        self.client = Groq(api_key=self.api_key)
        self.model_name = model_name or os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
        logger.info(f"Initialized LLMService with model: {self.model_name}")

    def is_generic_query(self, query: str) -> bool:
        """
        Determines if the query is conversational/informational or requires data retrieval.
        
        Args:
            query (str): The user's query.
            
        Returns:
            bool: True if the query is generic, False if it requires data retrieval.
        """
        query = query.lower()
        specific_keywords = [
            'salinity', 'temperature', 'temp', 'depth', 'pressure', 'trend', 'average', 'min', 'max',
            'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september',
            'october', 'november', 'december', '2018', '2019', '2020', '2021', '2022', '2023',
            '2024', '2025', 'profile', 'latitude', 'longitude', 'platform'
        ]
        return not any(keyword in query for keyword in specific_keywords)

    def sanitize_llm_output(self, text: str) -> str:
        """
        Extracts and sanitizes the SQL query from the LLM's raw output.
        
        Args:
            text (str): The raw output from the LLM.
            
        Returns:
            str: The sanitized SQL query.
        """
        m = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.S | re.I)
        if m:
            sql = m.group(1).strip()
        else:
            m2 = re.search(r"(SELECT\b.*?;)", text, flags=re.S | re.I)
            if m2:
                sql = m2.group(1).strip()
            else:
                m3 = re.search(r"SELECT\b.*", text, flags=re.S | re.I)
                sql = m3.group(0).strip() if m3 else text.strip()
                
        sql = re.sub(r'^[\s\w\-\:\.]*?(SELECT\b)', r'\1', sql, flags=re.I)
        
        sql = re.sub(r"CAST\s*([^ (]+)\.", r"CAST(\1.", sql, flags=re.I)
        sql = re.sub(r"CAST\s*([^ (]+)\s+AS", r"CAST(\1) AS", sql, flags=re.I)
        
        return sql

    def generate_generic_response(self, user_query: str) -> str:
        """
        Generates a generic response for non-data-specific queries.
        
        Args:
            user_query (str): The user's query.
            
        Returns:
            str: The LLM's response.
            
        Raises:
            LLMGenerationError: If the LLM call fails.
        """
        system_msg = (
            "You are a professional and highly knowledgeable AI assistant specializing in oceanography and climate science. "
            "Your objective is to answer general or informational questions about ocean data clearly and concisely. "
            "Ensure that your explanations connect the concepts to broader implications for climate systems and marine ecosystems "
            "(e.g., how salinity, temperature, or depth changes affect ocean circulation, carbon sequestration, or biodiversity). "
            "Keep the response professional, scientifically accurate, and under 200 words."
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": f"User query: {user_query}"},
                ],
                temperature=0.5,
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Failed to generate generic response: {e}")
            raise LLMGenerationError(f"LLM Error: {e}")

    def generate_sql_query(self, user_query: str, context_text: str) -> tuple[str, str]:
        """
        Generates a DuckDB SQL query based on the user's question and retrieved context.
        
        Args:
            user_query (str): The user's question.
            context_text (str): Retrieved context from the vector store.
            
        Returns:
            Tuple[str, str]: A tuple containing the raw LLM output and the sanitized SQL string.
            
        Raises:
            LLMGenerationError: If the LLM call fails.
        """
        schema_text = (
            "Schema Overview:\n"
            "- profiles(profile_id, latitude, longitude, time, cycle_number, platform_number)\n"
            "- measurements(profile_id, depth_m, temp, psal, sigma_theta)\n"
            "- calibration(profile_id, scientific_calib_equation, scientific_calib_coefficient, scientific_calib_comment)\n"
            "- platforms(platform_number, platform_type, project_name, pi_name)\n\n"
        )

        system_msg = (
            "You are an expert Data Engineer specializing in DuckDB SQL syntax. "
            "Your task is to generate exactly ONE valid SQL query that answers the user's question, using the provided schema and context. "
            "STRICT RULES:\n"
            "1. Output ONLY the SQL query. Do not provide explanations, markdown formatting (other than the sql code block), or conversational text.\n"
            "2. Use CAST(profiles.time AS TIMESTAMP) for any time-based filtering or extraction.\n"
            "3. Use `measurements.psal` for salinity and `measurements.temp` for temperature.\n"
            "4. There is no 'pressure' column; use `depth_m` as an approximation (1 dbar ≈ 1 meter).\n"
            "5. Ensure proper parentheses and standard SQL best practices."
        )

        user_msg = f"{schema_text}Retrieved Context:\n{context_text}\n\nUser Question: {user_query}"

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            raw_llm_text = response.choices[0].message.content
            sql_text = self.sanitize_llm_output(raw_llm_text)
            return raw_llm_text, sql_text
        except Exception as e:
            logger.error(f"Failed to generate SQL query: {e}")
            raise LLMGenerationError(f"LLM Error: {e}")

    def generate_summary(self, user_query: str, df: Optional[pd.DataFrame], sql_text: str, retrieved: List[Dict[str, Any]]) -> str:
        """
        Generates a professional summary of the queried data and its ecological implications.
        
        Args:
            user_query (str): The original user query.
            df (pd.DataFrame | None): The resulting dataframe from the SQL execution.
            sql_text (str): The executed SQL query.
            retrieved (List[Dict]): The retrieved context.
            
        Returns:
            str: The LLM-generated summary.
            
        Raises:
            LLMGenerationError: If the LLM call fails.
        """
        system_msg = (
            "You are a professional Data Analyst and Oceanographer. "
            "Your objective is to provide a comprehensive, executive-level summary of the query results. "
            "Highlight key insights, trends, and anomalies (e.g., changes over time, depth profiles). "
            "Crucially, translate these findings into broader climate and ecological contexts. "
            "For example, discuss the impact of warming waters on marine ecosystems, or how salinity variations influence ocean currents. "
            "Maintain an objective, scientific tone without over-speculating."
        )
        
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            sampled_df = df.sample(n=min(15, len(df))).sort_values(by=df.columns[0])
            data_str = sampled_df.to_string(index=False)
            user_msg = (f"User Question: {user_query}\n"
                        f"Data Sample (15 rows):\n{data_str}\n"
                        f"Executed SQL: {sql_text}\n"
                        f"Vector Search Context: {retrieved}")
        else:
            user_msg = (f"User Question: {user_query}\n"
                        f"Data Returned: None.\n"
                        f"Executed SQL: {sql_text}\n"
                        f"Vector Search Context: {retrieved}")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Failed to generate data summary: {e}")
            raise LLMGenerationError(f"LLM Error: {e}")
