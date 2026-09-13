import duckdb
import re
import pandas as pd
from exceptions import DatabaseExecutionError
from logger import get_logger

logger = get_logger(__name__)

class DuckDBClient:
    """Handles connections and query executions for DuckDB."""

    def __init__(self, db_path: str = "data/argo.duckdb"):
        """
        Initializes the DuckDB connection.

        Args:
            db_path (str): Path to the DuckDB database file.
        """
        try:
            self.con = duckdb.connect(db_path)
            logger.info(f"Successfully connected to DuckDB at {db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to DuckDB at {db_path}: {e}")
            raise DatabaseExecutionError(f"Database connection failed: {e}")

    def get_profiles_time_type(self) -> str:
        """
        Retrieves the data type of the 'time' column in the 'profiles' table.

        Returns:
            str: The data type of the column, or an empty string if not found/failed.
        """
        try:
            desc = self.con.execute("DESCRIBE profiles").df()
            for _, row in desc.iterrows():
                if str(row['column_name']).lower() == 'time':
                    return str(row['column_type']).lower()
        except Exception as e:
            logger.warning(f"Could not retrieve profiles time type: {e}")
        return ""

    def adjust_sql_for_time_cast(self, sql: str) -> str:
        """
        Adjusts the SQL query to cast the 'time' column to TIMESTAMP if necessary.
        This handles cases where EXTRACT or date functions are used on string columns.

        Args:
            sql (str): The original SQL query.

        Returns:
            str: The adjusted SQL query.
        """
        time_type = self.get_profiles_time_type()
        if time_type and 'timestamp' not in time_type and 'date' not in time_type:
            sql = re.sub(r"EXTRACT\s*\(\s*MONTH\s+FROM\s+([^\)]+)\)", r"EXTRACT(MONTH FROM CAST(\1 AS TIMESTAMP))", sql, flags=re.I)
            sql = re.sub(r"EXTRACT\s*\(\s*YEAR\s+FROM\s+([^\)]+)\)", r"EXTRACT(YEAR FROM CAST(\1 AS TIMESTAMP))", sql, flags=re.I)
            sql = re.sub(r"\bMONTH\s*\(\s*([^\)]+)\s*\)", r"EXTRACT(MONTH FROM CAST(\1 AS TIMESTAMP))", sql, flags=re.I)
            sql = re.sub(r"\bYEAR\s*\(\s*([^\)]+)\s*\)", r"EXTRACT(YEAR FROM CAST(\1 AS TIMESTAMP))", sql, flags=re.I)
        return sql

    def execute_query(self, sql_query: str) -> pd.DataFrame:
        """
        Executes a SQL query and returns the results as a Pandas DataFrame.

        Args:
            sql_query (str): The SQL query to execute.

        Returns:
            pd.DataFrame: Query results.

        Raises:
            DatabaseExecutionError: If the SQL execution fails.
        """
        adjusted_sql = self.adjust_sql_for_time_cast(sql_query)
        try:
            logger.info(f"Executing SQL query: {adjusted_sql}")
            df = self.con.execute(adjusted_sql).df()
            return df
        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            raise DatabaseExecutionError(f"Failed to execute query: {e}")
