import faiss
import numpy as np
import duckdb
from sentence_transformers import SentenceTransformer

print("1. Downloading embedding model (this might take a moment)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("2. Creating dummy FAISS index & ID map...")
dummy_texts = ["Measurement in profile 1.0 at depth 10.0m: Temp 15.0 °C, Salinity 35.0 PSU"]
embeddings = model.encode(dummy_texts, convert_to_numpy=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss.write_index(index, "data/argo_faiss.index")

id_map = [{"source": "measurements", "summary": dummy_texts[0]}]
np.save("data/id_map.npy", np.array(id_map, dtype=object))

print("3. Creating dummy DuckDB tables...")
con = duckdb.connect("data/argo.duckdb")

con.execute("""
    CREATE TABLE IF NOT EXISTS profiles (
        profile_id FLOAT, latitude FLOAT, longitude FLOAT, 
        time TIMESTAMP, cycle_number INT, platform_number INT
    )
""")
con.execute("""
    CREATE TABLE IF NOT EXISTS measurements (
        profile_id FLOAT, depth_m FLOAT, temp FLOAT, 
        psal FLOAT, sigma_theta FLOAT
    )
""")

con.execute("INSERT INTO measurements VALUES (1.0, 10.0, 15.0, 35.0, 26.5)")
con.execute("INSERT INTO profiles VALUES (1.0, 0.0, 0.0, CURRENT_TIMESTAMP, 1, 12345)")
con.close()

print("Success! Dummy data created. You can now start your server.")