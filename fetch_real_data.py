import ssl
ssl.create_default_context = ssl._create_unverified_context
from argopy import DataFetcher
import duckdb
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

print("1. Fetching real ocean data from Argo servers (this may take a minute)...")

argo_fetcher = DataFetcher().region([-75, -70, 20, 25, 0, 50])
df = argo_fetcher.to_dataframe()

print(f"Downloaded {len(df)} real measurements! Cleaning data...")

df = df.dropna(subset=['PRES', 'TEMP', 'PSAL'])

print("2. Downloading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("3. Building Text Summaries for AI Search...")
dummy_texts = []
id_map = []
measurements_for_db = []
profiles_for_db = []

profile_counter = 1.0

for (platform, cycle), group in df.groupby(['PLATFORM_NUMBER', 'CYCLE_NUMBER']):
    lat = group['LATITUDE'].iloc[0]
    lon = group['LONGITUDE'].iloc[0]
    time = group['TIME'].iloc[0]
    
    profiles_for_db.append((profile_counter, float(lat), float(lon), str(time), int(cycle), int(platform)))
    
    for _, row in group.iterrows():
        depth = row['PRES']
        temp = row['TEMP']
        psal = row['PSAL']
        
        summary = f"Measurement in profile {profile_counter} at depth {depth:.1f}m: Temp {temp:.2f} °C, Salinity {psal:.2f} PSU"
        dummy_texts.append(summary)
        id_map.append({"source": "measurements", "summary": summary})
        
        measurements_for_db.append((profile_counter, float(depth), float(temp), float(psal), 0.0))
        
    profile_counter += 1.0

print("4. Creating FAISS Index (this will take a moment)...")
embeddings = model.encode(dummy_texts, convert_to_numpy=True)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss.write_index(index, "data/argo_faiss.index")

np.save("data/id_map.npy", np.array(id_map, dtype=object))

print("5. Populating DuckDB...")
con = duckdb.connect("data/argo.duckdb")

con.execute("DROP TABLE IF EXISTS profiles")
con.execute("DROP TABLE IF EXISTS measurements")

con.execute("""
    CREATE TABLE profiles (
        profile_id FLOAT, latitude FLOAT, longitude FLOAT, 
        time TIMESTAMP, cycle_number INT, platform_number INT
    )
""")
con.execute("""
    CREATE TABLE measurements (
        profile_id FLOAT, depth_m FLOAT, temp FLOAT, 
        psal FLOAT, sigma_theta FLOAT
    )
""")

con.executemany("INSERT INTO profiles VALUES (?, ?, ?, ?, ?, ?)", profiles_for_db)
con.executemany("INSERT INTO measurements VALUES (?, ?, ?, ?, ?)", measurements_for_db)

con.close()
print(" Success! Real ocean data injected. You can now restart your FastAPI server!")