import sys
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

def upload_data_to_mongodb():
    try:
        print(" Uploading data to MongoDB...")

        csv_path = 'data/processed/clustered_data.csv'
        if not os.path.exists(csv_path):
            print(f"Error: File not found at {csv_path}")
            sys.exit(1)

        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} records from CSV")

        mongo_url = os.getenv('MONGO_DB_URL')
        if not mongo_url:
            print("Error: MONGO_DB_URL not found in .env file")
            sys.exit(1)

        client = MongoClient(mongo_url)
        db = client["CustomerDB"]
        collection = db["customer_0"]

        print("🔄 Clearing existing data...")
        collection.delete_many({})

        print("Uploading data to MongoDB...")
        data = df.to_dict('records')
        if data:
            collection.insert_many(data)
            print(f"✅ Successfully uploaded {len(data)} records to MongoDB!")
        else:
            print("⚠️ No data found in CSV.")

        print("📂 Database:", db.name)
        print("📁 Collection:", collection.name)

        client.close()

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    upload_data_to_mongodb()