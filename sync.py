import os
import pymongo
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")
DATABASE = os.getenv("DATABASE")
COLLECTION = os.getenv("COLLECTION")

db = pymongo.MongoClient(MONGO_URL)[DATABASE][COLLECTION]

os.rename("records.csv", "_records.csv")

df = pd.read_csv("_records.csv")

db.insert_many(df.to_dict('records'))