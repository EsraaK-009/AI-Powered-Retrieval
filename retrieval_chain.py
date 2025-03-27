"""
This file contains code to connect to the database and run the SQL query
"""
import os
import psycopg2
from langchain_core.runnables import RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv


load_dotenv(".env")

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

def execute_sql_query(answer_dict):
    #print(answer_dict)
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

    cursor = conn.cursor()
    cursor.execute(answer_dict["query"])
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    return result

def get_DB_data():
    return RunnableLambda(execute_sql_query)

