import os
import psycopg2
import requests
from dotenv import load_dotenv

# Load environment variables only in development
if os.path.exists('.env'):
    load_dotenv()
# Get database connection info from environment variables
POSTGRES_HOST = os.environ.get('POSTGRES_HOST1')
POSTGRES_PORT = os.environ.get('POSTGRES_PORT', '5432')
POSTGRES_DATABASE = os.environ.get('POSTGRES_DATABASE')
POSTGRES_USER = os.environ.get('POSTGRES_USER')
POSTGRES_PASSWORD = os.environ.get('POSTGRES_PASSWORD')

# Connect to PostgreSQL database
conn = psycopg2.connect(
    host=POSTGRES_HOST,
    port=POSTGRES_PORT,
    database=POSTGRES_DATABASE,
    user=POSTGRES_USER,
    password=POSTGRES_PASSWORD
)

# Create a cursor to execute queries
cursor = conn.cursor()

# Fetch all product names from the product table
cursor.execute("SELECT name FROM product")
product_names = [row[0] for row in cursor.fetchall()]

# Close the database connection
cursor.close()
conn.close()

# URL of your Flask API
url = 'http://localhost:5000/demand'

# Data to send in the request
data = {
    "product_names": product_names
}

# Send POST request
response = requests.post(url, json=data)

# Check the response status
if response.status_code == 200:
    print("Dự đoán:", response.json())
else:
    print(f"Lỗi {response.status_code}: {response.json()}")
