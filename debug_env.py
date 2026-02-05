import os
from dotenv import load_dotenv

load_dotenv()
print("SERVICE:", os.getenv("FIDZULU_DB_SERVICE"))