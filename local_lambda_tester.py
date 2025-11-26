import os
from lambda_handler import lambda_handler
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set required environment variables for local test
data_folder = os.getenv("S3_BUCKET_NAME")

# Example event (simulate API Gateway event)
event = {
    "body": '{"folder_key": "local:data/MahaAgri"}'
}

# Pass None for context or mock it if needed
result = lambda_handler(event, None)
print(result)


