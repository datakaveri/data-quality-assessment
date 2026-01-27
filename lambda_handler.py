print("Starting lambda_handler.py") 
import os
import boto3
import zipfile
import logging
import tempfile
import time
import json
from dotenv import load_dotenv


logging.info("Importing modules completed in lambda_handler.py")

# Configure logging
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger()

# Configure formatter for more detailed logs
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
for handler in logger.handlers:
    handler.setFormatter(formatter)

# Initialize AWS clients
s3_client = boto3.client('s3')
logger.info("AWS S3 client initialized successfully.")

def lambda_handler(event, context):
    """
    Lambda function handler to run data readiness framework on files in an S3 bucket.
    
    Parameters
    ----------
    event : dict
        Lambda event object with the following keys:
            bucket: str
                The name of the S3 bucket containing the files to process.
            prefix: str
                The prefix of the files to process within the given bucket.
    context : object
        Lambda context object.
    
    Returns
    -------
    dict
        A dictionary with a single key "status" which maps to "done".
    """

    try:
        logger.info("Lambda function started.")
        # Start timing the execution
        start_time = time.time()
        request_id = context.aws_request_id if context else 'local-run'
        logger.info(f"Lambda invocation started - RequestID: {request_id}")
        try:
            # Log the incoming event
            logger.info(f"Received event: {json.dumps(event)}")
            
            # Extract folder_key robustly from event
            body = event.get('body', event)
            if isinstance(body, str):
                try:
                    body = json.loads(body)
                except Exception as e:
                    logger.error(f"Error parsing body as JSON: {e}")
                    body = {}
            elif not isinstance(body, dict):
                body = {}
            folder_key = body.get('folder_key')
            logger.info(f"Extracted folder_key: {folder_key}")

            # For local testing, to be removed before deployment
            # if folder_key and (folder_key.startswith("local:") or os.path.isdir(folder_key)):
            #     # Local mode: folder_key is a local directory path
            #     local_dir = folder_key.replace("local:", "") if folder_key.startswith("local:") else folder_key
            #     # print(f"Running unstructured_main.main on local directory: {local_dir}")
            #     # import unstructured_main
            #     # unstructured_main.main(local_dir, folder_key)
            #     # return {"status": "done (local unstructured run)"}
            #     print(f"Running structured_main.main on local directory: {local_dir}")
            #     import structured_main
            #     structured_main.main(local_dir, folder_key)
            #     return {"status": "done (local structured run)"}

    
            if not folder_key:
                logger.error("Missing required parameter: folder_key")
                return {
                    'statusCode': 400,
                    'body': json.dumps({'error': 'Missing folder_key parameter'})
                }
            
            if isinstance(folder_key, str):
                folder_keys = [folder_key]
            else:
                folder_keys = folder_key

            # Get bucket name from environment variable
            bucket_name = os.environ.get('S3_BUCKET_NAME')
            if not bucket_name:
                logger.error("Missing required environment variable: S3_BUCKET_NAME")
                return {
                    'statusCode': 500,
                    'body': json.dumps({'error': 'Server configuration error: Missing S3_BUCKET_NAME'})
                }
            
            logger.info(f"Using bucket: {bucket_name}")
            

            for fk in folder_keys:
                logger.info(f"Processing folder: {fk}")
                # Check if the folder exists in S3
                logger.info(f"Checking if folder exists: {fk}")
                try:
                    # List objects with the given prefix and limit to 1 result
                    response = s3_client.list_objects_v2(
                        Bucket=bucket_name,
                        Prefix=fk,
                        MaxKeys=1
                    )
                    
                    # If no contents or the only item is the folder itself (ends with /)
                    if 'Contents' not in response or (
                        len(response['Contents']) == 1 and 
                        response['Contents'][0]['Key'] == fk and 
                        fk.endswith('/')
                    ):
                        logger.error(f"Folder not found or empty: {fk}")
                        return {
                            'statusCode': 404,
                            'body': json.dumps({
                                'error': 'Folder not found or empty',
                                'folder_key': fk
                            })
                        }
                        
                    logger.info(f"Folder exists: {fk}")
                except Exception as e:
                    logger.error(f"Error checking folder existence: {str(e)}", exc_info=True)
                    return {
                        'statusCode': 500,
                        'body': json.dumps({
                            'error': 'Error checking folder existence',
                            'error_message': str(e)
                        })
                    }
                
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Download files from S3
                    for obj in s3_client.list_objects_v2(Bucket=bucket_name, Prefix=fk).get('Contents', []):
                        if obj['Key'].endswith('/'):
                            continue
                        local_path = os.path.join(temp_dir, os.path.basename(obj['Key']))
                        s3_client.download_file(bucket_name, obj['Key'], local_path)
                        # Unzip if required
                        if local_path.endswith('.zip'):
                            with zipfile.ZipFile(local_path, 'r') as zf:
                                zf.extractall(temp_dir)
                    # Run your framework
                    # for structured datasets
                    if any(f.endswith(('.parquet', '.csv', '.json')) for f in os.listdir(temp_dir)):
                        try:
                            from structured_main import main
                            logging.info("main imported successfully")
                        except Exception as e:
                            logging.error(f"Error importing main: {e}", exc_info=True)

                        main(temp_dir, fk)
                    # for unstructured datasets
                    elif any(f.endswith(('.xlsx', '.xls', '.pdf', '.mp3', '.jpg', '.jpeg', '.png', '.tiff', '.tif', '.txt', '.md', '.dcm')) for f in os.listdir(temp_dir)):
                        try:
                            from unstructured_main import main
                            logging.info("main imported successfully")
                        except Exception as e:
                            logging.error(f"Error importing main: {e}", exc_info=True)
                        
                        main(temp_dir, fk)
                    logger.info("Data readiness framework executed successfully.")
                    # Upload reports to S3
                    for root, _, files in os.walk(temp_dir):
                        print(root, _, files)
                        for f in files:
                            if f.endswith(('.json', '.pdf')):
                                load_dotenv() 
                                reports_bucket_name = os.getenv('S3_REPORTS_BUCKET_NAME')
                                logger.info("Reports bucket name: %s", reports_bucket_name)
                                if not reports_bucket_name:
                                    logger.error("Missing required environment variable: S3_REPORTS_BUCKET_NAME")
                                    return {
                                        'statusCode': 500,
                                        'body': json.dumps({'error': 'Server configuration error: Missing S3_REPORTS_BUCKET_NAME'})
                                    }
                                logger.info(f"Uploading report: {f} to bucket: {reports_bucket_name}")
                                report_key = f"{os.path.basename(fk)}/{f}"
                                logger.info(f"Report key: {report_key}")
                                s3_client.upload_file(os.path.join(root, f), reports_bucket_name, report_key)   
                                logger.info(f"Uploaded report to S3: {report_key}")
            end_time = time.time()
            logger.info(f"Lambda invocation completed - RequestID: {request_id}")
            logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")
            return {"status": "done"}
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}", exc_info=True)
            total_duration = time.time() - start_time
            logger.info(f"Lambda execution failed after {total_duration:.2f} seconds")
            
            error_response = {
                'statusCode': 500,
                'body': json.dumps({
                    'error': 'Internal server error',
                    'error_message': str(e),
                    'request_id': request_id
                })
            }
    except Exception as e:
        logger.error(f"Unhandled error at top level: {e}", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Unhandled top-level error', 'error_message': str(e)})
        }


