# filepath: /home/ubuntu/sam2/services/s3_service.py

import os
import boto3
import re
from botocore.exceptions import ClientError

def download_images_from_s3(s3_path, local_dir):
    """
    Download all jpg images from the specified S3 path to a local directory
    and rename them in a sequential manner (000.jpg, 001.jpg, etc.)
    
    Args:
        s3_path (str): S3 path in the format 's3://bucket-name/device-name/Experiments/experiment-name/sample-name/'
        local_dir (str): Local directory to save the downloaded images
        
    Returns:
        tuple: (file_mappings, bucket_name, device_name, experiment_name, sample_name, data_dir, context_dir)
    """
    # Parse bucket name and prefix from s3_path
    s3_path = s3_path.strip()
    if not s3_path.startswith('s3://'):
        raise ValueError("S3 path must start with 's3://'")
    
    # Parse the path components
    # s3://<bucket_name>/<device_name>/Experiments/<experiment_name>/<sample_name>/
    path_parts = s3_path[5:].split('/')
    
    bucket_name = path_parts[0]
    device_name = path_parts[1]
    
    # Handle potential colon in experiment name
    experiment_name = path_parts[3]
    sample_name = path_parts[4]
    
    # Create data and context directories
    data_dir_name = f"data_{experiment_name}_{sample_name}"
    context_dir_name = f"context_{experiment_name}_{sample_name}"
    
    data_dir = os.path.join(local_dir, data_dir_name)
    context_dir = os.path.join(local_dir, context_dir_name)
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(context_dir, exist_ok=True)
    
    # Ensure prefix ends with a slash
    prefix = '/'.join(path_parts[1:])
    print(prefix)
    
    # Initialize S3 client
    s3_client = boto3.client('s3')
    
    try:
        # List objects in the specified S3 path
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter='/')
        
        # Collect all jpg files (ignoring subdirectories)
        jpg_files = []
        for obj in response.get('Contents', []):
            key = obj['Key']
            # Check if the file is directly in the specified directory (not in a subdirectory)
            if key.lower().endswith('.jpg') and '/' not in key[len(prefix):]:
                jpg_files.append(key)
        
        # Sort files by name to preserve relative order
        jpg_files.sort()
        
        # Download and rename files
        file_mappings = {}
        for idx, key in enumerate(jpg_files):
            original_filename = os.path.basename(key)
            new_filename = f"{idx:03d}.jpg"
            local_path = os.path.join(data_dir, new_filename)
            
            # Download the file
            s3_client.download_file(bucket_name, key, local_path)
            
            # Store mapping
            file_mappings[original_filename] = new_filename
        
        return file_mappings, bucket_name, device_name, experiment_name, sample_name, data_dir, context_dir
        
    except ClientError as e:
        raise Exception(f"Error accessing S3: {str(e)}")

def upload_file_to_s3(file_path, bucket_name, device_name, session_id, context_dir_name, file_name=None):
    """
    Upload a file to S3 and generate a presigned URL
    
    Args:
        file_path (str): Local path to the file
        bucket_name (str): S3 bucket name
        device_name (str): Device name
        session_id (str): Session ID
        context_dir_name (str): Context directory name
        file_name (str, optional): Custom file name for S3. Defaults to basename of file_path.
        
    Returns:
        str: Presigned URL for the uploaded file
    """
    if file_name is None:
        file_name = os.path.basename(file_path)
    
    # Construct S3 key
    # Format: <device_name>/Analysis/<session_id>/<context_dir_name>/<file_name>
    s3_key = f"{device_name}/Analysis/{session_id}/{context_dir_name}/{file_name}"
    
    # Initialize S3 client
    s3_client = boto3.client('s3')
    
    try:
        # Upload file to S3
        s3_client.upload_file(file_path, bucket_name, s3_key)
        
        # Generate presigned URL (valid for 1 hour)
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': s3_key},
            ExpiresIn=3600  # URL expires in 1 hour
        )
        
        return presigned_url
    
    except ClientError as e:
        raise Exception(f"Error uploading to S3: {str(e)}")