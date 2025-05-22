# filepath: /home/ubuntu/sam2/app.py

import os
import json
import uuid
import torch
import h5py
import numpy as np
import boto3
from flask import Flask, request, jsonify
from services.s3_service import download_images_from_s3, upload_file_to_s3
from services.session_service import SessionManager
from sam2.build_sam import build_sam2_video_predictor

app = Flask(__name__)
session_manager = SessionManager()

@app.route('/create_session', methods=['POST'])
def create_session():
    try:
        data = request.get_json()
        s3_path = data.get('s3_path')
        
        if not s3_path:
            return jsonify({"error": "S3 path is required"}), 400
        
        # Generate a unique session ID
        session_id = str(uuid.uuid4())
        session_dir = os.path.join('sessions', session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Download images from S3 with reorganized directory structure
        result = download_images_from_s3(s3_path, session_dir)
        file_mappings, bucket_name, device_name, experiment_name, sample_name, data_dir, context_dir = result
        
        # Initialize SAM2 predictor
        checkpoint = "./checkpoints/sam2.1_hiera_base_plus.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        predictor = build_sam2_video_predictor(model_cfg, checkpoint)
        
        # Create and store session
        session_manager.create_session(
            session_id=session_id,
            session_dir=session_dir,
            file_mappings=file_mappings,
            predictor=predictor,
            bucket_name=bucket_name,
            device_name=device_name,
            experiment_name=experiment_name,
            sample_name=sample_name,
            data_dir=data_dir,
            context_dir=context_dir
        )
        
        return jsonify({"session_id": session_id}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/open_session', methods=['POST'])
def open_session():
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({"error": "Session ID is required"}), 400
        
        if not session_manager.session_exists(session_id):
            return jsonify({"error": f"Session {session_id} not found"}), 404
        
        session_manager.set_current_session(session_id)
        return jsonify({"message": f"Session {session_id} opened successfully"}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/init_state', methods=['POST'])
def init_state():
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({"error": "Session ID is required"}), 400
        
        if not session_manager.session_exists(session_id):
            return jsonify({"error": f"Session {session_id} not found"}), 404
        
        session = session_manager.get_session(session_id)
        predictor = session.get('predictor')
        data_dir = session.get('data_dir')
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = predictor.init_state(data_dir)
        
        # Save inference state to session
        session_manager.update_session(session_id, 'inference_state', inference_state)
        
        return jsonify({"message": "State initialized successfully"}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/reset_state', methods=['POST'])
def reset_state():
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({"error": "Session ID is required"}), 400
        
        if not session_manager.session_exists(session_id):
            return jsonify({"error": f"Session {session_id} not found"}), 404
        
        session = session_manager.get_session(session_id)
        predictor = session.get('predictor')
        
        predictor.reset_state()
        
        return jsonify({"message": "State reset successfully"}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/add_prompt', methods=['POST'])
def add_prompt():
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        frame_idx = data.get('frame_idx')
        prompts = data.get('prompt')
        
        if not all([session_id, frame_idx is not None, prompts]):
            return jsonify({"error": "Session ID, frame_idx, and prompt are required"}), 400
        
        if not session_manager.session_exists(session_id):
            return jsonify({"error": f"Session {session_id} not found"}), 404
        
        session = session_manager.get_session(session_id)
        predictor = session.get('predictor')
        inference_state = session.get('inference_state')
        context_dir = session.get('context_dir')
        bucket_name = session.get('bucket_name')
        device_name = session.get('device_name')
        experiment_name = session.get('experiment_name')
        sample_name = session.get('sample_name')
        
        results = []
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for ann_obj_id, values in prompts.items():
                print(f"Adding object {ann_obj_id}")
                points, labels = values
                out_frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=ann_obj_id,
                    points=points,
                    labels=labels,
                )
                
                # Convert tensors to serializable format
                result = {
                    "out_frame_idx": out_frame_idx,
                    "out_obj_ids": out_obj_ids.tolist() if isinstance(out_obj_ids, torch.Tensor) else out_obj_ids,
                    "out_mask_logits": [logit.cpu().numpy().tolist() for logit in out_mask_logits] 
                                        if isinstance(out_mask_logits, list) else out_mask_logits.cpu().numpy().tolist()
                }
                results.append(result)
        
        # Save frame_idx to session context
        session_manager.update_session(session_id, 'current_frame_idx', frame_idx)
        
        # Save results to JSON file
        json_file_name = f"prompt_results_{frame_idx}.json"
        json_file_path = os.path.join(context_dir, json_file_name)
        with open(json_file_path, 'w') as f:
            json.dump(results, f)
        
        # Upload JSON file to S3 and get presigned URL
        context_dir_name = os.path.basename(context_dir)
        presigned_url = upload_file_to_s3(
            file_path=json_file_path,
            bucket_name=bucket_name,
            device_name=device_name,
            session_id=session_id,
            context_dir_name=context_dir_name,
            file_name=json_file_name
        )
        
        return jsonify({
            "message": "Prompt results saved and uploaded successfully",
            "file_url": presigned_url
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/propagate_in_video', methods=['POST'])
def propagate_in_video():
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({"error": "Session ID is required"}), 400
        
        if not session_manager.session_exists(session_id):
            return jsonify({"error": f"Session {session_id} not found"}), 404
        
        session = session_manager.get_session(session_id)
        predictor = session.get('predictor')
        inference_state = session.get('inference_state')
        session_dir = session.get('session_dir')
        frame_idx = session.get('current_frame_idx')
        context_dir = session.get('context_dir')
        bucket_name = session.get('bucket_name')
        device_name = session.get('device_name')
        
        video_segments = {}
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state, reverse=True, start_frame_idx=frame_idx
            ):
                video_segments[int(out_frame_idx)] = {
                    int(out_obj_id): (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
        
        # Save video_segments as h5 file in the context directory
        h5_file_name = 'video_segments.h5'
        h5_file_path = os.path.join(context_dir, h5_file_name)
        with h5py.File(h5_file_path, 'w') as f:
            for frame_idx, obj_dict in video_segments.items():
                frame_group = f.create_group(str(frame_idx))
                for obj_id, mask in obj_dict.items():
                    frame_group.create_dataset(str(obj_id), data=mask)
        
        # Upload h5 file to S3 and get presigned URL
        context_dir_name = os.path.basename(context_dir)
        presigned_url = upload_file_to_s3(
            file_path=h5_file_path,
            bucket_name=bucket_name,
            device_name=device_name,
            session_id=session_id,
            context_dir_name=context_dir_name,
            file_name=h5_file_name
        )
        
        return jsonify({
            "message": "Propagation completed and saved to H5 file", 
            "file_url": presigned_url
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/growth', methods=['POST'])
def calculate_growth():
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        time_difference = data.get('time')  # Time difference between frames in minutes
        
        if not session_id:
            return jsonify({"error": "Session ID is required"}), 400
        
        if not time_difference:
            return jsonify({"error": "Time difference is required"}), 400
        
        if not session_manager.session_exists(session_id):
            return jsonify({"error": f"Session {session_id} not found"}), 404
        
        session = session_manager.get_session(session_id)
        context_dir = session.get('context_dir')
        bucket_name = session.get('bucket_name')
        device_name = session.get('device_name')
        
        # Check if h5 file exists
        h5_file_path = os.path.join(context_dir, 'video_segments.h5')
        if not os.path.exists(h5_file_path):
            return jsonify({"error": "Video segments file not found. Run propagate_in_video first."}), 404
        
        # Create growth directory
        growth_dir = os.path.join(context_dir, 'growth')
        os.makedirs(growth_dir, exist_ok=True)
        
        # Load h5 file and calculate areas
        import pandas as pd
        import numpy as np
        
        video_segments = {}
        with h5py.File(h5_file_path, 'r') as f:
            # Get number of frames (images)
            frame_ids = sorted([int(k) for k in f.keys()])
            if not frame_ids:
                return jsonify({"error": "No frames found in video segments file"}), 404
            
            # Get all object IDs across all frames
            all_object_ids = set()
            for frame_id in f.keys():
                all_object_ids.update(f[frame_id].keys())
            
            all_object_ids = sorted([int(obj_id) for obj_id in all_object_ids])
            if not all_object_ids:
                return jsonify({"error": "No objects found in video segments file"}), 404
            
            # Load all frames and objects
            for frame_id in f.keys():
                frame_idx = int(frame_id)
                video_segments[frame_idx] = {}
                for obj_id in f[frame_id].keys():
                    video_segments[frame_idx][int(obj_id)] = f[frame_id][obj_id][:]
        
        # Calculate areas for each object in each frame
        csv_data = []
        for t in range(max(frame_ids) + 1):
            for obj in all_object_ids:
                if t in video_segments and obj in video_segments[t]:
                    time = float(t) * float(time_difference)
                    area = np.sum(video_segments[t][obj] > 0)
                    csv_data.append({
                        'Time': time,
                        'Object': int(obj),
                        'Area': float(area)
                    })
        
        # Create DataFrame and save CSV
        df = pd.DataFrame(csv_data)
        csv_file_path = os.path.join(growth_dir, 'areas.csv')
        df.to_csv(csv_file_path, index=False)
        
        # Upload CSV to S3
        s3_key = f"{device_name}/Analysis/{session_id}/growth/areas.csv"
        s3_client = boto3.client('s3')
        s3_client.upload_file(csv_file_path, bucket_name, s3_key)
        
        # Generate presigned URL
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': s3_key},
            ExpiresIn=3600  # URL expires in 1 hour
        )
        
        return jsonify({
            "message": "Growth analysis completed and saved to CSV file",
            "file_url": presigned_url
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/rugosity', methods=['POST'])
def calculate_rugosity():
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({"error": "Session ID is required"}), 400
        
        if not session_manager.session_exists(session_id):
            return jsonify({"error": f"Session {session_id} not found"}), 404
        
        session = session_manager.get_session(session_id)
        data_dir = session.get('data_dir')
        context_dir = session.get('context_dir')
        bucket_name = session.get('bucket_name')
        device_name = session.get('device_name')
        
        # Check if h5 file exists
        h5_file_path = os.path.join(context_dir, 'video_segments.h5')
        if not os.path.exists(h5_file_path):
            return jsonify({"error": "Video segments file not found. Run propagate_in_video first."}), 404
        
        # Create rugosity directory
        rugosity_dir = os.path.join(context_dir, 'rugosity')
        os.makedirs(rugosity_dir, exist_ok=True)
        
        # Import necessary libraries
        import pandas as pd
        import numpy as np
        from PIL import Image
        import glob
        from skimage import morphology
        from scipy.ndimage import binary_fill_holes, gaussian_filter
        
        # Define helper functions
        def preprocess_mask(mask):
            mask = np.squeeze(mask) > 0
            cleaned = morphology.remove_small_objects(mask, min_size=20)
            filled = binary_fill_holes(cleaned)
            return filled.astype(np.uint8)
        
        def gaussian_fft(intensity_array):
            low_pass = gaussian_filter(intensity_array, sigma=5)
            high_pass = intensity_array - low_pass
            fft_high_pass = np.fft.fftshift(np.fft.fft2(high_pass))
            radial_prof_hp = cal_radial_profile(np.abs(fft_high_pass))
            return high_pass, radial_prof_hp
        
        def cal_radial_profile(data):
            y, x = np.indices(data.shape)
            center = np.array([x.max() / 2, y.max() / 2])
            r = np.hypot(x - center[0], y - center[1])
            r = r.astype(np.int32)
            tbin = np.bincount(r.ravel(), data.ravel())
            nr = np.bincount(r.ravel())
            radialprofile = tbin / np.maximum(nr, 1)
            return radialprofile
        
        # Get image files
        frame_names = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
        if not frame_names:
            return jsonify({"error": f"No images found in {data_dir}"}), 404
        
        # Pass 1: Collect all object ids
        with h5py.File(h5_file_path, "r") as f:
            object_ids_set = set()
            frame_ids = sorted(f.keys(), key=lambda x: int(x))
            for frame_id in frame_ids:
                frame_group = f[frame_id]
                object_ids_set.update([int(obj_id) for obj_id in frame_group.keys()])
        object_ids = sorted(list(object_ids_set))
        
        if not object_ids:
            return jsonify({"error": "No objects found in video segments file"}), 404
        
        # Prepare dict to accumulate per-object, per-frame data
        object_frame_data = {object_id: {} for object_id in object_ids}
        
        # Pass 2: Process data and accumulate per-object, per-frame
        with h5py.File(h5_file_path, "r") as f:
            frame_ids = sorted(f.keys(), key=lambda x: int(x))
            for frame_id in frame_ids:
                print(f"Processing frame {frame_id}...")
                idx = int(frame_id)
                if idx >= len(frame_names):
                    print(f"Frame index {idx} out of range for images!")
                    continue
                    
                img_path = frame_names[idx]
                with Image.open(img_path) as img:
                    # Store image dimensions for resizing masks
                    img_width, img_height = img.size
                    gray_img = np.array(img.convert('L')) / 255.0
                
                print(f"Image dimensions: {gray_img.shape}")
                    
                frame_group = f[frame_id]
                # For each object in this frame, process if it exists
                for object_id in object_ids:
                    if str(object_id) not in frame_group:
                        continue
                    mask = frame_group[str(object_id)][:]
                    proc_mask = preprocess_mask(mask)
                    if np.sum(proc_mask) == 0:
                        continue
                        
                    # Print dimensions for debugging
                    print(f"Before resize: mask shape: {proc_mask.shape}, image shape: {gray_img.shape}")
                    
                    # Resize mask to match gray_img dimensions
                    if proc_mask.shape != gray_img.shape:
                        from skimage.transform import resize
                        proc_mask = resize(
                            proc_mask.astype(float), 
                            (gray_img.shape[0], gray_img.shape[1]),
                            mode='constant',
                            order=0,  # Nearest neighbor to preserve mask values
                            preserve_range=True
                        ).astype(np.uint8)
                        print(f"After resize: mask shape: {proc_mask.shape}")
                    
                    # Ensure the mask and image can be multiplied
                    if proc_mask.shape == gray_img.shape:
                        masked_image = gray_img * proc_mask
                        _, radial_profile = gaussian_fft(masked_image)
                    else:
                        print(f"ERROR: Shapes still don't match after resize: {proc_mask.shape} vs {gray_img.shape}")
                        continue
                    freqs = np.arange(1, len(radial_profile) + 1)
                    freqs_log = np.log(freqs)
                    radial_profile_log = np.log(radial_profile + 1e-8)
                    # Save per-frame data for this object
                    object_frame_data[object_id][frame_id] = {
                        'Frequency (log)': freqs_log,
                        'Radial Profile (log)': radial_profile_log
                    }
        
        # Create Excel files for each object
        excel_files = []
        for object_id, frame_dict in object_frame_data.items():
            if not frame_dict:  # Skip if no data for this object
                continue
                
            object_dir = os.path.join(rugosity_dir, f"object_{object_id}")
            os.makedirs(object_dir, exist_ok=True)
            excel_path = os.path.join(object_dir, f"object_{object_id}_radial_profiles.xlsx")
            
            # Create Excel file with a sheet per frame
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                for frame_id, data in frame_dict.items():
                    df = pd.DataFrame({
                        'Frequency (log)': data['Frequency (log)'],
                        'Radial Profile (log)': data['Radial Profile (log)']
                    })
                    sheet_name = f"frame_{frame_id}"
                    # Excel sheet names are limited to 31 characters
                    df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
            
            # Upload Excel to S3
            s3_key = f"{device_name}/Analysis/{session_id}/rugosity/object_{object_id}/object_{object_id}_radial_profiles.xlsx"
            s3_client = boto3.client('s3')
            s3_client.upload_file(excel_path, bucket_name, s3_key)
            
            excel_files.append({
                "object_id": object_id,
                "local_path": excel_path,
                "s3_key": s3_key
            })
        
        # Generate a presigned URL for the rugosity directory
        # We can't create a presigned URL for a directory, so we'll upload a summary JSON file
        summary = {
            "session_id": session_id,
            "objects_processed": list(object_ids),
            "excel_files": [f["s3_key"] for f in excel_files]
        }
        
        summary_path = os.path.join(rugosity_dir, "rugosity_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f)
        
        s3_summary_key = f"{device_name}/Analysis/{session_id}/rugosity/rugosity_summary.json"
        s3_client.upload_file(summary_path, bucket_name, s3_summary_key)
        
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': s3_summary_key},
            ExpiresIn=3600  # URL expires in 1 hour
        )
        
        return jsonify({
            "message": "Rugosity analysis completed",
            "objects_processed": len(object_ids),
            "excel_files_created": len(excel_files),
            "summary_url": presigned_url
        }), 200
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/generate_masked_images', methods=['POST'])
def generate_masked_images():
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({"error": "Session ID is required"}), 400
        
        if not session_manager.session_exists(session_id):
            return jsonify({"error": f"Session {session_id} not found"}), 404
        
        session = session_manager.get_session(session_id)
        data_dir = session.get('data_dir')
        context_dir = session.get('context_dir')
        bucket_name = session.get('bucket_name')
        device_name = session.get('device_name')
        
        # Check if h5 file exists
        h5_file_path = os.path.join(context_dir, 'video_segments.h5')
        if not os.path.exists(h5_file_path):
            return jsonify({"error": "Video segments file not found. Run propagate_in_video first."}), 404
        
        # Create masked_images directory
        masked_images_dir = os.path.join(context_dir, 'masked_images')
        os.makedirs(masked_images_dir, exist_ok=True)
        
        # Import necessary libraries
        import glob
        from PIL import Image, ImageDraw
        import numpy as np
        from skimage import measure
        import random
        
        # Function to generate random distinct colors for each object
        def get_random_color():
            return (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
        
        # Get all image files
        frame_names = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
        if not frame_names:
            return jsonify({"error": f"No images found in {data_dir}"}), 404
        
        # Generate a color for each object ID
        with h5py.File(h5_file_path, "r") as f:
            all_object_ids = set()
            for frame_id in f.keys():
                frame_group = f[frame_id]
                all_object_ids.update([obj_id for obj_id in frame_group.keys()])
            
        # Create a fixed color map for all objects
        color_map = {obj_id: get_random_color() for obj_id in all_object_ids}
        
        # Process each frame
        uploaded_files = []
        with h5py.File(h5_file_path, "r") as f:
            frame_ids = sorted(f.keys(), key=lambda x: int(x))
            
            for frame_id in frame_ids:
                idx = int(frame_id)
                if idx >= len(frame_names):
                    print(f"Frame index {idx} out of range for images!")
                    continue
                
                # Load original image
                img_path = frame_names[idx]
                with Image.open(img_path) as img:
                    # Create a copy of the original image to draw on
                    img_with_masks = img.copy().convert('RGB')
                    draw = ImageDraw.Draw(img_with_masks)
                    img_width, img_height = img.size
                    
                    # Get all object masks for this frame
                    frame_group = f[frame_id]
                    for obj_id in frame_group.keys():
                        # Get the mask for this object
                        mask = frame_group[obj_id][:]
                        
                        # Ensure the mask is 2D by squeezing any extra dimensions
                        mask_binary = np.squeeze(mask > 0)
                        
                        # Resize mask if needed to match image dimensions
                        if mask_binary.shape != (img_height, img_width):
                            print(f"Resizing mask from {mask_binary.shape} to {(img_height, img_width)}")
                            # Use skimage resize which handles boolean arrays better
                            from skimage.transform import resize
                            mask_binary = resize(
                                mask_binary.astype(np.uint8), 
                                (img_height, img_width),
                                order=0,  # Nearest neighbor to preserve binary nature
                                preserve_range=True,
                                anti_aliasing=False
                            ).astype(bool)
                            print(f"After resize: mask shape: {mask_binary.shape}")
                        
                        # Find contours (boundaries) of the mask
                        contours = measure.find_contours(mask_binary, 0.5)
                        
                        # Draw each contour on the image
                        color = color_map[obj_id]
                        line_width = 3  # Width of the outline
                        
                        for contour in contours:
                            # Convert contour points to image coordinates (y, x to x, y)
                            # and scale if necessary
                            points = [(int(x), int(y)) for y, x in contour]
                            
                            # Draw the contour with PIL
                            for i in range(len(points) - 1):
                                draw.line([points[i], points[i+1]], fill=color, width=line_width)
                            # Connect last point to first point to close the contour
                            if len(points) > 1:
                                draw.line([points[-1], points[0]], fill=color, width=line_width)
                                
                        # Add object ID label near the top of the contour
                        if contours:
                            # Find a good position for the label (top of the largest contour)
                            largest_contour = max(contours, key=len)
                            label_pos = (int(largest_contour[:, 1].mean()), int(largest_contour[:, 0].min()) - 15)
                            
                            # Use a larger font size with bold text for better visibility
                            from PIL import ImageFont
                            try:
                                # Try to load a system font with larger size
                                font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
                            except IOError:
                                # Fall back to default font if the specified font is not available
                                font = ImageFont.load_default()
                            
                            # Draw a background box for better text visibility
                            label_text = f"Object: {obj_id}"
                            text_bbox = draw.textbbox(label_pos, label_text, font=font)
                            draw.rectangle([
                                (text_bbox[0]-5, text_bbox[1]-5),
                                (text_bbox[2]+5, text_bbox[3]+5)
                            ], fill=(0, 0, 0))
                            
                            # Draw the text with the larger font
                            draw.text(label_pos, label_text, fill=color, font=font)
                    
                    # Save the image with masks
                    output_filename = f"masked_{idx:03d}.jpg"
                    output_path = os.path.join(masked_images_dir, output_filename)
                    img_with_masks.save(output_path)
                    
                    # Upload to S3
                    s3_key = f"{device_name}/Analysis/{session_id}/masked_images/{output_filename}"
                    s3_client = boto3.client('s3')
                    s3_client.upload_file(output_path, bucket_name, s3_key)
                    
                    uploaded_files.append(s3_key)
        
        # Create a summary JSON with information about all processed files
        summary = {
            "session_id": session_id,
            "total_frames_processed": len(uploaded_files),
            "masked_image_files": uploaded_files
        }
        
        # Save summary file
        summary_path = os.path.join(masked_images_dir, "masked_images_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f)
        
        # Upload summary to S3
        s3_summary_key = f"{device_name}/Analysis/{session_id}/masked_images/masked_images_summary.json"
        s3_client.upload_file(summary_path, bucket_name, s3_summary_key)
        
        # Generate presigned URL for the summary file
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': s3_summary_key},
            ExpiresIn=3600  # URL expires in 1 hour
        )
        
        return jsonify({
            "message": "Masked images generated successfully",
            "frames_processed": len(uploaded_files),
            "summary_url": presigned_url
        }), 200
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/delete_session', methods=['POST'])
def delete_session():
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({"error": "Session ID is required"}), 400
        
        if not session_manager.session_exists(session_id):
            return jsonify({"error": f"Session {session_id} not found"}), 404
            
        # Get session data before deletion
        session = session_manager.get_session(session_id)
        session_dir = session.get('session_dir')
        bucket_name = session.get('bucket_name')
        device_name = session.get('device_name')
        
        # Delete from S3
        # List all objects with the session prefix in the Analysis folder
        s3_client = boto3.client('s3')
        s3_prefix = f"{device_name}/Analysis/{session_id}/"
        
        try:
            # List all objects with this prefix
            response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=s3_prefix
            )
            
            # Delete each object
            if 'Contents' in response:
                for obj in response['Contents']:
                    s3_client.delete_object(
                        Bucket=bucket_name,
                        Key=obj['Key']
                    )
                    print(f"Deleted S3 object: {obj['Key']}")
            
            # Continue deleting until all objects are gone (pagination)
            while response.get('IsTruncated', False):
                response = s3_client.list_objects_v2(
                    Bucket=bucket_name,
                    Prefix=s3_prefix,
                    ContinuationToken=response['NextContinuationToken']
                )
                if 'Contents' in response:
                    for obj in response['Contents']:
                        s3_client.delete_object(
                            Bucket=bucket_name,
                            Key=obj['Key']
                        )
                        print(f"Deleted S3 object: {obj['Key']}")
        
        except Exception as e:
            print(f"Warning: Error deleting S3 objects: {str(e)}")
        
        # Delete local files
        if os.path.exists(session_dir) and os.path.isdir(session_dir):
            import shutil
            shutil.rmtree(session_dir)
            print(f"Deleted local directory: {session_dir}")
        
        # Remove from session manager
        session_manager.delete_session(session_id)
        
        return jsonify({
            "message": f"Session {session_id} and all associated data deleted successfully"
        }), 200
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/clear_sessions', methods=['POST'])
def clear_sessions():
    try:
        # Get all session IDs before clearing
        all_session_ids = list(session_manager.sessions.keys())
        if not all_session_ids:
            return jsonify({"message": "No sessions to clear"}), 200
        
        deleted_count = 0
        failed_count = 0
        
        # Delete each session one by one
        for session_id in all_session_ids:
            try:
                session = session_manager.get_session(session_id)
                session_dir = session.get('session_dir')
                bucket_name = session.get('bucket_name')
                device_name = session.get('device_name')
                
                # Delete S3 objects
                s3_client = boto3.client('s3')
                s3_prefix = f"{device_name}/Analysis/{session_id}/"
                
                try:
                    # Delete S3 objects (paginated)
                    response = s3_client.list_objects_v2(
                        Bucket=bucket_name,
                        Prefix=s3_prefix
                    )
                    
                    # Delete objects in current page
                    if 'Contents' in response:
                        for obj in response['Contents']:
                            s3_client.delete_object(
                                Bucket=bucket_name,
                                Key=obj['Key']
                            )
                    
                    # Continue with pagination if needed
                    while response.get('IsTruncated', False):
                        response = s3_client.list_objects_v2(
                            Bucket=bucket_name,
                            Prefix=s3_prefix,
                            ContinuationToken=response['NextContinuationToken']
                        )
                        if 'Contents' in response:
                            for obj in response['Contents']:
                                s3_client.delete_object(
                                    Bucket=bucket_name,
                                    Key=obj['Key']
                                )
                except Exception as e:
                    print(f"Warning: Error deleting S3 objects for session {session_id}: {str(e)}")
                
                # Delete local directory
                if os.path.exists(session_dir) and os.path.isdir(session_dir):
                    import shutil
                    shutil.rmtree(session_dir)
                
                deleted_count += 1
            
            except Exception as e:
                print(f"Error deleting session {session_id}: {str(e)}")
                failed_count += 1
        
        # Reset session manager
        session_manager.clear_sessions()
        
        # Delete all files in the sessions directory
        sessions_dir = 'sessions'
        if os.path.exists(sessions_dir) and os.path.isdir(sessions_dir):
            for item in os.listdir(sessions_dir):
                item_path = os.path.join(sessions_dir, item)
                if os.path.isdir(item_path):
                    import shutil
                    shutil.rmtree(item_path)
        
        return jsonify({
            "message": f"All sessions cleared. Successfully deleted {deleted_count} sessions. Failed to delete {failed_count} sessions."
        }), 200
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Create sessions directory if it doesn't exist
    os.makedirs('sessions', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)