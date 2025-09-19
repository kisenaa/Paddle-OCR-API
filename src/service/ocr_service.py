''' 
    Run OCR on an uploaded file and return the results via images and json
'''
import base64
from time import time, perf_counter
import cv2
from paddleocr import PaddleOCR
from paddlex.inference.pipelines.ocr.result import OCRResult
import os
from typing import BinaryIO, Optional
import io
import requests
import zipfile

# path
MODEL_NAME = 'PP-OCRv5_server_det'
current_file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(current_file_path)
model_path = os.path.join(dir_path, f'../../paddle_model/{MODEL_NAME}/')

# download and check model
def download(link: str, zip_name: str):
    print("Downloading model...")
    base_name = os.path.splitext(zip_name)[0]  # e.g. PP-OCRv5_server_det
    zip_path = os.path.join(dir_path, f"../../paddle_model/{zip_name}")
    extract_path = os.path.join(dir_path, f"../../paddle_model/{base_name}")

    response = requests.get(link)
    with open(zip_path, "wb") as f:
        f.write(response.content)

    # ensure target folder exists
    os.makedirs(extract_path, exist_ok=True)

    # extract into that folder
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    os.remove(zip_path)
    print("Model downloaded and extracted to:", extract_path)


def check_and_download_model():
    global model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)
        # Download the model from the official PaddleOCR GitHub repository
        
        download("https://drive.usercontent.google.com/download?id=1uJXl72pVGJdYSTfZXM-D_QjCYddo29pE&export=download&confirm=t", "PP-OCRv5_server_det.zip")
        download("https://drive.usercontent.google.com/download?id=1MpTYLMIvcZWt0USaQaH9I0LU2q2rlC9k&export=download&confirm=t", "PP-OCRv5_mobile_det.zip")
    else:
        print("Model already exists at:", model_path)

check_and_download_model()
OCR = PaddleOCR(
    use_doc_orientation_classify=False, 
    use_doc_unwarping=False, 
    use_textline_orientation=False,
    text_detection_model_name=MODEL_NAME,
    text_detection_model_dir=model_path,
    )
        
# reference: https://www.paddleocr.ai/main/en/quick_start.html#python-script-usage
def handle_ocr_inference(temp_image_path: str):
    global OCR
    print("Processing file: ", temp_image_path)
    start = perf_counter()
    results: list[OCRResult] = OCR.predict(temp_image_path)
    end = perf_counter()
    print(f"Time taken: {end - start:.2f} seconds")

    outputs = []
    start = perf_counter()
    for res in results:
        # Store recognized text in json_data as json
        json_data = {}
        for key in res.json.keys():
            json_data[key] = res.json[key].get('rec_texts', [])
        
        # Store image for OCR preview as base64 encoded JPEG
        # remove this to save post-processing time
        image_data = {}
        for key in res.img.keys():
            if key != 'ocr_res_img':
                continue
            
            buf = io.BytesIO()
            res.img[key].save(buf, format="JPEG", quality=30, optimize=False)
            b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")
            image_data[key] = f"data:image/jpeg;base64,{b64_img}"

        outputs.append({
            "json_result": json_data,
            "image_result": image_data
        })
    end = perf_counter()
    print(f"Post-processing time: {end - start:.2f} seconds")

    return outputs
