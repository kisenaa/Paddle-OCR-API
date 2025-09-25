''' 
    Run OCR on an uploaded file and return the results via images and json
'''
from time import time, perf_counter
import cv2
from modules.paddleocr_openvino.OCR import OCR, sav2Img
import os
import requests
import zipfile

# path
current_file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(current_file_path)

# download and check model
def download(link: str, zip_name: str):
    print("Downloading model...")
    base_name = os.path.splitext(zip_name)[0]  # e.g. PP-OCRv5_server_det
    zip_path = os.path.join(dir_path, f"{dir_path}/../modules/paddleocr_openvino/")
    extract_path = os.path.join(dir_path, f"{dir_path}/../modules/paddleocr_openvino/")

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
    if not os.path.exists(f"{dir_path}/../modules/paddleocr_openvino/models") or not os.path.exists(f"{dir_path}/../modules/paddleocr_openvino/fonts"):
        print("Model not found, downloading...")        

        # Download the model from the official PaddleOCR GitHub repository
        download("https://drive.usercontent.google.com/download?id=1CVL7hXL_kIVhMo8luDwPuBGSTfRMeUyS&export=download&authuser=0&confirm=t", "fonts_and_model.zip")
    else:
        print("Model already exists at:", {dir_path})

check_and_download_model()
OCR_MODEL = OCR(use_angle_cls=False, use_gpu=False)

# reference: https://www.paddleocr.ai/main/en/quick_start.html#python-script-usage
def handle_ocr_inference(temp_image_path: str):
    global OCR_MODEL

    print("Processing file: ", temp_image_path)
    img = cv2.imread(temp_image_path)
    start = perf_counter()
    results = OCR_MODEL.ocr(img)
    end = perf_counter()
    print(f"Time taken: {end - start:.2f} seconds")

    results_json = [[box[1][0], box[1][1]] for box in results[0]]

    result_image = sav2Img(img, results)

    return {"inference_result": results_json, "image_result": result_image}