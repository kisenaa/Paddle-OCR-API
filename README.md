## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
    - [Run the API server](#run-the-api-server)
    - [API Endpoint](#api-endpoint)
- [Model Info](#model-info)
- [Development](#development)
- [References](#references)

## OCR Rest API
This project provides an OCR (Optical Character Recognition) API using [FastAPI](https://fastapi.tiangolo.com/) and [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR). It supports document image text detection and recognition using pre-trained PaddleOCR models.

## Features

- REST API for OCR processing
- Supports image upload and returns detected text and preview images
- Uses PaddleOCR models that convertede to openvino models to run on intel CPU
- Returns results in JSON and base64-encoded image format

## Project Structure

```
.
├── Makefile
├── README.md
├── requirements.txt
└── src/
    ├── main.py
    ├── model/
    │   └── ocr_model.py
    └── service/
        └── ocr_service.py
    └── modules/paddleocr_openvino
        └── fonts
        └── models
        └── {py_files}.py
```

## Installation

1. **Clone the repository**

   ```sh
   git clone https://github.com/kisenaa/Paddle-OCR-API --depth 1
   cd Paddle-OCR-API
   ```

2. **Run makefile command to install depedencies**

   ```sh
   make install

   # activate virtual environment
   # On Windows
   venv\Scripts\activate
   # On Unix
   source venv/bin/activate
   ```

3. **Download PaddleOCR / openvino models and fonts**
   - The models will be automatically downloaded on first run.
   - Alternatively, place the model folders inside `modules/paddleocr_openvino`

4. **Open the API Documentation**
   - The server will be run on http://127.0.0.1:8000/
   - To open the docs, go to http://127.0.0.1:8000/docs

## Usage

### Run the API server

```sh
make run
# or
fastapi run .\src\main.py
```

### API Endpoint

#### `POST /api/v1/run_ocr/`

- **Description:** Run OCR on an uploaded image file.
- **Request:** `multipart/form-data` with an `image_file` field.
- **Response:** JSON containing filename, detected text, and preview images.

**Example using `curl`:**

```sh
curl -X POST "http://localhost:8000/api/v1/run_ocr/" \
     -F "image_file=@your_image.jpg"
```

**Response:**

```json
{
  "filename": "string",
  "outputs": {
    "inference_result": [
      [
        "NIK",
        0.998
      ],
      [
        "Nama",
        0.995
      ],
      [
        ": Johnny Andrean",
        0.997
      ]
    ],
    "image_result": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ..."
  }
}
```

## Model Info

- Models are stored in `paddle_model/PP-OCRv5_server_det/` and `paddle_model/PP-OCRv5_mobile_det/`.
- For more details, see the official [PaddleOCR documentation](https://paddlepaddle.github.io/PaddleOCR/latest/en/index.html).

## Development

- Main API logic: [`src/main.py`](src/main.py)
- OCR service: [`src/service/ocr_service.py`](src/service/ocr_service.py)
- Data models: [`src/model/ocr_model.py`](src/model/ocr_model.py)


## References

- [PaddleOCR Repo](https://github.com/PaddlePaddle/PaddleOCR)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
