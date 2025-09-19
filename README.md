# OCR FastAPI Service

This project provides an OCR (Optical Character Recognition) API using [FastAPI](https://fastapi.tiangolo.com/) and [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR). It supports document image text detection and recognition using pre-trained PaddleOCR models.

## Features

- REST API for OCR processing
- Supports image upload and returns detected text and preview images
- Uses PaddleOCR models (`PP-OCRv5_server_det`, `PP-OCRv5_mobile_det`)
- Asynchronous file handling for efficient processing
- Returns results in JSON and base64-encoded image format

## Project Structure

```
.
├── Makefile
├── README.md
├── requirements.txt
├── paddle_model/
│   ├── PP-OCRv5_mobile_det/
│   └── PP-OCRv5_server_det/
└── src/
    ├── main.py
    ├── model/
    │   └── ocr_model.py
    └── service/
        └── ocr_service.py
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

3. **Download PaddleOCR models**
   - The models will be automatically downloaded on first run.
   - Alternatively, place the model folders inside `paddle_model/` as shown above.

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
  "filename": "your_image.jpg",
  "outputs": [
    {
      "json_result": {
        "res": ["Detected text line 1", "Detected text line 2", ...]
      },
      "image_result": {
        "ocr_res_img": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ..."
      }
    }
  ]
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
