# Origianl parser and related helper functions code from https://gitlab.com/mattbas/python-lottie
# SVG parse using https://gitlab.com/mattbas/python-lottie/. 
# Change to original code : Generating Lottie using pydantic based object model.

import uvicorn
from fastapi import FastAPI, File, status
from core.svg import convert_svg_to_lottie
from core.svg import convert_svg_to_lottie_def
import shutil
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from fastapi import UploadFile
import xml.etree.cElementTree as et
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from enum import Enum
import cairosvg
import time
import uuid

from cli import convert_zip

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/files", StaticFiles(directory=UPLOADS_DIR), name="files")

origins = [
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "http://localhost",
    "http://localhost:8080"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def is_svg(filename):
    tag = None
    with open(filename, "r") as f:
        try:
            for _, el in et.iterparse(f, ('start',)):
                tag = el.tag
                break
        except et.ParseError:
            pass
    return tag == '{http://www.w3.org/2000/svg}svg'


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    return PlainTextResponse(str(exc.detail), status_code=exc.status_code)


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>SVG/ZIP to Lottie Converter</title>
        <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
        <style>
          body {
            font-family: Arial, sans-serif;
            background: #f7f7fb;
            margin: 0;
            padding: 0;
          }
          .container {
            max-width: 900px;
            margin: 40px auto;
            background: #ffffff;
            padding: 32px;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
          }
          h1 {
            margin-top: 0;
            font-size: 28px;
          }
          .upload-section {
            display: flex;
            flex-direction: column;
            gap: 16px;
            padding: 20px;
            border: 2px dashed #c7c7d6;
            border-radius: 12px;
            background: #fafafe;
          }
          input[type="file"] {
            padding: 12px;
          }
          button {
            padding: 12px 18px;
            background: #4f46e5;
            color: #fff;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
          }
          button:disabled {
            background: #9ca3af;
            cursor: not-allowed;
          }
          .status {
            font-size: 14px;
            color: #374151;
          }
          .result {
            margin-top: 24px;
            display: none;
          }
          .download-link {
            display: inline-block;
            margin-bottom: 16px;
            text-decoration: none;
            background: #10b981;
            color: #fff;
            padding: 10px 16px;
            border-radius: 8px;
            font-weight: 600;
          }
          .error {
            color: #dc2626;
          }
        </style>
      </head>
      <body>
        <div class="container">
          <h1>SVG/ZIP to Lottie Converter</h1>
          <p>Select an SVG/XML file or a ZIP containing SVG/XML frames.</p>
          <div class="upload-section">
            <input id="fileInput" type="file" accept=".svg,.xml,.zip" />
            <button id="uploadButton">Upload & Convert</button>
            <div id="status" class="status"></div>
          </div>

          <div id="result" class="result">
            <a id="downloadLink" class="download-link" href="#" download>Download Lottie JSON</a>
            <div>
              <lottie-player
                id="lottiePlayer"
                autoplay
                controls
                loop
                mode="normal"
                src=""
                style="width: 320px"
              ></lottie-player>
            </div>
          </div>
        </div>
        <script>
          const fileInput = document.getElementById("fileInput");
          const uploadButton = document.getElementById("uploadButton");
          const statusEl = document.getElementById("status");
          const resultEl = document.getElementById("result");
          const downloadLink = document.getElementById("downloadLink");
          const lottiePlayer = document.getElementById("lottiePlayer");

          uploadButton.addEventListener("click", async () => {
            if (!fileInput.files.length) {
              statusEl.textContent = "Please choose a file before uploading.";
              statusEl.classList.add("error");
              return;
            }

            statusEl.classList.remove("error");
            statusEl.textContent = "Uploading and converting... Please wait.";
            uploadButton.disabled = true;
            resultEl.style.display = "none";

            const formData = new FormData();
            formData.append("file", fileInput.files[0]);

            try {
              const response = await fetch("/convert/", {
                method: "POST",
                body: formData,
              });
              const data = await response.json();

              if (!response.ok) {
                throw new Error(data.message || "Conversion failed.");
              }

              statusEl.textContent = data.message || "Conversion complete.";
              downloadLink.href = data.download_url;
              downloadLink.setAttribute("download", data.filename);
              lottiePlayer.setAttribute("src", data.json_url);
              resultEl.style.display = "block";
            } catch (error) {
              statusEl.textContent = error.message;
              statusEl.classList.add("error");
            } finally {
              uploadButton.disabled = false;
            }
          });
        </script>
      </body>
    </html>
    """


def _sanitize_filename(filename: str) -> str:
    safe_name = Path(filename).name
    if not safe_name:
        safe_name = f"upload-{uuid.uuid4().hex}"
    return safe_name


def _convert_svg_file(input_path: Path, output_path: Path, optimize: bool) -> dict:
    if not is_svg(str(input_path)):
        raise ValueError("Invalid file type. Expected SVG/XML content.")

    with NamedTemporaryFile(delete=False, suffix=".svg") as tmp_svg:
        tmp_path = Path(tmp_svg.name)

    cairosvg.svg2svg(file_obj=open(input_path, 'rb'), write_to=tmp_path)

    try:
        if optimize:
            anim = convert_svg_to_lottie(str(tmp_path))
        else:
            anim = convert_svg_to_lottie_def(str(tmp_path))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(anim, f, indent=2, ensure_ascii=False)
        return anim
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@app.post("/convert/")
def convert_file(
    optimize: bool = False,
    file: UploadFile = File(...)
):
    safe_name = _sanitize_filename(file.filename or "")
    upload_path = UPLOADS_DIR / safe_name
    suffix = upload_path.suffix.lower()

    with open(upload_path, "wb") as out_file:
        shutil.copyfileobj(file.file, out_file)

    output_json = upload_path.with_suffix(".json")

    try:
        if suffix == ".zip":
            convert_zip(str(upload_path), str(output_json), optimize=optimize)
        else:
            _convert_svg_file(upload_path, output_json, optimize)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    file_url = f"/files/{output_json.name}"
    return {
        "success": True,
        "message": "Conversion complete.",
        "filename": output_json.name,
        "download_url": file_url,
        "json_url": file_url,
    }


@app.post("/uploadsvg/")
def create_upload_file_def(
    optimize: bool = False,
    output_path: str = None,
    file: UploadFile = File(...)
):
    """
    Convert SVG to Lottie JSON.
    
    Args:
        optimize: If True, use optimized conversion mode
        output_path: Optional path to save the Lottie JSON file (e.g., "/path/to/output.json")
        file: The SVG file to convert
    
    Returns:
        Lottie JSON object (and saves to file if output_path is specified)
    """
    try:
        suffix = Path(file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)
            file.file.close()

            newfile = NamedTemporaryFile(delete=False, suffix=".svg")

        cairosvg.svg2svg(file_obj=open(tmp_path, 'rb'), write_to=newfile.name)

    finally:
        if (is_svg(tmp_path)):
            if not optimize:
                anim = convert_svg_to_lottie_def(str(newfile.name))
            else:
                anim = convert_svg_to_lottie(str(newfile.name))

            # Clean up temp files
            newfilepath = newfile.name
            os.unlink(newfilepath)
            assert not os.path.exists(newfilepath)
            os.unlink(tmp_path)
            assert not os.path.exists(tmp_path)

            # Save to file if output_path is specified
            if output_path:
                output_file = Path(output_path)
                # Create parent directories if they don't exist
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(anim, f, indent=2, ensure_ascii=False)
                return {
                    "success": True,
                    "message": f"Lottie JSON saved to {output_path}",
                    "output_path": str(output_file.absolute()),
                    "data": anim
                }

            return anim
        else:
            error = {"success": False, "message": "Invalid file type"}
            return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=error)


if __name__ == "__main__":
    uvicorn.run("svgtolottie:app", host="0.0.0.0")
