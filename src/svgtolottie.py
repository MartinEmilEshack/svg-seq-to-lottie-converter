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

from enum import Enum
import cairosvg
import time
import base64
import tempfile

from cli import convert_zip, export_dotlottie

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML_PATH = BASE_DIR / "templates" / "index.html"

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
    return INDEX_HTML_PATH.read_text(encoding="utf-8")


def _sanitize_filename(filename: str) -> str:
    safe_name = Path(filename).name
    if not safe_name:
        safe_name = "upload"
    return safe_name


def _convert_svg_file(input_path: Path, output_path: Path, optimize: bool, frame_rate: int) -> dict:
    if not is_svg(str(input_path)):
        raise ValueError("Invalid file type. Expected SVG/XML content.")

    with NamedTemporaryFile(delete=False, suffix=".svg") as tmp_svg:
        tmp_path = Path(tmp_svg.name)

    cairosvg.svg2svg(file_obj=open(input_path, 'rb'), write_to=str(tmp_path))

    try:
        if optimize:
            anim = convert_svg_to_lottie(str(tmp_path), frame_rate=frame_rate)
        else:
            anim = convert_svg_to_lottie_def(str(tmp_path), frame_rate=frame_rate)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(anim, f, separators=(",", ":"), ensure_ascii=False)
        return anim
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


class OutputFormat(str, Enum):
    json = "json"
    dotlottie = "dotlottie"


@app.post("/convert/")
def convert_file(
    optimize: bool = False,
    frame_rate: int = 30,
    output_format: OutputFormat = OutputFormat.json,
    file: UploadFile = File(...)
):
    safe_name = _sanitize_filename(file.filename or "")
    suffix = Path(safe_name).suffix.lower()

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            upload_path = temp_path / f"upload{suffix or ''}"
            output_json = temp_path / "output.json"

            with open(upload_path, "wb") as out_file:
                shutil.copyfileobj(file.file, out_file)

            if suffix == ".zip":
                anim = convert_zip(str(upload_path), str(output_json), optimize=optimize, pretty=False, frame_rate=frame_rate)
            else:
                anim = _convert_svg_file(upload_path, output_json, optimize, frame_rate)

            json_bytes = output_json.read_bytes()
            json_payload = {
                "filename": "output.json",
                "content_type": "application/json",
                "content_base64": base64.b64encode(json_bytes).decode("ascii"),
                "data": anim,
            }

            if output_format == OutputFormat.dotlottie:
                output_file = temp_path / "output.lottie"
                export_dotlottie(anim, str(output_file))
                file_payload = {
                    "filename": "output.lottie",
                    "content_type": "application/zip",
                    "content_base64": base64.b64encode(output_file.read_bytes()).decode("ascii"),
                }
            else:
                file_payload = {
                    "filename": "output.json",
                    "content_type": "application/json",
                    "content_base64": json_payload["content_base64"],
                }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        file.file.close()

    return {
        "success": True,
        "message": "Conversion complete.",
        "output_file": file_payload,
        "json_file": json_payload,
    }


@app.post("/uploadsvg/")
def create_upload_file_def(
    optimize: bool = False,
    frame_rate: int = 30,
    output_path: str = None,
    file: UploadFile = File(...)
):
    """
    Convert SVG to Lottie JSON.
    
    Args:
        optimize: If True, use optimized conversion mode
        frame_rate: Frame rate for the output animation
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
                anim = convert_svg_to_lottie_def(str(newfile.name), frame_rate=frame_rate)
            else:
                anim = convert_svg_to_lottie(str(newfile.name), frame_rate=frame_rate)

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
