import base64
import datetime
from io import BytesIO
import os
import numpy as np
from PIL import Image
import uuid

output_dir = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'outputs', 'files'))
os.makedirs(output_dir, exist_ok=True)

static_serve_base_url = 'http://127.0.0.1:8888/files/'


def save_output_file(img: np.ndarray, use_webp: bool = False) -> str:
    current_time = datetime.datetime.now()
    date_string = current_time.strftime("%Y-%m-%d")
    if use_webp:
        ext = '.webp'
        format_type = 'webp'
    else:
        ext = '.png'
        format_type = 'PNG'
    filename = os.path.join(date_string, str(uuid.uuid4()) + ext)
    file_path = os.path.join(output_dir, filename)

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    Image.fromarray(img).save(file_path, format=format_type)
    return filename


def delete_output_file(filename: str):
    file_path = os.path.join(output_dir, filename)
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return
    try:
        os.remove(file_path)
    except OSError:
        print(f"Delete output file failed: {filename}")


def output_file_to_base64img(filename: str | None, use_webp: bool = False) -> str | None:
    if filename is None:
        return None
    file_path = os.path.join(output_dir, filename)
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return None
    if use_webp:
        format_type = 'webp'
    else:
        format_type = 'PNG'
    img = Image.open(file_path)
    output_buffer = BytesIO()
    img.save(output_buffer, format=format_type)
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str


def output_file_to_bytesimg(filename: str | None, use_webp: bool = False) -> bytes | None:
    if filename is None:
        return None
    file_path = os.path.join(output_dir, filename)
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return None
    if use_webp:
        format_type = 'webp'
    else:
        format_type = 'PNG'
    img = Image.open(file_path)
    output_buffer = BytesIO()
    img.save(output_buffer, format=format_type)
    byte_data = output_buffer.getvalue()
    return byte_data


def get_file_serve_url(filename: str | None) -> str | None:
    if filename is None:
        return None
    return static_serve_base_url + filename.replace('\\', '/')


def output_file_to_file_path(filename: str | None) -> str | None:
    if filename is None:
        return None
    file_path = os.path.join(output_dir, filename)
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return None
    return file_path


def create_output_file_name(use_webp: bool = False) -> str:
    current_time = datetime.datetime.now()
    date_string = current_time.strftime("%Y-%m-%d")

    if use_webp:
        ext = '.webp'
    else:
        ext = '.png'
    filename = os.path.join(date_string, str(uuid.uuid4()) + ext)
    file_path = os.path.join(output_dir, filename)

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    open(file_path, 'x')
    return filename
