import os
from tqdm import tqdm
from typing import List, Dict, Any
from PIL import Image
import cv2
import numpy as np


class BatchProcessor:
    SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

    def __init__(self, processor):
        self.processor = processor

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        settings: Dict[str, Any],
        recursive: bool = False,
        output_format: str = "JPEG",
        quality: int = 95,
    ) -> None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        files = self._collect_files(input_dir, recursive)
        for file_path in tqdm(files, desc="Обработка изображений"):
            try:
                self._process_single_file(
                    file_path, output_dir, settings, output_format, quality
                )
            except Exception as e:
                print(f"Ошибка при обработке {file_path}: {str(e)}")

    def _collect_files(self, directory: str, recursive: bool) -> List[str]:
        files = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if filename.lower().endswith(self.SUPPORTED_EXTENSIONS):
                    files.append(os.path.join(root, filename))
            if not recursive:
                break
        return files

    def _process_single_file(
        self,
        input_path: str,
        output_dir: str,
        settings: Dict[str, Any],
        output_format: str,
        quality: int,
    ) -> None:
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError("Не удалось прочитать изображение")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        processed_img = self.processor.apply_settings(img, settings)
        rel_path = os.path.relpath(input_path, os.path.dirname(input_path))
        output_path = os.path.join(output_dir, os.path.basename(input_path))
        if output_format == "JPEG" and not output_path.lower().endswith(
            (".jpg", ".jpeg")
        ):
            output_path = os.path.splitext(output_path)[0] + ".jpg"
        elif output_format == "PNG" and not output_path.lower().endswith(".png"):
            output_path = os.path.splitext(output_path)[0] + ".png"
        pil_img = Image.fromarray(processed_img)
        pil_img.save(output_path, format=output_format, quality=quality)
