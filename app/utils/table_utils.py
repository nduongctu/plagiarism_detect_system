import torch
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import math


# Hàm tự động xoay ảnh
def auto_rotate_image(image: Image.Image) -> Image:
    image_cv = np.array(image)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Dùng Hough Transform để tìm các đường thẳng trong ảnh
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90  # Chuyển angle về độ
            angles.append(angle)

        median_angle = np.median(angles)  # Lấy góc trung bình

        (h, w) = image_cv.shape[:2]
        center = (w // 2, h // 2)

        # Tạo ma trận xoay và xoay ảnh
        matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated_image_cv = cv2.warpAffine(image_cv, matrix, (w, h), flags=cv2.INTER_CUBIC,
                                          borderMode=cv2.BORDER_REPLICATE)

        rotated_image_pil = Image.fromarray(cv2.cvtColor(rotated_image_cv, cv2.COLOR_BGR2RGB))
        return rotated_image_pil
    else:
        print("Không tìm thấy đường thẳng, ảnh không được xoay.")
        return image  # Nếu không tìm được đường thẳng, trả về ảnh gốc


# Hàm tăng kích thước ảnh nếu cần
def upscale_image(image: Image.Image, target_min_size: int = 1240) -> Image.Image:
    w, h = image.size
    if max(w, h) >= target_min_size:
        return image  # Không cần resize nếu kích thước đủ lớn
    scale = target_min_size / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    return image.resize(new_size, Image.BICUBIC)


# Tăng độ tương phản để làm rõ nét văn bản
def enhance_contrast(image: Image.Image, factor: float = 2.0) -> Image.Image:
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


# Chuyển về grayscale và làm nét ảnh
def preprocess_for_ocr(image: Image.Image) -> Image.Image:
    gray = image.convert("L")  # Chuyển ảnh sang grayscale
    return gray.filter(ImageFilter.SHARPEN)  # Làm nét ảnh


# Hàm chuẩn bị ảnh trước khi OCR
def prepare_image_for_ocr(image: Image.Image) -> Image.Image:
    image = upscale_image(image)  # Tăng kích thước ảnh nếu cần
    image = enhance_contrast(image)  # Tăng độ tương phản
    image = preprocess_for_ocr(image)  # Chuyển grayscale và làm nét
    return image
