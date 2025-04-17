import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
import math

image_processor = AutoImageProcessor.from_pretrained(
    "microsoft/table-transformer-detection",
    use_fast=True
)
model = TableTransformerForObjectDetection.from_pretrained(
    "microsoft/table-transformer-detection"
)
model.eval()


def auto_rotate_image(image: Image.Image) -> Image:
    image_cv = np.array(image)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Dùng Hough Transform để tìm các đường thẳng trong ảnh
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            angles.append(angle)

        median_angle = np.median(angles)

        (h, w) = image_cv.shape[:2]
        center = (w // 2, h // 2)

        matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated_image_cv = cv2.warpAffine(image_cv, matrix, (w, h), flags=cv2.INTER_CUBIC,
                                          borderMode=cv2.BORDER_REPLICATE)

        rotated_image_pil = Image.fromarray(cv2.cvtColor(rotated_image_cv, cv2.COLOR_BGR2RGB))
        return rotated_image_pil
    else:
        return image


def crop_tables_from_image(image: Image.Image, threshold: float = 0.7,
                           padding_horizontal: int = 15, padding_vertical: int = 5):
    width, height = image.size

    inputs = image_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    results = image_processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=[(height, width)]
    )[0]

    cropped_tables = []
    for label, box in zip(results["labels"], results["boxes"]):
        if model.config.id2label[label.item()] == "table":
            x0 = max(0, int(box[0].item()) - padding_horizontal)
            y0 = max(0, int(box[1].item()) - padding_vertical)
            x1 = min(width, int(box[2].item()) + padding_horizontal)
            y1 = min(height, int(box[3].item()) + padding_vertical)
            cropped = image.crop((x0, y0, x1, y1))

            cropped = auto_rotate_image(cropped)

            cropped_tables.append(cropped)

    return cropped_tables
