import base64
import json
import os
import time
from io import BytesIO

import numpy as np
from PIL import ImageGrab, Image, ImageDraw, ImageFont
from sklearn.linear_model import LinearRegression
from tencentcloud.common import credential
from tencentcloud.common.exception import TencentCloudSDKException
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.ocr.v20181119 import ocr_client, models
from tencentcloud.tiia.v20190529 import tiia_client
from tencentcloud.tiia.v20190529 import tiia_client, models

def capture_screen():
    # Add a delay to switch windows
    time.sleep(3)  # Delay for 3 seconds

    # Capture the entire screen
    screenshot = ImageGrab.grab()
    screenshot.save("icons_identification2/screenshot.png")
    return screenshot


def detect_labels(image):
    """Detect labels (icons) in the given PIL.Image using Tencent Cloud API."""
    # Create credential object
    your_secret_id = os.getenv("SECRET_ID")
    your_secret_key = os.getenv("SECRET_KEY")

    cred = credential.Credential(your_secret_id, your_secret_key)
    http_profile = HttpProfile()
    http_profile.endpoint = "tiia.tencentcloudapi.com"
    client_profile = ClientProfile(httpProfile=http_profile)
    client = tiia_client.TiiaClient(cred, "ap-shanghai", client_profile)

    # Encode the image to base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Build the request
    req = models.DetectLabelRequest()
    req.ImageBase64 = encoded_image

    # Call the API
    try:
        response = client.DetectLabel(req)
        print(response.to_json_string())
        return response.Labels
    except TencentCloudSDKException as err:
        print(f"An error occurred: {err}")
        return []


def draw_labels(image, labels):
    """Draw labels and bounding boxes on the PIL.Image and return the marked image."""
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 16)  # 使用Arial字体

    for label in labels:
        label_name = label.Name
        confidence = label.Confidence
        description = f"{label_name}: {confidence:.2f}%"

        # 检查是否有BoundingBox信息
        if hasattr(label, 'BoundingBox'):
            box = label.BoundingBox
            left = box.Left * image.width
            top = box.Top * image.height
            width = box.Width * image.width
            height = box.Height * image.height

            # 绘制矩形框
            draw.rectangle([left, top, left + width, top + height], outline="red", width=2)

            # 绘制标签文本在框上方
            draw.text((left, top - 10), description, font=font, fill="red")
        else:
            print(f"No BoundingBox for label: {label_name}")

    return image


# Capture the screen
screenshot = capture_screen()
print("Screenshot saved as 'screenshot.png'")

# Detect labels in the image
labels = detect_labels(screenshot)

# Draw labels on the image
marked_image = draw_labels(screenshot, labels)

# Save the marked image
marked_image_path = "icons_identification2/marked_screenshot.png"
marked_image.save(marked_image_path)
print(f"Marked image saved as '{marked_image_path}'")
