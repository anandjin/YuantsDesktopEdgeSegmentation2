import io
import os
from PIL import ImageGrab
import cv2
from google.cloud import vision
import time

# 设置环境变量
os.environ[
    'GOOGLE_APPLICATION_CREDENTIALS'] = 'E:\project\pythonProjects\YuantsDesktopEdgeSegmentation\data\keys\icon-recognition-430313-605ee6b8129a.json'


def capture_screen():
    print("等待3秒后开始截屏...")
    time.sleep(3)  # 延迟3秒，用于切换窗口
    screenshot = ImageGrab.grab()
    screenshot.save("icons_identification/screenshot.png")
    print("截屏完成，保存为 screenshot.png")
    return screenshot


def detect_and_display_icons(file_path, output_path='output.png'):
    """Detects icons in the file, draws bounding boxes, and saves the image."""
    client = vision.ImageAnnotatorClient()

    # 读取图像文件
    with io.open(file_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    # 调用 Vision API 进行图标识别
    response = client.logo_detection(image=image)
    logos = response.logo_annotations

    # 读取图像用于展示
    image_cv = cv2.imread(file_path)

    # 绘制检测到的图标边界框
    for logo in logos:
        print(f'Logo description: {logo.description}')
        print(f'Confidence: {logo.score}')
        vertices = [(vertex.x, vertex.y) for vertex in logo.bounding_poly.vertices]
        for i in range(len(vertices)):
            start_point = vertices[i]
            end_point = vertices[(i + 1) % len(vertices)]
            cv2.line(image_cv, start_point, end_point, (0, 255, 0), 2)

    # 保存绘制完边界框的图像
    cv2.imwrite(output_path, image_cv)
    print(f"Output image saved to {output_path}")

    # 展示图像
    cv2.imshow('Detected Icons', image_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 先截屏
capture_screen()
screenshot_path = 'icons_identification/screenshot.png'

# 然后进行图标识别并保存和展示结果
output_image_path = 'icons_identification/output.png'
detect_and_display_icons(screenshot_path, output_image_path)
