import time

import cv2
import numpy as np
from PIL import ImageGrab


def capture_screen():
    # Add a delay to switch windows
    time.sleep(3)  # Delay for 3 seconds

    # Capture the entire screen
    screenshot = ImageGrab.grab()
    screenshot.save("color_distribution_thresholding/screenshot1.png")
    return screenshot


# 截图
image = capture_screen()

# 将 PIL.Image 对象转换为 numpy 数组
image_np = np.array(image)

# 确保图像是 BGR 格式，如果是 RGB 格式则需要转换
if image_np.shape[2] == 3:  # 检查是否为彩色图像
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

# 将图像转换为HSV颜色空间
hsv_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
cv2.imwrite('color_distribution_thresholding/output_image_hsv1.jpg', hsv_image)

# 设置HSV阈值范围
lower_bound = np.array([35, 100, 100])  # 示例值，需要根据实际情况调整
upper_bound = np.array([85, 255, 255])  # 示例值，需要根据实际情况调整

# 应用阈值分割
mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

# 应用掩码到原始图像
result_image = cv2.bitwise_and(image_np, image_np, mask=mask)

cv2.imwrite('color_distribution_thresholding/output_mask1.jpg', mask)
cv2.imwrite('color_distribution_thresholding/output_mask_result1.jpg', result_image)
