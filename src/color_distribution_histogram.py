import time
from scipy.signal import find_peaks
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

# 计算色调直方图
hist = cv2.calcHist([hsv_image], [0], None, [180], [0, 179])
hist = hist.flatten()

# 找到直方图的峰值
peaks, _ = find_peaks(hist, height=0.01*hist.max(), distance=10)

# 选择前几大峰值
num_colors = 10  # 选择主要颜色的数量
main_colors = peaks[np.argsort(hist[peaks])[-num_colors:]]

# 创建掩码并进行颜色分割
masks = []
for color in main_colors:
    lower_bound = np.array([color - 10, 100, 100])
    upper_bound = np.array([color + 10, 255, 255])
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    masks.append(mask)

# 合并掩码
combined_mask = np.zeros_like(masks[0])
for mask in masks:
    combined_mask = cv2.bitwise_or(combined_mask, mask)


# 应用掩码到原始图像
result_image = cv2.bitwise_and(image_np, image_np, mask=combined_mask)

cv2.imwrite('color_distribution_thresholding/output_mask1.jpg', combined_mask)
cv2.imwrite('color_distribution_thresholding/output_mask_result1.jpg', result_image)
