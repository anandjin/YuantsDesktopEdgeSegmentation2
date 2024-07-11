import time

import cv2
import numpy as np
from PIL import ImageGrab

time.sleep(3)
# 截取屏幕
screenshot = ImageGrab.grab()
screenshot = np.array(screenshot)

# 转换为BGR格式（OpenCV使用的格式）
screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

# 读取已知图片列表
templates = ['images/image3.png', 'images/image4.png']

# 转换已知图片为灰度图
gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
gray_templates = [cv2.cvtColor(cv2.imread(template), cv2.COLOR_BGR2GRAY) for template in templates]

# 定义一个函数来执行模板匹配
def find_template_positions(screenshot, template, threshold=0.5):
    result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= threshold)
    positions = []
    for pt in zip(*loc[::-1]):
        positions.append((pt[0], pt[1], pt[0] + template.shape[1], pt[1] + template.shape[0]))
    return positions

# 记录所有匹配的位置
all_positions = []

for gray_template in gray_templates:
    positions = find_template_positions(gray_screenshot, gray_template)
    all_positions.extend(positions)

# 在截屏上绘制所有匹配的位置
for (x1, y1, x2, y2) in all_positions:
    cv2.rectangle(screenshot, (x1, y1), (x2, y2), (0, 0, 255), 2)


# 保存结果图像
cv2.imwrite('result_screenshot.jpg', screenshot)

# 输出匹配的位置信息
for position in all_positions:
    print(f"Image found at: {position}")
