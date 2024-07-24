import cv2
import numpy as np

from sklearn.linear_model import RANSACRegressor
from PIL import ImageGrab

# 截取屏幕图像
bbox = (100, 100, 800, 600)  # 根据需要调整截屏区域
screen = ImageGrab.grab(bbox)
screen_np = np.array(screen)

# 转换颜色格式以便使用cv2
screen_np = cv2.cvtColor(screen_np, cv2.COLOR_RGB2BGR)

# 保存截屏图像
cv2.imwrite('lines_identification/screenshot.jpg', screen_np)

# 使用截屏图像进行处理
gray = cv2.cvtColor(screen_np, cv2.COLOR_BGR2GRAY)

# 边缘检测
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# 提取边缘点
points = np.column_stack(np.where(edges > 0))

# 使用RANSAC拟合直线
ransac = RANSACRegressor()
ransac.fit(points[:, 1].reshape(-1, 1), points[:, 0])

# 获取直线参数
slope = ransac.estimator_.coef_[0]
intercept = ransac.estimator_.intercept_

# 绘制直线
x = np.array([0, screen_np.shape[1]])
y = slope * x + intercept
cv2.line(screen_np, (x[0], int(y[0])), (x[1], int(y[1])), (0, 0, 255), 2)



# 保存处理后的截屏图像
cv2.imwrite('lines_identification/processed_screenshot.jpg', screen_np)
