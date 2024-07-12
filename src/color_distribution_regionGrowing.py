import time
import cv2
import numpy as np
from collections import Counter, defaultdict
from PIL import ImageGrab

def capture_screen():
    # Add a delay to switch windows
    time.sleep(3)  # Delay for 3 seconds

    # Capture the entire screen
    screenshot = ImageGrab.grab()
    screenshot.save("color_distribution_regionGrowing/screenshot1.png")
    return screenshot

def get_top_colors_with_coordinates(image_path, top_n=100):
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or path is incorrect")

    # 将图像从BGR转换为RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 获取图像的形状
    rows, cols, _ = img.shape

    # 将图像展平为二维数组
    flat_img = img_rgb.reshape(-1, 3)

    # 统计每种颜色的像素数量
    color_counts = Counter(map(tuple, flat_img))

    # 找出前top_n种颜色
    top_colors = color_counts.most_common(top_n)

    # 存储颜色及其坐标
    color_coords = defaultdict(list)

    # 提取每种颜色的坐标
    for r in range(rows):
        for c in range(cols):
            pixel_color = tuple(img_rgb[r, c])
            if pixel_color in dict(top_colors):
                color_coords[pixel_color].append((r, c))

    return top_colors, color_coords

def find_all_connected_components(img, color, block_size=(512, 512)):
    height, width = img.shape[:2]
    components = []

    for y in range(0, height, block_size[1]):
        for x in range(0, width, block_size[0]):
            block = img[y:y + block_size[1], x:x + block_size[0]]
            mask = np.zeros(block.shape[:2], np.uint8)
            mask[np.all(block == color, axis=-1)] = 1

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

            for label in range(1, num_labels):
                component_mask = (labels == label).astype(np.uint8) * 255
                stats[label, 0] += x  # Adjust x coordinate
                stats[label, 1] += y  # Adjust y coordinate
                components.append((stats[label, cv2.CC_STAT_AREA], component_mask, stats[label], (x, y)))

    return components

def draw_bounding_box(img, stats, color=(0, 255, 0)):
    x, y, w, h = stats[:4]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

def keep_largest_components(image_path, top_colors_n=100, largest_components_n=50, background_color=(0, 0, 0)):
    top_colors, color_coords = get_top_colors_with_coordinates(image_path, top_colors_n)

    # 读取图像
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    all_components = []

    for color, _ in top_colors:
        components = find_all_connected_components(img_rgb, color)
        all_components.extend(components)

    # 按连通域大小排序并取前largest_components_n个
    all_components.sort(key=lambda x: x[0], reverse=True)
    largest_components = all_components[:largest_components_n]

    # 创建一个掩码来保存所有最大的连通域
    combined_mask = np.zeros(img.shape[:2], np.uint8)
    result_img = np.zeros_like(img_rgb)
    for i in range(3):
        result_img[:, :, i] = background_color[i]

    for _, component_mask, stats, (x, y) in largest_components:
        temp_mask = np.zeros_like(combined_mask)
        temp_mask[y:y + component_mask.shape[0], x:x + component_mask.shape[1]] = component_mask
        combined_mask = cv2.bitwise_or(combined_mask, temp_mask)
        draw_bounding_box(img, stats)

    result_img[combined_mask == 255] = img_rgb[combined_mask == 255]

    # 保存结果图像和带边框的图像
    save_path = 'color_distribution_regionGrowing/result_image1.png'
    cv2.imwrite(save_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    print(f'Saved result image with largest components to {save_path}')

    save_path_with_boxes = 'color_distribution_regionGrowing/result_image_with_boxes1.png'
    cv2.imwrite(save_path_with_boxes, img)
    print(f'Saved result image with bounding boxes to {save_path_with_boxes}')

capture_screen()

image_path = 'color_distribution_regionGrowing/screenshot1.png'

keep_largest_components(image_path)
