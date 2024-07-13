import time
import cv2
import numpy as np
from collections import Counter, defaultdict
from PIL import ImageGrab
from sklearn.cluster import KMeans

def capture_screen():
    print("等待3秒后开始截屏...")
    time.sleep(3)  # 延迟3秒，用于切换窗口
    screenshot = ImageGrab.grab()
    screenshot.save("color_distribution_regionGrowing/screenshot1.png")
    print("截屏完成，保存为 screenshot1.png")
    return screenshot

def get_top_colors_with_coordinates(image_path, top_n=10):
    start_time = time.time()
    print(f"读取图像 {image_path} ...")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("图像未找到或路径不正确")

    # 将图像从BGR转换为RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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

    end_time = time.time()
    print(f"找到了前 {top_n} 种颜色，耗时 {end_time - start_time:.2f} 秒")
    return top_colors, color_coords

def apply_color_clustering(image_path, n_clusters=3):
    start_time = time.time()
    print(f"对图像 {image_path} 进行颜色聚类...")

    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("图像未找到或路径不正确")

    # 将图像从BGR转换为RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rows, cols, _ = img.shape

    # 将图像展平为二维数组
    flat_img = img_rgb.reshape(-1, 3)

    # 应用K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(flat_img)
    clustered_img = kmeans.cluster_centers_[kmeans.labels_].reshape(rows, cols, 3).astype(np.uint8)

    end_time = time.time()
    print(f"颜色聚类完成，耗时 {end_time - start_time:.2f} 秒")

    return clustered_img

def find_all_connected_components(img, color):
    start_time = time.time()
    height, width = img.shape[:2]
    components = []
    processed_pixels = set()

    print(f"查找颜色 {color} 的连通域...")
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[np.all(img == color, axis=-1)] = 1

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    for label in range(1, num_labels):
        component_mask = (labels == label).astype(np.uint8) * 255
        pixels = np.column_stack(np.where(labels == label))

        # 检查像素是否已经处理过
        pixel_tuple = tuple(map(tuple, pixels))
        if any(p in processed_pixels for p in pixel_tuple):
            continue

        components.append((stats[label, cv2.CC_STAT_AREA], component_mask, stats[label], pixels))
        processed_pixels.update(pixel_tuple)

    end_time = time.time()
    print(f"找到了 {len(components)} 个连通域，耗时 {end_time - start_time:.2f} 秒")
    return components

def draw_bounding_box(img, stats, color=(0, 255, 0)):
    x, y, w, h = stats[:4]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    print(f"绘制边框：x={x}, y={y}, w={w}, h={h}")

def are_adjacent(pixels1, pixels2):
    adjacent_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    pixels1_set = set(map(tuple, pixels1))
    pixels2_set = set(map(tuple, pixels2))
    for (y1, x1) in pixels1_set:
        for dy, dx in adjacent_offsets:
            if (y1 + dy, x1 + dx) in pixels2_set:
                return True
    return False

def is_similar_color(color1, color2, tolerance=10):
    return np.linalg.norm(np.array(color1) - np.array(color2)) <= tolerance

def merge_components(components):
    start_time = time.time()
    print("开始合并相邻的连通域...")
    merged_components = []
    used = set()

    while components:
        component1 = components.pop(0)
        if id(component1) in used:
            continue
        merged = [component1]
        used.add(id(component1))
        merge_count = 0

        i = 0
        while i < len(components):
            component2 = components[i]
            if id(component2) in used:
                i += 1
                continue
            if is_similar_color(component1[2][4:7], component2[2][4:7]) and are_adjacent(component1[3], component2[3]):
                merged.append(component2)
                used.add(id(component2))
                components.pop(i)
                merge_count += 1
            else:
                i += 1

        if len(merged) > 1:
            merged_area = sum(comp[0] for comp in merged)
            merged_mask = np.zeros_like(merged[0][1])
            merged_pixels = []
            for comp in merged:
                print(f"正在合并的掩码大小: {merged_mask.shape}, {comp[1].shape}, 类型: {merged_mask.dtype}, {comp[1].dtype}")
                if merged_mask.shape == comp[1].shape and merged_mask.dtype == comp[1].dtype:
                    merged_mask = cv2.bitwise_or(merged_mask, comp[1])
                else:
                    print(f"掩码大小或类型不匹配: {merged_mask.shape}, {comp[1].shape}, {merged_mask.dtype}, {comp[1].dtype}")
                    continue
                merged_pixels.extend(comp[3])
            merged_stats = merged[0][2].copy()
            merged_stats[cv2.CC_STAT_AREA] = merged_area
            x_coords, y_coords = zip(*merged_pixels)
            merged_stats[cv2.CC_STAT_LEFT] = min(x_coords)
            merged_stats[cv2.CC_STAT_TOP] = min(y_coords)
            merged_stats[cv2.CC_STAT_WIDTH] = max(x_coords) - min(x_coords) + 1
            merged_stats[cv2.CC_STAT_HEIGHT] = max(y_coords) - min(y_coords) + 1
            merged_components.append((merged_area, merged_mask, merged_stats, np.array(merged_pixels)))
            print(f"合并了 {merge_count} 个连通域，总面积为 {merged_area}")
        else:
            merged_components.append(component1)

    end_time = time.time()
    print(f"合并后剩余 {len(merged_components)} 个连通域，耗时 {end_time - start_time:.2f} 秒")
    return merged_components

def keep_largest_components(image_path, top_colors_n=5, largest_components_n=10, background_color=(0, 0, 0)):
    overall_start_time = time.time()

    # 对图像进行颜色聚类
    clustered_img = apply_color_clustering(image_path, n_clusters=3)

    # 将聚类后的图像保存以便检查
    cv2.imwrite('color_distribution_regionGrowing/clustered_image.png', cv2.cvtColor(clustered_img, cv2.COLOR_RGB2BGR))
    print("保存聚类后的图像到 clustered_image.png")

    # 获取聚类后的图像中的前 N 种颜色及其坐标
    top_colors, color_coords = get_top_colors_with_coordinates('color_distribution_regionGrowing/clustered_image.png', top_colors_n)
    img_rgb = clustered_img
    all_components = []

    for color, _ in top_colors:
        components = find_all_connected_components(img_rgb, color)
        all_components.extend(components)

    # 全图范围内合并连通域
    merged_components = merge_components(all_components)
    merged_components.sort(key=lambda x: x[0], reverse=True)
    largest_components = merged_components[:largest_components_n]

    print(f"保留最大的 {largest_components_n} 个连通域")
    combined_mask = np.zeros(img_rgb.shape[:2], np.uint8)
    result_img = np.zeros_like(img_rgb)
    for i in range(3):
        result_img[:, :, i] = background_color[i]

    for _, component_mask, stats, _ in largest_components:
        print(f"处理连通域，面积为 {stats[cv2.CC_STAT_AREA]}")
        temp_mask = np.zeros_like(combined_mask)
        temp_mask[np.where(component_mask == 255)] = 255
        combined_mask = cv2.bitwise_or(combined_mask, temp_mask)
        draw_bounding_box(img_rgb, stats)

    result_img[combined_mask == 255] = img_rgb[combined_mask == 255]

    save_path = 'color_distribution_regionGrowing/result_image1.png'
    cv2.imwrite(save_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    print(f"保存结果图像到 {save_path}")

    save_path_with_boxes = 'color_distribution_regionGrowing/result_image_with_boxes1.png'
    cv2.imwrite(save_path_with_boxes, img_rgb)
    print(f"保存带边框的结果图像到 {save_path_with_boxes}")

    overall_end_time = time.time()
    print(f"总体耗时 {overall_end_time - overall_start_time:.2f} 秒")

capture_screen()

image_path = 'color_distribution_regionGrowing/screenshot1.png'

keep_largest_components(image_path)
