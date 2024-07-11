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


def capture_screen():
    # Add a delay to switch windows
    time.sleep(3)  # Delay for 3 seconds

    # Capture the entire screen
    screenshot = ImageGrab.grab()
    screenshot.save("screenshot0.png")
    return screenshot


def convert_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('ascii')


def get_ocr_results(screenshot):
    image_base64 = convert_image_to_base64(screenshot)

    """Get OCR results from Tencent Cloud OCR for the given base64 image."""
    try:
        # Load credentials from environment variables
        SecretId = os.getenv("SECRET_ID")
        SecretKey = os.getenv("SECRET_KEY")

        cred = credential.Credential(SecretId, SecretKey)

        # Setup HTTP and client profile
        httpProfile = HttpProfile()
        httpProfile.endpoint = "ocr.tencentcloudapi.com"
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile

        # Create OCR client
        client = ocr_client.OcrClient(cred, "ap-shanghai", clientProfile)

        # Prepare request
        req = models.GeneralBasicOCRRequest()
        params = {"ImageBase64": image_base64}
        req.from_json_string(json.dumps(params))

        # Execute request and return results
        resp = client.GeneralBasicOCR(req)
        return json.loads(resp.to_json_string())
    except TencentCloudSDKException as err:
        return str(err)


def add_ocr_results_to_image(image, ocr_results):
    """Add OCR results to the image."""
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    if 'TextDetections' in ocr_results:
        for item in ocr_results['TextDetections']:
            text = item['DetectedText']
            item_polygon = item.get('ItemPolygon', {})
            x = item_polygon.get('X', 0)
            y = item_polygon.get('Y', 0)
            width = item_polygon.get('Width', 0)
            height = item_polygon.get('Height', 0)

            # Draw bounding box
            draw.rectangle([x, y, x + width, y + height], outline="blue", width=2)
            # # Draw text within the bounding box
            # draw.text((x, y), text, fill="red", font=font)

    return image


def simple_difference_edge_detection(image):
    # Convert the image to a numpy array of RGB
    image_np = np.array(image)

    # Calculate the differences with right and below neighbors
    right_diff = np.abs(image_np[:, :-1] - image_np[:, 1:])
    down_diff = np.abs(image_np[:-1, :] - image_np[1:, :])

    # Sum the differences across the color channels
    right_diff_sum = np.sum(right_diff, axis=2)
    down_diff_sum = np.sum(down_diff, axis=2)

    # Initialize the edge map with zeros
    edge_map = np.zeros(image_np.shape[:2])

    # Set edges where the difference exceeds the threshold
    edge_map[:, :-1][right_diff_sum > 1] = 1
    edge_map[:-1, :][down_diff_sum > 1] = 1

    # Get edge positions
    edge_positions = np.argwhere(edge_map == 1)

    return edge_map, edge_positions.tolist()


def is_valid(y, x, edge_set, visited):
    # Check that the coordinates (y, x) are in edge_set and have not been accessed
    return (y, x) in edge_set and not visited[y, x]


def bfs(start_y, start_x, edge_set, visited):
    queue = [(start_y, start_x)]
    component_pixels = []
    visited[start_y, start_x] = True

    while queue:
        cy, cx = queue.pop(0)
        component_pixels.append((cy, cx))

        # Check the top, bottom, and left four adjacent pixels
        for ny, nx in [(cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)]:
            if is_valid(ny, nx, edge_set, visited):
                visited[ny, nx] = True
                queue.append((ny, nx))

    return component_pixels


def get_boundary_coordinates(min_y, min_x, max_y, max_x):
    boundary_coords = []

    # Top and bottom edges
    for x in range(min_x, max_x + 1):
        boundary_coords.append((min_y, x))
        boundary_coords.append((max_y, x))

    # Left and right edges
    for y in range(min_y + 1, max_y):
        boundary_coords.append((y, min_x))
        boundary_coords.append((y, max_x))

    return boundary_coords


def is_straight_line(segment_pixels, direction='horizontal'):
    if len(segment_pixels) < 15:
        return False

        # Extract coordinates
    y_coords = np.array([pos[0] for pos in segment_pixels])
    x_coords = np.array([pos[1] for pos in segment_pixels])

    if direction == 'horizontal':
        reg = LinearRegression().fit(y_coords.reshape(-1, 1), x_coords)
        residuals = np.abs(reg.predict(y_coords.reshape(-1, 1)) - x_coords)
    elif direction == 'vertical':
        reg = LinearRegression().fit(x_coords.reshape(-1, 1), y_coords)
        residuals = np.abs(reg.predict(x_coords.reshape(-1, 1)) - y_coords)
    else:
        raise ValueError("direction should be 'horizontal' or 'vertical'")

    residual_threshold = 1
    if np.mean(residuals) >= residual_threshold:
        return False

    # min_y, max_y = min(y_coords)[0], max(y_coords)[0]
    # min_x, max_x = min(x_coords), max(x_coords)
    # length = np.sqrt((max_y - min_y) ** 2 + (max_x - min_x) ** 2)
    #
    # length_threshold = 7
    # if length < length_threshold:
    #     return False

    return True


def split_into_segments(component_pixels, window_size=15, step_size=5):
    n = len(component_pixels)
    segments = [
        component_pixels[i:i + window_size]
        for i in range(0, n - window_size + 1, step_size)
    ]
    if n % step_size != 0:
        segments.append(component_pixels[-window_size:])
    return segments


def get_image_dimensions(positions):
    if not positions:
        return None, None, np.zeros((1, 1), dtype=bool), set()

    # 确保 positions 中的每个位置是元组
    positions = [tuple(pos) for pos in positions]

    max_y = max(y for y, x in positions) + 1
    max_x = max(x for y, x in positions) + 1

    visited = np.zeros((max_y, max_x), dtype=bool)
    position_set = set(positions)

    return max_y, max_x, visited, position_set


def get_image_dimensions2(positions):
    if not positions:
        return None, None, np.zeros((1, 1), dtype=bool), set()

    # 确保 positions 是一个集合
    positions = set(positions)

    max_y = max(y for y, x in positions) + 1
    max_x = max(x for y, x in positions) + 1

    visited = np.zeros((max_y, max_x), dtype=bool)
    position_set = positions

    return max_y, max_x, visited, position_set


def classify_edge_positions(edge_positions, edge_map):
    total_pixels = edge_map.size
    total_height, total_width = edge_map.shape

    # text_icon_positions = []
    straight_line_positions = set()
    irregular_large_positions = []

    # 创建变量 edge_positions_image
    edge_positions_image = [(y, x) for y, x in edge_positions if
                            (40 / 2560) * total_width < x < (2460 / 2560) * total_width and (135 / 1440) * total_height < y < (1335 / 1440) * total_height]

    edge_positions_line = set((y, x) for y, x in edge_positions if
                              y < (1370 / 1440) * total_height)

    max_y, max_x, visited, edge_set = get_image_dimensions(edge_positions_image)
    if max_y is None:
        return []

    for (y, x) in edge_positions_image:
        if not visited[y, x]:
            component_pixels = bfs(y, x, edge_set, visited)
            if component_pixels:

                component_pixels.sort()  # 对连通区域的像素列表进行排序
                y_coords = [pos[0] for pos in component_pixels]
                x_coords = [pos[1] for pos in component_pixels]

                # Geometric properties
                min_y, max_y = min(y_coords), max(y_coords)
                min_x, max_x = min(x_coords), max(x_coords)
                width = max_x - min_x + 1
                height = max_y - min_y + 1
                area = len(component_pixels)
                aspect_ratio = width / height
                area_threshold = total_pixels / 100

                # Calculate bounding box area
                bounding_box_area = width * height

                if area / bounding_box_area >= 0.3 and area >= area_threshold:  # Adjust the threshold to identify
                    # large irregular shapes
                    # If the component is Large and irregular in position

                    if 0.1 <= aspect_ratio <= 10:
                        # 从 edge_positions_line 中删除 component_pixels 中的坐标
                        edge_positions_line -= set(component_pixels)
                        irregular_large_positions.extend(component_pixels)

                # if (0.3 <= aspect_ratio <= 20) and (10 <= area <= 5000):
                #     if area / bounding_box_area >= 0.25:
                #         text_icon_positions.extend(component_pixels)

                # else:
                #     if 0 < area / bounding_box_area <= 0.3 and area >= 2000:
                #         segments = split_into_segments(component_pixels)
                #         for segment in segments:
                #             if is_straight_line(segment, direction='horizontal') or is_straight_line(segment,
                #                                                                                      direction='vertical'):
                #                 straight_line_positions.update(segment)

    max_y, max_x, visited, edge_set = get_image_dimensions(edge_positions_line)
    if max_y is None:
        return []

    for (y, x) in edge_positions_line:
        if not visited[y, x]:
            component_pixels = bfs(y, x, edge_set, visited)
            if component_pixels:
                component_pixels.sort()  # 对连通区域的像素列表进行排序
                y_coords = [pos[0] for pos in component_pixels]
                x_coords = [pos[1] for pos in component_pixels]

                # Geometric properties
                min_y, max_y = min(y_coords), max(y_coords)
                min_x, max_x = min(x_coords), max(x_coords)
                width = max_x - min_x + 1
                height = max_y - min_y + 1
                area = len(component_pixels)
                aspect_ratio = width / height
                area_threshold = total_pixels / 55

                # Calculate bounding box area
                bounding_box_area = width * height

                if 0 < area / bounding_box_area <= 0.3 and area >= 2000:
                    segments = split_into_segments(component_pixels)
                    for segment in segments:
                        if is_straight_line(segment, direction='horizontal') or is_straight_line(segment,
                                                                                                 direction='vertical'):
                            straight_line_positions.update(segment)

    return list(straight_line_positions), irregular_large_positions


def extract_text_and_icons(input_img, text_icon_positions):
    # Converts the input image to a numpy array
    img_np = np.array(input_img)

    # 将text_icon_positions转换为NumPy数组
    positions = np.array(text_icon_positions)

    # 分别提取行和列索引
    y_coords, x_coords = positions[:, 0], positions[:, 1]

    # 利用NumPy的向量化操作直接提取颜色信息
    color_info = img_np[y_coords, x_coords]

    # 使用zip将位置和颜色信息结合
    new_text_icon_info = list(zip(text_icon_positions, map(tuple, color_info)))

    return new_text_icon_info


def add_label_to_straight_lines_info(straight_line_positions):
    # 将straight_line_positions转换为NumPy数组
    positions = np.array(straight_line_positions)

    # 创建颜色信息数组，所有元素均为(0, 255, 0)
    new_color_info = np.tile((0, 255, 0), (positions.shape[0], 1))

    # 使用zip将位置和颜色信息结合
    straight_lines_info = list(zip(map(tuple, positions), map(tuple, new_color_info)))

    return straight_lines_info


def add_label_boxes_info(irregular_large_positions):
    red_boxes_info = []
    max_y, max_x, visited, irregular_large_set = get_image_dimensions(irregular_large_positions)
    if max_y is None:
        return []

    red_boxes = []

    for (y, x) in irregular_large_positions:
        if not visited[y, x]:
            component_pixels = bfs(y, x, irregular_large_set, visited)
            if component_pixels:
                # 将 component_pixels 转换为 NumPy 数组
                component_pixels_np = np.array(component_pixels)

                # 找到边界框的坐标
                min_y, min_x = component_pixels_np.min(axis=0)
                max_y, max_x = component_pixels_np.max(axis=0)

                # Get the boundary coordinates
                boundary_coords = get_boundary_coordinates(min_y, min_x, max_y, max_x)

                # Add the boundary coordinates to the list of red boxes
                red_boxes.extend(boundary_coords)

    new_color_info = (255, 0, 0)  # 红色
    for (y, x) in red_boxes:
        red_boxes_info.append(((y, x), new_color_info))

    return red_boxes_info


def edge_map_to_info(edge_map):
    # 获取图像的高度和宽度
    height, width = edge_map.shape

    # 创建一个空的 NumPy 数组用于存储颜色信息
    edge_info = np.zeros((height, width, 3), dtype=np.uint8)

    # 将边缘像素设置为白色（255, 255, 255）
    edge_info[edge_map == 1] = [255, 255, 255]

    # 将非边缘像素设置为黑色（0, 0, 0），实际上这步可以省略，因为初始值就是黑色
    # edge_info[edge_map == 0] = [0, 0, 0]

    # 获取所有像素的坐标
    y_coords, x_coords = np.indices(edge_map.shape)

    # 将坐标和颜色信息组合成一个列表
    edge_info_list = list(zip(zip(y_coords.ravel(), x_coords.ravel()), edge_info.reshape(-1, 3)))

    return edge_info_list


def merge_info(edge_info, new_text_icon_info, straight_lines_info, red_boxes_info):
    # 将所有像素信息转换为字典，方便查找
    new_text_icon_dict = {pos: color for pos, color in new_text_icon_info}
    straight_lines_dict = {pos: color for pos, color in straight_lines_info}
    red_boxes_dict = {pos: color for pos, color in red_boxes_info}

    # 创建集合用于快速查找
    new_text_icon_set = set(new_text_icon_dict.keys())
    straight_lines_set = set(straight_lines_dict.keys())
    red_boxes_set = set(red_boxes_dict.keys())

    merged_info = []

    for pos, color in edge_info:
        if pos in red_boxes_set:
            new_color = red_boxes_dict[pos]
        elif pos in new_text_icon_set:
            new_color = new_text_icon_dict[pos]
        elif pos in straight_lines_set:
            new_color = straight_lines_dict[pos]
        else:
            new_color = color  # 保持原始颜色

        merged_info.append((pos, new_color))

    return merged_info


def merge_info2(screenshot, straight_lines_info, red_boxes_info):
    # 将所有像素信息转换为字典，方便查找
    straight_lines_dict = {pos: color for pos, color in straight_lines_info}
    red_boxes_dict = {pos: color for pos, color in red_boxes_info}

    # 创建集合用于快速查找
    straight_lines_set = set(straight_lines_dict.keys())
    red_boxes_set = set(red_boxes_dict.keys())

    # 获取图像的像素数据
    pixels = screenshot.load()
    width, height = screenshot.size

    for y in range(height):
        for x in range(width):
            pos = (y, x)
            if pos in red_boxes_set:
                new_color = red_boxes_dict[pos]
            elif pos in straight_lines_set:
                new_color = straight_lines_dict[pos]
            else:
                new_color = pixels[x, y]  # 保持原始颜色

            pixels[x, y] = new_color

    return screenshot


def save_image(data, filename):
    # Use PIL to save the binary image
    img = Image.fromarray(data * 255)  # Convert binary data to a format suitable for saving
    img = img.convert("L")  # Convert to grayscale
    img.save(filename)


def save_visualization_image(merged_info, filename):
    # 从 merged_info 中获取图像的尺寸
    max_x = max(pos[1] for pos, color in merged_info) + 1
    max_y = max(pos[0] for pos, color in merged_info) + 1

    # 创建一个黑色背景的 numpy 数组
    img_array = np.zeros((max_y, max_x, 3), dtype=np.uint8)

    # 提取位置和颜色信息
    positions = np.array([pos for pos, color in merged_info])
    colors = np.array([color for pos, color in merged_info])

    # 利用 NumPy 的向量化操作填充图像的像素
    img_array[positions[:, 0], positions[:, 1]] = colors

    # 将 numpy 数组转换为 PIL 图像
    img = Image.fromarray(img_array)

    # 保存图像
    img.save(filename)


def save_visualization_image2(image, filename):
    """
    Save the modified image to a file.

    :param image: Image object with merged information
    :param filename: Filename to save the image
    """
    # 直接保存传入的 Image 对象
    image.save(filename)


def main():
    # Step 1: Capture the screen
    screenshot = capture_screen()
    print("Screenshot saved as 'screenshot0.png'")

    # Step 2: Convert screenshot to edge map and get the position of edged elements
    edge_map, edge_positions = simple_difference_edge_detection(screenshot)

    # Step 3: Save the edge map
    save_image(edge_map, 'edge_map0.png')
    print("Edge map saved as 'edge_map0.png'")

    # Step 4: Extract text and save as
    ocr_results = get_ocr_results(screenshot)
    print(ocr_results)
    image_with_ocr = add_ocr_results_to_image(screenshot, ocr_results)
    output_path = "screenshot_with_ocr.png"
    image_with_ocr.save(output_path)
    print(f"OCR results added to image and saved as {output_path}")

    # Step 5: classify the type of edged elements
    straight_line_positions, irregular_large_positions = classify_edge_positions(edge_positions, edge_map)

    # # Step : extract text and icons
    # new_text_icon_info = extract_text_and_icons(screenshot, text_icon_positions)

    # Step 7: add label to straight lines
    straight_lines_info = add_label_to_straight_lines_info(straight_line_positions)

    # Step 8: label natural pics
    red_boxes_info = add_label_boxes_info(irregular_large_positions)

    # Step : get edge_map info
    edge_info = edge_map_to_info(edge_map)

    # Step 9: merge 4 info and get pic
    merged_info = merge_info2(screenshot, straight_lines_info, red_boxes_info)

    # Step 10: Save merged_info as a visualization image
    save_visualization_image2(merged_info, "labeled_screenshot0.png")
    print("Merged info saved as 'labeled_screenshot0.png'")


if __name__ == "__main__":
    main()
