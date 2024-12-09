from PIL import Image, ImageDraw, ImageFont
import os,copy,modelscope,random,shutil

from icon_localization import det
from merge_strategy import merge_all_icon_boxes

def cmyk_to_rgb(c, m, y, k):
    r = 255 * (1.0 - c / 255) * (1.0 - k / 255)
    g = 255 * (1.0 - m / 255) * (1.0 - k / 255)
    b = 255 * (1.0 - y / 255) * (1.0 - k / 255)
    return int(r), int(g), int(b)

def draw_coordinates_boxes_on_image(image_path, coordinates, output_image_path, font_path):
    image = Image.open(image_path)
    width, height = image.size
    draw = ImageDraw.Draw(image)
    total_boxes = len(coordinates)
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in
              range(total_boxes)]
    # padding = height * 0.005

    for i, coord in enumerate(coordinates):
        # color = generate_color_from_hsv_pil(i, total_boxes)
        c, m, y, k = colors[i]
        color = cmyk_to_rgb(c, m, y, k)
        # print(color)

        # coord[0] = coord[0] - padding
        # coord[1] = coord[1] - padding
        # coord[2] = coord[2] + padding
        # coord[3] = coord[3] + padding

        draw.rectangle(coord, outline=color, width=int(height * 0.0025))

        font = ImageFont.truetype(font_path, int(height * 0.012))
        text_x = coord[0] + int(height * 0.0025)
        text_y = max(0, coord[1] - int(height * 0.013))
        draw.text((text_x, text_y), str(i + 1), fill=color, font=font)
    # image.show()
    image = image.convert('RGB')
    image.save(output_image_path)

def split_image_into_4(input_image_prefix: str,input_image: str, output_image_prefix: str) -> None:
    img = Image.open(input_image_prefix+'/'+input_image)
    width, height = img.size

    sub_width = width // 2
    sub_height = height // 2

    # crop into 4 sub images
    quadrants = [
        (0, 0, sub_width, sub_height),
        (sub_width, 0, width, sub_height),
        (0, sub_height, sub_width, height),
        (sub_width, sub_height, width, height)
    ]

    for i, box in enumerate(quadrants):
        sub_img = img.crop(box)
        sub_img.save(f"{output_image_prefix}_part_{i+1}.png")

def crop(file, file_name, box, i):
    image = Image.open(file)
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    if x1 >= x2-10 or y1 >= y2-10:
        return
    cropped_image = image.crop((x1, y1, x2, y2))
    cropped_image.save(f"./My_agent/temp/{file_name}/icon_{i}.png")

def get_perception_infos(
    input_file_prefix: str,
    file:str, 
    output_file_prefix:str,
    output_file:str, 
    font_path:str,
    groundingdino_model:modelscope.Pipeline
    ) -> tuple[dict, int, int]:
    """
    # TODO: Write Summary here.
    
    Args:
        file (str): Input image.
        output_file (str): Destination of the output_file.
        font_path (str): The font for ocr.
        ocr_detection (modelscope.Pipeline): The detection model. TODO: Complete description.
        ocr_recognition (modelscope.Pipeline): The recognition model. TODO: Complete description.

    Returns:
        perception infos (dict): TODO: Complete the descrition.
        total width, total height(int,int): the size of the image.
    """
    
    total_width, total_height = Image.open(input_file_prefix + '/' + file).size

    # Partition Image into 4 parts
    # TODO: remove split.
    fileName, _ = os.path.splitext(file)
    temp_file_prefix = './My_agent/temp/'
    split_image_into_4(input_file_prefix, file, temp_file_prefix+f'{fileName}')
    img_list = [temp_file_prefix+f'{fileName}_part_1.png', temp_file_prefix+f'{fileName}_part_2.png',
                temp_file_prefix+f'{fileName}_part_3.png', temp_file_prefix+f'{fileName}_part_4.png']
    img_x_list = [0, total_width/2, 0, total_width/2]
    img_y_list = [0, 0, total_height/2, total_height/2]
    coordinates = []
    texts = []
    padding = total_height * 0.0025  # 10

    # 使用groundingDino 识别图标。然后绘制图标框。
    coordinates = []
    for i, img in enumerate(img_list):
        width, height = Image.open(img).size
        sub_coordinates = det(img, "icon", groundingdino_model)
        for coordinate in sub_coordinates:
            coordinate[0] = int(max(0, img_x_list[i] + coordinate[0] - padding))
            coordinate[2] = int(min(total_width, img_x_list[i] + coordinate[2] + padding))
            coordinate[1] = int(max(0, img_y_list[i] + coordinate[1] - padding))
            coordinate[3] = int(min(total_height, img_y_list[i] + coordinate[3] + padding))

        sub_coordinates = merge_all_icon_boxes(sub_coordinates)
        coordinates.extend(sub_coordinates)
    merged_icon_coordinates = merge_all_icon_boxes(coordinates)
    draw_coordinates_boxes_on_image(input_file_prefix+'/'+file, copy.deepcopy(merged_icon_coordinates), output_file_prefix+'/'+output_file, font_path)
    
    mark_number = 0
    perception_infos = []
    for i in range(len(merged_icon_coordinates)):
        mark_number += 1
        perception_info = {"text": "mark number: " + str(mark_number) + " icon", "coordinates": merged_icon_coordinates[i]}
        perception_infos.append(perception_info)
    
    # if args.icon_caption == 1:
    image_box = []
    image_id = []
    for i in range(len(perception_infos)):
        # if perception_infos[i]['text'] == 'icon':
        if 'icon' in perception_infos[i]['text']: # TODO
            image_box.append(perception_infos[i]['coordinates'])
            image_id.append(i)

    # 此处裁剪图片并且存在对应路径中。
    file_name, _ = os.path.splitext(file)
    if os.path.exists(f"./My_agent/temp/{file_name}"):
        shutil.rmtree(f"./My_agent/temp/{file_name}")
    os.mkdir(f"./My_agent/temp/{file_name}")
    for i in range(len(image_box)):
        crop(input_file_prefix+'/'+file, file_name, image_box[i], image_id[i])

    # if args.location_info == 'center':
    for i in range(len(perception_infos)):
        perception_infos[i]['coordinates'] = [int((perception_infos[i]['coordinates'][0]+perception_infos[i]['coordinates'][2])/2), int((perception_infos[i]['coordinates'][1]+perception_infos[i]['coordinates'][3])/2)]
    # elif args.location_info == 'icon_center':
    #     for i in range(len(perception_infos)):
    #         if 'icon' in perception_infos[i]['text']:
    #             perception_infos[i]['coordinates'] = [
    #                 int((perception_infos[i]['coordinates'][0] + perception_infos[i]['coordinates'][2]) / 2),
    #                 int((perception_infos[i]['coordinates'][1] + perception_infos[i]['coordinates'][3]) / 2)]

    return perception_infos, total_width, total_height