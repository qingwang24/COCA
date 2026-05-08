import os


def get_filepaths(file_dir, ext='.pdf'):
    '''
        goal: 提取当前文件夹及其子文件夹下的所有'.ext'文件
        param: file_dir, 需要提取的文件夹路径
        param: ext, 需要提取的文件类型
        output: 文件类型为ext的所有文件路径
    '''
    all_files = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[-1].lower() == ext:
                all_files.append(root+"/"+file)
    return all_files


def get_image_file_list(img_file):
    imgs_lists = []
    if img_file is None or not os.path.exists(img_file):
        raise Exception("not found any img file in {}".format(img_file))

    img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif', 'GIF'}
    if os.path.isfile(img_file) and imghdr.what(img_file) in img_end:
        imgs_lists.append(img_file)
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            file_path = os.path.join(img_file, single_file)
            if os.path.isfile(file_path) and imghdr.what(file_path) in img_end:
                imgs_lists.append(file_path)
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    imgs_lists = sorted(imgs_lists)
    return imgs_lists


import os
import cv2
import math
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def visualize_ocr(image, boxes, txts, font_path='/train21/mmu/permanent/zrzhang6/MultiModelMathAnswer/mv2mmu/simfang.ttf'):
    image = image.convert('RGB')
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))
    
    boxes = np.array(boxes, dtype=np.int32).reshape(-1, 4, 2)
    random.seed(0)
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        color = (random.randint(0, 255), random.randint(0, 255),
            random.randint(0, 255))
        draw_left.polygon([item for item in box.reshape(-1)], fill=color)
        draw_right.polygon(
            [
                box[0][0], box[0][1], box[1][0], box[1][1],
                box[2][0], box[2][1], box[3][0], box[3][1]
            ],
            outline=color
        )
        box_height = math.sqrt((box[0][0] - box[3][0])**2 + (box[0][1] - box[3][1])**2)
        box_width = math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][1])**2)
        if box_height > 2*box_width:
            font_size = max(int(box_width * 0.9), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            cur_y = box[0][1]
            for c in txt:
                char_size = font.getsize(c)
                draw_right.text(
                    (box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font
                )
                cur_y += char_size[1]
        else:
            font_size = max(int(box_height * 0.8), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            draw_right.text(
                [box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font
            )
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w*2, h), (255,255,255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w*2, h))
    return np.array(img_show)