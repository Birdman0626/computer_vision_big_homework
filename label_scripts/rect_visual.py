import cv2
import numpy as np
import xml.etree.ElementTree as ET
import os
import tqdm

def xml2rect(file_name: str): 
    info_list = []
    xml_tree = ET.parse(file_name)
    root = xml_tree.getroot()
    for single_object in root.findall('object'):
        label = single_object.find("name").text
        x_min = int(single_object.find("bndbox")[0].text)
        y_min = int(single_object.find("bndbox")[1].text)
        x_max = int(single_object.find("bndbox")[2].text)
        y_max = int(single_object.find("bndbox")[3].text)
        info_list.append((label, x_min, y_min, x_max, y_max))
    return info_list

def draw_rect(file_name):
    image = cv2.imread(file_name + ".png")
    info_list = xml2rect(file_name + ".xml")
    for label, x_min, y_min, x_max, y_max in info_list:
        if label not in ['root', 'level_0', 'level_1', 'level_2']:    
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        else:
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        cv2.putText(image, label, (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return image
    

for dirpath, subdirname, subfilename in os.walk("dataset_fixed_capital_scrollable"):
    if subdirname == []:
        for subfile in subfilename:
            if subfile[-4:] == ".png":
                file_name = dirpath + "\\" + subfile[:-4]
                image_rect = draw_rect(file_name)
                save_file_name = "dataset_fixed_capital_scrollable_visual\\" + dirpath[(dirpath.find("\\")+1):] +"\\" + subfile
                print(file_name, save_file_name)
                cv2.imwrite(save_file_name, image_rect)
                
print("finished")