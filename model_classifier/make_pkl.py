import xml.etree.ElementTree as ET
import os,shutil,pickle
import numpy as np
from PIL import Image
import cv2

def make_dict(path:str) -> dict:
    """
    To get the dataset into the dictionary.
    Args:
        path (str): The targeted xml document path.

    Returns:
        dict: The dictionary contains the name and the icons.
    
    Details: The return dictionary has the following format:
    image_info = Dict
    {
        'image_path' (str): The images path.
        'image_size' (list[int, int, 3]): The list of images RGB.
        'icon_num' (int): The icon number contain in image.
        'name' (list[str]): The type of these icons.
        'range' (list[list[xmin, ymin, xmax, ymax]]): The icon spreading box.
    }
    """
    root = ET.parse(path)

    # Get the image_size from xml. Remember in int.
    image_size = []
    for size in root.findall('size'):
        for child in size:
            image_size.append(int(child.text))
    
    # Read each icon
    icon_num = len(root.findall('object'))
    image_path, _ = os.path.splitext(path)
    image_path += '.png'
    image_info = {
        'image_path': image_path,
        'image_size': image_size,
        'icon_num': icon_num,
        'name':[],
        'range':[]
    }
    
    # find icons and ranges.
    for icon in root.findall('object'):
        image_info['name'].append(icon.find('name').text)
        # The coordinate should be: xmin, ymin, xmax, ymax.
        coord = []
        for data in icon.find('bndbox'):
            coord.append(int(data.text))
        image_info['range'].append(coord)
    
    return image_info
    
    
if __name__ == '__main__':
    """
        This program aims to save a dictionary into binary by the format as follows:
        ### images_info = List[Dict]
        
    """
    
    # For Mac, the ".DS_Store" is the directory who manages the structure. 
    # So we should skip it.
    images_info = []
    for directory in os.listdir('./dataset_fixed_capital_scrollable'):
        # inside the directory, list all files. Read the base name and skip the same file.
        prefix = f'./dataset_fixed_capital_scrollable/{directory}'
        fileBaseList=list()
        for file in os.listdir(prefix):
            fileBaseName, _ = os.path.splitext(file)
            if fileBaseList.count(fileBaseName) > 0: continue
            image_info = make_dict(f'{prefix}/{fileBaseName}.xml')
            images_info.append(image_info)
            fileBaseList.append(fileBaseName)
    
    if os.path.exists('./dataset.pkl'):
        os.remove('./dataset.pkl')
    fileHandler = open('./dataset.pkl','wb')
    pickle.dump(images_info, fileHandler)
    fileHandler.close()
    
    
    fileHandler = open('./dataset.pkl','rb')
    dataset = pickle.load(fileHandler)
    fileHandler.close()

    root_image_list = []
    icon_image_list = []
    icon_label_list = []
    icon_pos_list = []

    LABEL_LIST = ['clickable', 'disabled', 'selectable','scrollable']

    def get_noramlized_pos(root_pos, pos):
        root_xmin, root_ymin, root_xmax, root_ymax = root_pos
        pos_xmin, pos_ymin, pos_xmax, pos_ymax = pos
        def min_max_norm(min_, max_, value):
            if value < min_ : value = min_
            if value > max_ : value = max_
            assert min_ <= value <= max_, f"{min_} {max_} {value}"
            return (value-min_) / (max_-min_)
        return np.array([
                min_max_norm(root_xmin, root_xmax, pos_xmin),
                min_max_norm(root_ymin, root_ymax, pos_ymin),
                min_max_norm(root_xmin, root_xmax, pos_xmax),
                min_max_norm(root_ymin, root_ymax, pos_ymax)
                ])



    for i in dataset:
        base_pic = np.array(cv2.imread(i["image_path"]))
        root_pos = i["range"][i["name"].index("root")]
        root_xmin, root_ymin, root_xmax, root_ymax = root_pos
        root_pic = base_pic[root_ymin:(root_ymax+1), root_xmin:(root_xmax+1)]
        # cv2.imshow("",root_pic)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows
        for j in range(i["icon_num"]):
            if i["name"][j] in ['root', 'level_0', 'level_1', 'level_2']:
                continue
            icon_label = LABEL_LIST.index(i["name"][j])
            #print(icon_label)
            icon_pos = i["range"][j]
            #print(icon_pos)
            icon_xmin, icon_ymin, icon_xmax, icon_ymax = icon_pos
            #print(get_noramlized_pos(root_pos, icon_pos))
            icon_pos_list.append(get_noramlized_pos(root_pos, icon_pos))
            icon_pic = base_pic[icon_ymin:(icon_ymax+1), icon_xmin:(icon_xmax+1)]
            # cv2.imshow("", icon_pic)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows
            # if icon_pic.shape[0] == 0:
            #     print(icon_pic.shape)
            #     print(i["image_path"])
            #     exit()
            root_image_list.append(root_pic)
            icon_image_list.append(icon_pic)
            icon_label_list.append(icon_label)
            #exit()
            
    assert len(root_image_list) == len(icon_image_list) == len(icon_label_list) == len(icon_pos_list)
    print(len(root_image_list))

    if os.path.exists('./dataset_4lists.pkl'):
            os.remove('./dataset_4lists.pkl')
    fileHandler = open('./dataset_4lists.pkl','wb')
    pickle.dump((root_image_list, icon_image_list, icon_label_list, icon_pos_list), fileHandler)