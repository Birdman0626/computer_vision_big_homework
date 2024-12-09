import xml.etree.ElementTree as ET
import os,shutil,pickle
import numpy as np
from PIL import Image

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
        'image_size' (list[int]): The list of images RGB.
        'icon_num' (int): The icon number contain in image.
        'name' (list[str]): The type of these icons.
        'truncated' (list[int]): __I_don't_know_what_is_this__.
        'difficult' (list[int]): __I_don't_know_what_is_this__.
        'range' (list[list[int]]): The icon spreading box.
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
        'truncated':[],
        'difficult':[],
        'range':[]
    }
    
    # find icons and ranges.
    for icon in root.findall('object'):
        image_info['name'].append(icon.find('name').text)
        image_info['truncated'].append(int(icon.find('truncated').text))
        image_info['difficult'].append(int(icon.find('difficult').text))
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
    for directory in os.listdir('./datasets'):
        if directory == '.DS_Store' or directory == 'dataset.pkl':
            continue
        # inside the directory, list all files. Read the base name and skip the same file.
        prefix = f'./datasets/{directory}'
        fileBaseName = ''
        for file in os.listdir(prefix):
            fileNewBaseName, _ = os.path.splitext(file)
            if fileBaseName == fileNewBaseName: continue
            image_info = make_dict(f'{prefix}/{fileNewBaseName}.xml')
            images_info.append(image_info)
            fileBaseName = fileNewBaseName
    
    if os.path.exists('./datasets/dataset.pkl'):
        os.remove('./datasets/dataset.pkl')
    fileHandler = open('./datasets/dataset.pkl','wb')
    pickle.dump(images_info, fileHandler)
    