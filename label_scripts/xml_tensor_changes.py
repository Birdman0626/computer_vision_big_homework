import xml.etree.ElementTree as ET
from xml.etree.ElementTree import indent
import torch
import torch.nn.functional as F
 
LABEL_LIST = ['root', 'level_0', 'clickable', 'disabled', 'selectable', 'level_1', 'scrollable', 'level_2']

def xml2tensor(file_name: str) -> list[tuple[torch.Tensor, torch.Tensor]]: 
    """change an xml file to some tensors

    Args:
        file_name (str): an xml file_name

    Returns:
        info_list(list[tuple[torch.Tensor, torch.Tensor]]): list of (torch.tensor([x_center, y_center, width, height]), one_hot)
    """
    info_list: list[tuple[torch.Tensor, torch.Tensor]] = []
    xml_tree = ET.parse(file_name)
    root = xml_tree.getroot()
    for single_object in root.findall('object'):
        one_hot = F.one_hot(torch.tensor(LABEL_LIST.index(single_object.find("name").text)), len(LABEL_LIST))
        x_min = int(single_object.find("bndbox")[0].text)
        y_min = int(single_object.find("bndbox")[1].text)
        x_max = int(single_object.find("bndbox")[2].text)
        y_max = int(single_object.find("bndbox")[3].text)
        x_center = (x_min+x_max)/2
        y_center = (y_min+y_max)/2
        width = x_max-x_min
        height = y_max-y_min
        info_list.append((torch.tensor([x_center, y_center, width, height]), one_hot))
    return info_list

@torch.no_grad()
def tensor2xml(tensor_list: list[tuple[torch.Tensor, torch.Tensor]], file_name: str):
    """change a list info of tensor into an xml file

    Args:
        tensor_list (list[tuple[torch.Tensor, torch.Tensor]]): list of (torch.tensor([x_center, y_center, width, height]), one_hot)
        file_name (str): an output xml file_name
    """
    root = ET.Element("annotation")
    for info_vector, prob_vector in tensor_list:
        x_center, y_center, width, height = info_vector
        x_min, x_max = x_center - width/2, x_center + width/2
        y_min, y_max = y_center - height/2, y_center + height/2
        label = LABEL_LIST[torch.argmax(prob_vector)]
        
        object = ET.SubElement(root, "object")
        name = ET.SubElement(object, "name")
        name.text = label
        bndbox = ET.SubElement(object, "bndbox")
        x_min_element = ET.SubElement(bndbox, "x_min")
        x_min_element.text = str(int(x_min))
        y_min_element = ET.SubElement(bndbox, "y_min")
        y_min_element.text = str(int(y_min))
        x_max_element = ET.SubElement(bndbox, "x_max")
        x_max_element.text = str(int(x_max))
        y_max_element = ET.SubElement(bndbox, "y_max")
        y_max_element.text = str(int(y_max))
    
    tree = ET.ElementTree(root)
    indent(tree, '\t')
    tree.write(file_name)

@torch.no_grad()
def tensor2xml2(file_name: str, icon_boxes: list, pred: torch.Tensor, root_info: list):
    LABEL_LIST = ['clickable', 'disabled', 'selectable','scrollable']
    root = ET.Element("annotation")
    root_level = ET.SubElement(root, "object")
    name = ET.SubElement(root_level, "name")
    name.text = "root"
    bndbox = ET.SubElement(root_level, "bndbox")
    x_min, y_min, x_max, y_max = root_info
    x_min_element = ET.SubElement(bndbox, "x_min")
    x_min_element.text = str(int(x_min))
    y_min_element = ET.SubElement(bndbox, "y_min")
    y_min_element.text = str(int(y_min))
    x_max_element = ET.SubElement(bndbox, "x_max")
    x_max_element.text = str(int(x_max))
    y_max_element = ET.SubElement(bndbox, "y_max")
    y_max_element.text = str(int(y_max))
    
    for i, icon_rect in enumerate(icon_boxes):
        
        x_min, y_min, x_max, y_max = icon_rect
        idx = torch.argmax(pred[i])
        if idx == 4: continue
        label = LABEL_LIST[idx]
        
        object = ET.SubElement(root, "object")
        name = ET.SubElement(object, "name")
        name.text = label
        bndbox = ET.SubElement(object, "bndbox")
        x_min_element = ET.SubElement(bndbox, "x_min")
        x_min_element.text = str(int(x_min))
        y_min_element = ET.SubElement(bndbox, "y_min")
        y_min_element.text = str(int(y_min))
        x_max_element = ET.SubElement(bndbox, "x_max")
        x_max_element.text = str(int(x_max))
        y_max_element = ET.SubElement(bndbox, "y_max")
        y_max_element.text = str(int(y_max))
    
    tree = ET.ElementTree(root)
    indent(tree, '\t')
    tree.write(file_name)




input_file_name = "dataset_fixed_capital_scrollable/coredraw/frame_1.xml"
output_file_name = "test.xml"

if __name__ == "__main__":
    input_tensor = xml2tensor(input_file_name)
    print(input_tensor)
    tensor2xml(input_tensor, output_file_name)
        
        

        
        
        
        
        
    
        
        
        
