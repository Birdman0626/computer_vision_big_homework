import os, shutil, torch, pickle, yaml, cv2
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from Huggingface_agent.utils_fixed.percept import get_perception_infos
from data_utils.xml_tensor_changes import tensor2xml2
from model_part2.model_classifier import Classifier
from torch import optim, nn
from torchvision import transforms as tr
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd

#transform methods
root_transform_method = tr.Compose([
    tr.ToTensor(),
    tr.Resize(size=(512,512))
])

icon_transform_method = tr.Compose([
    tr.ToTensor(),
    tr.Resize(size=(32,32))
])

#all labels
LABEL_LIST = ['clickable', 'disabled', 'selectable','scrollable']

def load_checkpoint(
    model:nn.Module,
    optimizer:optim.Optimizer,
    file:str,
    map_location = 'cpu'
):
    """load model and optimizer from given checkpoint

    Args:
        model (nn.Module): the nn model
        optimizer (optim.Optimizer): the optimizer for model
        file (str): path for checkpoint file
        map_location (str, optional): device for model and optimizer. Defaults to 'cpu'.
    """
    d = torch.load(file, map_location, weights_only=True)
    model.load_state_dict(d['model_state'])
    optimizer.load_state_dict(d['optimizer_state'])

def get_slicer_image(image_path:str, box:list[int]):
    """get given box image from the origin image

    Args:
        image_path (str): path for image
        box (list[int]): given box whose structure should be [xmin, ymin, xmax, ymax]

    Returns:
        _type_(np.ndarray): the array for sliced image
    """
    
    return np.array(cv2.imread(image_path))[box[1]:box[3]+1, box[0]:box[2]+1]

def fix_boxes(image_boxes: list[list[int]], confidences: list[int], root_info:list[int]):
    """for image boxes and confidences, select the box and confidence that are in the root area
    
    Args:
        image_boxes (list[list[int]]): list of boxes whose structure should be
        [
            [xmin, ymin, xmax, ymax],
            [xmin, ymin, xmax, ymax],
            ...
        ]
        confidences (list[int]): the confidence with image boxes
        root_info (list[int]): the root info for origin image, whose structure should be [xmin, ymin, xmax, ymax]

    Returns:
        image_boxed_fixed (list[list[int]]): the selected boxes whose structure is still
        [
            [xmin, ymin, xmax, ymax],
            [xmin, ymin, xmax, ymax],
            ...
        ]
        confidences_fixed (list[int]): the selcted confidence
    """
    x_min, y_min, x_max, y_max = root_info
    image_boxed_fixed = []
    confidences_fixed = []
    for idx, box in enumerate(image_boxes):
        cond = ((x_min-5 <= box[0]) and 
                (box[2] <= x_max+5) and 
                (y_min-5 <= box[1]) and 
                (box[3] <= y_max+5))
        if cond: 
            image_boxed_fixed.append(box)
            confidences_fixed.append(confidences[idx])
        
    return image_boxed_fixed, confidences_fixed        
    
@torch.no_grad()
def main():
    #load ocr and grounding dino model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = AutoProcessor.from_pretrained('./model_dino')
    model = AutoModelForZeroShotObjectDetection.from_pretrained('./model_dino').to(device)
    model.load_adapter("./checkpoint-2900")
    
    #make essential directories
    output_dir = './output_4_with_finetuned_dino_1'
    temp_dir = './temp'
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.mkdir(temp_dir)
    
    #read all image paths and root infos
    fileHandler = open('./dataset.pkl','rb')
    dataset = pickle.load(fileHandler)
    fileHandler.close()
    
    image_list = []
    root_info_list = []
    for i in dataset:
        image_list.append(f'{i["image_path"]}')
        root_info_list.append(i["range"][i["name"].index("root")])
        
    #load classifier model and its params
    classifier = Classifier(num_classes=4)
    with open('./model_part2/config/class_4.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
    optimizer = optim.SGD(classifier.parameters(), lr=config['train']['learning_rate'], momentum=0.3)
    load_checkpoint(classifier, optimizer, "./model_part2/" + config["train"]["ckpt_dir"] 
                    + "/model-4-classes.ckpt", device)
    
    #go through the dataset
    relpath_list = []
    label_list = []
    confidence_list = []
    image_box_list = []
    
    for i in tqdm(range(len(image_list))):
        file = image_list[i]
        root_info = root_info_list[i]
        root_pic = root_transform_method(get_slicer_image(file, root_info)).to(torch.float32)
        
        #get image boxes and confidences
        image_boxes, confidences = get_perception_infos(
            temp_file_prefix = temp_dir,
            file = file, 
            output_file_prefix = output_dir,
            output_file = file,
            font_path = 'C:\\Windows\\Fonts\\times.ttf',
            groundingdino_model = model,
            preprocessor = processor,
            device = device
        )
        tqdm.write(str(image_boxes))
        
        # image_boxes = []
        # for j in perception_info:
        #     image_boxes.append(j["coordinates"])
        
        #fix boxes and confidence
        image_boxes, confidences = fix_boxes(image_boxes, confidences, root_info)
        if(len(image_boxes) == 0):
            tqdm.write("hack! in "+file)
            continue
        
        # max_confidence = max(confidences)
        # confidences = [x/max_confidence - np.random.rand()*0.05 for x in confidences]
        
        #get label from classifier model
        root_pic_batch = torch.repeat_interleave(root_pic.unsqueeze(0), len(image_boxes), dim = 0)
        icon_pic_batch = torch.zeros((len(image_boxes), 3, 32, 32))
        
        for idx, icon_coords in enumerate(image_boxes):
            icon_pic = icon_transform_method(get_slicer_image(file, icon_coords)).to(torch.float32)
            icon_pic_batch[idx] = icon_pic
        
        #print(root_pic_batch.shape, icon_pic_batch.shape, torch.tensor(image_boxes).shape)
        pred = classifier.forward(icon_pic_batch, root_pic_batch, torch.tensor(image_boxes)).squeeze()
        tqdm.write(f"{pred.shape[0]}")
        
        #write xml file
        file = Path(file)
        output_xml_file_path = Path(output_dir) / file.parent.name / file.name
        output_xml_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        relpath_list.extend([str(file.parent.name+ "\\" + file.name) for _ in range(len(image_boxes))])
        label_list.extend(LABEL_LIST[torch.argmax(pred[idx])] for idx in range(len(image_boxes)))
        confidence_list.extend(confidences)
        for image_box in image_boxes:
            string = f"{image_box[0]} {image_box[1]} {image_box[2] - image_box[0]} {image_box[3] - image_box[1]}"
            #print(string)
            image_box_list.append(string)
                
        tensor2xml2(file_name = str(output_xml_file_path).replace("png", "xml"), icon_boxes = image_boxes, pred = pred, root_info = root_info)
        tqdm.write(f"{file} finished")
        #break
        
    #write csv file
    data_csv = {
        "file_name":relpath_list,
        "image_box":image_box_list,
        "confidence":confidence_list,
        "label":label_list
    }
    df = pd.DataFrame(data_csv)
    df.to_csv('output_4_with_finetuned_dino_1.csv', index=False, header=False)
        
    

if __name__ == '__main__':
    main()
