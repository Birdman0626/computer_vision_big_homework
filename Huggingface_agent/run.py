import os,shutil,torch,pickle

from utils.percept import get_perception_infos
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

if __name__ == '__main__':
    ### Load ocr model and icon detection model.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = AutoProcessor.from_pretrained('./model')
    model = AutoModelForZeroShotObjectDetection.from_pretrained('./model').to(device)

    
    ### Get all the images name in the datasets.
    path_dir = './datasets/'
    output_dir = './Huggingface_agent/output'
    temp_dir = './Huggingface_agent/temp'
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.mkdir(temp_dir)
    
    fileHandler = open('./datasets/dataset.pkl','rb')
    dataset = pickle.load(fileHandler)
    fileHandler.close()
    
    image_list = []
    for i in dataset:
        image_list.append(f'{i["image_path"]}')
    
    ### Get the interception message by walking through the dataset.
    perception_infos_list, width_list, height_list = list(),list(),list()
    for i in range(len(image_list)):
        file = image_list[i]
        output_file = image_list[i]
        
        # TODO: argumentise.
        perception_infos, width, height = get_perception_infos(
            temp_file_prefix = temp_dir,
            file = file, 
            output_file_prefix = output_dir,
            output_file = output_file, 
            font_path = '/System/Library/Fonts/Times.ttc',
            groundingdino_model = model,
            preprocessor = processor,
            device = device
        )
        perception_infos_list.append(perception_infos)
        width_list.append(width)
        height_list.append(height)
        