import os,shutil

from percept import get_perception_infos
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

if __name__ == '__main__':
    ### Load ocr model and icon detection model.
    device = 'cpu'
    processor = AutoProcessor.from_pretrained('./model')
    model = AutoModelForZeroShotObjectDetection.from_pretrained('./model').to(device)

    
    ### Get all the images name in the datasets.
    path_dir = './Huggingface_agent/dataset_1'
    output_dir = './Huggingface_agent/dataset_1_output'
    temp_dir = './Huggingface_agent/temp'
    
    if not os.path.exists(path_dir):
        raise "No dataset error!"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.mkdir(temp_dir)
    
    file_list = os.listdir(path_dir)
    image_list, xml_list = list(), list()
    for file in file_list:
        _, ExtensionName = os.path.splitext(file)
        if ExtensionName == '.png':
            image_list.append(file)
        elif ExtensionName == '.xml':
            xml_list.append(file)

    
    ### Get the interception message by walking through the dataset.
    perception_infos_list, width_list, height_list = list(),list(),list()
    for i in range(len(image_list)):
        file = image_list[i]
        output_file = image_list[i]
        
        # TODO: argumentise.
        perception_infos, width, height = get_perception_infos(
            input_file_prefix = path_dir,
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
        