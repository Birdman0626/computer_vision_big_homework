from PIL import Image
import pickle
from peft import LoraConfig, get_peft_model


def get_finetune_model(model,device):
    peft_config = LoraConfig(
        target_modules=[
            'conv_encoder.projection',
            'input_proj_vision.0.0',
            'input_proj_vision.1.0',
            'input_proj_vision.2.0',
            'input_proj_vision.3.0',
            'fusion_layer.vision_proj',
            'fusion_layer.values_vision_proj',
            'fusion_layer.out_vision_proj',
            'decoder.bbox_embed.0-5',
            'encoder_output_bbox_embed.layers'
            'bbox_embed.0-5.layers',
            'deformable_layer.sampling_offsets',
            'deformable_layer.attention_weights',
            'deformable_layer.value_proj',
            'deformable_layer.output_proj',
            'deformable_layer.fc1',
            'deformable_layer.fc2'
        ], r=8, lora_dropout=0.1)
    model_2 = get_peft_model(model, peft_config).to(device)
    return model_2


def get_dataset():
    fileHandler = open('./datasets/dataset.pkl','rb')
    dataset = pickle.load(fileHandler)
    fileHandler.close()
    return dataset
