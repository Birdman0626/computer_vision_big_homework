from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch
from PIL import Image

from transformers import Trainer, TrainingArguments

from loss_utils import loss_helper
from finetune_utils import get_finetune_model, get_dataset

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else "cpu" 
    processor = AutoProcessor.from_pretrained('./model')
    model = AutoModelForZeroShotObjectDetection.from_pretrained('./model').to(device)
    finetune_model = get_finetune_model(model,device)
    train_dataset = get_dataset()
    
    def data_collator(batch:list[dict]):
        paths = [ image['image_path'] for image in batch]
        Images = [Image.open(f"./{i}").convert("RGB") for i in paths]
        text = ['an interface. an icon.' for _ in range(len(paths))]
        inputs = processor(images=Images, text=text,return_tensors='pt').to(device)
        inputs['labels'] = {
            'input_ids': inputs['input_ids'],
            'target_sizes': [image.size[::-1] for image in Images],
            'bbox': [image['range'] for image in batch],
            'icon_num': [image['icon_num'] for image in batch] 
        }
        return inputs
    
    
    def loss(outputs, labels, **kwargs):
        # Training with box threshold is zero.
        res = processor.post_process_grounded_object_detection(
            outputs,
            labels['input_ids'],
            box_threshold=0.,
            text_threshold=0.,
            target_sizes=[i[::-1] for i in labels['target_sizes']]
        )
        return loss_helper(res, labels, device)
    
    
    train_args = TrainingArguments(
        output_dir='./ckpt',
        overwrite_output_dir='./ckpt',
        do_train=True,
        do_eval=False,
        per_device_train_batch_size=1,
        save_steps=20,
        torch_empty_cache_steps=500,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        use_cpu = True if not torch.cuda.is_available() else False
    )
    
    trainer = Trainer(
        model = finetune_model,
        args = train_args,
        train_dataset=train_dataset,
        processing_class=processor,
        data_collator=data_collator,
        compute_loss_func=loss
    )
    
    trainer.train()