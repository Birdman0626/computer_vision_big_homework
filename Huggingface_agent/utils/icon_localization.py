from utils.crop import calculate_size, calculate_iou
from PIL import Image
import torch

def remove_boxes(boxes_filt, size, iou_threshold=0.5):
    boxes_to_remove = set()

    for i in range(len(boxes_filt)):
        if calculate_size(boxes_filt[i]) > 0.05*size[0]*size[1]:
            boxes_to_remove.add(i)
        for j in range(len(boxes_filt)):
            if calculate_size(boxes_filt[j]) > 0.05*size[0]*size[1]:
                boxes_to_remove.add(j)
            if i == j:
                continue
            if i in boxes_to_remove or j in boxes_to_remove:
                continue
            iou = calculate_iou(boxes_filt[i], boxes_filt[j])
            if iou >= iou_threshold:
                boxes_to_remove.add(j)

    boxes_filt = [box for idx, box in enumerate(boxes_filt) if idx not in boxes_to_remove]
    
    return boxes_filt


def det(input_image_path, caption, groundingdino_model, processor, device, box_threshold=0.05, text_threshold=0.5):
    image = Image.open(input_image_path).convert("RGB")
    size = image.size

    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith('.'):
        caption = caption + '.'
    
    inputs = processor(images=image, text=caption, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = groundingdino_model(**inputs)
    
    result = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]]
    )[0]
    
    boxes_filt = result['boxes'].detach().cpu().int().tolist()

    # H, W = size[1], size[0]
    # for i in range(boxes_filt.size(0)):
    #     boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
    #     boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
    #     boxes_filt[i][2:] += boxes_filt[i][:2]

    # boxes_filt = boxes_filt.cpu().int().tolist()
    
    filtered_boxes = remove_boxes(boxes_filt, size)  # [:9]
    coordinates = []
    for box in filtered_boxes:
        coordinates.append([box[0], box[1], box[2], box[3]])

    return coordinates