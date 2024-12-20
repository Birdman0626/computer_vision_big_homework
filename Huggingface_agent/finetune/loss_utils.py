import torch
import numpy
from torchvision.ops.boxes import box_area as area
from scipy.optimize import linear_sum_assignment

def box_iou(boxes1, boxes2):
    area1 = area(boxes1)
    area2 = area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

def loss_helper(res,labels, device):
    '''
    Calculate L1 loss in following steps:
    First: results sorted with scores to get the same shape of target.
    Second: Calculate results to the central points. (matmul [[0.5,0,0.5,0],...])
    Third: Get the L1_loss/loss by Hungarian alg.
    '''
    
    def distance(a,b):
        return (a[0]-b[0])**2 + (a[1]-b[1])**2
    
    outputs_with_score=[(i['scores'],i['boxes']) for i in res]
    targets_boxes = [l for l in labels['bbox']]
    image_size = [i[::-1] for i in labels['target_sizes']]
    # traverse with batch.
    overbatch_loss = None
    for i in range(len(outputs_with_score)):
        # Calculate center bias and l1 loss.
        # print(res[i]['labels'])
        scores,boxes = outputs_with_score[i][0],outputs_with_score[i][1]
        x, y = image_size[i]
        target_boxes = targets_boxes[i]
        ind = [(scores[i],i) for i in range(scores.shape[0])]
        indices_with_scores = sorted(ind, key = lambda i:i[0], reverse=True)
        indices = [i[1] for i in indices_with_scores][:len(target_boxes)]
        # Pad the same size with target.
        pred_boxes = boxes[indices]
        centralized = torch.tensor([[0.5,0.,-1.,0.],
                                    [0.,0.5,0.,-1.],
                                    [0.5,0.,1.,0.],
                                    [0.,0.5,0.,1]])
        normalized = torch.tensor([[x,y,x,y]],dtype=torch.float32)
        # print(f'size:({x},{y})')
        # print('pred_boxes',pred_boxes)
        # print('targ_boxes',target_box)
        # print('----------------------------------')
        # Find the center points.
        pred_cen = pred_boxes.cpu() @ centralized / normalized # icon x 4 @ 4 x 2 -> icon x 2
        targ_cen = torch.tensor(target_boxes,dtype = torch.float32) @ centralized / normalized
        # print('pred_cen',pred_cen)
        # print('targ_cen',targ_cen)
        cost_mat = torch.zeros((pred_cen.shape[0],pred_cen.shape[0]))
        for i in range(pred_cen.shape[0]):
            for j in range(targ_cen.shape[0]):
                cost_mat[i,j] = distance(pred_cen[i],targ_cen[j])
        
        row,col = linear_sum_assignment(cost_mat.detach().numpy())      # cpu may need!
        pred_cen, targ_cen = pred_cen[row],targ_cen[col]
        pred_cen, targ_cen =  pred_cen.to(device), targ_cen.to(device)
        # print('***********************************')
        # print('after pred_cen',pred_cen)
        # print('after targ_cen',targ_cen)
        loss_bbox = torch.nn.functional.l1_loss(pred_cen,targ_cen)
        
        # Calculate GIOU loss.
        target_boxes_tensor = torch.tensor(target_boxes, dtype=torch.float32).to(device)
        loss_giou = 1 - torch.diag(generalized_box_iou(target_boxes_tensor, pred_boxes))
        loss_giou = loss_giou.sum() / loss_giou.shape[0]
        if overbatch_loss == None:
            overbatch_loss = loss_giou + loss_bbox
        else:
            overbatch_loss += loss_giou + loss_bbox
    overbatch_loss /= len(image_size)
    return overbatch_loss

