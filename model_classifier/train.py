import torch, yaml, pickle
from torch.nn import functional as F, Module
from torch.nn import Module
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as tr
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path
from model_classifier import Classifier

#set seeds and constants
torch.manual_seed(42)
eps =1e-12

#transform methods
root_transform_method = tr.Compose([
    tr.ToTensor(),
    tr.Resize(size=(512,512))
])

icon_transform_method = tr.Compose([
    tr.ToTensor(),
    tr.Resize(size=(32,32))
])
        
class pic_dataset(Dataset):
    """training dataset
    """
    def __init__(self, dataset_path = "../dataset_5lists.pkl"):
        super(pic_dataset, self).__init__()
        fileHandler = open(dataset_path,'rb')
        dataset = pickle.load(fileHandler)
        fileHandler.close()
        root_image_list, icon_image_list, icon_label_list, icon_pos_list = dataset
        self.root_image_list = root_image_list
        self.icon_image_list = icon_image_list
        self.icon_label_list = icon_label_list
        self.icon_pos_list = icon_pos_list
        self.root_transform_method = root_transform_method
        self.icon_transform_method = icon_transform_method

    def __len__(self):
        return len(self.root_image_list)

    def __getitem__(self, idx):
        root_image = self.root_transform_method(self.root_image_list[idx]).to(torch.float32)
        icon_image = self.icon_transform_method(self.icon_image_list[idx]).to(torch.float32)
        icon_pos = torch.tensor(self.icon_pos_list[idx], dtype=torch.float32)
        icon_label = self.icon_label_list[idx]
        #print(root_image.shape, icon_image.shape, icon_pos.shape, icon_label)
        return icon_image, root_image, icon_pos, icon_label

class AverageMeter:
    """tool for calculating
    """
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.cnt = 0
        self.sum = 0
        self.avg = 0
    def update(self, val, n=1):
        self.val = val
        self.cnt += n
        self.sum += n*val
        self.avg = self.sum/self.cnt

class FocalLoss:
    def __init__(self, alpha:list[float], gamma:float = torch.e):
        #ALPHA = torch.tensor([0.8,1,1.1,0.7,1.5], dtype=torch.float32).to(pred.device)
        #GAMMA = torch.e
        self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.alpha /= torch.sum(self.alpha)
        self.gamma = gamma
    def __call__(self, pred:torch.Tensor, lbl:torch.Tensor):
        """
        loss = -\sum q \cdot \alpha \cdot (1-pred)^{\gamma} \cdot (pred + \eps)

        Args:
            pred (torch.Tensor): the pred label, shape should be (Batch, num_classes)
            lbl (torch.Tensor): the true label, shape should be (Batch,)

        Returns:
            _type_: loss
        """
        alpha, gamma = self.alpha.to(pred.device), self.gamma
        q = F.one_hot(lbl, pred.shape[1])
        return torch.sum(-q*alpha*((1-pred)**gamma)*torch.log(pred + eps))

def train_epoch(
    model: Module,
    train_loader:DataLoader,
    optimizer:optim.Optimizer,
    scheduler:ReduceLROnPlateau = None,
    epoch:int = 1,
    device = 'cpu',
    loss_function = F.cross_entropy
)->None:
    """train function

    Args:
        model (Module): the model
        train_loader (DataLoader): the training dataset
        optimizer (optim.Optimizer): the optimizer for model
        scheduler (ReduceLROnPlateau, optional): the scheduler for lr. Defaults to None.
        epoch (int, optional): epoches for training. Defaults to 1.
        device (str, optional): device for data. Defaults to 'cpu'.
        loss_function (function, optional): function to calculate loss. 
            should be loss_function(pred, lbl), 
            in which:
            pred (torch.Tensor): the pred label, shape should be (Batch, num_classes)
            lbl (torch.Tensor): the true label, shape should be (Batch,)
            Defaults to F.cross_entropy.
    """
    model.train()
    losses = AverageMeter()
    idx = 0
    with tqdm(train_loader, desc=f'training epoch {epoch}') as processBar:
        for icon, pic, pos, lbl in processBar:
            optimizer.zero_grad()
            idx += 1
            # 预测
            icon, pic, pos, lbl = icon.to(device), pic.to(device), pos.to(device), lbl.to(device)
            pred = model.forward(icon, pic, pos)
            # 更新loss
            loss = loss_function(pred, lbl)
            losses.update(loss.item())
            processBar.set_postfix(loss = f'{losses.avg:.4f}')
            # 更新
            
            loss.backward()
            optimizer.step()
            if idx%10==0:
                idx = 0
                if scheduler is not None:
                    scheduler.step(losses.avg)
                    
@torch.no_grad()
def test_epoch(
    model:Module,
    test_loader:DataLoader,
    device = 'cpu',
    epoch:int = 1,
    loss_function = F.cross_entropy
):
    """test function

    Args:
        model (Module): the model
        train_loader (DataLoader): the testing dataset
        device (str, optional): device for data. Defaults to 'cpu'.
        epoch (int, optional): epoches for testing. Defaults to 1.
        loss_function (function, optional): function to calculate loss. 
            should be loss_function(pred, lbl), 
            in which:
            pred (torch.Tensor): the pred label, shape should be (Batch, num_classes)
            lbl (torch.Tensor): the true label, shape should be (Batch,)
            Defaults to F.cross_entropy.

    Returns:
        accuracy: 准确率
    """
    model.eval()
    losses = AverageMeter()
    right_cnt = 0
    sum_cnt = 0
    with tqdm(test_loader, desc=f'testing epoch {epoch}') as processBar:
        for icon, pic, pos, lbl in processBar:
            icon, pic, pos, lbl = icon.to(device), pic.to(device), pos.to(device), lbl.to(device)
            pred = model.forward(icon, pic, pos)
            # 计算loss
            loss = loss_function(pred, lbl)
            losses.update(loss.item())
            # 计算right
            #print(torch.argmax(pred, 1))
            right_cnt += torch.sum(torch.argmax(pred, -1)==lbl).item()
            sum_cnt += pred.shape[0]
            # 显示
            processBar.set_postfix(acc = f'{right_cnt/sum_cnt*100:.3f}%', loss = f'{losses.avg:.4f}')
    return right_cnt/sum_cnt

            
def main():
    # args
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='class_5')
    args = parser.parse_args()
    with open('./config/'+args.config+'.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
        
    # data
    data_dir = Path(config['train']['data_dir']).absolute()
    ckpt_dir = Path(config['train']['ckpt_dir']).absolute()
    if not data_dir.exists():
        raise RuntimeError(f'data dir <{data_dir}> doesn\'t exist')
    if not ckpt_dir.exists():
        ckpt_dir.mkdir()
    tqdm.write(f'checkpoints is at: {ckpt_dir.absolute()}')
    data = pic_dataset()
    train_loader = DataLoader(
        data, batch_size=config["train"]["batch_size"],
        num_workers=config["train"]["num_workers"],
        shuffle=True
    )
    
    # train
    device = config['train']['device']
    model = Classifier(**config['model']).to(device=device)
    optimizer = optim.SGD(model.parameters(), lr=config['train']['learning_rate'], momentum=0.3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10)
    flag = False
    loss_function = FocalLoss(config['train']['alpha'], gamma=0.)
    tqdm.write(f"alpha: {config['train']['alpha']}")
    try:
        for epoch in range(1, config['train']['max_epoch']):
            train_epoch(
                model, train_loader, 
                optimizer, scheduler, 
                epoch, device, loss_function
            )
            flag = True
            acc = test_epoch(model, train_loader, device, epoch, loss_function)
    # auto saving
    except KeyboardInterrupt:
        tqdm.write('You exited\n'+ ('Saving...' if flag else ''))
    except Exception as err:
        raise err
    finally:
        if flag:
            # name = 'model-5-classes'
            name = config['name']
            torch.save({
                'model_state':model.state_dict(),
                'optimizer_state':optimizer.state_dict()
            }, str(ckpt_dir/f'{name}.ckpt'))
            tqdm.write('Saving as '+ str(ckpt_dir/f'{name}.ckpt'))
            
if __name__=='__main__':
    main()
