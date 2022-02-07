'''
A demo for leveraging cldice in training
'''

'''
imports
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from cldice_loss.pytorch.cldice import soft_dice_cldice
from torch.utils.data import DataLoader, Dataset

#-----------------------------------------------
'''
Constants
'''
EPOCHS = 10
BATCH_SIZE=16
LR = 1e-3
NUM_CLASSES = 6
#-----------------------------------------------

class DemoData(Dataset):
    def __init__(self):
        super(DemoData, self).__init__()
        self.data = torch.rand(100, 3, 256, 256).float()
        self.label = torch.randint(0, NUM_CLASSES-1, (100, 256, 256))
        
    def __getitem__(self, index):
        return self.data[index, ...], self.label[index, ...]

    def __len__(self):
        return self.data.size(0)

class Model(nn.Module):
    def __init__(self, in_channels_=3, out_channels_=NUM_CLASSES):
        super(Model, self).__init__()
        self._conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels_, out_channels=5, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            nn.Conv2d(in_channels=5, out_channels=out_channels_, kernel_size=1, stride=1, padding=0)
        )
    
    def forward(self, x):
        return self._conv(x)

def embedding_onehot(label, num_classes):
    '''
    embedding label to one-hot form
    
    Args:
        label: (LongTensor) [N, W, H]
        num_classes: (Int) number of classes in segmentation (including background)
    
    Returns:
        one-hot labels: (FloatTensor) [N, C, W, H]
    '''    
    b, w, h = label.shape
    label = label.flatten() # [N, W, H] -> [N*W*H]
    
    id_mtx = torch.eye(num_classes) # [C, C]
    
    label = id_mtx[label] # [N*W*H, C]
    
    label = label.view(b, w, h, -1).permute(0, 3, 1, 2) # [B, C, W, H]
    
    return label.float()    

def train():
    demoset = DemoData()
    loader = DataLoader(demoset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
    net = Model.cuda() if torch.cuda.is_available() else Model()
    criterion = soft_dice_cldice()
    optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=1e-8)
    
    for epoch in range(EPOCHS):
        total_loss = 0.0
        count = 0
        
        for img, mask in loader:
            if torch.cuda.is_available():
                img = img.cuda()
                mask = mask.cuda()
            batch_size = img.size(0)
            count += batch_size
            
            optimizer.zero_grad()
            
            logits = net(img) # [B, C, W, H]
            
            preds = torch.max(logits, dim=1)[1].long() # [B, W, H]
            
            preds = embedding_onehot(preds, NUM_CLASSES)
            mask = embedding_onehot(mask, NUM_CLASSES)
            
            loss = criterion(mask, preds)
            
            loss.backward()
            
            optimizer.step()
            
            total_loss += loss.item()*batch_size
            
        total_loss /= count
        
        print('epoch %d: training loss %2f'%(epoch, total_loss))
            

if __name__ == "__main__":
    train()
    
    
        