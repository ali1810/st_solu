
from mmap import ACCESS_DEFAULT
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from infrared.utils import convert_to_binary
import numpy as np
from torch.optim.lr_scheduler import StepLR, LinearLR


from infrared.utils import group_accuracy
from infrared.utils import pred_acc

class Finger(nn.Module):
    def __init__(self, inp):
        super().__init__()

        self.fc1 = nn.Linear(inp, inp)
        self.fc1_bn = nn.BatchNorm1d(inp)
        self.dropout_fc1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(inp, 256)
        self.fc2_bn = nn.BatchNorm1d(256)
        self.dropout_fc2 = nn.Dropout(0.3)

    def forward(self, x) : 

        x = F.relu(self.fc1_bn(self.fc1(x.squeeze(1))))#
        #x = F.relu(self.fc1(x))
        x = self.dropout_fc1(x)
        x = F.relu(self.fc2_bn(self.fc2(x)))
        #x = F.relu(self.fc2(x))
        x = self.dropout_fc2(x)

        x = torch.flatten(x,1)

        return x

class SepSpec(nn.Module):
    def __init__(self, num_classes, split_at):
        
        super().__init__()

        diff = 3400 - split_at

        self.fc1 = nn.Linear(diff, diff) #diff, 1056
        self.fc1_bn = nn.BatchNorm1d(diff)
        self.dropout_fc1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(diff, 256) #1056, 256
        self.fc2_bn = nn.BatchNorm1d(256)
        self.dropout_fc2 = nn.Dropout(0.3)

        self.finger = Finger(split_at)
        self.fc5 = nn.Linear(512, num_classes)  

    def forward(self, x_func, x_fing) : 
        x_func = F.relu(self.fc1_bn(self.fc1(x_func.squeeze(1))))
        #x_func = F.relu(self.fc1(x_func))
        x_func = self.dropout_fc1(x_func)
        x_func = F.relu(self.fc2_bn(self.fc2(x_func)))
        #x_func = F.relu(self.fc2(x_func))
        x_func = self.dropout_fc2(x_func)
        x_func = torch.flatten(x_func,1)

        x = torch.cat([x_func, self.finger(x_fing)], axis= 1)
        x = self.fc5(x)
        output = torch.sigmoid(x)

        return output




def model_init(lr, gamma ,num_classes, split_at):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SepSpec(num_classes, split_at).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)#, weight_decay=1e-5)
    criterion = nn.BCELoss(reduction='none')
    scheduler = LinearLR(optimizer, start_factor=0.5, total_iters=100)
    return model, optimizer, criterion, scheduler, device

def train(model, optimizer, scheduler, criterion, train_loader, device, split_at):
    batch_loss = []
    batch_accuracy = []
    model.train()
    for batch_idx, data in enumerate(train_loader):
        xs, ws, ys = data['xs'], data['ws'].to(device), data['ys'].to(device)
        x_func, x_fing  = xs[:,:,split_at:].to(device), xs[:,:,:split_at].to(device)

        optimizer.zero_grad()
        output = model(x_func, x_fing)
        loss = criterion(output, ys)
        
        loss = (loss*ws).mean()
        binary_preds = convert_to_binary(output.detach().cpu(), torch.tensor([0.5]))
        acc_1, acc_0 = group_accuracy(binary_preds, ys.detach().cpu(), agg='mean')
        batch_loss.append(loss.detach().cpu().numpy())

        batch_accuracy.append((acc_1, acc_0))
        
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    mean_batch_acc1 = np.mean([x[0] for x in batch_accuracy])
    mean_batch_acc0 = np.mean([x[1] for x in batch_accuracy])

    return np.array(batch_loss).mean(), mean_batch_acc1, mean_batch_acc0, model

def test(model, epoch, total_epochs, criterion, test_loader, device, split_at):
    model.eval()
    test_loss = []
    test_accuracy = []
    preds = []
    raw_preds = []
    with torch.no_grad():
        for data in test_loader:
            xs, ws, ys = data['xs'], data['ws'].to(device), data['ys'].to(device)
            x_func, x_fing  = xs[:,:,split_at:].to(device), xs[:,:,:split_at].to(device)

            predictions = model(x_func, x_fing)
            binary_preds = convert_to_binary(predictions.cpu(), torch.tensor([0.5]))
            acc_1, acc_0 = pred_acc(binary_preds[0], ys.cpu())
            
            loss = criterion(predictions, ys)

            loss = (loss).mean()

            if epoch == (total_epochs - 1):
                preds.append((binary_preds[0], ys.cpu()))
                raw_preds.append(predictions.cpu(), ys.cpu())
            test_loss.append(loss.cpu())
            test_accuracy.append((acc_1, acc_0))

    epoch_loss = np.array(test_loss).mean()
    epoch_acc1 = np.mean([j[0] for j in test_accuracy])
    epoch_acc0 = np.mean([j[1] for j in test_accuracy])

    return epoch_loss, epoch_acc1, epoch_acc0, preds, raw_preds




