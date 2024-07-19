import torch
import torch.nn as nn
from tqdm import tqdm
from utils import modify
device='cuda'

class SignClassifier(nn.Module):
    def __init__(self):
        super(SignClassifier,self).__init__()
        self.ff1=nn.Linear(21*3,1024)
        self.ff2=nn.Linear(1024,512)
        # self.ff3=nn.Linear(512,29)
        self.ff3=nn.Linear(512,28)
        self.relu=nn.ReLU()
        self.flatten=nn.Flatten()
        self.dropout1=nn.Dropout(0.2)
        self.dropout2=nn.Dropout(0.2)
    def forward(self,x):
        x=self.flatten(x)
        x=self.dropout1(self.relu(self.ff1(x)))
        x=self.dropout2(self.relu(self.ff2(x)))
        return self.ff3(x)

def _train(dataloader,model,loss_fn,optimizer):
    model.train()
    pbar=tqdm(dataloader)
    for y,X in pbar:
        X=[modify(i) for i in X.tolist()]
        X=torch.Tensor(X).to(device)
        y=torch.Tensor(y).to(device)
        pred=model(X)
        loss=loss_fn(pred,y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pbar.set_description(f"loss: {loss.item():.6f}")
def _test(dataloader,model,loss_fn):
    model.eval()
    pbar=tqdm(dataloader)
    losssum=0
    cnt=0
    acc=0
    for y,X in pbar:
        X=torch.Tensor(X).type(torch.float).to(device)
        y=torch.Tensor(y).to(device)
        pred=model(X)
        loss=loss_fn(pred,y)
        acc+=(pred.argmax(1) == y).type(torch.float).sum().item()
        losssum+=loss.item()
        cnt+=1
    print(f"loss {losssum/cnt} acc {acc/len(dataloader.dataset)}")

def train(train_loader,test_loader,epoch=5):
    model=SignClassifier()
    model.to(device)
    loss_fn=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)

    for i in range(epoch):
        print(f"Epoch {i+1}")
        _train(train_loader,model,loss_fn,optimizer)
        _test(test_loader,model,loss_fn)
    torch.save(model,'PointDetect_3d.pth')
