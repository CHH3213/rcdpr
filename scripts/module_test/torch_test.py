from numpy.core.fromnumeric import shape
import torch
from torch import nn
import numpy as np
from torch.nn import functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

class NeuralNetwork(nn.Module):
    def __init__(self, n_feature=2, n_hidden=128, n_output=1):
        super(NeuralNetwork, self).__init__()   # 继承__init__功能
        self.hidden_layer1 = torch.nn.Linear(n_feature, n_hidden)
        self.hidden_layer2 = torch.nn.Linear(n_hidden, int(n_hidden/2))
        self.output_layer = torch.nn.Linear(int(n_hidden/2), n_output)

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = F.relu(x)
        # for i in range(3):
        x = self.hidden_layer2(x)
        x = F.relu(x)

        pridect_y = self.output_layer(x)
        return pridect_y

def save(model):
    torch.save(model.state_dict(), 'test_model/model.pth')

def load(model):
    model.load_state_dict(torch.load('test_model/model.pth'))

def func(x,y):
    return x**2-y**2+np.sin(x)

def train():
    model = NeuralNetwork().to(device)
    print(model)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01,momentum=0.8)

    model.train()
    
    number = 50000
    np.random.seed(1)
    Xs = np.random.random((number,64,2))*2
    print(shape(Xs))
    # print(Xs[0])
    Zs = func(Xs[:,:,0], Xs[:,:,1])

    print(shape(Zs))
    Xs = torch.tensor(Xs, device=device,dtype=torch.float32)
    # print(Zs)
    
    for i in range(number):
        x = Xs[i]
        z = torch.tensor([Zs[i]],device=device,dtype=torch.float32)
        # print(x)

        pred = model(x)
        loss = loss_fn(pred,z)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.item()
        if i%200 == 0:
            print('loss:', loss)
            # print("x:{}\tz:{}\tpred:{}\tloss:{}\tstep{}".format(x,z.item(),pred.item(),loss,i))
    
    save(model)

def test():
    model = NeuralNetwork().to(device)
    print(model)

    load(model)
    model.eval()
    Xs = np.random.random((64,2)) * 2
    Zs = func(Xs[:,0], Xs[:,1])
    
    # Xs.grad.zero_()
    # pred[0].backward()
    for i in range(len(Zs)):
        x = torch.tensor(Xs[i], device=device,dtype=torch.float32,requires_grad=True)
        pred = model(x)
        pred.backward()
        print((2*x[0].item()+np.cos(x[0].item())),-2*x[1].item(),x.grad.detach())


if __name__ == '__main__':
    # train()
    test()

