# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

import inspect
import sys, os
import datetime


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.linear1(x).clamp(min=0)
        print(h_relu.grad_fn)
        y_pred = self.linear2(h_relu)

        print(y_pred.grad_fn)
        #print(os.path.abspath(sys.modules[y_pred.grad_fn.__class__.__module__].__file__))


        #print(dir(y_pred.grad_fn))
        #print([func for func in dir(y_pred.grad_fn) if callable(getattr(y_pred.grad_fn, func))])
        return y_pred

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(3810304, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 3810304)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
# x = torch.randn(N,D_in)
# y = torch.randn(N, D_out)

x = torch.randn(N, 3, 256, 256)
y = torch.randn(N, 10)

# Construct our model by instantiating the class defined above
model = Net().cuda() #TwoLayerNet(D_in, H, D_out).cuda()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(1):
    # Forward pass: Compute predicted y by passing x to the model
    start = datetime.datetime.now() 

    y_pred = model(x.cuda())

    # for idx, m in model.named_children(): #enumerate(model.modules()):
    #     #print(idx, '->', list(m.parameters()))
    #     for p in m.parameters():
    #         print(p.grad_fn)

    print(model.modules())
    # Compute and print loss
    loss = criterion(y_pred.cuda(), y.cuda())
    #if t % 100 == 99:
    print(t, loss.item())
    print(loss.get_device())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()


    end = datetime.datetime.now()
    gap=end-start
 #   print(gap)

    start2 = datetime.datetime.now() 
    loss.backward()
    end = datetime.datetime.now()
    gap=end-start2
    print(gap)

    for idx, m in model.named_children(): #enumerate(model.modules()):
        #print(idx, '->', list(m.parameters()))
        for p in m.parameters():
            print(p.grad_fn)
    optimizer.step()

print(torch.cuda.get_tensor_db().decode("utf-8") )