import torch.nn as nn
import torch.autograd as autograd
import numpy as np


x = torch.tensor([3.,1.,4.],requires_grad=True)
y = torch.tensor([2.,3.,1.],requires_grad=True)

op = x*y
op = torch.sum(op)

op.backward() #Backprop

#grad of x with respect to op
x.grad
#grad of y with respect to op
y.grad

