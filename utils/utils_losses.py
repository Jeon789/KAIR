import torch
import math


def heron_loss(x, y, z, current_step, regularizer=False):
    assert (x.size() == y.size() and y.size() == z.size()), "In heron_loss, the torch tensor's size not same."
    pdist = torch.nn.PairwiseDistance(p=2)
    batch_size = x.size()[0]

    x = x.view(batch_size, -1)
    y = y.view(batch_size, -1)
    z = z.view(batch_size, -1)
    HW = x.size()[1]

    a, b, c = pdist(x, y), pdist(y, z), pdist(z, x)
    if (current_step % 10) in [0,1,2,3,4,5,6] :
        a = a.detach()

    if current_step % 100 == 0 :
        print('-'*100)
        print(torch.mean(a), torch.mean(b), torch.mean(c))
        print('-'*100)

    s = (a+b+c)/2
    area = torch.sqrt( s*(s-a)*(s-b)*(s-c) )
    loss = torch.mean(area) / math.sqrt(HW)


    # 세변의 길이를 비슷하게 만든다. 정삼각형처럼 만든다.
    # 삼각형이 무너져버리는 경우를 방지하기 위하여.
    if regularizer:
        abc = torch.stack([a,b,c],dim=1)
        regularizer = torch.nn.MSELoss()(torch.max(abc,1).values, torch.min(abc,1).values)
        loss += 0.1 * regularizer
    
    if torch.isnan(loss):
        breakpoint()

    return loss


def heron_loss2(x, y, z, current_step, regularizer=False):
    assert (x.size() == y.size() and y.size() == z.size()), "In heron_loss, the torch tensor's size not same."
    pdist = torch.nn.PairwiseDistance(p=2)
    batch_size = x.size()[0]


    x = x.view(batch_size, -1)
    y = y.view(batch_size, -1)
    z = z.view(batch_size, -1)
    HW = x.size()[1]

    a2, b2, c2 = torch.pow(pdist(x, z),2), torch.pow(pdist(x, y),2), torch.pow(pdist(z, y),2)
    area = 0.25 * torch.sqrt(4*a2*b2- torch.pow(a2+b2-c2,2))
    loss = torch.mean(area) / math.sqrt(HW)


    # 세변의 길이를 비슷하게 만든다. 정삼각형처럼 만든다.
    # 삼각형이 무너져버리는 경우를 방지하기 위하여.
    if regularizer:
        abc = torch.stack([a,b,c],dim=1)
        regularizer = torch.nn.MSELoss()(torch.max(abc,1).values, torch.min(abc,1).values)
        loss += 0.1 * regularizer

    return loss


def stable_heron_loss(x, y, z, current_step, regularizer=False):
    assert (x.size() == y.size() and y.size() == z.size()), "In heron_loss, the torch tensor's size not same."
    pdist = torch.nn.PairwiseDistance(p=2)
    batch_size = x.size()[0]

    x = x.view(batch_size, -1)
    y = y.view(batch_size, -1)
    z = z.view(batch_size, -1)
    HW = x.size()[1]

    a, b, c = pdist(x, y), pdist(y, z), pdist(z, x)        

    abc = torch.stack([a,b,c])
    abc = torch.sort(abc,0, descending=True).values  # a >= b >= c
    a,b,c = abc[0], abc[1], abc[2]

    area = 0.25 * torch.sqrt( (a+(b+c) * (c-(a-b)) * (c+(a-b)) * (a+(b-c)) ))
    loss = torch.mean(area) / math.sqrt(HW)

    # 세변의 길이를 비슷하게 만든다. 정삼각형처럼 만든다.
    # 삼각형이 무너져버리는 경우를 방지하기 위하여.
    if regularizer:
        abc = torch.stack([a,b,c],dim=1)
        regularizer = torch.nn.MSELoss()(torch.max(abc,1).values, torch.min(abc,1).values)
        loss += 1e-5 * regularizer


    return loss

def triangle_loss(x, y, z):
    'Project z onto xy line to'
    assert (x.size() == y.size() and y.size() == z.size()), "In heron_loss, the torch tensor's size not same."
    pdist = torch.nn.PairwiseDistance(p=2)
    batch_size = x.size()[0]

    x = x.view(batch_size, -1)
    y = y.view(batch_size, -1)
    z = z.view(batch_size, -1)
    HW = x.size()[1]

    # project z onto xy line
    t = _pw_dot(y-x,z-x) / (_pw_dot(y-x,y-x) + 1e-6)
    h = x + torch.mul(t.unsqueeze(1),y-x)    # (B,1) x (B,HW), broadcasting by torch.mul
    area = _pw_dot(y-x,h-z)
    loss = torch.mean(area) / HW

    return loss

def _pw_dot(a,b):
    "pair-wise dot product"
    "a,b are (B,X) size tensors. Output is (B) size tensors"
    " (B,1,X)x(B,X,1) = (B)"
    a, b = a.unsqueeze(1), b.unsqueeze(2) 
    return torch.bmm(a,b).view(-1)