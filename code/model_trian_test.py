import time
import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import accuracy_score, f1_score
from snntorch import functional as SF
scaler = None
import math

torch.pi = math.pi
Epoch = 100
batch_size = 4
num_steps = 20

def train_model(net, train_loader, optimizer, device):
    start_time = time.time()
    net.train()
    train_loss = 0
    train_acc = 0
    snn_out = []

    for x1, x2, label in tqdm(train_loader):
        optimizer.zero_grad()
        x1, x2, label = x1.to(device), x2.to(device), label.to(device)

        with autocast():
            for step in range(num_steps):

                output = net(x1, x2)
                output = output.repeat(batch_size, 1)
                snn_out.append(output)




        loss = F.cross_entropy(output.float(), label.long())
        loss.backward(retain_graph=True)

        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            train_loss += loss.item()
            output_argmax = output.argmax(1).cpu().detach().numpy()
            label = label.cpu().detach().numpy()

            train_acc += accuracy_score(output_argmax, label)
            train_f1 = f1_score(label, output_argmax, average='micro')

    train_time = time.time() - start_time
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    return train_loss, train_acc, train_f1, train_time


def test_model(net, test_loader, optimizer, device):
    start_time = time.time()
    net.eval()

    test_loss = 0
    test_acc = 0

    for x1, x2, label in tqdm(test_loader):
        optimizer.zero_grad()
        x1, x2, label = x1.to(device), x2.to(device), label.to(device)



        with autocast():
            output = net(x1, x2)
            output = output.repeat(batch_size, 1)

        loss = F.cross_entropy(output.float(), label.long())
        loss.backward(retain_graph=True)

        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            test_loss += loss.item()
            output_argmax = output.argmax(1).cpu().detach().numpy()
            label = label.cpu().detach().numpy()

            test_acc += accuracy_score(output_argmax, label)
            test_f1 = f1_score(label, output_argmax, average='micro')

    test_time = time.time() - start_time
    test_loss /= len(test_loader)
    test_acc /= len(test_loader)

    return test_loss, test_acc, test_f1, test_time

