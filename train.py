import torch
import torch.nn as nn
import torch.optim as optim
from model import MulitNet
from dataLoader import TrainDS, TestDS

class_num = 16

x = torch.randn(1, 1, 30, 25, 25)
net = MulitNet(class_num=class_num)

out = net(x)
print(out.shape)


train_loader = torch.utils.data.DataLoader(dataset=TrainDS(), batch_size=1024, shuffle=True, num_workers=0)
test_loader  = torch.utils.data.DataLoader(dataset=TestDS(),  batch_size=1024, shuffle=False, num_workers=0)

print(len(train_loader))
print(len(test_loader))

# 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 网络放到GPU上
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 开始训练
total_loss = 0
for epoch in range(200):
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        # 优化器梯度归零
        optimizer.zero_grad()
        # 正向传播 +　反向传播 + 优化 
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch%10 == 0 :
        print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' %(epoch + 1, total_loss/(epoch+1), loss.item()))

print('Finished Training')
