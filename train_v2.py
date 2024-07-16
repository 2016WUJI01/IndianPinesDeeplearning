import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse

from model_v2 import MulitNet
from dataLoader import TrainDS, TestDS
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from logger import get_logger


def train(model,optimizer,criterion,train_loader,test_loader,logger,epochs):
    # 开始训练
    total_loss = 0
    for epoch in range(10):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 优化器梯度归零
            optimizer.zero_grad()
            # 正向传播 +　反向传播 + 优化 
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch%10 == 0 :
            print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' %(epoch + 1, total_loss/(epoch+1), loss.item()))

    print('Finished Training')
    

    count = 0
    # 模型测试
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test =  outputs
            count = 1
        else:
            y_pred_test = np.concatenate( (y_pred_test, outputs) )

    # 生成分类报告
    classification = classification_report(TestDS().y_data, y_pred_test, digits=4)
    print(classification)
    logger.info(classification)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # cuda settings
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size of training.')
    parser.add_argument('--lr', type=float, default=5e-3,
                        help='Initial learning rate. default:[0.00001]')
    parser.add_argument('--wd', type=float, default=1e-4,
                        help='Weight decay (L2 loss on parameters). default: 5e-3')
    # Training parameter settings
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs to train.')
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_num = 16
    model = MulitNet(class_num=class_num)
    model = model.to(device)

    train_loader = torch.utils.data.DataLoader(dataset=TrainDS(), batch_size=1024, shuffle=True, num_workers=0)
    test_loader  = torch.utils.data.DataLoader(dataset=TestDS(),  batch_size=1024, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.wd)

    logger = get_logger('./log/lr'+str(args.lr)+'epoch'+str(args.epochs)+'_class.log')

    train(model,optimizer,criterion,train_loader,test_loader,logger,args.epochs)