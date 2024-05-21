# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import my_dataset
from captcha_cnn_model import CNN
import os

# Hyper Parameters
num_epochs = 30
batch_size = 100
learning_rate = 0.001

def main():
    cnn = CNN()
    cnn.train()
    print('init net')
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    # Train the Model
    train_dataloader = my_dataset.get_train_data_loader()
    if os.path.exists("./model.pkl"):
        # 如果模型文件存在，则加载模型
        print("Model found, loading...")
        cnn = torch.load("./model.pkl")  # 假设model是一个可以直接加载的PyTorch模型
        cnn.train() # 设置模型为评估模式（对于训练好的模型进行推理）
        # 如果你的模型是在多GPU上训练的，并且你现在在单个GPU或CPU上运行，你可能还需要指定map_location参数
        # 例如：model = torch.load(model_path, map_location=torch.device('cpu'))

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_dataloader):
            images = Variable(images)
            labels = Variable(labels.float())
            predict_labels = cnn(images)
            # print(predict_labels.type)
            # print(labels.type)
            loss = criterion(predict_labels, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 10 == 0:
                print("epoch:", epoch, "step:", i, "loss:", loss.item())
            if (i+1) % 100 == 0:
                torch.save(cnn.state_dict(), "./model.pkl")   #current is model.pkl
                print("save model")
        print("epoch:", epoch, "step:", i, "loss:", loss.item())
    torch.save(cnn.state_dict(), "./model.pkl")   #current is model.pkl
    print("save last model")

if __name__ == '__main__':
    main()


