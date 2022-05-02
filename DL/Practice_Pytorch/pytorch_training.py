from doctest import OutputChecker
from tkinter import Variable
import torch.nn as nn
import torch.nn.functional as F


### Model Design
# class Model(nn.Module):
#     def __init__(self):
#         super(network_name, self).__init__()
#         # network layer
#     def forward(self, x):
#         # forward propagation
#         return output
    
# model = Model()
# output = model(data)

### Loss Functions, Optimizer
# import torch.nn as nn
# import torch.optim as optim

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

### Training
# CUDA = False
# epoch_size = 1000
# for epoch in range(epoch_size):
#     train_loss = 0.0
    
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = Variable(data), Variable(target)
        
#         if CUDA:
#             data, target = data.cuda(), target.cuda()
            
#         # clear gradient
#         optimizer.zero_grad()

#         # Forward propagation
#         output = model(data) 
#         loss = criterion(output, target) 

#         # Calculate gradients
#         loss.backward()

#         # Update parameters
#         optimizer.step()

#         predicted = torch.max(output.data, 1)[1]
#         train_loss += loss.item()

### Save model
# # 只儲存模型的權重
# torch.save(model.state_dict(), 'model_weights.pth')
# # 讀取權重
# model.load_state_dict(torch.load('model_weights.pth'))
# # 完整儲存整個模型
# torch.save(model, 'model.pth')

### TensorBoard
# from torch.utils.tensorboard import SummaryWriter
# log_dir = "./log_dir/"
# writer = SummaryWriter(log_dir)
# # 寫入資料
# writer.add_scalar(tag, scalar_value)
# writer.add_histogram(tag, values)
# writer.add_image(tag, img_tensor)

### 使用以下指令開啟 tensorboard
# tensorboard --logdir="./log_dir"