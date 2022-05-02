import os
from numpy import concatenate
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from datetime import datetime
import csv

# Global Parameters
CUDA = torch.cuda.is_available()
batch_size = 16
num_epochs = 10
lr = 0.001
num_classes = 2



class basic_block(nn.Module):
    # 輸出通道乘的倍數
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride, downsample):
        super(basic_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 在 shortcut 時，若維度不一樣，要更改維度
        self.downsample = downsample 
    
    def forward(self, x):
        res = x
        
        out = self.conv1(x)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            res = self.downsample(x)
        
        out += res
        out = self.relu(out)
        
        return out

class bottleneck_block(nn.Module):
    # 輸出通道乘的倍數
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, downsample):
        super(bottleneck_block, self).__init__()      
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        # 在 shortcut 時，若維度不一樣，要更改維度
        self.downsample = downsample 
        
    def forward(self, x):
        res = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            res = self.downsample(x)

        out += res
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, net_block, layers, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self.net_block_layer(net_block, 64, layers[0])
        self.layer2 = self.net_block_layer(net_block, 128, layers[1], stride=2)
        self.layer3 = self.net_block_layer(net_block, 256, layers[2], stride=2)
        self.layer4 = self.net_block_layer(net_block, 512, layers[3], stride=2)

        self.avgpooling = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * net_block.expansion, num_classes)

        # Parameters initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def net_block_layer(self, net_block, out_channels, num_blocks, stride=1):
        downsample = None
        
        # 在 shortcut 時，若維度不一樣，要更改維度
        if stride != 1 or self.in_channels != out_channels * net_block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * net_block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * net_block.expansion))
        
        layers = []
        layers.append(net_block(self.in_channels, out_channels, stride, downsample))
        
        self.in_channels = out_channels * net_block.expansion
        
        for i in range(1, num_blocks):
            layers.append(net_block(self.in_channels, out_channels, 1, None))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpooling(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpooling(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        
        return x

def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    total_train = 0
    correct_train = 0
    train_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target) 
        
        if CUDA:
            data, target = data.cuda(), target.cuda()

        # clear gradient
        optimizer.zero_grad()

        # Forward propagation
        output = model(data) 
        loss = criterion(output, target) 

        # Calculate gradients
        loss.backward()

        # Update parameters
        optimizer.step()

        predicted = torch.max(output.data, 1)[1]
        total_train += len(target)
        correct_train += sum((predicted == target).float())
        train_loss += loss.item()

        if batch_idx % 100 == 0:
            print("Train Epoch: {}/{} [iter： {}/{}], acc： {:.6f}, loss： {:.6f}".format(
               epoch+1, num_epochs, batch_idx+1, len(train_loader),
               correct_train / float((batch_idx + 1) * batch_size),
               train_loss / float((batch_idx + 1) * batch_size)))
            
    train_acc_ = 100 * (correct_train / float(total_train))
    train_loss_ = train_loss / total_train
                    
    return train_acc_, train_loss_

def validate(valid_loader, model, criterion, epoch): 
    model.eval()
    total_valid = 0
    correct_valid = 0
    valid_loss = 0
    
    for batch_idx, (data, target) in enumerate(valid_loader):
        data, target = Variable(data), Variable(target) 
        
        if CUDA:
            data, target = data.cuda(), target.cuda()

        output = model(data)
        loss = criterion(output, target) 

        predicted = torch.max(output.data, 1)[1]
        total_valid += len(target)
        correct_valid += sum((predicted == target).float())
        valid_loss += loss.item()

        if batch_idx % 100 == 0:
            print("Valid Epoch: {}/{} [iter： {}/{}], acc： {:.6f}, loss： {:.6f}".format(
               epoch+1, num_epochs, batch_idx+1, len(valid_loader),
               correct_valid / float((batch_idx + 1) * batch_size),
               valid_loss / float((batch_idx + 1) * batch_size)))
            
    valid_acc_ = 100 * (correct_valid / float(total_valid))
    valid_loss_ = valid_loss / total_valid
                    
    return valid_acc_, valid_loss_ 
   
def test(test_loader, model, criterion): 
    model.eval()
    total_test = 0
    correct_test = 0
    test_loss = 0
    data = None
    target = None
    predicted_log_ = []
    for idx, (data, target) in enumerate(test_loader):
        data, target = Variable(data), Variable(target) 
        
        if CUDA:
            data, target = data.cuda(), target.cuda()

        output = model(data)
        loss = criterion(output, target) 

        predicted = torch.max(output.data, 1)[1]
        predicted_log_.append([target.tolist()[0], predicted.tolist()[0]])
        # print('{}: target={}, predict={}'.format(idx, target.tolist()[0], predicted.tolist()[0]))
        total_test += len(target)
        correct_test += sum((predicted == target).float())
        test_loss += loss.item()
        
    test_acc_ = 100 * (correct_test / float(total_test))
    test_loss_ = test_loss / total_test
    print('test info {}/{}'.format(correct_test, total_test))
                    
    return test_acc_, test_loss_, predicted_log_

def training_loop(model, criterion, optimizer, train_loader, valid_loader, num_epochs):
    # set objects for storing metrics
    total_train_loss = []
    total_valid_loss = []
    total_train_accuracy = []
    total_valid_accuracy = []
 
    # Train model
    for epoch in range(num_epochs):
        # training
        train_acc_, train_loss_ = train(train_loader, model, criterion, optimizer, epoch)
        total_train_loss.append(train_loss_)
        total_train_accuracy.append(train_acc_)

        # validation
        with torch.no_grad():
            valid_acc_, valid_loss_ = validate(valid_loader, model, criterion, epoch)
            total_valid_loss.append(valid_loss_)
            total_valid_accuracy.append(valid_acc_)

        print('==========================================================================')
        print("Epoch: {}/{}， Train acc： {:.6f}， Train loss： {:.6f}， Valid acc： {:.6f}， Valid loss： {:.6f}".format(
               epoch+1, num_epochs, 
               train_acc_, train_loss_,
               valid_acc_, valid_loss_))
        print('==========================================================================')

    print("====== END ==========")

    return total_train_loss, total_valid_loss, total_train_accuracy, total_valid_accuracy

def plot_result(total_train, total_valid, label):
    plt.plot(range(num_epochs), total_train, 'b-', label=f'Training_{label}')
    plt.plot(range(num_epochs), total_valid, 'g-', label=f'validation_{label}')
    plt.title(f'Training & Validation {label}')
    plt.xlabel('Number of epochs')
    plt.ylabel(f'{label}')
    plt.legend()
    plt.show()
    
def save_model(model, out_path, content='Full'):
    if not(os.path.isdir(out_path)):
        os.makedirs(out_path)
    ### Save model
    if content=='weight':
        # 只儲存模型的權重
        torch.save(model.state_dict(), os.sep.join([out_path, 'model_weights.pth']))
    else:
        
        # 完整儲存整個模型
        torch.save(model, os.sep.join([out_path, 'model.pth']))
        torch.save(model.state_dict(), os.sep.join([out_path, 'model_weights.pth']))
        
    # 讀取權重
    # model.load_state_dict(torch.load('model_weights.pth'))

def main(do_training=True, model_path='', valid_data_path=''):
    
    device = torch.device('cuda' if CUDA else 'cpu')
    print(f'Device = {device}')
    
    
    if do_training:
        # ResNet18
        # model = ResNet(basic_block, [2, 2, 2, 2], num_classes)
        # ResNet50
        model = ResNet(bottleneck_block, [3, 4, 6, 3], num_classes)

        if CUDA:
            model = model.cuda()
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Transform
        my_transform = transforms.Compose(
            [
                transforms.Resize(size=(227, 227)),
                transforms.CenterCrop(224),
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, )),
            ]
        )
        
        # Data
        train_data_path = r'C:\Users\YuFamily\Documents\Will\Project\_DataSets\dogsandcats_500\training_set'
        valid_data_path = r'C:\Users\YuFamily\Documents\Will\Project\_DataSets\dogsandcats_500\test_set'
        t1 = datetime.now()
        print('Transform...training data')
        train_dataset = datasets.ImageFolder(root=train_data_path, transform=my_transform)
        t2 = datetime.now()
        print(f'spend {t2-t1}')
        print('Transform...testing data')
        valid_dataset = datasets.ImageFolder(root=valid_data_path, transform=my_transform)
        t1 = datetime.now()
        print(f'spend {t1-t2}')
        print('Loading...training data')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        t2 = datetime.now()
        print(f'spend {t2-t1}')
        print('Loading...test data')
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        t1 = datetime.now()
        print(f'spend {t1-t2}')
        
        t1 = datetime.now()
        print('Start training')
        total_train_loss, total_valid_loss, total_train_accuracy, total_valid_accuracy = training_loop(model, criterion, optimizer, train_loader, valid_loader, num_epochs)
        t2 = datetime.now()
        print(f'finished... spend{t2-t1}')
        plot_result(total_train_loss, total_valid_loss, 'loss')
        plot_result(total_train_accuracy, total_valid_accuracy, 'accuracy')
        
        save_model(model, r'C:\Users\YuFamily\Documents\Will\Project\_DataSets\dogsandcats_500\resnet18')
    else:
        model = torch.load(model_path)
        criterion = nn.CrossEntropyLoss()
        # Transform
        my_transform = transforms.Compose(
            [
                transforms.Resize(size=(227, 227)),
                transforms.CenterCrop(224),
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, )),
            ]
        )
        test_dataset = datasets.ImageFolder(root=valid_data_path, transform=my_transform)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        filename_list = []
        for r,d,fs in os.walk(valid_data_path):
            for f in fs:
                filename_list.append(os.sep.join([r, f]))
            
        with torch.no_grad():
            test_acc_, test_loss_, predicted_log_= test(test_loader, model, criterion)

            print('==========================================================================')
            print(" Test acc： {:.6f}， Valid loss： {:.6f}".format(
                test_acc_, test_loss_))
            print('==========================================================================')


        # print(len(filename_list[:12]))
        # print(predicted_log_)
        result = [[f, str(p1), str(p2)] for f, [p1, p2] in zip(filename_list, predicted_log_)]
        # print(result)
        with open(r'C:\Users\YuFamily\Documents\Will\Project\_DataSets\dogsandcats_500\resnet18\result.csv', mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Filename', 'Target', 'Predicted'])
            writer.writerows(result)
            writer.writerow('')
            writer.writerow('================')
            writer.writerow(['Test Acc', str(test_acc_)])
        
            
    return 
    
if __name__ == '__main__':
    # main()
    main(False, r'C:\Users\YuFamily\Documents\Will\Project\_DataSets\dogsandcats_500\resnet18\model.pth', valid_data_path=r'C:\Users\YuFamily\Documents\Will\Project\_DataSets\dogsandcats_500\test_set')