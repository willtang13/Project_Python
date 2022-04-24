# %%
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

class Dataset(object):
    def __init__(self):
        # 定義初始化參數
        # 讀取資料集路徑
        pass
    def __getitem__(self, index):
        # 讀取每次迭代的資料集中第 idx  資料
        # 進行前處理 (torchvision.Transform 等)
        # return 資料和 label
        pass
    def __len__(self):
        # 計算資料集總共數量
        # return 資料集總數
        pass


# %%

dataset_path = r'C:\Users\YuFamily\Documents\Will\Project\_DataSets\dogsandcats\dataset\training_set'
image_folder = ImageFolder(dataset_path, transform=None, target_transform=None)
print(image_folder.class_to_idx)
# %% 下載 CIFAR10
cifar_data = torchvision.datasets.CIFAR10(root=r'C:\Users\YuFamily\Documents\Will\Project\_DataSets', train=True, download=True)
# %% 建立 DataLoader 
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms

train_transform = transforms.Compose([
                  transforms.Resize((256, 256)),
                  transforms.ToTensor(), 
                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

dataset_path = r'C:\Users\YuFamily\Documents\Will\Project\_DataSets\dogsandcats_100\dataset'
image_folder = ImageFolder(dataset_path, transform=train_transform, target_transform=None)

data_loader = DataLoader(dataset=image_folder, batch_size=100, shuffle=True, num_workers=2)

for batch_idx, (data, target) in enumerate(data_loader):
    print('data: ', data)
    print('label: ', target)
# %%
