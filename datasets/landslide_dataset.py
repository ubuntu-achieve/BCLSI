import h5py
import random
import numpy as np
from PIL import Image, ImageEnhance

import torch
from torch.utils import data
from torch.utils.data import DataLoader

class LandslideDataSet(data.Dataset):
    def __init__(self, data_dir, list_path, max_iters=None,set='label', is_random=True, n_channels=14):
        self.list_path, self.is_random, self.n_channels = list_path, is_random, n_channels
        self.mean = [-0.4914, -0.3074, -0.1277, -0.0625, 0.0439, 0.0803, 0.0644, 0.0802, 0.3000, 0.4082, 0.0823, 0.0516, 0.3338, 0.7819]
        self.std = [0.9325, 0.8775, 0.8860, 0.8869, 0.8857, 0.8418, 0.8354, 0.8491, 0.9061, 1.6072, 0.8848, 0.9232, 0.9018, 1.2913]
        self.set = set
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        
        if not max_iters==None:
            n_repeat = int(np.ceil(max_iters / len(self.img_ids)))
            self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters-n_repeat*len(self.img_ids)]

        self.files = []

        if set=='labeled':
            for name in self.img_ids:
                img_file = data_dir + name
                label_file = data_dir + name.replace('img','mask').replace('image','mask')
                self.files.append({
                    'img': img_file,
                    'label': label_file,
                    'name': name
                })
        elif set=='unlabeled':
            for name in self.img_ids:
                img_file = data_dir + name
                self.files.append({
                    'img': img_file,
                    'name': name
                })
            
    def __len__(self):
        return len(self.files)

    def normalization(self, x):
        # 将像素归一化方便显示
        return (x - np.min(x))/(np.max(x) - np.min(x))

    def back_normalization(self, x, x_min_max):
        x_min, x_max = x_min_max
        return x*(x_max-x_min)+x_min

    def get_min_max(self, x):
        return np.min(x), np.max(x)

    def random_im(self, image):

        random_min, random_max = 0.90, 1.10
        # random_min, random_max = 0.50, 1.50
        # random_min, random_max = 1, 1
        img = np.array(image)
        min_max = [self.get_min_max(img[:,:,i]) for i in range(1,4)][::-1]
        B, G, R = [self.normalization(i) for i in np.array_split(img, self.n_channels, axis=2)[1:4]]
        RGB = np.squeeze(np.stack([R,G,B], axis=2))
        # 记得乘255 unit8范围问题
        im=Image.fromarray(np.uint8(RGB*255)) # numpy 转 image类
        # 明亮度参数
        brightness = random.uniform(random_min, random_max)
        # 对比度参数
        contrast = random.uniform(random_min, random_max)
        # 颜色
        color = random.uniform(random_min, random_max)

        # 明亮度
        enhancer = ImageEnhance.Brightness(im)
        im = enhancer.enhance(brightness)
        # 对比度
        enhancer = ImageEnhance.Contrast(im)
        im = enhancer.enhance(contrast)
        # 颜色
        enhancer = ImageEnhance.Color(im)
        im = enhancer.enhance(color)
        # # img 是 Image类
        img_np = np.array(im)
        # # (128,128,3) 倒回去
        # 拆分RGB
        R, G, B = [i/255 for i in np.array_split(img_np, 3, axis=2)]
        R, G, B = [self.back_normalization(i, min_max[index]) for index, i in enumerate([R,G,B])]
        # 注意是BGR
        BGR = np.squeeze(np.stack((B,G,R),axis=2))
        img[:,:,1:4] = BGR
        return img  # 扰动后

    def __getitem__(self, index):
        datafiles = self.files[index]
        
        if self.set=='labeled':
            with h5py.File(datafiles['img'], 'r') as hf:
                image = hf['img'][:]
            with h5py.File(datafiles['label'], 'r') as hf:
                label = hf['mask'][:]
            name = datafiles['name']
            
            if random.randint(1,10) > 7 and self.is_random:
                image = self.random_im(image)
            image = np.asarray(image, np.float32)
            label = np.asarray(label, np.float32)
            image = image.transpose((-1, 0, 1))
            size = image.shape

            for i in range(len(self.mean)):
                image[i,:,:] -= self.mean[i]
                image[i,:,:] /= self.std[i]

            return image.copy(), label.copy(), np.array(size), name

        else:
            with h5py.File(datafiles['img'], 'r') as hf:
                image = hf['img'][:]
            name = datafiles['name']
                
            image = np.asarray(image, np.float32)
            image = image.transpose((-1, 0, 1))
            size = image.shape

            for i in range(len(self.mean)):
                image[i,:,:] -= self.mean[i]
                image[i,:,:] /= self.std[i]

            return image.copy(), np.array(size), name

root_path = "./dataset"
list_path = "./dataset/train.txt"
if __name__ == '__main__':
    
    train_dataset = LandslideDataSet(data_dir=root_path, list_path=list_path)
    train_loader = DataLoader(dataset=train_dataset,batch_size=1,shuffle=True,pin_memory=True)

    channels_sum,channel_squared_sum = 0,0
    num_batches = len(train_loader)
    for data,_,_,_ in train_loader:
        channels_sum += torch.mean(data,dim=[0,2,3])   
        channel_squared_sum += torch.mean(data**2,dim=[0,2,3])       

    mean = channels_sum/num_batches
    std = (channel_squared_sum/num_batches - mean**2)**0.5
    print(mean,std) 
    #[-0.4914, -0.3074, -0.1277, -0.0625, 0.0439, 0.0803, 0.0644, 0.0802, 0.3000, 0.4082, 0.0823, 0.0516, 0.3338, 0.7819]
    #[0.9325, 0.8775, 0.8860, 0.8869, 0.8857, 0.8418, 0.8354, 0.8491, 0.9061, 1.6072, 0.8848, 0.9232, 0.9018, 1.2913]
