import os
import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.landslide_dataset import LandslideDataSet

name_classes = ['Non-Landslide','Landslide']
epsilon = 1e-14

def importName(modulename, name):
    """ Import a named object from a module in the context of this function.
    """
    try:
        module = __import__(modulename, globals(), locals(  ), [name])
    except ImportError:
        return None
    return vars(module)[name]


def make_mask_as_h5(model_path="./exp/unet_14.pth", datasets_path="./datasets/", data_list="./datasets/true_test.txt", save_dir="./predict", is_make_mask=True, n_channel=14):


    model = torch.load(model_path)
    model = model.cuda()

    print('Generating mask..........')
    model.eval()
   # main 修改batch_size和读取线程数
    test_loader = DataLoader(
            LandslideDataSet(datasets_path, data_list, set="unlabeled", is_random=False, n_channels=n_channel),
            batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    for index, batch in enumerate(test_loader):  
        image, _, name = batch
        image = image.float().cuda()
        name = name[0].split('.')[0].split('/')[-1].replace('image','mask')
        print(index+1, '/', len(test_loader), ': Testing ', name)  
        
        with torch.no_grad():
            pred = model(image)
        
        interp = nn.Upsample(size=(128, 128), mode='bilinear')
        _,pred = torch.max(interp(nn.functional.softmax(pred,dim=1)).detach(), 1)
        pred = pred.squeeze().data.cpu().numpy().astype('uint8')
        if is_make_mask:
            with h5py.File(os.path.join(datasets_path, "TestData/mask/")+name+'.h5','w') as hf:
                hf.create_dataset('mask', data=pred)
        else:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            plt.imsave(os.path.join(save_dir, str(index)+".png"), pred, cmap="gray")
            # Image.fromarray(np.uint8(pred*255)).save(os.path.join(save_dir, str(index)+".png"))
    del model
 
if __name__ == '__main__':
    # make_mask_as_h5(model_path="./exp/unet_14.pth", datasets_path="./datasets/", data_list="./datasets/true_test.txt", save_dir="pre_unet_14", is_make_mask=False)
    # make_mask_as_h5(model_path="./exp/unet_16.pth", datasets_path="./datasets_16/", data_list="./datasets/true_test.txt", save_dir="pre_unet_16", is_make_mask=False, n_channel=16)
    make_mask_as_h5(model_path="./exp/swin_unet_14.pth", datasets_path="./datasets/", data_list="./datasets/true_test.txt", save_dir="pre_swin_unet_14", is_make_mask=False)
    # make_mask_as_h5(model_path="./exp/swin_unet_16.pth", datasets_path="./datasets_16/", data_list="./datasets/true_test.txt", save_dir="pre_swin_unet_16", is_make_mask=False, n_channel=16)
