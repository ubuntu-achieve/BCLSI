import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.landslide_dataset import LandslideDataSet
class Args:
    num_classes = 2
    epsilon = 1e-14
    name_classes = ['Non-Landslide','Landslide']

def eval_image(predict,label,num_classes):
    index = np.where((label>=0) & (label<num_classes))
    predict = predict[index]
    label = label[index] 
    
    TP = np.zeros((num_classes, 1))
    FP = np.zeros((num_classes, 1))
    TN = np.zeros((num_classes, 1))
    FN = np.zeros((num_classes, 1))
    
    for i in range(0,num_classes):
        TP[i] = np.sum(label[np.where(predict==i)]==i)
        FP[i] = np.sum(label[np.where(predict==i)]!=i)
        TN[i] = np.sum(label[np.where(predict!=i)]!=i)
        FN[i] = np.sum(label[np.where(predict!=i)]==i)        
    
    return TP,FP,TN,FN,len(label)

if __name__ == "__main__":
    args = Args()
    model = torch.load("./exp/unet_16.pth")
    model = model.cuda()
    print('Testing..........')
    model.eval()
    TP_all = np.zeros((args.num_classes, 1))
    FP_all = np.zeros((args.num_classes, 1))
    TN_all = np.zeros((args.num_classes, 1))
    FN_all = np.zeros((args.num_classes, 1))
    n_valid_sample_all = 0
    F1 = np.zeros((args.num_classes, 1))
    test_loader = DataLoader(
                LandslideDataSet("./datasets_16/", "./datasets/true_test.txt", set="labeled", is_random=False),
                batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    interp = nn.Upsample(size=(128, 128), mode='bilinear')
    _tqdm_test = tqdm(test_loader)
    _tqdm_test.set_description_str("Test")
    for _, batch in enumerate(_tqdm_test):  
        image, label,_, name = batch
        label = label.squeeze().numpy()
        image = image.float().cuda()
        
        with torch.no_grad():
            pred = model(image)

        _,pred = torch.max(interp(nn.functional.softmax(pred,dim=1)).detach(), 1)
        pred = pred.squeeze().data.cpu().numpy()                       
                    
        TP,FP,TN,FN,n_valid_sample = eval_image(pred.reshape(-1),label.reshape(-1),args.num_classes)
        TP_all += TP
        FP_all += FP
        TN_all += TN
        FN_all += FN
        n_valid_sample_all += n_valid_sample

    OA = np.sum(TP_all)*1.0 / n_valid_sample_all
    P,R = 0,0
    for i in range(args.num_classes):
        P += TP_all[i]*1.0 / (TP_all[i] + FP_all[i] + args.epsilon)
        P1 = TP_all[i]*1.0 / (TP_all[i] + FP_all[i] + args.epsilon)
        R += TP_all[i]*1.0 / (TP_all[i] + FN_all[i] + args.epsilon)
        R1 = TP_all[i]*1.0 / (TP_all[i] + FN_all[i] + args.epsilon)
        F1[i] = 2.0*P1*R1 / (P1 + R1 + args.epsilon)
        if i==1:
            print('===>' + args.name_classes[i] + ' Precision: %.2f'%(P1 * 100))
            print('===>' + args.name_classes[i] + ' Recall: %.2f'%(R1 * 100))                
            print('===>' + args.name_classes[i] + ' F1: %.2f'%(F1[i] * 100))

    mF1 = np.mean(F1)
    print('===> mean F1: %.2f OA: %.2f'%(mF1*100,OA*100))
    print('===> mean P: %.2f OA: %.2f'%(P/2*100,OA*100))
    print('===> mean R: %.2f OA: %.2f'%(R/2*100,OA*100))