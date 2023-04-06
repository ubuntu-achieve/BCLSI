# To change the number of Swin Unet channels, you need to adjust the IN_CHANS entry in the configuration file configs\swin_tiny_patch4_window4_128_lite.yaml
# 改变Swin Unet通道数需要调整配置文件configs\swin_tiny_patch4_window4_128_lite.yaml中的IN_CHANS项
python Swin_Unet_Train.py --dataset Synapse --cfg configs/swin_tiny_patch4_window4_128_lite.yaml --max_epochs 150 --img_size 128 --base_lr 0.05 --batch_size 24
# Training Unet
# 训练Unet
python UNet_Train.py --data_dir ./datasets/ --snapshot_dir ./exp/ --num_channels 16 
python UNet_Train.py --data_dir ./datasets_16/ --snapshot_dir ./exp_16/ --num_channels 14 
shutdown