# Segment_dataset
https://github.com/linhandev/dataset

# 对比网络
file:///home/zhangzm/lzp/paper_read/Yan_AFTer-UNet_Axial_Fusion_Transformer_UNet_for_Medical_Image_Segmentation_WACV_2022_paper.pdf

file:///home/zhangzm/%E4%B8%8B%E8%BD%BD/Breast-Lesions-Segmentation-using-Dual-level-UNet-DL-UNet.pdf



### Data_dir
‘’‘
├── data
│   ├── BUS 存放用于训练的图片
│   ├── Kvasir-SEG
│   ├── STU-Hospital-master


├── bus(Kvasir-SEG, STU-Hospital-master)
│   ├── data_mask
│       ├── images
│       ├── masks

│   ├── data (Only need to create data and data_mask)
│       ├── train 存放用于训练的图片
│       ├── trainannot 存放用于训练的图片标注
│       ├── val 存放用于验证的图片
│       ├── valannot 存放用于验证的图片标注
│       ├── test 存放用于测试的图片
│       ├── testannot 存放用于测试的图片标注
’‘’

### Data_process:
'python data_spilt.py  # Spilt the data. Select data by changing 'filepath' in code'

### K-folder_experiment:
'python k_folder.py --log_name "./log/log_name.log" --data_name bus --batch_size 8 --EPOCH 100 --LR 0.00005 --LOSSK 0.1 --DNM 1 --M 10'

### Comparative_experiment:
'python compare_kfold.py --log_name "./log/log_name.log" --data_name bus --model_name unet --batch_size 8 --EPOCH 100 --LR 0.00005'

### Pic_experiment:
'python unet_train.py --log_name "./log/log_name.log" --data_name bus --batch_size 8 --EPOCH 100 --LR 0.00005 --LOSSK 0.1 --DNM 1 --M 10'
'python unet_compare_train.py --log_name "./log/log_name.log" --data_name bus --batch_size 8 --EPOCH 100 --LR 0.00005'

### Pic_plot:
'python predicted.py'
'python predicted_other_model.py'
'predicted_other_model_pic.py  # If the number of pictures and the name of the model are changed, the file needs to be changed.' 


