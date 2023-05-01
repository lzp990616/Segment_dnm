


python unet_compare_train.py --model_name U_Net --log_name ./log/pic_U_Net.log --batch_size 6 --EPOCH 100 --LR 0.0001
python unet_compare_train.py --model_name SegNet --log_name ./log/pic_SegNet.log --batch_size 6 --EPOCH 100 --LR 0.0001
python unet_compare_train.py --model_name R2U_Net --log_name ./log/pic_R2U_Net.log --batch_size 6 --EPOCH 100 --LR 0.0001
python unet_compare_train.py --model_name AttU_Net --log_name ./log/pic_AttU_Net.log --batch_size 6 --EPOCH 100 --LR 0.0001
python unet_compare_train.py --model_name R2AttU_Net --log_name ./log/pic_R2AttU_Net.log --batch_size 6 --EPOCH 100 --LR 0.0001
python unet_compare_train.py --model_name NestedUNet --log_name ./log/pic_NestedUNet.log --batch_size 6 --EPOCH 100 --LR 0.0001
python unet_compare_train.py --model_name transunet --log_name ./log/pic_transunet.log --batch_size 6 --EPOCH 100 --LR 0.0001
python unet_compare_train.py --model_name AAUnet --log_name ./log/pic_AAUnet.log --batch_size 6 --EPOCH 100 --LR 0.0001
