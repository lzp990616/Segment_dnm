import os
import pdb

path = ('./data/Dataset_BUSI_malignant/Dataset_BUSI_with_GT/data_mask/images')


filepath ='./data/Dataset_BUSI/Dataset_BUSI_with_GT/Train_images/'
file_path_busi_m = './data/Dataset_BUSI_malignant/Dataset_BUSI_with_GT/'
filepath_cloth = './data/archive/Train_images/'

# 获取该目录下所有文件，存入列表中
fileList = os.listdir(path)
#pdb.set_trace()
#fileList.sort(key = lambda x:int(x[4:-4])) # 正则表达
fileList.sort(key = lambda x:int(x[11:-5])) # 正则表达

print(fileList)
n = 0
for i in fileList:
    # 设置旧文件名（就是路径+文件名）
    oldname = path + os.sep + fileList[n]  # os.sep添加系统分隔符

    # 设置新文件名
    newname = path + os.sep + 'malignant (' + str(n + 1) + ')' + '.png'

    os.rename(oldname, newname)  # 用os模块中的rename方法对文件改名
    print(oldname, '======>', newname)

    n += 1
