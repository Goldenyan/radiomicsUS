import numpy as np
import pandas as pd
import SimpleITK as sitk
import scipy.io as scio
from scipy import signal
import os

sampleFrequency = 30000000
imageWidth = 38.2
soundSpeed = 1540
alineNumber = 256
pixelType = sitk.sitkInt8                                                               #定义nii格式的编码类型Int8
Path = os.getcwd()                                                                      #全体图像数据路径


folderLists = np.arange(4)                                                              #共分为4类数据
fileLists = [12, 41, 20, 12]                                                            #每类下病例数目

for folderList in folderLists:
    folder = folderList + 1
    folderPath = Path + '\\%s\\' %folder                                                #病例文件夹路径

    for fileList in np.arange(fileLists[folderList]):
        file = fileList + 1
        filePath = folderPath + '%s' %file                                              #病例路径

        #数据获取
        # rawImage_meta = scio.loadmat(filePath + '.mat')                                 #加载图像
        # try:
        #     rawImage_arr = rawImage_meta.get('b_data0')                                 #保存图像矩阵
        #     rawImage_hilbert = signal.hilbert(rawImage_arr, None, 0)                    # 希尔伯特变换 0表示沿y轴方向
        # except :
        #     rawImage_arr = rawImage_meta.get('b_data')
        #     rawImage_hilbert = signal.hilbert(rawImage_arr, None, 0)                    # 希尔伯特变换 0表示对列进行变换
        #
        # rawImage_RF = abs(rawImage_arr)
        # rawImage_env = abs(rawImage_hilbert)                                            #复数取模求包络
        # niiImageSize = rawImage_env.shape
        #
        #
        # # 对包络压缩以获取B-mode灰度图像
        # niiImage_nlz = (pow(10, 38.0 / 20) - 1) * (rawImage_env / rawImage_env.max())   #标准化
        # niiImage_log = 20 * np.log10(niiImage_nlz + 1)                                  #对数压缩

        # 创建BmodeLog/BmodeNor/RawEnv/RawRF图像nii格式
        # niiImage3D = np.reshape(rawImage_RF, (niiImageSize[0], niiImageSize[1], 1))
        # niiImage3D = niiImage3D.transpose(2, 0, 1)                                      #矩阵转置
        # niiImage3DSize = niiImage3D.shape                                               #获取image维度
        # print(niiImage3DSize)
        #
        # niiImage = sitk.Image(niiImage3DSize, pixelType)                                #新建图像
        # niiImage = sitk.GetImageFromArray(niiImage3D)                                   #数据矩阵填充新图像
        # niiImage.SetSpacing((imageWidth / alineNumber, 0.5* 1000 * soundSpeed / sampleFrequency, 1))                   #空间对应信息，顺序x,y,z
        # niiImage.SetOrigin((0.0, 0.0, 0.0))                                             #设置空间原点
        # sitk.WriteImage(niiImage, filePath +'_RF.nii.gz')                               #保存nii图像格式路径与名称



        #超声mask mat转nii格式
        niiMask_meta = scio.loadmat(filePath + '_mask.mat')                             #加载ROI
        niiMask_arr = niiMask_meta.get('ROI')                                           #保存ROI矩阵
        niiMaskSize = niiMask_arr.shape
        niiMask3D = np.reshape(niiMask_arr, (niiMaskSize[0], niiMaskSize[1], 1))       #将二维矩阵转为三维
        niiMask3D = niiMask3D.transpose(2, 0, 1)                                        #矩阵转置
        niiMask3DSize = niiMask3D.shape                                                 #获取mask维度，应于image相同


        # #创建mask nii格式
        # niiMask = sitk.Image(niiMask3DSize, pixelType)
        # niiMask = sitk.GetImageFromArray(niiMask3D)
        # niiMask.SetSpacing((imageWidth / alineNumber, 0.5* 1000 * soundSpeed / sampleFrequency, 1))
        # niiMask.SetOrigin((0.0, 0.0, 0.0))
        # sitk.WriteImage(niiMask, filePath + '_mask.nii.gz')                           #保存mask的路径与名称

        # 超声entropy/nakagami/HK mat转nii格式
        niiEntropy_meta = scio.loadmat(filePath + '_HK3.mat')  # 加载entropy图像
        niiEntropy_arr = niiEntropy_meta.get('ParaMap')  # 保存entropy矩阵
        niiEntropySize = niiEntropy_arr.shape
        niiEntropy3D = np.reshape(niiEntropy_arr, (niiEntropySize[0], niiEntropySize[1], 1))  # 将二维矩阵转为三维
        niiEntropy3D = niiEntropy3D.transpose(2, 0, 1)  # 矩阵转置
        niiEntropy3DSize = niiEntropy3D.shape  # 获取entropy维度

        # 创建entropy/nakagami图像 nii格式
        niiEntropy = sitk.Image(niiEntropy3DSize, pixelType)
        niiEntropy = sitk.GetImageFromArray(niiEntropy3D)
        niiEntropy.SetSpacing((imageWidth / alineNumber, 0.5* 1000 * soundSpeed / sampleFrequency, 1))
        niiEntropy.SetOrigin((0.0, 0.0, 0.0))
        sitk.WriteImage(niiEntropy, filePath + '_HK3.nii.gz')  # 保存entropy图像的路径与名称

        a = niiMask_arr * niiEntropy_arr
        exist = (a != 0)
        result = a.sum() / exist.sum()
        print(result)

        print('The Folder %s Patient %s has been transformed' %(folder, file))

print('All Patients have been done!')