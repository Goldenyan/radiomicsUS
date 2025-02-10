import numpy as np
import pandas as pd
import SimpleITK as sitk
import scipy.io as scio
from scipy import signal
import os

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
        rawImage_meta = scio.loadmat(filePath + '.mat')                                 #加载图像
        try :
            rawImage_arr = rawImage_meta.get('b_data0')                                     #保存图像矩阵
            rawImage_hilbert = signal.hilbert(rawImage_arr, None, 0)                    # 希尔伯特变换 0表示沿y轴方向
        except :
            rawImage_arr = rawImage_meta.get('b_data')
            rawImage_hilbert = signal.hilbert(rawImage_arr, None, 0)                    # 希尔伯特变换 0表示对列进行变换


        rawImage_env = abs(rawImage_hilbert)                                            #复数取模求包络
        niiImage_nlz = (pow(10, 38.0 / 20) - 1) * (rawImage_env / rawImage_env.max())   #标准化
        niiImage_log = 20 * np.log10(niiImage_nlz + 1)                                  #对数压缩
        niiImageSize = niiImage_log.shape



        #创建图像nii格式
        niiImage = sitk.Image(niiImageSize, pixelType)                                  #新建图像
        niiImage = sitk.GetImageFromArray(niiImage_log)                                 #数据矩阵填充新图像
        niiImage.SetSpacing((38.3 / 256, 500.0 * 1540 / 38000000))                      #空间对应信息，顺序x,y,z
        niiImage.SetOrigin((0.0, 0.0))                                                  #设置空间原点
        sitk.WriteImage(niiImage, filePath +'_2D.nii.gz')                               #保存nii图像格式路径与名称



        #超声mask mat转nii格式
        niiMask_meta = scio.loadmat(filePath + '_mask.mat')                             #加载ROI
        niiMask_arr = niiMask_meta.get('ROI')                                           #保存ROI矩阵
        niiMaskSize = niiMask_arr.shape


        #创建mask nii格式
        niiMask = sitk.Image(niiMaskSize, pixelType)
        niiMask = sitk.GetImageFromArray(niiMask_arr)
        niiMask.SetSpacing((38.3 / 256, 500.0 * 1540 / 38000000))
        niiMask.SetOrigin((0.0, 0.0))
        sitk.WriteImage(niiMask, filePath + '_2D_mask.nii.gz')                           #保存mask的路径与名称

        print('The Folder %s Patient %s has been transformed' %(folder, file))

print('All Patients have been done!')