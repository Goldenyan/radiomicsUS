import radiomics
from radiomics import featureextractor
import numpy as np
import pandas as pd
import SimpleITK as sitk
import scipy.io as scio
from scipy import signal
import os
import openpyxl


pixelType = sitk.sitkInt8                                                               #定义nii格式的编码类型Int8
data = pd.DataFrame()
Path = os.getcwd()                                                                      #全体图像数据路径


folderLists = np.arange(4)                                                              #共分为4类数据
fileLists = [12, 41, 20, 12]                                                            #每类下病例数目

for folderList in folderLists:
    folder = folderList + 1
    folderPath = Path + '\\%s\\' %folder                                                #病例文件夹路径
    for fileList in np.arange(fileLists[folderList]):
        file = fileList + 1


        # 读取nii图像和对应mask
        imageName =  folderPath + '%s_HK3.nii.gz' %file
        maskName = folderPath + '%s_mask.nii.gz' %file

        #特征提取
        extractor = featureextractor.RadiomicsFeatureExtractor('radiomicsParams.yaml')                        #实例化特征提取类
        featureVector = extractor.execute(imageName, maskName)                          #调用特征提取函数
        print(len(featureVector))

        #输出特征结果
        #for featureName in featureVector.keys():
        #    print("Computed %s: %s" % (featureName, featureVector[featureName]))

        #保存特征结果
        data_add = pd.DataFrame.from_dict(featureVector.values()).T
        data_add.columns = featureVector.keys()
        data = pd.concat([data, data_add])

        print('The Features of Folder %s Patients %s has been executed' %(folder, file))



#保存成excel
data.to_excel(Path + '\\feature_HK3.xlsx')
print('All Patients have been executed')
