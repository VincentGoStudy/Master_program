mg3_resnet50:
1.使用resnet50模型做三分类任务
2.训练集为非配对id文件
3.MG和MRI配对病人id作为独立测试集

mr3_resnet18:
1.使用3d resnet18模型做三分类任务
2.训练集为非配对id文件
3.MG和MRI配对病人id作为独立测试集

multi_resnet_maf:
使用两个模型得到的prob做自动权重结合得到双模态的prob进行预测

数据集使用：福州数据+ISPY2 (乳腺癌DCE-MRI)