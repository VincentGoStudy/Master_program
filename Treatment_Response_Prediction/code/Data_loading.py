import os
import SimpleITK as sitk
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class read_mr(Dataset):
    def __init__(self, file_paths, labels, file_ids, transform=None, save_dir=None):
        self.file_paths = file_paths
        self.labels = labels
        self.file_ids = file_ids  # 存储文件 ID
        self.transform = transform
        self.save_dir = save_dir

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_path = self.file_paths[idx]
        labels = torch.tensor(self.labels[idx], dtype=torch.long)  # 将标签转换为 LongTensor 类型
        file_id = self.file_ids[idx]  # 获取对应的文件 ID

        # 读取 MRI 数据
        img = sitk.ReadImage(file_path)
        img_array = sitk.GetArrayFromImage(img)
        img_array = img_array[np.newaxis, ...]

        # 进行预处理
        if self.transform:
            img_array = self.transform(img_array)

        # 转换为 PyTorch 张量
        image = torch.from_numpy(img_array).float()

        if self.save_dir:
            # 确保保存目录存在
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

            # 保存增强后的图像为 .nrrd 文件
            save_path = os.path.join(self.save_dir, f"{os.path.basename(file_path).replace('.nii.gz', '_enhanced.nrrd')}")
            enhanced_image = sitk.GetImageFromArray(image.numpy().squeeze())
            enhanced_image.SetDirection(img.GetDirection())
            enhanced_image.SetSpacing(img.GetSpacing())
            enhanced_image.SetOrigin(img.GetOrigin())
            sitk.WriteImage(enhanced_image, save_path)

        # 返回数据和标签
        return image, labels, file_id






