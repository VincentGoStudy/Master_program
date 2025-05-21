import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef, roc_curve, auc, roc_auc_score
from imblearn.over_sampling import RandomOverSampler
import seaborn as sns
import matplotlib.pyplot as plt
from monai.transforms import Compose
from Data_loading import read_mg

# 定义设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# 定义模型
# Cross Attention 模块
class CrossAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=4):
        super(CrossAttention, self).__init__()
        # 定义多头注意力机制
        self.multihead_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)

    def forward(self, query, key, value):
        # Cross Attention 的输入是 query, key 和 value
        attn_output, _ = self.multihead_attn(query, key, value)
        return attn_output

# CustomResNet50 模型
class CustomResNet50(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet50, self).__init__()
        self.num_classes = num_classes

        # 定义 ResNet50 模型
        self.base = models.resnet50(pretrained=True)
        self.resnet_feature_extractor = nn.Sequential(*list(self.base.children())[:-2])  # 去掉分类层，只保留特征提取部分

        # Cross Attention 模块，处理 2048 维度特征
        self.cross_attention = CrossAttention(feature_dim=2048)

        # 替换 ResNet50 的全连接层以适应自定义分类任务
        self.fc = nn.Linear(self.base.fc.in_features, num_classes)  # 将全连接层的输出维度设置为 3（num_classes）

    def forward(self, cc_image, mlo_image):
        # 提取 CC 和 MLO 图像的特征
        cc_features = self.resnet_feature_extractor(cc_image)  # [B, 2048, H, W]
        mlo_features = self.resnet_feature_extractor(mlo_image)  # [B, 2048, H, W]

        # 将特征调整为 [seq_length, batch_size, embed_dim] 以适应 MultiheadAttention
        cc_features = cc_features.flatten(2).permute(2, 0, 1)  # [B, C, H, W] -> [H*W, B, C]
        mlo_features = mlo_features.flatten(2).permute(2, 0, 1)  # [B, C, H, W] -> [H*W, B, C]

        # 通过 Cross Attention 融合特征
        fused_features = self.cross_attention(cc_features, mlo_features, mlo_features)  # 使用 MLO 作为 Key 和 Value
        fused_features = fused_features.permute(1, 0, 2).mean(dim=1)  # [H*W, B, C] -> [B, C]

        # 使用新的全连接层进行分类
        output = self.fc(fused_features)  # [B, num_classes]，在此为 [B, 3]

        return output


# class CustomResNet50(nn.Module):
#     def __init__(self, num_classes):
#         super(CustomResNet50, self).__init__()
#         self.num_classes = num_classes
#         self.base = models.resnet50(pretrained=True)
#
#         # 定义 MLP 层，逐步减少特征维度
#         self.base.fc = nn.Sequential(
#             # nn.Dropout(0.2),
#             nn.Linear(self.base.fc.in_features, num_classes)
#             # nn.Linear(self.base.fc.in_features, 1024),  # 2048 -> 1024
#             # nn.ReLU(),
#             # nn.Linear(1024, 512),  # 1024 -> 512
#             # nn.ReLU(),
#             # nn.Linear(512, 256),  # 512 -> 256
#             # nn.ReLU(),
#             # nn.Linear(256, 128),  # 256 -> 128
#             # nn.ReLU(),
#             # nn.Linear(128, num_classes)  # 128 -> num_classes
#         )
#
#     def forward(self, x):
#         return self.base(x)

# 数据增强
transform = Compose([
    transforms.ToPILImage(),
    transforms.Resize(size=(512, 512)),
    transforms.RandomApply([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-30, 30)),
        transforms.ColorJitter(brightness=(0.9, 1.1), contrast=(0.9, 1.1), saturation=(0.9, 1.1), hue=(-0.1, 0.1)),
    ], p=0.5),
    # transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    transforms.ToTensor(),
    # 加入归一化
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 定义数据集和路径
data_path = "data_new/data/preprocessed_MG"
# data_path_1 = "data/data1/preprocessed_MG"
# data_path_2 = "data/data2/preprocessed_MG"
# data_path_3 = "data/data3/processed_mg/preprocessed_cancer_1"
# data_path_4 = "data/data3/processed_mg/preprocessed_cancer_2"
# data_path_5 = "data/data4/preprocessed_mg"
label_path = "../file/all_label.csv"
model_path = "model/resnet50/mg3_resnet50.pth"
save_dir = "../data/mg_model_input"
output_path = 'result/new/mg3/resnet50/mg3_prob.csv'
output_dir = "result/new/mg3/resnet50"
paired_id_path = "../file/paired_id.csv"

# 读取标签
label_df = pd.read_csv(label_path)
file_ids = label_df['id'].values
labels = label_df['label'].values
side = label_df['side'].values

# 重新编码标签
def relabel(labels):
    new_labels = []
    for label in labels:
        if label in [0, 1]:
            new_labels.append(0)  # HR+/HER2-
        elif label in [2, 3]:
            new_labels.append(1)  # HER2+
        elif label == 4:
            new_labels.append(2)  # TN
    return np.array(new_labels)


new_labels = relabel(labels)


# 查找符合条件的图像路径
def find_image_paths(data_path, file_id, side):
    if side == 1:
        # 右侧图像：优先查找 RMLO/R_MLO 和 RCC/R_CC
        search_patterns = [
            os.path.join(data_path, f"*{file_id}*RMLO*.png"),
            os.path.join(data_path, f"*{file_id}*R_MLO*.png"),
            os.path.join(data_path, f"*{file_id}*RCC*.png"),
            os.path.join(data_path, f"*{file_id}*R_CC*.png")
        ]
    else:
        # 左侧图像：优先查找 LMLO/L_MLO 和 LCC/L_CC
        search_patterns = [
            os.path.join(data_path, f"*{file_id}*LMLO*.png"),
            os.path.join(data_path, f"*{file_id}*L_MLO*.png"),
            os.path.join(data_path, f"*{file_id}*LCC*.png"),
            os.path.join(data_path, f"*{file_id}*L_CC*.png")
        ]

    # 根据优先级搜索符合条件的图片路径
    file_paths = []
    for pattern in search_patterns:
        file_paths.extend(glob.glob(pattern))

    # 如果没有找到 RMLO/R_MLO 和 RCC/R_CC 或 LMLO/L_MLO 和 LCC/L_CC 的图像，再进行宽松的查找 MLO 和 CC
    if len(file_paths) == 0:
        if side == 1:
            # 右侧图像：查找 MLO 和 CC
            fallback_patterns = [
                os.path.join(data_path, f"*{file_id}*MLO*.png"),
                os.path.join(data_path, f"*{file_id}*CC*.png")
            ]
        else:
            # 左侧图像：查找 MLO 和 CC
            fallback_patterns = [
                os.path.join(data_path, f"*{file_id}*MLO*.png"),
                os.path.join(data_path, f"*{file_id}*CC*.png")
            ]

        # 根据宽松条件查找
        for pattern in fallback_patterns:
            file_paths.extend(glob.glob(pattern))

    # 返回找到的图像路径（可能是 RMLO/R_MLO 或 LMLO/L_MLO、RCC/R_CC 或 LCC/L_CC，也可能是 MLO 或 CC）
    return file_paths


# 查找所有路径中符合条件的图片
def find_all_image_paths(file_id, side_value):
    all_image_paths = (
            find_image_paths(data_path, file_id, side_value) +
            find_image_paths(data_path_1, file_id, side_value) +
            find_image_paths(data_path_2, file_id, side_value) +
            find_image_paths(data_path_3, file_id, side_value) +
            find_image_paths(data_path_4, file_id, side_value) +
            find_image_paths(data_path_5, file_id, side_value)
    )
    return all_image_paths


# 检查图片命名是否不同
def is_different_names(path1, path2):
    return os.path.basename(path1) != os.path.basename(path2)


# 将路径和标签组合成 DataFrame
file_paths = []
file_labels = []
file_ids_to_save = []

# 创建一个set来跟踪已保存的file_id，避免重复保存
saved_file_ids = set()

for file_id, label, side_value in zip(file_ids, new_labels, side):
    # 搜索所有数据路径中的图像
    all_image_paths = find_all_image_paths(file_id, side_value)

    if len(all_image_paths) >= 2:  # 至少找到 2 张图片
        # 按顺序找到最早的两张不同命名的图片
        selected_paths = []
        for i in range(len(all_image_paths)):
            for j in range(i + 1, len(all_image_paths)):
                if is_different_names(all_image_paths[i], all_image_paths[j]):
                    selected_paths = [all_image_paths[i], all_image_paths[j]]
                    break
            if selected_paths:
                break

        # 如果找到 2 张不同命名的图片，将其组合
        if selected_paths and file_id not in saved_file_ids:
            file_paths.append(selected_paths)
            file_labels.append(label)
            file_ids_to_save.append(file_id)
            saved_file_ids.add(file_id)
    else:
        # 如果没有找到至少两张图片，跳过此 file_id
        continue

# 创建 DataFrame
df = pd.DataFrame({'file_paths': file_paths, 'labels': file_labels, 'file_ids': file_ids_to_save})

# 读取file/paired_id.xlsx文件的第一列
paired_id_df = pd.read_csv(paired_id_path)
paired_ids = paired_id_df.iloc[:, 0].values  # 获取第一列的所有值

# 按照file_ids_to_save是否包含在paired_ids中划分数据集
test_df = df[df['file_ids'].isin(paired_ids)]
train_val_df = df[~df['file_ids'].isin(paired_ids)]

# print(train_val_df.file_paths[:5])
# print(train_val_df.labels[:5])
# print(train_val_df.file_ids[:5])
#
# # 保存train_val_df和过采样后的前五条记录为check.txt文件到output_dir
# check_df = pd.DataFrame({'file_paths': train_val_df.file_paths[:5], 'labels': train_val_df.labels[:5], 'file_ids': train_val_df.file_ids[:5]})
# check_df.to_csv(os.path.join(output_dir, 'check.txt'), index=False)

# 使用 RandomOverSampler 进行过采样
ros = RandomOverSampler()
train_val_paths_labels_ids_resampled, resampled_labels = ros.fit_resample(
    train_val_df[['file_paths', 'file_ids']], train_val_df['labels']
)

# 获取过采样后的文件路径、标签和文件ID
oversampled_file_paths = train_val_paths_labels_ids_resampled['file_paths'].values
oversampled_file_ids = train_val_paths_labels_ids_resampled['file_ids'].values
oversampled_labels = resampled_labels

# print(oversampled_file_paths[:5])
# print(oversampled_labels[:5])
# print(oversampled_file_ids[:5])

# 创建数据集
train_val_dataset = read_mg(oversampled_file_paths, oversampled_labels, oversampled_file_ids, transform=transform)
val_dataset = read_mg(test_df['file_paths'].values, test_df['labels'].values, test_df['file_ids'].values, transform=transform)
test_dataset = read_mg(test_df['file_paths'].values, test_df['labels'].values, test_df['file_ids'].values, transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_val_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 打印类别数量
# train_class_counts = train_df_resampled['labels'].value_counts().sort_index()
oversampled_class_counts = np.unique(oversampled_labels, return_counts=True)
val_class_counts = test_df['labels'].value_counts().sort_index()
test_class_counts = test_df['labels'].value_counts().sort_index()

print("训练集类别数量:")
for label, count in zip(oversampled_class_counts[0], oversampled_class_counts[1]):
    print(f'Class: {label}, Count: {count}')
print("验证集类别数量:")
for label, count in val_class_counts.items():
    print(f'Class: {label}, Count: {count}')
print("测试集类别数量:")
for label, count in test_class_counts.items():
    print(f'Class: {label}, Count: {count}')

# 生成数据分布直方图
def plot_combined_histogram(train_class_counts, val_class_counts, test_class_counts, title, file_name):
    plt.figure(figsize=(15, 8))
    x = np.arange(len(train_class_counts[0]))  # oversampled_class_counts[0] 是类别标签
    width = 0.3

    bars1 = plt.bar(x - width, train_class_counts[1], width=width,
                    label='Training')  # oversampled_class_counts[1] 是类别计数
    bars2 = plt.bar(x, val_class_counts.values, width=width, label='Val')
    bars3 = plt.bar(x + width, test_class_counts.values, width=width, label='Test')  # test_class_counts 是 pandas Series

    class_labels = ['HR+/HER2-', 'HER2+', 'TN']
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(title)
    plt.xticks(x, class_labels)
    plt.legend()

    # 在柱状图顶部添加数据标签
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, str(int(height)),
                     ha='center', va='bottom', fontsize=10)

    plt.savefig(os.path.join(output_dir, file_name))
    plt.show()

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plot_combined_histogram(oversampled_class_counts, val_class_counts, test_class_counts, 'mg Class Distribution', 'mg3_class_distribution.png')

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 使用指定的超参数值重新定义模型和优化器
learning_rate = 0.0001
num_epochs = 50
weight_decay = 1e-05

model = CustomResNet50(num_classes=3).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# 保存训练和验证精度、损失的列表
train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []

# 训练模型
# best_mcc = 0
best_auc = 0
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    total = 0
    correct = 0
    for images, labels, _ in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_accuracy = correct / total
    train_accuracies.append(train_accuracy)
    train_losses.append(train_loss / len(train_loader))

    # 验证模型
    model.eval()
    val_loss = 0
    total = 0
    correct = 0
    all_labels = []
    # all_predictions = []
    all_probs = []  # 用于存储输出概率
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            # all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())  # 获取概率分布
        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss / len(val_loader))

        # 计算 MCC
        # mcc = matthews_corrcoef(all_labels, all_predictions)
        # print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_loader):.4f}, '
        #       f'Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss / len(val_loader):.4f}, '
        #       f'Val Accuracy: {val_accuracy:.4f}, MCC: {mcc:.4f}')

        # if mcc > best_mcc:
        #     best_mcc = mcc
        #
        #     # 确保模型保存目录存在
        #     model_dir = os.path.dirname(model_path)
        #     if not os.path.exists(model_dir):
        #         os.makedirs(model_dir)
        #
        #     torch.save(model.state_dict(), model_path)
        #     print(f'Saved model with MCC: {best_mcc:.4f}')

        # 计算AUC
        auc_per_class = roc_auc_score(all_labels, all_probs, multi_class='ovr', average=None)  # 每类的AUC
        average_auc = auc_per_class.mean()  # 计算三类的平均AUC
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_loader):.4f}, '
              f'Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss / len(val_loader):.4f}, '
              f'Val Accuracy: {val_accuracy:.4f}, Average AUC: {average_auc:.4f}')

        # 判断是否保存模型
        if average_auc > best_auc:
            best_auc = average_auc

            # 确保模型保存目录存在
            model_dir = os.path.dirname(model_path)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            torch.save(model.state_dict(), model_path)
            print(f'Saved model with Average AUC: {best_auc:.4f}')

# 加载并测试最佳模型
model.load_state_dict(torch.load(model_path))
model.eval()

# 初始化一个空的 DataFrame，用于保存模型的结果
df_all_prob = pd.DataFrame(columns=['file_id', 'prob', 'label'])

# 初始化列表用于保存需要的数据
all_prob = []
all_labels = []
file_ids = []  # 用于保存文件 ID
all_predictions = []  # 用于保存预测结果
all_outputs = []  # 用于保存模型输出

# 初始化统计变量
total = 0  # 总样本数
correct = 0  # 预测正确的样本数

# 模型评估
with torch.no_grad():
    for images, labels, ids in test_loader:
        images, labels = images.to(device), labels.to(device)

        # 处理 ids 数据
        if isinstance(ids, tuple):
            ids = ids[0]  # 假设第一个元素是我们需要的 id
        if isinstance(ids, torch.Tensor):
            ids = ids.cpu().numpy()  # 转换为 numpy 数组
        elif isinstance(ids, str):
            ids = [ids]  # 将单个字符串转换为列表，以便处理
        elif not isinstance(ids, np.ndarray):
            print(f"Unexpected type of ids: {type(ids)}")
            continue  # 跳过错误类型的 id

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        # 保存每个批次的预测结果、模型输出、和其他数据
        all_predictions.extend(predicted.cpu().numpy())  # 保存预测结果
        all_outputs.extend(outputs.cpu().detach().numpy())  # 保存模型输出

        # 保存文件 ID、概率和标签
        for prob, label, file_id in zip(outputs.cpu().numpy(), labels.cpu().numpy(), ids):
            all_prob.append(','.join(map(str, prob)))  # 将概率转换为字符串以存储
            all_labels.append(label)
            file_ids.append(str(file_id))  # 保存文件 ID，并确保其为字符串

        # 计算预测正确的样本数
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# 计算测试集的准确率
test_accuracy = correct / total
print(f"Test Accuracy: {test_accuracy:.4f}")

# 将概率和标签保存到 DataFrame
df_all_prob = pd.DataFrame({
    'file_id': file_ids,
    'prob': all_prob,
    'label': all_labels
})

# 保存结果到 CSV 文件
# output_path = 'result/new/mg3/resnet50/mg3_prob.csv'
df_all_prob.to_csv(output_path, index=False)
print(f"数据已保存到 {output_path}")

# 输出分类报告
report = classification_report(all_labels, all_predictions,
                               target_names=['HR+/HER2-', 'HER2+', 'TN'], output_dict=False,
                               zero_division=0)
mcc = matthews_corrcoef(all_labels, all_predictions)

report_file_path = os.path.join(output_dir, 'mg3_resnet50_classification_report.txt')
with open(report_file_path, 'w') as f:
    f.write('\n\n')
    # 保存平均训练精度和验证精度
    f.write(f'Training Accuracy: {train_accuracy:.4f}\n')
    f.write(f'Validation Accuracy: {val_accuracy:.4f}\n\n')

    # 保存平均训练损失和验证损失
    f.write(f'Training Loss: {train_loss / len(train_loader):.4f}\n')
    f.write(f'Validation Loss: {val_loss / len(val_loader):.4f}\n\n')

    # 保存测试精度
    f.write(f'Test Accuracy: {test_accuracy:.4f}\n\n')
    f.write(f'MCC: {mcc:.4f}\n\n')
    f.write('Classification Report:\n')
    f.write(str(report))

# 混淆矩阵分类标签
class_labels = ['HR+/HER2-', 'HER2+', 'TN']
def plot_confusion_matrix(conf_matrix, class_labels, output_dir):
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_normalized, annot=True, fmt='.1%', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Ground truth')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'mg3_resnet50_confusion_matrix.png'))
    plt.show()

# 生成并绘制混淆矩阵
conf_matrix = confusion_matrix(all_labels, all_predictions)
plot_confusion_matrix(conf_matrix, class_labels, output_dir)

# 绘制训练和验证精度图表
def plot_accuracies(train_accuracies, val_accuracies, output_dir):
    plt.figure(figsize=(8, 6))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.savefig(os.path.join(output_dir, 'mg3_resnet50_accuracy_plot.png'))
    plt.show()

# 绘制训练和验证损失图表
def plot_losses(train_losses, val_losses, output_dir):
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(output_dir, 'mg3_resnet50_loss_plot.png'))
    plt.show()

plot_accuracies(train_accuracies, val_accuracies, output_dir)
plot_losses(train_losses, val_losses, output_dir)

# 绘制 ROC 曲线
n_classes = 3
class_labels = ['HR+/HER2-', 'HER2+', 'TN']
all_fpr = {i: [] for i in range(n_classes)}
all_tpr = {i: [] for i in range(n_classes)}
all_roc_auc = {i: [] for i in range(n_classes)}

def find_best_threshold(y_true, y_score):
    # Find the best threshold based on ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    best_threshold = thresholds[ix]
    return best_threshold

# 将标签二值化
y_test = label_binarize(all_labels, classes=range(len(class_labels)))
y_score = np.array(all_outputs)

# 计算每个类的最佳阈值
best_thresholds = []
for i in range(len(class_labels)):
    best_threshold = find_best_threshold(y_test[:, i], y_score[:, i])
    best_thresholds.append(best_threshold)
    print(f"Best threshold for class {class_labels[i]}: {best_threshold:.4f}")

# 基于最佳阈值生成预测结果
test_predictions = []
for i in range(len(y_score)):
    scores = y_score[i]
    predicted_label = np.argmax(scores >= best_thresholds)
    test_predictions.append(predicted_label)

# # 计算混淆矩阵
# cm = confusion_matrix(all_labels, test_predictions)
# cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#
# # 绘制混淆矩阵
# plt.figure(figsize=(10, 8))
# sns.heatmap(cm, annot=True, fmt=".1%", cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
# plt.xlabel('Predicted')
# plt.ylabel('Ground Truth')
# plt.title('mg3 Confusion Matrix')
# plt.savefig(os.path.join(output_dir, 'mg3_confusion_matrix.png'))
# plt.show()

# 用最佳模型在测试集上进行预测得到roc曲线
fpr = dict()
tpr = dict()
roc_auc = dict()

# 计算所有类的ROC曲线和AUC
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    # 打印调试信息
    print(f"Class {class_labels[i]}: fpr={fpr[i]}, tpr={tpr[i]}, roc_auc={roc_auc[i]:.2f}")

    # 保存fpr和tpr
    all_fpr[i].append(fpr[i])
    all_tpr[i].append(tpr[i])
    all_roc_auc[i].append(roc_auc[i])

# 画出roc曲线
plt.figure(figsize=(10, 10))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {class_labels[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('mg3 ROC Curve')
plt.legend()
plt.savefig(os.path.join(output_dir, 'mg3_roc_curve.png'))
plt.show()

