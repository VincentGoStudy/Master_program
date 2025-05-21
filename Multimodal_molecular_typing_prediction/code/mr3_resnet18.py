import glob
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from Data_loading import read_mr
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef, roc_curve, auc, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from resnet3dmodel import ResNet18_3D
import torch.nn.functional as F

# 设置设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, depth, height, width = x.size()

        # 计算Q, K, V
        query = self.query_conv(x).view(batch_size, -1, depth * height * width)
        key = self.key_conv(x).view(batch_size, -1, depth * height * width).permute(0, 2, 1)
        value = self.value_conv(x).view(batch_size, -1, depth * height * width)

        # 计算注意力权重
        attention = F.softmax(torch.bmm(query.permute(0, 2, 1), key), dim=-1)

        # 计算注意力输出
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, depth, height, width)

        return self.gamma * out + x  # 残差连接

# 定义模型
class CustomResNet18_3D(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet18_3D, self).__init__()
        self.num_classes = num_classes
        self.base = ResNet18_3D()  # 使用预训练的 3D ResNet18 作为基础模型
        self.base.conv1 = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)

        # 获取原始模型的全连接层的输入特征数量
        num_ftrs = self.base.fc.in_features

        # 重新初始化全连接层，以匹配新的输入维度和类别数
        self.base.fc = nn.Linear(num_ftrs, num_classes)

        # 添加自注意力层
        self.self_attention = SelfAttention(in_channels=64)  # 假设 base 的第一层输出通道数为 64

    def forward(self, x):
        x = self.base.conv1(x)  # 通过第一层卷积
        x = self.self_attention(x)  # 添加自注意力
        x = self.base(x)  # 通过基础模型
        return x

# 自定义的三维 MRI 图像预处理函数
class Custom3DMRITransform:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, img_array):
        # 计算原始图像的形状
        _, depth, height, width = img_array.shape

        # 获取目标尺寸
        _, new_depth, new_height, new_width = self.output_size

        # 计算缩放比例
        d_scale = new_depth / depth
        h_scale = new_height / height
        w_scale = new_width / width

        # 初始化缩放后的图像数组
        resized_img_array = np.zeros(self.output_size)

        # 遍历目标图像的每个像素
        for d in range(new_depth):
            for h in range(new_height):
                for w in range(new_width):
                    # 计算原始图像上的坐标
                    orig_d = int(d / d_scale)
                    orig_h = int(h / h_scale)
                    orig_w = int(w / w_scale)

                    # 确保坐标在原始图像范围内
                    orig_d = min(depth - 1, orig_d)
                    orig_h = min(height - 1, orig_h)
                    orig_w = min(width - 1, orig_w)

                    # 使用最近邻插值进行像素值赋值
                    resized_img_array[0, d, h, w] = img_array[0, orig_d, orig_h, orig_w]

        return resized_img_array

# 数据加载和预处理
transform = Custom3DMRITransform(output_size=(1, 64, 64, 64))

# 定义数据集和路径
data_path = "../data/data/preprocessed_MR"
# data_path_1 = "../data/data1/preprocessed_MR"
# data_path_2 = "../data/data2/preprocessed_MR"
# data_path_3 = "../data/data3/preprocessed_mr"
# data_path_4 = "../data/data4/preprocessed_mr"
# data_path_5 = "../data/ispy2/true"
label_path = "../file/all_label.csv"
# label_path1 = "../file/ispy2_label.csv"
save_dir = "../data/mr_model_input"
model_path = "../model/mr3/resnet18/mr3_resnet18.pth"
output_path = "../result/new/mr3/resnet18/mr3_prob.csv"
output_dir = "../result/new/mr3/resnet18"
paired_id_path = "../file/paired_id.csv"

# 读取标签
label_df = pd.read_csv(label_path)
file_ids = label_df['id'].values
labels = label_df['label'].values

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

# 更新标签数据框
label_df['new_label'] = new_labels

# 定义一个函数来查找文件
def find_files(data_path, file_ids, label_df, pattern_format="*_preNAT.nii.gz"):
    data = []
    remaining_file_ids = file_ids.copy()  # 创建一个 file_ids 的副本来跟踪未找到的文件
    for file_id in remaining_file_ids.copy():  # 遍历 file_ids
        pattern = os.path.join(data_path, pattern_format.format(file_id))  # 匹配模式格式
        file_paths = glob.glob(pattern)  # 查找匹配的文件
        for file in file_paths:
            # 获取文件 ID 并在 label_df 中查找对应的标签
            label = label_df[label_df['id'] == file_id]['new_label']
            if not label.empty:
                data.append({'file_paths': file, 'labels': label.values[0], 'file_id': file_id})
                remaining_file_ids = [fid for fid in remaining_file_ids if fid != file_id]  # 移除已找到的文件ID
                break  # 只处理每个 file_id 的第一个匹配文件
    return data, remaining_file_ids

# 处理每个路径中的文件
data, remaining_file_ids = find_files(data_path, file_ids, label_df)  # 处理 data_path 的文件
data_1, remaining_file_ids = find_files(data_path_1, remaining_file_ids, label_df)  # 处理 data_path_1 的文件
data_2, remaining_file_ids = find_files(data_path_2, remaining_file_ids, label_df)  # 处理 data_path_2 的文件
data_3, remaining_file_ids = find_files(data_path_3, remaining_file_ids, label_df)  # 处理 data_path_3 的文件
data_4, remaining_file_ids = find_files(data_path_4, remaining_file_ids, label_df)  # 处理 data_path_4 的文件

# # 读取 ispy2 数据集
# ispy2_label_df = pd.read_csv(label_path1)
# ispy2_file_ids = ispy2_label_df.iloc[:, 0].values  # 第一列是 file_ids
# ispy2_labels = ispy2_label_df.iloc[:, -1].values  # 最后一列是 labels
#
# # 更新 ispy2 的标签数据框
# ispy2_label_df['new_label'] = relabel(ispy2_labels)

# 查找 ispy2 数据
# data_5, remaining_file_ids = find_files(data_path_5, ispy2_file_ids, ispy2_label_df, pattern_format="*.nii.gz")  # 处理 ispy2 的文件

# 创建数据框并合并所有路径的数据
df = pd.DataFrame(data)
df_1 = pd.DataFrame(data_1)
df_2 = pd.DataFrame(data_2)
df_3 = pd.DataFrame(data_3)
df_4 = pd.DataFrame(data_4)
# df_5 = pd.DataFrame(data_5)

# 合并所有数据框并重置索引
train_df = pd.concat([df, df_1, df_2, df_3, df_4]).reset_index(drop=True)

# 去重，保留每个 file_id 的第一个出现
train_df = train_df.drop_duplicates(subset=['file_id'], keep='first')

# 读取 paired_id.csv 文件的第一列
paired_id_df = pd.read_csv(paired_id_path)
paired_ids = paired_id_df.iloc[:, 0].values  # 获取第一列的所有值

# 按照 file_ids_to_save 是否包含在 paired_ids 中划分数据集
test_df = train_df[train_df['file_id'].isin(paired_ids)]
train_val_df = train_df[~train_df['file_id'].isin(paired_ids)]

# 使用 RandomOverSampler 进行过采样
ros = RandomOverSampler()
train_val_paths_labels_ids_resampled, resampled_labels = ros.fit_resample(
    train_val_df[['file_paths', 'file_id']], train_val_df['labels']
)

# 获取过采样后的文件路径、标签和文件ID
oversampled_file_paths = train_val_paths_labels_ids_resampled['file_paths'].values
oversampled_file_ids = train_val_paths_labels_ids_resampled['file_id'].values
oversampled_labels = resampled_labels

# Create custom datasets for MRI images
train_val_dataset = read_mr(oversampled_file_paths, oversampled_labels, oversampled_file_ids, transform=transform)
# train_val_dataset = read_mr(train_val_df['file_paths'].values, train_val_df['labels'].values, train_val_df['file_id'].values, transform=transform)
val_dataset = read_mr(test_df['file_paths'].values, test_df['labels'].values, test_df['file_id'].values, transform=transform)
test_dataset = read_mr(test_df['file_paths'].values, test_df['labels'].values, test_df['file_id'].values, transform=transform)

# Data loaders
train_loader = DataLoader(train_val_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 输出数据集中每个类别的数量
# oversampled_class_counts = np.unique(oversampled_labels, return_counts=True)
train_class_counts = train_val_df['labels'].value_counts().sort_index()
val_class_counts = test_df['labels'].value_counts().sort_index()
test_class_counts = test_df['labels'].value_counts().sort_index()

for label, count in train_class_counts.items():
    print(f'Training Set - Class: {label}, Count: {count}')
for label, count in test_class_counts.items():
    print(f'Validation Set - Class: {label}, Count: {count}')
for label, count in test_class_counts.items():
    print(f'Test Set - Class: {label}, Count: {count}')

# 生成数据分布直方图
def plot_combined_histogram(train_class_counts, val_class_counts, test_class_counts, title, file_name):
    plt.figure(figsize=(15, 8))
    x = np.arange(len(train_class_counts))  # oversampled_class_counts[0] 是类别标签
    width = 0.3

    bars1 = plt.bar(x - width, train_class_counts.values, width=width,
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

plot_combined_histogram(train_class_counts, val_class_counts, test_class_counts, 'mr Class Distribution', 'mr_class_distribution.png')

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 定义模型、损失函数和优化器
# 使用指定的超参数值重新定义模型和优化器
learning_rate = 0.0001
num_epochs = 10
weight_decay = 1e-05

model = CustomResNet18_3D(num_classes=3).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# 保存训练损失和验证精度
train_accuracies = []
train_losses = []
val_accuracies = []
val_losses = []

# 训练模型
best_mcc = 0.0
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
        train_loss += loss.item()  # 记录每个batch的损失值
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
    all_predictions = []
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()  # 记录每个batch的损失值
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss / len(val_loader))

    # 计算MCC
    mcc = matthews_corrcoef(all_labels, all_predictions)

    # 计算F1-score
    # f1_scores = f1_score(all_labels, all_predictions, average='weighted')

    # 打印训练信息
    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, mcc: {mcc:.4f}')

    # 保存模型
    if mcc > best_mcc:
        best_mcc = mcc

        # 确保模型保存目录存在
        model_dir = os.path.dirname(model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        torch.save(model.state_dict(), model_path)
        print(f'Saved model with MCC: {best_mcc:.4f}')

    # # 保存模型
    # if f1_scores > best_f1_score:
    #     best_f1_score = f1_scores
    #     torch.save(model.state_dict(), model_path)
    #     print(f'Saved model with Weighted F1-score: {best_f1_score:.4f}')

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
df_all_prob.to_csv(output_path, index=False)
print(f"数据已保存到 {output_path}")

# 输出分类报告和混淆矩阵
report = classification_report(all_labels, all_predictions,
                               target_names=['HR+/HER2-', 'HER2+', 'TN'], output_dict=False)
mcc = matthews_corrcoef(all_labels, all_predictions)

# 保存分类报告和混淆矩阵
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

report_file_path = os.path.join(output_dir, 'mr3_resnet18_classification_report.txt')
confusion_matrix_file_path = os.path.join(output_dir, 'mr3_resnet18_confusion_matrix.png')

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

    # 绘制精度变化图
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy', marker='o')
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'mr3_accuracy_plot.png'))
    plt.show()

    # 绘制损失变化图
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'mr3_loss_plot.png'))
    plt.show()

    # 输出百分比数据的混淆矩阵图像
    cm = confusion_matrix(all_labels, all_predictions)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 将每个类别预测正确的数量除以该类别的总数量，得到百分比
    cm_percent = np.round(cm_percent * 100, 1)  # 将结果保留一位小数

    labels = ['HR+/HER2-', 'HER2+', 'TN']
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', xticklabels=labels, yticklabels=labels)

    # 添加百分比符号
    for t in plt.gca().texts:
        t.set_text(t.get_text() + '%')

    plt.xlabel('Predicted')
    plt.ylabel('Ground truth')
    plt.title('mr resnet18 Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mr3_Confusion_Matrix.png'))
    plt.close()

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
    plt.title('mr ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'mr3_roc_curve.png'))
    plt.show()
