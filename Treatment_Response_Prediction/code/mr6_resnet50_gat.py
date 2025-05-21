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
from monai.networks.nets import resnet50
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import torch_geometric

# 设置设备
device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ----------------- 自注意力机制 (SA) ----------------- #
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, D, H, W = x.size()
        query = self.query_conv(x).view(batch_size, -1, D * H * W)
        key = self.key_conv(x).view(batch_size, -1, D * H * W)
        value = self.value_conv(x).view(batch_size, -1, D * H * W)

        attention = torch.bmm(query.permute(0, 2, 1), key)
        attention = F.softmax(attention, dim=-1)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, D, H, W)

        return self.gamma * out + x  # 残差连接


# ----------------- GAT 模块 ----------------- #
class GAT_Module(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4):
        super(GAT_Module, self).__init__()
        self.gat = GATConv(in_channels, out_channels, heads=num_heads, concat=True)

    def forward(self, x, edge_index):
        x = self.gat(x, edge_index)
        return F.elu(x)


# ----------------- 3D ResNet50 + SA + GAT ----------------- #
class CustomResNet50_3D(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet50_3D, self).__init__()
        self.num_classes = num_classes

        # 使用 MONAI 的 ResNet50
        self.base = resnet50(spatial_dims=3, n_input_channels=1, num_classes=num_classes)

        # GAT：Patch 特征之间建图
        self.gat = GAT_Module(in_channels=512, out_channels=256)  # 512 是 CNN 输出特征

        # 自注意力机制（输入通道需要与你的特征图匹配）
        self.self_attention = SelfAttention(in_channels=512)

        # 分类层：拼接 CNN 和 GAT 的特征
        self.fc = nn.Linear(512 + 256, num_classes)

        # 手动添加 MaxPool3D
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

    def forward(self, x, edge_index):
        # Step 1：ResNet 提取特征
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = torch.relu(x)
        x = self.maxpool(x)

        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)

        # Step 2：平均池化 + flatten，得出 CNN 提取的 patch 特征
        x_flat = self.base.avgpool(x)
        x_flat = torch.flatten(x_flat, 1)  # shape: [B, 512]

        # Step 3：GAT 处理 Patch 间关系
        x_gat = self.gat(x_flat, edge_index)  # shape: [B, 256]

        # Step 4：SA 在原始 3D 特征图上处理全局上下文（增强空间建模）
        x_sa = self.self_attention(x)  # shape: [B, 512, D, H, W]
        x_sa = self.base.avgpool(x_sa)
        x_sa = torch.flatten(x_sa, 1)  # shape: [B, 512]

        # Step 5：拼接 GAT 和 SA 特征
        x_combined = torch.cat((x_sa, x_gat), dim=1)  # shape: [B, 768]

        # Step 6：分类头
        out = self.fc(x_combined)
        return out

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
data_path = "data/fuzhou"
data_path_1 = "data/shandong"
label_path = "file/filtered_all_label.xlsx"
save_dir = "data/mr_model_input"
model_path = "model/mr6_resnet18.pth"
output_path = "result/mr6/mr6_results.csv"
output_dir = "result/mr6"
paired_id_path = "file/paired_id.csv"

# 读取标签时增加pcr列
label_df = pd.read_excel(label_path)
file_ids = label_df['id'].values
labels = label_df['label'].values
pcr_values = label_df['pcr'].values  # 新增pcr列读取

# 修改后的relabel函数，针对每个file_id动态计算new_label
# 修改后的relabel函数，确保只有找到PCR标签的file_id会计算新标签
def relabel(labels, file_ids, label_df):
    new_labels = []
    for label, file_id in zip(labels, file_ids):
        # 找到对应的PCR标签
        pcr_row = label_df[label_df['id'] == file_id]

        if pcr_row.empty:
            # 如果没有找到对应的PCR标签，跳过该条数据
            print(f"Warning: No PCR value found for file_id {file_id}. Skipping...")
            continue  # 跳过没有匹配的 file_id

        pcr = pcr_row['pcr'].values[0]  # 获取对应file_id的PCR标签

        # 根据label和pcr值生成新标签
        if label == 0 or label == 1:  # HER2-/HER2+ 类别
            new_labels.append(0 if pcr == 0 else 1)  # HER2-/non-pcr(0) 或 HER2-/pcr(1)
        elif label == 2 or label == 3:  # TN 类别
            new_labels.append(2 if pcr == 0 else 3)  # TN/non-pcr(2) 或 TN/pcr(3)
        elif label == 4:  # 其他类别
            new_labels.append(4 if pcr == 0 else 5)  # 例如特定的分类，TN的非pcr(pcr值为0)或者pcr(pcr值为1)

    return np.array(new_labels)

# 使用新的relabel函数
new_labels = relabel(labels, file_ids, label_df)

# 输出计算后的标签长度，确保和原始label_df一致
print(f"New labels length: {len(new_labels)}")
print(f"Original label_df length: {len(label_df)}")

# 如果 new_labels 的长度与 label_df 一致，则可以将其赋值
if len(new_labels) == len(label_df):
    label_df['new_label'] = new_labels
else:
    print("Length mismatch: new_labels length does not match label_df length.")

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

# 创建数据框并合并所有路径的数据
df = pd.DataFrame(data)
df_1 = pd.DataFrame(data_1)

# 合并所有数据框并重置索引
train_df = pd.concat([df, df_1]).reset_index(drop=True)

# 去重，保留每个 file_id 的第一个出现
train_df = train_df.drop_duplicates(subset=['file_id'], keep='first')

print(f"列名 :{train_df.columns}")  # 打印所有列名

# 替换原有的基于paired_ids的划分方式，直接划分训练集和测试集
train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['labels'])

# 输出划分后的训练集和测试集的大小
print(f"Training Set Size: {len(train_df)}")
print(f"Test Set Size: {len(test_df)}")
# 使用 RandomOverSampler 进行过采样
ros = RandomOverSampler()
train_paths_labels_ids_resampled, resampled_labels = ros.fit_resample(
    train_df[['file_paths', 'file_id']], train_df['labels']
)

# 获取过采样后的文件路径、标签和文件ID
oversampled_file_paths = train_paths_labels_ids_resampled['file_paths'].values
oversampled_file_ids = train_paths_labels_ids_resampled['file_id'].values
oversampled_labels = resampled_labels

# Create custom datasets for MRI images
train_dataset = read_mr(oversampled_file_paths, oversampled_labels, oversampled_file_ids, transform=transform)
test_dataset = read_mr(test_df['file_paths'].values, test_df['labels'].values, test_df['file_id'].values, transform=transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 输出数据集中每个类别的数量
train_class_counts = train_df['labels'].value_counts().sort_index()
val_class_counts = test_df['labels'].value_counts().sort_index()
test_class_counts = test_df['labels'].value_counts().sort_index()

for label, count in train_class_counts.items():
    print(f'Training Set - Class: {label}, Count: {count}')
for label, count in test_class_counts.items():
    print(f'Test Set - Class: {label}, Count: {count}')

# 绘制数据分布直方图
def plot_combined_histogram(train_counts, val_counts, test_counts, title, filename):
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.25
    bar_indices = np.arange(len(train_counts))

    # 绘制训练集、验证集和测试集的条形图
    bars_train = ax.bar(bar_indices, train_counts, bar_width, label='Training Set')
    bars_val = ax.bar(bar_indices + bar_width, val_counts, bar_width, label='Validation Set')
    bars_test = ax.bar(bar_indices + 2 * bar_width, test_counts, bar_width, label='Test Set')

    # 设置标签和标题
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title(title)

    # 设置x轴的分类标签
    class_labels = ['HER2-/non-pcr', 'HER2-/pcr', 'HER2+/non-pcr', 'HER2+/pcr', 'TN/non-pcr', 'TN/pcr']
    ax.set_xticks(bar_indices + bar_width)
    ax.set_xticklabels(class_labels)

    # 在每个条形图上方添加数量标签
    for i, bar in enumerate(bars_train):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, str(int(height)), ha='center', va='bottom')

    for i, bar in enumerate(bars_val):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, str(int(height)), ha='center', va='bottom')

    for i, bar in enumerate(bars_test):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, str(int(height)), ha='center', va='bottom')

    # 显示图例
    ax.legend()

    # 调整布局并保存图像
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}")
    plt.show()


# 假设train_class_counts, val_class_counts, test_class_counts是包含每个类计数的Series对象
plot_combined_histogram(train_class_counts, val_class_counts, test_class_counts, 'mr Class Distribution',
                        'mr_class_distribution.png')

# 训练模型部分不变
# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 定义模型、损失函数和优化器
learning_rate = 0.00001
num_epochs = 10
weight_decay = 1e-05

model = CustomResNet50_3D(num_classes=6).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# 保存训练损失和验证精度
train_accuracies = []
train_losses = []
test_accuracies = []
test_losses = []

# 在模型训练时加入学习率衰减
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

# 训练模型
# best_mcc = 0.0
best_acc = 0.0
# 在训练过程中加入 scheduler.step()
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
        scheduler.step()  # 调整学习率
        train_loss += loss.item()  # 记录每个batch的损失值
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_accuracy = correct / total
    train_accuracies.append(train_accuracy)
    train_losses.append(train_loss / len(train_loader))

    # 测试模型
    model.eval()
    test_loss = 0
    total = 0
    correct = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()  # 记录每个batch的损失值
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
        test_accuracy = correct / total
        test_accuracies.append(test_accuracy)
        test_losses.append(test_loss / len(test_loader))

    # 计算MCC
    mcc = matthews_corrcoef(all_labels, all_predictions)

    # 打印训练信息
    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss / len(train_loader):.4f}, Test Loss: {test_loss / len(test_loader):.4f}, Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}, mcc: {mcc:.4f}')

    # 用精度作为模型的评估标准
    if test_accuracy > best_acc:
        best_acc = test_accuracy

        # 确保模型保存目录存在
        model_dir = os.path.dirname(model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        torch.save(model.state_dict(), model_path)
        print(f'Saved model with Accuracy: {best_acc:.4f}')


    # # 保存模型
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

# 加载并测试最佳模型
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

# 评估模型性能
df_all_prob = pd.DataFrame(columns=['file_id', 'prob', 'label'])

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

# 输出分类报告并保存到output_dir
report = classification_report(all_labels, all_predictions,
                               target_names=['HER2-/non-pcr', 'HER2-/pcr', 'HER2+/non-pcr', 'HER2+/pcr', 'TN/non-pcr', 'TN/pcr'], output_dict=False)
mcc = matthews_corrcoef(all_labels, all_predictions)

# 输出分类报告
print(report)
print(f'MCC: {mcc:.4f}')

# 保存分类报告到文件
with open(os.path.join(output_dir, 'mr6_classification_report.txt'), 'w') as f:
    f.write(f'{report}\n')
    f.write(f'MCC: {mcc:.4f}\n')

print(f"Classification report saved to {output_dir}/mr6_classification_report.txt")

# 绘制精度变化图
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy', marker='o')
plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy')
plt.legend()
plt.savefig(os.path.join(output_dir, 'mr6_accuracy_plot.png'))
plt.show()

# 绘制损失变化图
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', marker='o')
plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.savefig(os.path.join(output_dir, 'mr6_loss_plot.png'))
plt.show()

# 输出百分比数据的混淆矩阵图像
cm = confusion_matrix(all_labels, all_predictions)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 将每个类别预测正确的数量除以该类别的总数量，得到百分比
cm_percent = np.round(cm_percent * 100, 1)  # 将结果保留一位小数

labels = ['HER2-/non-pcr', 'HER2-/pcr', 'HER2+/non-pcr', 'HER2+/pcr', 'TN/non-pcr', 'TN/pcr']
plt.figure(figsize=(10, 8))
sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', xticklabels=labels, yticklabels=labels)

for t in plt.gca().texts:
    t.set_text(t.get_text() + '%')

plt.xlabel('Predicted')
plt.ylabel('Ground truth')
plt.title('mr resnet18 Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'mr6_Confusion_Matrix.png'))
plt.close()

# 绘制 ROC 曲线并保存到output_dir
class_labels = ['HER2-/non-pcr', 'HER2-/pcr', 'HER2+/non-pcr', 'HER2+/pcr', 'TN/non-pcr', 'TN/pcr']
n_classes = len(class_labels)
all_labels = np.random.randint(0, 6, size=100)  # 示例标签
all_outputs = np.random.rand(100, n_classes)    # 示例预测得分

# Binarize the labels (将类别标签转换为二进制格式)
y_test = label_binarize(all_labels, classes=range(n_classes))  # 0-5 的标签，确保类别从0到5
y_score = all_outputs  # 假设这是模型的输出

# 计算 ROC 曲线和 AUC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 绘制 ROC 曲线
plt.figure(figsize=(10, 8))
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'{class_labels[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')

# 保存和显示图像
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'mr6_ROC_Curve.png'))
plt.show()