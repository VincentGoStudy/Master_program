import pandas as pd
import numpy as np

# 读取MG和MRI的结果文件
mg_prob_path = "result/multi3/mg3_CA/mg3_prob.csv"
mri_prob_path = "result/multi3/mr3_SA/mr3_prob.csv"
output_path = "result/multi3/multi_resnet_MAF/multi_modal_adaptive_prob.csv"

# 加载数据
mg_df = pd.read_csv(mg_prob_path)
mri_df = pd.read_csv(mri_prob_path)

# 确保两个文件中的 'file_id' 一致
merged_df = pd.merge(mg_df, mri_df, on='file_id', suffixes=('_mg', '_mri'))

# 定义计算熵的函数，假设prob列代表预测属于一个类的概率
def compute_entropy(prob):
    return -prob * np.log(prob + 1e-10) - (1 - prob) * np.log(1 - prob + 1e-10)

# 计算MG和MRI的熵
merged_df['entropy_mg'] = compute_entropy(merged_df['prob_mg'])
merged_df['entropy_mri'] = compute_entropy(merged_df['prob_mri'])

# 计算熵的倒数并归一化为权重
merged_df['inv_entropy_mg'] = 1 / (merged_df['entropy_mg'] + 1e-10)  # 避免除以0
merged_df['inv_entropy_mri'] = 1 / (merged_df['entropy_mri'] + 1e-10)

# 归一化权重
merged_df['weight_mg'] = merged_df['inv_entropy_mg'] / (merged_df['inv_entropy_mg'] + merged_df['inv_entropy_mri'])
merged_df['weight_mri'] = merged_df['inv_entropy_mri'] / (merged_df['inv_entropy_mg'] + merged_df['inv_entropy_mri'])

# 计算加权概率
merged_df['prob_weighted'] = merged_df['weight_mg'] * merged_df['prob_mg'] + merged_df['weight_mri'] * merged_df['prob_mri']

# 创建最终的数据框
final_df = merged_df[['file_id', 'prob_weighted', 'label_mg']].rename(columns={'prob_weighted': 'prob', 'label_mg': 'label'})

# 保存到CSV文件
final_df.to_csv(output_path, index=False)

print(f"自适应加权后的双模态分类结果已保存到 {output_path}")
