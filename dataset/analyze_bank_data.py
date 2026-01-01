#!/usr/bin/env python3
"""
Bank Marketing 数据集详细分析
包括：数据分布、缺失值、异常值、相关性分析等
"""

import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 设置显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def load_data():
    """加载数据"""
    print("="*80)
    print("Bank Marketing 数据集分析")
    print("="*80)

    # 读取数据
    df = pd.read_csv('../../data/bank-full.csv', sep=';')
    print(f"\n✓ 数据加载成功")
    print(f"  样本数: {len(df):,}")
    print(f"  特征数: {df.shape[1]}")

    return df

def basic_info(df):
    """基本信息"""
    print("\n" + "="*80)
    print("1. 基本信息")
    print("="*80)

    print("\n数据集形状:", df.shape)
    print("\n列名和数据类型:")
    print(df.dtypes)

    print("\n内存使用:")
    print(df.memory_usage(deep=True))

    print("\n前5行数据:")
    print(df.head())

    print("\n数据集统计摘要:")
    print(df.describe(include='all'))

def check_missing_values(df):
    """检查缺失值"""
    print("\n" + "="*80)
    print("2. 缺失值分析")
    print("="*80)

    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)

    missing_df = pd.DataFrame({
        '列名': missing.index,
        '缺失数': missing.values,
        '缺失率(%)': missing_pct.values
    })
    missing_df = missing_df[missing_df['缺失数'] > 0].sort_values('缺失数', ascending=False)

    if len(missing_df) == 0:
        print("\n✓ 没有发现缺失值（NaN）")
    else:
        print(f"\n✗ 发现 {len(missing_df)} 个特征有缺失值:")
        print(missing_df.to_string(index=False))

    # 检查特殊标记的缺失值
    print("\n检查特殊缺失值标记（'unknown', '?', 'NA'等）：")
    special_missing = {}

    for col in df.columns:
        if df[col].dtype == 'object':  # 只检查字符串列
            unknown_count = (df[col] == 'unknown').sum()
            question_count = (df[col].str.contains('\?', regex=True, na=False)).sum()
            na_count = (df[col].str.upper() == 'NA').sum()

            total_special = unknown_count + question_count + na_count
            if total_special > 0:
                special_missing[col] = {
                    'unknown': unknown_count,
                    '?': question_count,
                    'NA': na_count,
                    'total': total_special,
                    'pct': 100 * total_special / len(df)
                }

    if special_missing:
        print(f"\n✗ 发现 {len(special_missing)} 个特征有特殊缺失值标记:")
        for col, counts in special_missing.items():
            print(f"\n  {col}:")
            print(f"    - 'unknown': {counts['unknown']:,} ({counts['unknown']/len(df)*100:.2f}%)")
            print(f"    - '?': {counts['?']:,}")
            print(f"    - 'NA': {counts['NA']:,}")
            print(f"    - 总计: {counts['total']:,} ({counts['pct']:.2f}%)")
    else:
        print("  ✓ 没有发现特殊缺失值标记")

def check_duplicates(df):
    """检查重复值"""
    print("\n" + "="*80)
    print("3. 重复值分析")
    print("="*80)

    # 完全重复
    dup_count = df.duplicated().sum()
    print(f"\n完全重复的行数: {dup_count:,} ({100*dup_count/len(df):.2f}%)")

    if dup_count > 0:
        print("\n前5个重复样本:")
        print(df[df.duplicated(keep=False)].head(10))

def analyze_target(df):
    """分析目标变量"""
    print("\n" + "="*80)
    print("4. 目标变量分析")
    print("="*80)

    target_col = 'y'

    print(f"\n目标变量: {target_col}")
    print("\n类别分布:")
    value_counts = df[target_col].value_counts()
    for val, count in value_counts.items():
        pct = 100 * count / len(df)
        print(f"  {val:5s}: {count:6,} ({pct:5.2f}%)")

    # 计算不平衡比例
    imbalance_ratio = value_counts.max() / value_counts.min()
    print(f"\n不平衡比例: {imbalance_ratio:.2f}:1")

    if imbalance_ratio > 1.5:
        print("  ⚠️  数据高度不平衡！需要使用加权损失或重采样。")

    return value_counts

def analyze_numerical_features(df):
    """分析数值特征"""
    print("\n" + "="*80)
    print("5. 数值特征分析")
    print("="*80)

    # 识别数值特征
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numerical_cols:
        print("\n✓ 没有数值特征")
        return

    print(f"\n数值特征列表 ({len(numerical_cols)} 个):")
    print(", ".join(numerical_cols))

    print("\n统计摘要:")
    stats = df[numerical_cols].describe().T
    stats['skew'] = df[numerical_cols].skew()
    stats['kurtosis'] = df[numerical_cols].kurtosis()
    print(stats)

    # 检查异常值（使用IQR方法）
    print("\n异常值检测（IQR方法）:")
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_pct = 100 * outlier_count / len(df)

        if outlier_count > 0:
            print(f"\n  {col}:")
            print(f"    范围: [{df[col].min():.2f}, {df[col].max():.2f}]")
            print(f"    正常范围: [{lower_bound:.2f}, {upper_bound:.2f}]")
            print(f"    异常值数量: {outlier_count:,} ({outlier_pct:.2f}%)")
            print(f"    异常值示例: {outliers[col].head(5).tolist()}")

def analyze_categorical_features(df):
    """分析类别特征"""
    print("\n" + "="*80)
    print("6. 类别特征分析")
    print("="*80)

    # 识别类别特征
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    if 'y' in categorical_cols:
        categorical_cols.remove('y')  # 移除目标变量

    print(f"\n类别特征列表 ({len(categorical_cols)} 个):")
    print(", ".join(categorical_cols))

    print("\n每个类别特征的唯一值数量:")
    for col in categorical_cols:
        n_unique = df[col].nunique()
        print(f"\n  {col}: {n_unique} 个唯一值")

        value_counts = df[col].value_counts()
        print(f"    分布:")
        for i, (val, count) in enumerate(value_counts.head(10).items()):
            pct = 100 * count / len(df)
            print(f"      {val:20s}: {count:6,} ({pct:5.2f}%)")

        if len(value_counts) > 10:
            print(f"      ... (还有 {len(value_counts)-10} 个类别)")

def check_data_quality_issues(df):
    """检查数据质量问题"""
    print("\n" + "="*80)
    print("7. 数据质量问题检测")
    print("="*80)

    issues = []

    # 1. 检查负数（对于应该是正数的特征）
    positive_cols = ['age', 'balance', 'duration', 'campaign', 'previous']
    for col in positive_cols:
        if col in df.columns:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                issues.append(f"  ✗ {col}: 有 {neg_count} 个负值（可能是缺失值标记）")

    # 2. 检查不合理的年龄
    if 'age' in df.columns:
        weird_age = df[(df['age'] < 18) | (df['age'] > 100)]
        if len(weird_age) > 0:
            issues.append(f"  ⚠️  age: 有 {len(weird_age)} 个不合理年龄 (<18 或 >100)")

    # 3. 检查 pdays = -1 的含义
    if 'pdays' in df.columns:
        pdays_neg = (df['pdays'] == -1).sum()
        if pdays_neg > 0:
            issues.append(f"  ℹ️  pdays: {pdays_neg:,} ({100*pdays_neg/len(df):.1f}%) 个样本 = -1（表示未联系过）")

    # 4. 检查 duration = 0
    if 'duration' in df.columns:
        zero_duration = (df['duration'] == 0).sum()
        if zero_duration > 0:
            issues.append(f"  ⚠️  duration: 有 {zero_duration} 个样本 = 0（通话时长为0）")

    # 5. 检查常数列
    for col in df.columns:
        if df[col].nunique() == 1:
            issues.append(f"  ✗ {col}: 常数列（所有值相同）")

    if issues:
        print("\n发现的问题:")
        for issue in issues:
            print(issue)
    else:
        print("\n✓ 未发现明显的数据质量问题")

def generate_recommendations(df):
    """生成数据清洗建议"""
    print("\n" + "="*80)
    print("8. 数据清洗建议")
    print("="*80)

    recommendations = []

    # 1. 缺失值处理
    for col in df.columns:
        if df[col].dtype == 'object':
            unknown_count = (df[col] == 'unknown').sum()
            if unknown_count > 0:
                pct = 100 * unknown_count / len(df)
                if pct > 50:
                    recommendations.append(
                        f"  • {col}: {pct:.1f}% 为 'unknown' → 建议删除该特征或使用模式填充"
                    )
                elif pct > 10:
                    recommendations.append(
                        f"  • {col}: {pct:.1f}% 为 'unknown' → 建议单独编码为一个类别"
                    )
                else:
                    recommendations.append(
                        f"  • {col}: {pct:.1f}% 为 'unknown' → 可以删除或保留为单独类别"
                    )

    # 2. pdays 特殊处理
    if 'pdays' in df.columns:
        pdays_neg = (df['pdays'] == -1).sum()
        if pdays_neg > 0:
            recommendations.append(
                f"  • pdays: -1 表示'未联系过' → 建议创建二值特征 'contacted_before'"
            )

    # 3. 不平衡数据
    target_counts = df['y'].value_counts()
    imbalance_ratio = target_counts.max() / target_counts.min()
    if imbalance_ratio > 2:
        recommendations.append(
            f"  • 目标变量: 不平衡比例 {imbalance_ratio:.1f}:1 → 建议使用加权损失函数（--weighted）"
        )

    # 4. 标准化
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numerical_cols:
        recommendations.append(
            f"  • 数值特征: 需要标准化（RRL 的 DBEncoder 会自动处理）"
        )

    # 5. One-Hot编码
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'y' in categorical_cols:
        categorical_cols.remove('y')
    if categorical_cols:
        recommendations.append(
            f"  • 类别特征: 需要 One-Hot 编码（RRL 的 DBEncoder 会自动处理）"
        )

    if recommendations:
        print("\n推荐的数据预处理步骤:")
        for rec in recommendations:
            print(rec)

    return recommendations

def main():
    """主函数"""
    # 加载数据
    df = load_data()

    # 1. 基本信息
    basic_info(df)

    # 2. 缺失值分析
    check_missing_values(df)

    # 3. 重复值分析
    check_duplicates(df)

    # 4. 目标变量分析
    analyze_target(df)

    # 5. 数值特征分析
    analyze_numerical_features(df)

    # 6. 类别特征分析
    analyze_categorical_features(df)

    # 7. 数据质量检测
    check_data_quality_issues(df)

    # 8. 清洗建议
    recommendations = generate_recommendations(df)

    # 保存分析报告
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)

if __name__ == '__main__':
    main()
