#!/usr/bin/env python3
"""
Boston Housing 数据集详细分析
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
pd.set_option('display.float_format', lambda x: '%.3f' % x)

def load_data():
    """加载Boston Housing数据"""
    print("="*80)
    print("Boston Housing 数据集分析")
    print("="*80)

    df = pd.read_csv('../../data/boston_housing.csv')

    print("\n✓ 数据加载成功")
    print(f"  样本数: {len(df):,}")
    print(f"  特征数: {df.shape[1] - 1}")  # 减去目标变量

    return df

def analyze_basic_info(df):
    """基本信息分析"""
    print("\n" + "="*80)
    print("1. 基本信息")
    print("="*80)

    print(f"\n数据集形状: {df.shape}")

    print(f"\n列名和数据类型:")
    print(df.dtypes)

    print(f"\n内存使用:")
    print(df.memory_usage())

    print(f"\n前5行数据:")
    print(df.head())

    print(f"\n数据集统计摘要:")
    print(df.describe())

def analyze_missing_values(df):
    """缺失值分析"""
    print("\n" + "="*80)
    print("2. 缺失值分析")
    print("="*80)

    # 统计NaN缺失值
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)

    if missing.sum() == 0:
        print("\n✓ 没有发现NaN缺失值")
    else:
        print("\n✗ 发现缺失值:")
        for col in missing[missing > 0].index:
            print(f"\n  {col}:")
            print(f"    - 缺失数量: {missing[col]} ({missing_pct[col]:.2f}%)")

    # 检查特殊缺失值标记
    print(f"\n检查特殊缺失值标记（'NA', '?', 'unknown'等）：")

    special_missing_found = False
    for col in df.columns:
        if df[col].dtype == 'object':
            special_markers = ['NA', '?', 'unknown', 'Unknown', 'UNKNOWN', '']
            for marker in special_markers:
                count = (df[col] == marker).sum()
                if count > 0:
                    special_missing_found = True
                    print(f"\n  {col}:")
                    print(f"    - '{marker}': {count} ({100*count/len(df):.2f}%)")

    if not special_missing_found and missing.sum() == 0:
        print("  ✓ 没有发现特殊缺失值标记")

def analyze_duplicates(df):
    """重复值分析"""
    print("\n" + "="*80)
    print("3. 重复值分析")
    print("="*80)

    dup_count = df.duplicated().sum()
    dup_pct = 100 * dup_count / len(df)

    print(f"\n完全重复的行数: {dup_count} ({dup_pct:.2f}%)")

def analyze_target_variable(df, target_col='MEDV'):
    """目标变量分析"""
    print("\n" + "="*80)
    print("4. 目标变量分析")
    print("="*80)

    print(f"\n目标变量: {target_col} (房价中位数, 单位: $1000)")

    print(f"\n统计摘要:")
    print(df[target_col].describe())

    print(f"\n分布特征:")
    print(f"  最小值: ${df[target_col].min():.1f}k")
    print(f"  25分位: ${df[target_col].quantile(0.25):.1f}k")
    print(f"  中位数: ${df[target_col].median():.1f}k")
    print(f"  75分位: ${df[target_col].quantile(0.75):.1f}k")
    print(f"  最大值: ${df[target_col].max():.1f}k")
    print(f"  均值:   ${df[target_col].mean():.1f}k")
    print(f"  标准差: ${df[target_col].std():.1f}k")

    # 偏度和峰度
    print(f"\n偏度 (skewness): {df[target_col].skew():.3f}")
    print(f"峰度 (kurtosis): {df[target_col].kurtosis():.3f}")

    if df[target_col].skew() > 1:
        print("  ⚠️  目标变量右偏，可能需要对数变换")
    elif df[target_col].skew() < -1:
        print("  ⚠️  目标变量左偏")
    else:
        print("  ✓ 目标变量接近正态分布")

def analyze_numerical_features(df, target_col='MEDV'):
    """数值特征分析"""
    print("\n" + "="*80)
    print("5. 数值特征分析")
    print("="*80)

    # 获取数值列（排除目标变量）
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    print(f"\n数值特征列表 ({len(numeric_cols)} 个):")
    print(", ".join(numeric_cols))

    print(f"\n统计摘要:")
    stats = df[numeric_cols].describe().T
    stats['skew'] = df[numeric_cols].skew()
    stats['kurtosis'] = df[numeric_cols].kurtosis()
    print(stats)

    # 异常值检测（IQR方法）
    print(f"\n异常值检测（IQR方法）:")

    for col in numeric_cols:
        if df[col].notna().sum() == 0:
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        outliers_pct = 100 * outliers / df[col].notna().sum()

        if outliers > 0:
            print(f"\n  {col}:")
            print(f"    范围: [{df[col].min():.2f}, {df[col].max():.2f}]")
            print(f"    正常范围: [{lower:.2f}, {upper:.2f}]")
            print(f"    异常值数量: {outliers} ({outliers_pct:.2f}%)")
            outlier_vals = df[col][(df[col] < lower) | (df[col] > upper)].head(5).tolist()
            print(f"    异常值示例: {[f'{v:.2f}' for v in outlier_vals]}")

def analyze_categorical_features(df):
    """类别特征分析"""
    print("\n" + "="*80)
    print("6. 类别特征分析")
    print("="*80)

    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    if len(cat_cols) == 0:
        print("\n✓ 没有类别特征（全部为数值特征）")
        return

    print(f"\n类别特征列表 ({len(cat_cols)} 个):")
    print(", ".join(cat_cols))

    print(f"\n每个类别特征的唯一值数量:")
    for col in cat_cols:
        unique_count = df[col].nunique()
        print(f"\n  {col}: {unique_count} 个唯一值")

        value_counts = df[col].value_counts()
        print(f"    分布:")
        for val, count in value_counts.head(10).items():
            pct = 100 * count / len(df)
            print(f"      {val:20s}: {count:5,} ({pct:5.2f}%)")

        if len(value_counts) > 10:
            print(f"      ... (还有 {len(value_counts)-10} 个类别)")

def analyze_feature_descriptions():
    """特征描述"""
    print("\n" + "="*80)
    print("7. 特征详细说明")
    print("="*80)

    feature_desc = {
        'CRIM': '人均犯罪率 (per capita crime rate by town)',
        'ZN': '超过25,000平方英尺的住宅用地比例 (proportion of residential land zoned for lots over 25,000 sq.ft.)',
        'INDUS': '城镇中非零售商业用地的比例 (proportion of non-retail business acres per town)',
        'CHAS': '是否邻近查尔斯河 (Charles River dummy variable: 1 if tract bounds river; 0 otherwise)',
        'NOX': '一氧化氮浓度 (nitric oxides concentration, parts per 10 million)',
        'RM': '平均房间数 (average number of rooms per dwelling)',
        'AGE': '1940年前建造的自住单位比例 (proportion of owner-occupied units built prior to 1940)',
        'DIS': '到5个波士顿就业中心的加权距离 (weighted distances to five Boston employment centres)',
        'RAD': '径向公路的可达性指数 (index of accessibility to radial highways)',
        'TAX': '每万美元的全额财产税率 (full-value property-tax rate per $10,000)',
        'PTRATIO': '城镇的师生比例 (pupil-teacher ratio by town)',
        'B': '1000(Bk - 0.63)^2，Bk是城镇中黑人的比例 (1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town)',
        'LSTAT': '人口中地位较低者的百分比 (% lower status of the population)',
        'MEDV': '自住房屋的中位数价值，单位:千美元 (Median value of owner-occupied homes in $1000s) [目标变量]'
    }

    print("\n特征名称及含义:")
    for i, (feat, desc) in enumerate(feature_desc.items(), 1):
        print(f"\n{i:2d}. {feat:8s}: {desc}")

def analyze_data_quality(df):
    """数据质量问题检测"""
    print("\n" + "="*80)
    print("8. 数据质量问题检测")
    print("="*80)

    issues = []

    # 检查负值（某些特征不应该有负值）
    non_negative_cols = ['ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'LSTAT', 'MEDV']
    for col in non_negative_cols:
        if col in df.columns:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                issues.append(f"✗ {col}: 有 {neg_count} 个负值")

    # 检查CHAS（应该只有0或1）
    if 'CHAS' in df.columns:
        unique_vals = df['CHAS'].dropna().unique()
        if not set(unique_vals).issubset({0, 1}):
            issues.append(f"✗ CHAS: 包含非0/1值: {unique_vals}")

    # 检查异常比例（应该在0-100或0-1之间）
    percentage_cols = ['ZN', 'INDUS']
    for col in percentage_cols:
        if col in df.columns:
            max_val = df[col].max()
            if max_val > 100:
                issues.append(f"⚠️  {col}: 最大值 {max_val:.2f} > 100 (可能不是百分比)")

    # 检查缺失值
    missing = df.isnull().sum()
    if missing.sum() > 0:
        for col in missing[missing > 0].index:
            issues.append(f"✗ {col}: 有 {missing[col]} 个缺失值 ({100*missing[col]/len(df):.2f}%)")

    # 打印问题
    if issues:
        print("\n发现的问题:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✓ 未发现明显的数据质量问题")

def generate_cleaning_recommendations(df):
    """生成清洗建议"""
    print("\n" + "="*80)
    print("9. 数据清洗建议")
    print("="*80)

    recommendations = []

    # 缺失值处理建议
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("\n推荐的数据预处理步骤:")
        for col in missing[missing > 0].index:
            missing_pct = 100 * missing[col] / len(df)
            if missing_pct < 5:
                recommendations.append(f"  • {col}: {missing_pct:.2f}% 缺失 → 建议删除含缺失值的行")
            elif missing_pct < 20:
                recommendations.append(f"  • {col}: {missing_pct:.2f}% 缺失 → 建议用中位数/均值填充")
            else:
                recommendations.append(f"  • {col}: {missing_pct:.2f}% 缺失 → 建议删除该特征")
    else:
        recommendations.append("  • 缺失值: ✓ 无缺失值（或已处理）")

    # 异常值处理建议
    recommendations.append("  • 异常值: 根据IQR检测到的异常值，需要判断是否为真实数据还是错误")
    recommendations.append("    - 如果是真实的极端值（如豪宅、贫民窟），建议保留")
    recommendations.append("    - 如果是数据错误，建议删除或修正")

    # 特征工程建议
    recommendations.append("  • 特征工程:")
    recommendations.append("    - CHAS 已经是二值特征（0/1），无需处理")
    recommendations.append("    - 其他特征都是连续数值，RRL会自动标准化")

    # 目标变量处理
    if 'MEDV' in df.columns:
        skew = df['MEDV'].skew()
        if abs(skew) > 0.5:
            recommendations.append(f"  • 目标变量 MEDV: 偏度={skew:.3f}")
            if skew > 0.5:
                recommendations.append("    - 右偏，回归时可能需要对数变换: log(MEDV)")

    # 打印建议
    for rec in recommendations:
        print(rec)

def main():
    """主函数"""
    # 加载数据
    df = load_data()

    # 1. 基本信息
    analyze_basic_info(df)

    # 2. 缺失值分析
    analyze_missing_values(df)

    # 3. 重复值分析
    analyze_duplicates(df)

    # 4. 目标变量分析
    analyze_target_variable(df)

    # 5. 数值特征分析
    analyze_numerical_features(df)

    # 6. 类别特征分析
    analyze_categorical_features(df)

    # 7. 特征描述
    analyze_feature_descriptions()

    # 8. 数据质量检测
    analyze_data_quality(df)

    # 9. 清洗建议
    generate_cleaning_recommendations(df)

    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)

if __name__ == '__main__':
    main()
