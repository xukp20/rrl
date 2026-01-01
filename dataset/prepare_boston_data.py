#!/usr/bin/env python3
"""
Boston Housing 数据集准备脚本
生成5个版本：1个原始 + 4个清洗组合（2×2）

版本命名规则（类似Bank数据）：
- boston-housing: 原始数据
- boston-housing-clean-[cd/cc]-[mo/ml]:
  - cd = CHAS discrete
  - cc = CHAS continuous
  - mo = MEDV original
  - ml = MEDV log
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# 特征名称（固定顺序）
FEATURE_NAMES = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
                 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
TARGET_NAME = 'MEDV'

def load_data():
    """加载Boston Housing数据"""
    data_path = '../../data/boston_housing.csv'
    df = pd.read_csv(data_path)

    print("=" * 80)
    print("Boston Housing 数据集准备")
    print("=" * 80)
    print(f"\n✓ 数据加载成功: {data_path}")
    print(f"  原始样本数: {len(df):,}")
    print(f"  特征数: {len(FEATURE_NAMES)}")
    print(f"  目标变量: {TARGET_NAME}")

    # 检查缺失值
    missing_count = df.isnull().sum().sum()
    missing_rows = df.isnull().any(axis=1).sum()
    print(f"\n缺失值统计:")
    print(f"  含缺失值的样本数: {missing_rows} ({100*missing_rows/len(df):.2f}%)")

    return df

def fill_missing_values(df):
    """
    填充缺失值
    - 连续特征：使用中位数填充
    - CHAS（离散特征）：使用众数（0）填充

    返回填充后的DataFrame和统计信息
    """
    df_filled = df.copy()
    stats = {}

    # 有缺失值的连续特征
    continuous_missing = ['CRIM', 'ZN', 'INDUS', 'AGE', 'LSTAT']

    print("\n填充缺失值:")
    for col in continuous_missing:
        if col in df_filled.columns:
            missing_count = df_filled[col].isnull().sum()
            if missing_count > 0:
                median_val = df_filled[col].median()
                df_filled[col] = df_filled[col].fillna(median_val)
                stats[col] = f'filled {missing_count} with median={median_val:.3f}'
                print(f"  {col}: 填充 {missing_count} 个缺失值，中位数={median_val:.3f}")

    # CHAS（离散特征）用众数填充
    if 'CHAS' in df_filled.columns:
        chas_missing = df_filled['CHAS'].isnull().sum()
        if chas_missing > 0:
            mode_val = df_filled['CHAS'].mode()[0]  # 众数是0
            df_filled['CHAS'] = df_filled['CHAS'].fillna(mode_val)
            stats['CHAS'] = f'filled {chas_missing} with mode={mode_val}'
            print(f"  CHAS: 填充 {chas_missing} 个缺失值，众数={mode_val}")

    # 验证无缺失值
    remaining_missing = df_filled.isnull().sum().sum()
    if remaining_missing == 0:
        print(f"\n✓ 所有缺失值已填充，剩余缺失: 0")
    else:
        print(f"\n⚠️  仍有 {remaining_missing} 个缺失值")

    return df_filled, stats

def save_to_rrl_format(df, dataset_name, chas_type='discrete', medv_transform=None):
    """
    保存数据为RRL格式（.data和.info文件）

    Args:
        df: DataFrame，包含所有特征和目标变量
        dataset_name: 数据集名称（无后缀）
        chas_type: 'discrete' 或 'continuous'
        medv_transform: None 或 'log'
    """
    # 准备特征类型列表
    feature_types = []
    for feat in FEATURE_NAMES:
        if feat == 'CHAS':
            feature_types.append(chas_type)
        else:
            feature_types.append('continuous')

    # 保存.info文件（特征类型定义）
    # 格式：每行 "feature_name feature_type" (空格分隔)
    # 最后一行：目标变量 + "LABEL_POS -1"
    info_path = f"{dataset_name}.info"
    with open(info_path, 'w') as f:
        # 写入每个特征：特征名 类型（空格分隔）
        for feat, ftype in zip(FEATURE_NAMES, feature_types):
            f.write(f'{feat} {ftype}\n')
        # 写入目标变量：目标名 类型（空格分隔）
        f.write(f'{TARGET_NAME} continuous\n')
        # 写入标签位置（-1表示最后一列）
        f.write('LABEL_POS -1\n')

    # 保存.data文件（数据内容）
    data_path = f"{dataset_name}.data"

    # 手动写入，处理缺失值（用'?'表示）
    with open(data_path, 'w') as f:
        for idx in range(len(df)):
            values = []
            # 写入特征
            for feat in FEATURE_NAMES:
                val = df.iloc[idx][feat]
                if pd.isna(val):
                    values.append('?')
                else:
                    values.append(str(val))
            # 写入目标变量
            target_val = df.iloc[idx][TARGET_NAME]
            if pd.isna(target_val):
                values.append('?')
            else:
                values.append(str(target_val))

            f.write(','.join(values) + '\n')

    return info_path, data_path

def generate_version_0(df):
    """
    版本0：原始数据（Baseline）
    - 保留所有缺失值
    - CHAS定义为continuous
    - MEDV原始值
    """
    print("\n" + "=" * 80)
    print("版本0：boston-housing（原始Baseline）")
    print("=" * 80)

    df_v0 = df.copy()

    info_path, data_path = save_to_rrl_format(
        df_v0,
        'boston-housing',
        chas_type='continuous',
        medv_transform=None
    )

    print(f"✓ 版本0生成完成")
    print(f"  样本数: {len(df_v0):,}")
    print(f"  缺失值: 保留（{df_v0.isnull().any(axis=1).sum()}个样本含缺失）")
    print(f"  CHAS: continuous")
    print(f"  MEDV: 原始值")
    print(f"  文件:")
    print(f"    - {info_path}")
    print(f"    - {data_path}")

    return {
        'version': 0,
        'name': 'boston-housing',
        'suffix': '(原始)',
        'samples': len(df_v0),
        'missing_handling': '保留',
        'chas_type': 'continuous',
        'medv_transform': '原始',
        'info_path': info_path,
        'data_path': data_path
    }

def generate_version_1(df_filled):
    """
    版本1：boston-housing-clean-cd-mo
    - 填充缺失值
    - CHAS discrete
    - MEDV original
    """
    print("\n" + "=" * 80)
    print("版本1：boston-housing-clean-cd-mo（标准清洗）")
    print("=" * 80)

    df_v1 = df_filled.copy()

    info_path, data_path = save_to_rrl_format(
        df_v1,
        'boston-housing-clean-cd-mo',
        chas_type='discrete',
        medv_transform=None
    )

    print(f"✓ 版本1生成完成")
    print(f"  样本数: {len(df_v1):,}")
    print(f"  缺失值: 填充（中位数/众数）")
    print(f"  CHAS: discrete (One-Hot → 2维)")
    print(f"  MEDV: 原始值")
    print(f"  后缀: cd-mo (CHAS discrete, MEDV original)")
    print(f"  文件:")
    print(f"    - {info_path}")
    print(f"    - {data_path}")

    return {
        'version': 1,
        'name': 'boston-housing-clean-cd-mo',
        'suffix': 'cd-mo',
        'samples': len(df_v1),
        'missing_handling': '填充',
        'chas_type': 'discrete',
        'medv_transform': '原始',
        'info_path': info_path,
        'data_path': data_path
    }

def generate_version_2(df_filled):
    """
    版本2：boston-housing-clean-cd-ml（推荐最优）
    - 填充缺失值
    - CHAS discrete
    - MEDV log
    """
    print("\n" + "=" * 80)
    print("版本2：boston-housing-clean-cd-ml（推荐最优）⭐")
    print("=" * 80)

    df_v2 = df_filled.copy()

    # 对MEDV进行log变换
    medv_min = df_v2[TARGET_NAME].min()
    medv_max = df_v2[TARGET_NAME].max()
    print(f"  MEDV原始值范围: [{medv_min:.1f}, {medv_max:.1f}]")

    df_v2[TARGET_NAME] = np.log(df_v2[TARGET_NAME])

    medv_log_min = df_v2[TARGET_NAME].min()
    medv_log_max = df_v2[TARGET_NAME].max()
    print(f"  log(MEDV)范围: [{medv_log_min:.3f}, {medv_log_max:.3f}]")

    info_path, data_path = save_to_rrl_format(
        df_v2,
        'boston-housing-clean-cd-ml',
        chas_type='discrete',
        medv_transform='log'
    )

    print(f"✓ 版本2生成完成")
    print(f"  样本数: {len(df_v2):,}")
    print(f"  缺失值: 填充（中位数/众数）")
    print(f"  CHAS: discrete (One-Hot → 2维)")
    print(f"  MEDV: log变换")
    print(f"  后缀: cd-ml (CHAS discrete, MEDV log)")
    print(f"  文件:")
    print(f"    - {info_path}")
    print(f"    - {data_path}")

    return {
        'version': 2,
        'name': 'boston-housing-clean-cd-ml',
        'suffix': 'cd-ml',
        'samples': len(df_v2),
        'missing_handling': '填充',
        'chas_type': 'discrete',
        'medv_transform': 'log',
        'info_path': info_path,
        'data_path': data_path
    }

def generate_version_3(df_filled):
    """
    版本3：boston-housing-clean-cc-mo
    - 填充缺失值
    - CHAS continuous
    - MEDV original
    """
    print("\n" + "=" * 80)
    print("版本3：boston-housing-clean-cc-mo（CHAS连续）")
    print("=" * 80)

    df_v3 = df_filled.copy()

    info_path, data_path = save_to_rrl_format(
        df_v3,
        'boston-housing-clean-cc-mo',
        chas_type='continuous',
        medv_transform=None
    )

    print(f"✓ 版本3生成完成")
    print(f"  样本数: {len(df_v3):,}")
    print(f"  缺失值: 填充（中位数/众数）")
    print(f"  CHAS: continuous (标准化)")
    print(f"  MEDV: 原始值")
    print(f"  后缀: cc-mo (CHAS continuous, MEDV original)")
    print(f"  文件:")
    print(f"    - {info_path}")
    print(f"    - {data_path}")

    return {
        'version': 3,
        'name': 'boston-housing-clean-cc-mo',
        'suffix': 'cc-mo',
        'samples': len(df_v3),
        'missing_handling': '填充',
        'chas_type': 'continuous',
        'medv_transform': '原始',
        'info_path': info_path,
        'data_path': data_path
    }

def generate_version_4(df_filled):
    """
    版本4：boston-housing-clean-cc-ml
    - 填充缺失值
    - CHAS continuous
    - MEDV log
    """
    print("\n" + "=" * 80)
    print("版本4：boston-housing-clean-cc-ml（CHAS连续+log）")
    print("=" * 80)

    df_v4 = df_filled.copy()

    # 对MEDV进行log变换
    df_v4[TARGET_NAME] = np.log(df_v4[TARGET_NAME])

    info_path, data_path = save_to_rrl_format(
        df_v4,
        'boston-housing-clean-cc-ml',
        chas_type='continuous',
        medv_transform='log'
    )

    print(f"✓ 版本4生成完成")
    print(f"  样本数: {len(df_v4):,}")
    print(f"  缺失值: 填充（中位数/众数）")
    print(f"  CHAS: continuous (标准化)")
    print(f"  MEDV: log变换")
    print(f"  后缀: cc-ml (CHAS continuous, MEDV log)")
    print(f"  文件:")
    print(f"    - {info_path}")
    print(f"    - {data_path}")

    return {
        'version': 4,
        'name': 'boston-housing-clean-cc-ml',
        'suffix': 'cc-ml',
        'samples': len(df_v4),
        'missing_handling': '填充',
        'chas_type': 'continuous',
        'medv_transform': 'log',
        'info_path': info_path,
        'data_path': data_path
    }

def generate_summary(versions_info):
    """生成汇总报告"""
    print("\n" + "=" * 80)
    print("汇总报告")
    print("=" * 80)

    print(f"\n共生成 {len(versions_info)} 个版本:")
    print(f"\n{'版本':<5} {'名称':<35} {'样本数':<10} {'CHAS':<12} {'MEDV':<10}")
    print("-" * 80)

    for info in versions_info:
        print(f"{info['version']:<5} {info['name']:<35} {info['samples']:<10,} "
              f"{info['chas_type']:<12} {info['medv_transform']:<10}")

    print("\n" + "-" * 80)
    print("后缀命名规则:")
    print("  cd = CHAS discrete (One-Hot编码)")
    print("  cc = CHAS continuous (标准化)")
    print("  mo = MEDV original (原始值)")
    print("  ml = MEDV log (log变换)")
    print("\n示例: boston-housing-clean-cd-ml = 删除缺失值 + CHAS discrete + MEDV log")

    print("\n" + "-" * 80)
    print("推荐使用顺序:")
    print("  1. 版本2 (cd-ml): 预期性能最好 ⭐⭐⭐⭐⭐")
    print("  2. 版本4 (cc-ml): 探索CHAS连续处理 ⭐⭐⭐⭐")
    print("  3. 版本1 (cd-mo): 标准清洗，无log变换 ⭐⭐⭐")
    print("  4. 版本3 (cc-mo): 对比CHAS处理方式 ⭐⭐⭐")
    print("  5. 版本0 (原始): Baseline参考 ⭐⭐")

    print("\n" + "-" * 80)
    print("实验设计建议:")
    print("  - 快速对比: 训练版本0, 1, 2")
    print("  - 完整探索: 训练所有5个版本")
    print("  - 关键对比维度:")
    print("    1. 清洗价值: 版本0 vs 其他")
    print("    2. CHAS类型: discrete vs continuous")
    print("    3. log变换: original vs log")

    # 保存汇总到文件
    summary_path = '../../data/boston_versions_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Boston Housing 数据集版本汇总\n")
        f.write("=" * 80 + "\n")
        f.write(f"\n生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n共 {len(versions_info)} 个版本:\n\n")

        f.write(f"{'版本':<5} {'名称':<35} {'样本数':<10} {'CHAS':<12} {'MEDV':<10}\n")
        f.write("-" * 80 + "\n")

        for info in versions_info:
            f.write(f"{info['version']:<5} {info['name']:<35} {info['samples']:<10,} "
                   f"{info['chas_type']:<12} {info['medv_transform']:<10}\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("后缀命名规则:\n")
        f.write("  cd = CHAS discrete (One-Hot编码)\n")
        f.write("  cc = CHAS continuous (标准化)\n")
        f.write("  mo = MEDV original (原始值)\n")
        f.write("  ml = MEDV log (log变换)\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("文件详情:\n\n")

        for info in versions_info:
            f.write(f"版本{info['version']}: {info['name']}\n")
            f.write(f"  样本数: {info['samples']:,}\n")
            f.write(f"  缺失值: {info['missing_handling']}\n")
            f.write(f"  CHAS: {info['chas_type']}\n")
            f.write(f"  MEDV: {info['medv_transform']}\n")
            f.write(f"  文件:\n")
            f.write(f"    - {info['info_path']}\n")
            f.write(f"    - {info['data_path']}\n")
            f.write("\n")

    print(f"\n✓ 汇总报告已保存至: {summary_path}")

def main():
    """主函数"""
    # 加载原始数据
    df = load_data()

    # 生成版本0（原始数据）
    v0_info = generate_version_0(df)

    # 准备清洗后的数据（填充缺失值）
    print("\n" + "=" * 80)
    print("应用确定性清洗：填充缺失值")
    print("=" * 80)
    df_filled, fill_stats = fill_missing_values(df)
    print(f"\n✓ 清洗后样本数: {len(df_filled):,} (保留所有样本，填充缺失值)")

    # 生成4个清洗版本（2×2组合）
    v1_info = generate_version_1(df_filled)
    v2_info = generate_version_2(df_filled)
    v3_info = generate_version_3(df_filled)
    v4_info = generate_version_4(df_filled)

    # 生成汇总报告
    versions_info = [v0_info, v1_info, v2_info, v3_info, v4_info]
    generate_summary(versions_info)

    print("\n" + "=" * 80)
    print("✓ 所有版本生成完成！")
    print("=" * 80)
    print("\n下一步:")
    print("  1. 修改RRL支持回归任务")
    print("  2. 选择实验方案（快速对比3个版本 或 完整探索5个版本）")
    print("  3. 训练并评估各版本")
    print("  4. 对比分析性能")

if __name__ == '__main__':
    main()
