#!/usr/bin/env python3
"""
Bank Marketing 数据准备 - 生成所有策略组合版本

生成9个数据集版本：
  1. 原始版本（不清洗）: bank-full.data
  2-9. 清洗版本（8种组合）: bank-full-clean-{suffix}.data

策略组合：
  - job:       'drop' (删除unknown行) / 'keep' (保留unknown)
  - education: 'drop' (删除unknown行) / 'keep' (保留unknown)
  - contact:   'keep' (保留unknown) / 'indicator' (创建指示变量+填充)

后缀命名规则：
  jd = job_drop, jk = job_keep
  ed = edu_drop, ek = edu_keep
  ck = contact_keep, ci = contact_indicator

示例：bank-full-clean-jd-ed-ck.data
"""

import pandas as pd
from itertools import product


def apply_certain_cleaning(df):
    """
    应用确定性清洗（所有版本都执行）

    1. 删除 poutcome (81.75% unknown)
    2. 转换 pdays → contacted_before + pdays_cleaned
    3. 删除 duration=0 的样本
    4. 删除重复行

    返回: (清洗后的df, 统计信息dict)
    """
    df = df.copy()
    stats = {}

    # 1. 删除 poutcome
    if 'poutcome' in df.columns:
        df = df.drop(columns=['poutcome'])
        stats['poutcome'] = 'dropped'

    # 2. 转换 pdays
    if 'pdays' in df.columns:
        df['contacted_before'] = (df['pdays'] != -1).astype(int)
        df['pdays_cleaned'] = df['pdays'].apply(lambda x: 999 if x == -1 else x)
        df = df.drop(columns=['pdays'])
        stats['contacted_before_count'] = int(df['contacted_before'].sum())

    # 3. 删除 duration=0
    zero_count = (df['duration'] == 0).sum()
    df = df[df['duration'] > 0]
    stats['duration_zero_dropped'] = int(zero_count)

    # 4. 删除重复行
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df = df.drop_duplicates()
    stats['duplicates_dropped'] = int(dup_count)

    return df, stats


def apply_strategy(df, job_strat, edu_strat, contact_strat):
    """
    应用特定策略组合

    参数:
        job_strat: 'drop' 或 'keep'
        edu_strat: 'drop' 或 'keep'
        contact_strat: 'keep' 或 'indicator'

    返回: (处理后的df, 统计信息dict)
    """
    df = df.copy()
    stats = {}

    # 1. 处理 job
    job_unknown = (df['job'] == 'unknown').sum()
    if job_strat == 'drop':
        df = df[df['job'] != 'unknown']
        stats['job'] = f'dropped {job_unknown} rows'
    else:
        stats['job'] = f'kept {job_unknown} as category'

    # 2. 处理 education
    edu_unknown = (df['education'] == 'unknown').sum()
    if edu_strat == 'drop':
        df = df[df['education'] != 'unknown']
        stats['education'] = f'dropped {edu_unknown} rows'
    else:
        stats['education'] = f'kept {edu_unknown} as category'

    # 3. 处理 contact
    contact_unknown = (df['contact'] == 'unknown').sum()
    if contact_strat == 'indicator':
        df['contact_was_unknown'] = (df['contact'] == 'unknown').astype(int)
        mode_val = df[df['contact'] != 'unknown']['contact'].mode()[0]
        df['contact'] = df['contact'].replace('unknown', mode_val)
        stats['contact'] = f'indicator created, filled with {mode_val}'
    else:
        stats['contact'] = f'kept {contact_unknown} as category'

    return df, stats


def generate_suffix(job_strat, edu_strat, contact_strat):
    """生成文件后缀"""
    j = 'jd' if job_strat == 'drop' else 'jk'
    e = 'ed' if edu_strat == 'drop' else 'ek'
    c = 'ck' if contact_strat == 'keep' else 'ci'
    return f"{j}-{e}-{c}"


def save_to_rrl_format(df, filename_prefix):
    """
    保存为RRL格式 (.data 和 .info 文件)

    参数:
        df: DataFrame
        filename_prefix: 文件名前缀（不含扩展名）
    """
    # 分离特征和标签
    X = df.drop(columns=['y'])
    y = df['y']

    # 保存 .data 文件
    data_path = f'{filename_prefix}.data'
    with open(data_path, 'w') as f:
        for idx in range(len(df)):
            features = []
            for col in X.columns:
                val = X.iloc[idx][col]
                if pd.isna(val):
                    features.append('?')
                else:
                    features.append(str(val))
            label = str(y.iloc[idx])
            line = ','.join(features) + ',' + label
            f.write(line + '\n')

    # 保存 .info 文件
    # 格式：每行 "feature_name feature_type" (空格分隔)
    # 最后一行：目标变量 + "LABEL_POS -1"
    info_path = f'{filename_prefix}.info'
    with open(info_path, 'w') as f:
        # 写入每个特征：特征名 类型（空格分隔）
        for col in X.columns:
            if df[col].dtype in ['int64', 'float64']:
                ftype = 'continuous'
            else:
                ftype = 'discrete'
            f.write(f'{col} {ftype}\n')
        # 写入目标变量：目标名 类型（空格分隔）
        f.write('y discrete\n')
        # 写入标签位置（-1表示最后一列）
        f.write('LABEL_POS -1\n')

    return data_path, info_path


def main():
    """主函数：生成所有数据集版本"""
    print("="*80)
    print("Bank Marketing 数据准备 - 生成所有版本")
    print("="*80)

    # 加载原始数据
    print("\n[1] 加载原始数据...")
    df_original = pd.read_csv('../../data/bank-full.csv', sep=';')
    print(f"    ✓ 原始数据: {len(df_original):,} 样本, {df_original.shape[1]} 特征")

    # 统计信息
    all_versions = []

    # ========== 版本0: 原始数据（不清洗）==========
    print("\n" + "="*80)
    print("[2] 生成版本 0: bank-full (原始数据，不清洗)")
    print("="*80)

    data_path, info_path = save_to_rrl_format(df_original, 'bank-full')
    print(f"    ✓ 已保存: {data_path}")
    print(f"    ✓ 已保存: {info_path}")
    print(f"    样本数: {len(df_original):,}, 特征数: {df_original.shape[1]}")

    all_versions.append({
        'version': 0,
        'name': 'bank-full',
        'description': '原始数据（不清洗）',
        'samples': len(df_original),
        'features': df_original.shape[1],
        'strategies': None,
    })

    # ========== 版本1-8: 清洗后的数据 ==========
    print("\n" + "="*80)
    print("[3] 生成 8 个清洗版本")
    print("="*80)

    job_options = ['drop', 'keep']
    edu_options = ['drop', 'keep']
    contact_options = ['keep', 'indicator']

    combinations = list(product(job_options, edu_options, contact_options))
    print(f"    策略组合数: {len(combinations)}")

    for i, (job_s, edu_s, contact_s) in enumerate(combinations, 1):
        # 先进行确定性清洗
        df_clean, certain_stats = apply_certain_cleaning(df_original)

        # 再应用策略性清洗
        df_final, strategy_stats = apply_strategy(df_clean, job_s, edu_s, contact_s)

        # 生成文件名
        suffix = generate_suffix(job_s, edu_s, contact_s)
        filename = f'bank-full-clean-{suffix}'

        # 保存
        data_path, info_path = save_to_rrl_format(df_final, filename)

        # 打印信息
        print(f"\n    版本 {i}: {filename}")
        print(f"      策略: job={job_s}, education={edu_s}, contact={contact_s}")
        print(f"      样本数: {len(df_final):,}, 特征数: {df_final.shape[1]}")

        # 记录
        all_versions.append({
            'version': i,
            'name': filename,
            'suffix': suffix,
            'description': f'j:{job_s[0]} e:{edu_s[0]} c:{contact_s[0]}',
            'samples': len(df_final),
            'features': df_final.shape[1],
            'strategies': {
                'job': job_s,
                'education': edu_s,
                'contact': contact_s,
            },
            'stats': {**certain_stats, **strategy_stats},
        })

    # ========== 生成汇总报告 ==========
    print("\n" + "="*80)
    print("[4] 生成汇总报告")
    print("="*80)

    summary_path = '../../data/bank_versions_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("Bank Marketing 数据集版本汇总\n")
        f.write("="*80 + "\n\n")
        f.write(f"生成时间: {pd.Timestamp.now()}\n")
        f.write(f"总版本数: {len(all_versions)}\n\n")

        for ver in all_versions:
            f.write(f"\n版本 {ver['version']}: {ver['name']}\n")
            f.write("-"*80 + "\n")
            f.write(f"说明: {ver['description']}\n")
            f.write(f"样本数: {ver['samples']:,}\n")
            f.write(f"特征数: {ver['features']}\n")

            if ver['strategies']:
                f.write(f"策略:\n")
                f.write(f"  - job:       {ver['strategies']['job']}\n")
                f.write(f"  - education: {ver['strategies']['education']}\n")
                f.write(f"  - contact:   {ver['strategies']['contact']}\n")

                if 'stats' in ver:
                    f.write(f"详细统计:\n")
                    for key, val in ver['stats'].items():
                        f.write(f"  - {key}: {val}\n")

    print(f"    ✓ 汇总报告: {summary_path}")

    # ========== 打印版本对照表 ==========
    print("\n" + "="*80)
    print("[5] 版本对照表")
    print("="*80)
    print(f"\n{'版本':<6} {'文件名':<40} {'样本数':<12} {'说明'}")
    print("-"*90)

    for ver in all_versions:
        print(f"{ver['version']:<6} {ver['name']:<40} {ver['samples']:<12,} {ver['description']}")

    # ========== 完成 ==========
    print("\n" + "="*80)
    print("完成！")
    print("="*80)
    print(f"\n✓ 生成了 {len(all_versions)} 个数据集版本")
    print(f"  - 1 个原始版本: bank-full")
    print(f"  - 8 个清洗版本: bank-full-clean-*")
    print(f"\n所有文件保存在: rrl/dataset/")
    print(f"汇总报告: {summary_path}")

    print("\n后缀含义:")
    print("  jd/jk = job_drop/keep")
    print("  ed/ek = education_drop/keep")
    print("  ck/ci = contact_keep/indicator")

    print("\n接下来可以:")
    print("  1. 对每个版本训练 RRL 模型")
    print("  2. 对比不同策略的性能（F1 Score, Accuracy等）")
    print("  3. 选择最优版本用于最终实验")


if __name__ == '__main__':
    main()
