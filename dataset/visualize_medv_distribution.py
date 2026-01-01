#!/usr/bin/env python3
"""
可视化 Boston Housing MEDV 分布
对比原始值和log变换后的分布
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_distribution_comparison():
    """对比MEDV原始值和log变换后的分布"""

    # 加载数据
    df = pd.read_csv('../../data/boston_housing.csv')

    # 删除缺失值
    df_clean = df.dropna()

    medv_original = df_clean['MEDV']
    medv_log = np.log(df_clean['MEDV'])

    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Boston Housing MEDV Distribution Analysis', fontsize=16, fontweight='bold')

    # ========== 第1行：原始MEDV ==========
    # 1. 直方图
    axes[0, 0].hist(medv_original, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 0].axvline(medv_original.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {medv_original.mean():.2f}')
    axes[0, 0].axvline(medv_original.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {medv_original.median():.2f}')
    axes[0, 0].set_xlabel('MEDV (House Price in $1000s)', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Original MEDV - Histogram', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # 添加统计信息
    skew_orig = medv_original.skew()
    kurt_orig = medv_original.kurtosis()
    axes[0, 0].text(0.65, 0.95, f'Skewness: {skew_orig:.3f}\nKurtosis: {kurt_orig:.3f}',
                    transform=axes[0, 0].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. 箱线图
    bp = axes[0, 1].boxplot(medv_original, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('skyblue')
    axes[0, 1].set_ylabel('MEDV (House Price in $1000s)', fontsize=11)
    axes[0, 1].set_title('Original MEDV - Boxplot', fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)

    # 添加截尾标记
    censored_count = (medv_original == 50.0).sum()
    axes[0, 1].text(0.5, 0.95, f'Censored at $50k: {censored_count} samples ({100*censored_count/len(medv_original):.2f}%)',
                    transform=axes[0, 1].transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # 3. Q-Q图
    stats.probplot(medv_original, dist="norm", plot=axes[0, 2])
    axes[0, 2].set_title('Original MEDV - Q-Q Plot', fontsize=12, fontweight='bold')
    axes[0, 2].grid(alpha=0.3)
    axes[0, 2].get_lines()[0].set_markerfacecolor('skyblue')
    axes[0, 2].get_lines()[0].set_markeredgecolor('blue')
    axes[0, 2].get_lines()[0].set_markersize(4)

    # ========== 第2行：log(MEDV) ==========
    # 1. 直方图
    axes[1, 0].hist(medv_log, bins=30, edgecolor='black', alpha=0.7, color='lightcoral')
    axes[1, 0].axvline(medv_log.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {medv_log.mean():.3f}')
    axes[1, 0].axvline(medv_log.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {medv_log.median():.3f}')
    axes[1, 0].set_xlabel('log(MEDV)', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].set_title('log(MEDV) - Histogram', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # 添加统计信息
    skew_log = medv_log.skew()
    kurt_log = medv_log.kurtosis()
    axes[1, 0].text(0.65, 0.95, f'Skewness: {skew_log:.3f}\nKurtosis: {kurt_log:.3f}',
                    transform=axes[1, 0].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. 箱线图
    bp = axes[1, 1].boxplot(medv_log, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightcoral')
    axes[1, 1].set_ylabel('log(MEDV)', fontsize=11)
    axes[1, 1].set_title('log(MEDV) - Boxplot', fontsize=12, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)

    # 3. Q-Q图
    stats.probplot(medv_log, dist="norm", plot=axes[1, 2])
    axes[1, 2].set_title('log(MEDV) - Q-Q Plot', fontsize=12, fontweight='bold')
    axes[1, 2].grid(alpha=0.3)
    axes[1, 2].get_lines()[0].set_markerfacecolor('lightcoral')
    axes[1, 2].get_lines()[0].set_markeredgecolor('red')
    axes[1, 2].get_lines()[0].set_markersize(4)

    plt.tight_layout()

    # 保存图片
    output_path = '../../docs/medv_distribution_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 图片已保存至: {output_path}")

    # 显示图片（如果在交互环境）
    # plt.show()

    return medv_original, medv_log

def print_statistics(medv_original, medv_log):
    """打印详细统计信息"""
    print("\n" + "="*80)
    print("MEDV 分布统计对比")
    print("="*80)

    print(f"\n{'统计量':<20} {'原始MEDV':>15} {'log(MEDV)':>15} {'改善'}")
    print("-"*80)

    # 基本统计
    print(f"{'样本数':<20} {len(medv_original):>15,} {len(medv_log):>15,}")
    print(f"{'均值':<20} {medv_original.mean():>15.3f} {medv_log.mean():>15.3f}")
    print(f"{'标准差':<20} {medv_original.std():>15.3f} {medv_log.std():>15.3f}")
    print(f"{'中位数':<20} {medv_original.median():>15.3f} {medv_log.median():>15.3f}")
    print(f"{'最小值':<20} {medv_original.min():>15.3f} {medv_log.min():>15.3f}")
    print(f"{'最大值':<20} {medv_original.max():>15.3f} {medv_log.max():>15.3f}")

    # 偏度和峰度
    skew_orig = medv_original.skew()
    skew_log = medv_log.skew()
    kurt_orig = medv_original.kurtosis()
    kurt_log = medv_log.kurtosis()

    print(f"\n{'偏度 (Skewness)':<20} {skew_orig:>15.3f} {skew_log:>15.3f} {'✓ 改善' if abs(skew_log) < abs(skew_orig) else '✗'}")
    print(f"{'峰度 (Kurtosis)':<20} {kurt_orig:>15.3f} {kurt_log:>15.3f} {'✓ 改善' if abs(kurt_log) < abs(kurt_orig) else '✗'}")

    # 正态性检验 (Shapiro-Wilk)
    _, p_orig = stats.shapiro(medv_original)
    _, p_log = stats.shapiro(medv_log)

    print(f"\n{'Shapiro-Wilk p-value':<20} {p_orig:>15.6f} {p_log:>15.6f} {'✓ 更接近正态' if p_log > p_orig else '✗'}")
    print(f"  (p>0.05 表示接近正态分布)")

    # 截尾分析
    censored = (medv_original == 50.0).sum()
    print(f"\n{'截尾样本 (=$50k)':<20} {censored:>15} {censored:>15} ({100*censored/len(medv_original):.2f}%)")

    print("\n" + "="*80)
    print("分析结论")
    print("="*80)

    print("\n原始MEDV:")
    print(f"  - 偏度 = {skew_orig:.3f} (右偏，>1表示明显右偏)")
    print(f"  - 峰度 = {kurt_orig:.3f} (尖峭程度)")
    print(f"  - Shapiro-Wilk p = {p_orig:.6f} ({'拒绝正态' if p_orig < 0.05 else '接近正态'})")
    print(f"  - 16个样本被截尾在$50k (3.16%)")

    print(f"\nlog(MEDV):")
    print(f"  - 偏度 = {skew_log:.3f} ({'显著改善！' if abs(skew_log) < abs(skew_orig) * 0.5 else '有所改善'})")
    print(f"  - 峰度 = {kurt_log:.3f}")
    print(f"  - Shapiro-Wilk p = {p_log:.6f} ({'拒绝正态' if p_log < 0.05 else '接近正态'})")
    print(f"  - 分布更接近正态，有利于回归模型")

    print("\n推荐:")
    if abs(skew_log) < abs(skew_orig) * 0.7:
        print("  ✓ 强烈推荐使用 log(MEDV)")
        print("    理由：偏度显著降低，分布更接近正态")
    else:
        print("  ⚠️  log变换改善有限，可以尝试但需对比效果")

    print("\n注意事项:")
    print("  1. 使用 log(MEDV) 训练后，预测值需要 exp() 反变换")
    print("  2. 评估指标应在反变换后的原始空间计算 RMSE")
    print("  3. log变换对截尾样本的影响较小（log(50)=3.912）")

def main():
    """主函数"""
    print("="*80)
    print("Boston Housing MEDV 分布可视化")
    print("="*80)

    # 生成可视化
    medv_original, medv_log = plot_distribution_comparison()

    # 打印统计信息
    print_statistics(medv_original, medv_log)

    print("\n" + "="*80)
    print("完成！请查看生成的图片")
    print("="*80)

if __name__ == '__main__':
    main()
