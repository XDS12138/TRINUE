import matplotlib.pyplot as plt
import numpy as np

# 数据集顺序（Ours 放第一个）
datasets = ['Ours', 'UIEB', 'RUIE', 'EUVP', 'UWCNN', 'LSUI', 'SeaThru', 'Varos', 'Atlantis']
paired_counts = np.array([
    100000,  # Ours
    890,     # UIEB
    726,     # RUIE
    2185,    # EUVP
    1449,    # UWCNN
    4279,    # LSUI
    1218,    # SeaThru
    4315,    # Varos
    3200     # Atlantis
])
resolutions = np.array([
    3840 * 2160,           # Ours
    868.90 * 594.79,       # UIEB
    400.00 * 300.00,       # RUIE
    320.00 * 240.00,       # EUVP
    620.00 * 460.00,       # UWCNN
    447.04 * 309.17,       # LSUI
    2369.59 * 1576.59,     # SeaThru
    1280.00 * 720.00,      # Varos
    672.00 * 512.00        # Atlantis
])

# 单位换算
paired_counts_k = paired_counts / 1000.0
resolutions_m = resolutions / 1e6

# 创建图形
fig, ax1 = plt.subplots(figsize=(12, 6), facecolor='none')
ax1.set_ylim(0, 12)

# 左轴：图像数量（单位：k）
bar_color = 'tab:blue'
# 去除 xlabel，防止与底部标题冲突
# ax1.set_xlabel('Dataset')
ax1.set_ylabel('Number of Paired Images (K)', color=bar_color)
bars = ax1.bar(datasets, paired_counts_k, color=bar_color, alpha=0.6)
ax1.tick_params(axis='y', labelcolor=bar_color)
ax1.set_yticks(np.arange(0, 13, 2))
ax1.set_yticklabels([f"{int(t)}k" for t in np.arange(0, 13, 2)])

# 数值标签：加粗、字体大一号
for bar, count_k, label in zip(bars, paired_counts_k, datasets):
    if label == "Ours":
        ax1.text(bar.get_x() + bar.get_width() / 2, 3.5,
                 '100K+', ha='center', va='bottom',
                 fontsize=11, color='black', fontweight='bold')
    else:
        ax1.text(bar.get_x() + bar.get_width() / 2, count_k + 0.2,
                 f'{count_k:.1f}k', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 右轴：分辨率（单位：百万像素）
ax2 = ax1.twinx()
line_color = 'tab:red'
ax2.set_ylabel('Average Resolution (MP)', color=line_color)
ax2.plot(datasets, resolutions_m, color=line_color, marker='o', linewidth=2)
ax2.tick_params(axis='y', labelcolor=line_color)
ax2.set_yticks(np.arange(0, 9, 1))
ax2.set_yticklabels([f"{t}M" for t in np.arange(0, 9, 1)])

# 移动标题到底部中央
fig.subplots_adjust(bottom=0.15)
plt.figtext(0.5, 0.02, 'Paired Underwater Image Datasets: Sample Size and Resolution',
            ha='center', fontsize=12)

# 保存为透明 SVG
plt.tight_layout()
plt.savefig('/mnt/data/paired_dataset_resolution_final_bottomtitle.svg', format='svg', transparent=True)
plt.show()
