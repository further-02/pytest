import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 从PDF中提取的表格数据
data = {
    'Country': ['Australia', 'Canada', 'France', 'Germany', 'Netherlands',
                'New Zealand', 'Norway', 'Sweden', 'Switzerland', 'UK', 'US'],
    'Code': ['AUS', 'CAN', 'FRA', 'GER', 'NETH', 'NZ', 'NOR', 'SWE', 'SWIZ', 'UK', 'US'],
    'Overall': [3, 10, 8, 5, 2, 6, 1, 7, 9, 4, 11],
    'Access to Care': [8, 9, 7, 3, 1, 5, 2, 6, 10, 4, 11],
    'Care Process': [6, 4, 10, 9, 3, 1, 8, 11, 7, 5, 2],
    'Administrative Efficiency': [2, 7, 6, 9, 8, 3, 1, 5, 10, 4, 11],
    'Equity': [1, 10, 7, 2, 5, 9, 8, 6, 3, 4, 11],
    'Health Care Outcomes': [1, 10, 6, 7, 4, 8, 2, 5, 3, 9, 11]
}

df = pd.DataFrame(data)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 整体排名条形图
plt.figure(figsize=(12, 8))
colors = ['#2E8B57' if x != 'US' else '#DC143C' for x in df['Country']]
bars = plt.barh(df['Country'], df['Overall'], color=colors)
plt.xlabel('排名 (1=最好, 11=最差)')
plt.title('11个国家医疗系统整体排名\n(美国排名最后)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()  # 让最好的排名在顶部

# 在条形上添加数值
for bar, value in zip(bars, df['Overall']):
    plt.text(bar.get_width() - 0.3, bar.get_y() + bar.get_height() / 2,
             f'{value}', ha='center', va='center', color='white', fontweight='bold')

plt.tight_layout()
plt.show()

# 2. 各维度排名热力图
categories = ['Access to Care', 'Care Process', 'Administrative Efficiency', 'Equity', 'Health Care Outcomes']
heatmap_data = df[categories].values
countries = df['Code'].values

plt.figure(figsize=(12, 8))
im = plt.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto')  # 红色表示差，绿色表示好

# 添加数值标签
for i in range(len(countries)):
    for j in range(len(categories)):
        text = plt.text(j, i, heatmap_data[i, j],
                        ha="center", va="center", color="black", fontweight='bold')

plt.colorbar(im, label='排名 (1=最好, 11=最差)')
plt.xticks(range(len(categories)), [cat.replace(' ', '\n') for cat in categories], rotation=0)
plt.yticks(range(len(countries)), countries)
plt.title('各国在不同医疗维度的排名表现', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# 3. 美国与其他国家对比雷达图
categories_radar = ['Access to Care', 'Care Process', 'Administrative Efficiency', 'Equity', 'Health Care Outcomes']

# 选择几个代表性国家：美国(最差)、挪威(最好)、德国(中等)、澳大利亚(前三)
selected_countries = ['US', 'Norway', 'Germany', 'Australia']
selected_data = df[df['Code'].isin(['US', 'NOR', 'GER', 'AUS'])]

# 准备雷达图数据
angles = np.linspace(0, 2 * np.pi, len(categories_radar), endpoint=False).tolist()
angles += angles[:1]  # 闭合图形

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
colors = ['#DC143C', '#2E8B57', '#1E90FF', '#FF8C00']  # 红, 绿, 蓝, 橙

for idx, (_, country) in enumerate(selected_data.iterrows()):
    values = country[categories_radar].values.tolist()
    values += values[:1]  # 闭合图形

    # 反转排名：让1(最好)在外圈，11(最差)在内圈
    # 使用12-value来反转，因为排名1-11
    inverted_values = [12 - v for v in values]

    ax.plot(angles, inverted_values, 'o-', linewidth=2,
            label=country['Country'], color=colors[idx])
    ax.fill(angles, inverted_values, alpha=0.1, color=colors[idx])

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels([cat.replace(' ', '\n') for cat in categories_radar])
ax.set_ylim(0, 11)

# 设置y轴标签为原始排名
y_ticks = [1, 3, 5, 7, 9, 11]
y_tick_labels = ['11', '9', '7', '5', '3', '1']  # 反转标签
ax.set_yticks([12 - y for y in y_ticks])
ax.set_yticklabels(y_tick_labels)

plt.title('美国与代表性国家医疗系统\n多维度对比雷达图', size=14, fontweight='bold', ha='center')
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
plt.tight_layout()
plt.show()

# 4. 各维度排名分布箱线图
plt.figure(figsize=(12, 8))
box_data = [df[col] for col in categories]

box_plot = plt.boxplot(box_data, labels=[cat.replace(' ', '\n') for cat in categories],
                       patch_artist=True)

# 设置箱线图颜色
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)

# 标记美国的位置
for i, category in enumerate(categories):
    us_rank = df[df['Code'] == 'US'][category].values[0]
    plt.plot(i + 1, us_rank, 'ro', markersize=8, markeredgecolor='black')
    plt.text(i + 1, us_rank + 0.2, 'US', ha='center', va='bottom', fontweight='bold')

plt.ylabel('排名 (1=最好, 11=最差)')
plt.title('各医疗维度排名分布及美国表现\n(红点表示美国排名)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 5. 美国在各维度具体表现条形图
us_data = df[df['Code'] == 'US'].iloc[0]
us_scores = [us_data[cat] for cat in categories]

plt.figure(figsize=(10, 6))
bars = plt.bar(range(len(categories)), us_scores,
               color=['#DC143C' if x > 8 else '#FFA07A' for x in us_scores])

plt.xlabel('医疗维度')
plt.ylabel('排名 (1=最好, 11=最差)')
plt.title('美国在各医疗维度的具体排名表现', fontsize=14, fontweight='bold')
plt.xticks(range(len(categories)), [cat.replace(' ', '\n') for cat in categories])

# 在条形上添加数值和评价
for i, (bar, score) in enumerate(zip(bars, us_scores)):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
             f'{score}', ha='center', va='bottom', fontweight='bold')
    if score == 2:
        plt.text(bar.get_x() + bar.get_width() / 2, -0.5,
                 '相对较好', ha='center', va='top', fontweight='bold', color='green')
    elif score == 11:
        plt.text(bar.get_x() + bar.get_width() / 2, -0.5,
                 '最差', ha='center', va='top', fontweight='bold', color='red')

plt.ylim(0, 12)
plt.tight_layout()
plt.show()

print("数据摘要:")
print(f"美国整体排名: {df[df['Code'] == 'US']['Overall'].values[0]}/11")
print(f"表现最好的维度: Care Process (第2名)")
print(f"表现最差的维度: 多个维度排名第11 (Access to Care, Administrative Efficiency, Equity, Health Care Outcomes)")