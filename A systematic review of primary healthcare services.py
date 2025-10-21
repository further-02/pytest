import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建表格1数据 - 心理测量特性
data1 = {
    'Country': ['Turkey', 'Germany', 'United States', 'Belgium', 'France', 'Luxembourg', 'Austria',
                'Australia', 'Switzerland', 'Spain', 'Hungary', 'Norway', 'New Zealand', 'Ireland',
                'Portugal', 'Mexico', 'Slovenia', 'Italy', 'Denmark', 'Latvia', 'Netherlands',
                'Canada', 'Slovakia', 'Estonia', 'Czech', 'United Kingdom', 'Israel', 'Finland',
                'Greece', 'Poland', 'Korea', 'Lithuania', 'Japan', 'Chile', 'Sweden'],
    'Language': ['Turkish', 'German', 'English', 'Dutch/French', 'French', 'Luxembourgish',
                 'Austrian', 'English', 'German', 'Spanish', 'Hungarian', 'Norwegian', 'English',
                 'English', 'Portuguese', 'Spanish', 'Slovenian', 'Italian', 'Danish', 'Latvian',
                 'Dutch', 'English', 'Slovak', 'Estonian', 'Czech', 'English', 'Hebrew', 'Finnish',
                 'Greek', 'Polish', 'Korean', 'Lithuanian', 'Japanese', 'Spanish', 'Swedish'],
    'Mean': [3.08, 3.01, 2.99, 2.98, 2.96, 2.92, 2.92, 2.90, 2.87, 2.87, 2.87, 2.85, 2.82, 2.82,
             2.80, 2.78, 2.77, 2.77, 2.77, 2.76, 2.75, 2.75, 2.71, 2.69, 2.69, 2.61, 2.61, 2.56,
             2.54, 2.52, 2.50, 2.47, 2.46, 2.41, 2.28],
    'Alpha': [0.95, 0.90, 0.94, 0.90, 0.91, 0.88, 0.90, 0.93, 0.90, 0.92, 0.92, 0.92, 0.92, 0.91,
              0.91, 0.92, 0.93, 0.94, 0.91, 0.91, 0.92, 0.95, 0.91, 0.90, 0.90, 0.92, 0.89, 0.91,
              0.93, 0.90, 0.92, 0.89, 0.94, 0.92, 0.93],
    'PEI_Corr': [0.74, 0.64, 0.69, 0.57, 0.62, 0.67, 0.69, 0.70, 0.62, 0.74, 0.63, 0.67, 0.62, 0.68,
                 0.67, 0.68, 0.69, 0.73, 0.61, 0.63, 0.59, 0.72, 0.71, 0.57, 0.68, 0.71, 0.63, 0.68,
                 0.67, 0.68, 0.67, 0.63, 0.73, 0.73, 0.74]
}

df1 = pd.DataFrame(data1)

# 创建表格2数据 - 各项目得分
data2 = {
    'Country': ['Turkey', 'Germany', 'United States', 'Belgium', 'France', 'Austria', 'Luxembourg',
                'Australia', 'Spain', 'Hungary', 'Switzerland', 'Norway', 'Ireland', 'New Zealand',
                'Portugal', 'Mexico', 'Denmark', 'Slovenia', 'Italy', 'Latvia', 'Canada', 'Netherlands',
                'Slovakia', 'Estonia', 'Czech Republic', 'United Kingdom', 'Israel', 'Finland', 'Greece', 'Poland',
                'South Korea', 'Lithuania', 'Japan', 'Chile', 'Sweden'],
    'Total': [3.08, 3.01, 2.99, 2.98, 2.96, 2.92, 2.92, 2.90, 2.87, 2.87, 2.87, 2.85, 2.82, 2.82,
              2.80, 2.78, 2.77, 2.77, 2.77, 2.76, 2.75, 2.75, 2.71, 2.69, 2.69, 2.61, 2.61, 2.56,
              2.54, 2.52, 2.50, 2.47, 2.46, 2.41, 2.28],
    'Q1': [3.3, 3.2, 3.1, 3.1, 3.2, 3.3, 3.3, 3.1, 3.2, 3.1, 3.1, 3.0, 3.1, 3.0, 3.1, 3.1, 3.0, 2.9,
           3.0, 3.0, 2.9, 2.9, 3.1, 3.0, 3.2, 2.9, 2.8, 3.0, 2.6, 2.9, 3.0, 2.9, 2.7, 2.7, 2.8],
    'Q2': [3.2, 3.1, 3.1, 3.0, 3.0, 3.1, 3.0, 3.1, 3.0, 2.8, 3.0, 3.0, 3.0, 3.1, 3.0, 3.1, 2.9, 2.8,
           2.8, 2.7, 2.9, 2.8, 3.0, 2.9, 3.0, 3.0, 2.8, 3.0, 2.6, 2.7, 2.9, 2.7, 2.7, 2.6, 2.7],
    'Q3': [3.2, 3.2, 3.2, 3.3, 3.3, 3.2, 3.1, 3.3, 3.3, 3.2, 3.1, 3.0, 3.2, 3.2, 3.2, 3.2, 3.1, 3.1,
           3.0, 3.0, 3.0, 3.0, 3.1, 3.1, 3.2, 3.0, 3.1, 2.9, 2.9, 2.9, 2.8, 2.9, 2.6, 2.9, 2.6],
    'Q4': [2.9, 3.0, 3.0, 2.9, 3.0, 3.0, 2.9, 3.0, 3.0, 3.0, 3.0, 2.8, 2.7, 2.9, 2.9, 2.7, 2.8, 2.8,
           2.6, 2.9, 2.7, 2.7, 2.8, 2.9, 2.8, 2.8, 2.5, 2.7, 2.5, 2.6, 2.6, 2.6, 2.3, 2.5, 2.5],
    'Q5': [2.8, 3.3, 3.0, 3.2, 3.2, 3.2, 3.0, 3.0, 2.9, 3.0, 3.0, 2.9, 2.9, 2.8, 2.9, 2.8, 2.6, 2.9,
           3.0, 2.8, 2.8, 2.8, 2.9, 2.6, 2.7, 2.5, 2.7, 2.3, 2.6, 2.8, 2.1, 2.5, 2.4, 2.2, 2.0],
    'Q6': [2.9, 2.6, 2.5, 2.6, 2.5, 2.4, 2.5, 2.5, 2.2, 2.8, 2.4, 2.6, 2.4, 2.3, 2.1, 2.2, 2.9, 2.3,
           2.2, 2.6, 2.4, 2.2, 2.2, 2.4, 1.8, 2.1, 2.1, 1.7, 2.4, 2.1, 2.0, 1.8, 2.1, 1.7, 1.8],
    'Q7': [3.1, 3.0, 2.8, 3.0, 2.8, 2.9, 2.9, 2.8, 2.8, 2.8, 2.9, 2.9, 2.8, 2.8, 2.8, 2.5, 2.8, 2.9,
           2.7, 2.6, 2.7, 2.8, 2.8, 2.7, 2.8, 2.5, 2.8, 2.7, 2.5, 2.4, 2.3, 2.5, 2.6, 2.1, 2.1],
    'Q8': [3.0, 2.8, 2.9, 3.0, 2.9, 2.6, 2.7, 2.8, 2.7, 2.8, 2.7, 2.7, 2.8, 2.7, 2.6, 2.7, 2.6, 2.7,
           2.8, 2.7, 2.7, 2.8, 2.6, 2.5, 2.6, 2.4, 2.5, 2.3, 2.6, 2.2, 2.4, 2.2, 2.3, 2.3, 1.8],
    'Q9': [3.1, 3.0, 2.9, 2.8, 2.5, 2.7, 2.7, 2.7, 2.7, 2.4, 2.7, 2.7, 2.5, 2.5, 2.4, 2.5, 2.5, 2.6,
           2.8, 2.5, 2.5, 2.7, 2.0, 2.5, 2.4, 2.4, 2.0, 2.1, 2.2, 2.2, 2.3, 2.1, 2.4, 2.3, 2.0],
    'Q10': [3.2, 3.0, 3.2, 3.1, 3.2, 3.0, 3.1, 3.0, 3.0, 3.0, 2.9, 2.9, 2.9, 2.9, 3.0, 3.0, 2.7, 2.9,
            2.9, 2.9, 2.9, 2.8, 2.8, 2.7, 2.8, 2.7, 2.8, 2.8, 2.7, 2.5, 2.6, 2.7, 2.5, 2.7, 2.4],
    'Q11': [3.1, 2.9, 3.0, 2.9, 2.9, 2.7, 2.8, 2.8, 2.8, 2.6, 2.8, 2.8, 2.7, 2.7, 2.8, 2.7, 2.8, 2.6,
            2.7, 2.6, 2.7, 2.7, 2.5, 2.4, 2.3, 2.5, 2.6, 2.7, 2.5, 2.3, 2.5, 2.3, 2.5, 2.4, 2.3]
}

df2 = pd.DataFrame(data2)

# 问题描述
question_descriptions = {
    'Q1': 'Easy to get care',
    'Q2': 'Provides most care',
    'Q3': 'Considers all health factors',
    'Q4': 'Coordinates multi-place care',
    'Q5': 'Knows me as a person',
    'Q6': 'Long-term relationship',
    'Q7': 'Stands up for me',
    'Q8': 'Considers family context',
    'Q9': 'Informed by community',
    'Q10': 'Helps stay healthy',
    'Q11': 'Helps meet goals'
}

print("数据加载完成，开始绘制图表...")

# 1. 各国PCPCM总分条形图
plt.figure(figsize=(14, 10))
plt.subplot(2, 2, 1)
sorted_df = df1.sort_values('Mean', ascending=True)
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_df)))
bars = plt.barh(range(len(sorted_df)), sorted_df['Mean'], color=colors)
plt.yticks(range(len(sorted_df)), sorted_df['Country'])
plt.xlabel('PCPCM Mean Score')
plt.title('PCPCM Mean Scores by Country (Ranked)')
plt.grid(axis='x', alpha=0.3)

# 添加数值标签
for i, bar in enumerate(bars):
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
             f'{bar.get_width():.2f}', va='center', fontsize=8)

# 2. 信度系数(Alpha)分布
plt.subplot(2, 2, 2)
plt.hist(df1['Alpha'], bins=15, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(df1['Alpha'].mean(), color='red', linestyle='--', label=f'Mean: {df1["Alpha"].mean():.3f}')
plt.xlabel('Cronbach\'s Alpha')
plt.ylabel('Frequency')
plt.title('Distribution of Reliability Coefficients (Alpha)')
plt.legend()
plt.grid(alpha=0.3)

# 3. PCPCM得分与PEI相关性的散点图
plt.subplot(2, 2, 3)
plt.scatter(df1['Mean'], df1['PEI_Corr'], alpha=0.7, s=60, c=df1['Alpha'], cmap='viridis')
plt.xlabel('PCPCM Mean Score')
plt.ylabel('PEI Correlation')
plt.title('PCPCM Scores vs PEI Correlations')
plt.colorbar(label='Alpha Coefficient')
plt.grid(alpha=0.3)

# 添加国家标签（只显示部分国家）
for i, row in df1.iterrows():
    if row['Mean'] > 3.0 or row['Mean'] < 2.5:  # 只显示极端值国家
        plt.annotate(row['Country'], (row['Mean'], row['PEI_Corr']),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.tight_layout()
plt.show()

# 4. 前5名国家的雷达图
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 1, polar=True)

# 选择前5个国家
top5 = df2.head(5)
categories = list(question_descriptions.keys())
N = len(categories)

# 计算角度
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # 闭合图形

# 定义颜色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# 为每个国家绘制雷达图
for i, (idx, row) in enumerate(top5.iterrows()):
    values = row[categories].values.tolist()
    values += values[:1]  # 闭合图形
    plt.plot(angles, values, 'o-', linewidth=2, label=row['Country'], color=colors[i])
    plt.fill(angles, values, alpha=0.1, color=colors[i])

plt.xticks(angles[:-1], [question_descriptions[q] for q in categories], size=8)
plt.yticks([1, 2, 3], ['1', '2', '3'], color="grey", size=7)
plt.ylim(0, 4)
plt.title('Top 5 Countries: Primary Care Dimensions', size=12, y=1.08)
plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0))

# 5. 后5名国家的雷达图
plt.subplot(1, 2, 2, polar=True)

# 选择后5个国家
bottom5 = df2.tail(5)

# 为每个国家绘制雷达图
for i, (idx, row) in enumerate(bottom5.iterrows()):
    values = row[categories].values.tolist()
    values += values[:1]  # 闭合图形
    plt.plot(angles, values, 'o-', linewidth=2, label=row['Country'], color=colors[i])
    plt.fill(angles, values, alpha=0.1, color=colors[i])

plt.xticks(angles[:-1], [question_descriptions[q] for q in categories], size=8)
plt.yticks([1, 2, 3], ['1', '2', '3'], color="grey", size=7)
plt.ylim(0, 4)
plt.title('Bottom 5 Countries: Primary Care Dimensions', size=12, y=1.08)
plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0))

plt.tight_layout()
plt.show()

# 6. 世界地图可视化
try:
    import plotly.express as px
    import pycountry

    # 创建国家名称映射（将数据中的国家名称转换为标准名称）
    country_mapping = {
        'United States': 'United States of America',
        'South Korea': 'Korea, Republic of',
        'Czech Republic': 'Czechia',
        'United Kingdom': 'United Kingdom'
    }

    # 准备地图数据
    map_data = df2[['Country', 'Total']].copy()

    # 转换国家名称
    map_data['Country_ISO'] = map_data['Country'].apply(
        lambda x: country_mapping[x] if x in country_mapping else x
    )


    # 添加ISO Alpha-3代码
    def get_iso_alpha_3(country_name):
        try:
            return pycountry.countries.lookup(country_name).alpha_3
        except:
            return None


    map_data['ISO_alpha'] = map_data['Country_ISO'].apply(get_iso_alpha_3)

    # 创建世界地图
    fig = px.choropleth(
        map_data,
        locations='ISO_alpha',
        color='Total',
        hover_name='Country',
        color_continuous_scale='RdYlGn',
        range_color=(2.0, 3.2),
        title='PCPCM Scores by Country - World Map',
        labels={'Total': 'PCPCM Score'}
    )

    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='equirectangular'
        )
    )

    # 显示地图
    fig.show()

except ImportError:
    print("注意: 未安装plotly和pycountry库，无法显示世界地图。")
    print("要安装这些库，请运行: pip install plotly pycountry")

    # 如果没有plotly，创建一个简单的替代可视化
    plt.figure(figsize=(12, 8))

    # 使用条形图作为替代
    sorted_map_data = df2.sort_values('Total', ascending=False)
    colors_map = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_map_data)))
    bars_map = plt.bar(range(len(sorted_map_data)), sorted_map_data['Total'], color=colors_map)

    plt.xticks(range(len(sorted_map_data)), sorted_map_data['Country'], rotation=90)
    plt.ylabel('PCPCM Score')
    plt.title('PCPCM Scores by Country (World Map Alternative)')
    plt.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for i, bar in enumerate(bars_map):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()

# 7. 热力图：各国在各维度的表现
plt.figure(figsize=(16, 12))

# 准备热力图数据
heatmap_data = df2.set_index('Country')[categories]
plt.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=1, vmax=4)

plt.colorbar(label='Score (1-4)')
plt.xticks(range(len(categories)), [f'Q{i + 1}' for i in range(len(categories))], rotation=45)
plt.yticks(range(len(heatmap_data)), heatmap_data.index)
plt.xlabel('Primary Care Dimensions')
plt.ylabel('Countries')
plt.title('Heatmap of Primary Care Dimension Scores by Country')

# 添加数值
for i in range(len(heatmap_data)):
    for j in range(len(categories)):
        plt.text(j, i, f'{heatmap_data.iloc[i, j]:.1f}',
                 ha='center', va='center', fontsize=7,
                 color='white' if heatmap_data.iloc[i, j] < 2 else 'black')

plt.tight_layout()
plt.show()

print("图表绘制完成！")
print(f"\n研究摘要:")
print(f"- 调查国家数量: {len(df1)}个OECD国家")
print(f"- PCPCM总分范围: {df1['Mean'].min():.2f} (瑞典) 到 {df1['Mean'].max():.2f} (土耳其)")
print(f"- 平均信度系数(Alpha): {df1['Alpha'].mean():.3f}")
print(f"- 前5名国家: {', '.join(df1.nlargest(5, 'Mean')['Country'].tolist())}")
print(f"- 后5名国家: {', '.join(df1.nsmallest(5, 'Mean')['Country'].tolist())}")