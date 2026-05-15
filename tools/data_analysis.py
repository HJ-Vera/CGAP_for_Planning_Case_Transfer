"""
数据分析模块 — 区域数据聚类分析、可视化、地图
"""

import os
import sys
import re

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from kneed import KneeLocator



# ==================== 字体初始化（模块级，只执行一次）====================

def _setup_font():
    """
    初始化中文字体，返回 (font_prop, font_name)。
    优先使用系统已有字体，失败则尝试下载，最终回退到无衬线字体。
    """
    priority_keywords = ['notosanscjktc', 'notoseriftc', 'msjh', 'uming', 'ukai',
                         'notosanscjk', 'simhei', 'arial unicode']
    system_fonts = fm.findSystemFonts()
    font_path = None

    for kw in priority_keywords:
        matches = [f for f in system_fonts if kw in f.lower().replace('-', '').replace('_', '')]
        if matches:
            font_path = matches[0]
            break

    if font_path is None:
        download_path = os.path.join(os.path.expanduser('~'), '.fonts', 'NotoSansTC-Regular.otf')
        url = ('https://github.com/googlefonts/noto-cjk/raw/main/'
               'Sans/OTF/TraditionalChinese/NotoSansTC-Regular.otf')
        try:
            import urllib.request
            os.makedirs(os.path.dirname(download_path), exist_ok=True)
            urllib.request.urlretrieve(url, download_path)
            if os.path.exists(download_path) and os.path.getsize(download_path) > 1000:
                font_path = download_path
                print(f"字体下载成功: {download_path}")
            else:
                raise RuntimeError("下载文件过小，可能失败")
        except Exception as e:
            print(f"警告: 字体下载失败({e})，将使用系统默认字体，中文可能无法正常显示。")
            matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
            matplotlib.rcParams['axes.unicode_minus'] = False
            return None, 'DejaVu Sans'

    fm.fontManager.addfont(font_path)
    font_prop = fm.FontProperties(fname=font_path)
    font_name = font_prop.get_name()

    matplotlib.rcParams['font.sans-serif'] = [font_name]
    matplotlib.rcParams['axes.unicode_minus'] = False
    sns.set_style("whitegrid")
    sns.set(font=font_name)

    print(f"字体设置成功: {font_name}")
    return font_prop, font_name


FONT_PROP, FONT_NAME = _setup_font()


def _fp():
    """返回全局字体属性对象，避免每次重复创建。"""
    return FONT_PROP


def _lkw(fp):
    """标签用字体参数字典。"""
    return dict(fontproperties=fp) if fp else {}


def _tkw(fp):
    """标题用字体参数字典。"""
    return dict(fontproperties=fp) if fp else {}


# ==================== 辅助函数 ====================


def _auto_optimal_k(inertia: list, k_range) -> int:
    k_list = list(k_range)
    if len(inertia) < 3:
        return 4

    kl = KneeLocator(
        k_list,
        inertia,
        curve="convex",
        direction="decreasing",
        S=0.1,           # 降低阈值，对微小的肘部更敏感 (尝试 0.5 或更低)
        online=True      # 开启在线模式，有助于捕捉后期的局部肘部
    )

    return kl.elbow if kl.elbow is not None else 8 # 找不到就回退到你认为的8


def _safe_percentile_high(z: float) -> float:
    return stats.norm.cdf(z) * 100


def _safe_percentile_low(z: float) -> float:
    return (1 - stats.norm.cdf(z)) * 100


def _shorten(name: str, max_len: int = 12) -> str:
    """截断过长的列名用于坐标轴显示。"""
    return name[:max_len] + '…' if len(name) > max_len else name


# ==================== 数据质量分析 ====================

def _analyze_data_quality(df: pd.DataFrame, numeric_cols: pd.Index) -> dict:
    """
    分析每列的数据质量，返回分组信息：
    - sparse_cols : 零值比例 > 80% 的稀疏列（如机场用地），不参与聚类
    - active_cols : 其余正常参与分析的列
    - col_groups  : 按均值数量级自动分组，用于分组可视化（解决量纲混用问题）
    - zero_ratio  : 每列零值比例
    """
    zero_ratio = (df[numeric_cols] == 0).mean()
    sparse_cols = zero_ratio[zero_ratio > 0.8].index.tolist()
    active_cols = [c for c in numeric_cols if c not in sparse_cols]

    if sparse_cols:
        print(f"\n检测到 {len(sparse_cols)} 个稀疏列（零值 >80%），将单独处理，不参与聚类：")
        for c in sparse_cols:
            print(f"   - {c}  (零值比例: {zero_ratio[c]:.1%})")

    # 按均值数量级分组，避免不同量纲混在同一张图
    means = df[active_cols].mean().abs()
    bins   = [0, 1, 10, 100, 1e4, np.inf]
    labels = ['0-1', '1-10', '10-100', '100-10000', '>10000']
    col_groups = {}
    for label, low, high in zip(labels, bins[:-1], bins[1:]):
        cols_in = [c for c in active_cols if low <= means.get(c, 0) < high]
        if cols_in:
            col_groups[label] = cols_in

    print(f"\n按数值量级自动分组（用于分组可视化，解决量纲混用问题）：")
    for g, cols in col_groups.items():
        print(f"   量级 {g}: {len(cols)} 列")

    return {
        'sparse_cols': sparse_cols,
        'active_cols': active_cols,
        'col_groups' : col_groups,
        'zero_ratio' : zero_ratio,
    }


# ==================== 可视化模块 ====================

def _plot_grouped_distribution(df, col_groups, fp, save_fig):
    """
    【修复量纲混用问题】
    按数值量级分组，每组生成一张宽幅箱线图：
    - Z-score 标准化，消除量纲差异
    - 箱子宽 = 各区域差异大（对聚类有贡献）
    - 箱子窄 + 右侧散点 = 大多数区域接近0，少数区域极端突出
    - 红色虚线 = 整体均值（Z=0）
    """
    print("\n生成分组标准化分布图（已按量级分组，消除量纲差异）...")
    scaler_vis = StandardScaler()

    for group_label, cols in col_groups.items():
        display_cols = cols[:20]  # 最多展示20列，防止图太挤
        n = len(display_cols)

        z_data = pd.DataFrame(
            scaler_vis.fit_transform(df[display_cols]),
            columns=display_cols
        )

        # 按箱体宽度（IQR）降序排列，差异大的指标排在上面，更容易阅读
        iqr_order = (z_data.quantile(0.75) - z_data.quantile(0.25)).sort_values(ascending=True)
        z_data = z_data[iqr_order.index]
        short = [_shorten(c, 18) for c in z_data.columns]  # 单图可显示更长的标签

        # 根据列数自动调整图高，保证每行有足够空间
        fig_height = max(6, n * 0.55 + 2)
        fig, ax = plt.subplots(figsize=(16, fig_height))

        fig.suptitle(
            f'指标分布分析  |  量级：{group_label}  |  共 {len(cols)} 列，展示前 {n} 列\n'
            f'已 Z-score 标准化（X轴单位：标准差）  |  '
            f'箱子越宽=各区域差异越大  |  右侧圆点=异常偏高的区域',
            fontsize=11, **_tkw(fp)
        )

        z_data.boxplot(ax=ax, vert=False, patch_artist=True,
                       boxprops=dict(facecolor='steelblue', alpha=0.55),
                       medianprops=dict(color='navy', linewidth=2),
                       flierprops=dict(marker='o', markersize=4,
                                       markerfacecolor='#d73027', alpha=0.5))
        ax.set_yticklabels(short, fontsize=10, **_lkw(fp))
        ax.set_xlabel('Z-score（标准差单位，0 = 所有区域的平均水平）', **_lkw(fp))
        ax.axvline(0, color='red', linestyle='--', alpha=0.45, linewidth=1.2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        # save_fig(f'分组分布_量级{group_label}.png')


def _plot_sparse_cols(df, sparse_cols, zero_ratio, fp, save_fig):
    """
    【稀疏列专项分析】
    图1：各稀疏列零值比例总览
    图2：各稀疏列非零值分布直方图（最多展示6列）
    """
    if not sparse_cols:
        return
    print(f"\n生成稀疏列专项分析图（共 {len(sparse_cols)} 列）...")

    # 图1：零值比例
    fig, ax = plt.subplots(figsize=(10, max(4, len(sparse_cols) * 0.5 + 2)))
    ratios = zero_ratio[sparse_cols].sort_values(ascending=True)
    bars = ax.barh(range(len(ratios)), ratios.values * 100,
                   color='#d73027', alpha=0.7)
    ax.set_yticks(range(len(ratios)))
    ax.set_yticklabels([_shorten(c, 20) for c in ratios.index], **_lkw(fp))
    ax.set_xlabel('零值比例 (%)', **_lkw(fp))
    ax.set_title('稀疏列零值比例（这些列零值过多，已排除出聚类）', **_tkw(fp))
    ax.axvline(80, color='black', linestyle='--', alpha=0.5, linewidth=1.2)
    ax.text(81, len(ratios) * 0.95, '80% 阈值', fontsize=9, **_lkw(fp))
    for bar, val in zip(bars, ratios.values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{val:.1%}', va='center', fontsize=9)
    plt.tight_layout()
    # save_fig('稀疏列零值比例.png')

    # 图2：非零值分布
    show_cols = sparse_cols[:6]
    ncols = min(3, len(show_cols))
    nrows = int(np.ceil(len(show_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes = np.array(axes).flatten()
    fig.suptitle('稀疏列：非零值分布（仅统计有值的区域）', fontsize=13, **_tkw(fp))

    for i, col in enumerate(show_cols):
        non_zero = df[col][df[col] != 0]
        if len(non_zero) > 0:
            axes[i].hist(non_zero, bins=20, color='steelblue', alpha=0.75, edgecolor='white')
            axes[i].set_title(
                f'{_shorten(col, 14)}\n非零: {len(non_zero)} 个 ({len(non_zero)/len(df):.1%})',
                fontsize=10, **_tkw(fp)
            )
            axes[i].set_xlabel('值', **_lkw(fp))
            axes[i].set_ylabel('频数', **_lkw(fp))
        else:
            axes[i].text(0.5, 0.5, '全为零值', ha='center', va='center',
                         transform=axes[i].transAxes)
            axes[i].set_title(_shorten(col, 14), **_tkw(fp))

    for j in range(len(show_cols), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    # save_fig('稀疏列非零值分布.png')


def _plot_correlation_clustermap(df, active_cols, desc_stats, output_dir, fp):
    """
    【修复相关性热图】
    使用层次聚类（ward）排序，相似指标自动聚集，结构更清晰。
    原来按方差排序选列会让热图结构混乱。
    """
    print("\n生成层次聚类相关性热图...")

    top_cols = (desc_stats.loc[active_cols].nlargest(40, 'std').index.tolist()
                if len(active_cols) > 40 else active_cols)

    corr = df[top_cols].corr()
    short_labels = [_shorten(c, 12) for c in top_cols]
    corr_display = corr.rename(index=dict(zip(top_cols, short_labels)),
                                columns=dict(zip(top_cols, short_labels)))

    g = sns.clustermap(
        corr_display,
        method='ward', metric='euclidean',
        cmap='coolwarm', center=0, vmin=-1, vmax=1,
        figsize=(22, 22), linewidths=0.3, annot=False,
        cbar_kws={'shrink': 0.5},
        xticklabels=True, yticklabels=True,
    )
    if fp:
        plt.setp(g.ax_heatmap.get_xticklabels(),
                 fontproperties=fp, fontsize=9, rotation=60, ha='right')
        plt.setp(g.ax_heatmap.get_yticklabels(),
                 fontproperties=fp, fontsize=9, rotation=0)

    g.fig.suptitle(
        '指标相关性热力图（层次聚类排序：高度相关的指标自动聚集在一起）',
        y=1.01, fontsize=14, **_tkw(fp)
    )

    save_path = os.path.join(output_dir, '相关性热力图_层次聚类.png')
    # g.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[IMAGE]相关性热力图_层次聚类.png[/IMAGE]")

    # 打印高相关对
    high_corr = [
        (top_cols[i], top_cols[j], corr.iloc[i, j])
        for i in range(len(top_cols))
        for j in range(i + 1, len(top_cols))
        if abs(corr.iloc[i, j]) > 0.8
    ]
    print(f"高度相关 (|r| > 0.8) 指标对数量: {len(high_corr)}")
    for a, b, v in high_corr[:20]:
        print(f"  {_shorten(a, 22)} — {_shorten(b, 22)}: {v:.3f}")


def _plot_pca_variance(explained_var, cumulative_var, n_components, fp, save_fig):
    """
    PCA 碎石图 + 累积方差曲线双轴图，直观展示选取主成分数量的依据。
    """
    n_show = min(20, len(explained_var))
    fig, ax1 = plt.subplots(figsize=(12, 6))
    x = range(1, n_show + 1)

    ax1.bar(x, explained_var[:n_show] * 100, color='steelblue', alpha=0.7, label='单个主成分')
    ax1.set_xlabel('主成分编号', **_lkw(fp))
    ax1.set_ylabel('解释方差比例 (%)', color='steelblue', **_lkw(fp))
    ax1.tick_params(axis='y', labelcolor='steelblue')

    ax2 = ax1.twinx()
    ax2.plot(x, cumulative_var[:n_show] * 100, 'ro-',
             linewidth=2, markersize=6, label='累积方差')
    ax2.axhline(80, color='gray', linestyle='--', alpha=0.6, linewidth=1)
    ax2.axvline(n_components, color='red', linestyle='--', alpha=0.7,
                linewidth=1.5, label=f'选取 {n_components} 个')
    ax2.set_ylabel('累积解释方差 (%)', color='red', **_lkw(fp))
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 105)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='center right', prop=fp) if fp else \
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    plt.title(
        f'PCA 方差解释图（选取 {n_components} 个主成分，'
        f'累积解释 {cumulative_var[n_components - 1]:.1%}）',
        **_tkw(fp)
    )
    plt.tight_layout()
    save_fig('PCA方差解释图.png')


def _plot_cluster_profile(df, numeric_cols, active_cols, mean_values,
                           optimal_k, fp, save_fig):
    """
    聚类画像双视角：
    图1：归一化均值热图（绿=高 红=低，每个聚类一行）
    图2：聚类规模条形图 + 各聚类与整体均值的 Z-score 偏差热图
    """
    print("\n生成聚类画像图...")

    # 取方差最大的前15列作为画像指标
    profile_cols = (pd.Series({c: df[c].std() for c in active_cols})
                    .nlargest(15).index.tolist())

    scaler_mm = MinMaxScaler()
    df_norm = pd.DataFrame(
        scaler_mm.fit_transform(df[profile_cols]),
        columns=profile_cols
    )
    df_norm['cluster'] = df['cluster'].values
    cluster_profiles = df_norm.groupby('cluster')[profile_cols].mean()

    short_profile = [_shorten(c, 10) for c in profile_cols]

    # ---- 图1：归一化热图 ----
    n_cols_fig = len(profile_cols)
    fig, ax = plt.subplots(figsize=(max(14, n_cols_fig * 0.75 + 4), optimal_k * 0.7 + 3))
    im = ax.imshow(cluster_profiles.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(n_cols_fig))
    ax.set_yticks(range(optimal_k))
    ax.set_xticklabels(short_profile, rotation=45, ha='right', **_lkw(fp))
    ax.set_yticklabels([f'聚类 {i}  ({int((df["cluster"]==i).sum())} 个区域)'
                        for i in range(optimal_k)], **_lkw(fp))
    ax.set_title('各聚类指标画像（Min-Max 归一化均值，绿=高 红=低）', **_tkw(fp))
    plt.colorbar(im, ax=ax, shrink=0.8)
    for i in range(optimal_k):
        for j in range(n_cols_fig):
            val = cluster_profiles.iloc[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8, color='black')
    plt.tight_layout()
    save_fig('聚类画像热图.png')

    # ---- 图2：规模 + Z-score 偏差 ----
    overall_std = df[active_cols].std().replace(0, 1)
    show_active = active_cols[:20]

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    fig.suptitle('聚类概览', fontsize=13, **_tkw(fp))

    # 左：规模条形图
    sizes = [int((df['cluster'] == i).sum()) for i in range(optimal_k)]
    bar_colors = plt.cm.viridis(np.linspace(0.1, 0.9, optimal_k))
    axes[0].bar(range(optimal_k), sizes, color=bar_colors, alpha=0.85, edgecolor='white')
    axes[0].set_xticks(range(optimal_k))
    axes[0].set_xticklabels([f'聚类 {i}\n({s} 个)' for i, s in enumerate(sizes)],
                              **_lkw(fp))
    axes[0].set_ylabel('区域数量', **_lkw(fp))
    axes[0].set_title('各聚类区域数量', **_tkw(fp))
    for i, s in enumerate(sizes):
        axes[0].text(i, s + 0.3, f'{s / len(df):.1%}', ha='center', fontsize=10)

    # 右：Z-score 偏差热图
    cluster_z = pd.DataFrame(index=range(optimal_k),
                              columns=[_shorten(c, 10) for c in show_active],
                              dtype=float)
    for cid in range(optimal_k):
        c_mean = df[df['cluster'] == cid][show_active].mean()
        cluster_z.loc[cid] = ((c_mean - mean_values[show_active]) /
                               overall_std[show_active]).values

    sns.heatmap(cluster_z, cmap='coolwarm', center=0, annot=False,
                linewidths=0.3, ax=axes[1], cbar_kws={'shrink': 0.8})
    axes[1].set_xticklabels(axes[1].get_xticklabels(),
                             rotation=45, ha='right', **_lkw(fp))
    axes[1].set_yticklabels([f'聚类 {i}' for i in range(optimal_k)],
                             rotation=0, **_lkw(fp))
    axes[1].set_title('各聚类与整体均值的 Z-score 偏差\n（红=显著高于均值，蓝=显著低于均值）',
                      **_tkw(fp))

    plt.tight_layout()
    save_fig('聚类概览.png')


# ==================== 地图可视化 ====================

def _plot_region_map(
    region_name: str,
    cluster_id: int,
    df: pd.DataFrame,
    region_col: str,
    geojson_path: str,
    output_dir: str,
    fp,
    highlight_regions: list = None,
):
    """
    绘制香港区议会分区地图：
    - 目标区域：深色高亮（透明度 20%，即 alpha=0.8）
    - 同聚类区域：中等高亮（透明度 50%）
    - 其他区域：浅灰色（透明度 80%，即 alpha=0.2）
    - 底图：CartoDB Positron（简洁浅色，适合叠加数据）
    - highlight_regions: 需要高亮的区域列表（如果提供，则高亮这些区域，而不是单个region_name）

    依赖：geopandas, contextily（首次运行会自动安装）
    """
    # 懒加载，避免没装包时整个模块报错
    try:
        import geopandas as gpd
        import contextily as ctx
        from matplotlib.patches import Patch
    except ImportError:
        print("\n地图依赖包缺失，尝试自动安装 geopandas / contextily ...")
        import subprocess
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "geopandas", "contextily", "-q"
        ])
        import geopandas as gpd
        import contextily as ctx
        from matplotlib.patches import Patch

    # 处理高亮区域列表
    if highlight_regions is None:
        highlight_regions = [region_name]
    print(f"\n生成区域地图：{region_name} ...")
    if len(highlight_regions) > 1:
        print(f"  高亮 {len(highlight_regions)} 个区域: {', '.join(highlight_regions[:5])}{'...' if len(highlight_regions) > 5 else ''}")

    # 读取 GeoJSON / SHP
    try:
        gdf = gpd.read_file(geojson_path)
    except Exception as e:
        print(f"地图文件读取失败，跳过地图生成：{e}")
        return

    # 自动匹配 GeoJSON 中对应区议会分区的字段名
    # 优先找与 region_col 同名的列，否则找包含"區議會"或"district"的列
    geo_col = None
    if region_col in gdf.columns:
        geo_col = region_col
    else:
        for c in gdf.columns:
            if any(kw in c for kw in ['區議會', '分區', 'district', 'District', 'DCCA']):
                geo_col = c
                break
    if geo_col is None:
        # 最后回退：用第一个非 geometry 列
        geo_col = [c for c in gdf.columns if c != 'geometry'][0]
        print(f"  警告：未找到明确的区名列，使用列 '{geo_col}' 作为匹配字段")

    # 将聚类结果合并进 GeoDataFrame
    cluster_map = df.set_index(region_col)['cluster'].to_dict()
    gdf['cluster'] = gdf[geo_col].map(cluster_map)

    # 转换坐标系到 Web Mercator（contextily 底图需要）
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    gdf = gdf.to_crs(epsg=3857)

    # ---- 配色方案：深色底图适配，青绿高亮 ----
    # 使用 matplotlib 的 to_rgba 将颜色+透明度合并成单一 RGBA，
    # 传给 geopandas 的 color 参数，实现向量化绘制（无需逐行循环）
    import matplotlib.colors as mcolors

    COLOR_TARGET  = "#1953ff"   # 亮青绿：目标区域（深色底图上更突出）
    COLOR_CLUSTER = "#89c2ff"   # 中青：同聚类区域
    COLOR_OTHER   = "#C9C9C9"   # 深灰：其他区域（配合黑色底图）
    EDGE_COLOR    = "#FFFFFF"   # 深色边线

    def to_rgba(hex_color, alpha):
        r, g, b, _ = mcolors.to_rgba(hex_color)
        return (r, g, b, alpha)

    # 向量化构建每行的 RGBA 颜色（一次性生成，无循环，速度快）
    def make_facecolor(row):
        name = row[geo_col]
        if name in highlight_regions:
            return to_rgba(COLOR_TARGET, 0.85)
        elif row['cluster'] == cluster_id:
            return to_rgba(COLOR_CLUSTER, 0.35)
        else:
            return to_rgba(COLOR_OTHER, 0.22)

    def make_edgecolor(row):
        if row[geo_col] in highlight_regions:
            return to_rgba("#ffffff", 0.9)
        elif row['cluster'] == cluster_id:
            return to_rgba("#ffffff", 0.6)
        else:
            return to_rgba(EDGE_COLOR, 0.4)

    def make_linewidth(row):
        if row[geo_col] in highlight_regions:
            return 0.8
        elif row['cluster'] == cluster_id:
            return 0.3
        else:
            return 0.3

    facecolors  = gdf.apply(make_facecolor,  axis=1).tolist()
    edgecolors  = gdf.apply(make_edgecolor,  axis=1).tolist()
    linewidths  = gdf.apply(make_linewidth,  axis=1).tolist()

    fig, ax = plt.subplots(figsize=(12, 13))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    # 1. 先画所有色块（让 ax 先确定 extent）
    # 用 positional index 映射颜色，避免 index 不连续导致颜色错位
    pos_map = {gdf_idx: pos for pos, gdf_idx in enumerate(gdf.index)}

    # 用 highlight_regions 列表判断目标区域（region_name 是拼接名，GeoJSON 里没有）
    is_target = gdf[geo_col].isin(highlight_regions)

    for layer_mask, zorder in [
        (~is_target, 2),
        ((gdf['cluster'] == cluster_id) & (~is_target), 3),
        (is_target, 4),
    ]:
        sub = gdf[layer_mask]
        if sub.empty:
            continue
        sub_fc = [facecolors[pos_map[i]] for i in sub.index]
        sub_ec = [edgecolors[pos_map[i]] for i in sub.index]
        sub_lw = [linewidths[pos_map[i]] for i in sub.index]
        sub.plot(
            ax=ax,
            color=sub_fc,
            edgecolor=sub_ec,
            linewidth=sub_lw,
            zorder=zorder,
        )

    # 2. 锁定 extent，防止底图加载后改变视野范围
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # 3. 底图加载已禁用，使用纯色背景

    basemap_loaded = False

    if not basemap_loaded:
        print("  所有底图源均加载失败，使用纯色背景（无底图）")
        # 已经设置了黑色背景，无需额外操作

    # 4. 恢复被底图撑开的 extent，保证色块始终在视野内
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # 5. 最后画文字，永远在最顶层 — 为每个高亮区域标注名称

    # 图例（深色风格）
    if len(highlight_regions) > 1:
        target_label = f'目标区域（{len(highlight_regions)}个区域）'
    else:
        target_label = f'目标区域：{region_name}'
    legend_elements = [
        Patch(facecolor=COLOR_TARGET,  alpha=0.85, edgecolor='white',   label=target_label),
        Patch(facecolor=COLOR_CLUSTER, alpha=0.55, edgecolor='#88dddd', label=f'同聚类区域（聚类 {cluster_id}）'),
        Patch(facecolor=COLOR_OTHER,   alpha=0.40, edgecolor='#555',    label='其他区域'),
    ]
    legend = ax.legend(
        handles=legend_elements, loc='lower left',
        prop=fp if fp else None, fontsize=10,
        framealpha=0.85, facecolor='#1a1a1a',
        edgecolor='#444444', labelcolor='white',
    )

    if len(highlight_regions) > 1:
        target_title = f'目标区域：{len(highlight_regions)}个区域'
    else:
        target_title = f'目标区域：{region_name}'
    ax.set_title(
        f'香港区议会分区地图  |  {target_title}  |  所属聚类：{cluster_id}',
        fontsize=13, pad=12, color='white',
        **({'fontproperties': fp} if fp else {})
    )
    ax.set_axis_off()

    plt.tight_layout()
    safe_name = re.sub(r'[\\/:*?"<>|、]', '_', region_name)
    save_path = os.path.join(output_dir, f'区域地图_{safe_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[IMAGE]区域地图_{safe_name}.png[/IMAGE]")


# ==================== 独立区域分析函数 ====================

def analyze_region(
    region_name: str,
    df: pd.DataFrame,
    region_col: str,
    numeric_cols: pd.Index,
    mean_values: pd.Series,
    desc_stats: pd.DataFrame,
    geojson_path: str = None,
    output_dir: str = '.',
    highlight_regions: list = None,
) -> str:
    """
    分析特定区域在所有数值指标上的表现，并与整体及所在聚类均值对比。

    参数:
        region_name : 要分析的区域名称
        df          : 已含 'cluster' 列的完整 DataFrame
        region_col  : 区域名所在列名
        numeric_cols: 数值列索引
        mean_values : 全体区域各指标均值
        desc_stats  : describe() 的转置，含 'std' 列
        highlight_regions: 需要在地图上高亮的区域列表（可选）

    返回:
        str: 分析报告文本
    """
    fp = _fp()
    out = ["\n" + "=" * 60, f"区域分析: {region_name}", "=" * 60]

    # 检查区域是否在数据表中
    if region_name not in df[region_col].values:
        out.append(f"未找到区域: {region_name}")
        return "\n".join(out)

    # 获取该区域的数据
    region_row = df[df[region_col] == region_name].iloc[0]
    cluster_id = region_row['cluster']
    cluster_size = int((df['cluster'] == cluster_id).sum())

    out.append(f"所属聚类: {cluster_id}")
    out.append(f"该聚类包含区域数: {cluster_size}")

    # 如果是平均值行，说明是基于多个区域的分析
    if highlight_regions and len(highlight_regions) > 0:
        out.append(f"注: 此分析基于 {len(highlight_regions)} 个区域的平均值")
        out.append(f"参与平均的区域: {', '.join(highlight_regions[:5])}{'...' if len(highlight_regions) > 5 else ''}")

    # 提取数值数据
    numeric_vals = {}
    for col in numeric_cols:
        try:
            numeric_vals[col] = float(region_row[col])
        except (ValueError, TypeError):
            continue
    region_values = pd.Series(numeric_vals)

    if region_values.empty:
        out.append("错误: 没有可用的数值数据")
        return "\n".join(out)

    common_cols = region_values.index.intersection(mean_values.index)
    if common_cols.empty:
        out.append("错误: 没有可对齐的数值列")
        return "\n".join(out)

    region_values = region_values[common_cols]
    mean_f = mean_values[common_cols]
    std_f  = desc_stats.loc[common_cols, 'std'].copy()
    std_f[std_f == 0] = 1

    z_scores = (region_values - mean_f) / std_f
    z_scores = z_scores.replace([np.inf, -np.inf], np.nan).dropna()

    if z_scores.empty:
        out.append("错误: 无法计算 z-score")
        return "\n".join(out)

    top5 = z_scores.nlargest(5)
    bot5 = z_scores.nsmallest(5)

    out.append("\n最突出的指标 (z-score 最高，远超平均):")
    for i, (col, z) in enumerate(top5.items(), 1):
        pct = _safe_percentile_high(z)
        out.append(f"  {i}. {col[:40]}: z={z:.2f}  (高于 {pct:.1f}% 的区域)")

    out.append("\n最落后的指标 (z-score 最低，远低于平均):")
    for i, (col, z) in enumerate(bot5.items(), 1):
        pct = _safe_percentile_low(z)
        out.append(f"  {i}. {col[:40]}: z={z:.2f}  (低于 {pct:.1f}% 的区域)")

    try:
        cluster_mean   = df[df['cluster'] == cluster_id][numeric_cols].mean()
        cluster_mean_f = cluster_mean[common_cols]
        safe_denom     = cluster_mean_f.copy()
        safe_denom[safe_denom == 0] = np.nan
        diff_pct = ((region_values - cluster_mean_f) / safe_denom.abs() * 100).dropna()

        top_diff_cols = diff_pct.abs().nlargest(5).index
        out.append(f"\n与所在聚类平均水平的差异 (前5项):")
        for i, col in enumerate(top_diff_cols, 1):
            d = diff_pct[col]
            direction = "高于" if d > 0 else "低于"
            out.append(f"  {i}. {col[:40]}: {direction} {abs(d):.1f}%")
    except Exception as e:
        out.append(f"\n无法计算与聚类均值的差异: {e}")


    # 地图可视化
    if geojson_path:
        if os.path.exists(geojson_path):
            _plot_region_map(
                region_name=region_name,
                cluster_id=int(cluster_id),
                df=df,
                region_col=region_col,
                geojson_path=geojson_path,
                output_dir=output_dir,
                fp=fp,
                highlight_regions=highlight_regions,
            )
        else:
            out.append(f"\n警告：地图文件未找到：{geojson_path}，跳过地图生成")

    return "\n".join(out)


# ==================== 主分析函数 ====================

def analyze_regional_data(
    df: pd.DataFrame,
    matched_area: str,
    output_dir: str = '.',
    geojson_path: str = None,
    matched_areas: list = None,
) -> str:
    """
    区域数据聚类分析函数。

    参数:
        df          : 包含区域数据的 DataFrame，第一列为区域名
        matched_area: 要重点分析的地区名称（或平均区域名称）
        output_dir  : 图表输出目录（默认当前目录）
        geojson_path: GeoJSON 文件路径（可选）
        matched_areas: 实际匹配的区域列表（可选，用于高亮多个区域）

    返回:
        str: 聚类特征分析 + 区域分析的完整报告文本
    """
    os.makedirs(output_dir, exist_ok=True)
    fp = _fp()

    def save_fig(filename: str):
        # 清理文件名，移除Windows非法字符
        safe_filename = re.sub(r'[\\/:*?"<>|]', '_', filename)
        path = os.path.join(output_dir, safe_filename)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[IMAGE]{safe_filename}[/IMAGE]")

    output_text = []

    # ==================== 1. 数据预处理 ====================
    print("=" * 60)
    print("数据基本信息")
    print("=" * 60)

    df = df.copy()
    region_col = df.columns[0]
    print(f"数据形状: {df.shape}  |  区域列: '{region_col}'")

    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"数值列数量: {len(numeric_cols)}")

    missing_count = df[numeric_cols].isna().sum().sum()
    if missing_count > 0:
        print(f"检测到缺失值: {missing_count} 个，使用列均值填充")
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # 如果提供了matched_areas，创建平均值行并添加到df中
    if matched_areas and len(matched_areas) > 1:
        print(f"\n处理 {len(matched_areas)} 个匹配区域，创建平均值行...")

        # 提取所有匹配区域的数据行
        matched_rows = df[df[region_col].astype(str).isin(matched_areas)]
        if not matched_rows.empty:
            print(f"  ✅ 找到 {len(matched_rows)} 个匹配区域的数据行")

            # 创建平均值行
            avg_data = {}
            # 区域列使用matched_area（多个区域的字符串表示）
            avg_data[region_col] = matched_area

            # 计算数值列的平均值
            for col in numeric_cols:
                avg_data[col] = float(matched_rows[col].mean())

            # 对于非数值列，使用第一个区域的值或留空
            non_numeric_cols = matched_rows.select_dtypes(exclude=[np.number]).columns
            for col in non_numeric_cols:
                if col != region_col:
                    avg_data[col] = matched_rows[col].iloc[0] if not matched_rows[col].empty else ""

            # 创建新的DataFrame行并添加到df中
            avg_df = pd.DataFrame([avg_data])
            df = pd.concat([df, avg_df], ignore_index=True)
            print(f"  ✅ 平均值行 '{matched_area}' 已添加到数据表中")
            print(f"  ✅ 新数据形状: {df.shape}")
        else:
            print(f"  ⚠️ 未找到匹配区域的数据行，跳过平均值行创建")

    desc_stats = df[numeric_cols].describe().T
    desc_stats['cv']    = desc_stats['std'] / desc_stats['mean'].replace(0, np.nan)
    desc_stats['range'] = desc_stats['max'] - desc_stats['min']
    mean_values = df[numeric_cols].mean()

    # ==================== 2. 数据质量分析（识别稀疏列、量纲分组）====================
    print("\n" + "=" * 60)
    print("数据质量分析")
    print("=" * 60)

    quality     = _analyze_data_quality(df, numeric_cols)
    sparse_cols = quality['sparse_cols']
    active_cols = quality['active_cols']   # 不含稀疏列，用于聚类
    col_groups  = quality['col_groups']    # 按量纲分组，用于可视化
    zero_ratio  = quality['zero_ratio']

    # ==================== 3. 分组标准化分布可视化（解决量纲混用问题）====================
    _plot_grouped_distribution(df, col_groups, fp, save_fig)

    # ==================== 4. 稀疏列专项分析 ====================
    _plot_sparse_cols(df, sparse_cols, zero_ratio, fp, save_fig)

    # ==================== 5. 相关性分析（层次聚类热图）====================
    print("\n" + "=" * 60)
    print("相关性分析")
    print("=" * 60)
    _plot_correlation_clustermap(df, active_cols, desc_stats, output_dir, fp)

    # ==================== 6. PCA + 聚类分析（仅使用非稀疏列）====================
    print("\n" + "=" * 60)
    print("PCA 降维 + 聚类分析")
    print("=" * 60)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[active_cols])

    max_components = min(20, len(active_cols), len(df))
    pca = PCA(n_components=max_components)
    X_pca_full = pca.fit_transform(X_scaled)

    explained_var  = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    print("前10个主成分解释的方差比例:")
    for i in range(min(10, len(explained_var))):
        print(f"  PC{i+1}: {explained_var[i]:.4f}  (累积: {cumulative_var[i]:.4f})")

    # 至少保留8个主成分（前2个仅解释约40%，8个可解释约65%）
    n_components_80 = int(np.argmax(cumulative_var >= 0.80)) + 1
    n_components    = max(n_components_80, 8)
    print(f"\n选择保留 {n_components} 个主成分 "
          f"(解释 {cumulative_var[n_components - 1]:.2%} 的方差)")

    _plot_pca_variance(explained_var, cumulative_var, n_components, fp, save_fig)

    X_pca = X_pca_full[:, :n_components]

    # 肘部法则
    k_range = range(2, 11)
    inertia = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_pca)
        inertia.append(km.inertia_)

    optimal_k = _auto_optimal_k(inertia, k_range)
    print(f"肘部法则自动检测最优聚类数: {optimal_k}")

    plt.figure(figsize=(10, 6))
    plt.plot(list(k_range), inertia, 'bo-', linewidth=2, markersize=8)
    plt.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7,
                label=f'最优 k={optimal_k}')
    plt.xlabel('聚类数量 (k)', **_lkw(fp))
    plt.ylabel('惯性 (Inertia)', **_lkw(fp))
    plt.title(f'肘部法则 — 最优聚类数（基于 {n_components} 个主成分）', **_tkw(fp))
    plt.legend(prop=fp) if fp else plt.legend()
    plt.grid(True, alpha=0.3)
    save_fig('肘部法则图.png')

    # K-Means 聚类
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
    df['cluster'] = kmeans.fit_predict(X_pca)

    # 散点图（前2个主成分，仅用于展示，聚类基于更多主成分）
    X_viz = X_pca_full[:, :2]
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(X_viz[:, 0], X_viz[:, 1],
                          c=df['cluster'], cmap='viridis', alpha=0.7, s=50)

    if len(df) <= 100:
        for i, region in enumerate(df[region_col]):
            plt.annotate(region, (X_viz[i, 0], X_viz[i, 1]),
                         fontsize=8, alpha=0.7, **_lkw(fp))
    else:
        for cid in range(optimal_k):
            mask = df['cluster'] == cid
            pts  = X_viz[mask.values]
            if len(pts) == 0:
                continue
            center  = pts.mean(axis=0)
            closest = np.argsort(np.linalg.norm(pts - center, axis=1))[:3]
            sub_df  = df[mask].reset_index(drop=True)
            for idx in closest:
                plt.annotate(
                    sub_df.loc[idx, region_col], pts[idx],
                    fontsize=10, alpha=0.8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
                    **_lkw(fp)
                )

    plt.xlabel(f'主成分1 (解释方差: {explained_var[0]:.2%})', **_lkw(fp))
    plt.ylabel(f'主成分2 (解释方差: {explained_var[1]:.2%})', **_lkw(fp))
    plt.title(
        f'区域聚类结果 (k={optimal_k}，基于 {n_components} 个主成分)\n'
        f'注：散点图仅用前2个主成分展示，聚类实际基于 {n_components} 个主成分',
        **_tkw(fp), fontsize=12
    )
    plt.colorbar(scatter, label='聚类标签')
    plt.grid(True, alpha=0.3)
    save_fig('聚类散点图.png')

    # ==================== 7. 聚类画像 ====================
    _plot_cluster_profile(df, numeric_cols, active_cols, mean_values,
                          optimal_k, fp, save_fig)

    # ==================== 8. 聚类特征文字报告 ====================
    print("\n" + "=" * 60)
    print("各聚类特征分析")
    print("=" * 60)

    output_text.append("\n" + "=" * 60)
    output_text.append("各聚类特征分析")
    output_text.append("=" * 60)

    for cid in range(optimal_k):
        cluster_data = df[df['cluster'] == cid]
        cluster_mean = cluster_data[numeric_cols].mean()
        size = len(cluster_data)

        diff_abs  = (cluster_mean - mean_values).abs()
        top_feats = diff_abs.nlargest(5).index.tolist()

        output_text.append(f"\n聚类 {cid}:")
        output_text.append(f"  包含区域数: {size} ({size / len(df) * 100:.1f}%)")
        output_text.append("  最显著特征（与整体均值偏差最大）：")
        for feat in top_feats:
            ov       = mean_values[feat]
            cv       = cluster_mean[feat]
            diff_pct = (cv - ov) / abs(ov) * 100 if ov != 0 else float('nan')
            output_text.append(f"    - {feat[:40]}: 比整体平均 {diff_pct:+.1f}%")

    # ==================== 9. 目标区域分析 ====================
    from config import GEOJSON_PATH
    
    region_text = analyze_region(
        region_name=matched_area,
        df=df,
        region_col=region_col,
        numeric_cols=numeric_cols,
        mean_values=mean_values,
        desc_stats=desc_stats,
        geojson_path=GEOJSON_PATH,
        output_dir=output_dir,
        highlight_regions=matched_areas,
    )
    output_text.append(region_text)

    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)

    return "\n".join(output_text)