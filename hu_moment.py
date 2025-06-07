import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def create_test_images():
    """テスト用のバイナリ画像を生成（歪な物体＋微細変化図形を含む）"""
    images = {}

    # 64x64の白画像を基本とする
    base = np.zeros((64, 64), dtype=np.uint8)

    # === 基本図形 ===

    # 1. 正方形（中央）
    square = base.copy()
    square[20:44, 20:44] = 255
    images['square'] = square

    # 2. 横長長方形
    h_rect = base.copy()
    h_rect[26:38, 16:48] = 255
    images['horizontal_rect'] = h_rect

    # 3. 縦長長方形
    v_rect = base.copy()
    v_rect[16:48, 26:38] = 255
    images['vertical_rect'] = v_rect

    # 3.5. 45度回転長方形
    rect_45 = base.copy()
    rect_45[26:38, 16:48] = 255  # まず横長長方形を作成
    # 45度回転変換
    center = (32, 32)
    rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
    rect_45 = cv2.warpAffine(rect_45, rotation_matrix, (64, 64))
    images['diagonal_rect'] = rect_45

    # 4. 円
    circle = base.copy()
    cv2.circle(circle, (32, 32), 12, 255, -1)
    images['circle'] = circle

    # 5. 横長楕円
    h_ellipse = base.copy()
    cv2.ellipse(h_ellipse, (32, 32), (18, 8), 0, 0, 360, 255, -1)
    images['horizontal_ellipse'] = h_ellipse

    # 6. 縦長楕円
    v_ellipse = base.copy()
    cv2.ellipse(v_ellipse, (32, 32), (8, 18), 0, 0, 360, 255, -1)
    images['vertical_ellipse'] = v_ellipse

    # 7. 斜め楕円（45度）
    d_ellipse = base.copy()
    cv2.ellipse(d_ellipse, (32, 32), (16, 8), 45, 0, 360, 255, -1)
    images['diagonal_ellipse'] = d_ellipse

    # === 複雑図形 ===

    # 8. L字型
    l_shape = base.copy()
    l_shape[15:44, 20:32] = 255  # 縦棒
    l_shape[32:44, 20:44] = 255  # 横棒
    images['l_shape'] = l_shape

    # Lを90度回転
    l_shape_90 = base.copy()
    l_shape_90[15:44, 20:32] = 255  # 縦棒
    l_shape_90[32:44, 20:44] = 255  # 横棒
    # 90度回転変換
    center = (32, 32)
    rotation_matrix = cv2.getRotationMatrix2D(center, 90, 1.0)
    l_shape_90 = cv2.warpAffine(l_shape_90, rotation_matrix, (64, 64))
    images['l_shape_90deg'] = l_shape_90

    # Lの鏡
    l_shape_mirror = base.copy()
    # l_shapeのx軸反転
    l_shape_mirror[15:44, 32:44] = 255  # 縦棒
    l_shape_mirror[32:44, 20:32] = 255  # 横棒
    images['l_shape_mirror'] = l_shape_mirror

    # 9. 三角形
    triangle = base.copy()
    pts = np.array([[32, 16], [20, 48], [44, 48]], np.int32)
    cv2.fillPoly(triangle, [pts], 255)
    images['triangle'] = triangle

    # 9.5. 90度回転三角形
    triangle_90 = base.copy()
    pts_90 = np.array([[32, 16], [20, 48], [44, 48]], np.int32)
    cv2.fillPoly(triangle_90, [pts_90], 255)
    # 90度回転変換
    center = (32, 32)
    rotation_matrix = cv2.getRotationMatrix2D(center, 90, 1.0)
    triangle_90 = cv2.warpAffine(triangle_90, rotation_matrix, (64, 64))
    images['triangle_90deg'] = triangle_90

    # 10. 非対称形状（右偏り）
    asymmetric = base.copy()
    asymmetric[24:40, 20:28] = 255  # 左部分（小）
    asymmetric[20:44, 28:48] = 255  # 右部分（大）
    images['asymmetric'] = asymmetric

    # === 歪な物体 ===

    # 11. 星型（5角星）- 完全対称版
    star = base.copy()
    outer_points = []
    inner_points = []
    for i in range(5):
        # 外側の点（半径16）
        angle_outer = i * 2 * np.pi / 5 - np.pi / 2
        x_outer = int(32 + 16 * np.cos(angle_outer))
        y_outer = int(32 + 16 * np.sin(angle_outer))
        outer_points.append([x_outer, y_outer])

        # 内側の点（半径7）
        angle_inner = (i + 0.5) * 2 * np.pi / 5 - np.pi / 2
        x_inner = int(32 + 7 * np.cos(angle_inner))
        y_inner = int(32 + 7 * np.sin(angle_inner))
        inner_points.append([x_inner, y_inner])

    # 星型の頂点を交互に配置
    star_points = []
    for i in range(5):
        star_points.append(outer_points[i])
        star_points.append(inner_points[i])

    star_pts = np.array(star_points, np.int32)
    cv2.fillPoly(star, [star_pts], 255)
    images['star_perfect'] = star

    # 12. ハート型
    heart = base.copy()
    heart_points = []
    for t in np.linspace(0, 2 * np.pi, 100):
        x = 16 * (np.cos(t) - np.cos(2 * t) / 2)
        y = 16 * (np.sin(t) - np.sin(2 * t) / 2)
        heart_points.append([int(32 + x / 2), int(32 - y / 2 + 5)])

    heart_pts = np.array(heart_points, np.int32)
    cv2.fillPoly(heart, [heart_pts], 255)
    images['heart'] = heart

    # 13. 不規則な多角形
    irregular = base.copy()
    irreg_points = np.array([
        [32, 10], [45, 20], [50, 35], [40, 50], [25, 48],
        [15, 35], [20, 25], [28, 15]
    ], np.int32)
    cv2.fillPoly(irregular, [irreg_points], 255)
    images['irregular_polygon'] = irregular

    # 14. 三日月型
    crescent = base.copy()
    cv2.circle(crescent, (32, 32), 16, 255, -1)  # 大きい円
    cv2.circle(crescent, (38, 28), 12, 0, -1)  # 小さい円（引く）
    images['crescent'] = crescent

    # 15. 歪んだ楕円（不規則形状）
    distorted = base.copy()
    cv2.ellipse(distorted, (32, 32), (15, 10), 0, 0, 360, 255, -1)
    distorted[25:35, 40:50] = 0  # 一部を削る
    distorted[15:25, 28:38] = 255  # 一部を追加
    images['distorted_ellipse'] = distorted

    # === Flusser効果検証用の微細変化図形 ===

    # 17. 基準楕円（Flusser比較用）
    ref_ellipse = base.copy()
    cv2.ellipse(ref_ellipse, (32, 32), (15, 10), 0, 0, 360, 255, -1)
    images['flusser_ellipse_ref'] = ref_ellipse

    # 18. 上部微細圧縮楕円
    compressed = base.copy()
    cv2.ellipse(compressed, (32, 32), (15, 10), 0, 0, 360, 255, -1)
    # 上部を微細に圧縮
    compressed[20:26, 22:42] = 0
    cv2.ellipse(compressed, (32, 23), (13, 8), 0, 0, 360, 255, -1)
    # 境界を滑らかに
    compressed = cv2.GaussianBlur(compressed, (3, 3), 0)
    compressed[compressed > 127] = 255
    compressed[compressed <= 127] = 0
    images['flusser_ellipse_compressed'] = compressed

    # 19. 右側微細膨張楕円
    expanded = base.copy()
    cv2.ellipse(expanded, (32, 32), (15, 10), 0, 0, 360, 255, -1)
    # 右側を微細に膨張
    cv2.ellipse(expanded, (35, 32), (18, 10), 0, 0, 360, 255, -1)
    images['flusser_ellipse_expanded'] = expanded

    # 20. 微細非対称星型（1つの角だけ長い）
    star_asym = base.copy()
    star_points_asym = star_points.copy()
    # 最初の外側の点だけ少し外に
    star_points_asym[0][1] -= 3  # 上の角を少し長く
    star_pts_asym = np.array(star_points_asym, np.int32)
    cv2.fillPoly(star_asym, [star_pts_asym], 255)
    images['flusser_star_asym'] = star_asym

    # 22. 微細変形円（わずかな楕円化）
    deformed_circle = base.copy()
    cv2.ellipse(deformed_circle, (32, 32), (15, 11), 0, 0, 360, 255, -1)
    # deformed_circleの左半分を黒に
    deformed_circle[16:48, 16:32] = 0  # 左半分を削除
    # 上半分を黒に
    deformed_circle[16:32, 16:48] = 0  # 上半分を削除
    cv2.ellipse(deformed_circle, (32, 32), (12, 11), 0, 0, 360, 255, -1)
    # deformed_circleの左半分を黒に
    deformed_circle[16:48, 16:32] = 0  # 左半分を削除
    # 半径11の円を追加
    cv2.circle(deformed_circle, (32, 32), 11, 255, -1)  # 中心に円を追加
    images['flusser_circle_deformed'] = deformed_circle

    # 23. 微細欠け円（小さな欠けあり）
    chipped_circle = base.copy()
    cv2.circle(chipped_circle, (32, 32), 12, 255, -1)
    # 小さな欠けを作る
    chipped_circle[28:32, 42:46] = 0
    images['flusser_circle_chipped'] = chipped_circle

    # 27. 医用画像風：異常細胞模擬（微細な変形）
    abnormal_cell = base.copy()
    cv2.ellipse(abnormal_cell, (32, 32), (10, 9), 15, 0, 360, 255, -1)
    # 微細な突起を追加
    cv2.circle(abnormal_cell, (38, 28), 2, 255, -1)
    # 軽いノイズ
    noise = np.random.randint(-10, 10, abnormal_cell.shape)
    abnormal_cell = np.clip(abnormal_cell.astype(int) + noise, 0, 255).astype(np.uint8)
    abnormal_cell[abnormal_cell > 127] = 255
    abnormal_cell[abnormal_cell <= 127] = 0
    images['flusser_circle_abnormal'] = abnormal_cell

    # 24. 基準三角形
    ref_triangle = base.copy()
    pts_ref = np.array([[32, 18], [22, 46], [42, 46]], np.int32)
    cv2.fillPoly(ref_triangle, [pts_ref], 255)
    images['flusser_triangle_ref'] = ref_triangle

    # 25. 微細非対称三角形（1つの角が微妙にずれ）
    asym_triangle = base.copy()
    pts_asym = np.array([[32, 18], [22, 46], [44, 46]], np.int32)  # 右下が微妙に右に
    cv2.fillPoly(asym_triangle, [pts_asym], 255)
    images['flusser_triangle_asym'] = asym_triangle

    # === h5 vs h6 違い検証用図形 ===

    # 28. 楕円+上突起（3次モーメントη₀₃変化）
    ellipse_top = base.copy()
    ellipse_top[26:38, 16:48] = 255
    cv2.circle(ellipse_top, (20, 27), 4, 255, -1)  # 上突起
    images['h5h6_rect_top1_spike'] = ellipse_top

    # 29. 楕円+下突起（3次モーメントη₀₃逆変化）
    ellipse_bottom = base.copy()
    ellipse_bottom[26:38, 16:48] = 255
    cv2.circle(ellipse_bottom, (22, 26), 3, 255, -1)  # 下突起
    images['h5h6_rect_top2_spike'] = ellipse_bottom

    return images


def calculate_all_moments(image):
    """Hu Moments、Flusser Moment、関連統計量を計算"""
    # OpenCVでHu Momentsを計算
    moments = cv2.moments(image)
    hu_moments = cv2.HuMoments(moments).flatten()

    # 正規化中心モーメントを手動計算
    m00 = moments['m00']
    if m00 == 0:
        return None

    # 重心
    cx = moments['m10'] / m00
    cy = moments['m01'] / m00

    # 正規化中心モーメント
    eta20 = moments['mu20'] / (m00 ** 2)
    eta02 = moments['mu02'] / (m00 ** 2)
    eta11 = moments['mu11'] / (m00 ** 2)
    eta30 = moments['mu30'] / (m00 ** 2.5)
    eta03 = moments['mu03'] / (m00 ** 2.5)
    eta21 = moments['mu21'] / (m00 ** 2.5)
    eta12 = moments['mu12'] / (m00 ** 2.5)

    eta_dict = {
        'eta20': eta20, 'eta02': eta02, 'eta11': eta11,
        'eta30': eta30, 'eta03': eta03, 'eta21': eta21, 'eta12': eta12
    }

    # 手動でHu Momentsを計算（全7つ）
    h1 = eta20 + eta02
    h2 = (eta20 - eta02) ** 2 + 4 * (eta11 ** 2)
    h3 = (eta30 - 3 * eta12) ** 2 + (3 * eta21 - eta03) ** 2
    h4 = (eta30 + eta12) ** 2 + (eta21 + eta03) ** 2

    # h5の計算（複雑な式）
    h5 = ((eta30 - 3 * eta12) * (eta30 + eta12) *
          ((eta30 + eta12) ** 2 - 3 * (eta21 + eta03) ** 2) +
          (3 * eta21 - eta03) * (eta21 + eta03) *
          (3 * (eta30 + eta12) ** 2 - (eta21 + eta03) ** 2))

    # h6の計算
    h6 = ((eta20 - eta02) * ((eta30 + eta12) ** 2 - (eta21 + eta03) ** 2) +
          4 * eta11 * (eta30 + eta12) * (eta21 + eta03))

    # h7の計算
    h7 = ((3 * eta21 - eta03) * (eta30 + eta12) *
          ((eta30 + eta12) ** 2 - 3 * (eta21 + eta03) ** 2) -
          (eta30 - 3 * eta12) * (eta21 + eta03) *
          (3 * (eta30 + eta12) ** 2 - (eta21 + eta03) ** 2))

    # Flusserの8番目の不変量 I8
    i8 = (eta11 * ((eta30 + eta12) ** 2 - (eta03 + eta21) ** 2) -
          (eta20 - eta02) * (eta30 + eta12) * (eta03 + eta21))

    manual_hu = [h1, h2, h3, h4, h5, h6, h7]

    result = {
        'name': '',
        'moments': moments,
        'centroid': (cx, cy),
        'eta': eta_dict,
        'hu_opencv': hu_moments,
        'hu_manual': manual_hu,
        'flusser_i8': i8,
        'area': m00
    }

    return result


def save_images_separately(images, output_dir='output_images'):
    """画像を個別にサブディレクトリに保存"""
    Path(output_dir).mkdir(exist_ok=True)

    for name, image in images.items():
        img_dir = Path(output_dir) / name
        img_dir.mkdir(exist_ok=True)

        # 画像を保存
        cv2.imwrite(str(img_dir / f'{name}.png'), image)

        # matplotlibでも保存（見やすい形式）
        plt.figure(figsize=(6, 6))
        plt.imshow(image, cmap='gray')
        plt.title(name.replace('_', ' ').title())
        plt.axis('off')
        plt.savefig(img_dir / f'{name}_display.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved: {name} -> {img_dir}")


def create_comparison_table(results):
    """比較表を作成してCSV出力"""
    data = []

    for name, result in results.items():
        row = {
            'image_name': name,
            'area': result['area'],
            'centroid_x': result['centroid'][0],
            'centroid_y': result['centroid'][1],
            'eta20': result['eta']['eta20'],
            'eta02': result['eta']['eta02'],
            'eta11': result['eta']['eta11'],
            'eta30': result['eta']['eta30'],
            'eta03': result['eta']['eta03'],
            'eta21': result['eta']['eta21'],
            'eta12': result['eta']['eta12'],
            'h1': result['hu_manual'][0],
            'h2': result['hu_manual'][1],
            'h3': result['hu_manual'][2],
            'h4': result['hu_manual'][3],
            'h5': result['hu_manual'][4],
            'h6': result['hu_manual'][5],
            'h7': result['hu_manual'][6],
            'flusser_i8': result['flusser_i8'],
            'h1_opencv': result['hu_opencv'][0],
            'h2_opencv': result['hu_opencv'][1],
            'h3_opencv': result['hu_opencv'][2],
            'h4_opencv': result['hu_opencv'][3],
            'h5_opencv': result['hu_opencv'][4],
            'h6_opencv': result['hu_opencv'][5],
            'h7_opencv': result['hu_opencv'][6]
        }
        data.append(row)

    df = pd.DataFrame(data)
    return df


def display_detailed_table(df):
    """詳細な表を標準出力に表示"""
    pd.options.display.float_format = '{:.6e}'.format

    print("\n" + "=" * 100)
    print("Hu Moments & Flusser Moment Analysis Results")
    print("=" * 100)

    # 基本情報
    print("\n--- 基本情報 ---")
    basic_cols = ['image_name', 'area', 'centroid_x', 'centroid_y']
    print(df[basic_cols].to_string(index=False, float_format='%.2f'))

    # 正規化中心モーメント
    print("\n--- 正規化中心モーメント (η) ---")
    eta_cols = ['image_name', 'eta20', 'eta02', 'eta11', 'eta30', 'eta03', 'eta21', 'eta12']
    print(df[eta_cols].to_string(index=False, float_format='%.6e'))

    # Hu Moments (手動計算)
    print("\n--- Hu Moments (手動計算) ---")
    hu_cols = ['image_name', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7']
    print(df[hu_cols].to_string(index=False, float_format='%.6e'))

    # Flusser Moment
    print("\n--- Flusser I8 ---")
    flusser_cols = ['image_name', 'flusser_i8']
    print(df[flusser_cols].to_string(index=False, float_format='%.6e'))


def analyze_shape_characteristics(df):
    """形状特徴の分析"""
    print("\n" + "=" * 80)
    print("形状特徴分析")
    print("=" * 80)

    # h2による異方性分析
    print("\n--- 異方性分析 (h2) ---")
    df_sorted = df.sort_values('h2')
    for _, row in df_sorted.iterrows():
        if row['h2'] < 1e-6:
            anisotropy = "等方的"
        elif row['h2'] < 1e-3:
            anisotropy = "やや異方的"
        else:
            anisotropy = "強い異方性"
        print(f"{row['image_name']:30s}: h2 = {row['h2']:.6e} ({anisotropy})")

    # h3による対称性分析
    print("\n--- 対称性分析 (h3) ---")
    df_sorted = df.sort_values(by='h3', key=lambda x: x.abs())
    for _, row in df_sorted.iterrows():
        if abs(row['h3']) < 1e-10:
            symmetry = "完全対称"
        elif abs(row['h3']) < 1e-7:
            symmetry = "ほぼ対称"
        else:
            symmetry = "非対称"
        print(f"{row['image_name']:30s}: h3 = {row['h3']:.6e} ({symmetry})")


def create_flusser_comparison_plots(df, output_dir='output_images'):
    """Flusserモーメントの比較プロットを作成"""
    # 微細変化ペアの可視化
    flusser_pairs = [
        ('flusser_ellipse_ref', 'flusser_ellipse_compressed'),
        ('flusser_ellipse_ref', 'flusser_ellipse_expanded'),
        ('star_perfect', 'flusser_star_asym'),
        ('flusser_circle_normal', 'flusser_circle_deformed')
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, (ref_name, mod_name) in enumerate(flusser_pairs[:4]):
        if ref_name in df['image_name'].values and mod_name in df['image_name'].values:
            ref_row = df[df['image_name'] == ref_name].iloc[0]
            mod_row = df[df['image_name'] == mod_name].iloc[0]

            i8_diff = abs(ref_row['flusser_i8'] - mod_row['flusser_i8'])
            h2_diff = abs(ref_row['h2'] - mod_row['h2'])

            axes[i * 2].bar(['h2_diff', 'i8_diff'], [h2_diff, i8_diff])
            axes[i * 2].set_title(f'{ref_name[:10]} vs {mod_name[:10]}')
            axes[i * 2].set_ylabel('Difference')
            axes[i * 2].set_yscale('log')

            # 感度比を表示
            sensitivity = i8_diff / h2_diff if h2_diff > 0 else float('inf')
            axes[i * 2 + 1].bar(['I8 Sensitivity'], [sensitivity])
            axes[i * 2 + 1].set_title(f'I8 Sensitivity: {sensitivity:.2f}x')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/flusser_effectiveness_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """メイン処理"""
    print("Enhanced Hu Moments & Flusser Moment Analysis")
    print("==============================================")

    # 1. テスト画像生成
    print("\n1. テスト画像を生成中...")
    images = create_test_images()
    print(f"生成された画像数: {len(images)}")
    print(f"- 基本図形: 7個")
    print(f"- 複雑図形: 10個")
    print(f"- Flusser検証用微細変化図形: {len(images) - 17}個")

    # 2. 画像を個別保存
    print("\n2. 画像を個別保存中...")
    save_images_separately(images)

    # 3. モーメント計算
    print("\n3. Hu Moments & Flusser Moment計算中...")
    results = {}
    for name, image in images.items():
        result = calculate_all_moments(image)
        if result:
            result['name'] = name
            results[name] = result

    # 4. データフレーム作成
    df = create_comparison_table(results)

    # 5. CSV出力
    output_file = 'hu_moments_analysis.csv'
    df.to_csv(output_file, index=False)
    print(f"\n4. CSV出力完了: {output_file}")

    # 5.5 指数表記バージョン
    output_file_formatted = 'hu_moments_analysis_formatted.csv'
    df.to_csv(output_file_formatted, index=False, float_format='%.6e')
    print(f"CSV出力完了: {output_file_formatted}")

    # 6. 詳細表示
    display_detailed_table(df)

    # 7. 形状特徴分析
    analyze_shape_characteristics(df)

    # 9. Flusser比較プロット作成（新機能）
    print("\n8. Flusser効果分析プロット作成中...")
    create_flusser_comparison_plots(df)

    # 10. 統合画像表示の作成
    print("\n9. 統合画像表示を作成中...")
    n_images = len(images)
    cols = 6
    rows = (n_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, 3 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for i, (name, image) in enumerate(images.items()):
        if i < len(axes):
            axes[i].imshow(image, cmap='gray')
            axes[i].set_title(name.replace('_', ' ').title(), fontsize=8)
            axes[i].axis('off')

    # 余った軸を非表示
    for i in range(len(images), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('all_test_images.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n分析完了！")
    print(f"- 個別画像: output_images/ ディレクトリ")
    print(f"- 統合画像: all_test_images.png")
    print(f"- 分析結果: {output_file}")


if __name__ == "__main__":
    main()
