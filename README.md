Gà làm khoa học

# KiTS23 Radiomics Clustering Project

Khám phá đặc trưng và phân cụm tự động khối u và u nang thận (KiTS23) sử dụng **Radiomics** (Pyradiomics) và các thuật toán **HDBSCAN**, **OPTICS**, **GMM**.

## Cấu trúc thư mục

```
kits23_clustering_project/
├── data/                     # Dữ liệu (ignored by git)
│   ├── raw/                  # Repo KiTS23 clone
│   └── processed/            # Features đã trích xuất
├── src/                      # Mã nguồn chính
│   ├── config.py             # Đường dẫn, hằng số, hyperparameters
│   ├── data_ingestion.py     # Tải dataset & quét lesions
│   ├── feature_extraction.py # Radiomics + PCA
│   ├── clustering.py         # HDBSCAN, OPTICS, GMM, K-Means
│   └── evaluation.py         # Metrics, t-SNE/UMAP, bootstrap
├── outputs/                  # Biểu đồ và báo cáo
├── main.py                   # Entry point
├── requirements.txt
└── README.md
```


## TMUX GUIDE
1. Tạo session
```
tmux new -s <ten_session>
```

2. Treo session
Ctrl + B, sau đó thả ra và bấm phím D (D là Detach)

3. Quay lại session
```
tmux attach -t <ten_session>
```

4. Xem số phòng đang chạy ngầm
```
tmux ls
```

5. Hủy session
```
tmux kill-session -t <ten_session>
```



## Yêu cầu hệ thống

- Python 3.10+
- Git
- ~50 GB dung lượng ổ cứng (dataset KiTS23)
- Khuyến nghị: RAM >= 16 GB

## Cài đặt

```bash
cd kits23_clustering_project

# Tạo và kích hoạt môi trường conda
conda create -n kits23 python=3.10 -y
conda activate kits23

# Cài đặt thư viện
pip install -r requirements.txt
```

## Chạy pipeline

### Chạy toàn bộ (từ tải dữ liệu đến đánh giá)

```bash
python main.py
```

### Chạy từng giai đoạn

```bash
# 1. Tải dataset KiTS23 (bỏ qua nếu đã có)
python main.py --stage ingest

# 2. Trích xuất Radiomics + PCA
python main.py --stage features

# 3. Phân cụm (HDBSCAN, OPTICS, GMM, K-Means)
python main.py --stage cluster

# 4. Đánh giá & trực quan hóa
python main.py --stage evaluate

# 5. Kiểm tra Supervised (LDA + Classifier)
python main.py --stage supervised-check

# 6. Quét ngưỡng kích thước khối u (Voxel Sweep)
python main.py --stage voxel-sweep --consistent-pca
```

### Tùy chọn CLI

| Flag | Mô tả |
|------|-------|
| `--stage <tên>` | `all`, `ingest`, `features`, `cluster`, `evaluate`, `supervised-check`, `voxel-sweep` |
| `--feature-selection` | Bật lựa chọn đặc trưng bán giám sát (ANOVA F-test) trước khi PCA để giữ lại các đặc trưng phân biệt tốt (Vd: HU/firstorder) |
| `--umap-cluster` | Dùng UMAP để phân cụm mật độ thay vì chỉ dùng PCA (giúp khuếch đại khoảng trống mật độ cục bộ) |
| `--min-voxels N` | Lọc bỏ các masks nhỏ hơn N voxels (mặc định: 250). Giúp loại bỏ nhiễu do masks quá nhỏ |
| `--consistent-pca` | Dùng cho voxel-sweep: áp dụng cùng một PCA space cho mọi ngưỡng cắt để so sánh công bằng |
| `--seed N` | Thiết lập seed toàn cục (mặc định 42) chi phối Numpy, stdlib, PCA, t-SNE, K-Means, GMM, RandomForest, đảm bảo tính tái lập |
| `--skip-download` | Bỏ qua bước tải dataset nếu đã có case folders |
| `--force-clone` | Clone lại repo KiTS23 |
| `--no-resume-extraction` | Chạy lại trích xuất Radiomics |
| `--use-tsne` | Dùng t-SNE thay UMAP cho visualization |
| `--no-bootstrap` | Bỏ qua bootstrap stability analysis |
| `--log-level DEBUG` | Log chi tiết hơn |

## Pipeline tóm tắt

1. **Data Ingestion** — Clone repo [KiTS23](https://github.com/neheller/kits23), cài package, tải dữ liệu, quét tumor/cyst masks, chuẩn hóa contrast phase (HU baseline).
2. **Feature Extraction** — Lọc masks nhỏ (mặc định `>= 250 voxels`), chạy Pyradiomics (firstorder, shape, texture).
3. **Feature Clean & Select** — Khử outliers, (Tùy chọn) Chọn đặc trưng bán giám sát bằng ANOVA F-test (`--feature-selection`), StandardScaler.
4. **Dimension Reduction** — PCA (giữ 95% variance) hoặc UMAP (`--umap-cluster`).
5. **Clustering** — Grid search HDBSCAN (thêm `leaf` method), OPTICS (thêm `dbscan` extraction), K-Means, GMM.
6. **Evaluation** — Đánh giá chéo bằng `Coverage × ARI` (metric chính). DBCV nội tại, ARI/NMI/Purity bên ngoài; UMAP plots; ANOVA khám phá subtype; bootstrap.
7. **Supervised Check** — Khẳng định tín hiệu tồn tại (LDA, ROC-AUC) nhưng bị các phương pháp không giám sát bỏ lỡ.

## Đầu ra

| File | Mô tả |
|------|-------|
| `data/processed/lesions.csv` | Metadata các tổn thương hợp lệ |
| `data/processed/features.csv` | Radiomics features thô |
| `data/processed/features_clean.csv` | Features sau làm sạch |
| `outputs/clustering_metrics.csv` | Bảng so sánh các chỉ số phân cụm (Coverage x ARI, DBCV, ARI, Purity,...) |
| `outputs/feature_selection_ranking.csv` | Bảng xếp hạng đặc trưng (nếu dùng `--feature-selection`) |
| `outputs/voxel_sweep_summary.csv` | Kết quả từ lệnh quét ngưỡng `--stage voxel-sweep` |
| `outputs/pca_loadings.csv` | Trọng số PCA (PCA loadings diagnostic) |
| `outputs/supervised_sanity_check.csv` | Kết quả phân loại Supervised Sanity Check |
| `outputs/*.png` | Biểu đồ trực quan hóa (UMAP, Heatmap, Reachability, LDA,...) |

## Cấu hình

Tất cả hyperparameters nằm trong `src/config.py`:

- **Tính tái lập:** `RANDOM_STATE = 42` (bị override bởi `--seed`)
- **Lọc nhiễu:** `LESION_MIN_VOXELS_FILTER = 250`
- **Lựa chọn đặc trưng:** `FEATURE_SELECTION_TOP_K = 25`
- Radiomics: `binWidth`, `resampledPixelSpacing`, CT window
- HDBSCAN grid: `min_cluster_size` (10-80), `min_samples`, `cluster_selection_method` (`eom`, `leaf`)
- OPTICS grid: `xi`, `eps`, `cluster_method` (`xi`, `dbscan`), min coverage = 0.5
- Bootstrap: 100 iterations x 80% subsample

## Ghi chú

- Bước ingestion **tự động bỏ qua** git clone / download nếu dữ liệu đã tồn tại.
- Bước extraction **tự động bỏ qua** nếu `features.csv` đã có (dùng `--no-resume-extraction` để chạy lại).
- Clustering được thực hiện trên **PCA space** (trừ phi dùng flag `--umap-cluster`) để tránh biến dạng khoảng cách.

## Liên hệ

Nhóm Gà làm khoa học — Khoa Toán - Tin học, Trường ĐH KHTN, ĐHQG-HCM.
