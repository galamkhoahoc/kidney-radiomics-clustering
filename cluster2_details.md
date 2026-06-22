# Chi tiết các mẫu thuộc Cụm 2 (Phân nhóm lâm sàng ẩn)

Dưới đây là danh sách chi tiết 107 mẫu bệnh (68 Tumors, 39 Cysts) được HDBSCAN gộp vào Cụm 2. Các mẫu được xếp hạng theo **độ phức tạp kết cấu bề mặt (Complexity)** giảm dần. 

> [!TIP]
> Các giá trị **Max Intensity** (cường độ sáng tối đa) ở đây lên tới vài trăm, thậm chí vài nghìn HU, là minh chứng rất mạnh mẽ cho sự xuất hiện của **Vôi hóa (Calcification)** hoặc các cấu trúc xương/vật thể cứng nằm trộn lẫn trong u/nang. Bình thường nang nước chỉ quanh mức 0-20 HU, còn u đặc quanh mức 60-120 HU.

## 1. Nhóm Khối u (Tumor - 68 mẫu)

| PatientID | Instance | Ground Truth | Max Intensity (HU) | Intensity Range | Complexity |
| --- | --- | --- | --- | --- | --- |
| case_00288 | tumor_instance-1 | Tumor | 714.95 | 1037.73 | 845.27 |
| case_00025 | tumor_instance-1 | Tumor | 556.29 | 772.84 | 565.44 |
| case_00102 | tumor_instance-1 | Tumor | 547.35 | 768.52 | 532.10 |
| case_00473 | tumor_instance-1 | Tumor | 674.96 | 933.39 | 509.25 |
| case_00517 | tumor_instance-1 | Tumor | 538.63 | 788.55 | 504.13 |
| case_00577 | tumor_instance-1 | Tumor | 567.53 | 799.01 | 495.03 |
| case_00492 | tumor_instance-1 | Tumor | 639.86 | 860.54 | 467.98 |
| case_00135 | tumor_instance-1 | Tumor | 541.87 | 746.32 | 456.51 |
| case_00053 | tumor_instance-1 | Tumor | 530.20 | 728.40 | 441.17 |
| case_00420 | tumor_instance-1 | Tumor | 621.59 | 827.82 | 432.21 |
| case_00426 | tumor_instance-1 | Tumor | 600.60 | 824.90 | 428.73 |
| case_00073 | tumor_instance-1 | Tumor | 605.18 | 803.18 | 422.75 |
| case_00477 | tumor_instance-1 | Tumor | 506.65 | 730.72 | 394.92 |
| case_00523 | tumor_instance-1 | Tumor | 610.32 | 842.90 | 376.48 |
| case_00224 | tumor_instance-1 | Tumor | 582.59 | 807.94 | 344.76 |
| case_00092 | tumor_instance-1 | Tumor | 524.33 | 710.63 | 307.22 |
| case_00237 | tumor_instance-1 | Tumor | 427.81 | 648.21 | 293.82 |
| case_00089 | tumor_instance-1 | Tumor | 396.34 | 626.37 | 255.14 |
| case_00474 | tumor_instance-1 | Tumor | 521.35 | 750.93 | 249.31 |
| case_00114 | tumor_instance-1 | Tumor | 328.47 | 538.96 | 247.15 |
| case_00538 | tumor_instance-1 | Tumor | 513.27 | 719.59 | 243.11 |
| case_00262 | tumor_instance-2 | Tumor | 384.18 | 579.34 | 235.67 |
| case_00535 | tumor_instance-1 | Tumor | 385.48 | 585.42 | 235.03 |
| case_00556 | tumor_instance-1 | Tumor | 514.45 | 695.29 | 234.84 |
| case_00196 | tumor_instance-1 | Tumor | 499.53 | 704.14 | 229.12 |
| case_00282 | tumor_instance-1 | Tumor | 524.51 | 688.02 | 222.93 |
| case_00545 | tumor_instance-1 | Tumor | 282.83 | 519.95 | 203.58 |
| case_00249 | tumor_instance-1 | Tumor | 289.33 | 512.11 | 201.15 |
| case_00293 | tumor_instance-1 | Tumor | 377.21 | 566.04 | 196.40 |
| case_00279 | tumor_instance-1 | Tumor | 496.15 | 714.46 | 191.00 |
| case_00431 | tumor_instance-1 | Tumor | 423.63 | 628.98 | 182.42 |
| case_00139 | tumor_instance-1 | Tumor | 441.95 | 606.68 | 181.21 |
| case_00030 | tumor_instance-1 | Tumor | 390.36 | 623.25 | 180.57 |
| case_00278 | tumor_instance-1 | Tumor | 391.40 | 616.59 | 179.81 |
| case_00067 | tumor_instance-1 | Tumor | 343.50 | 561.86 | 176.92 |
| case_00029 | tumor_instance-1 | Tumor | 398.10 | 605.15 | 162.73 |
| case_00571 | tumor_instance-1 | Tumor | 394.21 | 601.05 | 157.89 |
| case_00543 | tumor_instance-1 | Tumor | 306.21 | 528.98 | 152.03 |
| case_00116 | tumor_instance-1 | Tumor | 217.29 | 433.41 | 150.75 |
| case_00522 | tumor_instance-1 | Tumor | 305.00 | 480.05 | 133.23 |
| case_00586 | tumor_instance-2 | Tumor | 308.81 | 505.16 | 114.85 |
| case_00027 | tumor_instance-1 | Tumor | 139.26 | 352.94 | 98.75 |
| case_00021 | tumor_instance-1 | Tumor | 237.40 | 435.52 | 91.41 |
| case_00518 | tumor_instance-1 | Tumor | 298.66 | 516.37 | 90.30 |
| case_00443 | tumor_instance-1 | Tumor | 126.00 | 366.00 | 87.11 |
| case_00445 | tumor_instance-1 | Tumor | 172.09 | 400.09 | 85.29 |
| case_00514 | tumor_instance-1 | Tumor | 130.34 | 351.43 | 84.87 |
| case_00013 | tumor_instance-1 | Tumor | 185.01 | 402.24 | 80.89 |
| case_00423 | tumor_instance-1 | Tumor | 139.55 | 370.05 | 74.05 |
| case_00402 | tumor_instance-1 | Tumor | 151.07 | 365.31 | 73.71 |
| case_00425 | tumor_instance-1 | Tumor | 147.82 | 376.01 | 65.78 |
| case_00255 | tumor_instance-1 | Tumor | 113.02 | 334.21 | 59.19 |
| case_00048 | tumor_instance-1 | Tumor | 118.27 | 346.03 | 55.58 |
| case_00413 | tumor_instance-1 | Tumor | 114.54 | 338.14 | 53.79 |
| case_00453 | tumor_instance-1 | Tumor | 128.54 | 352.68 | 52.24 |
| case_00072 | tumor_instance-1 | Tumor | 117.69 | 338.00 | 51.36 |
| case_00190 | tumor_instance-1 | Tumor | 104.51 | 318.59 | 48.18 |
| case_00432 | tumor_instance-1 | Tumor | 92.46 | 294.98 | 45.31 |
| case_00242 | tumor_instance-2 | Tumor | 116.70 | 321.31 | 43.77 |
| case_00438 | tumor_instance-2 | Tumor | 86.42 | 293.79 | 40.01 |
| case_00478 | tumor_instance-1 | Tumor | 89.07 | 303.56 | 37.60 |
| case_00499 | tumor_instance-2 | Tumor | 97.59 | 307.09 | 35.47 |
| case_00064 | tumor_instance-1 | Tumor | 87.40 | 280.71 | 33.21 |
| case_00585 | tumor_instance-1 | Tumor | 63.93 | 268.85 | 31.96 |
| case_00171 | tumor_instance-1 | Tumor | 96.12 | 293.54 | 31.36 |
| case_00205 | tumor_instance-1 | Tumor | 55.96 | 245.96 | 27.94 |
| case_00410 | tumor_instance-1 | Tumor | 91.13 | 289.99 | 27.41 |
| case_00129 | tumor_instance-1 | Tumor | 52.18 | 249.09 | 26.24 |


## 2. Nhóm U nang (Cyst - 39 mẫu)

| PatientID | Instance | Ground Truth | Max Intensity (HU) | Intensity Range | Complexity |
| --- | --- | --- | --- | --- | --- |
| case_00183 | cyst_instance-1 | Cyst | 443.44 | 629.82 | 362.40 |
| case_00523 | cyst_instance-2 | Cyst | 393.73 | 595.90 | 205.15 |
| case_00468 | cyst_instance-14 | Cyst | 287.74 | 518.88 | 163.99 |
| case_00549 | cyst_instance-1 | Cyst | 273.94 | 466.88 | 137.16 |
| case_00551 | cyst_instance-1 | Cyst | 343.81 | 499.81 | 132.03 |
| case_00551 | cyst_instance-2 | Cyst | 378.88 | 539.83 | 127.80 |
| case_00468 | cyst_instance-9 | Cyst | 246.23 | 444.98 | 118.19 |
| case_00468 | cyst_instance-1 | Cyst | 216.14 | 428.75 | 107.41 |
| case_00468 | cyst_instance-7 | Cyst | 198.04 | 405.90 | 94.03 |
| case_00468 | cyst_instance-11 | Cyst | 220.69 | 418.50 | 93.30 |
| case_00508 | cyst_instance-1 | Cyst | 236.44 | 453.17 | 91.92 |
| case_00468 | cyst_instance-12 | Cyst | 185.81 | 392.38 | 90.32 |
| case_00468 | cyst_instance-2 | Cyst | 193.46 | 405.95 | 88.01 |
| case_00468 | cyst_instance-8 | Cyst | 185.42 | 395.29 | 87.15 |
| case_00468 | cyst_instance-13 | Cyst | 169.70 | 386.11 | 76.88 |
| case_00184 | cyst_instance-16 | Cyst | 111.49 | 316.52 | 74.32 |
| case_00184 | cyst_instance-11 | Cyst | 117.37 | 318.60 | 73.90 |
| case_00184 | cyst_instance-17 | Cyst | 113.75 | 314.52 | 72.86 |
| case_00238 | cyst_instance-2 | Cyst | 160.32 | 369.98 | 72.15 |
| case_00468 | cyst_instance-10 | Cyst | 171.29 | 383.27 | 71.66 |
| case_00184 | cyst_instance-13 | Cyst | 120.63 | 323.59 | 71.08 |
| case_00184 | cyst_instance-5 | Cyst | 114.40 | 328.78 | 66.35 |
| case_00426 | cyst_instance-1 | Cyst | 137.93 | 366.21 | 65.62 |
| case_00184 | cyst_instance-3 | Cyst | 114.35 | 315.33 | 65.45 |
| case_00529 | cyst_instance-2 | Cyst | 207.15 | 408.00 | 64.89 |
| case_00424 | cyst_instance-1 | Cyst | 107.07 | 320.38 | 56.32 |
| case_00198 | cyst_instance-1 | Cyst | 102.73 | 307.01 | 55.97 |
| case_00069 | cyst_instance-7 | Cyst | 123.60 | 333.41 | 53.91 |
| case_00184 | cyst_instance-4 | Cyst | 112.61 | 289.37 | 52.16 |
| case_00238 | cyst_instance-3 | Cyst | 122.25 | 333.39 | 51.35 |
| case_00504 | cyst_instance-2 | Cyst | 71.30 | 274.90 | 47.96 |
| case_00475 | cyst_instance-1 | Cyst | 96.07 | 308.23 | 47.47 |
| case_00238 | cyst_instance-6 | Cyst | 107.46 | 310.33 | 47.27 |
| case_00069 | cyst_instance-3 | Cyst | 113.56 | 303.43 | 46.87 |
| case_00224 | cyst_instance-3 | Cyst | 85.77 | 290.02 | 46.58 |
| case_00242 | cyst_instance-1 | Cyst | 124.47 | 312.47 | 44.81 |
| case_00508 | cyst_instance-2 | Cyst | 91.20 | 302.77 | 43.85 |
| case_00492 | cyst_instance-1 | Cyst | 123.32 | 314.25 | 42.28 |
| case_00097 | cyst_instance-2 | Cyst | 118.39 | 278.49 | 38.57 |

