# Giải thích ý nghĩa các đặc trưng Radiomics trong Y tế (Dữ liệu KiTS23)

**Radiomics** là lĩnh vực chuyển đổi hình ảnh y tế (như ảnh CT, MRI) thành dữ liệu định lượng có thể khai thác được. Thay vì bác sĩ chỉ nhìn bằng mắt và mô tả "khối u này có vẻ tròn, hơi sần sùi", thuật toán Radiomics sẽ tính toán ra chính xác độ tròn là bao nhiêu %, độ sần sùi (kết cấu) đạt bao nhiêu điểm. 

Gần 50 đặc trưng trong bộ dữ liệu của bạn được chia thành **3 nhóm chính**, mỗi nhóm giải quyết một khía cạnh sinh học khác nhau của khối u/nang.

---

## 1. Nhóm đặc trưng hình dạng (Shape Features)
Nhóm này đánh giá hình thái vật lý không gian (3D) của tổn thương, hoàn toàn độc lập với cường độ sáng.

*   `original_shape_Sphericity` (Độ cầu): Đo lường khối tổn thương giống hình cầu hoàn hảo tới mức nào. **Ý nghĩa lâm sàng:** U nang (Cyst) lành tính thường căng tròn như bong bóng nước (Sphericity gần bằng 1). Ngược lại, khối u ác tính (Tumor) thường phát triển xâm lấn, hình thù gồ ghề, kỳ dị (Sphericity thấp).
*   `original_shape_Elongation` (Độ thuôn dài): Đo mức độ kéo dài của tổn thương.
*   `original_shape_Flatness` (Độ bẹt): Đo mức độ dẹt của khối u. Khối u càng bẹt thì giá trị càng thấp.
*   `original_shape_LeastAxisLength`: Độ dài của trục ngắn nhất xuyên qua khối u. Giúp đánh giá độ dày của vùng tổn thương.

> [!TIP]
> Nhóm hình dạng là công cụ đắc lực nhất để phân biệt ban đầu giữa khối u phát triển ác tính (gai góc) và nang nước phát triển tự nhiên (tròn, nhẵn).

---

## 2. Nhóm đặc trưng thống kê bậc 1 (First-order / Intensity Features)
Nhóm này mô tả sự phân bố của cường độ pixel (Hounsfield Unit - HU trong ảnh CT) bên trong tổn thương, bằng cách đếm xem có bao nhiêu pixel màu xám, bao nhiêu màu trắng... mà **không quan tâm đến vị trí** của chúng.

*   `original_firstorder_Mean` / `Median`: Mức sáng trung bình/trung vị. Nang nước thường tối (0-20 HU), mô u đặc thường sáng hơn (60-120 HU).
*   `original_firstorder_Maximum` / `Minimum` / `Range`: Điểm sáng nhất, tối nhất và biên độ (Max - Min). **Ý nghĩa cực quan trọng:** Biên độ (Range) và Max rất cao chứng tỏ khối u đó có vôi hóa (rất sáng) hoặc xuất huyết/hoại tử (tối đen lẫn sáng trắng). Đây chính là dấu hiệu đã giúp thuật toán HDBSCAN tìm ra Cụm số 2!
*   `original_firstorder_Entropy`: Đo độ "hỗn loạn" của thông tin cường độ. U nang chứa nước đồng nhất sẽ có Entropy thấp. Khối u chứa máu, mô hoại tử, mạch máu sẽ có Entropy rất cao.
*   `original_firstorder_Skewness` (Độ lệch) & `Kurtosis` (Độ nhọn): Đánh giá hình dáng của biểu đồ phân bố độ sáng. Khối u không đồng nhất thường có biểu đồ lệch hẳn sang một bên.

---

## 3. Nhóm đặc trưng kết cấu bề mặt (Texture Features - Bậc 2 và Bậc cao)
Đây là nhóm phức tạp và mạnh mẽ nhất của Radiomics. Nó không chỉ quan tâm đến độ sáng, mà còn quan tâm đến **sự sắp xếp vị trí** của các pixel kề nhau, từ đó mô tả được tổn thương đó "nhẵn mịn" hay "thô ráp, sần sùi".

### A. Ma trận đồng xuất hiện mức xám (GLCM - Gray Level Co-occurrence Matrix)
Đánh giá mức độ lặp lại của các **cặp pixel kề nhau**.
*   `original_glcm_Contrast` (Độ tương phản không gian): Sự chênh lệch độ sáng giữa các điểm ảnh cạnh nhau. Độ tương phản cao nghĩa là mô rất thô ráp, sần sùi.
*   `original_glcm_ClusterProminence` & `ClusterShade`: Đánh giá mức độ lệch của bề mặt. Giá trị lớn (như trong Cụm số 2) cho thấy có các cụm "hạt" sáng rực lên nằm xen kẽ với các vùng tối, dấu hiệu của vôi hóa cục bộ.
*   `original_glcm_Correlation` (Độ tương quan): Đo mức độ lặp lại của một mẫu kết cấu. Mô lành tính thường có cấu trúc tuyến tính đều đặn (Correlation cao).

### B. Ma trận độ dài dải mức xám (GLRLM - Gray Level Run Length Matrix)
Đánh giá chiều dài của **các dải pixel liền kề** có cùng mức sáng.
*   `original_glrlm_LongRunLowGrayLevelEmphasis`: Nhấn mạnh các dải tối và dài. Nang nước đồng nhất sẽ có dải dài. Khối u gồ ghề sẽ bị đứt gãy thành các dải rất ngắn.

### C. Ma trận vùng kích thước mức xám (GLSZM - Gray Level Size Zone Matrix)
Phân vùng tổn thương thành **các cụm diện tích** có cùng độ sáng.
*   `original_glszm_GrayLevelVariance`: Mức độ biến động của các cụm xám.
*   `original_glszm_SmallAreaHighGrayLevelEmphasis`: Đánh giá xem có nhiều cụm diện tích NHỎ nhưng lại RẤT SÁNG hay không. Bất kỳ khối u nào có chỉ số này cao đều bị nghi ngờ là có các vi vôi hóa nhỏ li ti bên trong.

### D. Ma trận khác biệt sắc thái xám lân cận (NGTDM - Neighbourhood Gray Tone Difference Matrix)
Tính toán sự khác biệt giữa một pixel và tất cả các pixel lân cận nó.
*   `original_ngtdm_Complexity` (Độ phức tạp bề mặt): Tính toán sự thay đổi nhanh chóng và liên tục của ánh sáng. Đây là đặc trưng then chốt để phân biệt mô mềm sinh lý (đơn giản) với mô ung thư di căn/hoại tử (cực kỳ phức tạp).
*   `original_ngtdm_Contrast`: Thể hiện sự thay đổi không gian và dải động (dynamic range) của hình ảnh.

---

## TỔNG KẾT MỐI LIÊN HỆ VỚI THUẬT TOÁN HDBSCAN CỦA BẠN:
Thay vì phải dùng mắt người để phán đoán *"hình như khối u này hơi gồ ghề và có đốm trắng"*, bộ 50 đặc trưng Radiomics này cung cấp cho thuật toán HDBSCAN một bản đồ toán học tuyệt đối chính xác:
1.  HDBSCAN dựa vào **Shape** để chia đôi nhóm hình cầu (Nang) và nhóm gai góc (U ác).
2.  HDBSCAN dựa vào **First-order** để tách nhóm sáng mờ và nhóm sáng rực.
3.  HDBSCAN dựa vào **Texture (GLCM, NGTDM)** để tìm ra các nhóm mô mềm thô ráp, nham nhở. 

Sự kết hợp của cả 3 yếu tố này đã tạo nên ma trận dữ liệu khổng lồ giúp thuật toán gom cụm cực kỳ chính xác và phát hiện ra được cả những phân nhóm bệnh lý ẩn!
