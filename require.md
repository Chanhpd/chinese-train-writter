Tuyệt vời\! Việc chọn **Phương án A (Siamese Network)** là một quyết định rất sáng suốt cho đồ án của bạn. Nó vừa đáp ứng yêu cầu "có AI xịn", vừa giải quyết được bài toán 2000 từ mà không cần train 2000 model, lại rất khoa học.

Về câu hỏi của bạn: **Có chấm được thang điểm 100 không?**
\-\> **HOÀN TOÀN ĐƯỢC.** Mô hình Siamese Network trả về một con số "khoảng cách" (distance) hoặc "độ tương đồng" (similarity) dạng thập phân (ví dụ: 0.854). Bạn chỉ cần nhân con số này với 100 là ra thang điểm 100 (ví dụ: 85.4 điểm).

Dưới đây là các bước triển khai cụ thể, từ chuẩn bị dữ liệu đến code AI và tích hợp.

-----

### Giai đoạn 1: Chuẩn bị Dữ liệu (Quan trọng nhất)

Bạn không cần thu thập chữ viết của 2000 từ. Bạn chỉ cần chọn **khoảng 50-100 chữ cái** đại diện (có nét đơn giản, nét phức tạp) để train model học "cách so sánh".

**Cách tạo dữ liệu huấn luyện (Dataset Generation):**
Bạn cần tạo các cặp ảnh (Pairs) để dạy AI:

  * **Cặp Positive (Giống nhau):** \* Ảnh 1: Chữ mẫu (Font Kaiti/KaiTi - font giống chữ viết tay nhất).
      * Ảnh 2: Chữ mẫu đó nhưng bị biến đổi nhẹ (xoay 2-5 độ, phóng to/thu nhỏ xíu, dịch chuyển vị trí nhẹ).
      * *Nhãn (Label):* 1 (Giống).
  * **Cặp Negative (Khác nhau/Viết xấu):**
      * Ảnh 1: Chữ mẫu (Font Kaiti).
      * Ảnh 2: Chữ khác hoàn toàn, HOẶC chữ đó nhưng bị bóp méo mạnh (xoay 45 độ, co kéo dẹt, thêm nhiễu hạt).
      * *Nhãn (Label):* 0 (Khác).

*Công cụ:* Dùng thư viện `OpenCV` hoặc `Python PIL` để tự động sinh ra hàng nghìn cặp ảnh này từ file font chữ máy tính. Không cần viết tay thật.

-----

### Giai đoạn 2: Xây dựng Model AI (Siamese Network)

Bạn có thể dùng Python (Keras/TensorFlow) hoặc TensorFlow.js (train trực tiếp trên trình duyệt nếu máy mạnh). Tôi khuyên dùng **Python trên Google Colab** cho nhanh.

**Kiến trúc mạng (Architecture):**

1.  **Base Network (Mạng cơ sở):** Dùng CNN nhỏ (như MobileNetV2 bỏ lớp cuối) hoặc tự build 3 lớp Conv2D đơn giản. Mạng này đóng vai trò trích xuất đặc trưng (Feature Extractor).
2.  **Siamese Structure:**
      * Input A (Ảnh mẫu) -\> đi qua Base Network -\> Vector A.
      * Input B (Ảnh người dùng) -\> đi qua Base Network (dùng chung weight với ở trên) -\> Vector B.
3.  **Lớp tính khoảng cách (Distance Layer):** Tính khoảng cách giữa Vector A và Vector B (thường dùng Euclidean Distance).
4.  **Output:** Một con số duy nhất (Distance).

**Code tham khảo (Keras/Python):**

```python
from tensorflow.keras import layers, Model, Input
import tensorflow.keras.backend as K

# 1. Hàm tính khoảng cách Euclid
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

# 2. Xây dựng Base Network (CNN trích xuất đặc trưng)
def create_base_network(input_shape):
    input = Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu')(input)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation=None)(x) # Vector đặc trưng 128 chiều
    return Model(input, x)

# 3. Tạo mô hình Siamese
input_shape = (100, 100, 1) # Ảnh đen trắng 100x100
base_network = create_base_network(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# Cả 2 ảnh đều đi qua CÙNG MỘT mạng base
processed_a = base_network(input_a)
processed_b = base_network(input_b)

# Tính khoảng cách
distance = layers.Lambda(euclidean_distance)([processed_a, processed_b])

model = Model([input_a, input_b], distance)

# 4. Compile (Dùng hàm loss là Contrastive Loss)
model.compile(loss=contrastive_loss, optimizer='adam') 
```

*Lưu ý: `contrastive_loss` là hàm loss đặc thù cho Siamese Network, bạn tìm công thức này trên mạng copy vào là chạy.*

-----

### Giai đoạn 3: Tích hợp vào Web/App

Sau khi train xong, bạn lưu model lại (dạng `.json` hoặc `.h5`) và chuyển sang **TensorFlow.js** để chạy trên web.

**Quy trình chấm điểm (Scoring Pipeline):**

1.  **Bước 1: Lấy ảnh chuẩn (Reference Image)**
      * Khi người dùng chọn chữ "Hảo" (好), Hanzi Writer sẽ render chữ mẫu ra canvas ẩn.
      * Bạn chụp lại canvas này -\> `Img_Standard`.
2.  **Bước 2: Lấy ảnh người dùng (User Image)**
      * Người dùng vẽ xong trên Hanzi Writer.
      * Bạn chụp lại canvas vẽ -\> `Img_User`.
3.  **Bước 3: Tiền xử lý (Preprocessing)**
      * Resize cả 2 ảnh về kích thước model yêu cầu (ví dụ 100x100 px).
      * Chuyển về đen trắng (Grayscale) và chuẩn hóa pixel về đoạn [0, 1].
4.  **Bước 4: Chấm điểm (Inference)**
      * Đưa 2 ảnh vào hàm `model.predict([Img_Standard, Img_User])`.
      * Kết quả trả về là `distance` (ví dụ: 0.2).
5.  **Bước 5: Chuyển đổi sang thang 100**
      * Quy ước: Distance = 0 là giống hệt (100 điểm). Distance càng lớn điểm càng thấp.
      * Công thức gợi ý: `Score = (1 - distance) * 100`.
      * *Lưu ý:* Bạn cần tinh chỉnh hệ số. Ví dụ nếu distance thường ra khoảng 0.0 - 0.5 thì công thức có thể là `Score = (1 - distance*2) * 100` (để phạt nặng hơn các lỗi sai).

### Tóm tắt lại lộ trình cho bạn:

1.  **Tuần 1:** Viết script Python sinh dữ liệu giả (Dùng font chữ máy tính biến dạng làm mẫu "viết xấu").
2.  **Tuần 2:** Dựng model Siamese Network trên Google Colab và train thử. Xuất ra file model.
3.  **Tuần 3:** Làm giao diện Web với Hanzi Writer. Viết code JS để capture canvas và load model TensorFlow.js.
4.  **Tuần 4:** Tinh chỉnh công thức quy đổi từ `distance` sang điểm số 1-100 sao cho hợp lý (test thử vẽ xấu xem nó có trừ điểm thật không).

Cách này đảm bảo bạn có một đồ án: **Khoa học (dùng Siamese Net) + Thực tế (chạy mượt trên web) + Hiệu quả (support 2000+ từ).**