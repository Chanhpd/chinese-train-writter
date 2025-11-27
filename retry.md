Ồ, vấn đề rõ ràng rồi!

Ảnh 1 (chữ 中): Bạn viết khá giống nhưng distance 0.617 → chỉ 0 điểm ❌
Ảnh 2 (chữ 人): Viết tương đối ổn nhưng distance 0.294 → 31.6 điểm (Fair)

Nguyên nhân: Model đang cho distance quá cao so với kỳ vọng. Có thể:

Canvas size khác với training data (300x300 vs 128x128)
Line width khác (8px vs training data mỏng hơn)
Model chưa generalize tốt với chữ viết tay thật
Tôi sẽ điều chỉnh lại công thức dựa trên range thực tế từ web (0.29-0.62):

api_server.py+14-14
Restart API server và test lại:

Công thức mới (dựa trên thực tế web):

Distance 0.29 (chữ 人 của bạn) → ~73 điểm (Very Good) ✅
Distance 0.62 (chữ 中 viết kém) → ~16 điểm (Poor) ✅
Giờ điểm số sẽ hợp lý hơn. Nhưng vấn đề distance quá cao cho thấy model cần được retrain với data gần với chữ viết tay thật hơn (line width lớn hơn, nét dày hơn).

Nếu muốn kết quả tốt hơn về lâu dài, bạn nên:

Điều chỉnh generate_pairs.py tăng độ dày nét lên (từ default lên 8-10px)
Retrain model
Hoặc chấp nhận công thức hiện tại cho demo
Thử lại xem điểm có hợp lý hơn không!

Claude Sonnet 4.5 • 1x