# Hiểu video dài trong bối cảnh triệu Token

> Một video 4K dài 1 giờ ở tốc độ 24 FPS, được vá và nhúng, sản xuất theo thứ tự 60 triệu tokens. Một podcast dài 2 giờ episode phiên âm là 30,000 tokens. Một bộ phim Blu-ray feature đầy đủ, thậm chí được nén bằng gộp tích cực, là hàng trăm nghìn tokens. Gemini 1.5 của Google (tháng 3 năm 2024) đã mở ra kỷ nguyên này với bối cảnh 10 triệu token, thực hiện các recall đáng tin cậy trong các video dài hàng giờ. LWM (Liu và cộng sự, tháng 2 năm 2024) cho thấy đường mở rộng quy mô của ring attention. LongVILA và Video-XL đã mở rộng khả năng nhập hơn nữa. VideoAgent đã hoán đổi ngữ cảnh thô để truy xuất agentic. Mỗi cách tiếp cận là một sự đánh đổi khác nhau về độ phức tạp của điện toán, recall và kỹ thuật. Bài học này đọc chúng cạnh nhau.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib, needle-in-haystack simulator + agentic-retrieval router)
**Kiến thức tiên quyết:** Giai đoạn 12 · 17 (video thời gian tokens)
**Thời lượng:** ~180 phút

## Mục tiêu học tập

- Tính toán tổng số token hình ảnh cho video dài ở các FPS và tổng hợp khác nhau.
- Giải thích ba đường dẫn chia tỷ lệ: ngữ cảnh vũ phu (Gemini 1.5), attention vòng (LWM), token nén (LongVILA / Video-XL).
- So sánh VLMs video ngữ cảnh thô so với VLMs video truy xuất agentic (VideoAgent) về accuracy và độ trễ.
- Thiết kế thử nghiệm kim trong đống cỏ khô cho video dài 30 phút và đo recall vào một phút cụ thể.

## Vấn đề

Một khung duy nhất của các bản vá có kích thước Qwen2.5-VL ở độ phân giải gốc 384 là ~729 tokens. Ở 3x3 pooling, đó là 81 tokens mỗi khung hình. Clip dài 30 phút ở tốc độ 1 FPS = 1800 khung hình = 145.800 tokens. Có thể thực hiện được vào năm 2025 mở VLMs, chặt chẽ. Ở 2 FPS, 291.600 tokens - chỉ những bối cảnh lớn nhất mới phù hợp.

Một bộ phim dài 2 giờ ở tốc độ 1 FPS là 583k tokens. Ngoài hầu hết các models mở năm 2026; yêu cầu Gemini 2.5 Pro hoặc gộp tích cực hơn.

Ba con đường mở rộng quy mô đã xuất hiện.

## Khái niệm

### Con đường 1: Bối cảnh vũ phu (Gemini 1.5, Claude Opus)

Ném phần cứng vào vấn đề. Mở rộng ngữ cảnh lên hàng triệu tokens, process mọi thứ trong một forward pass.

Gemini 1.5 Pro ra mắt với 1 triệu tokens; Gemini 1,5 Ultra đến 10M; Gemini 2.5 Pro vào năm 2026 thực hiện hàng giờ video một cách đáng tin cậy. Bài báo (arXiv: 2403.05530) tài liệu recall kim trong đống cỏ khô ở mức 99,7% lên đến ~9,5 triệu tokens.

Kỹ thuật: triển khai attention tùy chỉnh với hệ thống phân cấp bộ nhớ (cục bộ + toàn cầu + thưa thớt ) cộng với định tuyến chuyên gia MoE để đạt hiệu quả ngữ cảnh dài. Không được công bố đầy đủ chi tiết. Không phải mã nguồn mở.

### Đường dẫn 2: Ring attention (LWM, LongVILA)

Ring attention phân phối các chuỗi dài trên các thiết bị trong một "vòng" trong đó mỗi thiết bị chứa một khối. Attention trên chuỗi đầy đủ xảy ra bằng cách mỗi thiết bị gửi đoạn của nó sang phần tiếp theo theo một mẫu vòng, tính toán attention một phần và tổng hợp.

LWM (Liu và cộng sự, 2024) đã huấn luyện bối cảnh 1M-token model cách này. Training tính toán mở rộng tuyến tính theo ngữ cảnh, không phải bậc hai - lần truy cập bậc hai trên attention được khấu hao trên các thiết bị của vòng.

LongVILA (arXiv: 2408.10188) đã điều chỉnh mô hình cho phù hợp với VLMs. Video 1400 khung hình ở tốc độ 192 tokens mỗi khung hình = ngữ cảnh 268k, được huấn luyện với attention vòng song song 8 chiều.

### Đường dẫn 3: Nén Token (Video-XL, LongVA)

Rẻ hơn bối cảnh vũ phu: nén mạnh trước khi LLM nhìn thấy trình tự.

Video-XL (arXiv: 2409.14485) sử dụng một token tóm tắt trực quan: mỗi clip của N khung hình tạo ra một token "tóm tắt" duy nhất tham dự trên N. Ở inference, LLM thấy một token tóm tắt cho mỗi clip, thu hẹp đáng kể bối cảnh.

LongVA mở rộng LLM ngữ cảnh từ 200k lên 2M với kỹ thuật "chuyển ngữ cảnh dài". Huấn luyện về văn bản ngữ cảnh dài, chuyển sang video ngữ cảnh dài thông qua biểu diễn được chia sẻ.

Token nén đánh đổi recall ở các dấu thời gian cụ thể để có khả năng mở rộng. Người model biết chung điều gì đã xảy ra nhưng đôi khi bỏ lỡ khung hình chính xác.

### Đường dẫn 4: Truy xuất Agentic (VideoAgent)

Không cung cấp toàn bộ video cho LLM. Thay vào đó, hãy coi video như một cơ sở dữ liệu và sử dụng LLM để truy vấn nó.

Tác nhân video (arXiv: 2403.10517):

1. LLM đọc câu hỏi.
2. LLM yêu cầu một công cụ truy xuất cho các clip có liên quan ("cho tôi xem các phân đoạn với một con mèo").
3. Công cụ trả về dấu thời gian clip phù hợp.
4. LLM đọc những clip đó qua VLM.
5. LLM soạn câu trả lời hoặc hỏi các truy vấn tiếp theo.

Đây là mẫu LLM-as-agent được áp dụng cho video dài. inference rẻ hơn (chỉ mã hóa các clip có liên quan), kỹ thuật khó hơn (chất lượng truy xuất trở thành nút thắt cổ chai).

### Kim trong đống cỏ khô benchmarks

Kiểm tra ngữ cảnh dài tiêu chuẩn: chèn một điểm đánh dấu hình ảnh hoặc văn bản duy nhất tại một điểm ngẫu nhiên trong video, sau đó hỏi một truy vấn yêu cầu gọi lại nó.

Chỉ số: Recall@k trên thời lượng video và vị trí điểm đánh dấu.

Điểm Gemini 2.5 Pro >99% recall ở video dài tối đa 90 phút. Mở 72B models (Qwen2.5-VL-72B, InternVL3-78B) đạt điểm ~85-90% sau 30 phút và giảm xuống sau 60.

VideoAgent có thể khớp hoặc đánh bại các models ngữ cảnh thô trong 2+ giờ vì việc truy xuất sẽ gặp khó khăn nếu công cụ này tốt.

### Chọn con đường nào

Đối với một clip dài 15 phút tại accuracy biên giới: mở 72B + ngữ cảnh gốc thường hoạt động. Chọn Qwen2.5-VL-72B.

Đối với nội dung từ 30 phút đến 1 giờ: LongVILA hoặc Video-XL để mở; Gemini 2.5 Pro để đóng. Thanh chất lượng rất quan trọng - biên giới đóng cửa.

Đối với nội dung 2+ giờ: VideoAgent hoặc các mẫu truy xuất tương tự. Ngoài ra, tóm tắt thành các phần nhỏ hơn và cung cấp các bản tóm tắt theo thứ bậc.

### Mô hình production 2026

Trong thực tế, pipelines video dài production là kết hợp:

1. Chạy sampling FPS động + tổng hợp tích cực trên toàn bộ video (nhận đại diện toàn cầu 100 nghìn token).
2. Chuyển đến VLM 72B để có bản tóm tắt toàn cầu.
3. Nếu người dùng đặt câu hỏi chi tiết, hãy chạy truy xuất agentic sử dụng tóm tắt làm chỉ mục.

Điều này kết hợp ngữ cảnh vũ phu để hiểu toàn cầu và truy xuất chi tiết cục bộ.

## Ứng dụng

`code/main.py`:

- Tính toán ngân sách token cho video từ 1 phút đến 3 giờ với FPS + gộp khác nhau.
- Mô phỏng một cuộc chạy kim trong đống cỏ khô: tiêm một điểm đánh dấu ở một dấu thời gian ngẫu nhiên, đặt câu hỏi, ghi điểm recall.
- Bao gồm trình mô phỏng bộ định tuyến truy xuất agentic chọn các clip cụ thể để cung cấp cho VLM hạ lưu.

Chạy bảng ngân sách và cảm nhận khoảng cách quy mô.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-long-video-strategy-planner.md`. Với thời lượng video và độ phức tạp của truy vấn, nó chọn giữa ngữ cảnh brute, nén và truy xuất agentic, đồng thời tính toán độ trễ + chất lượng kỳ vọng.

## Bài tập

1. Một bài giảng kéo dài 45 phút ở 1 FPS, 81 tokens mỗi khung hình. Tổng tokens? Phù hợp với bối cảnh models nào?

2. Thiết kế một bài kiểm tra kim trong đống cỏ khô: bạn tiêm điểm đánh dấu vào phút nào và định dạng truy vấn chính xác là gì?

3. So sánh ngữ cảnh vũ phu Qwen2.5-VL-72B (ngữ cảnh 80k) với VideoAgent (Claude 3.5 + truy xuất) trên video dài 1 giờ. Cái nào thắng trên recall? Cái nào chiến thắng về độ trễ?

4. Chi phí bộ nhớ của Ring attention thay đổi tỷ lệ tuyến tính theo độ dài trình tự và tuyến tính về số lượng thiết bị. Giải thích lý do tại sao và điều gì không thành công nếu bạn bỏ giai đoạn quay vòng.

5. Đọc Gemini 1.5 Phần 5 về kim trong đống cỏ khô. Bài báo tìm thấy gì về recall ở ranh giới token 1M so với 10M?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| Bối cảnh vũ phu | "Chỉ cần tokens hơn" | Mở rộng LLM ngữ cảnh lên hàng triệu tokens; process mọi thứ trong một lần |
| Nhẫn attention | "Song song kiểu LWM" | Mẫu attention phân tán trong đó mỗi thiết bị giữ một khối và xoay |
| Nén Token | "Tóm tắt tokens" | Giảm tokens trên mỗi clip thông qua máy nén đã học trước khi LLM |
| Kim trong đống cỏ khô | "Kiểm tra NIH" | Chèn một điểm đánh dấu duy nhất vào một điểm ngẫu nhiên, yêu cầu model recall nó vào thời điểm kiểm tra |
| Truy xuất Agentic | "LLM làm công cụ lập kế hoạch truy vấn" | LLM yêu cầu một công cụ truy xuất các clip có liên quan, đọc chúng qua VLM, soạn câu trả lời |
| Đại lý video | "Mẫu truy xuất cho video" | Thiết kế truy xuất agentic chuẩn: câu hỏi -> công cụ -> clip -> trả lời |

## Đọc thêm

- [Gemini Team — Gemini 1.5 (arXiv:2403.05530)](https://arxiv.org/abs/2403.05530)
- [Liu et al. — LWM / RingAttention (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Xue et al. — LongVILA (arXiv:2408.10188)](https://arxiv.org/abs/2408.10188)
- [Shu et al. — Video-XL (arXiv:2409.14485)](https://arxiv.org/abs/2409.14485)
- [Wang et al. — VideoAgent (arXiv:2403.10517)](https://arxiv.org/abs/2403.10517)
