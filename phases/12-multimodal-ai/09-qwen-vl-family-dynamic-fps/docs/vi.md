# Dòng Qwen-VL và video Dynamic-FPS

> Họ Qwen-VL — Qwen-VL (2023), Qwen2-VL (2024), Qwen2.5-VL (2025), Qwen3-VL (2025) — là dòng model ngôn ngữ tầm nhìn mở có ảnh hưởng nhất vào năm 2026. Mỗi thế hệ đã đặt cược kiến trúc quyết định duy nhất mà rest của hệ sinh thái mở được sao chép trong vòng mười hai tháng: độ phân giải động gốc thông qua M-RoPE, sampling FPS động với alignment thời gian tuyệt đối, attention cửa sổ trong ViT và các định dạng đầu ra agent có cấu trúc. Đến Qwen3-VL, công thức đã ổn định: encoder 2D-RoPE-ViT với đầu vào tỷ lệ khung hình gốc, máy chiếu MLP vào cơ sở ngôn ngữ Qwen3 lớn và training giai đoạn nhấn mạnh hành vi OCR, grounding và agent là mục tiêu class đầu tiên. Bài học này đọc gia đình theo thứ tự thời gian để bạn hiểu tại sao mọi núm vặn đều ở đúng vị trí của nó.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, M-RoPE encoder + dynamic-FPS sampler)
**Kiến thức tiên quyết:** Giai đoạn 12 · 06 (vá và gói)
**Thời lượng:** ~120 phút

## Mục tiêu học tập

- Tính toán các vòng quay ba trục của M-RoPE (thời gian, chiều cao, chiều rộng) và giải thích lý do tại sao cả ba đều cần thiết.
- Chọn chiến lược sampling FPS động cho video và lý do về accuracy phát hiện sự kiện tokens giây so với phát hiện sự kiện.
- Đặt tên cho bốn bản nâng cấp thế hệ Qwen-VL theo thứ tự và những gì mỗi lần đã bật.
- Kết nối định dạng đầu ra JSON agent kiểu Qwen2.5-VL và phân tích cú pháp lệnh gọi công cụ có cấu trúc từ phản hồi VLM.

## Vấn đề

Qwen-VL shipped vào tháng 8 năm 2023 như một phản ứng trực tiếp với LLaVA-1.5 và BLIP-2. Khoảng cách mà nhóm Qwen nhắm mục tiêu là gấp ba: độ phân giải, video và đầu ra có cấu trúc.

Độ phân giải: LLaVA-1.5 chạy ở 336x336. Tốt cho ảnh, vô dụng đối với hóa đơn tiếng Trung hoặc ảnh chụp màn hình bảng tính dày đặc. Cải tiến đầu tiên của Qwen-VL là 448x448 và đầu ra hộp giới hạn nối đất, cho phép model trỏ vào mọi thứ.

Video: Video LLaMA xếp chồng lên nhau trên mỗi khung hình encoders và đưa chúng vào LLM. Nó hoạt động cho các clip ngắn, không phải cho các video dài nhiều phút mà trục thời gian là tín hiệu. Nhóm Qwen muốn một encoder duy nhất hiểu được thời gian.

Đầu ra có cấu trúc: LLaVA phát ra văn bản dạng tự do. Một agent cần JSON. Qwen-VL được huấn luyện về các định dạng đầu ra JSON rõ ràng bao gồm tọa độ hộp giới hạn dưới dạng văn bản.

Mỗi thế hệ Qwen-VL mở rộng một trong ba trục này.

## Khái niệm

### Qwen-VL (Tháng Tám 2023)

Thế hệ đầu tiên: OpenCLIP ViT-bigG/14 dưới dạng encoder (tham số 2,5B), Q-Former tương thích với LLama (1 bước với 256 truy vấn), cơ sở Qwen-7B. Đóng góp:

- Độ phân giải 448x448 (sau đó là SOTA cho VLM mở).
- Grounding: được huấn luyện về các cặp hình ảnh-văn bản với đầu ra token tọa độ rõ ràng. "Con mèo ở (<box>112, 204), (280, 344)</box>".
- training đa ngôn ngữ tiếng Trung + tiếng Anh ngay từ đầu.

Benchmarks vào thời điểm đó: cạnh tranh với GPT-4V về tiếng Anh, chiếm ưu thế về tiếng Trung. Sự giám sát grounding là tiêu đề thực sự.

### Qwen2-VL (Tháng Chín 2024) — M-RoPE và độ phân giải gốc

Qwen2-VL đã thay thế stack độ phân giải cố định + Q-Former bằng encoder ViT có độ phân giải động nguyên bản. Những thay đổi chính:

- Độ phân giải động gốc. ViT chấp nhận bất kỳ HxW nào chia hết cho 28 (bản vá 14 với merge không gian 2x). Hình ảnh ở 1120x672 (40x24 merged bản vá) tạo ra 960 tokens hình ảnh. Không thay đổi kích thước, không lát gạch, không hình thu nhỏ.
- M-RoPE (RoPE đa phương thức). Mỗi token mang một vị trí 3D (t, h, w) thay vì 1D. Đối với hình ảnh t = 0, đối với video t = frame_index. RoPE quay query/key vectors theo tần số trên mỗi trục. Không có bảng embedding vị trí.
- Máy chiếu MLP. Bỏ Q-Former; sử dụng MLP 2 lớp trên tokens bản vá merged.
- Video với FPS động. Video được lấy mẫu ở tốc độ 1-2 FPS theo mặc định, nhưng model chấp nhận số khung hình tùy ý.

Kết quả: Qwen2-VL-7B phù hợp với GPT-4o trên một số benchmarks đa phương thức và đánh bại nó trên DocVQA (94,5 so với 88,4). Sự thay đổi kiến trúc là bước đi quyết định.

### Qwen2.5-VL (Tháng Hai 2025) — FPS động + thời gian tuyệt đối

Sự thay đổi lớn của Qwen2.5-VL là video. FPS động không chỉ là "lấy mẫu nhiều khung hình hơn khi cần". Bài báo chính thức hóa:

- Thời gian tuyệt đối tokens. Thay vì các chỉ số vị trí (khung 0, 1, 2...), hãy sử dụng dấu thời gian thực tế. "Vào lúc 0:04, con mèo nhảy." model thấy `<time>0.04</time>` tokens xen kẽ với khung tokens.
- FPS động. Lấy mẫu ở tốc độ 1 FPS cho cảnh quay chậm, 4+ FPS cho hành động. Người dùng hoặc huấn luyện viên chọn; M-RoPE thích ứng.
- attention cửa sổ trong ViT. attention không gian được cửa sổ (cục bộ trong các khối) cho thông lượng; attention toàn cầu cứ sau vài lớp.
- Định dạng đầu ra JSON rõ ràng. Được huấn luyện về dữ liệu gọi công cụ: "{\"tool\": \"click\", \"coords\": [380, 220]}". Agent sẵn sàng khi ra khỏi hộp.
- Tỷ lệ MRoPE-v2. Tỷ lệ vị trí với kích thước đầu vào tối đa để video dài 10 phút không vượt quá dải tần số.

Benchmarks: Qwen2.5-VL-72B đánh bại GPT-4o trên hầu hết các benchmarks video, khớp với Gemini 2.0 trên tài liệu và đặt SOTA model mở cho grounding GUI (ScreenSpot: 84% accuracy so với 38% cho GPT-4o).

### Qwen3-VL (Tháng Mười Một 2025)

Qwen3-VL là một bản nâng cấp gia tăng củng cố thay vì phát minh lại: xương sống LLM lớn hơn (Qwen3-72B), mở rộng dữ liệu training, cải thiện OCR, suy luận mạnh mẽ hơn thông qua "chế độ tư duy" Qwen3. ViT và M-RoPE vẫn ở lại. Bài báo tập trung vào dữ liệu và cải tiến training so với kiến trúc.

Bài học rút ra: đến năm 2025, kiến trúc Qwen-VL đã ổn định. Các thế hệ bổ sung thay đổi quy mô điện toán và dữ liệu chứ không phải primitives.

### M-RoPE về mặt toán học

RoPE cổ điển xoay một `q` truy vấn có kích thước `d` theo vị trí `m` sử dụng tọa độ được ghép nối:

```
q_rot[2i]   = q[2i]   * cos(m * theta_i) - q[2i+1] * sin(m * theta_i)
q_rot[2i+1] = q[2i]   * sin(m * theta_i) + q[2i+1] * cos(m * theta_i)
theta_i     = 10000^(-2i/d)
```

M-RoPE chia độ mờ ẩn thành ba dải. Nói `d = 96`. Gán 32 độ mờ cho thái dương, 32 độ mờ cho chiều cao, 32 độ mờ cho chiều rộng. Mỗi dải quay theo vị trí trục riêng của nó. Một bản vá tại (t=5, h=10, w=20) nhận được các vòng quay `R_t(5)`, `R_h(10)` `R_w(20)` áp dụng cho ba dải của nó.

Văn bản tokens sử dụng `t = text_index, h = 0, w = 0` (hoặc lựa chọn chuẩn hóa), giữ khả năng tương thích. Khung hình video sử dụng `t = frame_time, h = row, w = col`. Hình ảnh đơn lẻ sử dụng `t = 0`.

Lợi ích: mã hóa một vị trí xử lý văn bản, hình ảnh và video mà không cần mã phân nhánh hoặc các bảng vị trí khác nhau.

### Logic sampling FPS động

Với video có thời lượng `T` giây và ngân sách tokens mục tiêu `B`:

1. Tính toán FPS tối đa bạn có thể chi trả: `fps_max = B / (T * tokens_per_frame)`.
2. Chọn FPS mục tiêu từ `{1, 2, 4, 8}` đáp ứng `fps <= fps_max`.
3. Nếu chuyển động cao (yêu cầu người dùng phỏng đoán luồng quang học hoặc rõ ràng), hãy chọn FPS cao hơn. Nếu chuyển động thấp, hãy chọn thấp hơn.
4. Lấy mẫu đồng nhất ở FPS đã chọn; chèn `<time>t</time>` tokens giữa các khung.

Qwen2.5-VL huấn luyện logic này một cách ngầm; Tại inference người dùng điều khiển thông qua `fps` parameter. Một chuỗi hành động dài 60 giây ở tốc độ 4 FPS với 81 tokens mỗi khung hình = 19440 tokens, có thể quản lý được trong bối cảnh 32k.

### Đầu ra agent có cấu trúc

Các agent training của Qwen2.5-VL nhắm mục tiêu rõ ràng đến các cuộc gọi công cụ có cấu trúc:

```
{
  "tool": "mouse_click",
  "coords": [1024, 512],
  "button": "left",
  "modifier": null
}
```

Phân tích cú pháp là xác định: JSON phân tích cú pháp trên đầu ra của model. So sánh với dạng tự do "nhấp vào (1024, 512)" yêu cầu xử lý biểu thức chính quy và mơ hồ. Sự thay đổi này là lý do tại sao điểm ScreenSpot của Qwen2.5-VL tăng từ 55% của Qwen2-VL lên 84%.

## Ứng dụng

`code/main.py` thực hiện:

- Tính toán vị trí M-RoPE cho một chuỗi đóng gói trộn văn bản, bản vá hình ảnh và khung video.
- Bộ lấy mẫu FPS động: được đưa ra (thời lượng, ngân sách, motion_level), chọn FPS và phát ra dấu thời gian khung hình.
- Một trình phân tích cú pháp đầu ra JSON của đồ chơi Qwen2.5-VL xử lý các phản hồi lệnh gọi công cụ với các trường tọa độ.

Chạy nó, sau đó cảm nhận sự khác biệt khi bạn hoán đổi FPS cố định cho FPS động trên video dài 5 phút.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-qwen-vl-pipeline-designer.md`. Với tác vụ video (giám sát, agent, nhận dạng hành động, khả năng truy cập), nó phát ra configuration Qwen2.5-VL (ngân sách khung hình, chiến lược FPS, cờ attention cửa sổ, chế độ đầu ra agent) và ước tính độ trễ. Sử dụng tính năng này bất cứ khi nào bạn triển khai model dòng Qwen-VL cho sản phẩm video.

## Bài tập

1. Tính toán vòng quay M-RoPE cho một bản vá tại (t = 3, h = 5, w = 7) với 48 ẩn (16 mỗi dải, theta cơ sở 10000). Hiển thị góc quay cho ba cặp đầu tiên trong mỗi dải.

2. Một camera an ninh 10 phút quay ở tốc độ 1 FPS tạo ra bao nhiêu khung hình? Ở độ phân giải 384 với hồ bơi 3x, tổng cộng tokens bao nhiêu? Ngữ cảnh 32k mặc định của Qwen2.5-VL có xử lý nó không?

3. Chọn FPS cho cuộc biểu tình quần vợt 30 giây so với bản demo công thức 30 giây so với bản ghi UI-agent 30 giây. Biện minh cho mỗi người bằng logic FPS động.

4. Qwen2.5-VL loại bỏ hoàn toàn Q-Former Tại sao MLP đơn giản hoạt động vào năm 2025 nhưng không hoạt động vào năm 2023? (Gợi ý: quy mô dữ liệu và chất lượng encoder.)

5. Phân tích cú pháp ba đầu ra cuộc gọi công cụ JSON Qwen2.5-VL thành Python dicts. Điều gì không thành công đối với JSON dị dạng và sách dạy nấu ăn Qwen đề xuất chiến lược phục hồi nào?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| M-RoPE | "Dây thừng đa phương thức" | embedding vị trí quay 3D với các dải thái dương, chiều cao và chiều rộng trong độ mờ ẩn |
| FPS động | "sampling thông minh" | Tốc độ sampling khung hình được chọn cho mỗi video dựa trên chuyển động, thời lượng và ngân sách token |
| Thời gian tuyệt đối token | "Dấu thời gian token" | `<time>t</time>` xen kẽ trong trình tự để model nhìn thấy giây thực tế chứ không phải chỉ số khung hình |
| Cửa sổ attention | "attention địa phương" | self-attention không gian giới hạn ở windows nhỏ cho tốc độ; attention toàn cầu được thêm vào định kỳ |
| Đầu ra agent có cấu trúc | "Chế độ JSON" | Training giám sát dữ liệu dạy VLM phát ra JSON có thể phân tích cú pháp với tọa độ và tên công cụ |
| min_pixels / max_pixels | "Giới hạn độ phân giải" | Mỗi yêu cầu Qwen2.5-VL kiểm soát giới hạn tổng số pixel và do đó số lượng điểm ảnh token |
| Grounding | "Chỉ vào nó" | Xuất tọa độ hộp giới hạn dưới dạng tokens văn bản; được sử dụng từ Qwen-VL v1 |

## Đọc thêm

- [Bai et al. — Qwen-VL (arXiv:2308.12966)](https://arxiv.org/abs/2308.12966)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Qwen Team — Qwen3-VL (arXiv:2511.21631)](https://arxiv.org/abs/2511.21631)
- [Zhu et al. — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)
