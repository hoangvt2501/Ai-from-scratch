# Models ngôn ngữ video: Tokens và Grounding thời gian

> Video không phải là một stack ảnh. Clip dài 5 giây có thứ tự nhân quả, động từ hành động và thời gian sự kiện mà hình ảnh model thể hiện được. Video-LLaMA (Zhang và cộng sự, Tháng Sáu 2023) shipped LLM video mở đầu tiên với grounding nghe nhìn. VideoChat và Video-LLaVA đã mở rộng mô hình. Đến năm 2025, TMRoPE của Qwen2.5-VL đã thu hẹp khoảng cách với models độc quyền biên giới. Mỗi hệ thống giải quyết các tokens thời gian khác nhau - Q-former trên mỗi clip, concat-pool trên mỗi khung hình, TMRoPE trên token. Bài học này đọc các mẫu, xây dựng bộ lấy mẫu khung đồng nhất so với động và đánh giá các tác vụ grounding thời gian.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib, frame sampler + temporal-grounding evaluator)
**Kiến thức tiên quyết:** Giai đoạn 12 · 08 (LLaVA-OneVision)
**Thời lượng:** ~180 phút

## Mục tiêu học tập

- Giải thích lý do mã hóa vị trí thời gian thay đổi hiệu suất VLM video độc lập với encoder thị giác.
- So sánh sampling khung hình đồng nhất, FPS động và theo sự kiện trên tokens giây so với grounding accuracy.
- Mô tả các thiết kế Q-former-per-clip (Video-LLaMA) so với gộp trên mỗi khung hình (Video-LLaVA) so với M-RoPE-per-token (Qwen2.5-VL).
- Đặt tên cho bốn video benchmarks: VideoMME, TempCompass, EgoSchema, Video-MMMU.

## Vấn đề

Một video dài 1 phút ở tốc độ 30 FPS là 1800 khung hình. Với 196 tokens hình ảnh trên mỗi khung hình (ViT-B ở mức 224), đó là 352k tokens - lớn hơn bất kỳ bối cảnh LLM nào của thời đại 2024.

Có ba chiến lược giảm thiểu:

1. Khung hình mẫu phụ (1-8 FPS tùy thuộc vào nội dung).
2. Nhóm bản vá của mỗi khung tokens tích cực (nhóm hai tuyến tính 3x3 hoặc 4x4).
3. Nén thông qua Q-former lấy clip 16 khung hình và xuất ra 64 tokens.

Mỗi sự đánh đổi là khác nhau. Lấy mẫu phụ mất chi tiết thời gian. Gộp làm mất chi tiết không gian. Q-former thua cả hai một chút nhưng cứu được tokens.

Mã hóa vị trí thời gian là trục khác: làm thế nào để model biết khung 5 đến trước khung 6? Các tùy chọn bao gồm RoPE thời gian 1D đơn giản (Video-LLaMA), embeddings thời gian đã học (Video-LLaVA) và TMRoPE (Qwen2.5-VL, 3D đầy đủ).

## Khái niệm

### Video-LLaMA: Q-former trên mỗi clip + branch âm thanh

Video-LLaMA (2023) là LLM video mở đầu tiên. Kiến trúc:

- Clip 16 khung hình ở tốc độ 2 FPS (tức là 8 giây).
- ViT features trên mỗi khung hình -> Video Q-former tham dự chéo trên tất cả 16 khung hình -> 32 truy vấn đã học -> LLM.
- branch âm thanh song song: dạng sóng -> encoder âm thanh ImageBind -> Audio Q-former -> 32 truy vấn -> LLM.

Điểm mạnh: lý luận chung nghe nhìn. Điểm yếu: độ dài clip cố định, không có thời gian tùy ý grounding.

### Trò chuyện video và Video-LLaVA

VideoChat giữ ý tưởng LLaMA video nhưng bỏ âm thanh và đơn giản hóa. Video-LLaVA (Lin và cộng sự, 2023) đã huấn luyện một encoder hình ảnh duy nhất trên cả hình ảnh và khung hình video ("alignment trước khi chiếu"), đưa ra một đại diện thống nhất. Cả hai đều được đông lạnh-CLIP-encoder + MLP + LLM.

Cả hai đều không xử lý video dài. Cả hai đều là hệ thống khung 8-16.

### Qwen2.5-VL và TMRoPE

Qwen2.5-VL giới thiệu TMRoPE - Vị trí quay phương thức thời gian Embedding. Mỗi bản vá token mang một vị trí (t, h, w) trong đó t là dấu thời gian thực tế (không phải chỉ mục khung).

Sự khác biệt chính so với embedding thời gian đơn giản:

- Thời gian tuyệt đối, không phải chỉ mục. model thấy "ở 4,2 giây" chứ không phải "ở khung hình 15".
- Vòng quay trên mỗi token, không phải trên mỗi clip. Mỗi token hình ảnh xoay độc lập theo dấu thời gian của nó.
- Tương thích với FPS động. Nếu bạn lấy mẫu ở 2 FPS ở đây và 4 FPS ở đó, TMRoPE sẽ xử lý khoảng cách không đồng đều nguyên bản.

TMRoPE cho phép truy vấn "con mèo nhảy vào giây nào?". Máy model có thể xuất ra "ở 4,2 giây". Người LLaMA video chỉ có thể nói "đầu clip".

### Chiến lược sampling khung

Đồng phục: mẫu N khung đồng đều trong suốt thời gian. Đơn giản, mất đỉnh chuyển động.

FPS động: mẫu thích ứng dựa trên cường độ chuyển động. Lưu lượng quang học hoặc sự khác biệt khung hình chọn các phân đoạn chuyển động cao cho sampling dày đặc hơn. Qwen2.5-VL huấn luyện về điều này.

Theo hướng sự kiện: chạy một máy dò nhẹ, lấy mẫu nhiều hơn nơi hành động xảy ra. Được sử dụng bởi VideoAgent.

Khung hình chính + ngữ cảnh: lấy mẫu tại ranh giới cảnh quay + một vài khung hình liền kề. Được sử dụng cho nội dung điện ảnh.

### Gộp trên mỗi khung hình

Ở tốc độ 1 FPS và 576 tokens mỗi khung hình, một clip dài 5 phút là 172.800 tokens. Có thể thực hiện được với bối cảnh 2.5k của Qwen 72-VL-128B nhưng đắt tiền.

Hồ bơi hai tuyến tính 3x3 giảm xuống 64 tokens mỗi khung hình -> 19.200 tokens trong 5 phút. Điểm ngọt ngào cho hầu hết các nhiệm vụ.

Nhóm tích cực hơn (6x6 -> 16 tokens mỗi khung hình) cho agent quy trình làm việc mà chi tiết không gian ít quan trọng hơn.

### Bốn video benchmarks

- VideoMME: hiểu video toàn diện, ngắn + trung bình + dài.
- TempCompass: suy luận thời gian chi tiết, câu hỏi "trước" / "sau".
- EgoSchema: video góc nhìn thứ nhất có đường chân trời dài.
- Video-MMMU: câu hỏi video đa ngành đa phương thức.

Đánh giá VLM video đầy đủ đánh vào cả bốn. Chúng nhấn mạnh các trục khác nhau - TempCompass là tất cả về thứ tự, EgoSchema là suy luận khoảng 3+ phút, VideoMME spans thời lượng.

### Grounding định dạng đầu ra

Định dạng đầu ra cho grounding thời gian:

- Văn bản tự do: "Con mèo nhảy xung quanh mốc 4 giây." Dễ phân tích cú pháp nhưng không chính xác.
- JSON cấu trúc: `{"event": "jump", "start": 4.1, "end": 4.3}`. Qwen2.5-VL huấn luyện điều này.
- Dựa trên Token: `<time>4.1</time>` tokens đặc biệt xen kẽ với câu trả lời. Định dạng nội bộ của Qwen2.5-VL.

Dựa trên Token là chính xác nhất để sử dụng xuôi dòng. Định dạng đầu ra JSON của Qwen2.5-VL phân tích cú pháp trực tiếp.

### Thực tiễn tốt nhất năm 2026

Đối với VLMs video vào năm 2026:

- Encoder: SigLIP 2 với M-RoPE hoặc TMRoPE (Qwen2.5-VL).
- sampling khung hình: FPS động (1-4 tùy thuộc vào chuyển động) với giới hạn khung hình tối đa.
- Gộp trên mỗi khung: 3x3 hai tuyến.
- Đầu ra: JSON có cấu trúc với các trường thời gian + sự kiện.
- Benchmarks: VideoMME + TempCompass cho chung; EgoSchema cho chân trời dài.

## Ứng dụng

`code/main.py` bao gồm:

- Bộ lấy mẫu khung hình FPS đồng nhất và động.
- Một công cụ đánh giá grounding thời gian đồ chơi: cho một sự kiện "ground truth" tại thời điểm T và đầu ra model, điểm số accuracy với dung sai.
- So sánh giữa Video-LLaMA (16 khung hình, Q-former), Video-LLaVA (8 khung hình, MLP), Qwen2.5-VL (FPS động + TMRoPE).

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-video-vlm-frame-planner.md`. Được giao một tác vụ video (giám sát, nhận dạng hành động, grounding thời gian, tóm tắt), nó sẽ chọn bộ lấy mẫu khung hình, hệ số gộp, định dạng đầu ra và bậc accuracy dự kiến.

## Bài tập

1. Đối với bản demo nấu ăn kéo dài 3 phút, hãy chọn FPS đồng nhất so với FPS động. Biện minh bằng số lượng token.

2. TMRoPE bổ sung cụ thể điều gì mà một bảng embedding thời gian đơn giản không thể làm được?

3. Viết một JSON schema cho grounding thời gian mà một VLM có thể học cách phát ra. Bao gồm các trường hợp lỗi.

4. Đọc phần 3 của Video-LLaVA về "Alignment trước khi chiếu". Tại sao điều này tốt hơn training encoders hình ảnh và video riêng biệt?

5. Với bảng xếp hạng VideoMME, khoảng cách giữa model mở hàng đầu và model độc quyền hàng đầu tính đến năm 2026 là bao nhiêu? Bao nhiêu trong khoảng cách đó là do mã hóa thời gian so với quy mô LLM cơ sở?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| grounding thời gian | "Câu trả lời được định vị theo thời gian" | VLM xuất ra một phạm vi dấu thời gian cụ thể khi một sự kiện xảy ra |
| TMRoPE | "Dây thừng đa phương thức thời gian" | Vị trí quay 3D với dấu thời gian tuyệt đối, được sử dụng bởi Qwen2.5-VL |
| FPS động | "sampling nhận biết chuyển động" | Lấy mẫu nhiều khung hình hơn trong các phân đoạn chuyển động cao, ít hơn trong các phân đoạn tĩnh |
| Gộp khung | "Nén không gian trên mỗi khung hình" | Giảm các bản vá trên mỗi khung hình bằng nội suy hai tuyến trước khi LLM |
| Video Q-former | "Máy nén kẹp" | Cross-attention ánh xạ nút cổ chai N khung thành K truy vấn đã học |
| VideoMME | "Băng ghế video" | benchmark video short/medium/long toàn diện, 2500+ mẫu |

## Đọc thêm

- [Zhang et al. — Video-LLaMA (arXiv:2306.02858)](https://arxiv.org/abs/2306.02858)
- [Li et al. — VideoChat (arXiv:2305.06355)](https://arxiv.org/abs/2305.06355)
- [Lin et al. — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Qwen Team — Qwen2.5-VL (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Lin et al. — VILA-1.5 (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
