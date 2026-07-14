# Omni Models: Qwen2.5-Omni và sự phân chia của người suy nghĩ-người nói

> Bản demo sản phẩm của GPT-4o vào tháng 5 năm 2024 gây gián đoạn không phải vì model cơ bản mà vì hình dạng sản phẩm - giao diện giọng nói nơi bạn nói chuyện, model nhìn thấy những gì máy ảnh nhìn thấy và nó nói lại trong vòng chưa đầy 250ms. Hệ sinh thái mở đã dành rest năm 2024 và 2025 để chạy đua để đạt được bề mặt sản phẩm đó. Qwen2.5-Omni (tháng 3 năm 2025) là thiết kế mở tham chiếu: Thinker (transformer tạo văn bản lớn) cộng với Talker (transformer tạo giọng nói song song), được liên kết bằng tokens giọng nói streaming. Mini-Omni đơn giản hóa nó, Moshi phù hợp với độ trễ của nó, GLM-4-Voice mở rộng nó sang tiếng Trung. Bài học này đọc kiến trúc Thinker-Talker và ngân sách độ trễ giúp streaming cuộc đối thoại thời gian thực hoạt động.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib, streaming pipeline latency simulator + VAD loop)
**Kiến thức tiên quyết:** Giai đoạn 12 · 19 (âm thanh-LLMs), Giai đoạn 12 · 16 (bất kỳ đến bất kỳ)
**Thời lượng:** ~180 phút

## Mục tiêu học tập

- Chia inference pipeline thành Thinker (suy luận văn bản) và Talker (tổng hợp giọng nói) và giải thích lý do tại sao streaming song song hoạt động.
- Tính toán ngân sách thời gian đến byte âm thanh đầu tiên (TTFAB) cho tương tác đàm thoại, từng thành phần.
- Mô tả mã hóa vị trí được căn chỉnh theo thời gian của TMRoPE trên hình ảnh, âm thanh và văn bản trong Thinker.
- Kể tên ba mẫu hội thoại thời gian thực: bán song công, theo lượt, song công.

## Vấn đề

Một trợ lý giọng nói thời gian thực phải làm rất nhiều, nhanh chóng:

1. Lắng nghe người dùng. tokenization giọng nói thời gian thực, phát hiện hoạt động giọng nói (VAD) để biết khi nào họ nói xong.
2. Tùy chọn xem. Đầu vào máy ảnh ở tốc độ 2-4 FPS, được truyền vào Thinker cùng với âm thanh.
3. Hãy suy nghĩ. Soạn một câu trả lời dựa trên lịch sử cuộc trò chuyện.
4. Nói. Tổng hợp tokens âm thanh, giải mã thành dạng sóng, truyền đến loa của người dùng.

Mỗi bước sẽ tăng thêm độ trễ. Cảm giác đàm thoại yêu cầu tổng thời gian khứ hồi < 500ms - dưới mức đó, người dùng ngừng nhận thấy độ trễ. GPT-4o tuyên bố ~250ms. Moshi ~160ms. Qwen2.5-Omni ~350-500ms.

Mọi thành phần đều cần được phát trực tuyến. Không có gì có thể "batch mọi thứ sau đó giải mã".

## Khái niệm

### Người suy nghĩ và người nói chuyện

Sự phân hủy của Qwen2.5-Omni:

- Thinker: một transformer tạo văn bản 7B-80B. Tiêu thụ tokens văn bản + hình ảnh + âm thanh xen kẽ. Xuất văn bản tokens đại diện cho những gì cần nói.
- Talker: một transformer tạo giọng nói nhỏ hơn (200M-1B). Tiêu thụ đầu ra văn bản của Thinker tokens cộng với tokens ngữ cảnh giọng nói gần đây. Xuất ra tokens giọng nói rời rạc (chỉ số VQ dư).
- decoder giọng nói: một decoder dạng sóng streaming (SNAC, họ MoVQGAN) lấy tokens giọng nói thành các mẫu âm thanh trong thời gian thực.

Sự tách biệt rất quan trọng. Nhà tư tưởng phải lớn vì lý do chính đáng. Talker có thể nhỏ vì công việc của nó là cục bộ - chuyển đổi văn bản thành giọng nói tokens. Bigger Talker không biểu cảm hơn; nó chậm hơn.

Chạy song song cả hai:

1. Thinker phát ra văn bản token t_i.
2. Talker tiêu thụ t_i (thông qua streaming) và phát ra tokens s_i lời nói, s_{i+1}, ..., s_{i+k}.
3. Speech decoder tiêu thụ tokens giọng nói khi chúng đến và phát ra các mẫu âm thanh.
4. Vào thời điểm Thinker ở văn bản token t_{i+3}, Talker đã phát trực tuyến âm thanh cho t_0..t_{i+2}.

### TMRoPE — vị trí đa phương thức được căn chỉnh theo thời gian

Thinker cần tích hợp khung hình ảnh (đạt 4 FPS), khung âm thanh (đến 50 frames/second) và văn bản từ lịch sử hội thoại. Thứ tự trình tự ngây thơ (tất cả hình ảnh, sau đó là tất cả âm thanh, sau đó là văn bản) sẽ mất alignment thời gian.

TMRoPE chỉ định dấu thời gian tuyệt đối cho mọi token. Tầm nhìn token ở t = 2,3 giây. token âm thanh ở t = 2.32 giây. Nhắn tin token từ người dùng "dừng" ở t = 2,35 giây. RoPE xoay attention theo dấu thời gian; model coi chúng là đồng thời về mặt thời gian.

Đây là cơ sở hạ tầng để "anh ấy vẫy tay trong khi nói lời chào" làm việc - model nhìn thấy khung hình video và âm thanh tại cùng một thời điểm khái niệm.

### Tổng hợp giọng nói Streaming

Lời nói tokens phải phát trực tuyến. Mini-Omni (Xie & Wu, 2024) giới thiệu "ngôn ngữ models có thể nghe, nói trong khi suy nghĩ trong streaming": Đầu ra của Thinker tokens và đầu ra của Talker tokens xen kẽ theo cùng một trình tự. Talker sa thải ngay khi Thinker commits tin nhắn tiếp theo token. Không có ranh giới batch.

Moshi (Défossez và cộng sự, tháng 10 năm 2024) là triển khai mở nhanh nhất. TTFAB 160ms trên một A100. Kiến trúc: một transformer 7B duy nhất phát ra văn bản và lời nói tokens ở các vị trí xen kẽ, với một "độc thoại bên trong" ngăn cách luồng tư duy với luồng nói. Đây là Thinker + Talker hợp nhất thành một model với training cẩn thận.

### VAD và lượt

Phát hiện hoạt động giọng nói chạy ở phía đầu vào. Hai mẫu:

- Half-duplex: người dùng nói, model nghe. Model nói, người dùng lắng nghe. Xóa bàn giao thông qua phát hiện im lặng VAD (~200ms).
- Song công: cả hai đều có thể nói đồng thời. Model có thể kênh ngược ("uh-huh") hoặc gián đoạn. Khó hơn nhiều. Moshi ủng hộ điều này.

Qwen2.5-Omni hỗ trợ bán song công theo mặc định, với ngưỡng im lặng thông qua lượt chuyển nhượng. Full-duplex yêu cầu xử lý lớp ứng dụng.

### Qwen3-Omni (Tháng Mười Một 2025)

Người kế nhiệm. Qwen3-80B Thinker, Talker lớn hơn, cải tiến TMRoPE-v2. Độ trễ gần với 250ms của GPT-4o. Trọng lượng mở. Benchmarks trên OmniBench cạnh tranh với Gemini 2.0 Live.

### Production ngân sách độ trễ

Đối với một tương tác streaming điển hình:

- Mic -> tokens âm thanh: 40-80ms.
- Điền trước (prompt + lịch sử): 100-200ms ở 7B, nhiều hơn ở 70B.
- token văn bản của First Thinker: 40ms.
- Talker processes token văn bản đầu tiên: 20ms.
- Bài phát biểu đầu tiên tokens commit: 40ms.
- Giải mã VQ dư: 30ms.
- Giải mã dạng sóng giọng nói: 50-80ms.

Tổng TTFAB: 320-510ms ở 7B, 600-900ms ở 70B. Chất lượng biên giới thường có nghĩa là 70B +; do đó khoảng cách độ trễ biên giới.

### Toán học tỷ lệ Token

Ở giọng nói 16kHz với tokens giọng nói cơ bản 50 Hz, bạn cần 50 tokens giọng nói mỗi giây đầu ra. Talker phải phát ra ≥50 tok/s để theo kịp. Ở thông lượng LLM điển hình là 30-80 tok/s trên H100, một Talker nhỏ (200-300M) đủ nhanh; một 7B Talker sẽ bị tụt lại phía sau.

Đây là lý do tại sao Talker chuyên dụng nhỏ models tồn tại thay vì "chỉ sử dụng model chính".

## Ứng dụng

`code/main.py`:

- Mô phỏng một pipeline Thinker-Talker với tỷ lệ phát thải token giả.
- Tính toán TTFAB cho kích thước model có thể định cấu hình và tốc độ lấy mẫu micrô.
- Thể hiện khả năng quay bán song công với ngưỡng im lặng VAD.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-omni-streaming-budget.md`. Với TTFAB mục tiêu của sản phẩm thoại thời gian thực và bộ feature (tầm nhìn, song ngữ, song công), hãy chọn Qwen2.5-Omni, Qwen3-Omni, Moshi hoặc Mini-Omni và kích thước Thinker/Talker.

## Bài tập

1. TTFAB mục tiêu của bạn là 300ms. Trên 7B Thinker và 300M Talker, hãy viết ra độ trễ của mọi thành phần.

2. Qwen2.5-Omni sử dụng TMRoPE. Mô tả những gì model nhìn thấy trong một prompt mà người dùng bắt đầu nói ở t = 1 giây và máy ảnh bắt cử chỉ ở t = 1,2 giây.

3. Hỗ trợ song công yêu cầu model phát ra âm thanh trong khi nghe. Đề xuất một định dạng dữ liệu training dạy điều này.

4. Đọc bài báo của Moshi Phần 4. Mô tả sự tách biệt "độc thoại nội tâm" và lý do tại sao nó tránh được sự phân chia giữa Thinker-Talker.

5. Tính toán ngân sách thông lượng: Talker phải phát ra tokens nhanh như thế nào để theo kịp giọng nói 16kHz ở 50 tokens/sec lớp cơ sở?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| Nhà tư tưởng | "Bộ não lý luận" | transformer tạo văn bản lớn tạo ra những gì cần nói |
| Người nói chuyện | "Miệng tạo ra lời nói" | transformer nhỏ tạo ra lời nói rời rạc tokens từ văn bản của Thinker |
| TTFAB | "Ngân sách độ trễ" | Thời gian đến byte âm thanh đầu tiên: từ đầu giọng nói của người dùng đến đầu ra mẫu âm thanh đầu tiên |
| TMRoPE | "Dây thừng được sắp xếp theo thời gian" | Mã hóa vị trí bằng cách sử dụng dấu thời gian tuyệt đối trên thị giác, âm thanh, văn bản |
| Bán song công | "Lần lượt" | Người dùng và model thay thế; Chế độ im lặng VAD phát hiện người dùng thực hiện |
| Song công hoàn toàn | "Đồng thời" | Model có thể nói và nghe cùng một lúc; có khả năng backchannel |
| Độc thoại nội tâm | "Tách Moshi" | Thiết kế một model nơi luồng tư duy và luồng nói giao thoa |

## Đọc thêm

- [Xu et al. — Qwen2.5-Omni (arXiv:2503.20215)](https://arxiv.org/abs/2503.20215)
- [Qwen Team — Qwen3-Omni (arXiv:2509.17765)](https://arxiv.org/html/2509.17765v1)
- [Xie & Wu — Mini-Omni (arXiv:2408.16725)](https://arxiv.org/abs/2408.16725)
- [Défossez et al. — Moshi (arXiv:2410.00037)](https://arxiv.org/abs/2410.00037)
- [Zeng et al. — GLM-4-Voice (arXiv:2412.02612)](https://arxiv.org/abs/2412.02612)
