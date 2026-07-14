# Công thức VLM trọng lượng mở: Điều gì thực sự quan trọng

> Tài liệu VLM trọng lượng mở 2024-2026 là một rừng bàn cắt bỏ. MM1 của Apple đã thử nghiệm 13 sự kết hợp của encoder hình ảnh, trình kết nối và hỗn hợp dữ liệu. Molmo của Allen AI đã chứng minh chú thích chi tiết của con người đánh bại GPT-4V distillation. Cambri-1 đã so sánh 20+ encoder. Idefics2 chính thức hóa không gian thiết kế năm trục. Prismatic VLMs so sánh 27 công thức nấu ăn training trên một benchmark có kiểm soát. Trong số tất cả những nhiễu đó, một tập hợp nhỏ các kết quả được giữ trên các bài báo: encoder hình ảnh quan trọng hơn kiến trúc kết nối, hỗn hợp dữ liệu quan trọng hơn cả hai và chú thích chi tiết của con người đánh bại dữ liệu tổng hợp chưng cất. Bài học này đọc các bảng đó để bạn không cần phải làm vậy.

**Loại:** Tìm hiểu + phòng thí nghiệm
**Ngôn ngữ:** Python (stdlib, ablation table parser + recipe picker)
**Kiến thức tiên quyết:** Giai đoạn 12 · 05 (Đường cơ sở LLaVA)
**Thời lượng:** ~180 phút

## Mục tiêu học tập

- Đặt tên cho không gian thiết kế VLM năm trục: encoder hình ảnh, trình kết nối, LLM, hỗn hợp dữ liệu, lịch trình giải quyết.
- Đọc bảng cắt bỏ MM1 / Idefics2 / Cambrian-1 và dự đoán núm nào di chuyển một benchmark nhất định.
- Chọn một công thức (encoder, trình kết nối, dữ liệu, độ phân giải) cho một VLM mới với ngân sách điện toán và hỗn hợp tác vụ.
- Giải thích lý do tại sao chú thích chi tiết của con người đánh bại GPT-4V distillation cùng một token tính.

## Vấn đề

Hàng trăm VLMs trọng lượng mở tồn tại. Hầu hết khoảng cách giữa "tốt" và "hiện đại" không phải là kiến trúc. Đó là dữ liệu, lịch trình giải quyết và encoder lựa chọn. Biết nên xoay núm nào trước khi model của bạn hoạt động kém hiệu quả sẽ giúp bạn tiết kiệm được sai lầm 5 triệu GPU giờ.

Sóng năm 2023 (LLaVA-1.5, InstructBLIP, MiniGPT-4) chạy trên cặp phụ đề pretraining + LLaVA-Instruct-150k. Đường cơ sở tốt. Đạt đỉnh khoảng 35% MMMU.

Sóng năm 2024 (MM1, Idefics2, Molmo, Cambrian-1, Prismatic VLMs) đã thực hiện cắt bỏ toàn diện. Kết quả thật đáng ngạc nhiên và thiết thực.

## Khái niệm

### Không gian thiết kế năm trục

Idefics2 (Laurençon et al., 2024) đặt tên cho các trục:

1. Hình ảnh encoder. CLIP ViT-L/14, SO400m/14 SigLIP, ViT-g/14 DINOv2, InternViT-6B. Encoders khác nhau về kích thước bản vá, độ phân giải và mục tiêu pretraining.
2. Kết nối. MLP (2-4 lớp), Q-Former (32 truy vấn + truy vấn chéo), Perceiver Resampler (64 truy vấn), C-Abstractor (tích chập + gộp hai tuyến).
3. Ngôn ngữ model. Llama-3 8B / 70B, Mistral 7B, Phi-3, Gemma-2, Qwen2.5. Kích thước LLM là chi phí tham số chi phối.
4. Training dữ liệu. Cặp phụ đề (CC3M, LAION), xen kẽ (OBELICS, MMC4), hướng dẫn (LLaVA-Instruct, ShareGPT4V, PixMo, Cauldron).
5. Lịch trình giải quyết. Đã sửa 224/336/448, AnyRes, động gốc. Ramped trong training hoặc hằng số.

Mỗi production VLM đưa ra lựa chọn trên mỗi trục. Hầu hết các variance trong điểm MMMU được giải thích bằng trục 1, 4 và 5 - không phải bởi đầu nối bạn chọn.

### Trục 1: Đầu nối encoder >

MM1 Phần 3.2 cho thấy: chuyển đổi từ CLIP ViT-L/14 sang SigLIP SO400m/14 thêm 3+ điểm MMMU. Việc hoán đổi đầu nối từ MLP sang Perceiver Resampler được thêm ít hơn 1 điểm. Idefics2 được sao chép: SigLIP > CLIP, Q-Former ≈ MLP ≈ Perceiver ở cùng số token.

"Cambrian Vision Encoders Match-Up" của Cambrian-1 (Tong et al., 2024) chạy 20+ encoders trên benchmark tập trung vào tầm nhìn (CV-Bench). Đứng đầu bảng xếp hạng là sự kết hợp của DINOv2 và SigLIP; CLIP nằm ở giữa gói; ImageBind và ViT-MAE thấp hơn. Khoảng cách từ CLIP ViT-L đến DINOv2 ViT-g/14 là ~5-7 điểm trên CV-Bench.

encoder mặc định năm 2026 cho VLMs mở là SigLIP 2 SO400m/14 cho ngữ nghĩa + features dày đặc, đôi khi được nối với DINOv2 ViT-g/14 features ("Spatial Vision Aggregator" của Cambrian làm điều này).

### Trục 2: thiết kế đầu nối là rửa

MM1, Idefics2, Prismatic và MM-Interleaved đều đi đến cùng một kết luận: với số lượng token hình ảnh cố định, kiến trúc kết nối hầu như không quan trọng. MLP 2 lớp trên các bản vá gộp trung bình hoạt động trong phạm vi 1 điểm của Q-Former 32 truy vấn với cùng ngân sách token.

Điều quan trọng là số lượng token. Nhiều tokens trực quan hơn = nhiều LLM tính toán = hiệu suất tốt hơn đến một điểm, sau đó lợi nhuận giảm dần. 64 tokens cho mỗi hình ảnh là quá ít đối với OCR. 576-1024 tokens là điểm ngọt ngào cho hầu hết các VLMs mở. 2048+ chỉ trợ giúp cho tài liệu và biểu đồ.

Q-Former vs MLP là một câu hỏi về chi phí, không phải là một câu hỏi chất lượng: Q-Former giới hạn tokens ở mức 32-64 bất kể độ phân giải hình ảnh; MLP phát ra tất cả các bản vá tokens. Đối với đầu vào có độ phân giải cao, Q-Former lưu ngữ cảnh LLM; Đối với độ phân giải thấp, sự khác biệt là nhiễu.

### Trục 3: Kích thước LLM đặt trần

Tăng gấp đôi LLM từ 7B lên 13B một cách đáng tin cậy thêm 2-4 điểm trên MMMU trên mỗi VLM bài báo. Ở mức 70B, bạn bão hòa hầu hết benchmarks. Trần lý luận đa phương thức của VLM là trần lý luận văn bản của LLM - encoder trực quan chỉ có thể nuôi dưỡng nó, không thể lý do cho nó.

Đây là lý do tại sao Qwen2.5-VL-72B và Claude Opus 4.7 nghiền nát MMMU-Pro và ScreenSpot-Pro: bộ não ngôn ngữ rất lớn. VLM 7B không thể thay thế cho VLM 70B thông qua thiết kế đầu nối thông minh.

### Trục 4: dữ liệu — chú thích chi tiết của con người đánh bại distillation

Molmo + PixMo (Deitke et al., 2024) là kết quả năm 2024 mà mọi người nên đọc. Allen AI yêu cầu người chú thích mô tả hình ảnh trong 1-3 phút chuyển giọng nói thành văn bản dày đặc, mang lại 712 nghìn hình ảnh có chú thích dày đặc. Không GPT-4V distillation ở bất kỳ đâu trong dữ liệu training.

Molmo-72B đánh bại Llama-3.2-90B-Vision trên 11/11 benchmarks. Đồng bằng không phải là kiến trúc - đó là chất lượng phụ đề. Chú thích chi tiết của con người chứa nhiều thông tin hơn 5-10 lần trên mỗi hình ảnh so với phụ đề web ngắn và giữ nguyên thực tế nơi GPT-4V distillation ảo giác.

ShareGPT4V (Chen và cộng sự, 2023) và Cauldron (Idefics2) tuân theo cùng một cuốn sách với chú thích hỗn hợp giữa con người + GPT-4V. Xu hướng rất rõ ràng: đối với biên giới năm 2026, mật độ phụ đề > số lượng phụ đề > distillation sự tiện lợi.

### Trục 5: độ phân giải và lịch trình của nó

Số lần cắt bỏ của Idefics2: 384 -> 448 cộng thêm 1-2 điểm. 448 -> 980 với tính năng tách hình ảnh (AnyRes) thêm 3-5 nữa trên OCR benchmarks. Độ phân giải phẳng training ổn định ở accuracy trung bình; Độ phân giải (bắt đầu 224, kết thúc 448 hoặc bản địa) huấn luyện nhanh hơn và kết thúc cao hơn.

Cambrian-1 đã đánh đổi độ phân giải so với tokens: ở điện toán cố định, bạn có thể có nhiều tokens hơn ở độ phân giải thấp hơn hoặc ít tokens hơn ở độ phân giải cao hơn. Độ phân giải cao hơn sẽ giành chiến thắng cho OCR; độ phân giải thấp hơn tokens chiến thắng để hiểu cảnh chung.

Công thức production 2026: huấn luyện Giai đoạn 1 ở 384 cố định, Giai đoạn 2 với độ phân giải động lên đến 1280 cho các tác vụ nặng OCR.

### So sánh có kiểm soát lăng trụ

Prismatic VLMs (Karamcheti et al., 2024) là bài báo kiểm soát tất cả các trục. Cùng một LLM 13B, cùng dữ liệu hướng dẫn, cùng đánh giá - chỉ có một trục thay đổi tại một thời điểm. Kết quả:

- Số lượng token hình ảnh trên mỗi hình ảnh giải thích ~60% variance.
- Lựa chọn Encoder giải thích ~20%.
- Kiến trúc trình kết nối giải thích ~5%.
- Mọi thứ khác (hỗn hợp dữ liệu, bộ lập lịch, LR) còn lại ~15%.

Đây là một sự phân hủy thô, nhưng nó là câu trả lời rõ ràng nhất cho "tôi nên cắt bỏ cái gì trước" trong tài liệu.

### Người hái cho năm 2026

Với bằng chứng, công thức VLM mở mặc định cho một dự án mới vào năm 2026:

- Encoder: SigLIP 2 SO400m/14 ở độ phân giải gốc với NaFlex, được kết hợp với DINOv2 ViT-g/14 cho features dày đặc nếu bạn cần segmentation/grounding.
- Đầu nối: MLP 2 lớp trên bản vá tokens. Bỏ qua Q-Former trừ khi bạn bị hạn chế token.
- LLM: Qwen2.5 / Llama-3.1 / Gemma 2, 7B cho chi phí, 70B cho chất lượng, được chọn theo độ trễ mục tiêu.
- Dữ liệu: PixMo + ShareGPT4V + Cauldron, được bổ sung dữ liệu hướng dẫn theo nhiệm vụ cụ thể.
- Độ phân giải: động (tối thiểu 256, tối đa 1280 pixel mỗi cạnh dài).
- Lịch trình: Giai đoạn 1 alignment (chỉ dành cho máy chiếu), Giai đoạn 2 fine-tune đầy đủ, fine-tune nhiệm vụ cụ thể của Giai đoạn 3.

Mỗi một trong những mặc định đó traces quay trở lại với một sự cắt bỏ có đo lường trong các bài báo được trích dẫn ở cuối bài học này.

## Ứng dụng

`code/main.py` là một trình phân tích cú pháp bảng cắt bỏ và công cụ chọn công thức. Nó mã hóa các bảng cắt bỏ MM1 và Idefics2 (cô đọng) và cho phép bạn truy vấn:

- "Với ngân sách X và nhiệm vụ Y, công thức nào thắng?"
- "Nếu tôi hoán đổi SigLIP cho CLIP trên Llama 7B, MMMU delta dự kiến là bao nhiêu?"
- "Tôi nên cắt bỏ trục nào trước để có câu trả lời có độ tin cậy 80%?"

Đầu ra là danh sách công thức được xếp hạng với benchmark delta dự kiến và khuyến nghị "ablate trước".

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-vlm-recipe-picker.md`. Với hỗn hợp tác vụ mục tiêu, ngân sách tính toán và mục tiêu độ trễ, nó phát ra một công thức đầy đủ (encoder, trình kết nối, LLM, hỗn hợp dữ liệu, lịch trình giải quyết) với các trích dẫn cho việc cắt bỏ biện minh cho mỗi lựa chọn. Ngăn các kỹ sư phát minh lại bàn cắt bỏ Idefics2 mỗi khi một dự án mới VLM bắt đầu.

## Bài tập

1. Đọc MM1 Phần 3.2. Đối với LLM 2B cố định với ngân sách 50 triệu hình ảnh, encoder nào thắng? Câu trả lời có lật ở 13B LLM? Tại sao?

2. Cambrian-1 phát hiện ra rằng việc nối DINOv2 + SigLIP hoạt động tốt hơn một mình trên benchmarks tập trung vào thị giác nhưng không thêm tín hiệu trên MMMU. Dự đoán cái nào benchmarks tăng và cái nào đi ngang.

3. Mục tiêu của bạn là giao diện người dùng di động agent trên LLM 2B. Chọn encoder, trình kết nối, độ phân giải và hỗn hợp dữ liệu. Biện minh cho từng lựa chọn bằng một bảng cắt bỏ cụ thể.

4. Molmo ships 4B và 72B models. 4B cạnh tranh với VLMs 7B đóng; 72B đánh bại Llama-3.2-90B-Vision trên 11/11 benchmarks. Điều đó cho bạn biết gì về giả thuyết cao nguyên kích thước LLM?

5. Thiết kế bảng cắt bỏ để cách ly chất lượng hỗn hợp dữ liệu khỏi chất lượng encoder trên VLM 7B. Tối thiểu bao nhiêu lần chạy training? Đề xuất cài đặt bốn trục.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| Cắt bỏ | "Xoay một núm" | Training nhiều lần chạy khác nhau trong chính xác một trục không gian thiết kế, giữ mọi thứ khác không đổi |
| Kết nối | "Cầu" / "máy chiếu" | Mô-đun có thể huấn luyện ánh xạ tầm nhìn encoder đầu ra vào không gian token của LLM (MLP, Q-Former, Perceiver) |
| Chú thích chi tiết của con người | "Chú thích dày đặc" | Mô tả nhiều câu do con người viết (thường là 80-300 tokens) phong phú hơn văn bản thay thế web |
| Distillation | "GPT-4V chú thích" | Training dữ liệu được tạo ra bởi một VLM độc quyền mạnh mẽ hơn; thuận tiện nhưng dễ bị ảo giác di truyền |
| AnyRes / độ phân giải động | "Đường dẫn độ phân giải cao" | Chiến lược cung cấp hình ảnh lớn hơn độ phân giải gốc của encoder thông qua lát gạch hoặc M-RoPE |
| Độ phân giải ramp | "Chương trình giảng dạy" | Training lịch trình bắt đầu ở độ phân giải thấp và tăng lên, tăng tốc độ học tập alignment |
| Băng ghế tập trung vào tầm nhìn | "CV-Băng ghế / BLINK" | Đánh giá nhấn mạnh nhận thức thị giác chi tiết hơn là lý luận nặng về ngôn ngữ |
| PixMo | "Dữ liệu của Molmo" | Hình ảnh có chú thích dày đặc 712K của Allen AI dataset; Lời nói của con người được phiên âm thành chú thích dày đặc |

## Đọc thêm

- [McKinzie et al. — MM1 (arXiv:2403.09611)](https://arxiv.org/abs/2403.09611)
- [Laurençon et al. — Idefics2 / What matters building VLMs (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Deitke et al. — Molmo and PixMo (arXiv:2409.17146)](https://arxiv.org/abs/2409.17146)
- [Tong et al. — Cambrian-1 (arXiv:2406.16860)](https://arxiv.org/abs/2406.16860)
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865)
