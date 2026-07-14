# Inference Kinh tế nền tảng — Pháo hoa, Cùng nhau, Cơ sở, Phương thức, Sao chép, Mọi quy mô

> Thị trường inference 2026 không còn GPU thời gian cho thuê. Nó phân nhánh thành silicon tùy chỉnh (Groq, Cerebras, SambaNova), nền tảng GPU (Baseten, Together, Fireworks, Modal) và thị trường ưu tiên API (Replicate, DeepInfra). Pháo hoa tăng giá $1/hr per GPU on May 1, 2026, and $4B định giá trên 10T + tokens/day cho bạn biết công trình model được điều khiển bởi volume. Baseten đóng cửa $300M Series E at $5B vào tháng 1 năm 2026. Quy tắc định vị cạnh tranh rất đơn giản: Fireworks tối ưu hóa độ trễ, Cùng nhau tối ưu hóa độ rộng danh mục, Baseten tối ưu hóa đánh bóng doanh nghiệp, Modal tối ưu hóa DX gốc Python, Replicate tối ưu hóa phạm vi tiếp cận đa phương thức, Anyscale tối ưu hóa Python phân tán. Bài học này cung cấp cho bạn một ma trận mà bạn có thể trao cho người sáng lập.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, toy per-call economics comparator)
**Kiến thức tiên quyết:** Giai đoạn 17 · 01 (Nền tảng LLM được quản lý), Giai đoạn 17 · 04 (vLLM phục vụ nội bộ)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Đặt tên cho ba phân khúc thị trường (silicon tùy chỉnh, nền tảng GPU, API đầu tiên) và ánh xạ từng nhà cung cấp đến một phân khúc.
- Giải thích lý do tại sao giá API "trên mỗi token" model nén theo đường cong chi phí của công cụ phân phối, không phải của phần cứng.
- Tính toán chi phí hiệu quả trên mỗi yêu cầu trên ít nhất ba nhà cung cấp và giải thích khi nào mỗi phút (Baseten, Modal) vượt qua mỗi token.
- Xác định nền tảng nào là mặc định phù hợp cho một khối lượng công việc nhất định (bùng nổ không máy chủ, thông lượng cao ổn định, biến thể fine-tuned, đa phương thức).

## Vấn đề

Bạn đã đánh giá các nền tảng siêu quy mô được quản lý. Bạn quyết định cần một nhà cung cấp hẹp hơn, nhanh hơn - Fireworks cho độ trễ, Together cho chiều rộng, Baseten cho model tùy chỉnh fine-tuned. Bây giờ bạn có sáu lựa chọn thực sự và các trang định giá không thẳng hàng. Màn trình diễn pháo hoa $/M tokens; Baseten shows $/phút; Phương thức hiển thị $/second; Replicate shows $/dự đoán. Bạn không thể so sánh chúng trực tiếp mà không mô hình hóa khối lượng công việc.

Tệ hơn, model kinh doanh đằng sau mỗi trang định giá là khác nhau. Fireworks chạy động cơ tùy chỉnh của riêng mình (FireAttention) trên GPUs chung; Tỷ lệ trên mỗi token phản ánh đường cong sử dụng của chúng. Baseten cung cấp cho bạn Truss + GPUs chuyên dụng; mỗi phút phản ánh tính độc quyền. Phương thức đúng Python phi máy chủ — thanh toán theo giây với khởi động nguội dưới giây. Cùng một đầu ra (phản hồi LLM), ba hàm chi phí khác nhau.

Bài học này models sáu và cho bạn biết khi nào mỗi người chiến thắng.

## Khái niệm

### Ba phân đoạn

**Silicon tùy chỉnh **- Groq (LPU), Cerebras (WSE), SambaNova (RDU). Thông thường, giải mã nhanh hơn 5-10 lần so với GPUdựa trên cùng một cụm model. Cao hơn mỗitoken giá (Groq là ~$0.99/M trên Llama-70B vào cuối năm 2025) nhưng không thể đánh bại đối với các trường hợp sử dụng nhạy cảm với độ trễ. Groq là production Chọn cho giọng nói agents và dịch thuật thời gian thực.

**GPU nền tảng** — Baseten, Together, Fireworks, Modal, Anyscale. Chạy trên NVIDIA (H100, H200, B200 vào năm 2026) hoặc đôi khi là AMD. Lớp kinh tế giữa "cho thuê GPU thô" (RunPod, Lambda) và "dịch vụ quản lý siêu quy mô" (Bedrock).

**Thị trường ưu tiên API** — Sao chép, DeepInfra, OpenRouter, Fal. Danh mục rộng, trả tiền cho mỗi dự đoán hoặc trả tiền theo giây, nhấn mạnh thời gian đến cuộc gọi đầu tiên.

### Fireworks — nền tảng GPU được tối ưu hóa độ trễ

- Công cụ FireAttention (tùy chỉnh); được tiếp thị là độ trễ thấp hơn 4 lần so với vLLM trên các cấu hình tương đương.
- Batch bậc ở mức ~50% mức phí phi máy chủ đối với khối lượng công việc không tương tác.
- Fine-tuned model được phục vụ ở cùng mức giá như model cơ sở - một điểm khác biệt thực sự so với các nhà cung cấp tính phí bảo hiểm cho LoRA của bạn.
- Giữa năm 2026: tăng theo yêu cầu GPU Thuê $1/hour có hiệu lực từ ngày 1 tháng 5 năm 2026. Volume giá cả có thể thương lượng trên quy mô lớn.
- Tín hiệu tài chính: Định giá 4 tỷ đô la, 10T + tokens/day xử lý.

### Cùng nhau — tối ưu hóa chiều rộng

- 200+ models bao gồm các bản phát hành mã nguồn mở trong vòng vài ngày sau khi xuất bản ngược dòng.
- Rẻ hơn 50-70% so với Replicate trên LLM models tương đương - định vị "AI Native Cloud" là volume và danh mục.
- Inference + fine-tuning + training trong một API.

### Baseten — tối ưu hóa cho doanh nghiệp

- Truss framework: model bao bì với các phụ thuộc, bí mật, phục vụ config trong một bản kê khai.
- GPU nằm trong khoảng từ T4 đến B200. Thanh toán theo phút với khả năng giảm thiểu khởi động nguội hợp lý.
- SOC 2 Loại II, sẵn sàng cho HIPAA. Lựa chọn fintech và chăm sóc sức khỏe phổ biến.
- $5B valuation, January 2026 Series E ($300M từ CapitalG, IVP, NVIDIA).

### Phương thức — Tối ưu hóa Python-native

- Cơ sở hạ tầng dưới dạng mã trong Python thuần túy. Trang trí một hàm bằng `@modal.function(gpu="A100")` và triển khai bằng một lệnh.
- Thanh toán theo giây. Lạnh bắt đầu 2-4 giây với chế độ làm ấm trước; <1 giây cho models nhỏ.
- Định giá $87M Series B at $1,1B (2025). Điểm trải nghiệm nhà phát triển mạnh nhất trong các cuộc khảo sát độc lập.

### Sao chép — chiều rộng đa phương thức

- Trả tiền cho mỗi dự đoán. Nền tảng mặc định cho models hình ảnh, video và âm thanh.
- Hệ sinh thái tích hợp (plugin Zapier, Vercel, CMS).
- Ít cạnh tranh hơn về tỷ lệ LLM token nhưng chiến thắng về sự đa dạng đa phương thức.

### Anyscale - Ray-bản địa

- Được xây dựng trên Ray; RayTurbo là công cụ inference độc quyền của Anyscale (cạnh tranh với vLLM).
- Tốt nhất cho khối lượng công việc Python phân tán trong đó bước inference là một nút trong biểu đồ lớn hơn.
- Cụm tia được quản lý; tích hợp chặt chẽ với Ray AIR và Ray Serve.

### Mỗi token so với mỗi phút - khi mỗi người giành chiến thắng

Mỗi token có ý nghĩa khi khối lượng công việc không nhạy cảm với độ trễ và bùng nổ — bạn chỉ phải trả tiền cho những gì bạn sử dụng. Mỗi phút có ý nghĩa khi mức sử dụng cao và có thể dự đoán được - bạn đánh bại mỗi token khi bạn bão hòa GPU.

Quy tắc sơ bộ: đối với khối lượng công việc sử dụng liên tục trên ~30% GPU chuyên dụng, mỗi phút (Baseten, Modal) bắt đầu đánh bại mỗi token (Pháo hoa, Cùng nhau). Dưới mức đó, mỗi token thắng vì bạn tránh phải trả tiền cho việc nhàn rỗi.

### Động cơ tùy chỉnh là hào thực sự

Mọi nền tảng trên vLLM và SGLang đều yêu cầu một công cụ tùy chỉnh. FireAttention, RayTurbo, inference stack của Baseten. Công cụ tùy chỉnh tuyên bố tiếp thị bóng râm - khung trung thực là vLLM + SGLang đại diện cho khoảng 80% inference mã nguồn mở của production và các yếu tố khác biệt ở lớp nền tảng là DX, phân bổ và SLA.

### Những con số bạn nên nhớ

- Pháo hoa GPU Giá thuê: $1/hr tăng có hiệu lực từ ngày 1 tháng 5 năm 2026.
- Tuyên bố pháo hoa: Độ trễ thấp hơn 4 lần so với vLLM trên các cấu hình tương đương.
- Cùng nhau: rẻ hơn 50-70% so với Replicate trên LLMs.
- Định giá cơ sở: $5B (Series E, Jan 2026, $300M vòng).
- Định giá phương thức: 1,1 tỷ đô la (Series B, 2025).
- Nhịp mỗi phút mỗi token trên ~30% mức sử dụng liên tục.

```figure
cost-per-token
```

## Ứng dụng

`code/main.py` so sánh sáu nhà cung cấp trên khối lượng công việc tổng hợp trên các models giá. Báo cáo $/day and effective $/M tokens. Chạy nó để tìm điểm hòa vốn giữa mỗi token và mỗi phút.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-inference-platform-picker.md`. Với hồ sơ khối lượng công việc, SLA và ngân sách, hãy chọn nền tảng inference chính và đặt tên cho người về nhì.

## Bài tập

1. Chạy `code/main.py`. Baseten (mỗi phút) đánh bại Fireworks (mỗi token) với mức sử dụng bền vững nào cho model 70B trên một chiếc H100? Tự mình lấy sự giao thoa và so sánh với quy tắc ngón tay cái.
2. Sản phẩm của bạn phục vụ tạo hình ảnh cộng với trò chuyện cộng với giọng nói thành văn bản. Chọn nền tảng cho từng phương thức và đặt tên cho mẫu gateway thống nhất chúng.
3. Pháo hoa tăng giá thêm $1/hr trên chính model. Model tác động chi phí kết hợp nếu 40% lưu lượng truy cập của bạn chuyển sang batch bậc (giảm 50%).
4. Khách hàng được quản lý yêu cầu SOC 2 Loại II + HIPAA + GPUs chuyên dụng. Ba nền tảng nào khả thi và nền tảng nào giành chiến thắng trên FinOps?
5. So sánh chi phí trên 1.000 dự đoán cho Llama 3.1 70 tỷ trên Fireworks phi máy chủ, Cùng nhau theo yêu cầu, Baseten chuyên dụng và Sao chép API. Cái nào rẻ nhất ở 10 predictions/day? Ở mức 10.000?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Silicon tùy chỉnh | "Chip không GPU" | Groq LPU, Cerebras WSE, SambaNova RDU — được tối ưu hóa để giải mã |
| Chú ý lửa | "Động cơ pháo hoa" | Hạt nhân attention tùy chỉnh; được bán trên thị trường với độ trễ thấp hơn 4 lần so với vLLM |
| Giàn | "Định dạng của Baseten" | Model bản kê khai bao bì; phụ thuộc + bí mật + config phục vụ |
| Mỗi token | "Định giá API" | Sạc theo tokens tiêu thụ; Trả tiền không nhàn rỗi |
| Mỗi phút | "Định giá chuyên dụng" | Sạc bằng đồng hồ treo tường GPU thời gian; chiến thắng khi sử dụng cao |
| Mỗi dự đoán | "Sao chép định giá" | Phí cho mỗi model lần gọi; phổ biến cho image/video |
| Máy bay RayTurbo | "Công cụ Anyscale" | inference độc quyền trên Ray; cạnh tranh với vLLM trên các cụm Ray |
| Bậc Batch | "Giảm giá 50%" | Hàng đợi không tương tác với tỷ lệ giảm; phổ biến trên Pháo hoa, OpenAI |
| Fine-tuned theo tỷ lệ cơ bản | "Pháo hoa LoRA" | Tính phí các yêu cầu được phân phối LoRA theo mức giá của model cơ bản (bộ phân biệt) |

## Đọc thêm

- [Fireworks Pricing](https://fireworks.ai/pricing) — giá mỗi token, batch cấp GPU cho thuê.
- [Baseten Pricing](https://www.baseten.co/pricing/) — mức giá mỗi phút, dung lượng cam kết, cấp doanh nghiệp.
- [Modal Pricing](https://modal.com/pricing) — tốc độ GPU mỗi giây và bậc miễn phí.
- [Together AI Pricing](https://www.together.ai/pricing) — danh mục model và tỷ lệ mỗi token.
- [Anyscale Pricing](https://www.anyscale.com/pricing) — Định giá RayTurbo và Ray được quản lý.
- [Northflank — Fireworks AI Alternatives](https://northflank.com/blog/7-best-fireworks-ai-alternatives-for-inference) - đánh giá so sánh.
- [Infrabase — AI Inference API Providers 2026](https://infrabase.ai/blog/ai-inference-api-providers-compared) — cảnh quan nhà cung cấp.
