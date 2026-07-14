# Phục vụ LLM đa vùng và KV Cache địa phương

> Cân bằng tải vòng tròn chủ động có hại cho LLM inference bộ nhớ đệm. Một yêu cầu không đến nút giữ tiền tố của nó sẽ trả toàn bộ chi phí điền trước — khoảng 800 ms ở P50 trên một prompt dài so với ~80 ms với một lần truy cập vào bộ nhớ đệm. Vào năm 2026, mẫu production là bộ định tuyến nhận biết bộ nhớ cache (Bộ định tuyến vLLM trong bộ định tuyến Rust, llm-d) sử dụng các sự kiện và tuyến KV-cache trên kết hợp băm tiền tố. Nghiên cứu gần đây (GORGO) làm cho độ trễ mạng xuyên khu vực trở thành một thuật ngữ rõ ràng trong mục tiêu định tuyến. Các dịch vụ "inference xuyên vùng" thương mại (Bedrock cross-region inference, GKE multi-cluster gateways) coi inference là không rõ ràng - chúng xử lý tính khả dụng, không phải TTFT. JPMorgan và Mayo Clinic đã chạy failover us-east-1 vào tháng 11 năm 2024 với thời lượng ~22 phút. Thực tế DR: 32% các lỗi DR LLM là do các nhóm sao lưu trọng số nhưng quên tệp tokenizer hoặc cấu hình quantization.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, toy prefix-cache-aware router simulator)
**Kiến thức tiên quyết:** Giai đoạn 17 · 04 (Phục vụ vLLM), Giai đoạn 17 · 06 (SGLang RadixAttention)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Giải thích lý do cân bằng tải vòng tròn phá vỡ inference bộ nhớ đệm và định lượng hình phạt TTFT.
- Sơ đồ bộ định tuyến nhận biết bộ nhớ cache: đầu vào (sự kiện KV-cache), thuật toán (đối sánh băm tiền tố), tie-breaker (sử dụng GPU).
- Đặt tên cho trình điều khiển lỗi DR 32% cho LLMs (thiếu tokenizer files / cấu hình quantization) và nêu ba file danh sách kiểm tra DR.
- Phân biệt các dịch vụ thương mại xuyên khu vực (Bedrock CRI, GKE Multi-Cluster Gateway) với định tuyến nhận biết KV.

## Vấn đề

Dịch vụ của bạn chạy trong us-east-1, us-west-2 và eu-west-1. Bạn đặt một ALB ở phía trước với vòng tròn một lượt. Tỷ lệ trúng bộ nhớ đệm tiền tố trong production giảm xuống còn 8%. TTFT P50 tăng gấp ba. Nhật ký vLLM của bạn cho thấy mọi yêu cầu đều phải trả toàn bộ chi phí điền trước.

Vòng tròn là tối ưu cho các dịch vụ không trạng thái. LLM inference có trạng thái theo thiết kế - KV cache mã hóa mọi thứ mà model đã thấy. Định tuyến mù là định tuyến vào bộ nhớ đệm sai.

Riêng biệt, nhóm của bạn có kế hoạch DR. Bạn sao lưu trọng số model vào S3 liên vùng. Một sự cố mất điện trong khu vực xảy ra; bạn cố gắng failover; Bản sao từ chối khởi động. Bạn đã quên tokenizer. json, quantization config và config mở rộng quy mô RoPE nằm trong một nhóm riêng biệt mà bạn không đồng bộ hóa.

Phân phối LLM đa vùng là vấn đề bộ nhớ đệm, vấn đề định tuyến và vấn đề vệ sinh DR — không phải là vấn đề cân bằng tải.

## Khái niệm

### Định tuyến nhận biết bộ nhớ cache

Yêu cầu đến với một prompt. Bộ định tuyến băm tiền tố (giả sử, 512 tokens đầu tiên); Nó hỏi mỗi bản sao "Bạn có tiền tố này được lưu vào bộ nhớ cache không?". Bản sao xuất bản các sự kiện bộ nhớ đệm KV trên kênh pub/sub khi chúng phân bổ và loại bỏ các khối. Router chọn bản sao với trận đấu, rơi vào tie-breaker dựa trên GPU-util nếu không ai làm vậy.

**vLLM Router** (Rust, 2026 production-stack): đăng ký các sự kiện `kv.cache.block_added`, duy trì chỉ mục bản sao băm tiền tố →, định tuyến với tra cứu O(1). Rơi xuống độ sâu hàng đợi ít nhất khi không khớp.

**Bộ định tuyến llm-D**: cùng một mẫu, Kubernetes gốc. Phát hành các sự kiện thông qua API ControlPlane.

**SGLang RadixAttention** (Giai đoạn 17 · 06) là bản sao nội bộ tương đương. Định tuyến bản sao chéo là nghiêm ngặt ngược dòng.

### Con số

TTFT P50 trên 2K-token prompt, Llama 3.3 70B FP8, H100:
- Lần truy cập bộ nhớ cache (cùng một bản sao, tiền tố cư trú): ~80 ms.
- Bỏ lỡ bộ nhớ đệm (nạp trước lạnh): ~800 ms.

Khoảng cách 10x. Nếu bộ định tuyến của bạn đạt 60-80% bộ nhớ đệm tiền tố trên các bản sao, bạn sẽ ước tính hiệu suất của một bản sao ở dung lượng N-replica. Nếu nó đạt 10%, bạn gần đúng là tỷ lệ ngây thơ.

### Cross-region có một hạn chế mới - độ trễ mạng

RTT liên vùng:
- US-East-1 ↔ US-West-2: ~65 mili giây.
- US-East-1 ↔ EU-West-1: ~75 ms.
- US-East-1 ↔ AP-Đông Nam-1: ~220 ms.

Nếu định tuyến nhận yêu cầu từ us-east-1 đến tiền tố nóng trong ap-southeast-1, thì prefill đã lưu (800 → 80 ms) sẽ bị lùn bởi 440 ms khứ hồi. GORGO (nghiên cứu năm 2026) làm rõ điều này - giảm thiểu `prefill_time + network_latency` chung, không chỉ điền trước. Thường thì câu trả lời là tiếp tục định tuyến khu vực ngoại trừ các tiền tố nhiều MB lớn nơi điền trước chiếm ưu thế.

### "inference xuyên vùng" thương mại không giúp ích gì ở đây

AWS Bedrock liên khu vực inference tự động định tuyến các yêu cầu đến các khu vực khác trong thời gian áp lực dung lượng. Nó tối ưu hóa tính khả dụng, không phải TTFT và coi inference là mờ đục. GKE Multi-Cluster Gateway cũng giống nhau - failover cấp độ dịch vụ, không nhận thức được KV cache.

Bạn vẫn cần một bộ định tuyến nhận biết bộ nhớ cache lớp ứng dụng ngay cả khi sử dụng những bộ định tuyến này. Họ xử lý trường hợp "us-east-1 đang bốc cháy". Định tuyến nhận biết bộ nhớ cache xử lý trường hợp TTFT.

### Vệ sinh DR - vấn đề 32% tệp bị thiếu

Số liệu thống kê năm 2026 được trích dẫn rộng rãi: 32% LLM thất bại DR xảy ra do các đội đã sao lưu trọng lượng nhưng quên:

- `tokenizer.json` hoặc `tokenizer.model`
- Cấu hình Quantization (`quantize_config.json`, thang đo AWQ, điểm không GPTQ)
- Cấu hình dành riêng cho Model (mở rộng quy mô RoPE, mặt nạ attention, mẫu trò chuyện)
- config động cơ (`vllm_config.yaml`, mặc định sampling LoRA bảng kê khai bộ điều hợp)

Bản sửa lỗi là tệp kê khai DR tối thiểu ba tệp:

1. Tất cả các tệp trong HF model repo (trọng số + cấu hình + tokenizer).
2. Các config phục vụ dành riêng cho động cơ.
3. Tệp kê khai triển khai (K8s YAML, Dockerfile, khóa phụ thuộc).

Thêm vào đó: chạy một cuộc diễn tập DR hàng quý. Cuộc tập trận US-east-1 của JPMorgan đã đạt 22 phút phục hồi vào tháng 11 năm 2024 chỉ vì kịch bản đã được diễn tập.

### Nơi lưu trữ dữ liệu là trực giao

PHI của khách hàng EU không thể rời khỏi EU. Nếu bộ định tuyến nhận biết bộ nhớ cache của bạn gửi yêu cầu có nguồn gốc từ Paris đến us-east-1 để khớp tiền tố, bạn đã vi phạm GDPR bất kể mức tăng TTFT. Phân vùng bộ định tuyến theo ranh giới nơi cư trú trước khi tối ưu hóa cho bộ nhớ đệm.

### Những con số bạn nên nhớ

- Khoảng cách TTFT trúng bộ nhớ cache so với bỏ lỡ: ~10x (80 ms so với 800 ms trên 2K prompt).
- RTT US-EU liên vùng: ~75 ms.
- DR thất bại: 32% bỏ lỡ cấu hình tokenizer/quant.
- JPMorgan us-east-1 failover tháng 11 năm 2024: 22 phút (SLA 30 phút).

## Ứng dụng

`code/main.py` mô phỏng ba chiến lược định tuyến (vòng tròn, khu vực nhận biết bộ nhớ đệm, toàn cầu nhận biết bộ nhớ đệm) trên khối lượng công việc đa khu vực. Báo cáo tỷ lệ truy cập bộ nhớ đệm, P50/P99 TTFT và hóa đơn liên vùng.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-multi-region-router.md`. Với các khu vực, hạn chế về cư trú và SLA, thiết kế một kế hoạch định tuyến.

## Bài tập

1. Chạy `code/main.py`. Định tuyến giữa các khu vực có độ dài prompt nào so với định tuyến chỉ cục bộ, với 75 ms RTT?
2. Tỷ lệ trúng bộ nhớ đệm của bạn giảm từ 70% xuống 12%. Chẩn đoán ba nguyên nhân có thể xảy ra và các yếu tố quan sát được sẽ xác nhận từng nguyên nhân.
3. Thiết kế tệp kê khai DR cho model lượng tử hóa AWQ 70B được phục vụ trong vLLM với 5 bộ điều hợp LoRA. Liệt kê mọi tệp và config.
4. Tranh luận liệu inference xuyên khu vực của Bedrock có "đủ" cho một fintech với TTFT SLO nghiêm ngặt hay không. Trích dẫn các hành vi cụ thể.
5. Yêu cầu Paris-origin khớp với tiền tố trong us-east-1. Bạn có định tuyến nó không? Viết policy.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Định tuyến nhận biết bộ nhớ cache | "LB thông minh" | Định tuyến đối sánh băm tiền tố đến bản sao KV-cache-holding |
| Sự kiện KV-cache | "cache pub-sub" | Bản sao xuất bản add/evict khối; Chỉ mục bộ định tuyến |
| Hàm băm tiền tố | "Khóa bộ nhớ cache" | Hàm băm của N đầu tiên tokens được sử dụng để tra cứu bộ định tuyến |
| GORGO | "Nghiên cứu định tuyến xuyên vùng" | arXiv 2602.11688; Độ trễ mạng như thuật ngữ rõ ràng |
| inference xuyên vùng | "CRI nền tảng" | Sản phẩm AWS; tính khả dụng failover, không phải nhận thức về TTFT |
| Tệp kê khai DR | "Danh sách sao lưu" | Mọi tệp cần khôi phục — không chỉ trọng lượng |
| Nơi lưu trữ dữ liệu | "Ranh giới GDPR" | Hạn chế pháp lý về khu vực xem dữ liệu người dùng |
| RTT | "Thời gian khứ hồi" | Độ trễ mạng; 75 ms US-EU, 220 ms US-APAC |
| LB nhận biết LLM | "LB truy cập vào bộ nhớ cache" | Bộ định tuyến nhận biết bộ nhớ cache như một danh mục sản phẩm |

## Đọc thêm

- [BentoML — Multi-cloud and cross-region inference](https://bentoml.com/llm/infrastructure-and-operations/multi-cloud-and-cross-region-inference)
- [arXiv — GORGO (2602.11688)](https://arxiv.org/html/2602.11688v1) — sử dụng lại bộ nhớ đệm KV liên khu vực với thời hạn độ trễ mạng.
- [TianPan — Multi-Region LLM Serving Cache Locality](https://tianpan.co/blog/2026-04-17-multi-region-llm-serving-data-residency-routing)
- [AWS Bedrock Cross-Region Inference](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html) — tính khả dụng failover tài liệu.
- [vLLM Production Stack Router](https://github.com/vllm-project/production-stack) — nguồn bộ định tuyến nhận biết bộ nhớ cache.
