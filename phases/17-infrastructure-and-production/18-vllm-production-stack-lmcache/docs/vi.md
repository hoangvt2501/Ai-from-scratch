# vLLM Production Stack với LMCache KV Offloading

> stack production của vLLM là tham chiếu Kubernetes triển khai - bộ định tuyến, động cơ và observability được kết nối với nhau. LMCache là lớp giảm tải KV trích xuất KV cache ra khỏi bộ nhớ GPU và sử dụng lại nó trên các truy vấn và công cụ (CPU DRAM, sau đó là disk/Ceph). Đầu nối giảm tải vLLM 0.11.0 KV (Tháng Một 2026) làm cho điều này không đồng bộ và có thể cắm được thông qua Đầu nối API (v0.9.0+). Độ trễ giảm tải không phải là đối mặt với người dùng. LMCache có giá trị ngay cả khi không có tiền tố dùng chung — khi GPU hết khe cắm KV, các yêu cầu bị ưu tiên có thể được khôi phục từ CPU thay vì tính toán lại tính toán nạp trước. Được xuất bản benchmarks trên 16x H100 (80GB HBM) trên 4 a3-highgpu-4g: khi KV cache vượt quá HBM, cả giảm tải CPU gốc và LMCache đều cải thiện đáng kể thông lượng; ở diện tích KV thấp, tất cả các cấu hình đều khớp với đường cơ sở với chi phí nhỏ.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, toy KV-spill simulator)
**Kiến thức tiên quyết:** Giai đoạn 17 · 04 (vLLM Phục vụ Nội bộ), Giai đoạn 17 · 06 (SGLang/RadixAttention)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Sơ đồ các lớp stack production vLLM: bộ định tuyến, động cơ, giảm tải KV, observability.
- Giải thích API KV Offloading Connector (v0.9.0+) và cách đường dẫn không đồng bộ 0.11.0 ẩn độ trễ giảm tải.
- Định lượng thời điểm LMCache CPU-DRAM giúp ích (KV > HBM) so với thêm chi phí chung (KV đủ nhỏ để phù hợp với HBM).
- Chọn giữa vLLM gốc CPU giảm tải và trình kết nối LMCache với các ràng buộc triển khai.

## Vấn đề

Phân phát vLLM của bạn hiển thị GPUs ở mức 100% HBM với các sự kiện ưu tiên bất cứ khi nào tính đồng thời tăng lên. Các yêu cầu bị loại bỏ, xếp hàng lại và bạn điền lại 2K-token prompt bốn lần trong một phút. GPU tính toán được sử dụng cho các lần điền trước dư thừa; Goodput thấp hơn nhiều so với thông lượng thô.

Thêm nhiều chi phí GPUs tuyến tính. Không thể thêm HBM. Nhưng DRAM CPU rẻ - một ổ cắm có 512 GB + với độ trễ thấp hơn HBM nhưng tốt cho KV cache "ấm tạm thời".

LMCache trích xuất KV cache vào DRAM CPU để các yêu cầu bị ưu tiên khôi phục nhanh chóng và các tiền tố lặp đi lặp lại trên các công cụ chia sẻ bộ nhớ đệm mà không cần điền lại mỗi công cụ.

## Khái niệm

### vLLM production-stack

`github.com/vllm-project/production-stack` là tham chiếu Kubernetes triển khai:

- **Bộ định tuyến** - nhận biết bộ nhớ cache (Giai đoạn 17 · 11). Tiêu thụ các sự kiện KV.
- **Động cơ** — vLLM workers. Mỗi GPU hoặc mỗi nhóm TP/PP một người.
- **KV cache giảm tải** — Triển khai LMCache hoặc trình kết nối gốc.
- **Observability** - Prometheus scrape, bảng điều khiển Grafana, traces OTel.
- **Control plane** — khám phá dịch vụ, config, cập nhật luân phiên.

Shipped làm biểu đồ + toán tử Helm.

### Đầu nối giảm tải KV API (v0.9.0+)

vLLM 0.9.0 đã giới thiệu một API kết nối cho các phần phụ trợ KV cache có thể cắm được. Động cơ của bạn giảm tải các khối đến đầu nối; trình kết nối lưu trữ chúng (RAM, đĩa, lưu trữ đối tượng, LMCache). Yêu cầu cần một khối, trình kết nối tải nó trở lại.

vLLM 0.11.0 (tháng 1 năm 2026) bổ sung một đường dẫn giảm tải không đồng bộ — quá trình giảm tải có thể xảy ra trong nền để công cụ không chặn nó trong trường hợp thông thường. Độ trễ và thông lượng từ đầu đến cuối vẫn phụ thuộc vào hình dạng khối lượng công việc, tỷ lệ trúng KV cache và áp suất hệ thống; Các ghi chú riêng của vLLM chỉ ra rằng giảm tải hạt nhân tùy chỉnh có thể làm giảm thông lượng ở tỷ lệ truy cập thấp và lập lịch không đồng bộ có các vấn đề tương tác đã biết với giải mã suy đoán.

### Giảm tải CPU gốc so với LMCache

**Giảm tải vLLM gốc CPU**: engine-local. Lưu trữ các khối KV trong RAM máy chủ. Triển khai nhanh chóng, không có bước nhảy mạng. Không vượt qua động cơ.

**Đầu nối LMCache**: quy mô cụm. Lưu trữ các khối trong server LMCache được chia sẻ (CPU bậc DRAM + Ceph/S3). Các khối có thể truy cập được với bất kỳ công cụ nào. 16x H100 benchmarks xuất bản.

Chọn bản địa khi một động cơ có áp suất HBM. Chọn LMCache khi nhiều công cụ chia sẻ tiền tố (RAG với prompts hệ thống chung, nhiều tenant với các mẫu được chia sẻ).

### Benchmark hành vi

16x H100 (80 GB HBM) trải rộng trên 4 bài kiểm tra a3-highgpu-4g:

- Dấu chân KV thấp (prompts ngắn, đồng thời thấp): tất cả các cấu hình phù hợp với đường cơ sở, LMCache thêm ~3-5% chi phí.
- Dấu chân vừa phải: LMCache bắt đầu giúp sử dụng lại tiền tố trên các công cụ.
- KV vượt quá HBM: giảm tải CPU gốc và LMCache đều cải thiện đáng kể thông lượng; LMCache tăng lớn hơn vì chia sẻ giữa các công cụ.

### Khi LMCache mang tính quyết định

- Phục vụ đa tenant trong đó prompts hệ thống được chia sẻ trên tenants.
- RAG nơi các đoạn tài liệu lặp lại trên các truy vấn.
- Fine-tuned biến thể (LoRA) trên cùng một cơ sở nơi tái sử dụng KV model cơ sở cắt giảm công việc dư thừa.
- Khối lượng công việc nặng ưu tiên: khôi phục từ CPU rẻ hơn so với nạp lại.

### Khi nào KHÔNG bật

- Áp lực HBM nhỏ - bạn phải trả chi phí mà không có lợi ích.
- Ngữ cảnh ngắn (<1K tokens) — thời gian truyền > điền lại.
- Khối lượng công việc một tenant một prompt — không cần sử dụng lại để ghi lại.

### Tích hợp với phân phát phân tách

Giai đoạn 17 · 17 hợp chất phân tách + LMCache: KV chuyển từ bể chứa trước để giải mã đất bể bơi trong LMCache nếu không được sử dụng; các truy vấn tiếp theo lấy từ LMCache. Giai đoạn 17 · 11 bộ định tuyến nhận biết bộ nhớ cache có thể định tuyến đến công cụ có bộ nhớ đệm được chia sẻ cục bộ HOẶC LMCache phù hợp.

### Những con số bạn nên nhớ

- vLLM 0.9.0: API shipped kết nối.
- vLLM 0.11.0 (Jan 2026): đường dẫn giảm tải không đồng bộ; tác động của độ trễ từ đầu đến cuối phụ thuộc vào khối lượng công việc, tốc độ trúng KV và áp suất hệ thống (không đảm bảo tuyệt đối).
- 16x H100 benchmark: LMCache giúp khi dấu chân KV vượt quá HBM.
- Áp suất HBM nhỏ: 3-5% chi phí chung mà không có lợi ích.

```figure
zero-sharding
```

## Ứng dụng

`code/main.py` mô phỏng khối lượng công việc nặng về ưu tiên có và không có LMCache. Báo cáo đã tránh được việc nạp lại, tăng thông lượng và sử dụng HBM hòa vốn.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-vllm-stack-decider.md`. Với hình dạng khối lượng công việc và triển khai vLLM, quyết định gốc so với LMCache so với cả hai.

## Bài tập

1. Chạy `code/main.py`. LMCache bắt đầu thanh toán ở mức sử dụng HBM nào?
2. Một tenant chia sẻ 6K-token system prompt trên 200 queries/hour. Compute dự kiến tiết kiệm LMCache mỗi tenant.
3. LMCache server là một điểm lỗi duy nhất. Thiết kế chiến lược HA (bản sao, dự phòng về gốc).
4. LMCache lưu trữ cho Ceph trên đĩa quay. Đối với 4K-token KV ở 70B FP8 (500 MB), thời gian đọc so với nạp lại là bao lâu?
5. Tranh luận liệu đường dẫn không đồng bộ vLLM 0.11.0 có "miễn phí" hay không - chi phí ẩn ở đâu?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Production-stack | "Triển khai tham chiếu" | Biểu đồ + toán tử Kubernetes Helm của vLLM |
| Kết nối API | "Giao diện phụ trợ KV" | vLLM 0.9.0+ giao diện cửa hàng KV có thể cắm được |
| Giảm tải CPU gốc | "Động cơ-tràn cục bộ" | Lưu trữ KV trong RAM máy chủ của cùng một động cơ |
| LMCache | "cụm KV cache" | KV cache server động cơ chéo trên CPU DRAM + đĩa |
| 0.11.0 không đồng bộ | "Giảm tải không chặn" | Giảm tải ẩn sau luồng động cơ |
| Ưu tiên | "Đuổi ra khỏi nhà để nhường chỗ" | KV cache xáo trộn khi HBM đầy |
| Tái sử dụng tiền tố | "Cùng system prompt" | Nhiều truy vấn chia sẻ phần đầu; Truy cập bộ nhớ cache |
| Bậc Ceph | "Tầng đĩa" | Lưu trữ bền bỉ bên dưới DRAM trong hệ thống phân cấp bộ nhớ đệm |

## Đọc thêm

- [vLLM Blog — KV Offloading Connector (Jan 2026)](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [vLLM Production Stack GitHub](https://github.com/vllm-project/production-stack) - Biểu đồ lái + người điều khiển.
- [LMCache for Enterprise-Scale LLM Inference (arXiv:2510.09665)](https://arxiv.org/html/2510.09665v2)
- [LMCache GitHub](https://github.com/LMCache/LMCache) - Triển khai trình kết nối.
- [vLLM 0.11.0 release notes](https://github.com/vllm-project/vllm/releases) — chi tiết đường dẫn không đồng bộ.
