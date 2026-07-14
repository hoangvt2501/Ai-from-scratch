# Capstone 14 — Inference Server giải mã suy đoán

> EAGLE-3 trong vLLM 0.7 ships thông lượng gấp 2.5-3 lần trên lưu lượng truy cập thực. P-EAGLE (AWS 2026) đã đẩy suy đoán song song hơn nữa. SpecForge của SGLang đã huấn luyện những người đứng đầu dự thảo trên quy mô lớn. Trung tâm Speculators của Red Hat đã xuất bản các bản nháp phù hợp cho các models mở chung. TensorRT-LLM đã thực hiện giải mã đầu cơ class đầu tiên trên NVIDIA. production phục vụ năm 2026 stack là vLLM hoặc SGLang với bản nháp dòng EAGLE, FP8 hoặc INT4 quantization và HPA đang chờ đợi. Capstone này sẽ phục vụ hai models mở ở thông lượng cơ bản 2,5x + với báo cáo độ trễ đuôi đầy đủ.

**Loại:** Đá đỉnh
**Ngôn ngữ:** Python (serving), C++ / CUDA (kernel inspection), YAML (configs)
**Kiến thức tiên quyết:** Giai đoạn 3 (học sâu), Giai đoạn 7 (transformers), Giai đoạn 10 (LLMs từ đầu), Giai đoạn 17 (cơ sở hạ tầng)
**Các giai đoạn thực hiện:** P3 · P7 · P10 · Trang 17
**Thời lượng:** 30 giờ

## Vấn đề

Giải mã đầu cơ đã trở thành một mặt hàng vào năm 2026. Đầu dự thảo EAGLE-3 huấn luyện trên các trạng thái ẩn của model mục tiêu và dự đoán N tokens phía trước; Mục tiêu model xác minh trong một lần vượt qua. Tỷ lệ chấp nhận 60-80% chuyển thành thông lượng đầu cuối gấp 2-3 lần. vLLM 0.7 tích hợp điều này một cách nguyên bản. SGLang + SpecForge cung cấp cho bạn training pipeline. Red Hat's Speculators xuất bản các bản nháp phù hợp cho Llama 3.3 70B, Qwen3-Coder-30B MoE, GPT-OSS-120B.

Nghề thủ công nằm trong các hoạt động phục vụ, không phải model. Tỷ lệ chấp nhận trôi dạt theo phân phối lưu lượng truy cập (ShareGPT so với mã so với dữ liệu miền). Độ trễ đuôi khi bị từ chối tồi tệ hơn so với không có suy đoán - bạn phải báo cáo p99 ở nhiều kích thước batch, không chỉ ở trạng thái ổn định tokens/sec. Chi phí cho mỗi 1 triệu tokens so với Anthropic / OpenAI API là đòn bẩy đáng tin cậy.

## Khái niệm

Giải mã suy đoán có hai lớp. Một model **dự thảo** (đầu EAGLE-3, ngram hoặc model căn chỉnh mục tiêu nhỏ hơn) đề xuất k ứng cử viên tokens mỗi bước. model **mục tiêu** xác minh tất cả k trong một lần vượt qua; bất kỳ tiền tố nào được chấp nhận sẽ thay thế con đường tham lam. Tỷ lệ chấp nhận phụ thuộc vào alignment mục tiêu dự thảo và phân phối đầu vào.

EAGLE-3 đánh bại các bản nháp ngram trên hầu hết lưu lượng truy cập. P-EAGLE chạy suy đoán song song cho các cây nháp sâu hơn. Sự đánh đổi: Độ trễ P99 khi bị từ chối cao hơn vì vượt qua xác minh lớn hơn. Người config phân phát phải báo cáo độ trễ theo nhóm kích thước batch để hiển thị điều này.

Triển khai là Kubernetes. vLLM 0.7 chạy một bản sao cho mỗi phân đoạn song song GPU hoặc tensor. HPA tự động thay đổi tỷ lệ khi chờ đợi thay vì CPU. Các lượng tử FP8 (Marlin) và INT4 (AWQ) giữ bộ nhớ GPU bên trong phong bì H100 / H200. Báo cáo từ đầu đến cuối là thông lượng, tỷ lệ chấp nhận, p50/p99 ở mức batch 1/8/32 và tokens 1 triệu đô la.

## Kiến trúc

```
request ingress
    |
    v
vLLM server (0.7) or SGLang (0.4)
    |
    +-- draft: EAGLE-3 heads | P-EAGLE parallel | ngram fallback
    +-- target: Llama 3.3 70B | Qwen3-Coder-30B | GPT-OSS-120B
    |     quantized FP8-Marlin or INT4-AWQ
    |
    v
verify pass: batch k draft tokens through target
    |
    v (accept prefix; resample for rejected suffix)
    v
token stream back to client
    |
    v
Prometheus metrics: throughput, acceptance rate, queue wait, latency p50/p99
    |
    v
HPA on queue-wait metric
```

## Stack

- Phục vụ: vLLM 0.7 hoặc SGLang 0.4
- Phương pháp suy đoán: đầu dự thảo EAGLE-3, đầu cơ song song P-EAGLE, dự phòng ngram
- Dự thảo training: SpecForge (SGLang) hoặc Red Hat Speculators
- Mục tiêu models: Llama 3.3 70B, Qwen3-Coder-30B MoE, GPT-OSS-120B
- Quantization: FP8 (Marlin), INT4 AWQ
- Triển khai: Plugin thiết bị Kubernetes + NVIDIA; HPA trên chỉ số chờ đợi
- Đánh giá: ShareGPT, MT-Bench-v2, GSM8K, HumanEval để đo lường chấp nhận chênh lệch miền
- Tham khảo: Giải mã suy đoán TensorRT-LLM cho đường cơ sở của nhà cung cấp

## Tự xây dựng

1. **Chuẩn bị model mục tiêu.** Chọn Llama 3.3 70B. Lượng tử hóa thành FP8 thông qua Marlin. Triển khai theo vLLM 0.7 trên 1xH100 (hoặc 2x tensor-song song).

2. **Nguồn dự thảo.** Kéo một đầu dự thảo EAGLE-3 được căn chỉnh từ Red Hat Speculators (hoặc huấn luyện một đầu qua SpecForge). Tải vào config giải mã suy đoán của vLLM.

3. **Con số cơ sở.** Trước khi suy đoán: tokens/s ở batch 1/8/32, độ trễ p50/p99 GPU sử dụng. Xuất bản.

4. **Bật EAGLE-3.** Lật config; Chạy lại cùng một benchmark. Báo cáo tăng tốc, tỷ lệ chấp nhận, delta độ trễ đuôi p99.

5. **P-EAGLE.** Cho phép suy đoán song song; đo lường cây nháp sâu hơn so với EAGLE-3 nối tiếp. Báo cáo sự uốn cong mà P-EAGLE giúp so với tổn thương.

6. **Lưu lượng truy cập tên miền.** Chạy lưu lượng truy cập ShareGPT so với HumanEval so với tên miền cụ thể thông qua cùng một server. Đo lường tỷ lệ chấp nhận cho mỗi bản phân phối. Xác định thời điểm gió lùa.

7. **Mục tiêu thứ hai model.** Chạy cùng một pipeline trên MoE Qwen3-Coder-30B. Bản nháp phức tạp hơn (MoE nhiễu định tuyến). Báo cáo.

8. **K8s HPA.** Triển khai trong K8s với `queue_wait_ms` theo dõi HPA. Thể hiện khả năng mở rộng quy mô khi tải gấp ba lần.

9. **So sánh chi phí.** Tính toán $/1 triệu tokens so với Anthropic Claude Sonnet 4.7 và OpenAI GPT-5.4 trên cùng một đánh giá. Xuất bản.

## Ứng dụng

```
$ curl https://infer.example.com/v1/chat/completions -d '{"messages":[...]}'
[serve]     vLLM 0.7, Llama 3.3 70B FP8, EAGLE-3 active
[decode]    bs=8, accepted_tokens_per_step=3.2, acceptance_rate=0.76
[latency]   first-token 42ms, full-response 980ms (620 tokens)
[cost]      $0.34 per 1M output tokens at sustained throughput
```

## Sản phẩm bàn giao

`outputs/skill-inference-server.md` mô tả sản phẩm. Một stack phục vụ được đo lường với giải mã suy đoán, báo cáo benchmark đầy đủ và triển khai K8s.

| Trọng lượng | Tiêu chí | Cách đo lường |
| :-: | --- | --- |
| 25 | Tăng tốc đo được so với đường cơ sở | Thông lượng 2,5x+ ở chất lượng phù hợp trên hai models |
| 20 | Tỷ lệ chấp nhận trên lưu lượng truy cập thực tế | Báo cáo tỷ lệ chấp nhận mỗi phân phối |
| 20 | Kỷ luật độ trễ đuôi P99 | p99 tại batch 1/8/32 có và không có suy đoán |
| 20 | Hoạt động | K8 triển khai, HPA khi chờ đợi rollout mượt mà |
| 15 | Bài viết và phương pháp luận | Giải thích rõ ràng về những gì đã thay đổi và tại sao |
| **100** |||

## Bài tập

1. Đo lường sự suy giảm tỷ lệ chấp nhận khi bản nháp chậm hơn một phiên bản so với mục tiêu (ví dụ: Llama độ lệch 3,3 -> 3,4). Xây dựng cảnh báo giám sát.

2. Triển khai ngram-dự phòng: nếu chấp nhận EAGLE-3 giảm xuống dưới ngưỡng, hãy chuyển sang bản nháp ngram. Báo cáo cải thiện độ tin cậy.

3. Chạy thử nghiệm MoE có kiểm soát: cùng Qwen3-Coder-30B với nhiễu định tuyến được đưa vào so với không có. Đo độ nhạy chấp nhận dự thảo.

4. Mở rộng đến H200 (141 GB). Báo cáo khoảng trống kích thước model trên mỗi bản sao đạt được và liệu bạn có thể phân phối Llama 3.3 70B chưa định lượng hay không.

5. Benchmark giải mã suy đoán TensorRT-LLM trên cùng một phần cứng H100. Báo cáo nơi nó thắng so với vLLM.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| Dự thảo model | "Nhà đầu cơ" | model nhỏ đề xuất N tokens để mục tiêu xác minh |
| ĐẠI BÀNG-3 | "Dự thảo kiến trúc năm 2026" | Người đứng đầu dự thảo được huấn luyện về các trạng thái ẩn mục tiêu; ~75% chấp nhận |
| ĐẠI BÀNG P | "Suy đoán song song" | Cây bản nháp branches xác minh trong một lần vượt qua mục tiêu |
| Tỷ lệ chấp nhận | "Tỷ lệ trúng đích" | Phần tokens dự thảo được chấp nhận mà không lấy mẫu lại |
| Quantization | "FP8 / INT4" | Trọng lượng precision thấp hơn để phù hợp với nhiều model hơn trong bộ nhớ GPU |
| Chờ đợi | "Chỉ số HPA" | Thời gian một yêu cầu chờ trong hàng đợi đang chờ xử lý trước khi inference bắt đầu |
| Trung tâm đầu cơ | "Bản nháp căn chỉnh" | Trung tâm Red Hat Neural Magic của EAGLE dự thảo cho các models mở chung |

## Đọc thêm

- [vLLM EAGLE and P-EAGLE documentation](https://docs.vllm.ai) — tài liệu tham khảo phục vụ stack
- [P-EAGLE (AWS 2026)](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/) — giấy giải mã đầu cơ song song + tích hợp
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — training pipeline đầu dự thảo
- [Red Hat Speculators](https://github.com/neuralmagic/speculators) - trung tâm nháp căn chỉnh
- [TensorRT-LLM speculative decoding](https://nvidia.github.io/TensorRT-LLM/) — thay thế nhà cung cấp
- [Fireworks.ai serving architecture](https://fireworks.ai/blog) — tham khảo thương mại
- [EAGLE-3 paper (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) — giấy phương pháp
- [vLLM repository](https://github.com/vllm-project/vllm) - mã và benchmarks
