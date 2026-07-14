# Load Testing LLM APIs - Tại sao k6 và Locust Lie

> Máy kiểm tra tải truyền thống không được thiết kế cho phản hồi streaming, độ dài đầu ra thay đổi, số liệu cấp token hoặc độ bão hòa GPU. Hai cái bẫy cắn hầu hết các đội. Bẫy GIL: Phép đo cấp token của Locust chạy tokenization theo Python GIL, cạnh tranh với việc tạo yêu cầu trong điều kiện đồng thời nặng; tokenization tồn đọng sau đó làm tăng độ trễ giữa các token được báo cáo - khách hàng của bạn là nút thắt cổ chai chứ không phải server. Bẫy đồng nhất prompt: prompts giống hệt nhau trong một vòng lặp kiểm tra một điểm trên phân bố token; Lưu lượng truy cập thực có độ dài thay đổi và kết quả khớp tiền tố đa dạng. LLMPerf khắc phục điều này bằng `--mean-input-tokens` + `--stddev-input-tokens`. Lập bản đồ công cụ năm 2026: chuyên ngành LLM (GenAI-Perf, LLMPerf, LLM-Locust, guidellm) cho accuracy cấp token; **k6 v2026.1.0** + **k6 Operator 1.0 GA (Tháng 9 năm 2025)** — streaming-aware, Kubernetes-native được phân phối qua TestRun/PrivateLoadZone CRD, tốt nhất cho các cổng CI/CD; Vegeta cho Go bão hòa tốc độ không đổi; Locust 2.43.3 chỉ với phần mở rộng LLM-Locust cho streaming. Các mẫu tải: trạng thái ổn định, ramp, tăng đột biến (kiểm tra tự động chia tỷ lệ), ngâm (rò rỉ bộ nhớ).

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib, toy realistic-prompt generator + latency collector)
**Kiến thức tiên quyết:** Giai đoạn 17 · 08 (Inference Số liệu), Giai đoạn 17 · 03 (GPU Tự động chia tỷ lệ)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Giải thích hai phản mẫu (bẫy GIL, bẫy đồng nhất prompt) khiến máy kiểm tra tải chung nói dối LLM APIs.
- Chọn một công cụ cho một mục đích nhất định: LLMPerf (chạy benchmark), phần mở rộng k6 + streaming (cổng CI), guidellm (tổng hợp quy mô lớn), GenAI-Perf (NVIDIA tham chiếu).
- Thiết kế bốn kiểu tải (ổn định, đường dốc, tăng đột biến, ngâm) và đặt tên cho chế độ hỏng hóc mà mỗi kiểu bắt được.
- Xây dựng phân phối prompt thực tế bằng cách sử dụng trung bình + stddev của tokens đầu vào thay vì độ dài cố định.

## Vấn đề

Bạn đã kiểm tra LLM endpoint của mình ở 500 người dùng đồng thời. Nó được giữ vững. Bạn shipped. Trong production với 200 người dùng thực tế, dịch vụ đã bị hỏng - P99 TTFT phát nổ GPUs bị ghim.

Hai điều đã xảy ra. Đầu tiên, k6 gửi 500 prompts giống hệt nhau - kết hợp yêu cầu và bộ nhớ đệm tiền tố của bạn làm cho nó trông giống như bạn đang xử lý 500 giải mã đồng thời trong khi bạn thực sự đang xử lý một giải mã. Thứ hai, k6 không theo dõi độ trễ giữa các token trên các phản hồi streaming theo cách mắt trải nghiệm nó; nó thấy một kết nối HTTP, không phải 500 tokens đến trong các khoảng thời gian khác nhau.

Kiểm thử tải cho LLMs là kỷ luật riêng của nó.

## Khái niệm

### Bẫy GIL (Locust)

Locust sử dụng Python và chạy tokenization phía máy khách theo GIL. Trong điều kiện đồng thời cao, tokenizer hàng đợi đằng sau việc tạo yêu cầu. Độ trễ giữa các token được báo cáo bao gồm tokenization tồn đọng phía máy khách. Bạn nghĩ server chậm chạp; đó là bài kiểm tra harness.

Khắc phục: Phần mở rộng LLM-Locust di chuyển tokenization để tách processes hoặc sử dụng harness ngôn ngữ đã biên dịch (k6, LLMPerf sử dụng tokenizers.rs).

### Bẫy đồng nhất prompt

Tất cả các trình kiểm tra tải đã biết đều cho phép bạn định cấu hình một prompt. Trong một thử nghiệm vòng lặp gồm 10.000 lần lặp lại, prompt gửi chính xác mỗi lần. Server luôn nhìn thấy cùng một tiền tố - bộ nhớ đệm tiền tố đạt 100%, thông lượng trông tuyệt vời.

Khắc phục: mẫu từ phân phối prompt. LLMPerf sử dụng `--mean-input-tokens 500 --stddev-input-tokens 150` - độ dài đa dạng, nội dung đa dạng.

### Bốn mẫu tải

1. **Trạng thái ổn định** — RPS không đổi trong 30-60 phút. Bắt bóng: hồi quy hiệu suất cơ bản.
2. **Đường dốc** — tăng tuyến tính RPS từ 0 đến mục tiêu trong 15 phút. Bắt: điểm ngắt dung lượng, bất thường khởi động.
3. **Tăng đột biến** - RPS đột ngột 3-10x trong 2 phút rồi quay lại. Bắt giữ: độ trễ tự động thay đổi quy mô, độ bão hòa hàng đợi, tác động khởi động nguội.
4. **Ngâm** — trạng thái ổn định trong 4-8 giờ. Bắt: rò rỉ bộ nhớ, trôi nhóm kết nối observability tràn.

### Lập bản đồ công cụ năm 2026

**LLMPerf** (Anyscale) — Python nhưng tokenization được Rust hậu thuẫn. Mean/stddev prompts. Nhận thức Streaming. Mặc định tốt nhất cho hiệu suất chạy.

**NVIDIA GenAI-Perf** — tài liệu tham khảo của NVIDIA. Sử dụng ứng dụng khách Triton; Phạm vi số liệu toàn diện. Lưu ý rằng ITL của nó không bao gồm TTFT; LLMPerf bao gồm nó. Hai công cụ tạo ra TPOT khác nhau cho cùng một server.

**LLM-Locust** (TrueFoundry) — Phần mở rộng Locust sửa bẫy GIL. Các chỉ số Locust DSL + streaming quen thuộc.

**GUIDELLM** — điểm chuẩn tổng hợp quy mô lớn.

**k6 v2026.1.0** + **k6 Operator 1.0 GA (Tháng 9 năm 2025)**:
- bản thân k6 (Go, biên dịch, không có GIL) đã thêm các chỉ số nhận biết streaming.
- k6 Operator sử dụng TestRun / PrivateLoadZone CRD để kiểm tra phân tán gốc Kubernetes.
- Tốt nhất cho các cổng CI/CD và thử nghiệm SLA.

**Vegeta **- Đi, đơn giản hơn k6. Độ bão hòa HTTP tốc độ không đổi. Không nhận biết LLM nhưng tốt cho thử nghiệm gateway / giới hạn tốc độ.

**Locust 2.43.3 cổ phiếu** - có bẫy GIL cho LLM. Chỉ với phần mở rộng LLM-Locust.

### Cổng SLA trong CI

Chạy k6 trên PR với:

- 30-50 lần lặp lại mỗi lần ở RPS cơ bản.
- Cổng: P50/P95 TTFT, 5xx < 5%, TPOT dưới ngưỡng.
- Phá vỡ bản dựng khi vi phạm.

### Phân phối prompt thực tế

Xây dựng từ các mẫu lưu lượng truy cập thực (nếu có) hoặc từ các bản phân phối đã xuất bản (ví dụ: ShareGPT prompts cho trò chuyện, HumanEval cho mã). Cung cấp giá trị trung bình + stddev cho LLMPerf. Tránh vòng lặp với một prompt bằng mọi giá.

### Những con số bạn nên nhớ

- k6 Operator 1.0 GA: Tháng Chín 2025.
- K6 v2026.1.0: Chỉ số nhận biết streaming.
- Chạy LLMPerf điển hình: 100-1000 yêu cầu tại X.
- Cổng CI điển hình: 30-50 lần lặp lại mỗi PR.
- Bốn mẫu: ổn định, đường dốc, gai, ngâm.

## Ứng dụng

`code/main.py` mô phỏng thử tải với phân bố prompt thực tế, đo TPOT hiệu quả và thể hiện bẫy prompt đồng đều.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-load-test-plan.md`. Với khối lượng công việc và SLA, chọn công cụ và thiết kế bốn mẫu tải.

## Bài tập

1. Chạy `code/main.py`. So sánh phân phối đồng nhất và phân phối thực tế - khoảng cách ở đâu?
2. Viết script k6 cho cổng CI: TTFT P95 < 800 ms ở 100 đồng thời, runtime 5 phút.
3. Thử nghiệm ngâm của bạn cho thấy trí nhớ tăng 50 MB/hour. Kể tên ba nguyên nhân và thiết bị để chọn giữa chúng.
4. Kiểm tra tăng đột biến từ 10 RPS đến 100 RPS. Thời gian phục hồi dự kiến là bao nhiêu nếu Karpenter + vLLM production-stack được sử dụng (Giai đoạn 17 · 03 + 18)?
5. GenAI-Perf báo cáo TPOT=6ms; LLMPerf báo cáo TPOT=11ms trên cùng một server. Giải thích.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| LLMPerf | "LLM harness" | Công cụ benchmark Anyscale, nhận biết streaming |
| GenAI-Hiệu suất | "Công cụ NVIDIA" | NVIDIA tham khảo harness |
| LLM-Châu chấu | "Châu chấu cho LLMs" | Phần mở rộng châu chấu cố định bẫy GIL |
| hướng dẫn | "benchmark tổng hợp" | Công cụ tổng hợp quy mô lớn |
| Nhà điều hành k6 | "K8s k6" | K6 phân tán dựa trên CRD |
| Bẫy GIL | "Chi phí khách hàng Python" | Tokenization tồn đọng tăng độ trễ được báo cáo |
| Bẫy đồng nhất Prompt | "Nói dối một prompt" | Vòng lặp với cùng một prompt truy cập vào bộ nhớ cache, tăng thông lượng |
| Trạng thái ổn định | "tải không đổi" | RPS phẳng trong N phút |
| Đường dốc | "tuyến tính lên" | 0 để nhắm mục tiêu trong thời gian |
| Tăng đột biến | "Thử nghiệm liên tục" | Hệ số đột ngột sau đó hoàn nguyên |
| Ngâm | "Bài kiểm tra dài" | Giờ phát hiện rò rỉ |

## Đọc thêm

- [TianPan — Load Testing LLM Applications](https://tianpan.co/blog/2026-03-19-load-testing-llm-applications)
- [PremAI — Load Testing LLMs 2026](https://blog.premai.io/load-testing-llms-tools-metrics-realistic-traffic-simulation-2026/)
- [NVIDIA NIM — Introduction to LLM Inference Benchmarking](https://docs.nvidia.com/nim/large-language-models/1.0.0/benchmarking.html)
- [TrueFoundry — LLM-Locust](https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-performance)
- [LLMPerf](https://github.com/ray-project/llmperf)
- [k6 Operator](https://github.com/grafana/k6-operator)
