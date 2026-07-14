# Inference Chỉ số — TTFT, TPOT, ITL, Goodput, P99

> Bốn chỉ số quyết định liệu việc triển khai inference có hoạt động hay không. TTFT là điền trước cộng với hàng đợi cộng với mạng. TPOT (tương đương ITL) là chi phí giải mã giới hạn bộ nhớ trên mỗi token. Độ trễ từ đầu đến cuối là TTFT cộng với TPOT nhân với độ dài đầu ra. Thông lượng là tokens mỗi giây được tổng hợp trên toàn đội xe. Nhưng điều quan trọng đối với sản phẩm là goodput - một phần yêu cầu đáp ứng mọi SLO cùng một lúc. Thông lượng cao với sản lượng thấp có nghĩa là bạn đang xử lý tokens không bao giờ đến tay người dùng đúng hạn. Số tham chiếu cho Llama-3.1-8B-Instruct trên TRT-LLM vào năm 2026: TTFT trung bình 162 ms, TPOT trung bình 7.33 ms, trung bình E2E 1,093 ms. Luôn báo cáo P50, P90, P99 - không bao giờ chỉ có ý nghĩa. Và xem bẫy đo lường: GenAI-Perf loại trừ TTFT khỏi tính toán ITL, LLMPerf bao gồm nó; hai công cụ không đồng ý về TPOT cho cùng một lần chạy.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, toy percentile calculator and goodput reporter)
**Kiến thức tiên quyết:** Giai đoạn 17 · 04 (vLLM phục vụ nội bộ)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Xác định chính xác TTFT, TPOT, ITL, E2E, thông lượng và goodput và đặt tên cho thành phần mà mỗi người đo lường.
- Giải thích lý do tại sao trung bình là số liệu thống kê sai cho giao bóng LLM và cách đọc P50/P90/P99.
- Xây dựng một đa ràng buộc SLO (ví dụ: TTFT<500 ms VÀ TPOT<15 ms VÀ E2E<2 s) và tính toán goodput so với nó.
- Kể tên hai công cụ benchmark không đồng ý về TPOT cho cùng một lần chạy và giải thích lý do.

## Vấn đề

"Thông lượng của chúng tôi là 15.000 tokens mỗi giây." Vậy thì sao? Nếu 40% yêu cầu vượt quá 2 giây từ đầu đến cuối, người dùng sẽ từ bỏ session. Chỉ thông lượng không cho bạn biết liệu sản phẩm có hoạt động hay không.

Inference có nhiều trục độ trễ và mỗi trục thất bại khác nhau. Prefill có giới hạn tính toán và chia tỷ lệ có độ dài prompt. Giải mã bị ràng buộc với bộ nhớ và mở rộng theo kích thước batch. Độ trễ xếp hàng là một vấn đề vận hành. Mạng là một vấn đề về khoảng cách vật lý. Bạn cần các số liệu riêng biệt cho mỗi số liệu và bạn cần phần trăm và bạn cần một tổng hợp duy nhất cho biết "người dùng có nhận được những gì họ mong đợi không" - đó là điều tốt.

## Khái niệm

### TTFT - thời gian token đầu tiên

`TTFT = queue_time + network_request + prefill_time`

Điền trước chiếm ưu thế khi prompts dài. Trên Llama-3.3-70B FP8 trên H100, prompt 32k mất ~800 ms nạp trước tinh khiết. Thời gian hàng đợi là hành vi của bộ lập lịch khi tải. Yêu cầu mạng là thời gian nối bao gồm TLS. TTFT là độ trễ mà người dùng nhìn thấy trước khi bất kỳ thứ gì phát trở lại.

### TPOT / ITL — độ trễ giữa các token

Nhiều tên cho một số lượng. `TPOT` (thời gian trên mỗi đầu ra token), `ITL` (độ trễ giữa các token), `decode latency per token` — tất cả đều giống nhau. Đó là thời gian giữa tokens phát trực tuyến liên tiếp sau lần đầu tiên.

`TPOT = (decode_forward_time + scheduler_overhead) / tokens_produced`

Trên cùng một Llama-3.3-70B H100 stack với phần nạp trước theo khối, TPOT có nghĩa là ~7 ms. Nếu không có prefill theo khối, trong quá trình nạp trước dài trên một chuỗi lân cận, TPOT có thể tăng đột biến lên 50 ms. Xem P99, không có nghĩa là.

### Độ trễ E2E

`E2E = TTFT + TPOT * output_tokens + network_response`

Đối với đầu ra dài (>500 tokens), E2E bị chi phối bởi TPOT. Đối với đầu ra ngắn với prompts dài, E2E chiếm ưu thế của TTFT. Báo cáo E2E có điều kiện về độ dài đầu ra.

### Thông lượng

`throughput = total_output_tokens / elapsed_time`

Chỉ số tổng hợp. Cho bạn biết hiệu quả của đội xe. Không cho bạn biết sức khỏe theo yêu cầu cá nhân.

### Goodput — số liệu bạn thực sự quan tâm

`goodput = fraction of requests meeting (TTFT <= a) AND (TPOT <= b) AND (E2E <= c)`

SLO là một đa ràng buộc. Một yêu cầu chỉ là "tốt" nếu mọi ràng buộc được giữ vững. Goodput là chia sẻ. Thông lượng cao ở 60% sản lượng tốt là thất bại. Thông lượng thấp hơn ở mức sản lượng tốt 99% là mục tiêu.

Vào năm 2026, goodput là số liệu được sử dụng trong các đệ trình MLPerf Inference v6.0 và theo dõi SLA nội bộ tại các nhà cung cấp nền tảng AI.

### Tại sao mean là thống kê sai

LLM phân phối độ trễ bị lệch phải. Một batch giải mã có một hàng xóm lấp đầy dài có thể ship 500 tokens với TPOT ~7 ms và 20 tokens với TPOT ~60 ms. TPOT trung bình là 9 ms. P99 TPOT là 65 mili giây. Người dùng nhấn P99 thường xuyên - đó là lý do tại sao họ rời đi.

Luôn báo cáo bộ ba (P50, P90, P99). Đối với trải nghiệm người dùng, P99 là ứng dụng bạn tối ưu hóa.

### Số tham chiếu - Llama-3.1-8B-Hướng dẫn về TRT-LLM, 2026

- TTFT trung bình: 162 ms
- TPOT trung bình: 7.33 ms
- E2E trung bình: 1,093 ms
- P99 TPOT: thay đổi 10-25 ms tùy thuộc vào configuration lấp đầy trước theo khối.

Đây là các điểm tham khảo NVIDIA đã được công bố. Chúng thay đổi theo kích thước model (70B sẽ hiển thị 3-5x), phần cứng (H100 so với B200 ~3x) và tải.

### Bẫy đo lường

Hai trong số các công cụ benchmark năm 2026 được sử dụng nhiều nhất không đồng ý về TPOT trong cùng một thời gian:

- **NVIDIA GenAI-Perf**: loại trừ TTFT khỏi tính toán ITL. ITL bắt đầu từ token 2.
- **LLMPerf**: bao gồm TTFT. ITL bắt đầu từ token 1.

Đối với yêu cầu có TTFT 500 ms và 100 đầu ra tokens trong tổng thời gian giải mã 700 ms, GenAI-Perf báo cáo `ITL = 700/99 = 7.07 ms`, LLMPerf báo cáo `ITL = 1200/100 = 12.00 ms`. Lựa chọn công cụ thay đổi con số.

Luôn nêu rõ công cụ nào. Luôn công bố định nghĩa.

### Xây dựng SLO

Một SLO hợp lý dành cho người tiêu dùng cho model trò chuyện 70 tỷ vào năm 2026:

- TTFT P99 < = 800 mili giây.
- TPOT P99 < = 25 mili giây.
- E2E P99 < = 3 giây cho đầu ra <300-token.
- Mục tiêu Goodput > = 99%.

SLO doanh nghiệp thắt chặt TTFT (200-400 ms) và nới lỏng E2E. Vấn đề là viết chúng ra, đo lường cả ba và theo dõi goodput dưới dạng một tổng hợp duy nhất.

### Cách đo lường

- Chạy lưu lượng truy cập thực hoặc tổng hợp thực tế (LLMPerf với `--mean-input-tokens 800 --stddev-input-tokens 300 --mean-output-tokens 150`).
- Nhắm mục tiêu đồng thời cao nhất gấp 2 lần cho lần chạy benchmark.
- Chạy 30-50 lần lặp, lấy phần trăm của mẫu kết hợp.
- Xuất bản với tên công cụ, phiên bản công cụ, model, phần cứng, đồng thời prompt phân phối.

```figure
throughput-latency
```

## Ứng dụng

`code/main.py` là một máy tính goodput đồ chơi. Tạo phân phối độ trễ tổng hợp, áp dụng SLO và tính toán goodput. Đồng thời hiển thị sự khác biệt giữa GenAI-Perf và LLMPerf TPOT trên cùng một trace.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-slo-goodput-gate.md`. Với khối lượng công việc và SLO, nó tạo ra một công thức CI/CD-ready benchmark để triển khai trên goodput chứ không phải thông lượng.

## Bài tập

1. Chạy `code/main.py`. Tạo phân phối với 1% tăng đột biến đuôi. Goodput thay đổi như thế nào khi bạn siết chặt P99 TPOT từ 30 ms xuống 15 ms?
2. Một nhà cung cấp báo giá "15.000 tok/s trên Llama 3.3 70B H100". Kể tên ba câu hỏi để hỏi trước khi tin tưởng nó.
3. Tại sao prefill theo khối bảo vệ P99 TPOT nhưng không có nghĩa là TPOT?
4. Xây dựng SLO dành cho người tiêu dùng cho trợ lý giọng nói (token đầu tiên được nghe, không được đọc). Chỉ số nào người dùng hiển thị nhiều nhất?
5. Đọc tài liệu LLMPerf README và GenAI-Perf. Xác định ba chỉ số khác mà các công cụ không đồng ý.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| TTFT | "Đã đến lúc token đầu tiên" | Hàng đợi + mạng + điền trước; bị chi phối bởi prefill ở prompts dài |
| TPOT | "Thời gian trên mỗi đầu ra token" | Chi phí giải mã giới hạn bộ nhớ trên mỗi token sau lần đầu tiên |
| ITL | "Độ trễ giữa các token" | Giống như TPOT trong hầu hết các công cụ (không phải tất cả — xem GenAI-Perf) |
| Tập 2E | "từ đầu đến cuối" | TTFT + TPOT * output_len; Mạng phía phản hồi ở trên cùng |
| Thông lượng | "tok/s" | Hiệu quả đội xe; vô dụng mà không có phần trăm độ trễ |
| Goodput | "Tỷ lệ đáp ứng SLO" | Phần yêu cầu đáp ứng đồng thời mọi ràng buộc SLO |
| P99 · | "đuôi" | Độ trễ 1 trong 100 trường hợp xấu nhất; Chỉ số trải nghiệm người dùng |
| Đa ràng buộc SLO | "khớp" | VÀ của cả ba giới hạn độ trễ; Yêu cầu không thành công nếu bất kỳ yêu cầu nào bị vi phạm |
| GenAI-Perf so với LLMPerf | "Bẫy công cụ" | Các công cụ không đồng ý về việc ITL có bao gồm TTFT hay không |

## Đọc thêm

- [NVIDIA NIM — LLM Benchmarking Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) — định nghĩa chính tắc của TTFT, ITL, TPOT.
- [Anyscale — LLM Serving Benchmarking Metrics](https://docs.anyscale.com/llm/serving/benchmarking/metrics) - định nghĩa thay thế và công thức đo lường.
- [BentoML — LLM Inference Metrics](https://bentoml.com/llm/inference-optimization/llm-inference-metrics) - đo lường được áp dụng trên triển khai thực tế.
- [LLMPerf](https://github.com/ray-project/llmperf) - benchmark mã nguồn mở dựa trên Ray.
- [GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c++/perf_analyzer/genai-perf/README.html) - công cụ benchmark của NVIDIA.
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) - benchmark dựa trên goodput được ngành công nghiệp chấp nhận.
