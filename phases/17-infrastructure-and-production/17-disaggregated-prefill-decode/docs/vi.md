# Prefill/Decode phân tách - NVIDIA Dynamo và llm-d

> Điền trước bị ràng buộc tính toán; giải mã bị ràng buộc với bộ nhớ. Chạy cả hai trên cùng một GPU lãng phí một tài nguyên. Phân tách tách chúng thành các nhóm riêng biệt và chuyển KV cache giữa chúng qua NIXL (dự phòng RDMA/InfiniBand hoặc TCP). NVIDIA Dynamo (công bố GTC 2025, 1.0 GA) nằm trên vLLM/SGLang/TRT-LLM — Planner Profiler + SLA Planner Planner tự động khớp tỷ lệ trước: giải mã để đáp ứng SLO. NVIDIA công bố mức tăng thông lượng trong sân bóng này - developer.nvidia.com (2025-06) cho thấy sự cải thiện ~6 lần đối với DeepSeek-R1 MoE trên GB200 NVL72 + Dynamo trong chế độ độ trễ trung bình và trang sản phẩm Dynamo (developer.nvidia.com, không ghi ngày) quảng cáo thông lượng MoE lên đến 50 lần trên GB300 NVL72 + Dynamo vs Hopper. Con số "30x" là tổng hợp cộng đồng trên các báo cáo Blackwell + Dynamo + DeepSeek-R1 đầy đủ stack; Chúng ta chưa tìm thấy một nguồn chính nào nói chính xác 30x, vì vậy hãy coi nó như một tuyên bố định hướng. llm-d (Red Hat + AWS) là Kubernetes-native: điền trước / giải mã / bộ định tuyến dưới dạng Dịch vụ độc lập với HPA cho mỗi vai trò. llm-d 0.5 bổ sung giảm tải KV phân cấp, định tuyến LoRA nhận biết bộ nhớ đệm, mạng UCCL, mở rộng quy mô bằng không. Kinh tế: tổng hợp nội bộ của nhiều tiết lộ khách hàng cho thấy tiết kiệm 30–40% cho $2M-class inference spend (i.e., $ 600-800K/year) khi chuyển từ phân phối cùng vị trí sang phân phối với Dynamo ở SLA không đổi; con số cụ thể $2M→$600-800K là một tổng hợp nội bộ, không phải là một nghiên cứu điển hình được công bố duy nhất - sử dụng nó như một mỏ neo theo thứ tự độ lớn, không phải là một trích dẫn tham khảo. prompts ngắn hạn (<512 tokens, sản lượng ngắn) không biện minh cho chi phí chuyển nhượng.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, toy disaggregated-vs-colocated simulator)
**Kiến thức tiên quyết:** Giai đoạn 17 · 04 (vLLM Phục vụ Nội bộ), Giai đoạn 17 · 08 (Inference Chỉ số)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Giải thích lý do tại sao việc nạp trước và giải mã có phân bổ GPU tối ưu khác nhau và định lượng chất thải khi đặt máy chủ.
- Sơ đồ kiến trúc phân tách: prefill pool, decode pool, KV transfer qua NIXL, router.
- Đặt tên cho điều kiện khi phân tách KHÔNG được đền đáp (prompts ngắn, đầu ra ngắn).
- Phân biệt NVIDIA Dynamo (stack trở lên) với llm-d (Kubernetes-native) và khớp từng loại với ngữ cảnh hoạt động.

## Vấn đề

Bạn chạy Llama 3.3 70B trên 8 H100. Trong khối lượng công việc hỗn hợp (prompts dài + đầu ra ngắn), GPUs không hoạt động trong quá trình giải mã vì phần lớn điện toán được dành cho việc điền trước. Dưới khối lượng công việc khác nhau (prompts ngắn + đầu ra dài), điều ngược lại sẽ xảy ra. Điền trước / giải mã cùng vị trí có nghĩa là bạn cung cấp quá nhiều cả hai.

Tác động ngân sách: 20-40% thời gian GPU bị lãng phí vào sai nguồn lực. Bạn đang mua điện toán H100 để chạy giải mã giới hạn bộ nhớ hoặc mua băng thông H100 HBM để chạy tính năng nạp trước liên kết điện toán. Cả hai đều là chất thải đắt tiền.

Phân tách tách điền trước và giải mã thành các nhóm riêng biệt có kích thước cho nút cổ chai của mỗi nhóm. KV cache chuyển từ nhóm nạp trước sang nhóm giải mã thông qua kết nối băng thông cao.

## Khái niệm

### Tại sao các nút thắt cổ chai khác nhau

**Điền trước **- chạy transformer trên prompt đầu vào đầy đủ trong một chuyển tiếp. Phép nhân ma trận chiếm ưu thế; ràng buộc điện toán. H100 FP8 cung cấp ~2000 TFLOPS thông lượng hữu ích. Batch hiệu quả là tốt - một tiền đạo processes nhiều tokens.

**Giải mã** — tạo từng token một, đọc trọng số đầy đủ mỗi lần lặp. Giới hạn băng thông bộ nhớ. HBM3 cho ~3 TB/s. Batch hiệu suất chỉ tốt ở đồng thời cao - trọng số đọc được khấu hao trên toàn batch.

Sắp xếp chúng: bạn mua GPUs được tối ưu hóa cho cả hai. H100 tốt ở cả hai nhưng chi phí theo cách nào cũng như nhau. Trên quy mô lớn, bạn muốn nhóm điền trước trên H100 / điện toán nặng; giải mã nhóm trên H200 / bộ nhớ nặng hoặc với quantization mạnh.

### Kiến trúc

```
            ┌──────────────┐
  Request → │    Router    │ ───────────────────────┐
            └──────┬───────┘                        │
                   │                                │
                   ▼ (prompt only)                  │
            ┌──────────────┐    KV cache    ┌───────▼──────┐
            │ Prefill pool │ ─── NIXL ────► │ Decode pool  │
            │  (compute)   │                │  (memory)    │
            └──────────────┘                └──────┬───────┘
                                                   │ tokens
                                                   ▼
                                                 Client
```

NIXL là transport liên nút của NVIDIA. Sử dụng RDMA/InfiniBand khi có sẵn, TCP dự phòng nếu không. Độ trễ truyền là có thật - thường là 20-80 ms đối với KV cache 4K-token prompt trên 70B FP8. Đây là lý do tại sao prompts ngắn hạn không biện minh cho việc phân tách: thuế chuyển nhượng vượt quá số tiền tiết kiệm.

### Máy phát điện so với llm-d

**NVIDIA Dynamo** (công bố GTC 2025, 1.0 GA):
- Nằm trên vLLM, SGLang, TRT-LLM với tư cách là một người điều phối.
- Planner Profiler đo lường khối lượng công việc, SLA Planner tự động định cấu hình tỷ lệ điền trước:giải mã.
- Rust lõi, Python khả năng mở rộng.
- Tăng thông lượng: NVIDIA báo cáo gấp 6 lần đối với DeepSeek-R1 MoE trên GB200 NVL72 + Dynamo ở chế độ độ trễ trung bình (developer.nvidia.com, 2025-06); báo cáo cộng đồng về "lên đến 30 lần" trên Blackwell + Dynamo + DeepSeek-R1 đầy đủ stacks thiếu một nguồn chính duy nhất và nên được coi là định hướng.
- GB300 NVL72 + Dynamo: thông lượng MoE lên đến 50 lần so với Hopper theo trang sản phẩm Dynamo (developer.nvidia.com, không ghi ngày).

**llm-d** (Red Hat + AWS, gốc Kubernetes):
- Điền trước / giải mã / bộ định tuyến dưới dạng Dịch vụ Kubernetes độc lập.
- HPA cho mỗi vai trò với tín hiệu độ sâu hàng đợi (điền trước) / sử dụng KV (giải mã).
- `topologyConstraint packDomain: rack` đóng gói các nhóm điền sẵn + giải mã trên cùng một giá để truyền KV băng thông cao.
- llm-d 0.5 (2026): giảm tải KV phân cấp, định tuyến LoRA nhận biết bộ nhớ đệm, mạng UCCL, mở rộng quy mô về không.

Sử dụng Dynamo nếu bạn muốn có một trình điều phối stack được quản lý. Sử dụng llm-d nếu bạn muốn primitives bản địa Kubernetes và cam kết với hệ sinh thái CNCF.

### Kinh tế học

Tổng hợp nội bộ (không phải một nghiên cứu điển hình được công bố duy nhất - neo theo thứ tự độ lớn):

- $2M/year inference chi tiêu cho việc phục vụ cùng vị trí.
- Chuyển sang phân tách với Dynamo.
- Cùng một yêu cầu volume, cùng SLA độ trễ P99.
- Tiết kiệm được báo cáo: $600K–$ 800K/year (giảm 30–40%).
- Không có phần cứng mới.

Chúng ta tổng hợp con số này từ nhiều tiết lộ của khách hàng thay vì một nghiên cứu điển hình có thể trích dẫn duy nhất; Điểm dữ liệu được công bố gần nhất là TTFT nhanh hơn 2 lần / thông lượng cao hơn 61% của Baseten với định tuyến Dynamo KV (baseten.co, 2025-10) và dự báo của VAST + CoreWeave là tokens/$ cao hơn 60–130% ở tỷ lệ trúng KV 40–60% (vastdata.com, 2025-12). Khoản tiết kiệm đến từ việc kích thước phù hợp với từng hồ bơi; khối lượng công việc nặng chèn sẵn (RAG có tiền tố 8K+) được hưởng lợi nhiều hơn khối lượng công việc cân bằng.

### Khi nào KHÔNG phân tách

- Prompts < 512 tokens và đầu ra < 200 tokens: thuế chuyển nhượng chi phối lợi nhuận.
- Cụm nhỏ (< 4 GPUs): không đủ đa dạng hồ bơi.
- Nhóm không thể vận hành hai nhóm GPU với quy mô cho mỗi vai trò: Dynamo giúp nhưng không tầm thường.
- Không có vải RDMA: Thuế chuyển nhượng TCP nặng hơn.

### Bộ định tuyến tích hợp với Giai đoạn 17 · 11

Các bộ định tuyến phân tách nhận biết bộ nhớ cache KV (Giai đoạn 17 · 11). Một yêu cầu đến nhóm giải mã có tiền tố của nó - nếu không khớp, nó sẽ điền trước → giải mã. Tỷ lệ truy cập và hợp chất phân tách — bộ định tuyến nhận biết bộ nhớ cache xác định xem có cần nạp trước mới hay không.

### MoE trên Blackwell là nơi có những con số thực

GB300 NVL72 + Dynamo hiển thị thông lượng MoE gấp 50 lần so với đường cơ sở của Phễu. Định tuyến chuyên gia MoE tính toán nặng về điền trước nhưng nặng bộ nhớ khi giải mã (bộ nhớ đệm chuyên gia), vì vậy phân tách là một chiến thắng kép. Phục vụ model biên giới năm 2026 chiếm ưu thế MoE (DeepSeek-V3, các biến thể GPT-5 trong tương lai).

### Những con số bạn nên nhớ

Benchmark số trôi dạt - NVIDIA và bài đăng inference stack cập nhật kết quả mỗi quý. Kiểm tra lại trước khi báo giá.

- DeepSeek-R1 trên GB200 NVL72 + Dynamo: thông lượng ~6x so với đường cơ sở trong chế độ độ trễ trung bình (developer.nvidia.com, 2025-06); tuyên bố cộng đồng "lên đến 30 lần" trên stacks Blackwell + Dynamo đầy đủ là tổng hợp định hướng mà không có một nguồn chính duy nhất.
- GB300 NVL72 + Dynamo: thông lượng MoE lên đến 50 lần so với Hopper (developer.nvidia.com, không ghi ngày).
- Neo tiết kiệm (tổng hợp nội bộ, không phải một nghiên cứu điển hình duy nhất): $600-800K/year off a $2M chi tiêu hàng năm ở SLA không đổi.
- Ngưỡng phân tách: prompts >512 tokens + đầu ra >200 tokens.
- Truyền KV qua NIXL: 20-80 ms cho 4K-prompt KV trên 70B FP8.

## Ứng dụng

`code/main.py` mô phỏng khẩu phần cùng vị trí so với phân tách. Báo cáo thông lượng, chi phí cho mỗi yêu cầu và bộ phân tần có độ dài prompt.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-disaggregation-decider.md`. Cho khối lượng công việc và cụm, quyết định có phân tách hay không.

## Bài tập

1. Chạy `code/main.py`. Sự phân tách đánh bại đồng vị trí ở độ prompt nào?
2. Thiết kế nhóm điền sẵn và nhóm giải mã cho dịch vụ RAG với độ dài tiền tố P99 8K, đầu ra 300.
3. Dynamo vs llm-d: chọn một cửa hàng cho một cửa hàng thuần Kubernetes không có sở thích Python runtime.
4. Chi phí truyền KV tính toán: Nạp trước 4K trên 70B FP8 = ~500 MB KV. Ở RDMA 100 GB/s, truyền = 5 ms. Ở TCP 10 GB/s = 50 ms. Điều gì quan trọng đối với SLA của bạn?
5. MoE định tuyến chuyên gia thay đổi các mẫu truy cập KV. Phân tách hoạt động như thế nào với MoE kích hoạt các chuyên gia khác nhau mỗi token?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Khẩu phần ăn tách rời | "tách prefill/decode" | Nhóm GPU riêng biệt cho từng giai đoạn |
| NIXL | "NVIDIA transport" | Truyền KV giữa các nút của Dynamo (RDMA/TCP) |
| NVIDIA Máy phát điện | "Người dàn nhạc" | Điều phối viên Stack trên cho vLLM/SGLang/TRT-LLM |
| llm-d | "Kubernetes bản địa" | Red Hat + AWS K8s được phân tách stack |
| Trình lập hồ sơ lập kế hoạch | "Tự động config máy phát điện" | Đo lường khối lượng công việc, cấu hình tỷ lệ nhóm |
| Công cụ lập kế hoạch SLA | "Máy phát điện policy" | Tự động khớp tỷ lệ điền trước: giải mã để đáp ứng SLO |
| `packDomain: rack` | "Cấu trúc liên kết llm-d" | Đóng gói nạp trước + giải mã trên cùng một giá đỡ cho KV nhanh |
| UCCL | "Tập thể thống nhất" | Lớp mạng llm-D 0.5 để mở rộng quy mô đến bằng không |
| Định tuyến chuyên gia MoE | "Chuyên gia mỗi token" | Mô hình DeepSeek-V3; Phân tách giúp |

## Đọc thêm

- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)
- [TensorRT-LLM Disaggregated Serving blog](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
- [llm-d GitHub](https://github.com/llm-d/llm-d)
- [llm-d 0.5 release notes](https://github.com/llm-d/llm-d/releases)
