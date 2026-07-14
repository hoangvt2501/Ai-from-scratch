# GPU Tự động thay đổi quy mô trên Kubernetes — Karpenter, KAI Scheduler, Gang Scheduling

> Ba lớp, không phải một. Karpenter cung cấp các nút động (dưới một phút, nhanh hơn 40% so với Cluster Autoscaler). KAI Scheduler xử lý lập lịch nhóm, nhận thức cấu trúc liên kết và hàng đợi phân cấp - nó ngăn chặn bẫy phân bổ từng phần 7/8 trong đó bảy nút chờ đợi và đốt trên một GPU bị thiếu. Bộ tự động thay đổi quy mô cấp ứng dụng (NVIDIA Dynamo Planner, llm-d Workload Variant Autoscaler) thay đổi quy mô trên các tín hiệu inference cụ thể - độ sâu hàng đợi, KV cache mức sử dụng - không phải CPU/DCGM chu kỳ làm việc. Bẫy HPA cổ điển là `DCGM_FI_DEV_GPU_UTIL` là phép đo chu kỳ nhiệm vụ: 100% có thể là 10 yêu cầu hoặc 100. vLLM phân bổ trước bộ nhớ KV cache, vì vậy bộ nhớ không bao giờ triggers giảm quy mô. Bài học này hướng dẫn bạn soạn ba lớp và tránh `WhenEmptyOrUnderutilized` policy Karpenter mặc định chấm dứt việc chạy các công việc GPU giữa inference.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, toy queue-depth autoscaler simulator)
**Kiến thức tiên quyết:** Giai đoạn 17 · 02 (Inference Kinh tế nền tảng), Giai đoạn 17 · 04 (vLLM phục vụ nội bộ)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Lập sơ đồ ba lớp tự động thay đổi quy mô (cung cấp nút, lập lịch nhóm, cấp ứng dụng) và đặt tên cho công cụ được sử dụng ở mỗi lớp.
- Giải thích lý do tại sao `DCGM_FI_DEV_GPU_UTIL` tín hiệu HPA sai cho vLLM và đặt tên cho hai tín hiệu thay thế (độ sâu hàng đợi, KV cache sử dụng).
- Mô tả lập lịch nhóm và chế độ lỗi phân bổ một phần mà KAI Scheduler ngăn chặn (7/8 GPUs không hoạt động).
- Đặt tên cho policy hợp nhất Karpenter (`WhenEmptyOrUnderutilized`) chấm dứt việc chạy các công việc GPU và nêu giải pháp thay thế an toàn năm 2026.

## Vấn đề

Nhóm của bạn ships một dịch vụ phục vụ LLM trên Kubernetes. Bạn thiết lập HPA với `DCGM_FI_DEV_GPU_UTIL` làm tín hiệu. Các chốt dịch vụ ở mức sử dụng 100% trong giờ làm việc. HPA không bao giờ mở rộng quy mô — nó đã nghĩ rằng bạn đã no. Bạn thêm một bản sao theo cách thủ công; TTFT giảm. HPA vẫn chưa mở rộng quy mô. Tín hiệu đang nói dối bạn.

Riêng biệt, bạn sử dụng Cluster Autoscaler cho các nút. Một token prompt 1 triệu đến lúc 2 giờ sáng; Cụm dành 3 phút để cung cấp một nút và yêu cầu hết thời gian chờ.

Một lần nữa, bạn triển khai một model 70B yêu cầu 8 GPUs trên 2 nút. Cụm có 7 GPUs miễn phí và 1 trải rộng trên 3 nút. Cluster Autoscaler cung cấp một nút cho 1 GPU bị thiếu. Bảy nút đợi 4 phút đốt tiền trong khi Kubernetes nhận được GPU cuối cùng.

Ba lớp, ba chế độ hỏng hóc khác nhau. Tự động thay đổi quy mô nhận biết GPU vào năm 2026 không phải là "bật HPA". Nó đang soạn cung cấp nút, lập lịch nhóm và tự động thay đổi quy mô tín hiệu ứng dụng.

## Khái niệm

### Lớp 1 — cung cấp nút (Karpenter)

Karpenter theo dõi các nhóm đang chờ xử lý và cung cấp các nút trong vòng ~45-60 giây (Cluster Autoscaler thường mất 90-120 giây cho các nút GPU). Nó chọn các loại phiên bản linh hoạt theo ràng buộc `NodePool` — nếu pod của bạn cần 8 H100 và cụm không có nút phù hợp, Karpenter sẽ cung cấp một loại trực tiếp thay vì thay đổi quy mô nhóm hiện có.

**Bẫy hợp nhất**: `consolidationPolicy: WhenEmptyOrUnderutilized` mặc định của Karpenter rất nguy hiểm đối với các nhóm GPU. Nó sẽ chấm dứt một nút GPU đang chạy để di chuyển các pod sang một phiên bản có kích thước phù hợp rẻ hơn. Đối với khối lượng công việc inference, điều đó có nghĩa là loại bỏ các yêu cầu đang chạy và tải lại 70B model trên nút mới. Loss là số phút dung lượng cộng với lỗi yêu cầu.

Cài đặt an toàn cho hồ bơi GPU:

```yaml
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h
```

Cho phép Karpenter hợp nhất các nút trống thực sự sau một giờ nhưng không bao giờ loại bỏ một công việc đang chạy.

### Lớp 2 - lập lịch băng nhóm (KAI Scheduler)

KAI Scheduler (dự án "Karp" sau đó được đổi tên) xử lý những gì kube-scheduler mặc định không:

**Lên lịch băng đảng **- lên lịch tất cả hoặc không có gì. Một nhóm inference phân tán yêu cầu 8 GPUs tất cả 8 bắt đầu cùng nhau hoặc không bắt đầu. Nếu không có điều này, bạn sẽ nhận được bẫy phân bổ một phần: 7 trong số 8 nhóm bắt đầu, đợi vô thời hạn, đốt tiền.

**Nhận thức về cấu trúc liên kết** — biết GPUs nào chia sẻ NVLink,  nào nằm trên cùng một giá đỡ, có InfiniBand giữa chúng. Đặt vỏ cho phù hợp. Khối lượng công việc song song tensor DeepSeek-V3 67B phải nằm trên một miền NVLink; KAI Scheduler tôn trọng điều đó.

**Hàng đợi phân cấp** — nhiều đội cạnh tranh cho cùng một nhóm GPU với mức độ ưu tiên và hạn ngạch. Cú nhúm production của Đội A chỉ bị công việc training của Đội B chiếm ưu tiên nếu các quy tắc ưu tiên cho phép.

KAI được triển khai cùng với kube-scheduler như một bộ lập lịch phụ; Bạn chú thích khối lượng công việc để sử dụng nó. Ray và vLLM production-stack đều tích hợp.

### Lớp 3 — tín hiệu cấp ứng dụng

**Bẫy HPA**: `DCGM_FI_DEV_GPU_UTIL` là số liệu chu kỳ nhiệm vụ - nó đo lường liệu GPU có đang làm việc ở mỗi khoảng thời gian sampling hay không. Sử dụng 100% có thể có nghĩa là 10 yêu cầu đồng thời hoặc 100; GPU đều bận rộn. Mở rộng quy mô theo chu kỳ nhiệm vụ đang mở rộng quy mô một cách mù quáng.

Tệ hơn, vLLM và các công cụ tương tự phân bổ trước bộ nhớ KV cache (tối đa `--gpu-memory-utilization`). Mức sử dụng bộ nhớ vẫn gần 90% ngay cả khi có một yêu cầu. HPA dựa trên bộ nhớ không bao giờ giảm quy mô.

**Tín hiệu thay thế năm 2026**:

- Độ sâu hàng đợi (số lượng yêu cầu đang chờ điền trước).
- KV cache sử dụng (phần nào của khối được phân bổ cho các trình tự đang hoạt động).
- Mỗi bản sao P99 TTFT (tín hiệu SLA của bạn).
- Goodput (yêu cầu đáp ứng tất cả SLO mỗi giây).

NVIDIA Dynamo Planner và Autoscaler biến thể khối lượng công việc llm-d sử dụng các tín hiệu này và mở rộng các bản sao. Chúng thay thế hoàn toàn HPA để phục vụ LLM.

### Khi nào sử dụng những gì

| Quyết định quy mô | Công cụ |
|----------------|------|
| Add/remove nút | Karpenter |
| Lên lịch công việc nhiều GPU | Bộ lập lịch KAI |
| Add/remove bản sao | Dynamo Planner / llm-d WVA (hoặc HPA tùy chỉnh trên độ sâu hàng đợi) |
| Chọn loại GPU | Karpenter NodePool |
| Ưu tiên mức độ ưu tiên thấp | Hàng đợi KAI Scheduler |

### Phân tách prefill/decode làm phức tạp mọi thứ

Nếu bạn chạy prefill/decode phân tách (Giai đoạn 17 · 17), bạn có hai classes pod với triggers tỷ lệ khác nhau: điền trước tỷ lệ vỏ theo độ sâu hàng đợi, giải mã tỷ lệ vỏ theo áp suất KV cache. llm-d cho thấy chúng dưới dạng `Services` riêng biệt với HPA cho mỗi vai trò. Đừng cố gắng đặt một HPA duy nhất trước cả hai.

### Cold start cũng quan trọng ở đây

Giảm thiểu khởi động nguội (Giai đoạn 17 · 10) là nơi thời gian cung cấp nút trở nên hiển thị cho người dùng. Khởi động 45-60 giây của Karpenter cộng với tải model 20GB cộng với khởi động động cơ có nghĩa là yêu cầu từ số không mất 2-5 phút. Giữ một hồ bơi ấm (`min_workers=1`) cho các đường dẫn quan trọng của SLO hoặc sử dụng điểm kiểm tra kiểu Modal ở lớp ứng dụng.

### Những con số bạn nên nhớ

- Cung cấp nút Karpenter: ~45-60 giây so với Cluster Autoscaler ~90-120 giây (GPU nút).
- KAI Scheduler ngăn chặn lãng phí phân bổ một phần - bẫy 7/8.
- `DCGM_FI_DEV_GPU_UTIL` là tín hiệu HPA: bị hỏng; sử dụng độ sâu hàng đợi hoặc sử dụng KV.
- Karpenter `WhenEmptyOrUnderutilized`: chấm dứt các công việc GPU đang chạy. Sử dụng `WhenEmpty + consolidateAfter: 1h` để inference.

```figure
autoscaling
```

## Ứng dụng

`code/main.py` mô phỏng bộ tự động chia tỷ lệ ba lớp trên khối lượng công việc GPU tăng vọt. So sánh HPA (chu kỳ nhiệm vụ) ngây thơ, HPA độ sâu hàng đợi và tỷ lệ theo lịch trình của băng đảng KAI. Báo cáo các yêu cầu chưa được đáp ứng, số phút GPU nhàn rỗi và điểm tổng hợp.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-gpu-autoscaler-plan.md`. Với cấu trúc liên kết cụm, hình dạng khối lượng công việc và SLO, nó thiết kế một kế hoạch tự động thay đổi quy mô ba lớp.

## Bài tập

1. Chạy `code/main.py`. Trong khối lượng công việc tăng vọt, HPA chu kỳ nhiệm vụ ngây thơ làm giảm bao nhiêu yêu cầu mà HPA nắm bắt được độ sâu hàng đợi? Sự khác biệt đến từ đâu?
2. Thiết kế Karpenter NodePool cho một cụm phục vụ Llama 3.3 70B FP8 trên H100 SXM5. Chỉ định `capacity-type`, `disruption.consolidationPolicy`, `consolidateAfter` và nội dung để giữ khối lượng công việc không GPU khỏi các nút này.
3. Nhóm của bạn báo cáo rằng việc triển khai bị kẹt trong Đang chờ xử lý vì "GPUs khả dụng nhưng nhóm sẽ không lên lịch". Chẩn đoán - đây là Karpenter, kube-scheduler hay KAI Scheduler? Số liệu nào xác nhận?
4. Chọn một tín hiệu để tự động chia tỷ lệ các nhóm nạp trước được phân tách và một tín hiệu khác cho các nhóm giải mã. Biện minh cho cả hai.
5. Tính toán chi phí của bẫy hợp nhất `WhenEmptyOrUnderutilized` trên dịch vụ production 24x7 trung bình 60 events/day thả yêu cầu ở P99 TTFT > 10 giây.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Karpenter | "Trình cung cấp nút" | Kubernetes nút tự động mở rộng; Cung cấp dưới phút |
| Cluster Autoscaler | "Máy cạo cũ" | Kubernetes người tiền nhiệm của trình tự động mở rộng nút; chậm hơn, dựa trên nhóm |
| Bộ lập lịch KAI | "Bộ lập lịch GPU" | Bộ lập lịch phụ cho băng nhóm + cấu trúc liên kết + hàng đợi |
| Lập lịch băng đảng | "Tất cả hoặc không có gì" | Lên lịch N pod nguyên tử hoặc trì hoãn tất cả chúng |
| Nhận thức về cấu trúc liên kết | "nhận biết giá đỡ" | Đặt vỏ dựa trên vị trí NVLink/IB/rack |
| `DCGM_FI_DEV_GPU_UTIL` | "Sử dụng GPU" | Số liệu chu kỳ nhiệm vụ; KHÔNG phải là tín hiệu chia tỷ lệ cho LLMs |
| Độ sâu hàng đợi | "Yêu cầu chờ đợi" | Tín hiệu HPA chính xác để mở rộng quy mô liên kết nạp trước |
| Sử dụng KV cache | "Áp lực bộ nhớ" | Tín hiệu HPA chính xác để mở rộng quy mô liên kết giải mã |
| Hợp nhất | "Hợp nhất Karpenter" | Chấm dứt nút sang loại phiên bản rẻ hơn |
| `WhenEmpty + 1h` | "Hợp nhất an toàn" | Policy điều đó không loại bỏ việc chạy GPU công việc |

## Đọc thêm

- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler) - thiết kế tài liệu và configuration ví dụ.
- [Karpenter Disruption Controls](https://karpenter.sh/docs/concepts/disruption/) — hợp nhất policy ngữ nghĩa và mặc định an toàn GPU.
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) - Tín hiệu chia tỷ lệ Dynamo Planner.
- [Ray docs — KAI Scheduler for RayClusters](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kai-scheduler.html) - Mẫu tích hợp tia.
- [AWS EKS Compute and Autoscaling Best Practices](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html) — hướng dẫn dành riêng cho Kubernetes được quản lý.
- [llm-d GitHub](https://github.com/llm-d/llm-d) - Thiết kế Autoscaler biến thể khối lượng công việc.
