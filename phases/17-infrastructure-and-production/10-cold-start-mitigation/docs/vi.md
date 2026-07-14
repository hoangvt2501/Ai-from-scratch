# Cold Start Giảm thiểu cho LLMs phi máy chủ

> Hình ảnh model 20 GB mất 5-10 phút (7B) đến 20+ phút (70B) để chuyển từ lạnh sang phục vụ. Trong một thế giới serverless thực sự, đó không phải là khởi động — đó là sự ngừng hoạt động. Các biện pháp giảm thiểu hoạt động ở năm lớp: hình ảnh nút được chèn sẵn (Bottlerocket trên AWS, vòm volume kép), model streaming (NVIDIA Run:ai Model Streamer, gốc trong vLLM), GPU ảnh chụp nhanh bộ nhớ (Modal checkpoints, khởi động lại nhanh hơn tới 10 lần), nhóm ấm (`min_workers=1`), tải theo tầng (NVMe→DRAM→HBM pipeline của ServerlessLLM, giảm độ trễ 10-200 lần) và di chuyển trực tiếp di chuyển tokens đầu vào (KB) thay vì KV cache (GB). Modal xuất bản khởi động nguội 2-4 giây dưới dạng sàn; Mặc định 5-10 giây, dưới giây với tính năng làm ấm trước. Bài học này dạy bạn đo lường, lập ngân sách và stack năm lớp.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, toy cold-start path simulator)
**Kiến thức tiên quyết:** Giai đoạn 17 · 02 (Inference Kinh tế nền tảng), Giai đoạn 17 · 03 (GPU Tự động chia tỷ lệ)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Liệt kê năm lớp giảm thiểu khởi động nguội và đặt tên cho một công cụ hoặc mẫu ở mỗi lớp.
- Tính tổng thời gian khởi động nguội dưới dạng tổng (cung cấp nút) + (tải xuống trọng số) + (trọng số tải vào HBM) + (khởi động động cơ) cho model 70B.
- Giải thích lý do chuyển dữ liệu di chuyển trực tiếp tokens đầu vào (KB) không phải KV cache (GB) và hình phạt (tính toán lại) là bao nhiêu.
- Đặt tên cho sự đánh đổi bể bơi ấm (trả tiền cho GPU nhàn rỗi hoặc chấp nhận đuôi khởi động nguội) và ngưỡng SLA mà tại đó `min_workers > 0` trở thành bắt buộc.

## Vấn đề

LLM endpoint serverless của bạn thay đổi quy mô về không chỉ sau một đêm. Vào lúc 8 giờ sáng, giao thông tăng đột biến. Yêu cầu đầu tiên chờ trong khi:

1. Karpenter cung cấp một nút GPU: 45-60s.
2. container kéo hình ảnh 30 GB với trọng lượng: 120-300 giây.
3. Động cơ tải trọng lượng vào HBM: 45-120 giây tùy thuộc vào kích thước model và tốc độ lưu trữ.
4. vLLM hoặc TRT-LLM khởi tạo CUDA đồ thị, nhóm KV cache, tokenizer: 10-30 giây.

Tổng cộng: 220-510 giây (khoảng 3-8 phút) trước khi một token quay trở lại. SLA của bạn là 2 giây. Bạn ship một hồ bơi ấm (`min_workers=1`) và vấn đề dường như biến mất - nhưng bây giờ bạn phải trả tiền cho một GPU nhàn rỗi 24x7. Nếu dịch vụ của bạn có 5 sản phẩm, mỗi sản phẩm có một bản sao ấm, thì đó là 5 × 24 × 30 = 3.600 GPU-hours/month cho dù có một người dùng nào được gọi hay không.

Giảm thiểu khởi động nguội là cách duy trì nền kinh tế phi máy chủ trong khi ước tính độ trễ của tính năng luôn bật.

## Khái niệm

### Lớp 1 - hình ảnh nút được chèn sẵn (Bottlerocket)

Trên AWS, kiến trúc volume kép của Bottlerocket tách hệ điều hành khỏi dữ liệu. Chụp nhanh volume dữ liệu với hình ảnh container của bạn được kéo trước; tham chiếu mã ảnh chụp nhanh trong `EC2NodeClass` của bạn. Các nút mới khởi động với trọng số đã có trên NVMe cục bộ — bước 2 và một phần của 3 biến mất. Làm việc với Karpenter nguyên bản. Tiết kiệm điển hình: 2-4 phút mỗi cold start cho các models lớn.

Tương đương trên GCP: hình ảnh máy ảo tùy chỉnh với các lớp container được nướng sẵn. Trên Azure: ảnh chụp nhanh ổ đĩa được quản lý với cùng một mẫu.

### Lớp 2 - model streaming (Chạy: ai Model Streamer)

Thay vì tải toàn bộ tệp trước khi trả lời yêu cầu đầu tiên, luồng sẽ tập trung vào bộ nhớ GPU từng lớp và bắt đầu xử lý ngay khi khối transformer đầu tiên cư trú. Streamer NVIDIA Run:ai Model ships bản địa trong vLLM 2026. Hoạt động với S3, GCS và NVMe cục bộ. Giảm khoảng một nửa thời gian tải trọng lượng cho models lớn bằng cách chồng chéo I/O với thiết lập điện toán.

### Lớp 3 - GPU ảnh chụp nhanh bộ nhớ (Phương thức)

Phương thức lấy checkpoint trạng thái GPU (trọng số, đồ thị CUDA KV cache vùng) sau lần tải đầu tiên. Các lần khởi động lại tiếp theo sẽ giải tuần tự hóa trực tiếp vào HBM - nhanh hơn 10 lần so với khởi tạo lại. Đây là điều gần nhất để "khởi động một GPU ấm trong 2 giây". Đánh đổi: ảnh chụp nhanh là cấu trúc liên kết trên mỗi GPU, vì vậy nếu Karpenter di chuyển bạn sang một SKU khác, bạn sẽ checkpoint lại.

### Lớp 4 - hồ bơi ấm (min_workers = 1)

Giảm thiểu đơn giản nhất: luôn sẵn sàng cho một bản sao. Chi phí là một GPUcủa tỷ lệ hàng giờ 24x7. Số học là tàn bạo trên nhỏ models (bạn trả tiền $0.85-$ 1.50/hr để tránh 30 giây cold start) và tử tế đến lớn (trả $4/hr để tránh 5 phút cold start). Ngưỡng SLA nơi các hồ bơi ấm trở thành bắt buộc: thường là TTFT P99 < 60 giây trên 70B + model.

### Lớp 5 — tải theo bậc (ServerlessLLM)

ServerlessLLM coi lưu trữ như một hệ thống phân cấp: NVMe (nhanh nhưng lớn), DRAM (trung bình nhưng có tầng), HBM (nhỏ nhưng tức thì). Trọng lượng được tải sẵn vào DRAM; tải theo yêu cầu vào HBM. Giấy báo cáo giảm độ trễ 10-200 lần khi tải lạnh so với đĩa thành HBM ngây thơ. Production áp dụng còn sớm nhưng vẫn có tích hợp với vLLM.

### Lớp 6 — di chuyển trực tiếp (mẫu thưởng)

Khi một nút trở nên không khả dụng (loại bỏ tại chỗ, thoát nút), mẫu truyền thống là khởi động nguội một bản sao khác và hàng đợi yêu cầu thoát nước. Di chuyển trực tiếp di chuyển tokens đầu vào (kilobyte) đến đích đã tải model và tính toán lại KV cache trên đích. Tính toán lại rẻ hơn so với truyền GB KV cache qua mạng. Áp dụng cho các triển khai phân tách.

### Toán học bể bơi ấm

Đối với một dịch vụ có P99 TTFT SLA là 2 giây, câu hỏi không phải là "yes/no hồ bơi ấm" mà là "có bao nhiêu bản sao ấm áp và con đường nào có được chúng".

- Đường dẫn tương tác có giá trị cao (trò chuyện trực tiếp, agent thoại): `min_workers=1-2`.
- Đường dẫn batch nền (phân loại hàng đêm): chấp nhận tỷ lệ đến không, 5-10 phút cold start có thể chấp nhận được.
- Bậc cao cấp: `min_workers` mỗi tenant với dung lượng chuyên dụng.

### Đo lường trước khi tối ưu hóa

Giải phẫu khởi động nguội cho model 70B trên một nút mới (minh họa):

| Giai đoạn | Thời gian | Giảm thiểu |
|-------|------|-----------|
| Cung cấp nút | Thập niên 50 | Tên lửa chai + hình ảnh hạt sẵn, hồ bơi ấm áp |
| Kéo hình ảnh | Thập niên 180 | volume dữ liệu hạt giống trước (loại bỏ) |
| Trọng lượng theo HBM | 75 giây | Model streamer (giảm một nửa); GPU ảnh chụp nhanh (loại bỏ) |
| Động cơ khởi động | Tuổi 20 | Bộ nhớ đệm biểu đồ CUDA liên tục |
| Tiền đạo đầu tiên | 3 giây | Độ trễ vốn có tối thiểu |
| **Lạnh hoàn toàn** | **328 giây** ||
| **Tổng số với các biện pháp giảm thiểu** | **~15 giây **| Giảm 22 lần |

### Những con số bạn nên nhớ

- cold start phương thức: 2-4 giây (với GPU ảnh chụp nhanh).
- Cơ sở cold start mặc định: 5-10 giây; dưới giây với sự hâm nóng trước.
- Raw 70B cold start: 3-8 phút.
- Chạy: ai Model Streamer: ~2x tăng tốc độ tải trọng lượng.
- Tải theo cấp ServerlessLLM: Giảm độ trễ 10-200 lần (số giấy).

## Ứng dụng

`code/main.py` models một lộ trình khởi động nguội có và không có mỗi biện pháp giảm thiểu. Báo cáo tổng thời gian khởi động nguội, chi phí bể bơi ấm và tỷ lệ yêu cầu hòa vốn mà trên đó bể nước ấm tự chi trả.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-cold-start-planner.md`. Với SLA, quy mô model và hình dạng lưu lượng truy cập, hãy chọn những biện pháp giảm thiểu nào để stack.

## Bài tập

1. Chạy `code/main.py`. Tính toán tỷ lệ yêu cầu hòa vốn mà trên đó một bản sao ấm sẽ rẻ hơn so với việc trả thuế khởi động nguội thông qua các yêu cầu giảm bổ sung tại SLO.
2. Bạn triển khai model 13B với SLA TTFT P99 là 3s. Chọn stack giảm thiểu tối thiểu (ít lớp nhất) đạt được điều đó.
3. Hạt trước tên lửa chai loại bỏ lực kéo hình ảnh nhưng trọng lượng vẫn tải từ ảnh chụp nhanh đến HBM. Tính toán đồng hồ treo tường cho model 70B nếu NVMe được hỗ trợ bởi ảnh chụp nhanh đọc ở 7 GB/s.
4. Nhà cung cấp phi máy chủ của bạn cung cấp GPU bản kết xuất nhanh (Modal) và nhóm của bạn từ chối vì "bản kết xuất nhanh làm rò rỉ PII". Tranh luận cả hai bên - rủi ro thực tế là gì và giảm thiểu là gì (ảnh chụp nhanh tạm thời, mã hóa, cách ly không gian tên)?
5. Thiết kế policy bể bơi ấm theo tầng: có bao nhiêu bản sao ấm cho người dùng trả phí, người dùng dùng thử và khối lượng công việc batch? Hiển thị toán học.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Cold start | "Tạm dừng lớn" | Thời gian từ khi yêu cầu đến token đầu tiên trên một bản sao mới |
| Hồ bơi ấm áp | "tối thiểu luôn bật" | `min_workers >= 1` để chuẩn bị sẵn ít nhất một bản sao |
| Hình ảnh được chèn sẵn | "AMI nướng" | Hình ảnh nút có trọng số container trước |
| Tên lửa chai | "Hệ điều hành nút AWS" | Hệ điều hành được tối ưu hóa cho AWS container với hỗ trợ kết xuất nhanh volume kép |
| Model streamer | "Tải streaming" | Chồng chéo trọng số I/O với thiết lập điện toán |
| GPU ảnh chụp nhanh | "checkpoint đến HBM" | Tuần tự hóa trạng thái GPU sau tải; deserialize khi khởi động lại |
| Tải theo tầng | "NVMe + DRAM + HBM" | Hệ thống phân cấp các tầng lưu trữ; Tải theo yêu cầu |
| Di chuyển trực tiếp | "Di chuyển tokens" | Chuyển đầu vào (KB), tính toán lại KV trên đích |
| `min_workers` | "Bản sao ấm áp" | Số lượng duy trì hoạt động tối thiểu phi máy chủ |
| Mở rộng quy mô về không | "hoàn toàn phi máy chủ" | Không mất phí khi nhàn rỗi; Chấp nhận thuế khởi động nguội đầy đủ |

## Đọc thêm

- [Modal — Cold start performance](https://modal.com/docs/guide/cold-start) — Kiến trúc benchmarks và checkpoint đã xuất bản của Modal.
- [AWS Bottlerocket](https://github.com/bottlerocket-os/bottlerocket) — dữ liệu được tạo sẵn volume mẫu ảnh chụp nhanh.
- [NVIDIA Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer) - tải trọng số chồng chéo với thiết lập điện toán.
- [Baseten — Cold-start mitigation](https://www.baseten.co/blog/cold-start-mitigation/) - sách hướng dẫn trước khi làm ấm.
- [ServerlessLLM paper (USENIX OSDI'24)](https://www.usenix.org/conference/osdi24/presentation/fu) - thiết kế tải theo tầng.
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — di chuyển trực tiếp cho các triển khai phân tách.
