# Lưu lượng truy cập bóng, Canary Rollout và triển khai lũy tiến cho LLMs

> LLM rollouts kết hợp các phần khó nhất của việc triển khai phần mềm: không kiểm tra đơn vị, chế độ lỗi khuếch tán, tín hiệu trễ. Trình tự là (1) chế độ bóng - trùng lặp các yêu cầu sản xuất để ứng cử viên model, ghi nhật ký, so sánh với tác động của người dùng bằng không; nắm bắt các vấn đề phân phối rõ ràng nhưng không đảm bảo chất lượng; (2) canary rollout - chuyển giao thông lũy tiến 10% → 25% → 50% → 75% → 100% với cổng ở mỗi bước; theo dõi phần trăm độ trễ, tỷ lệ cost/request, error/refusal, phân bố độ dài đầu ra, tỷ lệ phản hồi của người dùng; (3) A/B thử nghiệm cho các lựa chọn thay thế riêng biệt sau khi xác nhận độ ổn định. Tính không xác định là không thể rút gọn - lên đến 15% accuracy biến đổi giữa các lần chạy với các đầu vào giống hệt nhau do GPU không liên kết FP cộng với variance kích thước batch. Chi phí là một biến số, không phải hằng số - model tốt hơn 20% có thể đắt hơn gấp 3 lần cho mỗi cuộc gọi. Rollback tốc độ là quyết định: nếu rollback yêu cầu triển khai lại, bạn quá chậm. Policy sống ở config/flags; model sống ở registry với các tiêu hóa được ghim; rollback = lật policy + hoàn nguyên ngưỡng + ghim model cũ trong vài giây.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, toy canary-progression simulator)
**Kiến thức tiên quyết:** Giai đoạn 17 · 13 (Observability), Giai đoạn 17 · 21 (Thử nghiệm A/B)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Phân biệt chế độ bóng (so sánh không tác động), canary (lưu lượng truy cập trực tiếp lũy tiến) và A/B (so sánh được xác nhận độ ổn định).
- Liệt kê năm chỉ số canary dành riêng cho LLM (độ trễ, cost/request, error/refusal, phân phối độ dài đầu ra, phản hồi của người dùng).
- Giải thích lý do tại sao tính không quyết định (lên đến 15%) LLM thay đổi ý nghĩa của "ổn định" trong một rollout.
- Thiết kế đường dẫn rollback mất vài giây (policy lật) chứ không phải vài giờ (triển khai lại).

## Vấn đề

Bạn ship một model mới. Đánh giá ngoại tuyến cho thấy mức tăng accuracy 3%. Bạn bật nó lên trong production. Trong vòng 24 giờ, chi phí tăng 40%, ngón tay cái của người dùng tăng 8%, ba phiếu yêu cầu của khách hàng báo cáo "câu trả lời kỳ lạ". Bạn quay trở lại. Quá trình triển khai lại mất 3 giờ. Cuối tuần của bạn bị hủy hoại.

Mọi phần của điều đó đều có thể tránh được. Chế độ Shadow sẽ bắt được mức chi phí tăng đột biến 40% trước khi bất kỳ người dùng nào nhìn thấy nó. Canary sẽ dừng lại ở mức 10% khi ngón tay cái di chuyển. rollback cờ Policy sẽ mất 30 giây. Kỷ luật là thứ lấp đầy khoảng trống giữa "đánh giá ngoại tuyến trông đẹp" và "người dùng thực sự hạnh phúc".

## Khái niệm

### Chế độ bóng

Ứng viên nhận được các yêu cầu tương tự như production; đầu ra được ghi lại, không trả lại cho người dùng. Không ảnh hưởng đến người dùng. Nhật ký:

- Nội dung đầu ra (chênh lệch so với production).
- Số lượng Token (delta chi phí).
- Độ trễ.
- Từ chối và sai sót.

Bắt: tăng chi phí, hồi quy độ dài, thay đổi từ chối rõ ràng, lỗi khó. KHÔNG bắt: người dùng delta chất lượng sẽ cảm nhận. Shadow là một bài kiểm tra khói, không phải một bài kiểm tra chất lượng.

### Canary rollout

Chuyển đổi giao thông lũy tiến với cổng. Tiến triển điển hình: 1% → 10% → 25% → 50% → 75% → 100%. Cổng trên 5 chỉ số ở mỗi bước:

1. **Phần trăm độ trễ **- P50, P95, P99. Vi phạm: canary có P99 > đường cơ sở gấp 1.5 lần.
2. **Chi phí cho mỗi yêu cầu** — kết hợp $. Vi phạm: >20% so với đường cơ sở.
3. **Tỷ lệ lỗi / từ chối** - 5xx cộng với từ chối rõ ràng. Vi phạm: 2x đường cơ sở.
4. **Phân bố chiều dài đầu ra** - trung bình + P99. Vi phạm: sự thay đổi phân phối.
5. **Tỷ lệ phản hồi của người dùng** — ngón tay cái xuống / nộp phiếu. Vi phạm: 1.5x đường cơ sở.

### Thuyết không quyết định là variance mới

Đầu vào giống hệt nhau tạo ra đầu ra không giống hệt nhau. Lý do:

- GPU FP không liên kết (thứ tự giảm dấu phẩy động thay đổi theo batch).
- variance cỡ Batch (cùng prompt trong batch 128 so với batch 16).
- Sampling (temperature > 0).

Đo lường: lên đến 15% accuracy biến thể chạy trên các bộ đánh giá giống hệt nhau. "Ổn định" trong rollout có nghĩa là các chỉ số nằm trong variance dự kiến, không giống với đường cơ sở. Đặt cổng phía trên sàn nhiễu.

### Chi phí là một biến số

Một model tốt hơn 20% có thể đắt hơn gấp 3 lần cho mỗi cuộc gọi. Cost/request là một trong năm cổng. Shipping một model "tốt hơn" phá vỡ kinh tế đơn vị là một trường hợp rollback.

### Rollback là vũ khí

- Cờ Policy (hệ thống cờ feature): tỷ lệ phần trăm lật trong config; mất vài giây.
- Ghim Model (registry thông báo): model được ghim không tự động nâng cấp.
- Rollback = hoàn nguyên cờ + đặt thông báo đã ghim về trước đó. Giây, không phải giờ.

Nếu stack của bạn yêu cầu triển khai lại cho rollback, hãy khắc phục điều đó trước khi triển khai.

### Dụng cụ

**Argo Rollouts** / **Flagger** — Kubernetes bộ điều khiển phân phối lũy tiến. Tích hợp với Istio/Linkerd định tuyến có trọng số.

**Định tuyến trọng số Istio** — phân chia lưu lượng truy cập cấp lưới dịch vụ.

**KServe / Seldon Core** - model phục vụ với canary tích hợp.

**Feature cờ** — LaunchDarkly, Flagsmith, Unleash. Lật cấp Policy, không cần triển khai lại.

### Nhịp số liệu

Canary cổng kiểm tra 5-15 phút một lần tùy thuộc vào volume giao thông. Lưu lượng truy cập 1% với 10 req/min cung cấp 50-150 điểm dữ liệu mỗi cửa sổ - đủ cho độ trễ nhưng nhiễu cho phản hồi của người dùng. 10% cho ~10 lần nhiều hơn. Tiến trình nên tạm dừng đủ lâu để tích lũy đủ mẫu ở mỗi bước.

### Bước A/B là tùy chọn

Nếu model mới khác biệt rõ rệt (hành vi khác nhau, đường cong chi phí khác nhau, giai điệu khác nhau), A/B kiểm tra nó ở mức 50% sau khi canary vượt qua. Nếu chỉ là phiên bản cải tiến, hãy bỏ qua 100% khi canary cổng đi qua.

### Những con số bạn nên nhớ

- Tiến triển Canary: 1% → 10% → 25% → 50% → 75% → 100%.
- Trần không xác định: lên đến 15% variance chạy trên các đầu vào giống hệt nhau.
- Năm chỉ số canary: độ trễ, chi phí, error/refusal, độ dài đầu ra, phản hồi của người dùng.
- Cổng chi phí: >20% so với mức cơ sở là vi phạm.
- Rollback: giây, không phải giờ.

## Ứng dụng

`code/main.py` mô phỏng một canary rollout với hồi quy được tiêm. Báo cáo giai đoạn nào rollout dừng và cổng nào được kích hoạt.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-rollout-runbook.md`. Với model ứng viên, đường cơ sở và khả năng chấp nhận rủi ro, thiết kế kế hoạch →canary→100% bóng.

## Bài tập

1. Chạy `code/main.py`. Tiêm hồi quy chi phí 25%. canary dừng ở giai đoạn nào?
2. model mới của bạn có 3% accuracy tăng ngoại tuyến nhưng cost/request là +18%. Nó có phải là một ship? Phụ thuộc vào policy - viết cả hai đường dẫn.
3. Thiết kế một rollback mất dưới 60 giây từ đầu đến cuối. Liệt kê cơ sở hạ tầng cần thiết.
4. Tính không xác định hiển thị ±7% trên đánh giá của bạn. Đặt cổng canary để bạn không báo động sai. Bạn sử dụng hệ số nhân nào?
5. Chế độ Shadow bắt được mức tăng đột biến 40% chi phí trước khi canary. Viết quy tắc cảnh báo kích hoạt trong bóng tối.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Chế độ bóng | "trùng lặp với mới" | Gửi đến ứng viên không tác động để ghi nhật ký |
| Canary | "Lưu lượng truy cập lũy tiến" | Dần dần người dùng tiếp xúc với rollout có cổng |
| Cổng | "Kiểm tra rollout" | Ngưỡng chỉ số chặn tiến trình |
| Thuyết không xác định | "LLM variance" | Sự khác biệt không thể rút gọn giữa các lần chạy |
| Cờ Policy | "Lật cờ rollback" | rollback cấp Config, giây không phải giờ |
| Model pin | "registry tiêu hóa" | Tham chiếu bất biến cho phiên bản model |
| Argo Rollouts | "K8s tiến bộ" | Bộ điều khiển canary/rollback gốc Kubernetes |
| KServe | "inference K8s" | Model ăn kèm với canary primitives |
| Istio có trọng số | "Tách lưới" | Bộ chia lưu lượng lưới dịch vụ |

## Đọc thêm

- [TianPan — Releasing AI Features Without Breaking Production](https://tianpan.co/blog/2026-04-09-llm-gradual-rollout-shadow-canary-ab-testing)
- [MarkTechPost — Safely Deploying ML Models](https://www.marktechpost.com/2026/03/21/safely-deploying-ml-models-to-production-four-controlled-strategies-a-b-canary-interleaved-shadow-testing/)
- [APXML — Advanced LLM Deployment Patterns](https://apxml.com/courses/mlops-for-large-models-llmops/chapter-4-llm-deployment-serving-optimization/advanced-llm-deployment-patterns)
- [Argo Rollouts docs](https://argo-rollouts.readthedocs.io/)
- [Flagger docs](https://docs.flagger.app/)
