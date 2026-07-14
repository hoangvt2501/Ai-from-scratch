# SRE for AI — Ứng phó sự cố đa Agent, Runbook, Phát hiện dự đoán

> AI SRE sử dụng LLMs dựa trên dữ liệu cơ sở hạ tầng (nhật ký, sổ chạy, cấu trúc liên kết dịch vụ) thông qua RAG để tự động hóa các giai đoạn điều tra, tài liệu và điều phối. Mô hình kiến trúc năm 2026 là đa agent orchestration - agents chuyên biệt (nhật ký, số liệu, sổ chạy) được điều phối bởi người giám sát; AI đề xuất các giả thuyết và truy vấn, con người chấp thuận các cuộc gọi phán đoán. Datadog Bits AI và Azure SRE Agent ship điều này dưới dạng sản phẩm được quản lý. Runbook đang phát triển: NeuBird Hawkeye sử dụng đánh giá đối nghịch (hai models phân tích cùng một sự cố; thỏa thuận = tin tưởng, bất đồng = không chắc chắn); bộ nhớ hoạt động vẫn tồn tại qua các thay đổi của nhóm. Tự động khắc phục vẫn thận trọng: AI gợi ý, con người chấp thuận. Hành động hoàn toàn tự động là hẹp (khởi động lại pod, rollback triển khai cụ thể) với các guardrails chặt chẽ - bất kỳ ai bán "đặt nó và quên nó" là bán quá mức. Biên giới mới nổi: dự đoán trước sự cố. Nghiên cứu của MIT báo cáo một LLM được huấn luyện về nhật ký lịch sử + nhiệt độ GPU + các mẫu lỗi API đã dự đoán 89% sự cố mất điện sớm 10-15 phút. Dự báo: 95% LLMs doanh nghiệp đã tự động hóa failover vào cuối năm 2026.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, toy multi-agent incident triage simulator)
**Kiến thức tiên quyết:** Giai đoạn 17 · 13 (Observability), Giai đoạn 17 · 24 (Kỹ thuật hỗn loạn)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Sơ đồ kiến trúc SRE đa agent AI: giám sát + agents chuyên biệt (nhật ký, số liệu, sổ chạy) + cổng phê duyệt của con người.
- Giải thích lý do tại sao tự động khắc phục hẹp (khởi động lại pod, hoàn nguyên triển khai) thay vì rộng (dịch vụ kiến trúc lại).
- Đặt tên cho mô hình đánh giá đối nghịch (NeuBird Hawkeye): hai models đồng ý = tin cậy; không đồng ý = leo thang.
- Trích dẫn kết quả phát hiện sớm 89% của MIT và hạn chế hoạt động: dự đoán không có kích hoạt chỉ là bảng điều khiển.

## Vấn đề

Một kỹ sư trực được nhắn tin vào lúc 3 giờ sáng "Tỷ lệ lỗi cao khi thanh toán". Họ kiểm tra Datadog, Loki, ba runbook, nhật ký triển khai. 30 phút sau, họ nhận ra nguyên nhân gốc rễ là do oom vLLM do KV cache tăng đột biến. Họ khởi động lại vỏ; Lỗi sẽ bị xóa.

Vào năm 2026, 20 phút đầu tiên của cuộc điều tra đó là tự động. Nhóm nhật ký theo dịch vụ, tương quan với các triển khai gần đây, khớp với runbook - tất cả đều RAG + sử dụng công cụ. Một agent được giám sát có thể thực hiện phân loại lần đầu tiên và đưa ra giả thuyết trước khi con người mở Datadog.

Khắc phục hoàn toàn tự động là một vấn đề khác. Khởi động lại pod: an toàn. Quy mô GPU hồ bơi: an toàn nếu policy cho phép. Kiến trúc lại dịch vụ: hoàn toàn không. Kỷ luật đang vạch ra ranh giới hẹp.

## Khái niệm

### Kiến trúc đa agent

```
          Incident
             │
             ▼
        Supervisor
        /    |    \
       ▼     ▼     ▼
  Log agent  Metric agent  Runbook agent
       │     │     │
       └─────┴─────┘
             │
             ▼
        Hypothesis + evidence
             │
             ▼
        Human approval
             │
             ▼
        Action (narrow set)
```

Người giám sát chia sự cố thành các truy vấn phụ. Chuyên agents có quyền truy cập công cụ (tìm kiếm nhật ký, PromQL, truy xuất tài liệu). Giám sát tổng hợp, trình bày giả thuyết + bằng chứng cho con người. Con người phê duyệt hoặc chuyển hướng.

### Phạm vi tự động khắc phục

**An toàn (hẹp)**: khởi động lại pod, hoàn nguyên triển khai cụ thể, mở rộng quy mô nhóm trong giới hạn đã được phê duyệt trước, bật cờ feature được phê duyệt trước.

**Không an toàn (rộng)**: thay đổi cấu trúc liên kết dịch vụ, sửa đổi giới hạn tài nguyên, triển khai mã mới, thay đổi IAM, thay đổi cơ sở dữ liệu.

Bất cứ ai bán "đặt nó và quên nó" là bán quá mức. Bộ an toàn phát triển khi AI SRE trưởng thành, nhưng ranh giới là có thật.

### Đánh giá đối nghịch (NeuBird Hawkeye)

Hai models phân tích độc lập cùng một sự cố. Nếu họ đồng ý về nguyên nhân gốc rễ, sự tự tin là cao. Nếu họ không đồng ý, hãy leo thang lên con người với cả hai giả thuyết có thể nhìn thấy. Mẫu đơn giản, lọc hiệu quả chống lại các nguyên nhân gốc rễ ảo giác.

### Bộ nhớ hoạt động

Doanh thu của đội là sự giết chết thầm lặng của SRE truyền thống - kiến thức bộ lạc rời đi. AI SRE lưu trữ runbook + post-mortems trong một vector DB; agents truy xuất trên mỗi sự cố mới. Khi các kỹ sư mới tham gia, AI đã có đầy đủ lịch sử.

### Dự đoán trước sự cố

Nghiên cứu của MIT 2025: LLM được huấn luyện về nhật ký lịch sử, nhiệt độ GPU API các mẫu lỗi đã dự đoán 89% sự cố mất điện 10-15 phút trước khi chúng xảy ra trên bộ thử nghiệm.

Kiểm tra thực tế: dự đoán không có kích hoạt là bảng điều khiển. Câu hỏi hoạt động là "khi chúng ta dự đoán, chúng ta sẽ làm gì?" Cống trước? Máy nhắn tin? Tự động mở rộng quy mô? Câu trả lời là cụ thể policy.

### Sản phẩm năm 2026

- **Datadog Bits AI** — phi công phụ SRE được quản lý bên trong Datadog.
- **Azure SRE Agent** — Azure-native.
- **NeuBird Hawkeye** — đánh giá đối nghịch + bộ nhớ hoạt động.
- **PagerDuty AIOps** — phân loại + loại bỏ trùng lặp.
- **Incident.io Autopilot** - chỉ huy sự cố + điều phối.

### Runbooks dưới dạng mã

Runbook phát triển từ các trang Confluence sang đánh dấu phiên bản với các phần có cấu trúc (triệu chứng, giả thuyết, xác minh, hành động). Runbook có cấu trúc cung cấp RAG truy xuất tốt hơn. Bắt đầu bất kỳ rollout AI-SRE nào bằng cách biến sổ chạy không có cấu trúc thành có cấu trúc.

### Những con số bạn nên nhớ

- Phát hiện sớm MIT: 89% sự cố mất điện, thời gian thực hiện 10-15 phút.
- Phân loại đa agent: giám sát + (nhật ký, số liệu, sổ chạy) + con người.
- Bộ tự động khắc phục an toàn: khởi động lại pod, hoàn nguyên triển khai, mở rộng quy mô trong giới hạn.
- Đánh giá đối nghịch: hai models độc lập; thỏa thuận = sự tự tin.

## Ứng dụng

`code/main.py` mô phỏng phân loại nhiều agent: nhật ký agent tìm lỗi, agent chỉ số tìm thấy CPU tăng đột biến, runbook agent khớp với vấn đề đã biết. Giám sát xếp hạng các giả thuyết.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-ai-sre-plan.md`. Với hiện tại, sự volume sự cố, sự trưởng thành của nhóm, thiết kế một rollout SRE AI.

## Bài tập

1. Chạy `code/main.py`. Điều gì sẽ xảy ra nếu nhật ký và số liệu agents không đồng ý? Người giám sát giải quyết như thế nào?
2. Xác định ba hành động tự động khắc phục "an toàn" cho dịch vụ của bạn. Biện minh cho mỗi người.
3. Viết một mẫu runbook có cấu trúc: các phần, trường bắt buộc, lệnh xác minh.
4. Phát hiện dự đoán bắn ở 12 phút dẫn đầu. policy của bạn là gì - máy nhắn tin, cống trước hoặc cả hai?
5. Tranh luận xem nhóm 3 người nên áp dụng SRE AI vào năm 2026 hay chờ đợi. Xem xét sự trưởng thành, volume, rủi ro.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| AI SRE | "agent cho cuộc gọi" | Điều tra + điều phối sự cố được hỗ trợ bởi LLM |
| Giám sát agent | "Người dàn nhạc" | agent cấp cao nhất chia các sự cố thành các truy vấn phụ |
| agent chuyên ngành | "agent miền" | agent phụ có quyền truy cập công cụ (nhật ký, số liệu, sổ chạy) |
| Tự động khắc phục | "AI sửa nó" | Hành động hẹp đã được phê duyệt trước; KHÔNG phải kiến trúc lại rộng rãi |
| Bộ nhớ hoạt động | "vector runbooks" | Post-mortems + runbook trong vector DB for RAG |
| Đánh giá đối nghịch | "Kiểm tra hai model" | Phân tích độc lập; thỏa thuận = sự tự tin |
| Mắt diều hâu NeuBird | "kẻ nghịch thù" | Sản phẩm có mẫu bộ nhớ + đánh giá đối nghịch |
| Bits AI | "SRE agent của Datadog" | AI SRE do Datadog quản lý |
| Dự đoán trước sự cố | "Phát hiện sớm" | Thời gian thực hiện 10-15 phút khi dự đoán mất điện |

## Đọc thêm

- [incident.io — AI SRE Complete Guide 2026](https://incident.io/blog/what-is-ai-sre-complete-guide-2026)
- [InfoQ — Human-Centred AI for SRE](https://www.infoq.com/news/2026/01/opsworker-ai-sre/)
- [DZone — AI in SRE 2026](https://dzone.com/articles/ai-in-sre-whats-actually-coming-in-2026)
- [Datadog Bits AI](https://www.datadoghq.com/product/bits-ai/)
- [NeuBird Hawkeye](https://www.neubird.ai/)
- [awesome-ai-sre](https://github.com/agamm/awesome-ai-sre)
