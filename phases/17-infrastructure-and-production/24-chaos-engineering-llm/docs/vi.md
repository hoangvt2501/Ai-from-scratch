# Chaos Engineering cho LLM Production

> Kỹ thuật hỗn loạn đối với LLMs là ngành học riêng của nó vào năm 2026. Điều kiện tiên quyết trước khi chạy thử nghiệm trong production: SLI/SLO được xác định, trace+chỉ số+observability nhật ký, rollback tự động, runbook, on-call. Kiến trúc có bốn mặt phẳng: điều khiển (bộ lập lịch thử nghiệm), mục tiêu (dịch vụ, cơ sở hạ tầng, kho dữ liệu), an toàn (bảo vệ + hủy bỏ + bộ lọc lưu lượng truy cập), observability (chỉ số + traces + nhật ký), phản hồi (vào điều chỉnh SLO). Guardrails là bắt buộc: cảnh báo tốc độ đốt tạm dừng thử nghiệm nếu cháy ngân sách lỗi hàng ngày > gấp 2 lần dự kiến; triệt tiêu windows + tương quan trace-ID khử trùng lặp nhiễu cảnh báo. Nhịp điệu: đánh giá canary nhỏ + SLO hàng tuần; ngày thi đấu hàng tháng + khám nghiệm tử thi; Kiểm tra khả năng phục hồi giữa các nhóm hàng quý + Lập bản đồ phụ thuộc. Các thử nghiệm dành riêng cho LLM: quá tải bộ nhớ, lỗi mạng, ngừng hoạt động của nhà cung cấp, prompts dị dạng KV cache bão trục xuất. Công cụ: Harness Chaos Engineering (các khuyến nghị có nguồn gốc từ LLM, giảm tỷ lệ bán kính vụ nổ MCP tích hợp công cụ); Hỗn loạn (CNCF); Chaos Mesh (CNCF Kubernetes-bản địa).

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, toy chaos experiment runner)
**Kiến thức tiên quyết:** Giai đoạn 17 · 23 (SRE cho AI), Giai đoạn 17 · 13 (Observability)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Kể tên năm điều kiện tiên quyết về kỹ thuật hỗn loạn (SLI/SLO, observability, rollback, runbook, on-call) và giải thích lý do tại sao bỏ qua bất kỳ điều kiện nào sẽ phá vỡ thực hành.
- Sơ đồ bốn mặt phẳng (điều khiển, mục tiêu, an toàn, observability) và vòng phản hồi thành SLO.
- Liệt kê năm thử nghiệm dành riêng cho LLM (quá tải bộ nhớ, lỗi mạng, ngừng hoạt động của nhà cung cấp, prompt dị dạng, cơn bão trục xuất KV).
- Chọn một công cụ - Harness, LitmusChaos, Chaos Mesh - stack.

## Vấn đề

Thử nghiệm hỗn loạn trong stacks truyền thống được thiết lập. LLM stacks thêm các chế độ lỗi mới. Một token prompt 4K với một nhân vật độc sẽ dừng tokenizer trong 12 giây. Một nhà cung cấp thượng nguồn 429s; gateway của bạn thử lại; OOM dịch vụ của bạn trên đồng thời khuếch đại thử lại. Một cơn bão trục xuất KV cache dưới tải trọng bùng nổ gây ra các tầng đổ đầy lại làm bão hòa tính toán.

Không có cái nào trong số này hiển thị trong các bài kiểm tra đơn vị. Kỹ thuật hỗn loạn là cách bạn khám phá chúng trước khi người dùng làm.

## Khái niệm

### Điều kiện tiên quyết

Đừng gây hỗn loạn trong production mà không có:

1. **SLI/SLO** — các chỉ số và mục tiêu cấp độ dịch vụ được xác định.
2. **Observability** — traces, số liệu, nhật ký, kết nối với bảng thông tin.
3. **rollback tự động **- Giai đoạn 17 · 20 rollback cờ policy.
4. **Runbooks** — có cấu trúc, Giai đoạn 17 · 23.
5. **Đang gọi** — ai đó để trả lời.

Bỏ lỡ bất kỳ phương tiện nào sự hỗn loạn trở thành sự cố có thật.

### Bốn mặt phẳng + phản hồi

**Control plane** — bộ lập lịch thử nghiệm (quy trình làm việc Litmus, lịch trình Chaos Mesh Harness giao diện người dùng).

**Mặt phẳng mục tiêu** — dịch vụ, nhóm, nút, load balancers, kho dữ liệu.

**Mặt phẳng an toàn **- kill switch, windows triệt tiêu, giới hạn bán kính vụ nổ, cổng ngân sách lỗi.

**Mặt phẳng Observability **- số liệu bình thường + tương quan trace-ID để phân biệt sự hỗn loạn gây ra với các lỗi tự nhiên.

**Vòng lặp phản hồi** — các phát hiện được đưa trở lại điều chỉnh SLO, cập nhật runbook, sửa mã.

### Guardrails là bắt buộc

- **Cảnh báo tốc độ ghi**: tạm dừng thử nghiệm nếu mức đốt ngân sách lỗi hàng ngày vượt quá 2 lần dự kiến.
- **windows triệt tiêu**: tắt tiếng các cảnh báo không phải thử nghiệm trong bán kính vụ nổ trong quá trình thử nghiệm.
- **Tương quan Trace-ID**: tất cả các lỗi do thử nghiệm gây ra đều mang một thẻ để On-Call có thể loại bỏ trùng lặp.

### Năm thử nghiệm dành riêng cho LLM

1. **Quá tải bộ nhớ** — gây bão ưu tiên KV cache bằng cách gửi các yêu cầu ngữ cảnh dài với tính đồng thời cao. Quan sát: dịch vụ có rụng hay sụp đổ một cách duyên dáng không?

2. **Lỗi mạng** — cắt kết nối giữa inference gateway và nhà cung cấp. Quan sát: dự phòng có bắt đầu trong SLA không? (Giai đoạn 17 · 19)

3. **Mô phỏng ngừng hoạt động của nhà cung cấp** — 100% 429 từ OpenAI. Quan sát: định tuyến có failover đến Anthropic không? (Giai đoạn 17 · 16, 19)

4. **prompt sai định dạng** — chèn payload trì hoãn tokenizer (ví dụ: unicode lồng sâu, điểm mã UTF-8 khổng lồ). Quan sát: một yêu cầu có khóa worker không?

5. **Cơn bão trục xuất KV **- buộc trục xuất bằng cách bão hòa ngân sách khối vLLM. Quan sát: LMCache có phục hồi hay dịch vụ bị suy giảm?

### Nhịp điệu

- **Hàng tuần** — thử nghiệm canary nhỏ trong dàn dựng, có thể là 5% prod.
- **Hàng tháng** — ngày thi đấu theo lịch trình trên một kịch bản cụ thể; tham dự giữa các đội; khám nghiệm tử thi.
- **Hàng quý** — kiểm tra khả năng phục hồi giữa các nhóm; cập nhật bản đồ phụ thuộc.

### Dụng cụ

- **Harness Chaos Engineering** - thương mại; Khuyến nghị thí nghiệm có nguồn gốc từ AI; giảm tỷ lệ bán kính nổ; MCP tích hợp công cụ.
- **LitmusChaos** - CNCF tốt nghiệp; Kubernetes dựa trên quy trình làm việc.
- **Lưới hỗn loạn** - CNCF sandbox; Phong cách CRD gốc Kubernetes.
- **Gremlin** — thương mại; hỗ trợ rộng rãi.
- **AWS FIS** / **Azure Chaos Studio** — các dịch vụ cloud được quản lý.

### Bắt đầu từ quy mô nhỏ

Thử nghiệm đầu tiên: pod-kill một bản sao giải mã trong lưu lượng truy cập ổn định. Quan sát việc định tuyến lại và phục hồi. Nếu điều này hoạt động và có vẻ an toàn, hãy chuyển sang hỗn loạn mạng.

Thí nghiệm dành riêng cho LLM đầu tiên: tiêm một nhà cung cấp 429 trong 5 phút. Quan sát dự phòng. Hầu hết các đội đều phát hiện ra dự phòng của họ chưa được kiểm tra đầy đủ.

### Những con số bạn nên nhớ

- Bốn mặt phẳng: điều khiển, mục tiêu, an toàn, observability.
- Tạm dừng tốc độ ghi: Đốt ngân sách hàng ngày dự kiến gấp 2 lần.
- Nhịp điệu: canary hàng tuần, ngày thi đấu hàng tháng, kiểm toán hàng quý.
- Năm thí nghiệm LLM: bộ nhớ, mạng, nhà cung cấp, prompt dị dạng, bão KV.

## Ứng dụng

`code/main.py` mô phỏng ba thí nghiệm hỗn loạn với cổng máy bay an toàn. Các báo cáo thí nghiệm nào sẽ vấp ngã tỷ lệ đốt cháy bị hủy bỏ.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-chaos-plan.md`. Với stack và độ trưởng thành, chọn ba thí nghiệm đầu tiên và dụng cụ.

## Bài tập

1. Chạy `code/main.py`. Thí nghiệm nào vượt qua cổng tốc độ đốt cháy và tại sao?
2. Thiết kế năm thử nghiệm hỗn loạn đầu tiên cho dịch vụ RAG dựa trên vLLM. Bao gồm tiêu chí thành công.
3. Cảnh báo tốc độ đốt của bạn đã tạm dừng thử nghiệm. Làm thế nào để bạn xác định nguyên nhân gốc rễ - hỗn loạn hay tự nhiên?
4. Tranh luận liệu sự hỗn loạn nên diễn ra trong production hay chỉ dàn dựng. Khi nào production là câu trả lời đúng?
5. Kể tên ba chế độ lỗi dành riêng cho LLM mà sự hỗn loạn mạng chung không thể tái tạo.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| SLI / SLO | "Mục tiêu dịch vụ" | Chỉ số + mục tiêu; Điều kiện tiên quyết bắt buộc |
| Bán kính vụ nổ | "Phạm vi" | Tập hợp dịch vụ / người dùng bị ảnh hưởng bởi thử nghiệm |
| Cảnh báo tốc độ đốt cháy | "Cổng ngân sách" | Cháy khi tỷ lệ đốt cháy ngân sách lỗi > gấp 2 lần dự kiến |
| Ngày thi đấu | "Diễn tập hàng tháng" | Bài tập hỗn loạn giữa các đội theo lịch trình |
| Hỗn loạn quỳ | "Quy trình làm việc CNCF" | Công cụ hỗn loạn CNCF Kubernetes tốt nghiệp |
| Lưới hỗn loạn | "CNCF CRD" | CNCF hỗn loạn sandbox Kubernetes bản địa |
| Harness CE | "Hỗ trợ AI thương mại" | Harness sự hỗn loạn với các đề xuất AI |
| prompt dị dạng | "tokenizer bom" | Đầu vào khiến tokenization bị đình trệ |
| Cơn bão trục xuất KV | "Dòng thác ưu tiên" | Trục xuất hàng loạt kích hoạt nạp lại |

## Đọc thêm

- [DevSecOps School — Chaos Engineering 2026 Guide](https://devsecopsschool.com/blog/chaos-engineering/)
- [Ankush Sharma — Observability for LLMs (book)](https://www.amazon.com/Observability-Large-Language-Models-Engineering-ebook/dp/B0DJSR65TR)
- [LitmusChaos (CNCF)](https://litmuschaos.io/)
- [Chaos Mesh (CNCF)](https://chaos-mesh.org/)
- [Harness Chaos Engineering](https://www.harness.io/products/chaos-engineering)
- [AWS FIS](https://aws.amazon.com/fis/)
