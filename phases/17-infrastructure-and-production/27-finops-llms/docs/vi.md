# FinOps cho LLMs — Kinh tế đơn vị và Phân bổ đa Tenant

> FinOps truyền thống phá vỡ chi tiêu LLM. Chi phí là giao dịch token, không phải thời gian hoạt động của tài nguyên. Thẻ không ánh xạ - lệnh gọi API là một giao dịch, không phải là một tài sản. Quyết định kỹ thuật (prompt thiết kế, context window, độ dài đầu ra) là quyết định tài chính. Cẩm nang năm 2026 có ba khía cạnh phân bổ cho công cụ vào ngày đầu tiên: mỗi người dùng (`user_id`) để định giá và mở rộng chỗ ngồi, mỗi tác vụ (`task_id` + `route`) cho chi phí bề mặt sản phẩm và ưu tiên, trên mỗi tenant (`tenant_id`) cho tính kinh tế đơn vị và gia hạn. Bốn lớp token - prompt, công cụ, bộ nhớ, phản hồi - một nhóm ẩn chi tiêu. Thang thực thi cho các sản phẩm nhiều tenant: rate limits mỗi tenant (2-3x đỉnh dự kiến, xóa 429 + thử lại); giới hạn chi tiêu hàng ngày (trần hợp đồng 1,5-3x; triggers thắt chặt tỷ lệ + cảnh báo); Ngắt kết nối khi chi tiêu điểm Z > 4 (tự động tạm dừng + trang theo cuộc gọi). Các mẫu phân bổ: gắn thẻ và tổng hợp, telemetry-joiner (thanh toán → trace-ID; accuracy cao nhất), sampling và ngoại suy, phân bổ dựa trên model, nguồn sự kiện, streaming theo thời gian thực. Chỉ số đơn vị: chi phí trên mỗi truy vấn đã giải quyết, chi phí trên mỗi artifact được tạo — không phải tokens USD/M. Gắn thẻ hồi tố luôn bị bỏ lỡ; công cụ khi tạo yêu cầu.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, toy cost-attribution simulator with kill switch)
**Kiến thức tiên quyết:** Giai đoạn 17 · 13 (Observability), Giai đoạn 17 · 14 (Bộ nhớ đệm)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Giải thích lý do tại sao FinOps truyền thống (thẻ + bậc) phá vỡ chi tiêu LLM và đặt tên cho ba thứ nguyên phân bổ mới.
- Liệt kê bốn lớp token (prompt, công cụ, bộ nhớ, phản hồi) và lý do tại sao thanh toán một nhóm ẩn chi phí.
- Thiết kế thang thực thi (tỷ lệ → giới hạn chi tiêu → kill switch) cho sản phẩm nhiều tenant.
- Chọn một chỉ số đơn vị (chi phí trên mỗi truy vấn đã giải quyết / artifact) thay vì tokens USD/M.

## Vấn đề

Hóa đơn của bạn ghi 40.000 đô la. Bạn không biết:
- Mà tenant đã chi tiêu nó.
- Sản phẩm nào feature lái nó.
- Liệu có người dùng cá nhân nào lạm dụng hay không.
- Cho dù prompt cồng kềnh, gọi công cụ hay khuếch đại bộ nhớ là thủ phạm.

Tag-and-aggregate ở phía nhà cung cấp hoạt động cho các tài nguyên cloud (EC2, S3) nơi các thẻ lan truyền đến mục hàng. LLM API cuộc gọi không tự động gắn thẻ - bạn phải đóng dấu user/task/tenant tại trang web cuộc gọi và thực hiện. Phân bổ hồi tố luôn bỏ lỡ các trường hợp biên.

## Khái niệm

### Ba thứ nguyên phân bổ

**Mỗi người dùng** (`user_id`): ai có chi phí gì. Thúc đẩy định giá chỗ ngồi, các cuộc trò chuyện mở rộng, xác định người dùng thành thạo.

**Mỗi tác vụ** (`task_id` + `route`): bề mặt sản phẩm nào có giá bao nhiêu. Thúc đẩy feature ưu tiên, giết chết các quyết định tốn kém features.

**Per-tenant** (`tenant_id`): khách hàng nào có lãi. Thúc đẩy tính kinh tế đơn vị, định giá gia hạn, ngưỡng bậc.

Thiết bị cả ba tại địa điểm cuộc gọi vào ngày đầu tiên. Hồi tố luôn tồi tệ hơn.

### Bốn lớp token

| Lớp | Ví dụ | % điển hình của tổng số |
|-------|---------|---------------------|
| Prompt | hệ thống + đầu vào của người dùng | 40-60% |
| Công cụ | Kết quả cuộc gọi công cụ được phản hồi lại | 20-40% (agent khối lượng công việc) |
| Bộ nhớ | prior cuộc trò chuyện / tài liệu đã truy xuất | 10-30% |
| Câu trả lời | model đầu ra | 10-30% |

Nhóm cả bốn với nhau làm cho tối ưu hóa bị mù. Chia nhỏ chúng trong schema phân bổ của bạn.

### Thang thực thi

1. **Rate limit** mỗi tenant. 2-3x đỉnh dự kiến. Trả lại 429 với `Retry-After`. Tenant thấy ma sát; không có hóa đơn bất ngờ.

2. **Giới hạn chi tiêu hàng ngày** mỗi tenant. Trần hợp đồng 1,5-3x. Trigger: thắt chặt rate limit + cảnh báo khách hàng thành công.

3. **Kill switch** trên điểm Z chi tiêu > 4 so với tenant cơ sở. Tự động tạm dừng tenant; trang đang gọi; leo thang lên OPS + CS.

### Mẫu phân bổ

- **Tag-and-aggregate**: tiêu đề siêu dữ liệu tem; tổng hợp sau. Đơn giản; thô ráp.
- **Telemetry người tham gia**: tham gia traces thanh toán qua ID trace. accuracy cao nhất. Những gì các nhóm trưởng thành làm.
- **Sampling + ngoại suy**: mẫu 5-10%, nhân lên. Hiệu quả về chi phí cho chi tiêu thô; bỏ lỡ đuôi.
- **Phân bổ dựa trên Model**: hồi quy để suy ra trình điều khiển chi phí. Đối với dữ liệu cũ không có thẻ.
- **Nguồn sự kiện**: chi phí dưới dạng sự kiện trong luồng (Kafka / Kinesis). Thời gian thực.
- **streaming thời gian thực**: bảng điều khiển cập nhật dưới giây.

### Chi phí trên mỗi X là số liệu đơn vị

$ / M tokens là lời nói của nhà cung cấp. Chỉ số sản phẩm:

- Chi phí cho mỗi phiếu hỗ trợ đã giải quyết.
- Chi phí cho mỗi bài viết được tạo.
- Chi phí cho mỗi nhiệm vụ agent thành công.
- Chi phí cho mỗi người dùng-session-phút.

Gắn chi phí với kết quả sản phẩm. Nếu không, tối ưu hóa sẽ không được neo.

### Phân bổ chi phí trace hình dạng

```
trace_id: abc123
  user_id: u_42
  tenant_id: t_7
  task_id: task_classify_doc
  route: model_haiku
  layers:
    prompt_tokens: 1800
    tool_tokens: 600
    memory_tokens: 400
    response_tokens: 150
  cost_usd: 0.0135
  cached_input: true
  batch: false
```

Phát ra trong mỗi cuộc gọi. Lưu trữ trong hồ dữ liệu. Tổng hợp trên mỗi thứ nguyên. Giai đoạn 17 · 13 observability stack là nơi điều này sống.

### Tiết kiệm kép stack

Stack: bộ nhớ đệm + batch + tuyến + gateway. Với cả bốn:
- Bộ nhớ đệm L2 (Giai đoạn 17 · 14): Đầu vào rẻ hơn ~10 lần.
- Batch (Giai đoạn 17 · 15): Giảm 50%.
- Lộ trình đến model giá rẻ (Giai đoạn 17 · 16): Giảm 60% chi phí.
- Hiệu quả Gateway (Giai đoạn 17 · 19): dự phòng + thử lại.

Trường hợp tốt nhất xếp chồng lên nhau: ~5-10% đường cơ sở ngây thơ. Hầu hết các đội đều có 2-3 đòn bẩy tham gia; rất ít stack cả bốn.

### Những con số bạn nên nhớ

- Thứ nguyên phân bổ: mỗi người dùng, mỗi tác vụ, mỗi tenant.
- Bốn lớp token: prompt, công cụ, bộ nhớ, phản hồi.
- Kill switch: Chi tiêu điểm Z > 4.
- Chỉ số đơn vị: chi phí cho mỗi truy vấn đã giải quyết, không phải tokens USD/M.
- Tối ưu hóa xếp chồng: ~5-10% đường cơ sở có thể.

## Ứng dụng

`code/main.py` mô phỏng một dịch vụ đa tenant LLM với thang thực thi ba tầng. Tiêm một tenant lạm dụng và chứng minh việc bắn kill switch.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-finops-plan.md`. Với sản phẩm và quy mô, thiết kế schema phân bổ và thang thực thi.

## Bài tập

1. Chạy `code/main.py`. kill switch bắn ở điểm z nào? Làm thế nào để bạn chọn ngưỡng?
2. Thiết kế bảng thông tin chi phí cho mỗi tenant, mỗi tác vụ. 5 chế độ xem bạn xây dựng đầu tiên là gì?
3. tenant lớn nhất của bạn là đơn vị-kinh tế-tiêu cực. Đề xuất ba biện pháp can thiệp theo lệnh tác động của khách hàng.
4. Chi phí điện toán trên mỗi yêu cầu đã giải quyết cho một sản phẩm hỗ trợ: 3 triệu tokens/ticket, ~800 tickets/day GPT-5 tốc độ được lưu trong bộ nhớ đệm.
5. Tranh luận xem tính năng gắn thẻ hồi tố có thể hoạt động hay không. Khi nào nó được chấp nhận?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Phân bổ cho mỗi người dùng | "Chi phí cấp người dùng" | `user_id` đóng dấu trên mỗi cuộc gọi |
| Phân bổ cho mỗi nhiệm vụ | "Chi phí feature" | `task_id` + `route` xác định bề mặt sản phẩm |
| Phân bổ trên mỗi tenant | "Chi phí khách hàng" | `tenant_id`; Thúc đẩy kinh tế đơn vị |
| Bốn lớp token | "Lớp chi phí" | prompt + công cụ + bộ nhớ + phản hồi |
| Rate limit | "429 lính gác" | Trần trên tenant được thực thi ở gateway |
| Giới hạn chi tiêu hàng ngày | "Trần hàng ngày" | Ngân sách trong phạm vi Tenant với cảnh báo |
| Kill switch | "Tự động tạm dừng" | Chi tiêu z-score > 4 triggers tự động tạm ngưng |
| Chi phí cho mỗi giải quyết | "Số liệu đơn vị sản phẩm" | Chi phí gắn liền với kết quả sản phẩm, không phải tokens |
| Telemetry nối | "trace để lập hóa đơn" | Mô hình phân bổ accuracy cao nhất |
| Tối ưu hóa xếp chồng | "Bộ nhớ cache + batch + tuyến đường + gateway" | Tiết kiệm kép đến ~5-10% đường cơ sở |

## Đọc thêm

- [FinOps Foundation — FinOps for AI Overview](https://www.finops.org/wg/finops-for-ai-overview/)
- [FinOps School — Cost per Unit 2026 Guide](https://finopsschool.com/blog/cost-per-unit/)
- [Digital Applied — LLM Agent Cost Attribution 2026](https://www.digitalapplied.com/blog/llm-agent-cost-attribution-guide-production-2026)
- [PointFive — Managed LLMs in Azure OpenAI](https://www.pointfive.co/blog/finops-for-ai-economics-of-managed-llms-in-azure-open-ai)
