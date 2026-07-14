# LLM Lớp định tuyến — LiteLLM, OpenRouter, Portkey

> Khóa nhà cung cấp rất tốn kém. Khối lượng công việc gọi công cụ khác nhau phù hợp với các models khác nhau. Định tuyến gateways cung cấp cho một bề mặt API, thử lại, failover, theo dõi chi phí và guardrails. Ba nguyên mẫu thống trị năm 2026: LiteLLM (tự lưu trữ mã nguồn mở), OpenRouter (SaaS được quản lý), Portkey (cấp production, mã nguồn mở vào tháng 3 năm 2026). Bài học này đặt tên cho các tiêu chí quyết định và thực hiện một gateway định tuyến stdlib.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, routing + failover + cost tracker)
**Kiến thức tiên quyết:** Giai đoạn 13 · 02 (function calling), Giai đoạn 13 · 17 (gateways)
**Thời lượng:** ~45 phút

## Mục tiêu học tập

- Phân biệt các tùy chọn định tuyến tự lưu trữ, được quản lý và cấp production.
- Triển khai chuỗi dự phòng thử lại các lỗi của nhà cung cấp theo thứ tự ưu tiên đã xác định.
- Theo dõi chi phí cho mỗi yêu cầu và mức sử dụng token giữa các nhà cung cấp.
- Quyết định giữa LiteLLM, OpenRouter và Portkey cho một ràng buộc production nhất định.

## Vấn đề

Các tình huống mà định tuyến nhà cung cấp quan trọng:

1. **Chi phí.** Claude Sonnet có giá gấp 3 lần chi phí của Haiku. Đối với một nhiệm vụ phân loại, Haiku là đủ; đối với một nhiệm vụ tổng hợp, Sonnet rất đáng giá. Lộ trình cho mỗi yêu cầu.

2. **Failover.** OpenAI có một giờ tồi tệ. Mọi yêu cầu đều thất bại. Bạn muốn dự phòng tự động Anthropic mà không cần triển khai lại.

3. **Độ trễ.** Giao diện người dùng live chat cần thời gian nhanh chóng để token trước. Một batch tóm tắt thì không. Định tuyến theo SLA độ trễ.

4. **Tuân thủ.** Người dùng EU phải ở lại các khu vực EU. Lộ trình theo khu vực.

5. **Thử nghiệm.** A/B hai models trên cùng một khối lượng công việc. Định tuyến bằng nhóm thử nghiệm.

Mã hóa thủ công tất cả những điều này cho mỗi tích hợp là lặp đi lặp lại. Một gateway định tuyến cung cấp cho một API tương thích với OpenAI và xử lý rest.

## Khái niệm

### Hình dạng proxy tương thích với OpenAI

Mọi người đều nói OpenAI hình. gateway định tuyến hiển thị `/v1/chat/completions`, chấp nhận OpenAI schema và proxy bên trong cho Anthropic / Gemini / Cohere / Ollama / bất cứ thứ gì. Khách hàng không quan tâm.

### Model bí danh

Thay vì `claude-3-5-sonnet-20251022`, mã của bạn cho biết `our_smart_model`. Các bản đồ gateway có bí danh với models thực. Khi Anthropic ships Claude 4, bạn thay đổi bí danh server mặt; Mã của bạn không chạm vào bất cứ thứ gì.

### Chuỗi dự phòng

```
primary: openai/gpt-4o
on 5xx: anthropic/claude-3-5-sonnet
on 5xx: google/gemini-1.5-pro
on 5xx: refuse
```

Gateways định nghĩa điều này trong một config. Thử lại được tính vào ngân sách để các tầng dự phòng không làm tăng chi phí.

### Bộ nhớ đệm ngữ nghĩa

Giống hệt hoặc gần giống hệt nhau prompts đánh vào bộ nhớ đệm thay vì nhà cung cấp. Tiết kiệm cho các vòng lặp agent lặp lại có thể là 30 đến 60 phần trăm. Các phím dựa trên embedding; gần giống hệt nhau prompts chia sẻ một khe cắm bộ nhớ đệm.

### Guardrails

Cấp Gateway:

- **Biên tập PII.** Regex hoặc thẻ dựa trên ML trước khi gửi prompts.
- **Policy vi phạm.** Từ chối prompts có nội dung bị cấm.
- **Bộ lọc đầu ra.** Chà hoàn thành để tìm rò rỉ.

Portkey và Kong đều ship guardrails cố chấp. LiteLLM để chúng tùy chọn.

### rate limits trên mỗi phím

Một chìa khóa API = một đội. Ngân sách cho mỗi khóa ngăn một nhóm sử dụng hạn ngạch được chia sẻ. Hầu hết gateways đều ủng hộ điều này.

### Tự lưu trữ và đánh đổi được quản lý

| Yếu tố | LiteLLM (tự lưu trữ) | OpenRouter (được quản lý) | Phím cổng (production) |
|--------|----------------------|----------------------|----------------------|
| Mã | Mã nguồn mở, Python | SaaS được quản lý | Mã nguồn mở (Mar 2026) + được quản lý |
| Thành lập | Triển khai proxy | Đăng ký | Hoặc |
| Nhà cung cấp | 100+ | 300+ | 100+ |
| Thanh toán | Chìa khóa của riêng bạn | Tín dụng OpenRouter | Chìa khóa của riêng bạn |
| Observability | Đo từ xa mở | Bảng điều khiển | Biên tập OTel + PII đầy đủ |
| Tốt nhất cho | Các nhóm muốn kiểm soát hoàn toàn | Tạo mẫu nhanh | Production tuân thủ |

LiteLLM chiến thắng khi bạn có một nhóm SRE và muốn có chủ quyền dữ liệu. OpenRouter chiến thắng khi bạn muốn một đăng ký duy nhất và không có cơ sở hạ tầng. Portkey chiến thắng khi bạn cần guardrails và tuân thủ ngay lập tức.

### Theo dõi chi phí

Mọi yêu cầu đều mang `provider`, `model`, `input_tokens`, `output_tokens`. Nhân với giá mỗi model trên token (lấy từ bảng giá mà gateway duy trì). Tổng hợp cho mỗi người dùng / mỗi nhóm / mỗi dự án.

### Định tuyến MCP plus

Một gateway có thể định tuyến cả cuộc gọi LLM VÀ yêu cầu MCP sampling. Khi modelPreferences của yêu cầu sampling thích một model cụ thể, gateway sẽ chuyển sang phần phụ trợ phù hợp. Đây là nơi Giai đoạn 13 · 17 (MCP gateway) Và việc định tuyến bài học này đôi khi gateway merge thành một buổi phụng.

### Chiến lược định tuyến

- **Ưu tiên tĩnh.** Đầu tiên trong danh sách; quay trở lại với lỗi.
- **Cân bằng tải.** Vòng tròn hoặc có trọng số.
- **Nhận biết chi phí.** Chọn độ trễ / chất lượng cuộc họp model rẻ nhất.
- **Nhận biết độ trễ.** Chọn model nhanh nhất trong N phút cuối.
- **Nhận biết nhiệm vụ.** Prompt bộ phân loại định tuyến mã hóa đến model này, tóm tắt đến một  khác.

## Ứng dụng

`code/main.py` triển khai gateway định tuyến trong ~150 dòng: chấp nhận các yêu cầu có hình OpenAI, chuyển sang sơ khai cho mỗi nhà cung cấp, chạy chuỗi dự phòng ưu tiên, theo dõi chi phí cho mỗi yêu cầu và áp dụng chuyển biên tập PII trên đầu vào. Chạy nó với ba tình huống: yêu cầu bình thường, ngừng hoạt động của nhà cung cấp chính kích hoạt dự phòng, rò rỉ PII bị phát hiện do biên tập.

Những gì cần xem:

- `ROUTES` dict: bí danh -> danh sách các nhà cung cấp cụ thể theo thứ tự ưu tiên.
- Thử lại vòng lặp dự phòng trên 5xx.
- Trình theo dõi chi phí nhân mức sử dụng token với tỷ lệ trên mỗi model.
- Trình biên tập PII chà các mẫu hình SSN trước khi chuyển tiếp.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-routing-config-designer.md`. Với hồ sơ khối lượng công việc (độ trễ, chi phí, tuân thủ), skill chọn LiteLLM / OpenRouter / Portkey và tạo ra config định tuyến.

## Bài tập

1. Chạy `code/main.py`. Trigger kịch bản mất điện; Xác nhận dự phòng đến nhà cung cấp thứ hai và chi phí được phân bổ chính xác.

2. Thêm bộ nhớ đệm ngữ nghĩa: SHA256 của prompt là một khóa tra cứu; lần truy cập bộ nhớ cache trở lại ngay lập tức. Đo lường mức tiết kiệm chi phí trong một cuộc gọi lặp lại.

3. Thêm một bộ phân loại prompt định tuyến "mã ..." prompts bí danh ủng hộ trí thông minh và "tóm tắt..." prompts bí danh ủng hộ tốc độ.

4. Thiết kế ngân sách cho mỗi nhóm: mỗi nhóm có giới hạn chi tiêu hàng tháng; gateway từ chối yêu cầu khi đạt đến giới hạn. Chọn mức độ chi tiết thực thi (theo yêu cầu hoặc có cửa sổ).

5. Đọc các tài liệu LiteLLM, OpenRouter và Portkey cạnh nhau. Đặt tên cho một feature mỗi ships mà hai người kia không có.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Định tuyến gateway | "LLM proxy" | Lớp một API bề mặt trước nhiều nhà cung cấp |
| Tương thích OpenAI | "Nói OpenAI schema" | Chấp nhận hình dạng `/v1/chat/completions`, dịch sang bất kỳ phần phụ trợ nào |
| Model bí danh | "our_smart_model" | Đặt tên trong mã của bạn mà gateway ánh xạ đến một model cụ thể |
| Chuỗi dự phòng | "Danh sách thử lại" | Danh sách các nhà cung cấp được sắp xếp khi thất bại |
| Bộ nhớ đệm ngữ nghĩa | "Bộ nhớ đệm Prompt-embedding" | Chìa khóa là embedding của prompt; Gần như trùng lặp chia sẻ lần truy cập bộ nhớ cache |
| Guardrails | "Input/output bộ lọc" | Biên tập PII, từ chối vi phạm policy |
| rate limit trên mỗi phím | "Ngân sách nhóm" | Hạn ngạch được giới hạn theo khóa API |
| Theo dõi chi phí | "Chi tiêu cho mỗi yêu cầu" | Tổng mức sử dụng token x giá mỗi model |
| LiteLLM | "Các proxy mở" | gateway định tuyến OSS có thể tự lưu trữ |
| Bộ định tuyến mở | "SaaS được quản lý" | gateway được lưu trữ với thanh toán dựa trên tín dụng |
| Chìa khóa cổng | "Tùy chọn production" | Mã nguồn mở + được quản lý với guardrails tích hợp sẵn |

## Đọc thêm

- [LiteLLM — docs](https://docs.litellm.ai/) — gateway định tuyến tự lưu trữ
- [OpenRouter — quickstart](https://openrouter.ai/docs/quickstart) — SaaS định tuyến được quản lý
- [Portkey — docs](https://portkey.ai/docs) — định tuyến production với guardrails
- [TrueFoundry — LiteLLM vs OpenRouter](https://www.truefoundry.com/blog/litellm-vs-openrouter) - hướng dẫn quyết định
- [Relayplane — LLM gateway comparison 2026](https://relayplane.com/blog/llm-gateway-comparison-2026) - Khảo sát nhà cung cấp
