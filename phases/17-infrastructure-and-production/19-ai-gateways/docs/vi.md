# AI Gateways - LiteLLM, Portkey, Kong AI Gateway, Bifrost

> Một gateway nằm giữa các ứng dụng của bạn và nhà cung cấp model. Các features cốt lõi là định tuyến nhà cung cấp, dự phòng, thử lại, giới hạn tốc độ, tham chiếu bí mật, observability guardrails. Phân chia thị trường vào năm 2026: **LiteLLM** là MIT OSS với 100+ nhà cung cấp, tương thích với OpenAI, nhưng chia nhỏ khoảng ~2000 RPS (bộ nhớ 8 GB, lỗi xếp tầng trong benchmarks đã xuất bản); tốt nhất cho Python, <500 RPS dev/prototyping. **Portkey** được định vị control-plane (guardrails, biên tập PII, phát hiện bẻ khóa, dấu vết kiểm tra), đã chuyển sang mã nguồn mở Apache 2.0 vào tháng 3 năm 2026, chi phí độ trễ 20-40 mili giây, giá $49/mo production tier. **Kong AI Gateway** built on Kong Gateway — Kong's own benchmark on same 12 CPUs: 228% faster than Portkey, 859% faster than LiteLLM; $ 100/model/month (tối đa 5 trên bậc Plus); phù hợp với doanh nghiệp nếu bạn đã sử dụng Kong. **Bifrost **(Maxim AI) - tự động thử lại với tính năng lùi có thể định cấu hình, dự phòng về Anthropic trên OpenAI 429. **Cloudflare / Vercel AI Gateways** — được quản lý, không hoạt động, thử lại cơ bản. Nơi lưu trữ dữ liệu thúc đẩy quyết định tự lưu trữ; Portkey và Kong ngồi ở giữa với OSS + tùy chọn được quản lý.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, toy gateway-routing simulator)
**Kiến thức tiên quyết:** Giai đoạn 17 · 01 (Nền tảng LLM được quản lý), Giai đoạn 17 · 16 (Model Định tuyến)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Liệt kê sáu gateway features cốt lõi (định tuyến, dự phòng, thử lại, rate limits, bí mật, observability, guardrails).
- Lập bản đồ bốn gateways 2026 (LiteLLM, Portkey, Kong AI, Bifrost) để mở rộng trần và các trường hợp sử dụng.
- Trích dẫn benchmark Kong (228% so với Portkey, 859% so với LiteLLM) và giải thích lý do tại sao nó lại quan trọng đối với >500 RPS.
- Chọn tự lưu trữ so với được quản lý, nơi lưu trữ dữ liệu và ngân sách hoạt động.

## Vấn đề

Sản phẩm của bạn gọi OpenAI, Anthropic và Llama tự lưu trữ. Mỗi nhà cung cấp có một SDK, model lỗi, rate limit và sơ đồ xác thực khác nhau. Bạn muốn failover (nếu OpenAI 429, hãy thử Anthropic), một kho thông tin xác thực duy nhất, observability hợp nhất và rate limits mỗi tenant.

Phát minh lại điều này ở lớp ứng dụng kết hợp mọi dịch vụ với mọi nhà cung cấp. Một lớp gateway hợp nhất nó thành một process với một API (thường tương thích với OpenAI) mà các nhà cung cấp.

## Khái niệm

### features sáu lõi

1. **Định tuyến nhà cung cấp** — OpenAI, Anthropic, Gemini, tự lưu trữ, v.v. đằng sau một API.
2. **Dự phòng** — trên 429, 5xx hoặc lỗi chất lượng, hãy thử lại ở nơi khác.
3. **Thử lại** — lùi theo cấp số nhân, nỗ lực có giới hạn.
4. **Rate limits** — mỗi tenant, mỗi khóa, mỗi model.
5. **Tài liệu tham khảo bí mật** — lấy thông tin đăng nhập từ vault tại runtime (không bao giờ có trong ứng dụng).
6. **Observability** — Thuộc tính OTel + GenAI (Giai đoạn 17 · 13) + phân bổ chi phí.
7. **Guardrails** — Biên tập PII, phát hiện bẻ khóa, bộ lọc chủ đề được phép.

### LiteLLM — MIT OSS, Python

- 100+ nhà cung cấp, tương thích với OpenAI, config bộ định tuyến, dự phòng, observability cơ bản.
- Phá vỡ khoảng 2000 RPS trong benchmark của Kong; Dấu chân bộ nhớ 8 GB, lỗi xếp tầng khi tải liên tục.
- Phù hợp nhất: ứng dụng Python, <500 RPS, dev/staging gateways, định tuyến thử nghiệm.
- Chi phí: $ 0 cho OSS; cloud cấp miễn phí tồn tại.

### Phím cổng - định vị control plane

- Apache 2.0 OSS kể từ tháng 3 năm 2026. Guardrails, biên tập PII, phát hiện bẻ khóa, dấu vết kiểm tra.
- Chi phí độ trễ 20-40 ms cho mỗi yêu cầu.
- $49/mo cho production cấp với tỷ lệ giữ chân + SLA.
- Phù hợp nhất: các ngành công nghiệp được quản lý cần guardrails + observability đi kèm.

### Kong AI Gateway — vở kịch quy mô

- Được xây dựng trên Kong Gateway (sản phẩm API gateway trưởng thành, lua + OpenResty).
- benchmark của Kong tương đương 12 CPU: nhanh hơn 228% so với Portkey, nhanh hơn 859% so với LiteLLM.
- Giá cả: $100/model/month, tối đa 5 trên bậc Plus.
- Phù hợp nhất: đã có trên Kong; >1000 vòng / giây; sẵn sàng cấp phép.

### Bifrost (Maxim AI)

- Tự động thử lại với tính năng dự phòng có thể định cấu hình.
- Dự phòng Anthropic trên OpenAI 429 là một công thức kinh điển.
- Người mới nhập cảnh; thương mại.

### Cloudflare AI Gateway / Vercel AI Gateway

- Được quản lý, không hoạt động. Thử lại cơ bản và observability.
- Phù hợp nhất: Các ứng dụng JavaScript phục vụ cạnh trên Cloudflare/Vercel.
- Hạn chế so với Kong/Portkey trên guardrails và rate limits.

### Tự lưu trữ so với được quản lý

Nơi lưu trữ dữ liệu là chức năng cưỡng bức. Tự lưu trữ mặc định về chăm sóc sức khỏe và tài chính (LiteLLM hoặc Portkey OSS hoặc Kong). Sản phẩm tiêu dùng được quản lý mặc định (Cloudflare AI Gateway) hoặc cấp trung (quản lý Portkey). Kết hợp: tự lưu trữ cho các tenant được quản lý, được quản lý cho những người khác.

### Ngân sách độ trễ

- LiteLLM: 5-15 ms trên cao điển hình.
- Phím cổng: 20-40 ms trên cao.
- Kong: 3-8 ms trên cao.
- Cloudflare/Vercel: 1-3 ms trên cao (lợi thế cạnh).

Độ trễ Gateway trực tiếp thêm vào TTFT. Đối với TTFT P99 < 100 ms SLA, Kong hoặc Cloudflare. Đối với P99 < 500 ms, bất kỳ.

### Ngữ nghĩa giới hạn tốc độ quan trọng

Gầu token đơn giản hoạt động ở quy mô vừa phải. Multi-tenant yêu cầu cửa sổ trượt + phụ cấp liên tục + phân tầng trên mỗi tenant. LiteLLM ships token-bucket; Kong ships cửa sổ trượt; Chìa khóa cổng ships phân tầng.

### Gateway + observability + soạn thảo định tuyến

Giai đoạn 17 · 13 (observability) + 16 (định tuyến model) + 19 (gateways) là cùng một lớp trong production. Chọn một công cụ bao gồm cả ba hoặc kết nối chúng cẩn thận: hầu hết các triển khai năm 2026 kết hợp Helicone (observability) hoặc Portkey (guardrails) với Kong (thang đo) để phân chia vai trò.

### Những con số bạn nên nhớ

- LiteLLM: ngắt ở ~2000 RPS, bộ nhớ 8 GB.
- Chìa khóa cổng: 20-40 ms trên cao; Apache 2.0 kể từ tháng 3 năm 2026.
- Kong: Nhanh hơn 228% so với Portkey, nhanh hơn 859% so với LiteLLM.
- Định giá Kong: $100/model/month, tối đa 5 trên bậc Plus.
- Cloudflare/Vercel: 1-3 ms trên cao ở rìa.

## Ứng dụng

`code/main.py` mô phỏng định tuyến gateway với dự phòng trên 3 nhà cung cấp dưới 429/5xx tiêm. Báo cáo độ trễ, tỷ lệ thử lại và tỷ lệ truy cập dự phòng.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-gateway-picker.md`. Cho quy mô, tư thế hoạt động, tuân thủ, ngân sách độ trễ, chọn một gateway.

## Bài tập

1. Chạy `code/main.py`. Cấu hình dự phòng từ OpenAI→Anthropic→tự lưu trữ. Tỷ lệ truy cập dự kiến ở tỷ lệ lỗi 5% của nhà cung cấp là bao nhiêu?
2. SLA của bạn là TTFT P99 < 200 mili giây trên đường cơ sở 300 mili giây. Những gateways nào nằm trong ngân sách?
3. Khách hàng chăm sóc sức khỏe yêu cầu tự lưu trữ + biên tập PII + kiểm tra. Chọn Portkey OSS hoặc Kong.
4. So sánh LiteLLM và Kong: một đội nên di chuyển ở mức trần RPS nào?
5. Thiết kế policy giới hạn tốc độ cho SaaS nhiều tenant: bậc miễn phí, bậc dùng thử, bậc trả phí. Xô Token hay cửa sổ trượt?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Gateway | "Nhà môi giới API" | Process ngồi giữa các ứng dụng và nhà cung cấp |
| LiteLLM | "MIT một" | Python OSS, 100+ nhà cung cấp, ngắt ở tốc độ 2K RPS |
| Chìa khóa cổng | "guardrails gateway" | Control plane + observability, Apache 2.0 |
| Kong AI Gateway | "Quy mô một" | Được xây dựng dựa trên Kong Gateway, benchmark lãnh đạo |
| Bifrost | "Maxim's gateway" | Thử lại + Anthropic công thức dự phòng |
| Cloudflare AI Gateway | "Quản lý cạnh" | gateway được quản lý triển khai biên, không hoạt động |
| Biên tập PII | "Quét dữ liệu" | Mặt nạ Regex + NER trước khi gửi đến model |
| Phát hiện bẻ khóa | "prompt bảo vệ phun" | Bộ phân loại trên đầu vào của người dùng |
| Dấu vết kiểm tra | "Nhật ký theo quy định" | Bản ghi bất biến của mọi cuộc gọi LLM |
| Xô Token | "rate limit đơn giản" | Bộ giới hạn tốc độ dựa trên nạp tiền |
| Cửa sổ trượt | "rate limit chính xác" | Bộ giới hạn tỷ lệ cửa sổ thời gian; Công bằng tốt hơn |

## Đọc thêm

- [Kong AI Gateway Benchmark](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — AI Gateways 2026 Comparison](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — Top LLM Gateway Tools 2026](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway docs](https://docs.konghq.com/gateway/latest/ai-gateway/)
