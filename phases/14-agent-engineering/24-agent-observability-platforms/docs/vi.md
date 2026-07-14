# Agent Observability: Langfuse, Phượng hoàng, Opik

> Ba nền tảng agent observability mã nguồn mở thống trị năm 2026. Langfuse (MIT) - 6 triệu + installs/month, theo dõi + quản lý prompt + đánh giá + phát lại session. Arize Phoenix (Elastic 2.0) — đánh giá sâu agent cụ thể, mức độ liên quan RAG, thiết bị đo lường tự động OpenInference. Sao chổi Opik (Apache 2.0) — tối ưu hóa prompt tự động, guardrails LLM phát hiện ảo giác của thẩm phán.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 14 · 23 (Thế hệ OTel)
**Thời lượng:** ~45 phút

## Mục tiêu học tập

- Kể tên ba nền tảng agent observability mã nguồn mở hàng đầu và giấy phép của chúng.
- Phân biệt những gì mỗi người mạnh nhất: Langfuse (prompt mgmt + sessions), Phoenix (RAG + tự động đo đạc), Opik (tối ưu hóa + guardrails).
- Giải thích lý do tại sao 89% tổ chức báo cáo có agent observability vào năm 2026.
- Triển khai pipeline trace đến bảng điều khiển stdlib với đánh giá LLM đánh giá.

## Vấn đề

OTel GenAI (Bài 23) mang đến cho bạn schema. Bạn vẫn cần nền tảng thu nạp spans, chạy đánh giá, lưu trữ prompt phiên bản và hiển thị hồi quy. Ba ứng cử viên đều nhấn mạnh các phần khác nhau của vòng đời.

## Khái niệm

### Cầu chì (MIT)

- 6 triệu + SDK installs/month, 19k + GitHub sao.
- Features: theo dõi, quản lý prompt với phiên bản + sân chơi, đánh giá (LLM với tư cách là giám khảo, phản hồi của người dùng, tùy chỉnh) session phát lại.
- Tháng 6 năm 2025: trước đây là các mô-đun thương mại (LLM-as-a-judge, hàng đợi chú thích, prompt thí nghiệm, Playground) mã nguồn mở của MIT.
- Mạnh nhất cho: observability đầu cuối với vòng prompt quản lý chặt chẽ.

### Arize Phoenix (Giấy phép đàn hồi 2.0)

- Đánh giá sâu hơn agent cụ thể: phân cụm trace, phát hiện bất thường, mức độ liên quan của truy xuất đối với RAG.
- Tự động đo lường OpenInference gốc.
- Ghép nối với Arize AX được quản lý cho production.
- Không có phiên bản prompt - được định vị như một công cụ drift/behavioral-regression cùng với các nền tảng rộng hơn.
- Mạnh nhất cho: Mức độ liên quan RAG, trôi dạt hành vi, phát hiện bất thường.

### Sao chổi Opik (Apache 2.0)

- Tối ưu hóa prompt tự động thông qua các thử nghiệm A/B.
- Guardrails (biên tập PII, ràng buộc theo chủ đề).
- Phát hiện ảo giác LLM thẩm phán.
- Benchmark từ phép đo của chính Comet: Nhật ký Opik + đánh giá trong 23,44 giây so với Langfuse 327,15 giây (khoảng cách ~14x) - lấy benchmarks của nhà cung cấp làm định hướng.
- Mạnh nhất cho: vòng lặp tối ưu hóa, thử nghiệm tự động guardrail thực thi.

### Dữ liệu ngành

Theo Maxim (phân tích thực địa năm 2026): 89% tổ chức có agent observability tại chỗ; Vấn đề chất lượng là rào cản hàng đầu production (32% người được hỏi trích dẫn chúng).

### Chọn một

| Nhu cầu | Chọn |
|------|------|
| Tất cả trong một với quản lý prompt | Cầu chì |
| Đánh giá RAG sâu + trôi dạt | Phượng hoàng |
| Tối ưu hóa tự động + guardrails | Thuốc lá |
| Cấp phép mở, không có ELv2 | Langfuse (MIT) hoặc Opik (Apache 2.0) |
| Tích hợp Datadog / New Relic | Bất kỳ - tất cả đều xuất OTel |

### Mô hình này sai ở đâu

- **Không có chiến lược đánh giá.** Truy tìm mà không đánh giá chỉ là ghi nhật ký tốn kém.
- **Tự cuộn LLM đánh giá không có grounding.** Áp dụng mẫu CRITIC (Bài 05) - giám khảo cần các công cụ bên ngoài để xác minh thực tế.
- **Prompt phiên bản không gắn với traces.** Khi prod thoái lui, bạn không thể chia đôi thành prompt gây ra nó.

## Tự xây dựng

`code/main.py` triển khai bộ thu thập trace stdlib + trình đánh giá LLM thẩm phán:

- Ăn spans hình GenAI.
- Nhóm theo session, gắn thẻ chạy không thành công (guardrail chuyến đi, đánh giá độ tin cậy thấp).
- Đánh giá LLM theo kịch bản chấm điểm agent câu trả lời trên một bảng đánh giá.
- Một bản tóm tắt giống như bảng điều khiển: tỷ lệ thất bại, lý do thất bại hàng đầu, phân phối điểm đánh giá.

Chạy nó:

```
python3 code/main.py
```

Đầu ra: điểm đánh giá mỗi session và phân loại thất bại phù hợp với những gì Langfuse/Phoenix/Opik sẽ hiển thị.

## Ứng dụng

- **Langfuse** tự lưu trữ hoặc cloud; dây qua OTel hoặc SDK của họ.
- **Arize Phoenix **tự lưu trữ; công cụ tự động OpenInference.
- **Sao chổi Opik** tự lưu trữ hoặc cloud; vòng lặp tối ưu hóa tự động.
- **Datadog LLM Observability** dành cho các nhóm hoạt động + ML hỗn hợp đã chạy Datadog.

## Sản phẩm bàn giao

`outputs/skill-obs-platform-wiring.md` chọn một nền tảng và kết nối các phiên bản traces + đánh giá + prompt vào một agent hiện có.

## Bài tập

1. Xuất một tuần OTel traces sang Langfuse cloud (bậc miễn phí). sessions nào thất bại? Tại sao?
2. Viết một bảng đánh giá LLM cho miền của bạn (tính đúng thực tế, giọng điệu, tuân thủ phạm vi). Kiểm tra trên 50 traces.
3. So sánh phiên bản Langfuse prompt với phân cụm trace của Phoenix. Điều gì cho bạn biết điều gì bị hỏng nhanh hơn?
4. Đọc tài liệu guardrail của Opik. Chuyển guardrail biên tập PII đến một trong các lần chạy agent của bạn.
5. Benchmark ba trong kho dữ liệu của bạn. Bỏ qua các con số do nhà cung cấp công bố; đo lường của riêng bạn.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Truy tìm | "Nhà sưu tập Spans" | Ăn OTel / SDK spans; Chỉ mục theo session |
| Quản lý Prompt | "Prompt CMS" | Phiên bản prompts gắn liền với traces |
| LLM với tư cách là thẩm phán | "Đánh giá tự động" | Phân tách điểm LLM agent đầu ra dựa trên một bảng đánh giá |
| Session phát lại | "Trace phát lại" | Bước qua các lần chạy trước đây để gỡ lỗi |
| RAG mức độ liên quan | "Chất lượng truy xuất" | Ngữ cảnh được truy xuất có khớp với truy vấn không |
| Phân cụm Trace | "Nhóm hành vi" | Cụm chạy tương tự để phát hiện trôi dạt |
| Guardrail thực thi | "Policy vào thời gian nhật ký" | PII/toxicity/scope kiểm tra nội dung đã ghi |

## Đọc thêm

- [Langfuse docs](https://langfuse.com/) - truy tìm, đánh giá prompt mgmt
- [Arize Phoenix docs](https://docs.arize.com/phoenix) — tự động đo đạc, trôi dạt
- [Comet Opik](https://www.comet.com/site/products/opik/) - tối ưu hóa + guardrails
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) - schema cả ba tiêu thụ
