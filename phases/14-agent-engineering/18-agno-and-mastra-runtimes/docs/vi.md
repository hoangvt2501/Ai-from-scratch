# Agno và Mastra: Production Runtimes

> Agno (Python) và Mastra (TypeScript) là cặp đôi production-runtime năm 2026. Agno nhắm đến việc khởi tạo agent micro giây và các phần phụ trợ FastAPI không trạng thái. Mastra ships agents, công cụ, quy trình làm việc, định tuyến model hợp nhất và lưu trữ tổng hợp trên chất nền Vercel AI SDK.

**Loại:** Học
**Ngôn ngữ:** Python, TypeScript
**Kiến thức tiên quyết:** Giai đoạn 14 · 01 (Vòng lặp Agent), Giai đoạn 14 · 13 (Đồ thị)
**Thời lượng:** ~45 phút

## Mục tiêu học tập

- Xác định các mục tiêu hiệu suất của Agno và thời điểm chúng quan trọng.
- Kể tên ba primitives của Mastra - Agents, Công cụ, Quy trình làm việc - và các bộ điều hợp server được hỗ trợ.
- Giải thích lý do tại sao chương trình phụ trợ FastAPI phạm vi session không trạng thái là đường dẫn Agno production được đề xuất.
- Chọn Agno vs Mastra cho một stack nhất định (Python trước so với TypeScript trước).

## Vấn đề

LangGraph, AutoGen, CrewAI nặng về framework. Các đội muốn "chỉ vòng lặp agent, nhanh chóng, trong runtime của tôi" tiếp cận Agno (Python) hoặc Mastra (TypeScript). Cả hai đều đánh đổi một số primitives thuộc sở hữu của framework để lấy tốc độ thô và phù hợp hơn với stack xung quanh.

## Khái niệm

### Bất động lực

- Python runtime, trước đây là Phi-data.
- "Không có đồ thị, chuỗi hoặc mô hình phức tạp - chỉ là python thuần túy."
- Mục tiêu hiệu suất từ tài liệu của họ: ~2μs agent khởi tạo, ~3,75 KiB bộ nhớ mỗi agent, ~23 nhà cung cấp model.
- Đường dẫn Production: phần phụ trợ FastAPI có phạm vi session không trạng thái. Mỗi yêu cầu bắt đầu một agent mới; session trạng thái sống trong một DB.
- Đa phương thức gốc (văn bản, hình ảnh, âm thanh, video, tệp) và agentic RAG.

Mục tiêu tốc độ quan trọng khi bạn có hàng nghìn agents ngắn mỗi giây (trò chuyện, người hâm mộ, đánh giá pipelines). Chúng ít quan trọng hơn khi một agent chạy trong 10 phút.

### Mastra

- TypeScript, được xây dựng trên Vercel AI SDK.
- Ba primitives: **Agents**, **Công cụ** (kiểu Zod), **Quy trình làm việc**.
- Bộ định tuyến Model hợp nhất — 3,300+ models trên 94 nhà cung cấp (Tháng Ba 2026).
- Lưu trữ tổng hợp: bộ nhớ, quy trình làm việc observability đến các chương trình phụ trợ khác nhau; ClickHouse được đề xuất cho observability trên quy mô lớn.
- Apache 2.0 với `ee/` thư mục theo giấy phép doanh nghiệp có sẵn nguồn.
- Server bộ điều hợp cho Express, Hono, Fastify, Koa; class Next.js đầu tiên và tích hợp Astro.
- Ships Mastra Studio (localhost:4111) để gỡ lỗi.
- 22k+ GitHub sao, 300k+ lượt tải xuống npm hàng tuần ở mức 1.0 (tháng 1 năm 2026).

### Định vị

Cả hai đều không cố gắng trở thành LangGraph. Họ cạnh tranh trên:

- **Phù hợp với ngôn ngữ.** Agno cho các đội Python đầu; Mastra cho TypeScript đầu tiên.
- **Runtime công thái học.** Agno = chi phí gần bằng không; Mastra = tích hợp với hệ sinh thái Vercel.
- **Observability.** Cả hai đều tích hợp với Langfuse/Phoenix/Opik (Bài 24) nhưng Mastra Studio là bên thứ nhất.

### Khi nào nên chọn từng loại

- **Agno** — Python phụ trợ, nhiều agents tồn tại trong thời gian ngắn, yêu cầu hiệu suất cao, cửa hàng FastAPI.
- **Mastra** — TypeScript phụ trợ, triển khai Next.js / Vercel, định tuyến model đa nhà cung cấp hợp nhất, các công cụ kiểu Zod.
- **LangGraph** (Bài 13) — khi trạng thái bền và suy luận đồ thị rõ ràng quan trọng hơn tốc độ thô.
- **OpenAI / Claude Agent SDK** — khi bạn muốn hình dạng sản phẩm của nhà cung cấp (Bài 16–17).

### Mô hình này sai ở đâu

- **Hiệu suất vì lợi ích của hiệu suất.** Chọn Agno vì "2μs" nghe có vẻ tốt khi khối lượng công việc chậm một agent cuộc gọi cho mỗi yêu cầu. Trên cao không phải là nút thắt cổ chai.
- **Khóa hệ sinh thái.** Tích hợp hương vị Vercel của Mastra là một điểm cộng trên Vercel, một điểm trừ ở những nơi khác.
- **Nhầm lẫn về giấy phép doanh nghiệp.** Thư mục `ee/` của Mastra có sẵn nguồn, không phải Apache 2.0. Đọc giấy phép nếu bạn định phân nhánh.

## Tự xây dựng

Bài học này chủ yếu mang tính so sánh - không có mã nào artifact có thể làm được cả hai frameworks công bằng. Xem `code/main.py` để biết đồ chơi cạnh nhau: một luồng "chạy agent, truyền phát đầu ra, duy trì session" tối thiểu được thực hiện hai lần (một lần hình Agno, một lần hình Mastra).

Chạy nó:

```
python3 code/main.py
```

Hai traces khác nhau về cấu trúc nhưng tương đương về mặt chức năng.

## Ứng dụng

- **Agno** — Python phần phụ trợ cần tốc độ và hình dạng FastAPI.
- **Mastra** — TypeScript phụ trợ với nhiều nhà cung cấp và primitives quy trình làm việc.
- Cả hai đều ship observability hooks của bên thứ nhất. Cả hai đều tích hợp với Langfuse.

## Sản phẩm bàn giao

`outputs/skill-runtime-picker.md` chọn Agno, Mastra, LangGraph hoặc nhà cung cấp SDK dựa trên stack, ngân sách độ trễ và hình dạng hoạt động.

## Bài tập

1. Đọc tài liệu của Agno. Chuyển vòng lặp ReAct stdlib (Bài 01) vào Agno. Điều gì đã biến mất? Điều gì ở lại?
2. Đọc tài liệu của Mastra. Chuyển cùng một vòng lặp sang Mastra. Điều gì đã thay đổi trong việc gõ công cụ (Zod so với không có gì)?
3. Benchmark: đo agent độ trễ khởi tạo trên stack của bạn. 2μs của Agno có quan trọng đối với khối lượng công việc của bạn không?
4. Thiết kế di chuyển: nếu bạn đang chạy CrewAI vào năm Python, điều gì sẽ xảy ra nếu bạn chuyển đến Agno?
5. Đọc các điều khoản cấp phép `ee/` của Mastra. Những hạn chế nào sẽ ảnh hưởng đến một fork mã nguồn mở?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Bất động lực | "Nhanh Python agents" | Stateless session scoped agent runtime |
| Mastra | "TypeScript agents trên Vercel AI SDK" | Agents + Công cụ + Quy trình làm việc + Bộ định tuyến Model |
| Bộ định tuyến Model hợp nhất | "Quyền truy cập nhiều nhà cung cấp" | Khách hàng duy nhất cho 3.300+ models trên 94 nhà cung cấp |
| Lưu trữ tổng hợp | "Nhiều phần phụ trợ" | Memory/workflows/observability từng cửa hàng đến một cửa hàng khác nhau |
| Phòng thu Mastra | "Trình gỡ lỗi cục bộ" | localhost:4111 Giao diện người dùng để xem xét nội tâm agents |
| Nguồn có sẵn | "Không phải OSS" | Giấy phép cho phép đọc nguồn nhưng hạn chế sử dụng thương mại |

## Đọc thêm

- [Agno Agent Framework docs](https://www.agno.com/agent-framework) - mục tiêu hiệu suất, tích hợp FastAPI
- [Mastra docs](https://mastra.ai/docs) - Bộ điều hợp primitives, server Model Bộ định tuyến
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — giải pháp thay thế biểu đồ trạng thái
- [Comet Opik](https://www.comet.com/site/products/opik/) — observability so sánh được trích dẫn bởi tích hợp Mastra
