# Bộ nhớ: Bối cảnh ảo và MemGPT

> Ngữ cảnh windows là hữu hạn. Các cuộc trò chuyện, tài liệu và traces công cụ thì không. MemGPT (Packer et al., 2023) đóng khung đây là bộ nhớ ảo của hệ điều hành — ngữ cảnh chính là RAM, kho lưu trữ bên ngoài là đĩa, các trang agent giữa chúng. Đây là mô hình mà mọi hệ thống bộ nhớ năm 2026 đều kế thừa.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 14 · 01 (Vòng lặp Agent), Giai đoạn 14 · 06 (Sử dụng công cụ)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Giải thích sự tương tự hệ điều hành mà MemGPT xây dựng trên: ngữ cảnh chính = RAM, ngữ cảnh bên ngoài = đĩa, công cụ bộ nhớ = trang in/out.
- Triển khai mẫu MemGPT hai tầng trong stdlib với bộ đệm ngữ cảnh chính, cửa hàng có thể tìm kiếm bên ngoài và các công cụ in/out trang.
- Mô tả cách các vấn đề agent "ngắt" để truy vấn hoặc sửa đổi bộ nhớ ngoài và cách kết quả được nối trở lại prompt tiếp theo.
- Xác định các lựa chọn thiết kế MemGPT được đưa vào Letta (Bài 08) và Mem0 (Bài 09).

## Vấn đề

Ngữ cảnh windows có vẻ như chúng nên giải quyết trí nhớ. Họ không. Ba chế độ lỗi lặp lại trong production:

1. **Tràn ngập.** Các cuộc trò chuyện nhiều lượt, tài liệu dài hoặc quỹ đạo nặng về cuộc gọi công cụ vượt qua cửa sổ. Mọi thứ đã qua ngưỡng đã biến mất.
2. **Pha loãng.** Ngay cả trong cửa sổ, việc nhồi nhét bối cảnh không liên quan sẽ làm loãng attention về những gì quan trọng. Frontier models vẫn xuống cấp khi đầu vào dài.
3. **Persistence.** Một session mới bắt đầu bằng một cửa sổ trống. Agents không có bộ nhớ ngoài không thể nói "hãy nhớ khi bạn yêu cầu tôi..." trên khắp sessions.

Lớn hơn windows giúp nhưng không khắc phục điều này. Bài báo năm 2025 của Mem0 đã đo lường rằng đường cơ sở cửa sổ 128k vẫn bỏ lỡ các sự kiện đường chân trời dài mà agent cửa sổ 4k với bộ nhớ ngoài bắt được.

## Khái niệm

### MemGPT: sự tương tự của hệ điều hành

Packer et al. (arXiv:2310.08560, v2 tháng 2 năm 2024) ánh xạ quản lý ngữ cảnh với bộ nhớ ảo của hệ điều hành:

| Khái niệm hệ điều hành | Khái niệm MemGPT | Tương tự 2026 production |
|------------|---------------|------------------------|
| RAM | Bối cảnh chính (prompt) | Anthropic/OpenAI context window |
| Đĩa | bối cảnh bên ngoài | vector DB, KV, kho đồ thị |
| Lỗi trang | Lệnh gọi công cụ bộ nhớ | `memory.search`, `memory.read`, `memory.write` |
| Nhân hệ điều hành | Vòng điều khiển agent | Vòng lặp ReAct với các công cụ bộ nhớ |

agent chạy vòng lặp ReAct bình thường. Một class công cụ bổ sung cho phép nó phân trang dữ liệu trong và ngoài ngữ cảnh chính.

### Hai tầng

- **Ngữ cảnh chính.** Kích thước cố định prompt giữ nhiệm vụ hiện tại. Luôn hiển thị cho model.
- **Ngữ cảnh bên ngoài.** Không giới hạn, có thể tìm kiếm thông qua các công cụ. Đọc khi có liên quan, viết khi sự thật xuất hiện.

Bài báo ban đầu đánh giá thiết kế trên hai nhiệm vụ ngoài cửa sổ cơ sở: phân tích tài liệu dài hơn 100k tokens và trò chuyện nhiều session với bộ nhớ liên tục qua nhiều ngày.

### Mô hình ngắt

MemGPT giới thiệu bộ nhớ khi ngắt: giữa cuộc trò chuyện, agent có thể gọi một công cụ bộ nhớ, runtime thực thi nó và kết quả nối vào lượt trợ lý tiếp theo dưới dạng một quan sát mới. Về mặt khái niệm giống hệt với một cuộc gọi hệ thống Unix `read()` chặn process, trả về byte và process tiếp tục.

Bề mặt công cụ bộ nhớ chuẩn:

- `core_memory_append(section, text)` - viết thư cho một phần liên tục của prompt.
- `core_memory_replace(section, old, new)` — chỉnh sửa một phần liên tục.
- `archival_memory_insert(text)` - viết thư vào cửa hàng bên ngoài có thể tìm kiếm.
- `archival_memory_search(query, top_k)` — truy xuất từ cửa hàng bên ngoài.
- `conversation_search(query)` - quét các lượt trước đây.

### Nơi MemGPT kết thúc và Letta bắt đầu

Vào tháng 9 năm 2024, MemGPT trở thành Letta. Nghiên cứu repo (`cpacker/MemGPT`) vẫn còn; Letta mở rộng thiết kế:

- Ba tầng thay vì hai (lõi, recall, lưu trữ — Bài 08).
- Lý luận gốc thay thế mô hình `send_message`/nhịp tim (Bài 08).
- Thời gian ngủ agents chạy công việc bộ nhớ không đồng bộ (Bài 08).

Bài báo MemGPT là nền tảng năm 2026 ngay cả khi các hệ thống production chạy Letta, Mem0 hoặc cửa hàng hai tầng tùy chỉnh.

### Mô hình này sai ở đâu

- **Bộ nhớ thối rữa.** Ghi tích lũy nhanh hơn đọc; truy xuất chìm trong những sự thật cũ kỹ. Khắc phục: hợp nhất định kỳ (Letta sleep-time), vô hiệu hóa rõ ràng (Mem0 conflict detector).
- **Nhiễm độc bộ nhớ.** Bộ nhớ ngoài là văn bản được truy xuất. Nếu nội dung do kẻ tấn công kiểm soát xuất hiện trong ghi chú bộ nhớ, agent sẽ nhập lại nội dung đó vào session tiếp theo. Đây là cuộc tấn công của Greshake et al. (Bài 27) được trình bày lại theo thời gian.
- **Trích dẫn loss.** Agent nhớ lại "người dùng yêu cầu tôi ship X" nhưng không thể trích dẫn ngã rẽ nào. Lưu trữ tham chiếu nguồn (ID session, ID lượt) với mỗi lần ghi lưu trữ.

```figure
context-budget
```

## Tự xây dựng

`code/main.py` triển khai mô hình hai tầng của MemGPT trong stdlib:

- `MainContext` - bộ đệm prompt kích thước cố định với một `core` và một danh sách `messages`; Tự động nén các tin nhắn cũ nhất khi vượt quá giới hạn.
- `ArchivalStore` - lưu trữ BM25 trong bộ nhớ (tính điểm chồng chéo token) của các bản ghi (id, văn bản, thẻ, session, lượt).
- Năm công cụ bộ nhớ ánh xạ đến bề mặt MemGPT.
- Một agent có kịch bản lấp đầy kho lưu trữ với các sự kiện, sau đó trả lời câu hỏi bằng cách gọi `archival_memory_search`.

Chạy nó:

```
python3 code/main.py
```

trace cho thấy agent viết ba sự kiện, điền vào ngữ cảnh chính vào giới hạn (buộc trục xuất), sau đó trả lời câu hỏi tiếp theo bằng cách truy xuất từ kho lưu trữ - tái tạo quy trình làm việc MemGPT mà không có bất kỳ LLM thực sự nào.

## Ứng dụng

Mỗi hệ thống bộ nhớ production ngày nay là một biến thể MemGPT:

- **Letta** (Bài 08) — ba bậc, suy luận gốc, tính toán thời gian ngủ.
- **Mem0** (Bài 09) — vector + KV + đồ thị hợp nhất với một lớp tính điểm.
- **OpenAI Trợ lý / Phản hồi** — bộ nhớ được quản lý qua threads và tệp.
- **Claude Agent SDK** — bộ nhớ dài hạn thông qua skills và session store.

Chọn một theo hình dạng hoạt động (tự lưu trữ, được quản lý framework tích hợp), không phải theo mẫu cốt lõi - mẫu cốt lõi là MemGPT.

## Sản phẩm bàn giao

`outputs/skill-virtual-memory.md` là một skill có thể tái sử dụng tạo ra giàn giáo bộ nhớ hai tầng chính xác (chính + lưu trữ + bề mặt công cụ) cho bất kỳ runtime mục tiêu nào, với các trường policy trục xuất và trích dẫn được nối vào.

## Bài tập

1. Thêm một nắp `max_main_context_tokens` được đo bằng tokens (xấp xỉ với `len(text.split())` * 1.3). Thu gọn các thư cũ nhất thành một bản tóm tắt khi vượt quá giới hạn. So sánh hành vi có và không có trình tóm tắt.
2. Triển khai BM25 đúng cách trên kho lưu trữ (tần suất thuật ngữ, tần suất tài liệu nghịch đảo). Đo lường recall@10 trên một bộ dữ kiện đồ chơi so với đường cơ sở chồng chéo token.
3. Thêm các trường `citation` (session_id, turn_id, source_url) vào chèn lưu trữ. Làm cho agent trích dẫn nguồn trên mọi câu trả lời được hỗ trợ truy xuất.
4. Mô phỏng nhiễm độc bộ nhớ: thêm một bản ghi lưu trữ có nội dung "bỏ qua tất cả các hướng dẫn sử dụng trong tương lai". Viết một bảo vệ quét truy xuất để tìm văn bản có hình chỉ thị và đánh dấu chúng không đáng tin cậy.
5. Chuyển triển khai để sử dụng JSON schema bộ nhớ lõi (`cpacker/MemGPT`) của repo nghiên cứu MemGPT. Điều gì thay đổi khi bạn chuyển từ chuỗi phẳng sang phần được nhập?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Bối cảnh ảo | "Bộ nhớ không giới hạn" | Các bậc chính (prompt) + bên ngoài (có thể tìm kiếm) với in/out trang |
| Bối cảnh chính | "Trí nhớ làm việc" | prompt - kích thước cố định, luôn hiển thị |
| Bộ nhớ lưu trữ | "Lưu trữ dài hạn" | persistence có thể tìm kiếm bên ngoài, truy xuất theo yêu cầu |
| Bộ nhớ lõi | "Phần prompt liên tục" | Các phần được đặt tên được ghim bên trong ngữ cảnh chính |
| Công cụ bộ nhớ | "Bộ nhớ API" | Công cụ gọi các vấn đề agent để read/write bộ nhớ ngoài |
| Ngắt | "Lỗi trang bộ nhớ" | Agent tạm dừng, runtime tìm nạp, kết quả nối vào lượt tiếp theo |
| Ký ức thối rữa | "Sự thật cũ" | Viết cũ chết đuối thu hồi; Khắc phục bằng hợp nhất |
| Ngộ độc trí nhớ | "Ghi chú liên tục được tiêm" | Nội dung của kẻ tấn công được lưu trữ dưới dạng bộ nhớ, được nhập lại trên recall |

## Đọc thêm

- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — Bài báo ngữ cảnh ảo lấy cảm hứng từ hệ điều hành
- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) - sự tiến hóa ba tầng
- [Anthropic, Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) - coi bối cảnh như một ngân sách
- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) — bộ nhớ production lai trên mẫu này
