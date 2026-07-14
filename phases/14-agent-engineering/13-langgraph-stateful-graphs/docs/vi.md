# LangGraph: Biểu đồ trạng thái và thực thi bền bỉ

> LangGraph là tài liệu tham khảo năm 2026 cho orchestration trạng thái cấp thấp. Agent là một cỗ máy nhà nước; các nút là hàm; các cạnh là chuyển tiếp; trạng thái là bất biến và được kiểm tra sau mỗi bước. Tiếp tục từ bất kỳ lỗi nào chính xác ở nơi nó đã dừng lại.

**Loại:** Tìm hiểu + Xây dựng
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 14 · 01 (Vòng lặp Agent), Giai đoạn 14 · 12 (Mẫu quy trình làm việc)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Mô tả model cốt lõi của LangGraph: máy trạng thái với trạng thái bất biến, các nút hàm, cạnh có điều kiện và checkpoints sau bước.
- Kể tên bốn khả năng mà tài liệu làm nổi bật: thực thi bền bỉ, streaming, con người trong vòng lặp, bộ nhớ toàn diện.
- Giải thích ba cấu trúc liên kết orchestration mà LangGraph hỗ trợ: giám sát, ngang hàng (swarm), phân cấp (đồ thị con lồng nhau).
- Triển khai biểu đồ trạng thái stdlib với trạng thái bất biến, cạnh có điều kiện và chu kỳ checkpoint/resume.

## Vấn đề

Agents và quy trình làm việc có chung một vấn đề: Khi chạy 40 bước không thành công ở bước 38, bạn muốn tiếp tục từ bước 38 chứ không phải bắt đầu lại. Trạng thái class thứ hai models khiến các toán tử hack thử lại xung quanh một thư viện giả định các lần chạy mới.

Câu trả lời thiết kế của LangGraph: state là một đối tượng được gõ class đầu tiên, các đột biến rõ ràng và checkpoints tồn tại sau mỗi nút. Resume là một cuộc gọi `load_state(session_id)`.

## Khái niệm

### Biểu đồ

Đồ thị được xác định bởi:

- **Loại trạng thái.** Một dict được gõ (hoặc Pydantic model) mà mọi nút đọc và thay đổi.
- **Các nút.** Chức năng thuần túy `(state) -> state_update`. Các bản cập nhật được merged trạng thái sau khi trả về.
- **Cạnh.** Chuyển đổi có điều kiện hoặc trực tiếp giữa các nút.
- **Vào và ra.** `START` và `END` nút lính gác đánh dấu ranh giới.

Ví dụ: một agent có các nút `classify`, `refund`, `bug`, `sales` `done` — quy trình định tuyến dưới dạng biểu đồ.

### Thực thi bền bỉ

Sau khi mỗi nút trả về, runtime sẽ tuần tự hóa trạng thái và ghi nó vào một con trỏ kiểm tra (SQLite, Postgres, Redis, tùy chỉnh). Khi thất bại ở bước N, runtime có thể `resume(session_id)` và tiếp tục từ bước N + 1 với trạng thái chính xác.

Tài liệu LangGraph nêu bật rõ ràng production người dùng khi điều này quan trọng: Klarna, Uber, JP Morgan. Tuyên bố không phải là hình dạng đồ thị; Đó là hình dạng đồ thị cộng với điểm kiểm tra làm cho việc phục hồi trở nên rẻ.

### Streaming

Mỗi nút có thể mang lại đầu ra một phần. Biểu đồ truyền các sự kiện trên mỗi nút delta đến người gọi để giao diện người dùng cập nhật khi biểu đồ chạy.

### Con người trong vòng lặp

Kiểm tra và sửa đổi trạng thái giữa các nút. Triển khai: tạm dừng trước một nút quan trọng, trạng thái bề mặt cho con người, chấp nhận sửa đổi, tiếp tục. Checkpointer làm cho điều này dễ dàng vì state đã được tuần tự hóa.

### Bộ nhớ

Ngắn hạn (trong một lần chạy - lịch sử cuộc trò chuyện trong trạng thái) và dài hạn (giữa các lần chạy - liên tục thông qua kiểm tra cộng với một kho lưu trữ dài hạn riêng biệt). LangGraph tích hợp với các hệ thống bộ nhớ ngoài (Mem0, tùy chỉnh) thông qua các công cụ.

### Ba cấu trúc liên kết

1. **Giám sát.** Bộ định tuyến trung tâm LLM gửi đến subagents chuyên gia. `create_supervisor()` vào năm `langgraph-supervisor` (mặc dù nhóm LangChain vào năm 2026 khuyên bạn nên thực hiện điều này thông qua các cuộc gọi công cụ trực tiếp để kiểm soát ngữ cảnh nhiều hơn).
2. **Swarm / ngang hàng.** Agents chuyển giao trực tiếp thông qua bề mặt công cụ dùng chung. Không có bộ định tuyến trung tâm.
3. **Phân cấp.** Người giám sát quản lý người giám sát phụ, được triển khai dưới dạng biểu đồ con lồng nhau.

### Mô hình này sai ở đâu

- **Checkpoints quá nhỏ.** Chỉ có lượt hội thoại điểm kiểm tra mới khiến trạng thái công cụ và ghi bộ nhớ không thể khôi phục được. Trạng thái đầy đủ phải tuần tự.
- **Các nút không xác định.** Resume giả định các đầu vào nút tạo ra cùng một cập nhật trạng thái. Hạt giống ngẫu nhiên, đồng hồ treo tường, APIs bên ngoài phải được bắt.
- **Sử dụng quá mức các cạnh có điều kiện.** Đồ thị với mọi cạnh có điều kiện là một cỗ máy trạng thái không thể lý luận được. Thích chuỗi tuyến tính với branches không thường xuyên.

## Tự xây dựng

`code/main.py` triển khai biểu đồ trạng thái stdlib:

- `State` — một câu chính tả được đánh máy với `messages`, `step`, `route`, `output`, `human_approval`.
- `Node` - có thể gọi lấy trạng thái và trả về một lệnh cập nhật.
- `StateGraph` — nút + cạnh + cạnh có điều kiện + chạy + tiếp tục.
- `SQLiteCheckpointer` (giả mạo trong bộ nhớ) — tuần tự hóa trạng thái sau mỗi nút; `load(session_id)` khôi phục.
- Biểu đồ demo: phân loại -> branch(hoàn tiền / lỗi / bán hàng) -> cổng con người -> gửi.

Chạy nó:

```
python3 code/main.py
```

trace cho thấy lần chạy đầu tiên thất bại ở cổng con người, persistence, sau đó tiếp tục tạo ra đầu ra cuối cùng.

## Ứng dụng

- **LangGraph** — tài liệu tham khảo, sẵn sàng production. Sử dụng `create_react_agent`, `create_supervisor` hoặc xây dựng biểu đồ của riêng bạn.
- **AutoGen v0.4** (Bài 14) — actor model giải pháp thay thế cho các tình huống đồng thời cao.
- **Claude Agent SDK** (Bài 17) — harness được quản lý với cửa hàng session tích hợp.
- **Tùy chỉnh** — khi bạn cần kiểm soát chính xác hình dạng trạng thái hoặc phần phụ trợ của con trỏ kiểm tra.

## Sản phẩm bàn giao

`outputs/skill-state-graph.md` tạo biểu đồ trạng thái hình LangGraph trong bất kỳ runtime đích nào có điểm kiểm tra và tiếp tục có dây.

## Bài tập

1. Thêm cạnh có điều kiện từ `classify` đến `end` khi độ tin cậy phân loại dưới ngưỡng. Tiếp tục chạy sau khi con người đặt `route` theo cách thủ công.
2. Hoán đổi hàng giả giống SQLite cho một con trỏ kiểm tra SQLite thật. Đo lường chi phí tuần tự hóa theo bước.
3. Triển khai các cạnh song song: hai nút chạy đồng thời, merge bằng một bộ giảm tốc tùy chỉnh. Trạng thái bất biến mua gì ở đây?
4. Đọc `langgraph-supervisor` tham khảo. Chuyển đồ chơi vào `create_supervisor`. So sánh các hình dạng trace.
5. Thêm streaming: mỗi nút mang lại trạng thái một phần trong khi chạy. In các delta khi chúng đến.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Biểu đồ trạng thái | "Agent như cỗ máy trạng thái" | Trạng thái gõ + nút + cạnh + bộ giảm tốc |
| Kiểm tra | "Persistence phụ trợ" | Tuần tự hóa trạng thái sau mỗi nút; Cho phép sơ yếu lý lịch |
| Giảm tốc | "Sáp nhập tiểu bang" | Chức năng kết hợp trạng thái hiện tại với cập nhật của nút |
| Cạnh có điều kiện | "Branch" | Cạnh được chọn bởi một hàm của trạng thái |
| Biểu đồ con | "Đồ thị lồng nhau" | Một biểu đồ được sử dụng làm nút bên trong một biểu đồ khác |
| Thực thi bền bỉ | "Tiếp tục từ thất bại" | Khởi động lại ở nút thành công cuối cùng với trạng thái chính xác |
| Giám sát viên | "Bộ định tuyến LLM" | Điều phối viên trung tâm cho subagents chuyên gia |
| Swarm | "agents P2P" | Agents bàn giao thông qua các công cụ được chia sẻ; Không có bộ định tuyến trung tâm |

## Đọc thêm

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — tài liệu tham khảo
- [langgraph-supervisor reference](https://reference.langchain.com/python/langgraph/supervisor/) - API mẫu giám sát
- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — thay thế diễn viên model
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — session cửa hàng và subagents
