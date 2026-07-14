# Giao diện công cụ — Tại sao Agents cần I/O có cấu trúc

> Một ngôn ngữ model tạo ra tokens. Một chương trình thực hiện hành động. Khoảng cách giữa hai điều đó là giao diện công cụ: một hợp đồng cho phép model yêu cầu một hành động và máy chủ thực hiện nó. Mỗi năm 2026 stack - function calling trên OpenAI, Anthropic và Gemini; MCP `tools/call`; Các phần nhiệm vụ của A2A - là một mã hóa khác của cùng một vòng lặp bốn bước. Bài học này đặt tên cho vòng lặp và hiển thị máy móc tối thiểu để chạy nó.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, no LLM)
**Kiến thức tiên quyết:** Giai đoạn 11 (LLM hoàn thành APIs)
**Thời lượng:** ~45 phút

## Mục tiêu học tập

- Giải thích lý do tại sao một LLM chỉ có thể tạo văn bản không thể tự mình thực hiện các hành động chống lại thế giới thực.
- Vẽ vòng lặp gọi công cụ bốn bước (mô tả → quyết định → thực hiện → quan sát) và nêu tên người sở hữu mỗi bước.
- Viết mô tả công cụ thành ba phần: tên, đầu vào JSON Schema và hàm thực thi xác định.
- Phân biệt các công cụ tinh khiết và tác dụng phụ và nêu lý do tại sao việc phân tách lại quan trọng đối với sự an toàn.

## Vấn đề

Một LLM phát ra phân phối xác suất trong token tiếp theo. Đó là toàn bộ bề mặt đầu ra. Nếu bạn hỏi một cuộc trò chuyện model "thời tiết ở Bengaluru hiện tại như thế nào", nó có thể viết một câu hợp lý, nhưng nó không thể quay số thành một API thời tiết. Câu này có thể đúng là trùng hợp ngẫu nhiên hoặc ba ngày cũ.

Thu hẹp khoảng cách đó là mục đích của giao diện công cụ. Chương trình lưu trữ — agent runtime, Claude Desktop, ChatGPT, Con trỏ hoặc script tùy chỉnh của bạn — quảng cáo danh sách các công cụ có thể gọi cho model. model, khi quyết định một hành động là cần thiết, sẽ phát ra một payload có cấu trúc đặt tên cho một công cụ và các đối số của nó. Máy chủ phân tích cú pháp payload đó, chạy công cụ thực và cung cấp lại kết quả. Vòng lặp tiếp tục cho đến khi model quyết định không cần gọi nữa.

Phiên bản đầu tiên của hợp đồng này shipped vào tháng 6 năm 2023 khi "chức năng" của OpenAI parameter. Anthropic theo sau với `tool_use` khối trong Claude 2.1. Gemini thêm `functionDeclarations` vài tháng sau đó. Giờ đây, mọi nhà cung cấp đều hiển thị cùng một hình dạng: danh sách công cụ được gõ JSON Schema, gọi công cụ JSON payload. Giao thức ngữ cảnh Model (tháng 11 năm 2024) đã khái quát hóa hợp đồng để một công cụ registry phục vụ mọi model. A2A (tháng 4 năm 2026, phiên bản 1.0) đã phân lớp cùng một primitive cho ủy quyền agent-agent.

Vòng lặp bốn bước là bất biến bên dưới tất cả những điều này. Mọi thứ khác trong Giai đoạn 13 là một sự trau chuốt.

## Khái niệm

### Bước một: mô tả

Máy chủ khai báo mỗi công cụ với ba trường.

- **Tên.** Một mã định danh ổn định, có thể đọc được bằng máy. `get_weather`, không phải "điều thời tiết".
- **Mô tả.** Một bản tóm tắt bằng ngôn ngữ tự nhiên một đoạn. "Sử dụng khi người dùng hỏi về các điều kiện hiện tại của một thành phố cụ thể. Không sử dụng cho dữ liệu lịch sử."
- **Đầu vào schema.** Một đối tượng JSON Schema (bản nháp 2020-12) mô tả các đối số của công cụ.

model nhận được danh sách. Các nhà cung cấp hiện đại tuần tự hóa các khai báo này vào system prompt bằng cách sử dụng mẫu dành riêng cho nhà cung cấp, vì vậy bạn với tư cách là người gọi chỉ xử lý biểu mẫu có cấu trúc.

### Bước hai: quyết định

Với thông điệp của người dùng và các công cụ có sẵn, model chọn một trong ba hành vi.

1. **Trả lời trực tiếp** bằng văn bản. Không cần gọi công cụ.
2. **Gọi một hoặc nhiều công cụ.** Phát ra các đối tượng cuộc gọi có cấu trúc. Trong `parallel_tool_calls: true` (mặc định trên OpenAI và Gemini, chọn tham gia Anthropic), model có thể phát ra nhiều cuộc gọi trong một lượt.
3. **Từ chối.** Đầu ra có cấu trúc ở chế độ nghiêm ngặt có thể tạo ra một khối `refusal` được nhập thay vì một cuộc gọi.

Một payload gọi công cụ có ba trường ổn định: `id` gọi, `name` công cụ và đối tượng JSON `arguments`. ID tồn tại để máy chủ có thể tương quan kết quả sau đó với lệnh gọi cụ thể, điều này quan trọng khi các cuộc gọi song song trở lại không theo thứ tự.

### Bước ba: thực hiện

Máy chủ nhận lệnh gọi, xác thực các đối số chống lại schema đã khai báo và chạy trình thực thi. Đối số không hợp lệ có nghĩa là model ảo giác một trường hoặc sử dụng sai loại - một chế độ lỗi rất phổ biến trên models yếu. Máy chủ Production thực hiện một trong ba điều đối với các đối số không hợp lệ: thất bại nhanh chóng và hiển thị lỗi cho model, sửa chữa JSON bằng trình phân tích cú pháp bị ràng buộc hoặc thử lại model với lỗi xác thực có trong prompt.

Bản thân trình thực thi là mã thông thường. Python, TypeScript, lệnh shell, truy vấn cơ sở dữ liệu. Nó tạo ra một kết quả, thường là một chuỗi nhưng có thể là bất kỳ giá trị JSON nào hoặc một khối nội dung có cấu trúc (tham chiếu văn bản, hình ảnh hoặc tài nguyên trong MCP). Kết quả phải có thể tuần tự hóa.

### Bước bốn: quan sát

Người chủ trì thêm kết quả công cụ vào cuộc trò chuyện (dưới dạng thông báo vai trò `tool` với các `id` phù hợp) và gọi lại model. model hiện có đầu ra công cụ trong ngữ cảnh và có thể tạo ra câu trả lời cuối cùng hoặc yêu cầu nhiều cuộc gọi hơn. Điều này tiếp tục cho đến khi model ngừng phát ra cuộc gọi hoặc máy chủ đạt đến giới hạn an toàn về số lần lặp lại.

### Sự chia rẽ niềm tin

Các công cụ có hai hương vị quan trọng đối với sự an toàn.

- **Thuần túy.** Chỉ đọc, xác định, không có tác dụng phụ. `get_weather`, `search_docs`, `get_current_time`. An toàn để gọi một cách suy đoán.
- **Hậu quả.** Thay đổi trạng thái, chi tiền, chạm vào dữ liệu người dùng. `send_email`, `delete_file`, `execute_trade`. Phải được kiểm soát.

"Quy tắc hai" năm 2026 của Meta cho bảo mật agent cho biết một lượt duy nhất có thể kết hợp nhiều nhất là hai: đầu vào không đáng tin cậy, dữ liệu nhạy cảm, hành động do hậu quả. Giao diện công cụ là nơi bạn thực thi quy tắc đó - bằng cách từ chối cuộc gọi, yêu cầu xác nhận của người dùng hoặc leo thang phạm vi. Xem Giai đoạn 13 · 15 cho chương bảo mật đầy đủ và Giai đoạn 14 · 09 đối với policies cấp phép cấp agent.

### Vòng lặp tồn tại ở đâu

| Bối cảnh | Ai mô tả | Ai quyết định | Ai thực hiện |
|---------|---------------|-------------|--------------|
| function calling một lượt (OpenAI/Anthropic/Gemini) | Nhà phát triển ứng dụng | LLM | Nhà phát triển ứng dụng |
| MCP | MCP server | LLM qua MCP khách | MCP server |
| A2A | Nhà xuất bản thẻ Agent | Gọi agent | Được gọi là agent |
| Trình duyệt web (agent gọi hàm) | Tiện ích mở rộng trình duyệt / WebMCP | LLM | Trình duyệt runtime |

Mọi nơi, bốn bước giống nhau. Tên cột thay đổi; cấu trúc thì không.

### Tại sao không chỉ prompt model để phát ra JSON?

"Yêu cầu model trả lời bằng JSON" là mẫu gọi trước chức năng. Nó thất bại ~5 đến 15 phần trăm thời gian trên models biên giới và nhiều hơn trên các models nhỏ hơn. Các chế độ lỗi bao gồm thiếu dấu ngoặc nhọn, dấu phẩy đuôi, trường ảo giác và sai loại. Sau đó, bạn cần một thẻ sửa chữa JSON, thử lại hoặc một decoder hạn chế.

function calling bản địa tốt hơn vì ba lý do. Đầu tiên, nhà cung cấp huấn luyện model từ đầu đến cuối về hình dạng cuộc gọi chính xác, vì vậy tỷ lệ JSON hợp lệ tăng lên 98 đến 99% ở chế độ nghiêm ngặt. Thứ hai, cuộc gọi payload nằm trong khe giao thức riêng của nó, không phải bên trong văn bản tự do - vì vậy lệnh gọi công cụ không bao giờ bị rò rỉ vào câu trả lời mà người dùng nhìn thấy. Thứ ba, các nhà cung cấp thực thi schema tuân thủ giải mã ràng buộc (chế độ nghiêm ngặt của OpenAI, `tool_use` của Anthropic, `responseSchema` của Gemini). Đầu ra được đảm bảo xác thực.

Giai đoạn 13 · 02 đi bộ ba nhà cung cấp APIs cạnh nhau. Giai đoạn 13 · 04 đi sâu vào đầu ra có cấu trúc.

### Bộ ngắt mạch

Vòng lặp kết thúc khi model ngừng phát ra cuộc gọi hoặc máy chủ đạt đến số lượt tối đa. Production chủ nhà đặt điều này từ 5 đến 20 lượt. Ngoài ra, bạn gần như chắc chắn đang ở trong một vòng lặp mà model không thể thoát ra. Mã Claude mặc định là 20; OpenAI Trợ lý cho 10; Chế độ agent của con trỏ thành 25.

Giải pháp thay thế - vòng lặp không giới hạn - hiển thị sáu tháng một lần là "agent đã chi 400 đô la cho các cuộc gọi API qua đêm". Đừng ship mà không có giới hạn.

Giai đoạn 14 · 12 bao gồm phục hồi lỗi và tự phục hồi chuyên sâu; Giai đoạn 17 bao gồm production rate limits.

### Giai đoạn 13 sẽ đi đến đâu từ đây

- Bài học từ 02 đến 05 đánh bóng bề mặt gọi công cụ cấp nhà cung cấp.
- Bài 06 đến 14 khái quát hóa vòng lặp thành MCP.
- Bài học từ 15 đến 18 bảo vệ vòng lặp chống lại servers thù địch, người dùng đối nghịch và các bề mặt xác thực từ xa chưa được xác thực.
- Các bài học từ 19 đến 22 mở rộng mô hình sang cộng tác agent-agent, observability, định tuyến và đóng gói.
- Bài 23 ships một hệ sinh thái hoàn chỉnh sử dụng mọi primitive.

Mỗi bài học còn lại là một sự trau chuốt của vòng lặp bốn bước này. Hãy ghi nhớ nó như một bất biến.

## Ứng dụng

`code/main.py` chạy vòng lặp bốn bước mà không cần LLM. Chức năng "quyết định" giả mô phỏng model bằng cách khớp mẫu trên tin nhắn người dùng; Executor, schema validator và observe-step harness là có thật. Chạy nó để xem toàn bộ vũ đạo request/response với trạng thái trung gian có thể in được, sau đó thay thế bộ quyết định giả bằng bất kỳ nhà cung cấp thực sự nào trong bài học sau.

Những gì cần xem:

- Công cụ registry chứa ba trường cho mỗi công cụ: tên, mô tả, schema và tham chiếu trình thực thi.
- Trình xác thực là một tập hợp con JSON Schema tối thiểu (types, required, enum, min/max) chỉ được viết bằng stdlib. Giai đoạn 13 · 04 ships một cái đầy đủ hơn.
- Vòng lặp giới hạn số lần lặp lại là năm. Production agents cần chính xác loại cầu dao này.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-tool-interface-reviewer.md`. Với định nghĩa công cụ dự thảo (tên + mô tả + schema + phác thảo trình thực thi), skill kiểm tra nó về tính phù hợp của vòng lặp: có phải tên máy ổn định không, mô tả có phải là bản tóm tắt sử dụng hoàn chỉnh không, schema có sử dụng chính xác JSON Schema 2020-12 không và phân loại thuần túy so với hậu quả rõ ràng.

## Bài tập

1. Thêm công cụ thứ tư vào `code/main.py` có tên là `get_stock_price(ticker)`. Viết mô tả của nó là "Sử dụng khi người dùng yêu cầu giá cổ phiếu hiện tại bằng mã chứng khoán. Không sử dụng cho lịch sử giá hoặc tóm tắt thị trường." Chạy harness và xác nhận các truy vấn định tuyến quyết định giả mạo đề cập đến mã đến công cụ mới.

2. Phá vỡ trình xác thực schema. Chuyển một cuộc gọi có đối tượng `arguments` thiếu trường bắt buộc và xác nhận máy chủ từ chối nó trước khi thực thi. Sau đó, vượt qua một cuộc gọi với một trường không xác định bổ sung. Quyết định: người dẫn chương trình nên từ chối hay bỏ qua? Biện minh cho lựa chọn của bạn bằng một lập luận an toàn.

3. Phân loại từng công cụ trong harness là thuần túy hoặc do hậu quả. Thêm cờ `consequential: true` vào registry mục nhập cần nó và thay đổi vòng lặp để in dòng "sẽ xác nhận với người dùng" bất cứ khi nào một công cụ hậu quả được chọn. Đây là hình dạng của cổng xác nhận mà mọi production chủ nhà cần.

4. Vẽ vòng lặp bốn bước trên giấy với bảng cột nhà cung cấp ở trên được điền cho ứng dụng khách yêu thích của bạn (Claude Màn hình nền, Con trỏ, ChatGPT hoặc stack tùy chỉnh). Tham khảo chéo với biến thể dành riêng cho MCP trong Giai đoạn 13 · 06.

5. Đọc hướng dẫn gọi hàm của OpenAI từ trên xuống dưới. Xác định một trường nằm trong yêu cầu nhưng không nằm trong vòng lặp bốn bước như được trình bày ở đây. Giải thích những gì nó thêm vào và tại sao nó thuận tiện hơn là thiết yếu.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Công cụ | "Một thứ mà model có thể gọi" | Bộ ba tên + đầu vào gõ JSON Schema + chức năng thực thi |
| Function calling | "Sử dụng công cụ gốc" | Hỗ trợ API cấp nhà cung cấp để phát ra các cuộc gọi công cụ có cấu trúc thay vì văn xuôi |
| Lệnh gọi công cụ | "Yêu cầu hành động của model" | Một JSON payload có `id`, `name` `arguments` do model phát ra |
| Kết quả công cụ | "Công cụ trả lại gì" | Đầu ra của người thực thi, được bao bọc trong một thông báo vai trò `tool` với id phù hợp |
| Lệnh gọi công cụ song song | "Nhiều cuộc gọi cùng một lúc" | Nhiều đối tượng gọi trong một model lượt, độc lập và có thể sắp xếp theo id |
| Chế độ nghiêm ngặt | "Đảm bảo JSON" | Giải mã ràng buộc buộc đầu ra của model xác thực dựa trên schema đã khai báo |
| Công cụ thuần túy | "Công cụ chỉ đọc" | Không có tác dụng phụ; An toàn để chạy lại |
| Công cụ hậu quả | "Công cụ hành động" | Đột biến trạng thái bên ngoài; yêu cầu cổng, kiểm tra hoặc xác nhận người dùng |
| Vòng lặp bốn bước | "Chu kỳ gọi công cụ" | mô tả → quyết định → thực hiện → quan sát |
| Chủ nhà | "Agent runtime" | Chương trình giữ công cụ registry, gọi model và chạy trình thực thi |

## Đọc thêm

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — tham chiếu chuẩn cho các khai báo công cụ kiểu OpenAI và hình dạng cuộc gọi
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — Định dạng khối `tool_use` / `tool_result` của Claude
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) — ngữ nghĩa `functionDeclarations` và gọi song song trong Gemini
- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — tổng quát hóa giao diện công cụ bất khả tri của nhà cung cấp
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) - phương ngữ schema mà mọi công cụ hiện đại API nói
