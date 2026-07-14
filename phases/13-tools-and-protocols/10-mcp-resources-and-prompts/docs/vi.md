# MCP Tài nguyên và Prompts - Tiếp xúc ngữ cảnh ngoài các công cụ

> Các công cụ nhận được 90% MCP attention. Hai cách còn lại server primitives giải quyết các vấn đề khác nhau. Tài nguyên hiển thị dữ liệu để đọc; prompts hiển thị các mẫu có thể tái sử dụng dưới dạng lệnh gạch chéo. Nhiều servers nên sử dụng tài nguyên thay vì bao bọc các bài đọc trong các công cụ và prompts thay vì quy trình làm việc mã hóa cứng trong prompts máy khách. Bài học này đặt tên cho quy tắc quyết định và thực hiện các thông điệp `resources/*` và `prompts/*`.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib, resource + prompt handler)
**Kiến thức tiên quyết:** Giai đoạn 13 · 07 (MCP server)
**Thời lượng:** ~45 phút

## Mục tiêu học tập

- Quyết định giữa việc hiển thị một khả năng dưới dạng công cụ, tài nguyên hoặc prompt cho một miền nhất định.
- Thực hiện `resources/list`, `resources/read`, `resources/subscribe` và xử lý `notifications/resources/updated`.
- Triển khai `prompts/list` và `prompts/get` với các mẫu đối số.
- Nhận biết khi máy chủ xuất hiện prompts dưới dạng lệnh gạch chéo so với ngữ cảnh tự động chèn.

## Vấn đề

Một MCP server ngây thơ cho một ứng dụng ghi chú hiển thị mọi thứ dưới dạng công cụ: `notes_read`, `notes_list` `notes_search`. Điều này bao bọc mọi quyền truy cập dữ liệu trong một lệnh gọi công cụ dựa trên model. Hậu quả:

- model phải quyết định có gọi `notes_read` cho mọi truy vấn có thể được hưởng lợi từ ngữ cảnh hay không.
- Không thể đăng ký hoặc phát trực tuyến nội dung chỉ đọc lên bảng điều khiển bên của máy chủ.
- Giao diện người dùng máy khách (Claude bảng đính kèm tài nguyên của Máy tính để bàn, bộ chọn "Bao gồm tệp" của Con trỏ) không thể hiển thị dữ liệu.

Phân chia phù hợp: hiển thị dữ liệu dưới dạng tài nguyên, hiển thị các hành động đột biến hoặc được tính toán dưới dạng công cụ, hiển thị quy trình làm việc nhiều bước có thể tái sử dụng dưới dạng prompts. Mỗi primitive đều có khả năng chi trả UX và mô hình truy cập của nó.

## Khái niệm

### Công cụ vs tài nguyên so với prompts - quy tắc quyết định

| Khả năng | Primitive |
|------------|-----------|
| Người dùng muốn tìm kiếm, lọc hoặc chuyển đổi dữ liệu | Công cụ |
| Người dùng muốn máy chủ bao gồm dữ liệu này làm ngữ cảnh | Tài nguyên |
| Người dùng muốn có một quy trình làm việc theo mẫu mà họ có thể chạy lại | prompt |

Hướng dẫn: nếu model sẽ được hưởng lợi từ việc gọi nó trên mọi truy vấn liên quan, thì đó là một công cụ. Nếu người dùng sẽ được hưởng lợi từ việc đính kèm nó vào một cuộc trò chuyện, thì đó là một tài nguyên. Nếu toàn bộ quy trình làm việc nhiều bước là đơn vị mà người dùng muốn sử dụng lại, thì đó là một prompt.

### Tài nguyên

`resources/list` trả về `{resources: [{uri, name, mimeType, description?}]}`. `resources/read` lấy `{uri}` và trả lại `{contents: [{uri, mimeType, text | blob}]}`.

URI có thể là bất cứ thứ gì có thể định địa chỉ:

- `file:///Users/alice/notes/mcp.md`
- `postgres://my-db/query/SELECT ...`
- `notes://note-14` (lược đồ tùy chỉnh)
- `memory://session-2026-04-22/recent` (dành riêng cho server)

`contents[]` hỗ trợ cả văn bản và nhị phân. Nhị phân sử dụng `blob` như một chuỗi được mã hóa base64 cộng với một `mimeType`.

### Đăng ký tài nguyên

Khai báo `{resources: {subscribe: true}}` về khả năng. Khách hàng gọi `resources/subscribe {uri}`. Server gửi `notifications/resources/updated {uri}` khi tài nguyên thay đổi. Khách hàng đọc lại.

Trường hợp sử dụng: một server ghi chú có tài nguyên là tệp trên đĩa; trình theo dõi tệp triggers thông báo cập nhật; Claude Máy tính để bàn kéo lại tệp vào ngữ cảnh khi được chỉnh sửa bên ngoài máy chủ.

### Mẫu tài nguyên (bổ sung 2025-11-25)

`resourceTemplates` cho phép bạn hiển thị mẫu URI được tham số: `notes://{id}` với `id` làm mục tiêu hoàn thành. Máy khách có thể tự động hoàn thành id trong bộ chọn tài nguyên.

### Prompts

`prompts/list` trả về `{prompts: [{name, description, arguments?}]}`. `prompts/get` lấy `{name, arguments}` và trả lại `{description, messages: [{role, content}]}`.

prompt là một mẫu điền vào danh sách các tin nhắn mà máy chủ cung cấp cho model của mình. Ví dụ: một `code_review` prompt lấy một đối số `file_path` và trả về một chuỗi ba tin nhắn: một thông báo hệ thống, một tin nhắn người dùng với nội dung tệp và một trợ lý khởi động với một mẫu suy luận.

### Chủ nhà và prompts

Claude Desktop, VS Code và Cursor hiển thị prompts dưới dạng lệnh gạch chéo trong giao diện người dùng trò chuyện. Người dùng nhập `/code_review` và chọn đối số từ biểu mẫu. prompt của server là hợp đồng giữa "phím tắt người dùng" và "prompt đầy đủ được gửi đến model".

Không phải khách hàng nào cũng hỗ trợ prompts - kiểm tra đàm phán năng lực. Một server có khả năng prompt được khai báo nhưng một máy khách không có hỗ trợ prompt đơn giản là sẽ không thấy các lệnh gạch chéo.

### Thông báo "danh sách đã thay đổi"

Cả tài nguyên và prompts đều phát ra `notifications/list_changed` khi tập hợp đột biến. Một server ghi chú vừa nhập 20 tờ tiền mới phát ra `notifications/resources/list_changed`; Khách hàng gọi lại `resources/list` để nhận các bổ sung.

### Quy ước kiểu nội dung

Đối với văn bản: `mimeType: "text/plain"`, `text/markdown`, `application/json`.
Đối với nhị phân: `image/png`, `application/pdf`, cộng với trường `blob`.
Đối với ứng dụng MCP (Bài 14): `text/html;profile=mcp-app` trong URI `ui://`.

### Tài nguyên động

URI tài nguyên không nhất thiết phải tương ứng với tệp tĩnh. `notes://recent` có thể trả về năm ghi chú mới nhất cho mỗi lần đọc. `db://query/users/active` có thể thực hiện một truy vấn được tham số hóa. server được tự do tính toán nội dung động.

Quy tắc: nếu máy khách có thể lưu vào bộ nhớ đệm bằng URI, URI phải ổn định. Nếu tính toán one-shot, URI phải bao gồm dấu thời gian hoặc nonce để bộ nhớ đệm của máy khách không bị cũ.

### Đăng ký so với thăm dò ý kiến

Khách hàng có khả năng đăng ký nhận được server thúc đẩy thông qua `notifications/resources/updated`. Các khách hàng hoặc máy chủ đăng ký trước không hỗ trợ nó thăm dò ý kiến bằng cách đọc lại. Cả hai đều tuân thủ thông số kỹ thuật. Khai báo khả năng của server cho khách hàng biết nó hỗ trợ gì.

Chi phí đăng ký: mỗi tiểu bang session trên server (ai đăng ký cái gì). Giữ giới hạn tập hợp đã đăng ký; Máy khách bị ngắt kết nối sẽ hết thời gian chờ.

### Prompts so với hệ thống prompts

Prompts trong MCP không phải là prompts hệ thống. system prompt của máy chủ (hướng dẫn vận hành riêng) và MCP prompts (các mẫu do người dùng gọi server) của máy chủ nằm cạnh nhau. Một khách hàng cư xử tốt không bao giờ để server prompt ghi đè lên system prompt của chính mình; nó xếp lớp chúng.

## Ứng dụng

`code/main.py` mở rộng các ghi chú server từ Bài 07 với:

- Tài nguyên cho mỗi ghi chú (`notes://note-1`, v.v.) với sự hỗ trợ `resources/subscribe`.
- Một `review_note` prompt hiển thị thành mẫu ba tin nhắn.
- Mô phỏng trình xem tệp phát ra `notifications/resources/updated` khi ghi chú được sửa đổi.
- Một tài nguyên động `notes://recent` luôn trả về năm ghi chú mới nhất.

Chạy bản demo để xem toàn bộ quy trình.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-primitive-splitter.md`. Với một MCP server được đề xuất, skill phân loại từng khả năng là công cụ / tài nguyên / prompt với lý do.

## Bài tập

1. Chạy `code/main.py`. Quan sát danh sách tài nguyên ban đầu, sau đó trigger chỉnh sửa ghi chú và xác minh sự kiện `notifications/resources/updated` kích hoạt.

2. Thêm bộ phát `resources/list_changed`: khi một ghi chú mới được tạo, hãy gửi thông báo để khách hàng khám phá lại.

3. Thiết kế ba prompts cho một GitHub MCP server: `summarize_pr`, `triage_issue`, `release_notes`. Mỗi người có lập luận schemas. Phần thân prompt có thể chạy được mà không cần chỉnh sửa thêm.

4. Lấy một công cụ hiện có trong bài học 07 server và phân loại xem nó nên vẫn là một công cụ hay được chia thành một cặp công cụ cộng với tài nguyên. Biện minh trong một câu.

5. Đọc phần `server/resources` và `server/prompts` của thông số kỹ thuật. Xác định một trường trong `resources/read` hiếm khi được điền nhưng được hỗ trợ thông số kỹ thuật. Gợi ý: xem `_meta` về nội dung tài nguyên.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Tài nguyên | "Dữ liệu bị lộ" | Nội dung có thể định địa chỉ URI mà máy chủ có thể đọc |
| URI tài nguyên | "Con trỏ đến dữ liệu" | Mã định danh có tiền tố lược đồ (`file://`, `notes://`, v.v.) |
| `resources/subscribe` | "Theo dõi những thay đổi" | Cập nhật đẩy server chọn sử dụng khách cho một URI cụ thể |
| `notifications/resources/updated` | "Tài nguyên đã thay đổi" | Báo hiệu cho khách hàng rằng tài nguyên đã đăng ký có nội dung mới |
| Mẫu tài nguyên | "URI được tham số hóa" | Mẫu URI với gợi ý hoàn thành cho bộ chọn máy chủ |
| Prompt | "Mẫu lệnh gạch chéo" | Mẫu nhiều tin nhắn được đặt tên với các vị trí đối số |
| Prompt đối số | "Đầu vào mẫu" | Đã nhập parameters máy chủ thu thập trước khi hiển thị |
| `prompts/get` | "Mẫu kết xuất" | Server trả về danh sách tin nhắn đã điền |
| Khối nội dung | "Khối được đánh máy" | '{type: văn bản \ | Hình ảnh \ | Tài nguyên \ | ui_resource}' |
| UX lệnh gạch chéo | "Phím tắt người dùng" | Bề mặt máy chủ prompts dưới dạng lệnh bắt đầu bằng `/` |

## Đọc thêm

- [MCP — Concepts: Resources](https://modelcontextprotocol.io/docs/concepts/resources) — URI tài nguyên, đăng ký và mẫu
- [MCP — Concepts: Prompts](https://modelcontextprotocol.io/docs/concepts/prompts) — prompt mẫu và tích hợp lệnh gạch chéo
- [MCP — Server resources spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/resources) — tham chiếu thông báo `resources/*` đầy đủ
- [MCP — Server prompts spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts) — tham chiếu thông báo `prompts/*` đầy đủ
- [MCP — Protocol info site: resources](https://modelcontextprotocol.info/docs/concepts/resources/) — Hướng dẫn cộng đồng mở rộng trên tài liệu chính thức
