# MCP Nguyên tắc cơ bản - Cơ sở Primitives, Vòng đời JSON-RPC

> Mọi tích hợp trước MCP là một lần. Giao thức ngữ cảnh Model, lần đầu tiên được Anthropic shipped vào tháng 11 năm 2024 và hiện được quản lý bởi Agentic AI Foundation của Linux Foundation, tiêu chuẩn hóa việc khám phá và gọi để bất kỳ khách hàng nào cũng có thể nói chuyện với bất kỳ server nào. Thông số kỹ thuật 2025-11-25 nêu tên sáu primitives (ba server, ba máy khách), vòng đời ba pha và định dạng dây JSON-RPC 2.0. Tìm hiểu những điều đó và rest của chương MCP của giai đoạn này trở thành đọc.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, JSON-RPC parser)
**Kiến thức tiên quyết:** Giai đoạn 13 · 01 đến 05 (giao diện công cụ và function calling)
**Thời lượng:** ~45 phút

## Mục tiêu học tập

- Đặt tên cho tất cả sáu MCP primitives (công cụ, tài nguyên, prompts trên server; gốc, sampling elicitation trên máy khách) và đưa ra một trường hợp sử dụng cho mỗi người.
- Đi qua vòng đời ba giai đoạn (khởi tạo, vận hành, tắt máy) và cho biết ai gửi tin nhắn nào ở mỗi giai đoạn.
- Phân tích cú pháp và phát phong bì yêu cầu, phản hồi và thông báo JSON-RPC 2.0.
- Giải thích đàm phán năng lực tại `initialize` là gì và điều gì sẽ gặp lỗi nếu không có nó.

## Vấn đề

Trước MCP, mọi agent sử dụng công cụ đều có giao thức riêng. Cursor có một hệ thống công cụ hình MCP nhưng không tương thích. Claude shipped máy tính để bàn với một  khác. Tiện ích mở rộng Copilot của VS Code có phần thứ ba. Một nhóm đã xây dựng công cụ "Truy vấn Postgres" đã viết cùng một công cụ ba lần, mỗi lần đến API của một máy chủ khác nhau. Việc sử dụng lại nó yêu cầu sao chép mã.

Kết quả là một sự bùng nổ kỷ Cambri của các tích hợp một lần và mức trần về vận tốc hệ sinh thái.

MCP khắc phục điều này bằng cách chuẩn hóa định dạng dây. Một MCP server duy nhất hoạt động trong mọi MCP khách hàng: Claude Desktop, ChatGPT, Cursor, VS Code, Gemini, Goose, Zed, Windsurf, 300+ khách hàng vào tháng 4 năm 2026. 110 triệu lượt tải xuống SDK hàng tháng. 10.000+ servers công cộng. Quỹ Linux đảm nhận quyền quản lý vào tháng 12 năm 2025 dưới Quỹ Agentic AI mới.

Bản sửa đổi thông số kỹ thuật được sử dụng trong giai đoạn này là **2025-11-25**. Nó bổ sung Tác vụ không đồng bộ (SEP-1686), elicitation chế độ URL (SEP-1036), sampling với các công cụ (SEP-1577), sự đồng ý phạm vi gia tăng (SEP-835) và ngữ nghĩa chỉ báo tài nguyên OAuth 2.1. Giai đoạn 13 · 09 đến 16 bao gồm các phần mở rộng đó. Bài học này dừng lại ở cơ sở.

## Khái niệm

### Ba server primitives

1. **Công cụ.** Hành động có thể gọi. Vòng lặp bốn bước tương tự từ Giai đoạn 13 · 01.
2. **Tài nguyên.** Dữ liệu bị lộ. Nội dung chỉ đọc có thể định địa chỉ bằng URI: `file:///path`, `db://query/...`, lược đồ tùy chỉnh.
3. **Prompts.** Mẫu có thể tái sử dụng. Lệnh gạch chéo trong giao diện người dùng máy chủ; server cung cấp mẫu, khách hàng điền vào các đối số.

### Ba primitives khách hàng

4. **Roots.** Bộ URI mà server được phép chạm vào. Khách hàng khai báo chúng; server tôn trọng họ.
5. **Sampling.** Server yêu cầu model của khách hàng thực hiện hoàn thành. Cho phép các vòng lặp agent được lưu trữ server mà không cần phím API phía server.
6. **Elicitation.** Server yêu cầu người dùng của khách hàng cung cấp đầu vào có cấu trúc giữa chuyến bay. Biểu mẫu hoặc URL (SEP-1036).

Mọi khả năng trong MCP đều thuộc về chính xác một trong sáu khả năng này. Giai đoạn 13 · 10 đến 14 bao gồm mỗi cái một chiều sâu.

### Định dạng dây: JSON-RPC 2.0

Mỗi tin nhắn là một đối tượng JSON với các trường sau:

- Yêu cầu: `{jsonrpc: "2.0", id, method, params}`.
- Câu trả lời: `{jsonrpc: "2.0", id, result | error}`.
- Thông báo: `{jsonrpc: "2.0", method, params}` - không có `id`, không có phản hồi.

Thông số cơ sở có ~15 phương pháp, được nhóm theo primitive. Những điều quan trọng:

- `initialize` / `initialized` (bắt tay)
- `tools/list`, `tools/call`
- `resources/list`, `resources/read`, `resources/subscribe`
- `prompts/list`, `prompts/get`
- `sampling/createMessage` (server đến khách hàng)
- `notifications/tools/list_changed`, `notifications/resources/updated`, `notifications/progress`

### Vòng đời ba pha

**Giai đoạn 1: khởi tạo.**

Khách hàng gửi `initialize` với `capabilities` và `clientInfo` của nó. Server phản hồi bằng `capabilities`, `serverInfo` và phiên bản thông số kỹ thuật riêng mà nó nói. Khách hàng gửi `notifications/initialized` khi nó đã tiêu hóa phản hồi. Từ đây trở đi, một trong hai bên có thể gửi yêu cầu theo khả năng thương lượng.

**Giai đoạn 2: hoạt động.**

Hai chiều. Khách hàng gọi `tools/list` để khám phá, sau đó `tools/call` để gọi. Server có thể gửi `sampling/createMessage` nếu tuyên bố khả năng đó. Server có thể gửi `notifications/tools/list_changed` khi bộ công cụ của nó đột biến. Khách hàng có thể gửi `notifications/roots/list_changed` khi người dùng thay đổi phạm vi gốc.

**Giai đoạn 3: tắt máy.**

Một trong hai bên đóng transport. Không có phương pháp tắt có cấu trúc trong MCP; transport (stdio hoặc Streamable HTTP, Giai đoạn 13 · 09) mang tín hiệu kết thúc kết nối.

### Đàm phán năng lực

`capabilities` trong cái bắt tay `initialize` là hợp đồng. Ví dụ từ một server:

```json
{
  "tools": {"listChanged": true},
  "resources": {"subscribe": true, "listChanged": true},
  "prompts": {"listChanged": true}
}
```

server tuyên bố nó có thể phát ra thông báo `tools/list_changed` và hỗ trợ `resources/subscribe`. Khách hàng đồng ý bằng cách tuyên bố của riêng mình:

```json
{
  "roots": {"listChanged": true},
  "sampling": {},
  "elicitation": {}
}
```

Nếu khách hàng không khai báo `sampling` thì server không được gọi `sampling/createMessage`. Đối xứng: nếu server không khai báo `resources.subscribe`, khách hàng không được cố gắng đăng ký.

Đây là điều ngăn chặn sự trôi dạt của hệ sinh thái. Khách hàng không hỗ trợ sampling vẫn là khách hàng MCP hợp lệ; Một server không gọi `sampling` vẫn là một MCP server hợp lệ. Họ chỉ không sử dụng feature đó cùng nhau.

### Nội dung có cấu trúc và hình dạng lỗi

`tools/call` trả về một mảng `content` các khối được nhập: `text`, `image`, `resource`. Giai đoạn 13 · 14 thêm MCP Ứng dụng (`ui://` giao diện người dùng tương tác) vào danh sách đó.

Lỗi sử dụng mã lỗi JSON-RPC. Các bổ sung được xác định theo thông số kỹ thuật: `-32002` "Không tìm thấy tài nguyên" `-32603` "Lỗi nội bộ", cộng với dữ liệu lỗi dành riêng cho MCP dưới dạng `error.data`.

### Khả năng của máy khách so với chi tiết lệnh gọi công cụ

Một nhầm lẫn phổ biến: `capabilities.tools` là liệu máy khách có hỗ trợ thông báo thay đổi danh sách công cụ hay không. Liệu máy khách SẼ gọi các công cụ cụ thể hay không là một lựa chọn runtime được thúc đẩy bởi model của nó, không phải là cờ khả năng. Cờ khả năng là hợp đồng cấp thông số kỹ thuật. Sự lựa chọn của model là trực giao.

### Tại sao lại JSON-RPC mà không phải REST?

JSON-RPC 2.0 (2010) là một giao thức hai chiều nhẹ. REST do khách hàng khởi tạo. MCP cần thông báo do server khởi tạo (sampling, thông báo), vì vậy JSON-RPC với hình dạng request/response đối xứng của nó là một sự phù hợp tự nhiên. JSON-RPC cũng sáng tác rõ ràng trên stdio và WebSocket/Streamable HTTP mà không cần phát minh lại hình dạng yêu cầu của HTTP.

```figure
mcp-tool-call
```

## Ứng dụng

`code/main.py` ships một trình phân tích cú pháp và bộ phát JSON-RPC 2.0 tối thiểu, sau đó thực hiện trình tự `initialize` → `tools/list` → `tools/call` → `shutdown` bằng tay, in mọi tin nhắn. Không có transport thực sự; chỉ có hình dạng tin nhắn. So sánh với thông số kỹ thuật được liên kết trong Đọc thêm để xác minh từng phong bì.

Những gì cần xem:

- `initialize` tuyên bố khả năng theo cả hai cách; Phản ứng có `serverInfo` và `protocolVersion: "2025-11-25"`.
- `tools/list` trả về một mảng `tools`; Mỗi mục đều có `name`, `description`, `inputSchema`.
- `tools/call` sử dụng `params.name` và `params.arguments`.
- Phản hồi `content` là một mảng các khối `{type, text}`.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-mcp-handshake-tracer.md`. Với bản ghi kiểu pcap của tương tác server khách MCP, skill chú thích từng tin nhắn với primitive nào, giai đoạn vòng đời nào và khả năng mà nó phụ thuộc vào.

## Bài tập

1. Chạy `code/main.py`. Xác định ranh giới nơi đàm phán năng lực diễn ra và mô tả điều gì sẽ thay đổi nếu server không tuyên bố `tools.listChanged`.

2. Mở rộng trình phân tích cú pháp để xử lý `notifications/progress`. Hình dạng thông điệp: `{method: "notifications/progress", params: {progressToken, progress, total}}`. Phát ra nó trong khi `tools/call` chạy lâu đang diễn ra và xác nhận trình xử lý máy khách sẽ hiển thị thanh tiến trình.

3. Đọc thông số kỹ thuật MCP 2025-11-25 từ trên xuống dưới - toàn bộ tài liệu dài khoảng 80 trang. Xác định một cờ khả năng mà hầu hết servers KHÔNG cần. Gợi ý: nó liên quan đến đăng ký tài nguyên.

4. Phác thảo trên giấy primitive một "công việc cron" giả định mà feature sẽ thuộc về. (Gợi ý: server muốn khách hàng gọi nó vào một thời điểm đã định. Không ai trong số sáu primitives phù hợp với ngày hôm nay.) Lộ trình năm 2026 của MCP có dự thảo SEP cho việc này.

5. Phân tích cú pháp một session nhật ký từ một MCP server mở trên GitHub. Đếm yêu cầu so với phản hồi so với tin nhắn thông báo. Tính toán phần nào của lưu lượng truy cập là vòng đời so với hoạt động.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| MCP | "Giao thức ngữ cảnh Model" | Giao thức mở để khám phá và gọi model đến công cụ |
| Server primitive | "Thật là một server phơi bày" | Công cụ (hành động), tài nguyên (dữ liệu), prompts (mẫu) |
| Khách hàng primitive | "Những gì khách hàng cho phép servers sử dụng" | Gốc (phạm vi), sampling (LLM callbacks), elicitation (đầu vào của người dùng) |
| JSON-RPC 2.0 | "Định dạng dây" | Phong bì request/response/notification đối xứng |
| `initialize` bắt tay | "Đàm phán năng lực" | Cặp tin nhắn đầu tiên; servers và khách hàng khai báo features họ hỗ trợ |
| `tools/list` | "Khám phá" | Khách hàng yêu cầu server cung cấp bộ công cụ hiện tại của nó |
| `tools/call` | "Lời cầu nguyện" | Client yêu cầu server thực thi một công cụ có đối số |
| `notifications/*_changed` | "Sự kiện đột biến" | Server cho khách hàng biết rằng danh sách primitive của họ đã thay đổi |
| Khối nội dung | "Kết quả đã nhập" | '{type: "văn bản" \ | "hình ảnh" \ | "tài nguyên" \ | "ui_resource"}' trong kết quả công cụ |
| THÁNG CHÍN | "Đề xuất phát triển thông số kỹ thuật" | Đề xuất dự thảo được đặt tên (ví dụ: SEP-1686 cho Tác vụ không đồng bộ) |

## Đọc thêm

- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — tài liệu thông số kỹ thuật chuẩn
- [Model Context Protocol — Architecture concepts](https://modelcontextprotocol.io/docs/concepts/architecture) - model tinh thần sáu primitive
- [Anthropic — Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) - Bài ra mắt Tháng Mười Một 2024
- [MCP blog — First MCP anniversary](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) - hồi tưởng một năm và thay đổi thông số kỹ thuật 2025-11-25
- [WorkOS — MCP 2025-11-25 spec update](https://workos.com/blog/mcp-2025-11-25-spec-update) — tóm tắt SEP-1686, 1036, 1577, 835 và 1724
