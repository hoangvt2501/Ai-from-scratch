# MCP Apps — Tài nguyên giao diện người dùng tương tác qua `ui://`

> Đầu ra của công cụ chỉ có văn bản giới hạn những gì agents có thể hiển thị. MCP Apps (SEP-1724, chính thức ngày 26 tháng 1 năm 2026) cho phép một công cụ trả về HTML tương tác sandbox được hiển thị nội tuyến trong Claude Desktop, ChatGPT, Cursor, Goose và VS Code. Bảng điều khiển, biểu mẫu, bản đồ, cảnh 3D, tất cả thông qua một tiện ích mở rộng. Bài học này hướng dẫn sơ đồ tài nguyên `ui://`, MIME `text/html;profile=mcp-app`, giao thức postMessage sandbox iframe và bề mặt bảo mật đi kèm với việc cho phép server hiển thị HTML.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib, UI resource emitter), HTML (sample app)
**Kiến thức tiên quyết:** Giai đoạn 13 · 07 (MCP server), Giai đoạn 13 · 10 (tài nguyên)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Trả về tài nguyên `ui://` từ lệnh gọi công cụ và đặt MIME và siêu dữ liệu chính xác.
- Khai báo giao diện người dùng được liên kết của công cụ với `_meta.ui.resourceUri`, `_meta.ui.csp` và `_meta.ui.permissions`.
- Triển khai iframe sandbox postMessage JSON-RPC để giao tiếp giữa giao diện người dùng với máy chủ.
- Áp dụng các giá trị mặc định CSP và permissions-policy để bảo vệ chống lại các cuộc tấn công có nguồn gốc từ giao diện người dùng.

## Vấn đề

Một công cụ `visualize_timeline` của kỷ nguyên 2025 có thể trả về "Đây là 14 nốt nhạc được sắp xếp theo thứ tự thời gian:...". Đó là một đoạn văn. Người dùng thực sự muốn dòng thời gian tương tác. Trước Ứng dụng MCP, các tùy chọn là: APIs tiện ích dành riêng cho ứng dụng khách (Claude artifacts, OpenAI GPT HTML tùy chỉnh) hoặc không có giao diện người dùng.

MCP Apps (SEP-1724, shipped ngày 26 tháng 1 năm 2026) chuẩn hóa hợp đồng. Kết quả công cụ chứa một `resource` có URI là `ui://...` và MIME của nó là `text/html;profile=mcp-app`. Máy chủ hiển thị nó trong một iframe sandbox với CSP hạn chế và không có quyền truy cập mạng trừ khi được cấp một cách rõ ràng. Giao diện người dùng bên trong iframe đăng tin nhắn đến máy chủ lưu trữ thông qua một phương ngữ postMessage nhỏ JSON-RPC.

Mọi ứng dụng khách tương thích (Claude Desktop, ChatGPT, Goose, VS Code) hiển thị cùng một tài nguyên `ui://` theo cùng một cách. Một server, một gói HTML, giao diện người dùng phổ quát.

## Khái niệm

### Sơ đồ tài nguyên `ui://`

Một công cụ trả về:

```json
{
  "content": [
    {"type": "text", "text": "Here is your notes timeline:"},
    {"type": "ui_resource", "uri": "ui://notes/timeline"}
  ],
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline",
      "csp": {
        "defaultSrc": "'self'",
        "scriptSrc": "'self' 'unsafe-inline'",
        "connectSrc": "'self'"
      },
      "permissions": []
    }
  }
}
```

Sau đó, máy chủ gọi `resources/read` trên URI `ui://notes/timeline` và nhận lại:

```json
{
  "contents": [{
    "uri": "ui://notes/timeline",
    "mimeType": "text/html;profile=mcp-app",
    "text": "<!doctype html>..."
  }]
}
```

### Iframe sandbox

Máy chủ hiển thị HTML bên trong `<iframe>` sandbox với:

- `sandbox="allow-scripts allow-same-origin"` (hoặc nghiêm ngặt hơn theo server khai báo)
- CSP được khai báo Server được áp dụng thông qua tiêu đề phản hồi.
- Không có cookie, không có localStorage từ nguồn gốc của máy chủ.
- Quyền truy cập mạng giới hạn ở `connectSrc` trong CSP.

### postGiao thức tin nhắn

iframe giao tiếp với máy chủ qua `window.postMessage`. Một phương ngữ JSON-RPC 2.0 nhỏ:

Luôn ghim `targetOrigin` vào nguồn gốc chính xác của ứng dụng ngang hàng và ở phía nhận xác thực `event.origin` dựa trên danh sách cho phép trước khi xử lý bất kỳ payload nào. Không bao giờ sử dụng `"*"` cho cả hai bên của kênh này - phần thân thực hiện các cuộc gọi công cụ và đọc tài nguyên.

```js
// iframe to host  (pin to host origin)
window.parent.postMessage({
  jsonrpc: "2.0",
  id: 1,
  method: "host.callTool",
  params: { name: "notes_update", arguments: { id: "note-14", title: "..." } }
}, "https://host.example.com");

// host to iframe  (pin to iframe origin)
iframe.contentWindow.postMessage({
  jsonrpc: "2.0",
  id: 1,
  result: { content: [...] }
}, "https://iframe.example.com");

// receiver on both sides
window.addEventListener("message", (event) => {
  if (event.origin !== "https://expected-peer.example.com") return;
  // safe to process event.data
});
```

Các phương thức phía máy chủ có sẵn mà giao diện người dùng có thể gọi:

- `host.callTool(name, arguments)` — gọi một công cụ server.
- `host.readResource(uri)` — đọc tài nguyên MCP.
- `host.getPrompt(name, arguments)` — tìm nạp một mẫu prompt.
- `host.close()` — loại bỏ giao diện người dùng.

Mọi cuộc gọi vẫn đi qua giao thức MCP và kế thừa quyền của server.

### Quyền

Danh sách `_meta.ui.permissions` yêu cầu các chức năng bổ sung:

- `camera` - truy cập máy ảnh của người dùng (được sử dụng để quét giao diện người dùng tài liệu).
- `microphone` - nhập liệu bằng giọng nói.
- `geolocation` - vị trí.
- `network:*` - truy cập mạng rộng hơn so với `connectSrc` cho phép.

Mỗi quyền là một prompt người dùng nhìn thấy trước khi giao diện người dùng hiển thị.

### Rủi ro bảo mật

HTML trong iframe vẫn HTML. Bề mặt tấn công mới:

- **Prompt-injection qua giao diện người dùng.** Giao diện người dùng server độc hại có thể hiển thị văn bản trông giống như thông báo hệ thống và lừa người dùng. Kết xuất máy chủ phải phân biệt rõ ràng giao diện người dùng server với giao diện người dùng máy chủ.
- **Trích xuất qua `connectSrc`.** Nếu CSP cho phép `connect-src: *`, giao diện người dùng có thể gửi dữ liệu đến bất cứ đâu. Mặc định phải nghiêm ngặt.
- **Clickjacking.** Giao diện người dùng phủ chrome máy chủ. Máy chủ phải ngăn thao túng chỉ mục z và thực thi các quy tắc về độ mờ.
- **Đánh cắp tiêu điểm.** Giao diện người dùng lấy nét bàn phím và ghi lại tin nhắn tiếp theo. Máy chủ phải chặn.

Giai đoạn 13 · 15 đề cập sâu đến những điều này như một phần của an ninh MCP; Bài học này giới thiệu chúng.

### `ui/initialize` bắt tay

Sau khi iframe tải, nó sẽ gửi `ui/initialize` qua postMessage:

```json
{"jsonrpc": "2.0", "id": 0, "method": "ui/initialize",
 "params": {"theme": "dark", "locale": "en-US", "sessionId": "..."}}
```

Máy chủ phản hồi bằng khả năng và session token. Giao diện người dùng sử dụng session token trên mỗi lệnh gọi máy chủ tiếp theo.

### AppRenderer / AppFrame SDK primitives

Các ứng dụng mở rộng SDK tiết lộ hai primitives tiện lợi:

- `AppRenderer` (bên server) — bao bọc một thành phần React / Vue / Solid và phát ra một tài nguyên `ui://` với MIME và siêu dữ liệu phù hợp.
- `AppFrame` (phía máy khách) — nhận tài nguyên, gắn iframe và trung gian postMessage.

Bạn có thể sử dụng những thứ này hoặc cuộn tay HTML và JSON-RPC.

### Trạng thái hệ sinh thái

Ứng dụng MCP shipped ngày 26 tháng 1 năm 2026. Hỗ trợ khách hàng tính đến tháng 4 năm 2026:

- **Claude Máy tính để bàn.** Hỗ trợ đầy đủ kể từ tháng 1 năm 2026.
- **ChatGPT.** Hỗ trợ đầy đủ thông qua Ứng dụng SDK (cùng giao thức Ứng dụng MCP cơ bản).
- **Con trỏ.** Beta; Bật thông qua cài đặt.
- **VS Code.** Chỉ xây dựng Insider.
- **Goose.** Hỗ trợ đầy đủ.
- **Zed, Windsurf.** Lộ trình.

Servers trong production: bảng thông tin, trực quan hóa bản đồ, bảng dữ liệu, trình tạo biểu đồ sandbox IDE xem trước.

## Ứng dụng

`code/main.py` mở rộng server ghi chú bằng công cụ `visualize_timeline` trả về tài nguyên `ui://notes/timeline`, cộng với trình xử lý cho `resources/read` trên URI đó trả về một gói HTML nhỏ nhưng đầy đủ với dòng thời gian SVG. HTML được tạo mẫu stdlib - không có hệ thống xây dựng. postMessage được phác thảo trong các bình luận JS vì stdlib không thể điều khiển trình duyệt.

Những gì cần xem:

- `_meta.ui` trên phản hồi của công cụ mang resourceUri, CSP, quyền.
- HTML hiển thị mà không cần truy cập mạng; Tất cả dữ liệu đều được nội tuyến.
- JS gọi `host.callTool` qua `window.parent.postMessage` (được ghi lại nhưng trơ trong bản demo stdlib này).

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-mcp-apps-spec.md`. Với một công cụ sẽ được hưởng lợi từ giao diện người dùng tương tác, skill tạo ra hợp đồng MCP Apps đầy đủ: `ui://` URI, CSP, quyền, điểm nhập postMessage và danh sách kiểm tra bảo mật.

## Bài tập

1. Chạy `code/main.py` và kiểm tra HTML phát ra. Mở HTML trực tiếp trong trình duyệt; xác minh kết xuất SVG. Sau đó, phác thảo hợp đồng postMessage mà giao diện người dùng sẽ sử dụng để gọi `host.callTool("notes_update", ...)`.

2. Siết chặt CSP: tháo `'unsafe-inline'` và sử dụng script policy dựa trên nonce. Những thay đổi nào trong mã thế hệ HTML?

3. Thêm `ui://notes/editor` tài nguyên giao diện người dùng thứ hai với biểu mẫu để chỉnh sửa ghi chú tại chỗ. Khi người dùng gửi, iframe sẽ gọi `host.callTool("notes_update", ...)`.

4. Kiểm tra bề mặt tấn công của giao diện người dùng. Một server độc hại có thể chèn nội dung ở đâu? iframe sandbox bảo vệ chống lại điều gì và nó không bảo vệ điều gì?

5. Đọc thông số kỹ thuật SEP-1724 và xác định một khả năng trong Ứng dụng MCP SDK mà triển khai đồ chơi này không sử dụng. (Gợi ý: đồng bộ hóa trạng thái cấp thành phần.)

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Ứng dụng MCP | "Tài nguyên giao diện người dùng tương tác" | SEP-1724 gia hạn shipped 2026-01-26 |
| `ui://` | "Lược đồ URI ứng dụng" | Lược đồ tài nguyên cho gói giao diện người dùng |
| `text/html;profile=mcp-app` | "Kịch câm" | Loại nội dung cho HTML ứng dụng MCP |
| Iframe sandbox | "Kết xuất container" | Sandbox trình duyệt của giao diện người dùng với CSP và quyền |
| postMessage JSON-RPC | "Dây giao diện người dùng đến máy chủ" | Phương ngữ JSON-RPC-over-postTin nhắn nhỏ cho các cuộc gọi máy chủ |
| `_meta.ui` | "Liên kết công cụ-giao diện người dùng" | Siêu dữ liệu liên kết kết quả công cụ với tài nguyên giao diện người dùng |
| CSP | "Bảo mật-Nội dung-Policy" | Khai báo các nguồn được phép cho scripts, mạng, kiểu |
| Trình kết xuất ứng dụng | "Server SDK primitive" | Chuyển đổi thành phần framework thành tài nguyên `ui://` |
| Khung ứng dụng | "Khách hàng SDK primitive" | Trình trợ giúp gắn kết Iframe trung gian postMessage |
| `ui/initialize` | "Bắt tay" | Bài đăng đầu tiênTin nhắn từ giao diện người dùng đến máy chủ |

## Đọc thêm

- [MCP ext-apps — GitHub](https://github.com/modelcontextprotocol/ext-apps) — triển khai và SDK tham khảo
- [MCP Apps specification 2026-01-26](https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx) - tài liệu thông số kỹ thuật chính thức
- [MCP — Apps extension overview](https://modelcontextprotocol.io/extensions/apps/overview) — tài liệu cấp cao
- [MCP blog — MCP Apps launch](https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/) - Bài ra mắt tháng 1 năm 2026
- [MCP Apps API reference](https://apps.extensions.modelcontextprotocol.io/api/) — Tham chiếu SDK kiểu JSDoc
