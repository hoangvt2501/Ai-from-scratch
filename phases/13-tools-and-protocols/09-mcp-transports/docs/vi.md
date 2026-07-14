# MCP Transports - stdio vs Streamable HTTP vs SSE Migration

> Stdio hoạt động tại địa phương và không ở đâu khác. HTTP có thể phát trực tuyến (26/03/2025) là tiêu chuẩn từ xa. HTTP+SSE transport cũ không còn được dùng nữa và sẽ bị xóa vào giữa năm 2026. Chọn sai transport sẽ tốn kém cho việc di chuyển; chọn đúng sẽ mua một MCP server có thể lưu trữ từ xa với tính liên tục session và bảo vệ liên kết lại DNS.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, Streamable HTTP endpoint skeleton)
**Kiến thức tiên quyết:** Giai đoạn 13 · 07, 08 (MCP server và khách hàng)
**Thời lượng:** ~45 phút

## Mục tiêu học tập

- Chọn giữa stdio và HTTP có thể phát trực tuyến dựa trên hình thức triển khai (cục bộ so với từ xa, một process so với nhóm).
- Triển khai mẫu một endpoint HTTP có thể phát trực tuyến: POST cho yêu cầu, GET cho session luồng.
- Thực thi xác thực `Origin` và ngữ nghĩa session-id để đánh bại liên kết lại DNS.
- Di chuyển HTTP+SSE server cũ sang HTTP có thể phát trực tuyến trước thời hạn xóa vào giữa năm 2026.

## Vấn đề

transport từ xa MCP đầu tiên (2024-11) là HTTP+SSE: hai endpoints, một kênh cho POST của khách hàng và một kênh Server-Sent-Events cho luồng server đến máy khách. Nó đã hoạt động. Nó cũng vụng về: hai endpoints mỗi session, bộ nhớ cache bị hỏng trước một số CDN và sự phụ thuộc khó khăn vào các kết nối SSE tồn tại lâu dài mà một số WAF chấm dứt một cách tích cực.

Thông số kỹ thuật 2025-03-26 đã thay thế nó bằng HTTP có thể phát trực tuyến: một endpoint, POST cho yêu cầu của khách hàng, GET để thiết lập luồng session, cả hai đều chia sẻ tiêu đề `Mcp-Session-Id`. Mọi server được xây dựng hoặc di chuyển kể từ đó đều sử dụng HTTP có thể phát trực tuyến. Chế độ SSE cũ đang không được dùng nữa - Atlassian Rovo đã loại bỏ nó vào ngày 30 tháng 6 năm 2026; Keboola ngày 1 tháng 4 năm 2026; hầu hết các doanh nghiệp còn lại servers đến cuối năm 2026.

Và stdio vẫn quan trọng đối với servers địa phương. Claude Desktop, VS Code và mọi ứng dụng khách hình IDE đều servers qua stdio. Các model tinh thần phù hợp: stdio cho "máy này", Streamable HTTP cho "qua mạng". Không giao nhau.

## Khái niệm

### STDIO

- Trẻ em process transport. Máy khách sinh server, giao tiếp qua stdin/stdout.
- Một đối tượng JSON trên mỗi dòng. Dòng mới được phân cách.
- Không có id session; Bản sắc process là session.
- Không cần xác thực (đứa trẻ kế thừa ranh giới tin cậy của cha mẹ).
- Không bao giờ sử dụng cho servers từ xa — bạn sẽ cần SSH hoặc socat để tạo đường hầm, tại thời điểm đó sử dụng HTTP có thể phát trực tuyến.

### HTTP có thể phát trực tuyến

Một endpoint `/mcp` (hoặc bất kỳ đường dẫn nào). Hỗ trợ ba phương pháp HTTP:

- **POST /mcp.** Máy khách gửi thông báo JSON-RPC. Server trả lời bằng một phản hồi JSON hoặc một luồng SSE của một hoặc nhiều phản hồi (hữu ích cho các phản hồi và thông báo hàng loạt liên quan đến yêu cầu đó).
- **GET /mcp.** Khách hàng mở một kênh SSE tồn tại lâu dài. Server sử dụng nó cho các yêu cầu server đến khách hàng (sampling, thông báo, elicitation).
- **DELETE /mcp.** Khách hàng chấm dứt session một cách rõ ràng.

Sessions được xác định bằng tiêu đề `Mcp-Session-Id` mà server đặt trên phản hồi đầu tiên và máy khách lặp lại trên mọi yêu cầu tiếp theo. Session id PHẢI ngẫu nhiên về mặt mật mã (128+ bit); ID do khách hàng chọn bị từ chối vì sự an toàn.

### Một endpoint so với hai

Chế độ hai endpoint từ thông số kỹ thuật cũ vẫn có thể gọi được vào năm 2026 - thông số kỹ thuật tuyên bố nó "tương thích kế thừa". Nhưng tất cả các servers mới nên là một endpoint. SDKs chính thức phát ra một endpoint; Chỉ sử dụng chế độ kế thừa khi nói chuyện với điều khiển từ xa chưa di chuyển.

### Xác thực `Origin` và liên kết lại DNS

Trình duyệt không phải là ứng dụng khách MCP (ngày nay), nhưng kẻ tấn công có thể tạo ra một trang web thuyết phục trình duyệt POST to `localhost:1234/mcp` - nơi MCP server cục bộ của người dùng lắng nghe. Nếu server không kiểm tra `Origin`, policy cùng nguồn gốc của trình duyệt sẽ không lưu vì `Origin: http://evil.com` là nguồn gốc chéo hợp lệ.

Thông số kỹ thuật 2025-11-25 yêu cầu servers từ chối các yêu cầu có `Origin` không có trong danh sách cho phép. Danh sách cho phép thường chứa các biến thể máy chủ máy khách MCP (`https://claude.ai`, `vscode-webview://*`) và máy chủ cục bộ cho giao diện người dùng cục bộ.

### Vòng đời id Session

1. Khách hàng gửi yêu cầu đầu tiên mà không cần `Mcp-Session-Id`.
2. Server gán một id ngẫu nhiên, đặt `Mcp-Session-Id` trên tiêu đề phản hồi.
3. Máy khách lặp lại tiêu đề đó trên tất cả các yêu cầu tiếp theo và trên `GET /mcp` cho luồng.
4. Session có thể bị thu hồi bởi server; Khách hàng nhìn thấy 404 trên các yêu cầu tiếp theo và phải khởi tạo lại.
5. Khách hàng có thể XÓA rõ ràng session để tắt máy sạch.

### Keepalive và kết nối lại

SSE kết nối bị ngắt. Khách hàng thiết lập lại bằng cách GETing lại với cùng một `Mcp-Session-Id`. Server PHẢI xếp hàng các sự kiện bị bỏ lỡ trong thời gian ngừng hoạt động (tối đa một khoảng thời gian hợp lý) và phát lại qua tiêu đề `last-event-id` mà máy khách lặp lại.

Giai đoạn 13 · 13 bao gồm Tasks, cho phép công việc chạy lâu dài tồn tại ngay cả khi kết nối lại toàn session.

### Đầu dò tương thích ngược

Một khách hàng muốn hỗ trợ cả những servers cũ và mới:

1. POST lên `/mcp`.
2. Nếu phản hồi `200 OK` với JSON hoặc SSE, đây là HTTP có thể phát trực tuyến.
3. Nếu phản hồi `200 OK` với tiêu đề `Content-Type: text/event-stream` AND `Location` trỏ đến endpoint phụ, thì đây là HTTP+SSE kế thừa; làm theo `Location`.

### Cloudflare, ngrok và lưu trữ

Production MCP servers từ xa vào năm 2026 chạy trên Cloudflare Workers (với MCP Agents SDK của chúng), Vercel Functions hoặc Node/Python. Key trong bộ chứa: lưu trữ của bạn phải hỗ trợ các kết nối HTTP tồn tại lâu dài cho SSE GET. Bậc miễn phí của Vercel giới hạn ở 10 giây và không phù hợp. Cloudflare Workers hỗ trợ luồng vô thời hạn.

### Thành phần Gateway

Khi bạn đứng trước nhiều MCP servers bằng một gateway (Giai đoạn 13 · 17), gateway là một HTTP endpoint có thể phát trực tuyến duy nhất ghi lại session id và ghép kênh ngược dòng. Các công cụ được merged ở lớp gateway; Khách hàng thấy một server logic duy nhất.

### Transport chế độ thất bại

- **stdio SIGPIPE.** Cái chết process trẻ em giữa chừng ghi làm tăng SIGPIPE; servers nên thoát ra sạch sẽ. Khách hàng nên phát hiện EOF và đánh dấu session đã chết.
- **HTTP 502/504.** Cloudflare, nginx và các proxy khác phát ra những thứ này khi lỗi ngược dòng. Các ứng dụng HTTP có thể phát trực tuyến nên thử lại một lần sau một thời gian ngắn.
- **SSE ngắt kết nối.** Thay đổi TCP RST, proxy timeout hoặc mạng máy khách sẽ đóng luồng. Khách hàng kết nối lại với `Mcp-Session-Id` và `last-event-id` tùy chọn để tiếp tục.
- **Session thu hồi.** Server làm mất hiệu lực ID session; Khách hàng thấy 404 trong yêu cầu tiếp theo. Khách hàng phải bắt tay lại.
- **Độ lệch đồng hồ.** Tính toán tài nguyên-TTL trên máy khách khác với server. Khách hàng nên coi dấu thời gian server là có thẩm quyền.

### Khi nào nên bỏ qua HTTP có thể phát trực tuyến

Một số doanh nghiệp triển khai MCP servers đằng sau gRPC hoặc transports hàng đợi tin nhắn bên trong mạng của riêng họ. Điều này là không chuẩn - thông số kỹ thuật của MCP không chính thức xác định những điều này. Gateways có thể hiển thị bề mặt HTTP có thể phát trực tuyến cho máy khách MCP trong khi sử dụng gRPC nội bộ. Giữ cho bề mặt bên ngoài tuân thủ thông số kỹ thuật; gateway sở hữu bản dịch.

## Ứng dụng

`code/main.py` triển khai HTTP endpoint có thể phát trực tuyến tối thiểu bằng cách sử dụng `http.server` (stdlib). Nó xử lý POST, GET và DELETE trên `/mcp`, đặt `Mcp-Session-Id` trên phản hồi đầu tiên, xác thực `Origin` và từ chối các yêu cầu từ các nguồn không nằm trong danh sách cho phép. Trình xử lý sử dụng lại ghi chú Bài học 07 server logic điều phối.

Những gì cần xem:

- Trình xử lý POST đọc nội dung JSON-RPC, gửi và ghi phản hồi JSON (biến thể phản hồi đơn; SSE biến thể có cấu trúc tương tự).
- Kiểm tra `Origin` từ chối đầu dò `http://evil.example` mặc định nhưng chấp nhận `http://localhost`.
- Session id là chuỗi hex 128 bit ngẫu nhiên; server giữ trạng thái mỗi session trong bộ nhớ.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-mcp-transport-migrator.md`. Với MCP server HTTP+SSE (cũ), skill tạo kế hoạch di chuyển sang HTTP có thể phát trực tuyến với tính liên tục session-id, kiểm tra nguồn gốc và hỗ trợ đầu dò tương thích ngược.

## Bài tập

1. Chạy `code/main.py`. ĐĂNG một `initialize` từ `curl` và quan sát tiêu đề phản hồi `Mcp-Session-Id`. ĐĂNG yêu cầu thứ hai lặp lại tiêu đề và xác minh tính liên tục session.

2. Thêm trình xử lý GET để mở luồng SSE. Gửi một sự kiện `notifications/progress` cứ sau năm giây. Kết nối lại bằng cách GET lại với cùng một id session và xác nhận server chấp nhận nó.

3. Triển khai logic phát lại `last-event-id`. Khi kết nối lại, phát lại bất kỳ sự kiện nào được tạo từ id đó.

4. Mở rộng xác thực `Origin` để hỗ trợ mẫu ký tự đại diện (`https://*.example.com`) và xác nhận mẫu đó chấp nhận `https://app.example.com` nhưng từ chối `https://evil.example.com.attacker.net`.

5. Lấy HTTP+SSE server cũ từ registry chính thức (có một số) và phác thảo quá trình di chuyển: những thay đổi nào trong cách xử lý endpoint, tạo id session và ngữ nghĩa tiêu đề.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| transport stdio | "Trẻ em địa phương process" | JSON-RPC trên stdin/stdout, được phân tách bằng dòng mới |
| HTTP có thể phát trực tuyến | "Điều khiển từ xa transport" | Một endpoint POST + GET + SSE tùy chọn, thông số kỹ thuật 2025-03-26 |
| HTTP+SSE | "Di sản" | Hai endpoint model được loại bỏ vào giữa năm 2026 |
| `Mcp-Session-Id` | "Tiêu đề Session" | ID ngẫu nhiên được chỉ định Server lặp lại trên mọi yêu cầu tiếp theo |
| `Origin` danh sách cho phép | "Bảo vệ liên kết lại DNS" | Từ chối các yêu cầu có Origin không được phê duyệt |
| Đơn endpoint | "Một URL" | `/mcp` xử lý POST / GET / DELETE cho tất cả các hoạt động session |
| `last-event-id` | "SSE phát lại" | Tiêu đề được sử dụng để tiếp tục sự kiện trực tiếp bị loại bỏ mà không bỏ lỡ sự kiện |
| Đầu dò tương thích ngược | "Phát hiện cũ và phát hiện mới" | Kiểm tra hình dạng phản hồi của khách hàng tự động chọn transport |
| Tuổi thọ cao HTTP | "SSE streaming" | Server đẩy các sự kiện trong vài phút hoặc hàng giờ trên một kết nối TCP |
| Thu hồi Session | "Buộc khởi tạo lại" | Server làm mất hiệu lực của id session; Khách hàng phải bắt tay một lần nữa |

## Đọc thêm

- [MCP — Basic transports spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports) — tham chiếu chuẩn cho stdio và HTTP có thể phát trực tuyến
- [MCP — Basic transports spec 2025-03-26](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports) — bản sửa đổi giới thiệu Streamable HTTP
- [Cloudflare — MCP transport](https://developers.cloudflare.com/agents/model-context-protocol/transport/) — Các mẫu HTTP có thể phát trực tuyến được lưu trữ Workers
- [AWS — MCP transport mechanisms](https://builder.aws.com/content/35A0IphCeLvYzly9Sw40G1dVNzc/mcp-transport-mechanisms-stdio-vs-streamable-http) — so sánh giữa các hình thức triển khai
- [Atlassian — HTTP+SSE deprecation notice](https://community.atlassian.com/forums/Atlassian-Remote-MCP-Server/HTTP-SSE-Deprecation-Notice/ba-p/3205484) — ví dụ cụ thể về thời hạn di chuyển
