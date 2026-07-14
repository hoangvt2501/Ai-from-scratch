# Xây dựng khách hàng MCP - Khám phá, Gọi Session Quản lý

> Hầu hết nội dung MCP ships server hướng dẫn và vẫy tay với khách hàng. Mã máy khách là nơi orchestration khó tồn tại: process sinh sản, đàm phán khả năng, hợp nhất danh sách công cụ trên nhiều servers, sampling callbacks, kết nối lại và giải quyết xung đột không gian tên. Bài học này xây dựng một ứng dụng đa server nâng ba MCP servers khác nhau vào một không gian tên công cụ phẳng cho model.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib, multi-server MCP client)
**Kiến thức tiên quyết:** Giai đoạn 13 · 07 (xây dựng MCP server)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Sinh ra một MCP server khi còn nhỏ process, hoàn thành `initialize` và gửi một `notifications/initialized`.
- Duy trì trạng thái mỗi server session (khả năng, danh sách công cụ, id thông báo nhìn thấy lần cuối).
- Merge công cụ liệt kê trên nhiều servers vào một không gian tên với tính năng xử lý xung đột.
- Định tuyến lệnh gọi công cụ đến server sở hữu nó và tập hợp lại phản hồi.

## Vấn đề

Một máy chủ agent thực sự (Claude Desktop, Cursor, Goose, Gemini CLI) tải nhiều MCP servers cùng một lúc. Người dùng có thể có server hệ thống tệp, server Postgres và GitHub server chạy đồng thời. Công việc của khách hàng:

1. Sinh sản mỗi server.
2. Bắt tay từng người một cách độc lập.
3. Gọi `tools/list` trên mỗi và làm phẳng kết quả.
4. Khi model phát ra `notes_search`, hãy tra cứu nó trong không gian tên merged và định tuyến đến server bên phải.
5. Xử lý thông báo từ bất kỳ server (`tools/list_changed`) nào mà không bị chặn.
6. Kết nối lại khi transport lỗi.

Lăn tay tất cả những điều đó là điều phân biệt "đồ chơi" với "có thể sử dụng được". Chính thức SDKs kết thúc điều này, nhưng model tinh thần phải là của bạn.

## Khái niệm

### Sinh sản process trẻ em

`subprocess.Popen` với `stdin=PIPE, stdout=PIPE, stderr=PIPE`. Đặt `bufsize=1` và sử dụng chế độ văn bản để đọc từng dòng. Mỗi server là một process; Khách hàng giữ một `Popen` tay cầm mỗi server.

### Tiểu bang trên mỗi server session

Một đối tượng `Session` trên mỗi server giữ:

- `process` - tay cầm Popen.
- `capabilities` - những gì server tuyên bố tại `initialize`.
- `tools` - kết quả `tools/list` cuối cùng.
- `pending` — bản đồ ID yêu cầu đến promise/future đang chờ phản hồi.

Các yêu cầu có bản chất không đồng bộ; một `tools/call` được gửi đến server A trong khi server B đang gọi giữa chừng, không được chặn. Sử dụng threads với hàng đợi hoặc không đồng bộ.

### Không gian tên Merged

Khi máy khách nhìn thấy danh sách công cụ tổng hợp, tên có thể va chạm. Hai servers đều có thể phơi bày `search`. Khách hàng có ba tùy chọn:

1. **Tiền tố bằng tên server.** `notes/search`, `files/search`. Rõ ràng nhưng xấu xí.
2. **Im lặng đến trước.** Sau đó, `search` của server ghi đè lên trước đó. Rủi ro; che giấu các va chạm.
3. **Từ chối va chạm.** Từ chối tải server thứ hai; Thông báo cho người dùng. An toàn nhất cho các máy chủ nhạy cảm về bảo mật.

Claude Máy tính để bàn sử dụng tiền tố theo server. Con trỏ sử dụng loại bỏ va chạm với một lỗi rõ ràng. VS Code MCP cũng sử dụng tiền tố theo server.

### Định tuyến

Sau khi hợp nhất, bảng điều phối ánh xạ `tool_name -> session`. model phát ra một cuộc gọi bằng tên; Khách hàng tìm thấy session và viết tin nhắn `tools/call` đến STDIN của server đó, sau đó chờ phản hồi.

### Sampling callback

Nếu server khai báo khả năng `sampling` ở `initialize`, nó có thể gửi `sampling/createMessage` yêu cầu máy khách chạy LLM của mình. Khách hàng phải:

1. Chặn các yêu cầu khác đối với server đó cho đến khi mẫu được giải quyết hoặc pipeline xem việc triển khai của mẫu có hỗ trợ đồng thời hay không.
2. Gọi cho nhà cung cấp LLM của nó.
3. Gửi phản hồi trở lại server.

Bài 11 bao gồm sampling từ đầu đến cuối. Bài học này sơ khai nó cho hoàn chỉnh.

### Xử lý thông báo

`notifications/tools/list_changed` có nghĩa là gọi lại `tools/list`. `notifications/resources/updated` có nghĩa là đọc lại tài nguyên nếu nó đang được sử dụng. Thông báo không được tạo ra phản hồi - đừng cố gắng phản hồi chúng.

Một lỗi phổ biến của máy khách: chặn vòng lặp đọc trên `tools/call` trong khi một thông báo nằm trong luồng. Sử dụng thread đọc nền đẩy mọi tin nhắn vào hàng đợi; thread chính là dequeues và gửi hàng.

### Kết nối lại

Transport có thể thất bại: server bị sập, hệ điều hành giết chết process, đường ống stdio bị vỡ. Khách hàng phát hiện EOF trên stdout và coi session là đã chết. Tùy chọn:

- Im lặng khởi động lại server và bắt tay lại. OK cho servers chỉ đọc thuần túy.
- Hiển thị lỗi cho người dùng. OK cho servers có trạng thái với sessions hiển thị của người dùng.

Giai đoạn 13 · 09 bao gồm ngữ nghĩa kết nối lại HTTP có thể phát trực tuyến; stdio đơn giản hơn.

### Keepalive và id session

HTTP có thể phát trực tuyến sử dụng tiêu đề `Mcp-Session-Id`. Stdio không có id session - danh tính process LÀ session. Ping Keepalive là tùy chọn; Ống STDIO không bị vỡ khi không hoạt động.

## Ứng dụng

`code/main.py` tạo ra ba MCP servers mô phỏng dưới dạng quy trình con, mỗi lần bắt tay merges danh sách công cụ của chúng và định tuyến các cuộc gọi công cụ đến đúng quy trình. Các "servers" thực sự là những người phản ứng đồ chơi chạy Python processes khác (không có LLM thực sự). Chạy nó để xem:

- Ba lần khởi tạo, mỗi lần có bộ khả năng riêng.
- Ba kết quả `tools/list` merged vào một không gian tên 7 công cụ.
- Quyết định định tuyến dựa trên tên công cụ.
- Một xung đột được ngăn chặn bởi tiền tố không gian tên.

Những gì cần xem:

- Lớp dữ liệu `Session` giữ trạng thái trên mỗi server một cách rõ ràng.
- Trình đọc nền thread loại bỏ mọi dòng trên stdout mà không chặn thread chính.
- Bảng điều phối là một `dict[str, Session]` đơn giản.
- Xử lý xung đột là rõ ràng: khi hai servers khai báo cùng một tên, người sau sẽ được đổi tên bằng tiền tố.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-mcp-client-harness.md`. Cho một danh sách khai báo các MCP servers (tên, lệnh, đối số), skill tạo ra một harness sinh ra chúng, merges danh sách công cụ và ships một hàm định tuyến với độ phân giải xung đột.

## Bài tập

1. Chạy `code/main.py` và xem nhật ký xuất hiện server. Giết một trong những server processes mô phỏng bằng SIGTERM và quan sát cách khách hàng phát hiện EOF và đánh dấu session đó là đã chết.

2. Triển khai tiền tố không gian tên. Khi hai servers hiển thị `search`, hãy đổi tên  thứ hai thành `<server>/search`. Cập nhật bảng điều phối và xác minh định tuyến lệnh gọi công cụ một cách chính xác.

3. Thêm backoff kiểu nhóm kết nối để khởi động lại server: backoff theo cấp số nhân khi các lỗi liên tiếp, giới hạn ở 30 giây, phát ra thông báo cho người dùng sau ba lần lỗi.

4. Phác thảo một máy khách hỗ trợ 100 MCP servers đồng thời. Cấu trúc dữ liệu nào thay thế lệnh điều phối đơn giản? (Gợi ý: trie cho khoảng cách tên tiền tố, cộng với số liệu cho số lượng công cụ trên server.)

5. Chuyển máy khách sang MCP Python SDK chính thức. SDK bọc `stdio_client` và `ClientSession`. Mã sẽ thu nhỏ từ ~200 dòng xuống ~40 dòng trong khi vẫn duy trì định tuyến nhiều server.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| MCP khách hàng | "Người dẫn chương trình agent" | Process tạo ra servers và điều phối các lệnh gọi công cụ |
| Session | "Tiểu bang trên mỗi server" | Khả năng, danh sách công cụ và sổ sách kế toán yêu cầu đang chờ xử lý |
| Không gian tên Merged | "Danh sách một công cụ" | Tập hợp tên công cụ phẳng trên tất cả các servers đang hoạt động |
| Xung đột không gian tên | "Hai servers cùng một công cụ" | Khách hàng phải có tiền tố, từ chối hoặc xuất hiện trước bản sao |
| Định tuyến | "Ai nhận được cuộc gọi này?" | Gửi từ tên công cụ đến sở hữu server |
| Trình đọc nền | "Không chặn stdout" | Thread hoặc nhiệm vụ rút server stdout vào hàng đợi |
| Sampling callback | "LLM dưới dạng dịch vụ" | Trình xử lý máy khách cho `sampling/createMessage` từ server |
| `notifications/*_changed` | "Primitive bị đột biến" | Báo hiệu khách hàng phải khám phá lại hoặc đọc lại |
| Kết nối lại policy | "Khi server chết" | Khởi động lại ngữ nghĩa khi transport không thành công |
| Stdio session | "Process = session" | Không có id session; Đứa trẻ process cuộc đời là session |

## Đọc thêm

- [Model Context Protocol — Client spec](https://modelcontextprotocol.io/specification/2025-11-25/client) — hành vi của khách hàng chuẩn
- [MCP — Quickstart client guide](https://modelcontextprotocol.io/quickstart/client) — hướng dẫn khách hàng hello-world với Python SDK
- [MCP Python SDK — client module](https://github.com/modelcontextprotocol/python-sdk) — `ClientSession` tham khảo và `stdio_client`
- [MCP TypeScript SDK — Client](https://github.com/modelcontextprotocol/typescript-sdk) — TS song song
- [VS Code — MCP in extensions](https://code.visualstudio.com/api/extension-guides/ai/mcp) — cách VS Code ghép kênh nhiều MCP servers trong một máy chủ chỉnh sửa duy nhất
