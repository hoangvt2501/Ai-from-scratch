# MCP Security II — OAuth 2.1, Chỉ báo tài nguyên, Phạm vi gia tăng

> MCP servers từ xa cần authorization, không chỉ xác thực. Thông số kỹ thuật 2025-11-25 phù hợp với OAuth 2.1 + PKCE + chỉ báo tài nguyên (RFC 8707) + siêu dữ liệu tài nguyên được bảo vệ (RFC 9728). SEP-835 bổ sung sự đồng ý phạm vi gia tăng với authorization bước lên trên 403 WWW-Authenticate. Bài học này triển khai luồng bước lên dưới dạng máy trạng thái để bạn có thể xem mọi bước nhảy.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib, OAuth state machine simulator)
**Kiến thức tiên quyết:** Giai đoạn 13 · 09 (transports), Giai đoạn 13 · 15 (bảo mật I)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Phân biệt server nguồn lực với trách nhiệm authorization server.
- Thực hiện quy trình mã OAuth 2.1 authorization được bảo vệ bởi PKCE.
- Sử dụng `resource` (RFC 8707) và siêu dữ liệu tài nguyên được bảo vệ (RFC 9728) để ngăn chặn các cuộc tấn công gây nhầm lẫn.
- Thực hiện authorization bước lên: server trả lời 403 bằng WWW-Authenticate yêu cầu phạm vi cao hơn; Khách hàng prompts lại sự đồng ý của người dùng và thử lại.

## Vấn đề

Đầu MCP (trước năm 2025) shipped servers từ xa với các phím API đặc biệt hoặc thậm chí không cần xác thực. Thông số kỹ thuật 2025-11-25 thu hẹp khoảng cách đó với cấu hình OAuth 2.1 đầy đủ.

Ba nhu cầu thực tế:

- **servers từ xa thông thường.** Người dùng cài đặt một MCP server từ xa truy cập vào Notion / GitHub / Gmail của họ. OAuth 2.1 với PKCE là hình dạng phù hợp.
- **Leo thang phạm vi.** Một ghi chú server được cấp `notes:read` sau này có thể cần `notes:write` cho một hành động cụ thể. Thay vì làm lại toàn bộ luồng, step-up (SEP-835) yêu cầu phạm vi bổ sung.
- **Ngăn chặn phó bối rối.** Khách hàng giữ token phạm vi đối tượng cho Server A. Server A độc hại và cố gắng trình bày token cho Server B. Các chỉ báo tài nguyên (RFC 8707) ghim token cho đối tượng dự định của nó.

OAuth 2.1 không phải là mới. Điểm mới là hồ sơ của MCP: các luồng bắt buộc cụ thể (chỉ mã authorization + PKCE; không ngầm, không có thông tin đăng nhập khách hàng theo mặc định), các chỉ báo tài nguyên bắt buộc trên mọi yêu cầu token và siêu dữ liệu tài nguyên được bảo vệ được xuất bản để khách hàng biết phải đi đâu.

## Khái niệm

### Vai trò

- **Máy khách.** Máy khách MCP (Claude Máy tính để bàn, Con trỏ, v.v.).
- **Tài nguyên server.** MCP server (ghi chú, GitHub, Postgres, bất cứ thứ gì).
- **Authorization server.** Các vấn đề tokens. Có thể là cùng một dịch vụ với server tài nguyên hoặc một IdP riêng biệt (Auth0, Keycloak, Cognito).

Trong hồ sơ của MCP, tài nguyên và authorization servers CÓ THỂ là cùng một máy chủ nhưng NÊN được phân biệt bằng URL.

### Mã Authorization + PKCE

Dòng chảy:

1. Máy khách tạo `code_verifier` (ngẫu nhiên) và `code_challenge` (SHA256).
2. Máy khách chuyển hướng người dùng đến `/authorize?response_type=code&client_id=...&redirect_uri=...&scope=notes:read&code_challenge=...&resource=https://notes.example.com`.
3. Sự đồng ý của người dùng. Authorization server chuyển hướng đến `redirect_uri?code=...`.
4. Khách hàng POSTs đến `/token?grant_type=authorization_code&code=...&code_verifier=...&resource=...`.
5. Authorization server xác thực hàm băm của trình xác minh đối với thử thách được lưu trữ và đưa ra token truy cập.
6. Máy khách sử dụng token: `Authorization: Bearer ...` trên mọi yêu cầu đến server tài nguyên.

PKCE ngăn chặn các cuộc tấn công đánh chặn mã authorization. Các chỉ số tài nguyên ngăn không cho token hợp lệ ở nơi khác.

### Siêu dữ liệu tài nguyên được bảo vệ (RFC 9728)

Tài nguyên server xuất bản tài liệu `.well-known/oauth-protected-resource`:

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:read", "notes:write", "notes:delete"]
}
```

Khách hàng phát hiện ra authorization server từ server tài nguyên. Giảm configuration - máy khách chỉ cần URL tài nguyên.

### Chỉ số tài nguyên (RFC 8707)

`resource` parameter trong yêu cầu token ghim đối tượng mục tiêu của token. token đã ban hành có chứa `aud: "https://notes.example.com"`. Một MCP server khác nhận được token này kiểm tra `aud` và từ chối nó.

### Phạm vi model

Phạm vi là các chuỗi được phân tách bằng không gian. Quy ước MCP phổ biến:

- `notes:read`, `notes:write`, `notes:delete`
- `admin:*` cho các chức năng quản trị (sử dụng một cách tiết kiệm)
- `profile:read` cho danh tính

Lựa chọn phạm vi nên là đặc quyền tối thiểu: yêu cầu những gì bạn cần ngay bây giờ, tăng cường khi bạn cần nhiều hơn.

### authorization bước lên (SEP-835)

Người dùng cấp `notes:read`. Sau đó, họ yêu cầu agent xóa một ghi chú. Người server trả lời:

```
HTTP/1.1 403 Forbidden
WWW-Authenticate: Bearer error="insufficient_scope",
    scope="notes:delete", resource="https://notes.example.com"
```

Máy khách thấy lỗi insufficient_scope prompts người dùng có hộp thoại đồng ý cho phạm vi bổ sung, thực hiện quy trình OAuth nhỏ cho phạm vi đó, thử lại yêu cầu với token mới.

### Xác thực đối tượng Token

Mọi yêu cầu: server kiểm tra `token.aud == self.resource_url`. Không khớp = 401. Điều này ngăn chặn việc tái sử dụng chéo server token.

### tokens và xoay vòng trong thời gian ngắn

Truy cập tokens NÊN tồn tại trong thời gian ngắn (mặc định 1 giờ). Làm mới tokens xoay sau mỗi lần làm mới. Máy khách xử lý làm mới im lặng trong nền.

### Không token chuyển qua

Sampling servers (Giai đoạn 13 · 11) KHÔNG ĐƯỢC chuyển token của khách hàng sang các dịch vụ khác. Yêu cầu sampling là ranh giới.

### Phó phòng ngừa bối rối

Token liên kết với `aud`. Khách hàng liên kết với `client_id`. Mọi yêu cầu đều được xác thực dựa trên cả hai. Thông số kỹ thuật cấm rõ ràng mô hình "vượt qua token" cũ phổ biến trong hệ sinh thái công cụ từ xa trước khi MCP.

### Khám phá ID khách hàng

Mỗi ứng dụng MCP xuất bản siêu dữ liệu của nó tại một URL cố định. Authorization servers có thể tìm nạp tài liệu siêu dữ liệu của khách hàng để khám phá URI chuyển hướng và thông tin liên hệ. Thao tác này sẽ xóa đăng ký ứng dụng thủ công.

### Gateways và OAuth

Giai đoạn 13 · 17 cho thấy cách gateway doanh nghiệp xử lý OAuth: gateway giữ thông tin đăng nhập cho servers ngược dòng, tokens cho máy khách được cấp gateway và tokens ngược dòng không bao giờ rời khỏi gateway. Điều này lật ngược model tin cậy - người dùng xác thực với gateway một lần; gateway xử lý ủy quyền N server.

## Ứng dụng

`code/main.py` mô phỏng toàn bộ quy trình bước lên OAuth 2.1 dưới dạng máy trạng thái. Nó thực hiện:

- Trình xác minh mã PKCE / tạo thử thách.
- Authorization luồng mã với chỉ báo tài nguyên.
- Siêu dữ liệu tài nguyên được bảo vệ endpoint.
- Token xác thực với kiểm tra đối tượng.
- Tăng cường `insufficient_scope`.

Không có HTTP server trong bài học này; Máy trạng thái chạy trong bộ nhớ để bạn có thể trace mọi bước nhảy. Giai đoạn 13 · Bài học gateway của 17 kết nối nó với một transport thực tế.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-oauth-scope-planner.md`. Với một MCP server từ xa với các công cụ, skill thiết kế bộ phạm vi, quy tắc ghim và policy bước lên.

## Bài tập

1. Chạy `code/main.py`. Trace luồng bước lên hai phạm vi. Lưu ý bước nhảy nào lặp lại khi bước lên.

2. Thêm vòng quay token làm mới: mỗi lần làm mới sẽ đưa ra một token làm mới mới và làm mất hiệu lực của lần làm mới cũ. Mô phỏng quá trình làm mới bị đánh cắp token đang được sử dụng sau khi xoay và xác nhận rằng nó không thành công.

3. Triển khai siêu dữ liệu tài nguyên được bảo vệ endpoint dưới dạng phản hồi HTTP thực bằng cách sử dụng http stdlib. server. Phản chiếu /mcp endpoint từ Bài 09.

4. Thiết kế hệ thống phân cấp phạm vi cho một GitHub MCP server: đọc repo, viết PR, phê duyệt PR, merge PR, quản trị viên. Sử dụng bước lên giữa mỗi cấp độ.

5. Đọc RFC 8707 và RFC 9728. Xác định một trường trong 9728 mà MCP sử dụng khác với ví dụ của RFC. (Gợi ý: nó liên quan đến `scopes_supported`.)

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| OAuth 2.1 | "OAuth hiện đại" | RFC hợp nhất bắt buộc PKCE và cấm dòng chảy ngầm |
| PKCE | "Bằng chứng sở hữu" | Trình xác minh mã + thử thách đánh chặn mã authorization |
| Chỉ báo tài nguyên | "Token đối tượng" | RFC 8707 `resource` parameter ghim token vào một server |
| Siêu dữ liệu tài nguyên được bảo vệ | "Tài liệu khám phá" | RFC 9728 `.well-known/oauth-protected-resource` |
| Nâng cấp authorization | "Sự đồng ý gia tăng" | Quy trình SEP-835 để thêm phạm vi theo yêu cầu |
| `insufficient_scope` | "403 với WWW-Xác thực" | Server tín hiệu đồng ý lại cho phạm vi lớn hơn |
| Cấp phó bối rối | "Token tái sử dụng trên các dịch vụ" | Tấn công khi chủ sở hữu đáng tin cậy chuyển tiếp token không phù hợp |
| token tồn tại trong thời gian ngắn | "Truy cập token TTL" | Người mang hết hạn nhanh chóng; Làm mới token gia hạn |
| Hệ thống phân cấp phạm vi | "Đặc quyền tối thiểu stack" | Phạm vi chia độ được thiết lập với bước lên giữa các cấp độ |
| Siêu dữ liệu ID khách hàng | "Tài liệu khám phá khách hàng" | URL mà tại đó máy khách xuất bản siêu dữ liệu OAuth của riêng mình |

## Đọc thêm

- [MCP — Authorization spec](https://modelcontextprotocol.io/specification/draft/basic/authorization) — hồ sơ OAuth MCP chuẩn
- [den.dev — MCP November authorization spec](https://den.dev/blog/mcp-november-authorization-spec/) - hướng dẫn về những thay đổi 2025-11-25
- [RFC 8707 — Resource indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — RFC ghim khán giả
- [RFC 9728 — OAuth 2.0 protected resource metadata](https://datatracker.ietf.org/doc/html/rfc9728) — RFC tài liệu khám phá
- [Aembit — MCP OAuth 2.1, PKCE and the future of AI authorization](https://aembit.io/blog/mcp-oauth-2-1-pkce-and-the-future-of-ai-authorization/) — hướng dẫn thực tế
