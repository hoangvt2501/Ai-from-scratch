# MCP Auth in Production — Đăng ký, Làm mới JWKS, Tokens ghim khán giả

> Bài 16 đã đưa máy trạng thái OAuth 2.1 lên trong bộ nhớ. Đến năm 2026, mỗi MCP server bạn ship vào một tổ chức thực sự đều nằm sau production xác thực: đăng ký máy khách mở rộng quy mô theo số lượng khách hàng không giới hạn (Tài liệu siêu dữ liệu ID khách hàng trước, đăng ký máy khách động làm dự phòng tương thích ngược), khám phá siêu dữ liệu authorization server (RFC 8414 *hoặc* Khám phá OpenID Connect), làm mới bộ nhớ đệm JWKS không phá vỡ xác thực token lúc 3 giờ sáng,  và các tokens được ghim bởi khán giả từ chối phát lại nhiều tài nguyên. Bài học này models toàn bộ bề mặt với ba vai trò - một authorization server, một server tài nguyên (MCP server) và một khách hàng - vì vậy bạn có thể trace mọi bước nhảy từ khám phá đến một cuộc gọi công cụ đã được xác thực.
>
> **Ghi chú thông số kỹ thuật (2025-11-25):** thông số kỹ thuật MCP authorization tháng 11 năm 2025 đã hạ cấp Đăng ký khách hàng động từ `SHOULD` xuống `MAY` và biến **Tài liệu siêu dữ liệu ID khách hàng (CIMD)** thành cơ chế đăng ký mặc định được đề xuất. Bài học này dạy cả hai, theo thứ tự ưu tiên của thông số kỹ thuật và mã giữ DCR cho hướng dẫn vì nó hoàn toàn khép kín trong một process.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 13 · 16 (máy trạng thái OAuth 2.1), Giai đoạn 13 · 17 (gateways)
**Thời lượng:** ~90 phút

## Mục tiêu học tập

- Khám phá một authorization server thông qua siêu dữ liệu RFC 8414 và xác minh hợp đồng.
- Triển khai đăng ký ứng dụng khách động RFC 7591 để MCP khách hàng đăng ký mà không cần sự can thiệp của quản trị viên.
- Bộ nhớ đệm và làm mới các khóa JWKS theo lịch trình để xác minh chữ ký vẫn tồn tại sau khi chuyển đổi khóa.
- Ghim tokens vào một tài nguyên MCP duy nhất bằng cách sử dụng các chỉ báo tài nguyên RFC 8707 và từ chối sử dụng lại phó lẫn lộn.
- Tách biệt ba vai trò một cách rõ ràng - authorization server, server tài nguyên, khách hàng - để mỗi vai trò chỉ thực thi các kiểm tra thuộc về nó.
- Đọc ma trận khả năng IdP và từ chối triển khai khi IdP không thể đáp ứng hồ sơ xác thực của MCP.

## Vấn đề

Trình mô phỏng Bài 16 chạy OAuth 2.1 trong bộ nhớ. Production có ba lỗ hổng hoạt động mà trình mô phỏng chỉ bộ nhớ không nhìn thấy.

Khoảng cách đầu tiên là tuyển sinh. Một tổ chức thực sự điều hành hàng trăm MCP servers và hàng nghìn khách hàng MCP. Người vận hành không đăng ký thủ công mọi người dùng Con trỏ làm ứng dụng OAuth. Thông số kỹ thuật 2025-11-25 cung cấp cho khách hàng thứ tự ưu tiên để giải quyết vấn đề này: sử dụng `client_id` đăng ký trước nếu bạn có, nếu không thì sử dụng **Tài liệu siêu dữ liệu ID khách hàng** (máy khách tự xác định mình bằng URL HTTPS mà nó kiểm soát và authorization server *kéo* siêu dữ liệu), nếu không thì quay trở lại **Đăng ký máy khách động RFC 7591** (máy khách *đẩy* `POST /register` và nhận `client_id` ngay tại chỗ),  nếu không prompt người dùng. CIMD là mặc định được khuyến nghị vì nó xóa hoàn toàn đăng ký theo server trong khi vẫn giữ model tin cậy gốc DNS; DCR được giữ lại để tương thích ngược. Cả hai đều khám phá ra các điểm vào của họ từ siêu dữ liệu của authorization server: `client_id_metadata_document_supported` cho CIMD, `registration_endpoint` cho DCR.

Khoảng trống thứ hai là xoay phím. Xác thực JWT phụ thuộc vào khóa ký của authorization server, được xuất bản dưới dạng Bộ khóa web JSON (JWKS). authorization server luân phiên chúng theo lịch trình (thường là hàng giờ, đôi khi nhanh hơn khi ứng phó sự cố). Một MCP server tìm nạp JWK một lần khi khởi động sẽ xác thực tốt cho đến khi cửa sổ xoay - sau đó mọi yêu cầu đều thất bại cho đến khi khởi động lại. Production kết nối JWK dưới dạng giá trị được lưu trong bộ nhớ đệm với một công việc làm mới ghi đè lên bộ nhớ đệm trước khi các khóa trước đó hết hạn, cộng với việc tìm nạp dự phòng khi bỏ lỡ bộ nhớ đệm cho trường hợp một token được ký bởi một khóa mới hơn bộ nhớ đệm đến.

Khoảng cách thứ ba là ràng buộc khán giả. Bài 16 giới thiệu các chỉ báo tài nguyên RFC 8707. Trong production, chỉ báo đó trở thành một kiểm tra yêu cầu khó khăn đối với mọi yêu cầu. MCP server so sánh `token.aud` với URL tài nguyên chính tắc của chính nó và từ chối các thông tin không khớp với HTTP 401. Đây là biện pháp bảo vệ duy nhất chống lại một MCP server ngược dòng (hoặc một khách hàng độc hại đang nắm giữ một token dành cho một server) phát lại token đó chống lại một server khác trong cùng một mạng lưới tin cậy.

Bài học này lập bản đồ từng khoảng trống lên một mảnh bê tông của bề mặt. Tài liệu siêu dữ liệu là một HTTP endpoint. Làm mới bộ nhớ đệm JWKS là một công việc đã lên lịch cộng với bộ nhớ đệm khóa-giá trị. Xác thực JWT là một quy trình mà tài nguyên server chạy trước khi gửi bất kỳ công cụ nào. Giữ ba vai trò riêng biệt và mỗi vai trò chỉ thực thi các kiểm tra mà nó sở hữu: các khóa authorization server và luân phiên, tài nguyên server lưu trữ và xác thực, khách hàng phát hiện và đăng ký.

## Khái niệm

### RFC 8414 — Siêu dữ liệu OAuth Authorization Server

Một tài liệu tại `/.well-known/oauth-authorization-server` mô tả mọi thứ mà khách hàng cần:

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "registration_endpoint": "https://auth.example.com/register",
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

Một máy khách được phát hiện chuỗi URL tài nguyên MCP: `oauth-protected-resource` từ RFC 9728 (tài liệu của server tài nguyên) đặt tên cho nhà phát hành, sau đó `oauth-authorization-server` (RFC này) đặt tên cho mỗi endpoint. Máy khách không bao giờ mã hóa cứng một URL authorization.

Hợp đồng bạn xác minh trước khi tin tưởng IdP cho MCP:

- `code_challenge_methods_supported` bao gồm `S256` (PKCE theo RFC 7636). Thông số kỹ thuật rõ ràng: nếu trường này **vắng mặt**, authorization server không hỗ trợ PKCE và khách hàng **PHẢI** từ chối tiếp tục.
- `grant_types_supported` bao gồm `authorization_code` và từ chối `password` và `implicit`.
- Ít nhất một lộ trình đăng ký được quảng cáo: `client_id_metadata_document_supported: true` (CIMD, ưu tiên) **hoặc** `registration_endpoint` (RFC 7591 DCR, dự phòng). Hoặc thỏa mãn hợp đồng; bạn không còn yêu cầu DCR nữa.
- `response_types_supported` chính xác là `["code"]` cho OAuth 2.1.

Nếu thiếu `S256`, MCP server từ chối triển khai chống lại IdP này - không có chế độ xuống cấp cho PKCE. Nếu * không có * lộ trình đăng ký được quảng cáo và bạn không có `client_id` đăng ký trước, bạn cũng không thể đăng ký; Tệp kê khai triển khai sai, không phải mã.

### RFC 9728 (tóm tắt) — Siêu dữ liệu tài nguyên được bảo vệ

Bài 16 bao gồm RFC 9728. Đồng bằng trong production: tài liệu này là nơi duy nhất khách hàng tìm kiếm để tìm authorization servers được tin cậy bởi *this* MCP server. Một MCP server có thể chấp nhận tokens từ nhiều IdP (một cho nhân viên, một cho đối tác). RFC 9728 khai báo tập hợp đó; RFC 8414 ghi lại những gì mỗi IdP hỗ trợ.

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### Tài liệu siêu dữ liệu ID khách hàng (mặc định được đề xuất)

CIMD đảo ngược đăng ký từ *push* sang *pull*. Thay vì yêu cầu authorization server đúc `client_id`, máy khách sử dụng URL HTTPS mà nó kiểm soát **như **`client_id` của nó. URL phân giải thành tài liệu siêu dữ liệu JSON; authorization server tìm nạp theo yêu cầu trong luồng OAuth. Niềm tin bắt nguồn từ DNS: nếu nhà điều hành server tin tưởng `app.example.com`, nó tin tưởng máy khách được phục vụ từ `https://app.example.com/client.json`. Không cần đăng ký khứ hồi, không có không gian tên `client_id` để cạn kiệt, không có trạng thái server để đồng bộ.

Tài liệu siêu dữ liệu mà khách hàng lưu trữ:

```json
{
  "client_id": "https://app.example.com/oauth/client.json",
  "client_name": "Example MCP Client",
  "client_uri": "https://app.example.com",
  "redirect_uris": ["http://127.0.0.1:7333/callback", "http://localhost:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none"
}
```

Giá trị `client_id` trong tài liệu **MUST** bằng với URL mà nó được phục vụ (authorization server xác minh điều này; sự không khớp bị từ chối). authorization server quảng cáo hỗ trợ với `client_id_metadata_document_supported: true` trong siêu dữ liệu RFC 8414 của nó.

Hai sự thật bảo mật mà thông số kỹ thuật nói thẳng:

- **SSRF.** authorization server tìm nạp URL do kẻ tấn công cung cấp. Nó phải bảo vệ chống lại việc giả mạo yêu cầu phía server (không tìm nạp để internal/admin endpoints).
- **Mạo danh localhost.** Chỉ riêng CIMD không thể ngăn kẻ tấn công cục bộ yêu cầu URL siêu dữ liệu của khách hàng hợp pháp và ràng buộc bất kỳ chuyển hướng `localhost` nào. Các authorization server **MUST** hiển thị rõ ràng tên máy chủ URI chuyển hướng trong quá trình đồng ý và **NÊN **cảnh báo về chuyển hướng chỉ dành cho `localhost`.

Bởi vì CIMD không cần nhà nước phía server, không có nhà đăng ký để đứng lên theo cách DCR yêu cầu. Phía máy khách là chỉ đọc: phục vụ tài liệu siêu dữ liệu của bạn từ một HTTPS endpoint tĩnh và để authorization server kéo nó.

### RFC 7591 — Đăng ký máy khách động (tương thích dự phòng / ngược)

DCR hiện là một `MAY`, được giữ để tương thích ngược với các triển khai trước 2025-11-25 và IdP chưa hỗ trợ CIMD. Nếu không có nó (và không có CIMD hoặc đăng ký trước), mọi ứng dụng khách MCP (Con trỏ, Claude Máy tính để bàn, một agent tùy chỉnh) cần trao đổi ngoài băng tần với quản trị viên IdP. Với DCR, khách hàng đăng:

```json
POST /register
Content-Type: application/json

{
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

server phản hồi bằng `client_id` và `registration_access_token` cho các bản cập nhật sau:

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

`token_endpoint_auth_method: none` là giá trị mặc định phù hợp cho MCP client chạy trên thiết bị của người dùng. Họ chỉ nhận được `client_id` - không có `client_secret` để trốn thoát. PKCE cung cấp bằng chứng sở hữu mà khách hàng công cần.

Ba cạm bẫy production:

- Các endpoint đăng ký phải giới hạn tỷ lệ theo IP nguồn. Nếu không có điều đó, một tác nhân thù địch scripts hàng triệu đăng ký giả mạo và làm cạn kiệt không gian tên `client_id`. Chạy kiểm tra giới hạn tốc độ trước khi tổ chức đăng ký tên miền xử lý yêu cầu.
- `software_statement` (xác nhận JWT đã ký cho khách hàng) được yêu cầu bởi một số IdP doanh nghiệp. Bài học giả bỏ qua nó; production kết nối một bước xác minh từ chối đăng ký chưa ký từ bất kỳ thứ gì khác ngoài URI chuyển hướng localhost.
- `registration_access_token` phải được lưu trữ dưới dạng hàm băm, không phải văn bản thuần túy. Đánh cắp token này có nghĩa là kẻ tấn công có thể viết lại URI chuyển hướng của máy khách.

### RFC 8707 (tóm tắt) — Chỉ số tài nguyên

Bài 16 đã thiết lập hình dạng. Quy tắc production: mọi yêu cầu token đều bao gồm `resource=<canonical-mcp-url>` và MCP server xác minh `token.aud` khớp với URL tài nguyên của chính nó trên mỗi lệnh gọi. URI chuẩn là mã định danh *cụ thể nhất* cho server: nó sử dụng lược đồ và máy chủ viết thường, không có phân đoạn và thông thường không có dấu gạch chéo đuôi. Thành phần đường dẫn **không** bị tước theo quy tắc - thông số kỹ thuật giữ nó khi cần thiết để xác định một MCP server riêng lẻ. `https://mcp.example.com`, `https://mcp.example.com/mcp`, `https://mcp.example.com:8443` và `https://mcp.example.com/server/mcp` đều là URI chính tắc hợp lệ. Chọn một server và ghim `aud` vào chính xác điều đó. (Mô phỏng của bài học này sử dụng các đối tượng chủ nhà trần như `https://notes.example.com` để ngắn gọn; một triển khai đồng lưu trữ một số MCP servers dưới một nguồn gốc phân biệt chúng theo đường dẫn.)

### RFC 7636 (tóm tắt) - PKCE

PKCE là bắt buộc trong OAuth 2.1. Luồng authorization-code của bài học luôn mang `code_challenge` và `code_verifier`. server từ chối bất kỳ yêu cầu token nào mà không có trình xác minh hoặc với trình xác minh không băm đối với thử thách được lưu trữ.

### MCP Spec 2025-11-25 Hồ sơ xác thực

Thông số kỹ thuật MCP (2025-11-25) chính xác về những gì lớp authorization của MCP server phải làm:

- Triển khai siêu dữ liệu tài nguyên được bảo vệ RFC 9728 và cung cấp vị trí của nó thông qua tiêu đề `WWW-Authenticate: Bearer resource_metadata="..."` trên 401 **hoặc **`/.well-known/oauth-protected-resource` URI nổi tiếng (SEP-985 đã làm cho tiêu đề tùy chọn với một dự phòng nổi tiếng). Trường `authorization_servers` siêu dữ liệu **MUST** đặt tên ít nhất một server.
- Chỉ chấp nhận tokens thông qua `Authorization: Bearer ...` trên **mọi **yêu cầu - không bao giờ trong chuỗi truy vấn, không bao giờ được xác thực chỉ khi bắt đầu session.
- Xác thực `aud`, `iss`, `exp` và phạm vi bắt buộc cho mỗi yêu cầu. server **PHẢI **xác nhận rằng token được phát hành dành riêng cho nó (khán giả); Một `aud` bị thiếu hoặc không khớp sẽ bị từ chối, không bao giờ được coi là ký tự đại diện.
- Trên 401/403, trả về `WWW-Authenticate: Bearer` mang `error=...`, `resource_metadata="<PRM-URL>"` parameter (URL của tài liệu siêu dữ liệu, * không * tài nguyên trần) và `scope="..."` trên `insufficient_scope` (403). Lưu ý: parameter là `resource_metadata`, một con trỏ khám phá - không có `resource` parameter trong thử thách.
- Khám phá Authorization-server chấp nhận **RFC 8414 OAuth siêu dữ liệu **hoặc** OpenID Connect Discovery 1.0; Khách hàng phải thử cả hai hậu tố nổi tiếng theo thứ tự ưu tiên.
- Máy khách (không phải server) bảo vệ chống lại **các cuộc tấn công hỗn hợp**: nó ghi lại `issuer` dự kiến trước khi chuyển hướng và xác thực parameter phản hồi `iss` authorization (RFC 9207) trước khi đổi mã. Một mình PKCE không ngăn chặn sự nhầm lẫn, bởi vì khách hàng giao `code_verifier` của mình cho bất kỳ token endpoint nào mà nó được chỉ đạo.

Bản nháp OAuth 2.1 là chất nền; RFC 8414/7591/8707/9728/9207 + RFC 7636 + CIMD là bề mặt; Thông số kỹ thuật MCP là hồ sơ.

### Ma trận khả năng IdP

Không phải mọi IdP đều hỗ trợ hồ sơ MCP đầy đủ. Ma trận dưới đây ghi lại các tuyên bố khả năng thực tế kể từ thông số kỹ thuật 2025-11-25. Nó là một *cổng triển khai*, không phải là một khuyến nghị.

CIMD shipped trong thông số kỹ thuật 2025-11-25 và dự thảo OAuth cơ bản chỉ được thông qua vào tháng 10 năm 2025, vì vậy hỗ trợ của nhà cung cấp vẫn đang đến - coi "CIMD" bên dưới là "vị trí hiện tại, xác minh trong tenant của bạn", không phải là tuyên bố vĩnh viễn.

| Danh mục IdP | Siêu dữ liệu AS (8414/OIDC) | CIMD | RFC 7591 DCR | Tài nguyên RFC 8707 | RFC 7636 S256 PKCE | Ghi chú |
|---|---|---|---|---|---|---|
| Tự lưu trữ (Keycloak) | Có | mới nổi | Có | Có (kể từ 24.x) | Có | Tham khảo IdP cho hồ sơ MCP trong bài học này; đường dẫn DCR đầy đủ từ đầu đến cuối, CIMD theo dõi thông số kỹ thuật mới. |
| SSO doanh nghiệp (Microsoft Entra ID) | Có | mới nổi | Có (Bậc cao cấp) | Có | Có | Tính khả dụng của DCR khác nhau tùy theo tenant cấp; Xác minh trong tenant mục tiêu trước khi triển khai. |
| SSO doanh nghiệp (Okta) | Có | mới nổi | Có (Okta CIC / Auth0) | Có | Có | DCR có sẵn trên Auth0 (nay là Okta CIC); các tổ chức Okta cổ điển yêu cầu đăng ký trước quản trị viên. |
| IdP đăng nhập xã hội (chung) | khác nhau | Không | hiếm khi | hiếm khi | Có | Hầu hết các IdP xã hội coi khách hàng là đối tác tĩnh; không đăng ký tự phục vụ. Chỉ sử dụng làm nguồn nhận dạng, xếp lớp authorization server nhận biết MCP của riêng bạn lên trên. |
| Tùy chỉnh / cây nhà lá vườn | phụ thuộc | phụ thuộc | phụ thuộc | phụ thuộc | phụ thuộc | Nếu bạn ship của riêng mình, hãy ship hồ sơ đầy đủ và thích CIMD hơn. Bỏ qua PKCE hoặc liên kết đối tượng sẽ phá vỡ hợp đồng xác thực MCP. |

Quy tắc từ chối đối với tệp kê khai triển khai: nếu IdP đã chọn không liệt kê `S256` trong `code_challenge_methods_supported`, MCP server từ chối khởi động - PKCE không có chế độ suy giảm. Ghi danh là một cánh cổng mềm hơn: bạn cần *một* lộ trình làm việc (`client_id`, `client_id_metadata_document_supported: true` hoặc `registration_endpoint` đã đăng ký trước). Sự vắng mặt của DCR không còn là một trigger từ chối nữa, bởi vì CIMD hoặc đăng ký trước có thể bao gồm nó.

### Mẫu làm mới JWKS (xoay ở AS, làm mới ở server tài nguyên)

Giữ hai động từ riêng biệt, bởi vì kết hợp chúng là một lỗi production thực sự:

- **Xoay vòng** là những gì *authorization server* làm: đúc một khóa ký mới, xuất bản nó trong JWKS, ngừng hoạt động khóa cũ sau. Tài nguyên server không tham gia vào việc này và không thể làm điều đó - nó không giữ các khóa riêng tư của IdP.
- **Làm mới** là những gì *tài nguyên server* làm: `GET` lại JWK đã xuất bản vào bộ nhớ đệm của nó. Đó là hành động JWKS duy nhất mà tài nguyên server từng thực hiện.

Chế độ lỗi production là bộ nhớ cache cũ. Giải quyết vấn đề này bằng một tác vụ làm mới theo lịch trình cộng với bộ nhớ đệm khóa-giá trị. Tài nguyên server chạy một công việc (cron, bộ đếm thời gian, bất cứ thứ gì runtime của bạn cung cấp), trong một khoảng thời gian cố định, tìm nạp `<issuer>/.well-known/jwks.json` và ghi đè lên `cache[issuer] = {keys, fetched_at}`. Trình xác thực đọc từ bộ nhớ cache đó. Một token bị thiếu `kid` trong bộ nhớ đệm triggers làm mới đồng bộ **one** làm dự phòng, sau đó kiểm tra lại. Điều này xử lý hai trường hợp cùng một lúc: làm mới theo lịch trình và windows chồng chéo khóa trong đó token được ký bằng khóa hoàn toàn mới đến trước lần làm mới theo lịch trình tiếp theo.

Dự phòng **phải là tìm nạp lại, không bao giờ là luân phiên**. Nếu bạn kết nối đường dẫn bỏ lỡ bộ nhớ đệm đến một rotate-and-mint, hai thứ sẽ bị hỏng: (1) đúc một khóa mới tạo ra một `kid` *vẫn* không khớp với token, vì vậy việc tra cứu vẫn thất bại; và (2) một kẻ tấn công phun tokens với các giá trị `kid` ngẫu nhiên buộc một loạt các sáng tạo quan trọng không giới hạn - một DoS tự gây ra. Tìm nạp lại là idempotent, vì vậy một `kid` không có thật tốn nhiều nhất là một lần tìm nạp lãng phí.

Hình dạng bộ nhớ đệm:

```json
{
  "https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}
```

Hai phím cùng một lúc là trạng thái ổn định. Authorization servers xoay vòng bằng cách giới thiệu khóa tiếp theo (`k_2026_04`) trước khi nghỉ hưu (`k_2026_03`) trước đó, vì vậy tokens được phát hành theo khóa cũ vẫn có hiệu lực cho đến khi chúng hết hạn. Bộ nhớ đệm chứa liên minh; Trình xác thực chọn theo `kid`.

### Quy trình xác thực

MCP server chạy xác thực trước khi gửi bất kỳ công cụ nào. Hình dạng `code/main.py` sử dụng:

```python
result = server.validate(bearer_token, required_scope="mcp:tools.invoke")
if not result["valid"]:
    return {"status": result["status"], "WWW-Authenticate": result["www_authenticate"]}
```

`validate` giải mã JWT, giải quyết khóa ký từ bộ nhớ đệm JWKS (làm mới một lần khi bỏ lỡ), xác minh chữ ký, sau đó kiểm tra `iss` với danh sách cho phép `aud` với tài nguyên chuẩn của server này, `exp` và phạm vi bắt buộc — trả về thử thách `WWW-Authenticate` cho lần thất bại đầu tiên. Giữ nó một quy trình duy nhất trên server tài nguyên có nghĩa là mọi điểm vào (mọi lệnh gọi công cụ, mọi transport) đều trải qua cùng một lần kiểm tra; Không có đường dẫn nào đến một công cụ mà không xác thực trước.

### Hướng dẫn phát lại khán giả (hạn chế đặc quyền truy cập token)

Server A (`notes.example.com`) và Server B (`tasks.example.com`) đều đăng ký dựa trên cùng một authorization server. Server A bị xâm phạm. Kẻ tấn công lấy ghi chú của người dùng token và phát lại nó với Server B.

Trình xác thực của Server B:

1. Giải mã JWT, tìm nạp JWKS theo `kid`, xác minh chữ ký.
2. Kiểm tra `iss` so với `authorization_servers` của siêu dữ liệu tài nguyên được bảo vệ. (Vượt qua - cùng một IdP.)
3. Kiểm tra `aud == "https://tasks.example.com"`. (Thất bại - `aud` của token là `https://notes.example.com`.)
4. Trả lại 401 với `WWW-Authenticate: Bearer error="invalid_token", error_description="audience mismatch", resource_metadata="https://tasks.example.com/.well-known/oauth-protected-resource"`.

Tuyên bố của khán giả là biện pháp bảo vệ duy nhất chống lại cuộc tấn công này ở lớp giao thức. Bỏ qua nó vì hiệu suất là sai lầm phổ biến nhất production; Trình xác thực phải chạy trên mọi yêu cầu, không chỉ khi bắt đầu session. Thông số kỹ thuật gọi đây là **hạn chế đặc quyền truy cập token**: một MCP server `MUST` từ chối bất kỳ token nào không nêu tên nó trong khán giả.

> **Ghi chú đặt tên.** Thông số kỹ thuật dành riêng thuật ngữ *phó nhầm lẫn* cho một vấn đề liên quan nhưng khác biệt: một MCP server hoạt động như một OAuth **proxy** cho một API của bên thứ ba, sử dụng ID máy khách tĩnh, chuyển tiếp một token mà không có sự đồng ý của mỗi người dùng. Liên kết khán giả khắc phục phát lại ở trên; Sửa lỗi Confused-Deputy là sự đồng ý của mỗi khách hàng **cộng với **không bao giờ chuyển token đến APIs ngược dòng (MCP server `MUST` có token ngược dòng riêng biệt).

### Các cuộc tấn công hỗn hợp (phòng thủ phía máy khách mà server không thể cung cấp)

Một khách hàng nói chuyện với nhiều authorization servers trong suốt cuộc đời của nó. Một AS độc hại có thể cố gắng khiến khách hàng đổi mã authorization của AS trung thực tại token endpoint của kẻ tấn công. Ràng buộc khán giả không giúp ích gì ở đây - cuộc tấn công xảy ra trước khi bất kỳ token nào tồn tại. Người bào chữa sống trong máy khách (RFC 9207):

1. Trước khi chuyển hướng, máy khách ghi lại `issuer` dự kiến từ siêu dữ liệu AS đã được xác thực.
2. Trên phản hồi authorization, máy khách so sánh `iss` parameter trả về với trình phát hành được ghi lại đó (so sánh chuỗi đơn giản, không chuẩn hóa) trước khi gửi mã đến bất cứ đâu.
3. Không khớp (hoặc `iss` vắng mặt khi AS được quảng cáo `authorization_response_iss_parameter_supported`) → từ chối và thậm chí không hiển thị các trường `error`.

Một mình PKCE không ngăn chặn sự nhầm lẫn, bởi vì khách hàng giao `code_verifier` của mình cho bất kỳ token endpoint nào mà nó được chỉ đạo. Đây là lý do tại sao thông số kỹ thuật ghi lại nhà phát hành theo yêu cầu cùng với trình xác minh PKCE và `state`.

### Chế độ thất bại

- **JWKS cũ.** Trình xác thực từ chối tokens hợp lệ sau khi AS xoay khóa. Cách khắc phục là mẫu cron-refresh + cache-miss-refetch ở trên. Không bao giờ lưu JWK vào bộ nhớ đệm mà không có công việc làm mới.
- **Rotate-as-fall-back.** Kết nối đường dẫn bỏ lỡ bộ nhớ đệm đến rotate-and-mint thay vì tìm nạp lại là một lỗi thực sự: nó không bao giờ tạo ra `kid` bị thiếu và nó biến các giá trị `kid` do kẻ tấn công kiểm soát thành DoS tạo khóa. Dự phòng phải là `refresh-jwks` idempotent.
- **Thiếu `aud` xác nhận quyền sở hữu.** Một số IdP mặc định bỏ qua `aud` trừ khi `resource` có trong yêu cầu token. Trình xác thực phải từ chối tokens thiếu `aud`, không coi vắng mặt là ký tự đại diện.
- **Nhầm lẫn thông qua kiểm tra `iss` bị thiếu.** Một máy khách không xác thực parameter phản hồi `iss` authorization RFC 9207 chống lại nhà phát hành mà nó đã ghi lại trước khi chuyển hướng có thể được điều khiển để đổi mã AS trung thực tại token endpoint của kẻ tấn công. Đây là một lỗi phía máy khách; tài nguyên server không thể bù đắp cho nó.
- **Cuộc đua nâng cấp phạm vi.** Hai luồng bước lên đồng thời cho cùng một người dùng có thể thành công và tạo ra hai tokens truy cập với các phạm vi khác nhau. Trình xác thực phải sử dụng token được trình bày trên yêu cầu, không phải tra cứu "phạm vi hiện tại của người dùng" - tạo ra một cửa sổ TOCTOU.
- **Đăng ký token trộm cắp.** Một `registration_access_token` bị rò rỉ cho phép kẻ tấn công viết lại URI chuyển hướng. Băm những thứ này tại rest; yêu cầu khách hàng trình bày văn bản rõ ràng trên mỗi bản cập nhật; xoay khi nghi ngờ.
- **`iss` không được ghim.** Trình xác thực chấp nhận bất kỳ `iss` nào cho phép kẻ tấn công đứng lên authorization server của riêng họ, đăng ký ứng dụng khách cho đối tượng mục tiêu và đưa ra tokens. Danh sách `authorization_servers` của siêu dữ liệu tài nguyên được bảo vệ là danh sách cho phép; thực thi nó.

## Ứng dụng

`code/main.py` đi bộ toàn bộ dòng production với stdlib Python và ba vai trò - `AuthorizationServer`, `ResourceServer` và `Client`. Dòng chảy:

1. Authorization server xuất bản siêu dữ liệu RFC 8414 tại `/.well-known/oauth-authorization-server`.
2. MCP khách gọi siêu dữ liệu endpoint và kiểm tra các tùy chọn đăng ký của nó (`client_id_metadata_document_supported` cho CIMD, `registration_endpoint` cho DCR) và `S256` hỗ trợ PKCE.
3. Hướng dẫn sử dụng đường dẫn dự phòng DCR: máy khách đăng đến `/register` (RFC 7591) và nhận được `client_id`. (Thay vào đó, một ứng dụng CIMD sẽ hiển thị URL HTTPS `client_id` của riêng mình và bỏ qua bước này.)
4. MCP khách chạy luồng mã authorization được bảo vệ bởi PKCE (RFC 7636) với chỉ báo `resource` (RFC 8707).
5. MCP khách gọi một công cụ trên MCP server với `Authorization: Bearer ...`.
6. MCP server chạy `validate`, giải quyết khóa ký từ bộ nhớ đệm JWKS.
7. IdP xoay một phím; quá trình làm mới theo lịch trình sẽ kéo lại JWK vào bộ nhớ đệm.
8. Lệnh gọi tiếp theo xác thực với các khóa đã làm mới mà không cần khởi động lại và token trước đó vẫn xác thực trong khoảng thời gian trùng lặp.
9. Nỗ lực phát lại khán giả đối với một tài nguyên MCP khác sẽ nhận được 401 với `audience mismatch` và con trỏ `resource_metadata`.

JWT ở đây sử dụng HS256 với một bí mật được chia sẻ (vì vậy bài học chỉ chạy trên stdlib). Production sử dụng RS256 hoặc EdDSA với mẫu JWKS ở trên; logic xác thực giống hệt nhau. Bởi vì IdP và tài nguyên server sống trong một process, `refresh_jwks` đọc trực tiếp danh sách khóa của authorization server; trên dây, nó là một HTTP `GET` để `jwks_uri`.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-mcp-auth.md`. Với một MCP server config và một bộ khả năng IdP, skill sẽ phát ra bề mặt xác thực để đứng lên — siêu dữ liệu tài nguyên được bảo vệ, đường dẫn đăng ký để sử dụng (CIMD, đăng ký trước hoặc dự phòng DCR), lịch làm mới JWKS, ánh xạ phạm vi và các quy tắc từ chối áp dụng khi IdP không hỗ trợ hồ sơ RFC đầy đủ.

## Bài tập

1. Chạy `code/main.py`. Trace dòng chảy. Lưu ý cách IdP xoay khóa ở bước 6, `refresh_jwks` đã lên lịch kéo lại tập đã xuất bản và cả token cũ (cửa sổ chồng chéo) và token mới đều xác thực mà không cần khởi động lại.

2. Thêm IdP mới vào danh sách `authorization_servers` của siêu dữ liệu tài nguyên được bảo vệ. Phát hành một token được ký bởi IdP mới và xác nhận trình xác thực chấp nhận nó. Phát hành token được ký bởi IdP không công khai và xác nhận trình xác thực từ chối bằng `WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"`.

3. Thêm kiểm tra giới hạn tốc độ vào `register_client` chạy trước khi nhà đăng ký chấp nhận yêu cầu. Sử dụng bộ chứa token cho mỗi IP nguồn được giữ trong một dict nhỏ được khóa bởi IP.

4. Đọc RFC 7591 và xác định hai trường mà trình xử lý `/register` của bài học không xác thực. Thêm xác thực. (Gợi ý: `software_statement` và `redirect_uris` lược đồ URI.)

5. Thêm đường dẫn Tài liệu siêu dữ liệu ID khách hàng. Phân phát một `client.json` có `client_id` bằng URL của chính nó, đồng thời yêu cầu authorization server tìm nạp và xác minh URL đó (từ chối nếu `client_id` ≠ URL). Xác nhận khách hàng CIMD đăng ký mà không cần cuộc gọi `register_client`.

6. Chứng minh bản sửa lỗi DoS. Gửi cho trình xác thực một token với một `kid` ngẫu nhiên và xác nhận `refresh_jwks` chạy nhiều nhất một lần và số lượng khóa của authorization server không tăng lên. Sau đó, cố tình nối lại dự phòng trở lại một vòng quay và đúc và xem số lượng khóa tăng lên trên mỗi token không có thật - khôi phục quá trình tìm nạp lại sau đó.

7. Triển khai kiểm tra RFC 920 `iss` 7 phía máy khách từ phần trộn lẫn: ghi lại nhà phát hành dự kiến trước yêu cầu authorization, sau đó từ chối phản hồi authorization có `iss` không khớp.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| ASM | "Tài liệu siêu dữ liệu OAuth" | RFC 8414 `/.well-known/oauth-authorization-server` JSON |
| CIMD | "URL siêu dữ liệu máy khách" | Tài liệu siêu dữ liệu ID khách hàng — một URL HTTPS được sử dụng làm `client_id`; AS kéo JSON. Mặc định được đề xuất kể từ 2025-11-25 |
| DCR | "Đăng ký khách hàng tự phục vụ" | RFC 7591 `POST /register` luồng; xuống dự phòng `MAY` vào 2025-11-25 |
| JWKS | "Khóa công khai để xác thực JWT" | JSON Web Key Set, được tìm nạp từ `jwks_uri`, được lập chỉ mục bởi `kid` |
| Xoay vs làm mới | "Cập nhật các phím" | *Rotate* = AS mints/retires khóa ký; *refresh* = resource server tìm nạp lại tập hợp đã xuất bản. Tài nguyên chỉ servers làm mới |
| Chỉ báo tài nguyên | "Khán giả parameter" | RFC 8707 `resource` parameter ghim token vào một server |
| `aud` yêu cầu bồi thường | "Đối tượng" | Xác nhận quyền sở hữu JWT so sánh trình xác thực với URL tài nguyên chính tắc |
| Phát lại khán giả | "Token phát lại" | Token cấp cho Server A được trình bày cho Server B; Được bảo vệ bằng xác thực đối tượng (thông số kỹ thuật: Hạn chế đặc quyền truy cập token) |
| Cấp phó bối rối | "Proxy token lạm dụng" | Một MCP proxy có ID khách hàng tĩnh chuyển tiếp một token mà không có sự đồng ý của từng khách hàng; Khác biệt với phát lại khán giả |
| Tấn công hỗn hợp | "Sai token endpoint" | Khách hàng đã chỉ đạo để đổi mã AS trung thực tại endpoint của kẻ tấn công; bảo vệ phía máy khách thông qua RFC 9207 `iss` |
| `iss` danh sách cho phép | "authorization servers đáng tin cậy" | Tập hợp được đặt tên trong `authorization_servers` của siêu dữ liệu tài nguyên được bảo vệ |
| `resource_metadata` | "Tìm tài liệu PRM ở đâu" | `WWW-Authenticate` parameter đặt tên cho URL siêu dữ liệu RFC 9728 trên 401/403 |
| Khách hàng công khai | "Máy khách gốc hoặc trình duyệt" | Ứng dụng OAuth không có `client_secret`; PKCE bù đắp |
| `WWW-Authenticate` | "Tiêu đề phản hồi 401/403" | Mang `Bearer error=...` chỉ thị thúc đẩy phục hồi khách hàng |

## Đọc thêm

- [MCP — Authorization spec (2025-11-25)](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization) — hồ sơ xác thực MCP mà bài học này triển khai
- [MCP blog — One Year of MCP: November 2025 Spec Release](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — điều gì đã thay đổi vào ngày 25 tháng 11 năm 2025 (giáng chức CIMD, XAA, DCR)
- [Aaron Parecki — Client Registration in the November 2025 MCP Authorization Spec](https://aaronparecki.com/2025/11/25/1/mcp-authorization-spec-update) - cơ sở lý luận CIMD-over-DCR
- [OAuth Client ID Metadata Document (draft-ietf-oauth-client-id-metadata-document-00)](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-client-id-metadata-document-00) - CIMD
- [RFC 8414 — OAuth 2.0 Authorization Server Metadata](https://datatracker.ietf.org/doc/html/rfc8414) - hợp đồng khám phá
- [RFC 7591 — OAuth 2.0 Dynamic Client Registration Protocol](https://datatracker.ietf.org/doc/html/rfc7591) - DCR (đường dẫn dự phòng)
- [RFC 7636 — Proof Key for Code Exchange (PKCE)](https://datatracker.ietf.org/doc/html/rfc7636) — bằng chứng sở hữu khách hàng công cộng
- [RFC 8707 — Resource Indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — ghim khán giả
- [RFC 9728 — OAuth 2.0 Protected Resource Metadata](https://datatracker.ietf.org/doc/html/rfc9728) — tài nguyên server khám phá
- [RFC 9207 — OAuth 2.0 Authorization Server Issuer Identification](https://datatracker.ietf.org/doc/html/rfc9207) - `iss` parameter bảo vệ chống lại các cuộc tấn công hỗn hợp
- [OAuth 2.1 draft](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1) — nền OAuth hợp nhất
