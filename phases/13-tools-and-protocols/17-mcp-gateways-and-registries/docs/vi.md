# MCP Gateways và Registries — Mặt phẳng điều khiển doanh nghiệp

> Doanh nghiệp không thể để mọi nhà phát triển cài đặt MCP servers ngẫu nhiên. Một gateway tập trung xác thực, RBAC, kiểm tra, giới hạn tốc độ, bộ nhớ đệm và phát hiện ngộ độc công cụ, sau đó hiển thị bề mặt công cụ merged dưới dạng một MCP endpoint duy nhất. Official MCP Registry (Anthropic + GitHub + PulseMCP + Microsoft, xác minh không gian tên) là thượng nguồn chính tắc. Bài học này nêu tên nơi phù hợp với gateway, thực hiện tối thiểu và khảo sát bối cảnh nhà cung cấp năm 2026.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, minimal gateway)
**Kiến thức tiên quyết:** Giai đoạn 13 · 15 (ngộ độc dụng cụ), Giai đoạn 13 · 16 (OAuth 2.1)
**Thời lượng:** ~45 phút

## Mục tiêu học tập

- Giải thích vị trí của một MCP gateway (giữa MCP khách hàng và nhiều MCP servers phụ trợ).
- Thực hiện năm trách nhiệm gateway: xác thực, RBAC, kiểm toán, rate limit, policy.
- Thực thi tệp kê khai băm công cụ được ghim ở lớp gateway.
- Phân biệt MCP Registry chính thức với siêu đăng ký (Glama, MCPMarket, MCP.so, Smithery, LobeHub).

## Vấn đề

Fortune 500 có 30 MCP servers được phê duyệt, 5000 nhà phát triển, các yêu cầu tuân thủ và kiểm tra cũng như một nhóm bảo mật muốn có policy tập trung. Cho phép mọi nhà phát triển cài đặt servers tùy ý trong IDE của họ là điều không thể bắt đầu.

Mô hình gateway:

1. Gateway chạy dưới dạng một HTTP endpoint có thể phát trực tuyến duy nhất mà các nhà phát triển kết nối.
2. Gateway giữ thông tin đăng nhập cho mỗi MCP server phụ trợ.
3. Mọi yêu cầu của nhà phát triển đều được xác thực và xác định phạm vi thông qua OAuth riêng của gateway.
4. Gateway định tuyến cuộc gọi đến server phụ trợ, áp dụng policy.
5. Tất cả các cuộc gọi được ghi lại để kiểm tra.

Cloudflare MCP Portals, Kong AI Gateway, IBM ContextForge, MintMCP, TrueFoundry, Envoy AI Gateway - tất cả đều shipped gateways hoặc gateway features vào năm 2025-2026.

Trong khi đó, Official MCP Registry ra mắt dưới dạng thượng nguồn chuẩn: được quản lý, xác minh không gian tên, đặt tên DNS ngược servers gateway có thể lấy từ đó. Siêu đăng ký (Glama, MCPMarket, MCP.so, Smithery, LobeHub) tổng hợp servers trên nhiều nguồn.

## Khái niệm

### Năm trách nhiệm gateway

1. **Auth.** OAuth 2.1 để xác định nhà phát triển; ánh xạ đến vai trò của người dùng.
2. **RBAC.** policy cho mỗi người dùng: servers nào, công cụ nào, phạm vi nào.
3. **Kiểm tra.** Mọi cuộc gọi được ghi lại với ai, cái gì, khi nào, kết quả.
4. **Rate limit.** Mũ cho mỗi người dùng / mỗi công cụ / mỗi server để ngăn chặn lạm dụng.
5. **Policy.** Từ chối mô tả bị nhiễm độc, thực thi Quy tắc Hai, biên tập PII.

### Gateway dưới dạng một endpoint

Đối với các nhà phát triển, gateway trông giống như một MCP server. Bên trong nó định tuyến đến N phần phụ trợ. Session id (Giai đoạn 13 · 09) được viết lại ở ranh giới.

### Kho thông tin xác thực

Các nhà phát triển không bao giờ thấy tokens phụ trợ. gateway giữ chúng (hoặc ủy quyền cho nhà cung cấp danh tính). Một nhà phát triển có `notes:read` trên gateway có thể truy cập chuyển tiếp các ghi chú MCP server bằng thông tin đăng nhập phụ trợ của chính gateway - nhưng chỉ dưới policy ràng buộc quyền truy cập chuyển tiếp.

### Ghim băm công cụ ở gateway

gateway chứa một bản kê khai các mô tả công cụ đã được phê duyệt (hàm băm SHA256). Tại thời điểm khám phá, nó tìm nạp `tools/list` của từng phần phụ trợ, so sánh hàm băm với tệp kê khai và xóa bất kỳ công cụ nào có mô tả đã bị thay đổi. Đây là phòng thủ kéo thảm từ Giai đoạn 13 · 15 áp dụng tập trung.

### Policy dưới dạng mã

Advanced gateways express policy bằng OPA/Rego, Kyverno hoặc Styra. Các quy tắc như "`alice` người dùng chỉ có thể gọi `github.open_pr` trên repos trong `acme` của tổ chức" được mã hóa khai báo. Đơn giản gateways sử dụng Python được mã hóa bằng tay. Cả hai hình dạng đều hợp lệ.

### Định tuyến nhận biết Session

Khi session của người dùng bao gồm kết hợp các servers, gateway ghép kênh: MCP session duy nhất của nhà phát triển giữ N sessions phụ trợ, mỗi server một lần. Thông báo từ bất kỳ tuyến phụ trợ nào thông qua gateway đến session của nhà phát triển.

### Hợp nhất không gian tên

Gateways merge không gian tên công cụ từ tất cả các phần phụ trợ, thường có tiền tố khi va chạm. `github.open_pr`, `notes.search`. Điều này làm cho định tuyến trở nên rõ ràng.

### Registries

- **MCP Registry chính thức (`registry.modelcontextprotocol.io`).** Ra mắt dưới sự quản lý của Anthropic, GitHub, PulseMCP, Microsoft. Xác minh không gian tên (DNS ngược: `io.github.user/server`). Được lọc trước cho chất lượng cơ bản.
- **Glama.** Siêu đăng ký tập trung vào tìm kiếm tổng hợp nhiều nguồn.
- **MCPMarket.** Thư mục nghiêng về thương mại với danh sách nhà cung cấp.
- **MCP.so.** Thư mục cộng đồng; Mở bài gửi.
- **Smithery.** Quy trình cài đặt kiểu trình quản lý gói.
- **LobeHub.** registry tích hợp giao diện người dùng trong ứng dụng LobeChat của họ.

Enterprise gateways lấy từ Registry chính thức theo mặc định, cho phép bổ sung do quản trị viên quản lý từ siêu đăng ký và từ chối bất kỳ nội dung nào không được ghim.

### Đặt tên DNS ngược

Các Registry chính thức bắt buộc đặt tên DNS ngược cho các servers công khai: `io.github.alice/notes`. Không gian tên ngăn chặn việc ngồi xổm và làm cho ủy quyền tin cậy rõ ràng hơn.

### Khảo sát nhà cung cấp, Tháng Tư 2026

| Nhà cung cấp | Sức mạnh |
|--------|----------|
| Cổng thông tin MCP Cloudflare | Lưu trữ cạnh; Tích hợp OAuth; Bậc miễn phí |
| Kong AI Gateway | K8s-bản địa; policy hạt mịn; nhật ký vào OpenTelemetry |
| IBM ContextForge | IAM doanh nghiệp; tuân thủ; Xuất kiểm toán |
| Xưởng đúc thật | nghiêng về DevOps; số liệu đầu tiên |
| Đúc MCP | Định hướng nền tảng dành cho nhà phát triển |
| Đặc phái viên AI Gateway | Mã nguồn mở; Bộ lọc có thể tùy chỉnh |

Giai đoạn 17 (cơ sở hạ tầng production) đi sâu hơn vào các hoạt động gateway.

## Ứng dụng

`code/main.py` ships gateway tối thiểu trong ~150 dòng: xác thực người dùng bằng token Bearer giả mạo, giữ policy RBAC cho mỗi người dùng, định tuyến yêu cầu đến hai MCP servers phụ trợ, ghi mọi lệnh gọi vào nhật ký kiểm tra, thực thi rate limit và từ chối bất kỳ công cụ phụ trợ nào có hàm băm mô tả không khớp với tệp kê khai được ghim.

Những gì cần xem:

- `RBAC` dict được khóa bởi `user_id` với các mục `server_tool` được phép.
- `AUDIT_LOG` là danh sách các sự kiện chỉ nối thêm.
- Rate limit sử dụng vùng lưu trữ token cho mỗi người dùng.
- Tệp kê khai được ghim là một dict của `server::tool -> hash`.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-gateway-bootstrap.md`. Với kế hoạch MCP doanh nghiệp (người dùng, phụ trợ, tuân thủ), skill tạo ra một thông số kỹ thuật gateway configuration.

## Bài tập

1. Chạy `code/main.py`. Thực hiện cuộc gọi với tư cách là người dùng được phép; sau đó là người dùng không được phép; sau đó là một đợt bùng nổ vượt quá giới hạn tỷ lệ. Xác minh cả ba luồng.

2. Thêm một policy để biên tập PII khỏi kết quả trước khi trả lại cho máy khách. Sử dụng một pass regex đơn giản cho các chuỗi hình SSN; Lưu ý khoảng cách (email, số điện thoại).

3. Mở rộng nhật ký kiểm tra để phát ra OpenTelemetry GenAI spans. Giai đoạn 13 · 20 bao gồm các thuộc tính chính xác.

4. Thiết kế policy RBAC cho nhóm 50 nhà phát triển với năm backend (ghi chú, github, postgres, jira, slack). Ai được chỉ đọc trên mỗi loại? Ai được viết?

5. Đọc bài đăng MCP doanh nghiệp Cloudflare từ trên xuống dưới. Xác định một feature ships Cloudflare mà gateway stdlib này không có.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Gateway | "MCP proxy" | Tập trung server giữa khách hàng và phụ trợ |
| Kho thông tin xác thực | "Backend tokens ở bên server" | Các nhà phát triển không bao giờ nhìn thấy tokens ngược dòng |
| Định tuyến nhận biết Session | "session đa phụ trợ" | Gateway ghép kênh N sessions phụ trợ cho mỗi nhà phát triển session |
| Ghim băm công cụ | "Bản kê khai được phê duyệt" | SHA256 của mọi mô tả công cụ đã được phê duyệt; chặn rug-pull ở trung tâm |
| RBAC | "policy cho mỗi người dùng" | Kiểm soát truy cập dựa trên vai trò cho các công cụ và servers |
| Policy dưới dạng mã | "Quy tắc khai báo" | OPA/Rego, Kyverno, Styra policies thi hành tại gateway |
| Nhật ký kiểm tra | "Ai, cái gì, khi nào" | Nhật ký sự kiện chỉ nối để tuân thủ |
| Rate limit | "Vùng lưu trữ token mỗi người dùng" | Giới hạn mỗi phút để ngăn chặn lạm dụng |
| MCP Registry chính thức | "thượng nguồn chính tắc" | `registry.modelcontextprotocol.io`, xác minh không gian tên |
| Đặt tên DNS ngược | "Không gian tên Registry" | `io.github.user/server` quy ước |

## Đọc thêm

- [Official MCP Registry](https://registry.modelcontextprotocol.io/) — thượng nguồn chuẩn, xác minh không gian tên
- [Cloudflare — Enterprise MCP](https://blog.cloudflare.com/enterprise-mcp/) — gateway mẫu với OAuth và policy
- [agentic-community — MCP gateway registry](https://github.com/agentic-community/mcp-gateway-registry) — gateway tham khảo mã nguồn mở
- [TrueFoundry — What is an MCP gateway?](https://www.truefoundry.com/blog/what-is-mcp-gateway) — bài viết so sánh feature
- [IBM — MCP context forge](https://github.com/IBM/mcp-context-forge) — gateway doanh nghiệp từ IBM
