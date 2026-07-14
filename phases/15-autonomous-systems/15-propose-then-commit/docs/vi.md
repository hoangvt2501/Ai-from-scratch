# Con người trong vòng lặp: Đề xuất sau đó Commit

> Sự đồng thuận năm 2026 về HITL là cụ thể. Nó không phải là "agent hỏi, người dùng nhấp vào Phê duyệt". Đó là đề xuất sau đó commit: hành động được đề xuất được duy trì cho một cửa hàng lâu dài với khóa idempotecy; hiển thị cho người đánh giá với ý định, dòng dữ liệu, quyền được chạm, bán kính vụ nổ và kế hoạch rollback; chỉ cam kết sau khi được thừa nhận tích cực; đã xác minh sau khi thực hiện để xác nhận tác dụng phụ thực sự xảy ra. `interrupt()` của LangGraph cộng với điểm kiểm tra PostgreSQL, `RequestInfoEvent` của Microsoft Agent Framework và `waitForApproval()` của Cloudflare đều triển khai cùng một hình dạng. Chế độ lỗi chuẩn là phê duyệt tem cao su: "Phê duyệt?" được nhấp vào mà không cần xem xét. Việc giảm thiểu được ghi lại là thách thức và phản ứng với một danh sách kiểm tra rõ ràng.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, propose-then-commit state machine with idempotency)
**Kiến thức tiên quyết:** Giai đoạn 15 · 12 (Thực thi bền bỉ), Giai đoạn 15 · 14 (Dây vấy)
**Thời lượng:** ~60 phút

## Vấn đề

Một agent thực hiện một hành động. Người dùng phải quyết định: phê duyệt hay không. Nếu quyết định là ngay lập tức, nó có thể không phải là một đánh giá. Nếu quyết định có cấu trúc, nó chậm nhưng đáng tin cậy. Câu hỏi kỹ thuật là làm thế nào để làm cho một đánh giá có cấu trúc trở thành con đường ít kháng cự nhất.

Mô hình HITL thời kỳ 2023 là một prompt đồng bộ: "Agent muốn gửi email đến X với nội dung Y - chấp thuận?" Người dùng nhấp vào Approve (Phê duyệt). Mọi người đều cảm thấy hệ thống an toàn. Trong thực tế, bề mặt này được đóng dấu cao su rất nhiều: người dùng phê duyệt nhanh, phê duyệt dự đoán ít và khi agent sai, dấu vết kiểm tra cho thấy lịch sử phê duyệt lâu dài mà người dùng không thể recall.

Mô hình năm 2026 - đề xuất sau đó commit - di chuyển HITL lên một chất nền bền, đính kèm siêu dữ liệu có cấu trúc và yêu cầu commit tích cực. Mỗi agent SDK ships được quản lý đều có một phiên bản: LangGraph `interrupt()`, Microsoft Agent Framework `RequestInfoEvent`, Cloudflare `waitForApproval()`. Các API tên khác nhau; hình dạng thì không.

## Khái niệm

### Bộ máy nhà nước đề xuất sau đó commit

1. **Đề xuất.** Agent đưa ra một hành động được đề xuất. Tồn tại với một cửa hàng bền bỉ (PostgreSQL, Redis, Durable Object). Bao gồm:
   - Ý định (tại sao agent lại làm điều này)
   - dòng dữ liệu (nguồn nào dẫn đến đề xuất này)
   - Quyền được chạm vào (phạm vi / tệp / endpoints)
   - bán kính vụ nổ (trường hợp xấu nhất là gì)
   - rollback kế hoạch (nếu đã cam kết, làm thế nào để chúng tôi hoàn tác nó)
   - idempotency key (duy nhất cho mỗi đề xuất; gửi lại trả về cùng một bản ghi)
2. **Surface.** Người đánh giá xem đề xuất với tất cả siêu dữ liệu. Người phản biện là một người (không phải agent tự đánh giá).
3. **Commit.** Lời cảm ơn tích cực. Hành động thực thi.
4. **Xác minh.** Sau khi thực hiện, tác dụng phụ được đọc lại và xác nhận. Nếu bước xác minh không thành công, hệ thống sẽ ở trạng thái xấu đã biết và cảnh báo sẽ hoạt động.

### Chìa khóa idempotency

Nếu không có khóa idempotency thì việc thử lại sau khi thất bại tạm thời có thể thực hiện hai hành động đã được phê duyệt. Ví dụ cụ thể: người dùng chấp thuận "chuyển 100 đô la từ A sang B". Mạng nhấp nháy. Thử lại quy trình làm việc. Người dùng đã phê duyệt một lần nhưng quá trình chuyển sẽ thực hiện hai lần. Chìa khóa idempotency gắn sự chấp thuận với một tác dụng phụ duy nhất; lần thực hiện thứ hai là không hoạt động.

Đây là cùng một mô hình idempotency mà Stripe và AWS APIs sử dụng. Việc sử dụng lại nó để phê duyệt agent được nêu rõ trong tài liệu Agent Framework của Microsoft.

### Độ bền: tại sao phê duyệt tồn tại lâu hơn processes

Phòng chờ phê duyệt là một phần của nhà nước mà agent không sở hữu. Quy trình làm việc bị tạm dừng (Bài 12). Khi được phê duyệt, quy trình làm việc sẽ tiếp tục từ chính xác thời điểm đó. Đây là lý do tại sao LangGraph ghép nối `interrupt()` với điểm kiểm tra PostgreSQL chứ không chỉ trạng thái trong bộ nhớ - một phê duyệt hai ngày sau đó vẫn thấy quy trình làm việc còn nguyên vẹn.

### Phê duyệt tem cao su và giảm thiểu thách thức và ứng phó

Giao diện người dùng mặc định cho HITL (nút "Phê duyệt" / "Từ chối") tạo ra phê duyệt nhanh chóng mà không cần xem xét chính hãng. Giảm thiểu được ghi lại: danh sách kiểm tra thử thách và phản hồi yêu cầu câu trả lời tích cực cho các câu hỏi cụ thể trước khi bật nút Phê duyệt. Hình dạng bê tông:

- "Bạn có hiểu điều này chạm đến nguồn tài nguyên nào không? [ ]"
- "Bạn đã xác minh bán kính vụ nổ có thể chấp nhận được không? [ ]"
- "Bạn có kế hoạch rollback nếu điều này thất bại không? [ ]"

Không phải bộ máy quan liêu vì lợi ích của chính nó - một chức năng cưỡng bức. Người đánh giá không thể đánh dấu vào các ô sẽ yêu cầu làm rõ (leo thang) hoặc từ chối (mặc định an toàn). Nghiên cứu an toàn Anthropic agent trích dẫn rõ ràng HITL theo danh sách kiểm tra như một biện pháp giảm thiểu cho các mẫu phê duyệt tem cao su.

### Điều gì được coi là hậu quả

Không phải mọi hành động đều cần đề xuất sau đó commit. Hướng dẫn năm 2026:

- **Hành động do hậu quả** (luôn luôn là HITL): ghi không thể đảo ngược, giao dịch tài chính, giao tiếp ra ngoài production thay đổi cơ sở dữ liệu, hoạt động phá hoại hệ thống tệp.
- **Hành động có thể đảo ngược** (đôi khi là HITL): chỉnh sửa các tệp cục bộ, thay đổi môi trường dàn dựng, ghi có thể đảo ngược với rollback rõ ràng.
- **Đọc và kiểm tra** (không bao giờ HITL): đọc tệp, liệt kê tài nguyên, gọi API chỉ đọc.

### Xác minh sau hành động

"commit chạy" không giống như "tác dụng phụ đã xảy ra". Phân vùng mạng và điều kiện chạy đua có thể tạo ra một quy trình làm việc nghĩ rằng nó đã thành công trong khi phần phụ trợ không tồn tại. Bước xác minh sẽ đọc lại tài nguyên đích sau commit để xác nhận. Đây là mẫu tương tự như các giao dịch cơ sở dữ liệu có mệnh đề `RETURNING` hoặc AWS `GetObject` sau `PutObject`.

### Đạo luật AI EU Điều 14

Điều 14 yêu cầu giám sát hiệu quả của con người đối với các hệ thống AI có nguy cơ cao ở EU. "Hiệu quả" không phải là trang trí. Ngôn ngữ quy định đặc biệt loại trừ các mẫu tem cao su. Đề xuất sau đó commit với thách thức và phản hồi là hình thức tồn tại sau khi xem xét kỹ lưỡng Điều 14 trong tài liệu tuân thủ Bộ công cụ quản trị Microsoft Agent.

## Ứng dụng

`code/main.py` triển khai một máy trạng thái đề xuất sau đó commit trong Python stdlib. Durable store là một tệp JSON. Khóa Idempotency là một hàm băm của (thread_id, action_signature). Trình điều khiển mô phỏng ba trường hợp: quy trình phê duyệt sạch, thử lại sau khi lỗi tạm thời (không được thực hiện hai lần) và mặc định tem cao su so với quy trình thử thách và phản hồi.

## Sản phẩm bàn giao

`outputs/skill-hitl-design.md` xem xét quy trình làm việc HITL được đề xuất cho hình dạng đề xuất sau đó commit và gắn cờ các lớp siêu dữ liệu, idempotency, xác minh hoặc thử thách và phản hồi bị thiếu.

## Bài tập

1. Chạy `code/main.py`. Xác nhận rằng việc thử lại đề xuất đã được phê duyệt sử dụng bản ghi lâu dài và không thực hiện lại. Bây giờ thay đổi khóa idempotency để bao gồm dấu thời gian và hiển thị các lần thực hiện thử lại.

2. Mở rộng bản ghi đề xuất bằng trường `rollback`. Mô phỏng một quá trình thực thi có bước xác minh không thành công. Hiển thị rollback tự động kích hoạt.

3. Đọc tài liệu `RequestInfoEvent` của Microsoft Agent Framework. Xác định một trường siêu dữ liệu mà API bao gồm mà công cụ đồ chơi bị thiếu. Thêm nó và giải thích những gì nó bảo vệ.

4. Thiết kế danh sách kiểm tra thử thách và phản hồi cho một hành động cụ thể (ví dụ: "đăng lên tài khoản Twitter công khai"). Người phản biện phải trả lời ba câu hỏi nào? Tại sao lại là ba điều đó?

5. Chọn một trường hợp đồng bộ "Phê duyệt?" prompt là đủ (không cần kho bền). Giải thích lý do tại sao và nêu tên rủi ro class bạn đang chấp nhận.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|---|---|---|
| Đề xuất sau đó commit | "Phê duyệt hai giai đoạn" | Đề xuất liên tục + commit tích cực + xác minh |
| Phím Idempotency | "Thử lại an toàn token" | Duy nhất cho mỗi đề xuất; Thực hiện lần thứ hai không hoạt động |
| Dòng dữ liệu | "Nó đến từ đâu" | Nội dung nguồn cụ thể dẫn đến đề xuất |
| Bán kính vụ nổ | "Trường hợp xấu nhất" | Phạm vi ảnh hưởng nếu hành động xảy ra sai |
| Tem cao su | "Phê duyệt nhanh" | Nhấp vào "Phê duyệt" mà không có đánh giá chính hãng |
| Thách thức và ứng phó | "Danh sách kiểm tra cưỡng bức" | Người phản biện phải tích cực thừa nhận các câu hỏi cụ thể |
| RequestInfoSự kiện | "Cô Agent Framework primitive" | Yêu cầu HITL bền bỉ với siêu dữ liệu có cấu trúc |
| `interrupt()` / `waitForApproval()` | "Framework primitives" | LangGraph / Cloudflare tương đương có cùng hình dạng |

## Đọc thêm

- [Microsoft Agent Framework — Human in the loop](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) - phê duyệt `RequestInfoEvent`, lâu dài.
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) - các vật thể `waitForApproval()` và bền.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) - HITL như một biện pháp giảm thiểu rủi ro dài hạn.
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) — đường cơ sở quy định cho các hệ thống có rủi ro cao.
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) - khung hiến pháp xung quanh việc giám sát.
