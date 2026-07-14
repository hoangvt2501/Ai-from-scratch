# Checkpoints và Rollback

> Mọi chuyển đổi trạng thái đồ thị vẫn tồn tại. Khi một worker gặp sự cố, hợp đồng thuê của nó sẽ hết hạn và một worker khác sẽ bắt đầu vào checkpoint muộn nhất. Cloudflare Durable Objects giữ trạng thái trong nhiều giờ hoặc vài tuần. Đề xuất sau đó commit (Bài 15) xác định một kế hoạch rollback cho mỗi hành động. Xác minh sau hành động đóng vòng lặp. Điều 14 của Đạo luật AI EU quy định sự giám sát hiệu quả của con người là bắt buộc đối với các hệ thống có rủi ro cao - trong thực tế, điều này có nghĩa là checkpoints phải có thể truy vấn được, việc khôi phục phải được diễn tập và dấu vết kiểm toán phải tồn tại sau khi triển khai. Chế độ lỗi mạnh: không có khóa idempotency và kiểm tra điều kiện tiên quyết, thử lại sau khi thất bại tạm thời có thể thực hiện hai hành động đã được phê duyệt. Xác minh sau hành động là thứ nắm bắt nó.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, checkpoint and rollback state machine)
**Kiến thức tiên quyết:** Giai đoạn 15 · 12 (Thực thi bền bỉ), Giai đoạn 15 · 15 (Đề xuất sau đó commit)
**Thời lượng:** ~60 phút

## Vấn đề

Thực thi bền bỉ (Bài 12) làm cho một agent bị treo có thể tiếp tục được. Đề xuất sau đó commit (Bài 15) làm cho một hành động đã được phê duyệt có thể kiểm toán được. Bài học này kết hợp với họ: điều gì xảy ra khi một hành động được phê duyệt thực hiện một phần, gặp sự cố và tiếp tục? Khi nào rollback chạy, và ngược lại trạng thái nào?

Các hệ thống thực tế kết nối điều này theo cách khác:

- **LangGraph** checkpoints mọi chuyển đổi trạng thái đồ thị sang PostgreSQL. Khi worker vụ sụp đổ, hợp đồng thuê được giải phóng và một worker khác tiếp tục vào checkpoint muộn nhất. Quy trình làm việc tạm dừng trên `interrupt()`, bản thân quy trình này vẫn tồn tại.
- **Cloudflare Durable Objects** giữ trạng thái trên mỗi phím trong nhiều giờ hoặc vài tuần. Cùng định vị tính toán với bộ lưu trữ cho hành động đã được phê duyệt.
- **Microsoft Agent Framework** hiển thị `Checkpoint` primitives trong API quy trình làm việc; phát lại cộng với idempotency bao gồm các lần thử lại.

Trong mọi trường hợp, sự kết hợp thực sự hoạt động là: khóa idempotency (ngăn chặn thực thi kép) + kiểm tra điều kiện trước (trạng thái vẫn là những gì chúng tôi đã chấp thuận) + xác minh sau hành động (tác dụng phụ thực sự đã xảy ra) + rollback khi xác minh-thất bại.

## Khái niệm

### Mọi quá trình chuyển đổi vẫn tồn tại

Chuyển đổi trạng thái đồ thị là bất kỳ bước nào di chuyển quy trình làm việc từ trạng thái được đặt tên này sang trạng thái được đặt tên khác. Việc triển khai ngây thơ chỉ tồn tại tại các điểm commit cụ thể; production triển khai vẫn tồn tại sau mỗi quá trình chuyển đổi. Chi phí (một vài lần ghi thêm) là nhỏ so với mức tăng độ tin cậy (phát lại ở bất cứ đâu, thu hồi hợp đồng thuê là chính xác).

### Thu hồi hợp đồng thuê

Khi một worker gặp sự cố, quy trình làm việc không bị mất; hợp đồng thuê (một tuyên bố ngắn hạn rằng worker này đang thực hiện lần chạy này) chỉ đơn giản là hết hạn. Một worker khác nhận checkpoint mới nhất và tiếp tục. Cơ chế cho thuê là thứ cho phép các hệ thống production tồn tại sau khi triển khai luân phiên mà không làm mất công việc trên chuyến bay.

### Idempotency cộng với điều kiện tiên quyết

Chỉ riêng idempotency là không đủ. Hãy xem xét: quy trình làm việc được phê duyệt để "chuyển $100 from A to B when balance > $1000". Quy trình làm việc được cam kết, gặp sự cố giữa quá trình thực thi và tiếp tục. Nếu chỉ khóa idempotency được chọn và quá trình thực thi tiếp tục, quá trình chuyển sẽ chạy một lần (đúng). Nhưng hãy xem xét rằng giữa sự cố và tiếp tục, số dư của A giảm xuống còn 500 đô la thông qua một quy trình làm việc khác. Kiểm tra idempotency vẫn được thông qua; Điều kiện tiên quyết thì không. Nếu không có kiểm tra điều kiện tiên quyết, chúng tôi ship thấu chi.

Mọi hành động do hậu quả đều cần cả hai:

- **Khóa Idempotency**: ngăn chặn thực thi kép.
- **Kiểm tra điều kiện tiên quyết**: xác nhận trạng thái vẫn phù hợp với những gì đã được phê duyệt.

### Xác minh sau hành động

"Công cụ trả về 200" không phải là xác minh. Xác minh thực đọc lại trạng thái mục tiêu và xác nhận tác dụng phụ thực sự đã xảy ra. Các mẫu:

- Cập nhật cơ sở dữ liệu: `UPDATE ... RETURNING *` sau đó xác nhận hàng trả về khớp với trạng thái dự kiến.
- Gửi email: kiểm tra thư mục gửi để biết ID thư sau khi gửi.
- Ghi tệp: đọc lại tệp và băm nó.
- Cuộc gọi API: `GET` theo dõi về nguồn lực đích.

Nếu xác minh không thành công, quy trình làm việc sẽ ở trạng thái đã biết-xấu. Rollback tham gia.

### Rollback kế hoạch

Mọi hành động do hậu quả trong đề xuất sau đó commit (Bài 15) đều mang một kế hoạch rollback. Các loại:

- **rollback trong băng tần**: đảo ngược tác dụng phụ trực tiếp (`DELETE` sau khi `INSERT`, `Send-correction-email` sau khi gửi).
- **Giao dịch bù đắp**: một hành động mới vô hiệu hóa bản gốc (mô hình SAGA tiêu chuẩn).
- **Out-of-band rollback**: cảnh báo con người, tạm dừng quy trình làm việc, để trạng thái xấu để điều tra.

No-op rollback ("chúng tôi không thể hoàn tác điều này") phải được nêu tên trong đề xuất. Các hành động không có rollback yêu cầu HITL mạnh hơn tại commit thời điểm (Bài 15 thách thức và phản hồi).

### Đạo luật AI EU Điều 14 đọc hoạt động

Điều 14 yêu cầu "sự giám sát hiệu quả của con người" đối với các hệ thống có rủi ro cao. Về mặt hoạt động, người triển khai đọc nó là:

- Checkpoints có thể truy vấn bởi kiểm toán viên.
- Các lần khôi phục được diễn tập (thử nghiệm từ đầu đến cuối ít nhất một lần).
- Dấu vết kiểm tra tồn tại sau khi triển khai (checkpoint phụ trợ không phải là tạm thời).
- Xác minh không thành công sẽ được cảnh báo, không được ghi lại một cách âm thầm.

Quy trình làm việc gặp sự cố giữa commit, tiếp tục và hoàn thành tác dụng phụ mà không có lộ trình xác minh + rollback sẽ không tồn tại qua thử nghiệm Điều 14.

### Chế độ thất bại mạnh: thực hiện kép

Sự cố production phổ biến nhất trong không gian này:

1. Hành động được phê duyệt, khóa idempotency k.
2. Commit bắt đầu, thực thi, trả về 200.
3. Quy trình làm việc gặp sự cố trước khi duy trì trạng thái "đã cam kết".
4. Tiếp tục quy trình làm việc; thấy "được chấp thuận nhưng không cam kết"; thực hiện lại.
5. Tác dụng phụ bắn hai lần.

Giảm thiểu: duy trì ý định "trong chuyến bay" trước khi thực thi, thực thi bằng khóa idempotency sau đó đánh dấu "đã cam kết" chỉ sau khi xác minh sau hành động thành công. Nếu hành động kích hoạt và ghi trạng thái không thành công, bạn biết xác minh và (nếu cần) kích hoạt lại. Nếu ghi trạng thái thành công và hành động không thành công, bạn sẽ xác minh và kích hoạt chính xác một lần thông qua đường dẫn khôi phục.

## Ứng dụng

`code/main.py` thực hiện quy trình làm việc có điểm kiểm tra với idempotency, điều kiện tiên quyết, xác minh và rollback. Trình điều khiển mô phỏng bốn tình huống: chạy sạch, thử lại sau khi gặp sự cố (idempotency bắt), điều kiện tiên quyết thất bại (quy trình làm việc hủy bỏ mà không kích hoạt), xác minh thất bại (rollback cháy).

## Sản phẩm bàn giao

`outputs/skill-rollback-rehearsal.md` thiết kế một bài kiểm tra diễn tập rollback cho quy trình làm việc được đề xuất và kiểm tra phần phụ trợ checkpoint cho persistence theo dõi kiểm tra.

## Bài tập

1. Chạy `code/main.py`. Xác minh bốn tình huống. Đối với trường hợp sự cố trong commit, hãy xác nhận hành động kích hoạt chính xác một lần sau khi thử lại.

2. Sửa đổi mẫu "đánh dấu là xong trước, sau đó thực hiện" để ghi trạng thái kích hoạt sau hành động. Chạy lại kịch bản sự cố. Đo lường số lượng hành động trùng lặp kích hoạt.

3. Thiết kế kế hoạch rollback cho một hành động production cụ thể (ví dụ: "đăng lên kênh Slack"). Phân loại là trong băng tần, bù hoặc ngoài băng tần. Biện minh cho sự lựa chọn.

4. Hãy lấy một quy trình làm việc mà bạn biết. Xác định mọi chuyển đổi trạng thái. Đánh dấu mỗi yêu cầu về độ bền (kiên trì / không tồn tại). Đếm những cái bạn hiện không kiên trì.

5. Kiểm tra rollback diễn tập: thiết kế một bài kiểm tra đầu cuối chạy quy trình làm việc thực, gặp sự cố và xác nhận đường dẫn rollback kích hoạt. Bài kiểm tra khẳng định điều gì?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|---|---|---|
| Checkpoint | "Điểm lưu" | Mọi chuyển đổi trạng thái đồ thị vẫn tồn tại trong một cửa hàng bền vững |
| Cho thuê | "Worker yêu cầu" | Tuyên bố ngắn hạn rằng một worker đang thực hiện một cuộc chạy; Hết hạn khi gặp sự cố |
| Điều kiện tiên quyết | "Cổng nhà nước" | Khẳng định rằng nhà nước vẫn nhất quán với hành động đã được phê duyệt |
| Xác minh sau hành động | "Kiểm tra đọc lại" | Xác nhận tác dụng phụ thực sự xảy ra trong hệ thống mục tiêu |
| rollback trong băng tần | "Hoàn tác trực tiếp" | Đảo ngược tác dụng phụ với thao tác nghịch đảo |
| Giao dịch bồi thường | "Hoàn tác SAGA" | Một hành động mới vô hiệu hóa bản gốc |
| Đánh dấu là xong trước | "Lệnh ghi trạng thái" | Duy trì trạng thái đã cam kết trước khi quay lại từ commit |
| Điều 14 | "Đạo luật AI của EU giám sát con người" | Hoạt động: checkpoints có thể truy vấn, khôi phục diễn tập, dấu vết có thể kiểm tra |

## Đọc thêm

- [Microsoft Agent Framework — Checkpointing and HITL](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) - thu hồi checkpoint primitives và cho thuê.
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — Đối tượng bền bỉ làm chất nền trạng thái.
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) - đường cơ sở quy định.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — khung hình đáng tin cậy cho quy trình làm việc dài hạn.
- [Anthropic — Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) — hình dạng quy trình làm việc cho Claude Code Routines.
