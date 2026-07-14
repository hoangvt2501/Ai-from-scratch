# Capstone 82 — Phân loại bẻ khóa

> Một harness an toàn không có phân loại là tung đồng xu. Đặt tên cho cuộc tấn công trước khi bạn bảo vệ nó.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Bài học an toàn Giai đoạn 18, Bài học Giai đoạn 19 Bài A 25-29
**Thời lượng:** ~90 phút

## Vấn đề

Một model được triển khai mà không có cuộc tấn công model là một model được bảo vệ chống lại không có gì đặc biệt. Các nhà khai thác đọc một thread Twitter, nhận ra thủ thuật, viết một regex, ship nó và tiếp tục. prompt tiếp theo là một diễn giải. Biểu thức chính quy bị bỏ lỡ. Một tuần sau, ai đó chỉ ra thủ thuật tương tự được bọc trong base64 và toán tử viết biểu thức chính quy thứ hai. Đến tháng thứ ba, hệ thống có 40 quy tắc được vá lỗi, không có từ vựng được chia sẻ, không có cách nào để nói về cuộc tấn công thực sự là gì và tồn đọng phát triển nhanh hơn các bản vá.

Trước khi bất kỳ trình phát, bộ phân loại hoặc công cụ quy tắc nào trong bài hát này thực hiện bất kỳ điều gì hữu ích, nhóm cần một cách chung để gắn nhãn các cuộc tấn công. Không phải vì nhãn ngăn chặn các cuộc tấn công, mà vì nhãn biến luồng tấn công thành biểu đồ. Biểu đồ trở thành biểu đồ bao phủ. Biểu đồ độ bao phủ thúc đẩy sprint tiếp theo. Các harness trong bài 83-87 dành thời gian để quyết định xem một prompt có phải là một cuộc tấn công nhập vai chống lại một policy từ chối hay một cuộc tấn công buôn lậu ngữ cảnh chống lại một công cụ. Quyết định đó là không thể nếu không có phân loại.

Capstone này xác định một phân loại sáu loại đủ rộng để bao gồm hầu hết các cuộc tấn công được thấy trong tự nhiên, đủ hẹp để hai người đánh giá thường đồng ý về thể loại và đủ cụ thể để mỗi danh mục có ít nhất bảy đồ đạc được chế tạo bằng tay. Phân loại là sóng mang cho mọi thứ ở hạ lưu.

## Khái niệm

Sáu loại cắt dọc theo một trục duy nhất: cuộc tấn công lạm dụng ranh giới tin cậy nào? Mỗi tên tương ứng với một ranh giới.

| Thể loại | Ranh giới tin cậy bị lạm dụng |
|---|---|
| Nhập vai | tính cách của trợ lý |
| instruction-override | thẩm quyền của system prompt |
| buôn lậu ngữ cảnh | Khoảng cách giữa nội dung người dùng và nội dung hướng dẫn |
| nhiều rãnh-ramp | Lịch sử cuộc trò chuyện dưới dạng hợp đồng |
| thủ thuật mã hóa | Dạng bề mặt của tokens bị cấm |
| tiền tố-tiêm | Quyết định token tiếp theo của trợ lý |

Một cuộc tấn công nhập vai định hình lại trợ lý như một agent khác ("bạn là một model nghiên cứu không hạn chế được gọi là QX") để các quy tắc từ chối gắn liền với tính cách ban đầu không còn kích hoạt nữa. Ghi đè hướng dẫn prompts nói "bỏ qua các hướng dẫn trước đó" và cố gắng ghi đè trực tiếp lên system prompt. Buôn lậu ngữ cảnh ẩn các hướng dẫn bên trong những gì trông giống như dữ liệu: một tài liệu được dán, một kết quả công cụ, một khối mã. Đường dốc nhiều lượt làm ấm model bằng những khúc cua vô hại và sau đó đi xuống sàn từng bước một, khai thác xu hướng nhất quán với cuộc trò chuyện của model. Thủ thuật mã hóa (base64, rot13, leet-speak, zero-width insertion) ẩn tokens bị cấm khỏi các bộ lọc từ khóa ngây thơ. Prefix-injection kết thúc prompt bằng "Chắc chắn, đây là cách làm" để model tiếp tục từ câu trả lời giả định thay vì từ chối.

```mermaid
flowchart TB
  P[prompt] --> C{which boundary}
  C -->|persona| RP[role-play]
  C -->|system prompt| IO[instruction-override]
  C -->|data vs instruction| CS[context-smuggling]
  C -->|history| MT[multi-turn-ramp]
  C -->|surface form| ET[encoding-trick]
  C -->|continuation| PI[prefix-injection]
  RP --> H[(histogram)]
  IO --> H
  CS --> H
  MT --> H
  ET --> H
  PI --> H
  H --> D[coverage chart]
```

Mỗi lịch thi đấu là một bản ghi với `id`, `category`, `subtype`, `prompt`, `target_behavior` và `severity`. Đối tượng phân loại tải các đồ đạc, nhóm chúng theo danh mục và hiển thị một `match` API: cho một prompt ứng viên, trả về vật cố định gần nhất và danh mục của nó. Match là cosin tam giác ký tấu: thô, nhanh, không phụ thuộc. Nó không phải là một máy dò. Máy dò nằm trong bài 83. Đây là nhà sản xuất nhãn.

Mức độ nghiêm trọng theo thang điểm 1-5. 1 là một cuộc tấn công vụng về chống lại một mục tiêu lành tính ("vui lòng giả vờ là cướp biển"). 5 là một cuộc tấn công, nếu thành công, tạo ra đầu ra mà hệ thống được triển khai không được phát ra (chi tiết hoạt động cho một hoạt động nguy hiểm). Hầu hết các trận đấu đều ở mức 2-3 vì các cuộc tấn công thực sự ở quy mô triển khai nghiêng về phía dễ dàng và lười biếng. Mức độ nghiêm trọng được thiết lập bởi tác giả cố định. Hai người phản biện không đồng ý với nhiều hơn một thứ hạng là một dấu hiệu cho thấy bảng đánh giá cần được mài giũa.

## Tự xây dựng

Kho dữ liệu tồn tại trong `code/fixtures.py` như một danh sách Python duy nhất. Phân loại class trong `code/main.py` tải nó, xác nhận rằng mọi danh mục có ít nhất bảy đồ đạc, hiển thị các phương thức `by_category`, `match` và `stats` và ships một bản demo có thể chạy được để in biểu đồ. Trigram cosine được thực hiện từ đầu với `numpy`.

Vượt qua xác thực kiểm tra bốn bất biến: mỗi thiết bị cố định có một prompt không trống, mọi danh mục trong schema được thể hiện, mọi mức độ nghiêm trọng đều `1..5` và mỗi id cố định là duy nhất. Thất bại ở đây là một lối thoát khó khăn, không phải là một cảnh báo, bởi vì rest của bản nhạc phụ thuộc vào kho dữ liệu nhất quán bên trong.

## Ứng dụng

Chạy `python3 main.py` từ thư mục `code/` bài học. Bản demo in số lượng cố định cho mỗi danh mục, chạy ba đầu dò mẫu đối với `match` và ghi `taxonomy.json` vào thư mục đầu ra bài học. Các bài học xuôi dòng đọc `taxonomy.json` thay vì nhập mô-đun Python, vì vậy kho dữ liệu là một artifact ổn định.

## Sản phẩm bàn giao

`outputs/skill-jailbreak-taxonomy.md` tài liệu sáu danh mục và bảng đánh giá. Hãy coi nó như vốn từ vựng chung của nhóm. Mọi phát hiện được ghi lại bởi harness trong bài 87 đều tham chiếu đến một id phân loại.

## Bài tập

1. Thêm một danh mục thứ bảy cho indirect-prompt-injection (lệnh được nhúng trong tài liệu được truy xuất, không phải trong lượt người dùng). Tác giả mười đồ đạc và chạy lại trình xác thực.
2. Thay thế cosin tam giác bằng bộ ghi điểm khoảng cách chỉnh sửa token và đo lường cách phân công trận đấu thay đổi trên kho dữ liệu hiện có.
3. Lấy ba mươi đồ đạc bổ sung từ nhật ký sản phẩm của riêng bạn (đã bị xóa) và xác nhận phân phối danh mục phù hợp với những gì nhóm của bạn mong đợi một cách trực quan.

## Thuật ngữ chính

| Thuật ngữ | Cách sử dụng phổ biến | Ý nghĩa chính xác |
|---|---|---|
| bẻ khóa | bất kỳ đầu ra model không an toàn nào | một prompt tạo ra đầu ra vi phạm một policy đã nêu |
| Phân loại | Danh sách các danh mục | một sự phân chia các cuộc tấn công mà họ lạm dụng ranh giới tin cậy |
| Lịch thi đấu | Một ví dụ thử nghiệm | prompt được gắn nhãn với danh mục, mức độ nghiêm trọng và hành vi mục tiêu |
| Mức độ nghiêm trọng | đầu ra tồi tệ như thế nào | xếp hạng 1-5 cho tác động nếu cuộc tấn công thành công |
| Trận đấu | quyết định phát hiện | Vật cố định gần nhất của Trigram Cosine, được sử dụng để gán một danh mục cho một prompt mới |

## Đọc thêm

Bài học này là điểm khởi đầu. Bài 83-87 được xây dựng trực tiếp trên kho dữ liệu.
