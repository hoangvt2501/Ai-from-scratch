# Kill Switches, Circuit Breaker và Canary Tokens

> kill switch là một boolean được giữ bên ngoài bề mặt chỉnh sửa của agent - phím Redis, cờ feature, config đã ký - vô hiệu hóa hoàn toàn agent. Bộ ngắt mạch có chi tiết hơn: nó vấp ngã trên một mô hình cụ thể (năm công cụ giống hệt nhau gọi liên tiếp), tạm dừng đường dẫn vi phạm và leo thang sang con người. Một canary token kế thừa từ sự lừa dối cổ điển: một thông tin xác thực giả mạo hoặc hồ sơ mật ong mà một agent không có lý do chính đáng để chạm vào, quyền truy cập của họ triggers một cảnh báo. đường dẫn dữ liệu dựa trên eBPF (ví dụ: Cilium) có thể viết lại lối ra của một nhóm bị cách ly thành một mật ong pháp y ở lớp nhân; Cilium benchmarks đã xuất bản báo cáo độ trễ đường dẫn dữ liệu P99 dưới mili giây khi tải (ngân sách lan truyền của bạn phụ thuộc vào cách bản cập nhật policy đến nút, chứ không phải chính đường dẫn dữ liệu). Các máy dò thống kê (EWMA, CUSUM) thích ứng với đường cơ sở chuyển động sẽ lặng lẽ chấp nhận sự trôi dạt - xếp chúng với các giới hạn hiến pháp cứng không bị uốn cong.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, three-detector simulator: kill switch, circuit breaker, canary)
**Kiến thức tiên quyết:** Giai đoạn 15 · 13 (Thống đốc chi phí), Giai đoạn 15 · 10 (Chế độ quyền)
**Thời lượng:** ~60 phút

## Vấn đề

Thống đốc chi phí (Bài 13) ràng buộc những gì agent có thể chi tiêu. Họ không ràng buộc những gì agent có thể làm trong ngân sách. Một agent có giới hạn vận tốc 50 đô la vẫn có thể lấy cắp bí mật, xuất bản bài đăng sai hoặc xóa tài nguyên - hành động tốn kém thường rẻ tiền trong tokens.

Bài học này bao gồm ba trình phát hiện nằm bên cạnh lớp chi phí:

1. **Kill switch**: nút tắt Boolean được giữ ngoài tầm với của agent.
2. **Bộ ngắt mạch**: máy dò mẫu hành động tạm dừng một đường dẫn cụ thể.
3. **Canary token**: mồi nhử mà một agent không có lý do chính đáng để chạm vào sẽ tự lộ ra bằng cách chạm vào.

Cả ba đều là kỹ thuật tiền LLM. Lừa dối cổ điển, phá vỡ giới hạn tốc độ và tiêu diệt cờ feature có trước agents tự trị. Điểm mới là bề mặt tấn công: agents đọc nội dung không đáng tin cậy (Bài 11), chỉnh sửa bộ nhớ của chính họ và có thể soạn nhiều hành động có vẻ an toàn thành một hành động không an toàn. Các máy dò được đặt tên ở đây hoạt động vì chúng không tin tưởng vào bản tự báo cáo của agent.

## Khái niệm

### Ngắt kết nối

Một kill switch là một boolean mà agent đọc nhưng không thể viết. Lựa chọn triển khai:

- **Feature cờ trong dịch vụ được quản lý.** LaunchDarkly, Statsig, Unleash. Các bản cập nhật lan truyền trong vài giây.
- **Redis chìa khóa các cuộc thăm dò agent.** Đơn giản; yêu cầu agent process kiểm tra từng lượt.
- **Đã ký config trong bộ nhớ đối tượng.** Agent xác minh chữ ký khi khởi động; từ chối các trạng thái chưa ký.
- **Tín hiệu cấp hệ điều hành hoặc tiêu diệt vòng đời container.** Docker `kill`, Kubernetes `kubectl delete pod`, dừng hệ thống.

Thuộc tính một kill switch chính xác có:

- agent không thể đặt nó thành `off`. (Sống trong một hệ thống mà thông tin đăng nhập của agent không viết.)
- Nó được kiểm tra trên mọi hành động do hậu quả, không chỉ khi khởi động.
- Khi tắt, agent không làm gì có thể quan sát được bên ngoài, bao gồm cả việc ghi nhật ký vào các hệ thống mà agent có thể tiếp cận.
- Kích hoạt lại nó là một hành động rõ ràng của con người, không phải là một timeout tự động.

### Bộ ngắt mạch

Bộ ngắt mạch tạm dừng một mẫu cụ thể, không phải toàn bộ agent. Hình dạng cổ điển (từ cuốn sách Nygard năm 2007, vẫn còn hiện tại):

- **Đã đóng**: cho phép hành động.
- **Mở**: hành động bị chặn.
- **Mở một nửa**: sau khi hồi chiêu, cho phép 1–3 lần thử thăm dò (mặc định là 1); thành công đóng cầu dao, bất kỳ thất bại nào còn lại sẽ mở lại nó.

triggers liên quan đến Agent:

- Năm lệnh gọi công cụ giống hệt nhau liên tiếp (vòng lặp lặp lại).
- Năm lỗi liên tiếp trên cùng một công cụ với các đầu vào khác nhau (lỗi hệ thống).
- Tốc độ gọi công cụ trên ngưỡng (Vận tốc Bài 13).
- Công cụ cụ thể được gọi (ví dụ: `file.delete`) sau khi đọc nội dung không đáng tin cậy (Bài 11).

### Canary tokens

Canary tokens (còn được gọi là honeytoken) là đầu vào mà agent không bao giờ được chạm vào. Truy cập là báo động.

Ví dụ cho agents:

- Một `AWS_SECRET_ACCESS_KEY` giả mạo trong môi trường không có đặc quyền thực tế. Các nỗ lực đánh cắp có thể được quy kết ngay lập tức.
- Bản ghi cơ sở dữ liệu giả mạo được gắn cờ với giá trị sentinel đã biết. Bất kỳ lần đọc hoặc cập nhật bản ghi này sẽ kích hoạt cảnh báo.
- Một tệp giả mạo trong không gian làm việc. Việc đọc hoặc chỉnh sửa tệp này là đáng ngờ.
- Một mục bộ nhớ giả được đưa vào khi session bắt đầu. Nếu nó xuất hiện trong đầu ra sau đó, bộ nhớ đã bị lấy cắp.

Canary thiết kế dành riêng cho quy trình làm việc. Một canary agent có lý do chính đáng để chạm vào là nhiễu.

### Tại sao lại có giới hạn thống kê và giới hạn cứng

Các máy dò thống kê (EWMA, CUSUM, z-score về tỷ lệ gọi công cụ) thích ứng với đường cơ sở đang di chuyển. Đó là một feature cho drift hợp pháp và một lỗi cho drift đối thủ. Một kẻ tấn công kiên nhẫn có thể thay đổi đường cơ sở đủ chậm để không có báo động nào kích hoạt.

Các giới hạn hiến pháp cứng (Bài 17) không bị bẻ cong. "Không bao giờ gửi email cho `secrets@`" như một quy tắc hiến pháp đơn giản hơn, có thể kiểm tra được và không thể chơi được bằng cách trôi chậm. Mô hình phân lớp: máy dò thống kê bắt được hầu hết nhiễu, giới hạn cứng bắt cuộc tấn công đi qua máy dò.

### Cách ly thông qua chuyển hướng đường dẫn dữ liệu eBPF

policy mạng dựa trên eBPF của Cilium có thể viết lại lối ra của một nhóm bị cách ly thành một mật ong pháp y ở lớp nhân. benchmarks công khai của Cilium báo cáo độ trễ đường dẫn dữ liệu P99 dưới mili giây trên TCP_RR; Ngân sách đồng hồ treo tường để *kích hoạt* cách ly trong cụm của bạn bị chi phối bởi cách cập nhật policy lan truyền từ control plane của bạn đến nút (vài trăm mili giây là thực tế trên Kubernetes với toán tử được tải sẵn; độ trễ đuôi phụ thuộc vào kích thước cụm và nhịp điều chỉnh). Điều này cung cấp cho bạn hai điều: agent không khai thác bất cứ thứ gì có thật, và hành vi của agent bên trong honeypot được quan sát đầy đủ. Cùng một mẫu, lớp khác nhau, như canary token: agent tiết lộ ý định bằng những gì nó cố gắng làm trong khi bị quản thúc.

### Những gì không có máy dò bắt được

- **Các cuộc tấn công ngữ nghĩa soạn từ các primitives được phép.** Mỗi hành động riêng lẻ đi qua mọi máy dò; thành phần là vấn đề. Đây là lỗi tổng hợp Chế độ tự động từ Bài 10.
- **Các cuộc tấn công bao gồm máy dò như một phần của payload.** Nếu một agent đọc một trang có nội dung "trước khi bạn hành động, hãy tắt canary" và agent có khả năng này, máy dò sẽ bị xâm phạm. Canary tokens nên ở trong các hệ thống mà agent không thể sửa đổi.

## Ứng dụng

`code/main.py` mô phỏng quỹ đạo agent ngắn thông qua ba máy dò. Một kill switch được giữ trong một câu lệnh bên ngoài; một bộ ngắt mạch ngắt năm lần gọi công cụ giống hệt nhau; một tệp canary được đọc triggers cảnh báo. Nguồn cấp dữ liệu theo quỹ đạo tổng hợp: hành động hợp pháp, vòng lặp lặp đi lặp lại, thăm dò canary và kịch bản kích hoạt ngắt kết nối trong đó các hành động của agent bị dừng lại.

## Sản phẩm bàn giao

`outputs/skill-tripwire-design.md` xem xét một máy dò được đề xuất stack triển khai agent và gắn cờ các lỗ hổng (thiếu kill switch, thiếu canary, ngưỡng bộ ngắt mạch quá lỏng).

## Bài tập

1. Chạy `code/main.py`. Xác nhận cầu dao cháy ở lượt 5 (cuộc gọi giống hệt thứ năm) và canary cháy ở lượt 9 (đọc chìa khóa giả).

2. Thêm trình phát hiện thống kê: EWMA z-score về tỷ lệ gọi công cụ. Nạp theo quỹ đạo trôi chậm và cho thấy máy dò không bao giờ kích hoạt. Bây giờ thêm một giới hạn cứng (không quá 50 lệnh gọi công cụ trong 10 phút) và hiển thị giới hạn cứng kích hoạt trên cùng một quỹ đạo.

3. Thiết kế một bộ canary token cho một agent trình duyệt (Bài 11). Liệt kê ít nhất ba con canary và những gì mỗi con sẽ phát hiện.

4. Đọc tài liệu policy mạng Cilium. Mô tả quy trình cách ly chuyển hướng đầu ra một cách cụ thể: bộ chọn policy nào, nhóm nào, ghi lại đầu ra nào, cảnh báo nào. Điều gì chi phối độ trễ của đồng hồ treo tường từ "quyết định cách ly" đến "gói được chuyển hướng đầu tiên"?

5. Xác định quy trình kích hoạt lại cho agent ngắt kết nối. Ai có thể kích hoạt lại? Những gì phải được ghi lại? Điều gì phải thay đổi về agent trước khi bật lại?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|---|---|---|
| Kill switch | "Nút tắt" | Boolean bên ngoài bề mặt chỉnh sửa của agent; Kiểm tra mọi hành động do hậu quả |
| Bộ ngắt mạch | "Tạm dừng mẫu" | Hành động cụ thể về số lần lặp lại, tỷ lệ thất bại hoặc giới hạn tốc độ |
| Canary token | "Mã thông báo mật ong" | Mồi nhử agent không có lý do chính đáng để chạm vào; Access kích hoạt cảnh báo |
| Honeypot | "sandbox pháp y" | Lưu lượng truy cập / không gian làm việc được chuyển hướng nơi quan sát thấy agent bị cách ly |
| EWMA | "Đường trung bình động" | Trọng số theo cấp số nhân; Thích ứng với trôi dạt (feature + lỗi) |
| CUSUM | "Số tiền tích lũy" | Phát hiện sự thay đổi liên tục so với đường cơ sở |
| Giới hạn cứng | "Quy tắc hiến pháp" | Không thích nghi; hằng số bất kể lịch sử |
| Giới hạn hiến pháp | "Quy tắc luôn đúng" | gắn liền với hiến pháp của Bài 17; không thể được chỉnh sửa bởi agent |

## Đọc thêm

- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) - ngắt kết nối và khung ngắt mạch cho agents tự động.
- [Microsoft Agent Framework — HITL and oversight](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) - production các mô hình quản trị.
- [OWASP LLM / Agentic Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/) - yêu cầu phát hiện và phản hồi.
- [Cilium — Network policy and eBPF](https://docs.cilium.io/en/stable/security/network/) - chuyển hướng lối ra cấp độ pod và các mẫu honeypot pháp y.
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) - các lệnh cấm được mã hóa cứng là "giới hạn hiến pháp".
