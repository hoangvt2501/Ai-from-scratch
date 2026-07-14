# Agents trình duyệt và các tác vụ web dài hạn

> ChatGPT agent (tháng 7 năm 2025) merged Nhà điều hành và nghiên cứu sâu vào một browser/terminal agent và đặt BrowseComp SOTA ở mức 68,9%. OpenAI đóng cửa Nhà điều hành vào ngày 31 tháng 8 năm 2025 - hợp nhất ở lớp sản phẩm. Thương vụ mua lại Vercept của Anthropic đã chuyển Claude Sonnet trên OSWorld từ dưới 15% xuống 72,5%. WebArena-Verified (ServiceNow, ICLR 2026) đã cố định 11,3 điểm phần trăm tỷ lệ âm tính giả trong WebArena ban đầu và shipped tập con Hard 258 nhiệm vụ. Những con số là có thật. Bề mặt tấn công cũng vậy: Trưởng bộ phận chuẩn bị của OpenAI tuyên bố công khai rằng việc tiêm prompt gián tiếp vào agents trình duyệt "không phải là một lỗi có thể được vá đầy đủ". Các cuộc tấn công được ghi nhận từ năm 2025–2026: Tainted Memories (Atlas CSRF), HashJack (Cato Networks) và các vụ chiếm quyền điều khiển bằng một cú nhấp chuột trong Perplexity Comet.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, indirect prompt-injection attack surface model)
**Kiến thức tiên quyết:** Giai đoạn 15 · 10 (Chế độ cho phép), Giai đoạn 15 · 01 (Đường chân trời dài agents)
**Thời lượng:** ~45 phút

## Vấn đề

agent trình duyệt là một agent dài đọc nội dung không đáng tin cậy và thực hiện các hành động do hậu quả. Mỗi trang mà agent truy cập là một đầu vào mà người dùng không viết. Mỗi biểu mẫu trên mỗi trang là một kênh lệnh tiềm năng. Kho dữ liệu tấn công 2025–2026 cho thấy đây không phải là giả thuyết: Tainted Memories cho phép kẻ tấn công liên kết các hướng dẫn độc hại với bộ nhớ của agent thông qua một trang được tạo ra; HashJack ẩn các lệnh trong các đoạn URL mà agent truy cập; Perplexity vụ cướp sao chổi chỉ bằng một cú nhấp chuột.

Bức tranh phòng ngự không thoải mái. Trưởng bộ phận chuẩn bị của OpenAI cho biết phần yên tĩnh: tiêm prompt gián tiếp "không phải là một lỗi có thể được vá hoàn toàn". Điều này là do cuộc tấn công nằm trong ranh giới đọc và hành động của agent, vốn không rõ ràng về mặt kiến trúc - về nguyên tắc, mọi token model đọc đều có thể được đọc như một hướng dẫn.

Bài học này đặt tên cho bề mặt tấn công, đặt tên cho bối cảnh benchmark (BrowseComp, OSWorld, WebArena-Verified) và models một kịch bản tiêm prompt gián tiếp tối thiểu để bạn có thể suy luận về các biện pháp phòng thủ thực sự trong Bài 14 và 18.

## Khái niệm

### Bối cảnh năm 2026, trong một đoạn văn cho mỗi hệ thống

**ChatGPT agent (OpenAI).** Ra mắt Tháng Bảy 2025. Hợp nhất Operator (duyệt) và Nghiên cứu sâu (nghiên cứu nhiều giờ). Đóng cửa Nhà điều hành độc lập Tháng Tám 31, 2025. SOTA trên BrowseComp ở mức 68,9%; những con số mạnh mẽ trên OSWorld và WebArena-Verified.

**Claude Sonnet + Vercept (Anthropic).** Việc mua lại Vercept của Anthropic tập trung vào khả năng sử dụng máy tính. Di chuyển Claude Sonnet trên OSWorld từ <15% lên 72,5%. Claude Máy tính Sử dụng ships như một công cụ API.

**Gemini 3 Pro với Sử dụng trình duyệt (DeepMind).** Tích hợp sử dụng trình duyệt ships điều khiển sử dụng máy tính; FSF v3 (Tháng 4 năm 2026, Bài học 20) theo dõi quyền tự chủ trong lĩnh vực R&D ML cụ thể.

**Đã xác minh WebArena (ServiceNow, ICLR 2026).** Khắc phục sự cố được ghi chép đầy đủ: WebArena ban đầu có tỷ lệ âm tính giả ~11,3% (các nhiệm vụ được đánh dấu thất bại nhưng thực sự đã được giải quyết). Bản phát hành Đã xác minh chấm điểm lại với các tiêu chí thành công do con người quản lý và thêm một tập hợp con Cứng gồm 258 nhiệm vụ (bài báo ICLR 2026, openreview.net/forum?id=94tlGxmqkN).

### BrowseComp so với OSWorld so với WebArena

| Benchmark | Những gì nó đo lường | Chân trời |
|---|---|---|
| Duyệt Comp | Tìm kiếm các sự kiện cụ thể trên web mở dưới áp lực thời gian | phút |
| Hệ điều hànhTrang web | Agent vận hành máy tính để bàn đầy đủ (chuột, bàn phím, vỏ) | hàng chục phút |
| Đã xác minh WebArena | Tác vụ web giao dịch trong các trang web mô phỏng | phút |
| Tập con cứng | Các tác vụ WebArena-Verified với chuyển đổi trạng thái nhiều trang | hàng chục phút |

Các trục khác nhau. Điểm BrowseComp cao cho biết agent tìm thấy sự thật; Nó không nói rằng agent có thể đặt một chuyến bay. Điểm OSWorld gần với "nó có hoạt động trên máy tính để bàn của tôi không". WebArena-Verified gần với "nó có thể hoàn thành một luồng không". Bất kỳ quyết định production nào cũng cần benchmark phù hợp với phân phối nhiệm vụ.

### Bề mặt tấn công, được đặt tên là

1. **Chèn prompt gián tiếp.** Nội dung trang không đáng tin cậy chứa hướng dẫn. Người agent đọc chúng. Người agent thực hiện chúng. Ví dụ công khai: 2024 Kai Greshake và cộng sự, 2025 Bài báo Tainted Memories, 2026 HashJack (Cato Networks).
2. **Đoạn URL / chèn truy vấn.** `#fragment` hoặc chuỗi truy vấn của URL được thu thập dữ liệu chứa các lệnh. Không bao giờ hiển thị rõ ràng; vẫn nằm trong bối cảnh của agent.
3. **Các cuộc tấn công liên kết bộ nhớ.** Page hướng dẫn agent ghi một bộ nhớ liên tục (Bài 12 đề cập đến trạng thái bền bỉ). session tiếp theo, bộ nhớ sẽ kích hoạt payload mà không có trigger nhìn thấy được.
4. **Các cuộc tấn công hình dạng CSRF vào sessions đã xác thực.** Tainted Memories class: agent đăng nhập ở đâu đó; Trang của kẻ tấn công đưa ra các yêu cầu thay đổi trạng thái mà agent thực thi bằng cookie của người dùng.
5. **Chiếm đoạt bằng một cú nhấp chuột.** Một nút vô hại trực quan đi payload agent theo sau. Sao chổi class.
6. **Content-Security-Policy lỗ hổng trên bề mặt máy chủ của agent.** Bản thân các lớp kết xuất và công cụ có thể bị tấn công vectors; trình duyệt trong trình duyệt agent stack rộng.

### Tại sao "không thể vá lỗi hoàn toàn"

Cuộc tấn công là đồng cấu với khả năng của agent. agent phải đọc nội dung không đáng tin cậy để thực hiện công việc của mình. Bất kỳ nội dung nào mà agent đọc đều có thể chứa hướng dẫn. Bất kỳ hướng dẫn nào mà agent sau có thể không phù hợp với yêu cầu thực tế của người dùng. Phòng thủ (ranh giới tin cậy, bộ phân loại, danh sách cho phép công cụ, HITL đối với các hành động do hậu quả) làm tăng chi phí của cuộc tấn công và giảm bán kính vụ nổ của nó. Họ không đóng class.

Đây là mô hình suy luận tương tự như định lý Lob (Bài 8): agent không thể chứng minh token tiếp theo là an toàn; Nó chỉ có thể thiết lập một hệ thống mà tokens không an toàn dễ phát hiện hơn.

### Tư thế phòng thủ thực sự ships

- **Đọc / ghi ranh giới.** Đọc không bao giờ là hệ quả. Viết (gửi biểu mẫu, đăng nội dung, gọi một công cụ có tác dụng phụ) yêu cầu sự chấp thuận mới của con người nếu nội dung bắt đầu đến từ bên ngoài ranh giới tin cậy.
- **Danh sách cho phép công cụ cho mỗi tác vụ.** Người agent có thể duyệt; nó không thể bắt đầu chuyển khoản ngân hàng trừ khi công cụ đó được bật rõ ràng cho tác vụ. Bài học 13 bao gồm ngân sách.
- **Session cách ly.** Trình duyệt chỉ agent sessions chạy với thông tin đăng nhập có giới hạn. Không production xác thực, không có email cá nhân. Nhật ký của mọi yêu cầu HTTP được giữ lại để kiểm tra.
- **Chất khử trùng nội dung.** Các HTML được tìm nạp được loại bỏ các mẫu đã biết-xấu trước khi được nối vào ngữ cảnh model. (Giảm các cuộc tấn công dễ dàng; không ngăn chặn payloads tinh vi.)
- **HITL về các hành động do hậu quả.** Đề xuất sau đó commit mô hình (Bài 15).
- **Canary tokens trên bộ nhớ.** Nếu một mục nhập bộ nhớ kích hoạt, người dùng sẽ nhìn thấy mục đó (Bài 14).

## Ứng dụng

`code/main.py` models một agent trình duyệt nhỏ chạy trên ba trang tổng hợp. Một trang lành tính, một trang có blob chèn prompt trực tiếp trong văn bản hiển thị, một trang có chèn đoạn URL (không hiển thị nhưng nằm trong ngữ cảnh của agent). script cho thấy (a) những gì một agent ngây thơ sẽ làm, (b) những gì ranh giới read/write bắt được, (c) những gì chất khử trùng bắt được, (d) những gì không bắt được.

## Sản phẩm bàn giao

`outputs/skill-browser-agent-trust-boundary.md` phạm vi triển khai agent trình duyệt được đề xuất: vùng tin cậy nào mà nó chạm vào, những gì nó được phép ghi và những biện pháp phòng thủ nào phải được thực hiện trước lần chạy đầu tiên.

## Bài tập

1. Chạy `code/main.py`. Xác định đòn tấn công nào mà chất khử trùng bắt được nhưng ranh giới read/write không bắt được và đòn tấn công nào chỉ tấn công read/write ranh giới bắt được.

2. Mở rộng trình làm sạch để phát hiện một class chèn đoạn URL kiểu HashJack. Đo lường tỷ lệ dương tính giả trên các URL lành tính có các đoạn hợp pháp.

3. Chọn một quy trình làm việc thực sự agent trình duyệt mà bạn biết (ví dụ: "đặt chuyến bay"). Liệt kê mọi lần đọc và mỗi lần ghi. Đánh dấu viết nào cần HITL và tại sao.

4. Đọc bài báo ICLR 2026 được xác minh bởi WebArena. Xác định một loại nhiệm vụ mà điểm của WebArena ban đầu không đáng tin cậy và giải thích cách tập con Đã xác minh giải quyết nó.

5. Thiết kế canary bộ nhớ cho cài đặt agent trình duyệt. Bạn sẽ lưu trữ gì, ở đâu và triggers gì báo động?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|---|---|---|
| Tiêm prompt gián tiếp | "Văn bản trang không hợp lệ" | Nội dung không đáng tin cậy trong trang mà agent đọc chứa các hướng dẫn mà agent thực thi |
| Ký ức bị ô nhiễm | "Tấn công trí nhớ" | Agent viết một lệnh do kẻ tấn công cung cấp cho bộ nhớ bền; Kích hoạt session tiếp theo |
| HashJack | "Tấn công phân đoạn URL" | Payload ẩn trong đoạn URL / chuỗi truy vấn nằm trong ngữ cảnh của agent nhưng không được hiển thị rõ ràng |
| Chiếm quyền điều khiển bằng một cú nhấp chuột | "Nút xấu" | Khả năng chi trả có thể nhìn thấy là một phần tiếp theo payload agent thực hiện |
| Duyệt Comp | "Tìm kiếm web benchmark" | Tìm kiếm các sự kiện cụ thể trên web mở; Chân trời quy mô phút |
| Hệ điều hànhTrang web | "Máy tính để bàn benchmark" | Kiểm soát hệ điều hành đầy đủ; Tác vụ GUI nhiều bước |
| Đã xác minh WebArena | "Đã sửa benchmark tác vụ web" | WebArena được phân loại lại của ServiceNow với tập con Hard |
| Ranh giới Read/write | "Cổng tác dụng phụ" | Đọc không bao giờ là hậu quả; Việc viết yêu cầu phê duyệt mới nếu nội dung không đáng tin cậy |

## Đọc thêm

- [OpenAI — Introducing ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/) - merge của Nhà điều hành và nghiên cứu sâu; Duyệt qua Comp SOTA.
- [OpenAI — Computer-Using Agent](https://openai.com/index/computer-using-agent/) - dòng dõi Operator và kiến trúc đã trở nên ChatGPT agent.
- [Zhou et al. — WebArena](https://webarena.dev/) - benchmark ban đầu.
- [WebArena-Verified (OpenReview)](https://openreview.net/forum?id=94tlGxmqkN) - Bài báo tập con cố định ICLR 2026.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — bao gồm thảo luận về bề mặt tấn công cho agents sử dụng máy tính.
