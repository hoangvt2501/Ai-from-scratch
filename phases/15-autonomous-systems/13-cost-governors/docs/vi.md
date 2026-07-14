# Ngân sách hành động, giới hạn lặp lại và bộ điều chỉnh chi phí

> Chi phí LLM hàng tháng của một agent thương mại điện tử cỡ trung đã tăng từ $1,200 to $4.800 sau khi nhóm của họ kích hoạt skill "theo dõi đơn hàng". Đó không phải là một lỗi về giá cả. Đó là một agent đã tìm thấy một vòng lặp mới và tiếp tục chi tiêu bên trong nó. Bộ công cụ quản trị Agent của Microsoft (ngày 2 tháng 4 năm 2026) hệ thống hóa khả năng bảo vệ chống lại class này: `max_tokens` cho mỗi yêu cầu, token cho mỗi tác vụ và ngân sách đô la, giới hạn per-day/month, giới hạn lặp lại, định tuyến model theo tầng, bộ nhớ đệm prompt, cửa sổ ngữ cảnh, checkpoints HITL đối với các hành động tốn kém, ngắt kết nối khi vi phạm ngân sách. Mã Claude của Anthropic Agent SDK ships cùng một primitives dưới các tên khác nhau. Giới hạn tốc độ tài chính - ví dụ: cắt quyền truy cập > 50 đô la trong 10 phút - bắt vòng lặp nhanh hơn giới hạn hàng tháng.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, layered cost-governor simulator)
**Kiến thức tiên quyết:** Giai đoạn 15 · 10 (Chế độ cho phép), Giai đoạn 15 · 12 (Thực thi bền bỉ)
**Thời lượng:** ~60 phút

## Vấn đề

Tự động agents tiêu tiền thật vào mỗi lượt. Đầu ra xấu của chatbot là một câu trả lời tồi; Vòng lặp xấu của một agent là một hóa đơn. Thuật ngữ được ghi nhận trong ngành cho chế độ thất bại là "Từ chối ví" - agent tiếp tục suy luận, tiếp tục gọi công cụ, tiếp tục thanh toán và không có gì ngăn cản nó vì không có gì được thiết kế.

Bản sửa lỗi không phải là một con số. Đây là một stack giới hạn ở các thang thời gian và mức độ chi tiết khác nhau: mỗi yêu cầu, mỗi tác vụ, mỗi giờ, mỗi ngày, mỗi tháng. Một stack được thiết kế tốt sẽ bắt được một vòng lặp chạy trốn trong vòng vài phút, rò rỉ chậm trong vòng vài giờ và phát hành kém trong vòng một ngày. Cùng một stack giữ ngân sách khi agent có chân trời dài và tự chủ.

Đây là một bài học kỹ thuật: toán học là tầm thường, kỷ luật là nơi các đội thất bại. Danh sách các giới hạn dưới đây đều được đặt tên trong Bộ công cụ Quản trị Microsoft Agent hoặc tài liệu Agent SDK Mã Anthropic Claude.

## Khái niệm

### Thống đốc chi phí stack

1. **`max_tokens` cho mỗi yêu cầu.** Đơn giản. Ngăn chặn bất kỳ một cuộc gọi nào phát ra kết thúc không giới hạn.
2. **Mỗi tác vụ token ngân sách.** Trong toàn bộ quá trình chạy, không vượt quá N tokens. Dừng lại ở nắp.
3. **Ngân sách đô la cho mỗi nhiệm vụ.** Tương tự như tokens nhưng bằng tiền tệ. `max_budget_usd` trong Claude Code.
4. **Giới hạn cuộc gọi cho mỗi công cụ.** Không quá N cuộc gọi `WebFetch`, N cuộc gọi `shell_exec`, v.v.
5. **Giới hạn lặp lại (`max_turns`).** Tổng số vòng lặp agent; ngăn chặn các vòng lặp suy luận vô hạn.
6. **Giới hạn mỗi phút / mỗi giờ / mỗi ngày / mỗi tháng.** windows luân phiên. Bắt rò rỉ ở các thang thời gian khác nhau.
7. **Giới hạn tốc độ tài chính.** Ví dụ: "nếu chi tiêu vượt quá $50 trong 10 phút, hãy cắt quyền truy cập." Bắt đốt dựa trên vòng lặp trước khi cháy mũ hàng tháng.
8. **Định tuyến model theo tầng.** Mặc định là một model nhỏ hơn; chỉ leo thang lên một cái lớn hơn khi một bộ phân loại đánh giá nhiệm vụ đảm bảo nó.
9. **Prompt bộ nhớ đệm.** ngữ cảnh System prompt và ổn định được lưu trữ trong bộ nhớ đệm của nhà cung cấp; Chi phí gửi lại token gần bằng không.
10. **Cửa sổ ngữ cảnh.** Nén / tóm tắt để giữ ngữ cảnh đang hoạt động dưới ngưỡng; giảm chi phí token trực tiếp.
11. **HITL checkpoints các hành động tốn kém.** Trước khi một hành động được biết là tốn kém (gọi công cụ dài, tải xuống lớn, nâng cấp model tốn kém), yêu cầu một cú chạm của con người.
12. **Kill switch vi phạm ngân sách.** Session hủy bỏ khi có bất kỳ giới hạn nào kích hoạt. Nắp được ghi lại; yêu cầu một đường dẫn bật lại riêng biệt.

### Tại sao lại stack, không phải một mũ

Một giới hạn hàng tháng duy nhất chỉ bắt được một agent chạy trốn sau khi ví đã hết. Một giới hạn cho mỗi yêu cầu duy nhất không bắt được gì ở cấp độ session. Các chế độ lỗi khác nhau yêu cầu các thang thời gian khác nhau:

- **Vòng lặp chạy trốn** (agent bị kẹt trong lần thử lại 5 giây): bị bắt bởi giới hạn vận tốc.
- **Rò rỉ chậm** (agent thực hiện ~2 lần công việc dự kiến cho mỗi nhiệm vụ): bị bắt bởi giới hạn hàng ngày.
- **Bản phát hành xấu** (phiên bản mới sử dụng 5x tokens): bị bắt bởi giới hạn hàng tuần / hàng tháng.
- **Tăng hợp pháp** (nhu cầu thực, không phải lỗi): bị bắt bởi giới hạn giờ / ngày với nhật ký rõ ràng.

### Bề mặt ngân sách của Claude Code

Bộ luật Claude Agent SDK phơi bày (tài liệu công khai):

- `max_turns` — giới hạn lặp lại.
- `max_budget_usd` — giới hạn đô la; session hủy bỏ khi vi phạm.
- `allowed_tools` / `disallowed_tools` — danh sách cho phép và danh sách từ chối của công cụ.
- Hook điểm trước khi sử dụng công cụ để hạch toán chi phí tùy chỉnh.

Kết hợp với thang chế độ cho phép (Bài 10). Một `autoMode` session không có `max_budget_usd` là quyền tự trị không được quản lý. Anthropic rõ ràng đóng khung Chế độ tự động là yêu cầu kiểm soát ngân sách; Bộ phân loại trực giao với chi phí.

### Đạo luật AI EU, OWASP Agentic Top 10

Bộ công cụ quản trị Agent của Microsoft bao gồm các yêu cầu về OWASP Agentic Top 10 và Điều 14 (giám sát con người) của Đạo luật AI của EU. Đối với production ở EU, việc ghi nhật ký và thực thi giới hạn không phải là tùy chọn.

### $1,200 → $4.800 trường hợp quan sát được

Trường hợp thực sự trong tài liệu của Microsoft: một agent thương mại điện tử có chi phí hàng tháng tăng gấp ba lần sau khi một công cụ mới được thêm vào. Công cụ này cho phép agent thăm dò trạng thái đơn hàng trong mỗi session. Không phát hiện vòng lặp. Không có nắp cho mỗi dụng cụ. Không có cảnh báo về tăng trưởng hàng tuần. Bản sửa lỗi là giới hạn cho mỗi công cụ cộng với cảnh báo tăng trưởng hàng ngày. Đây là một khuôn mẫu: mỗi bề mặt công cụ mới là một vòng lặp tiềm năng mới; Mỗi công cụ mới đều cần nắp riêng và cảnh báo riêng.

## Ứng dụng

`code/main.py` mô phỏng một agent chạy có và không có stack điều chỉnh chi phí nhiều lớp. agent mô phỏng trôi dạt vào vòng lặp bỏ phiếu sau một số lượt; stack phân lớp bắt nó trong cửa sổ vận tốc trong khi một giới hạn hàng tháng duy nhất sẽ không kích hoạt cho đến vài ngày sau đó.

## Sản phẩm bàn giao

`outputs/skill-agent-budget-audit.md` kiểm tra stack điều chỉnh chi phí của triển khai agent được đề xuất và gắn cờ các lớp bị thiếu.

## Bài tập

1. Chạy `code/main.py`. Xác nhận các vụ nổ giới hạn vận tốc trước khi giới hạn lặp lại trên quỹ đạo vòng bỏ phiếu. Bây giờ tắt giới hạn vận tốc và đo số tiền agent "chi tiêu" trước khi giới hạn lặp lại bắt được nó.

2. Thiết kế bộ giới hạn cho mỗi công cụ cho agent trình duyệt (Bài 11). Công cụ nào cần nắp chặt nhất? Công cụ nào có thể chạy không giới hạn mà không gặp rủi ro?

3. Đọc tài liệu Bộ công cụ Quản trị Microsoft Agent. Liệt kê mọi loại mũ tên bộ công cụ. Ánh xạ từng chế độ đến một trong các chế độ lỗi (vòng lặp chạy, rò rỉ chậm, nhả kém, tăng vọt).

4. Định giá một cuộc chạy qua đêm mà không cần giám sát cho một nhiệm vụ thực tế (ví dụ: "phân loại 50 vấn đề trong một repo"). Đặt `max_budget_usd` ở mức gấp 2 lần ước tính điểm của bạn. Biện minh cho 2x.

5. `max_budget_usd` của Claude Code dựa trên session chi phí tổng hợp. Thiết kế một giới hạn vận tốc bổ sung mà bạn sẽ thực thi bên ngoài. Giới hạn triggers gì và kích hoạt lại trông như thế nào?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|---|---|---|
| Từ chối ví | "Hóa đơn bỏ trốn" | Agent vòng lặp tạo ra chi tiêu mà không có giới hạn để dừng nó |
| max_tokens | "Giới hạn cho mỗi yêu cầu" | Trần trên kích thước của một lần hoàn thành |
| max_turns | "Giới hạn lặp lại" | Trần trên agent lặp lại vòng lặp trong một session |
| max_budget_usd | "Đô la kill switch" | Session giới hạn chi phí; hủy bỏ khi vi phạm |
| Giới hạn vận tốc | "Giới hạn giá" | Giới hạn chi tiêu cho mỗi khoảng thời gian ngắn (ví dụ: 50 USD / 10 phút) |
| Định tuyến theo tầng | "Nhỏ model trước" | Mặc định model rẻ; Chỉ leo thang khi bộ phân loại đảm bảo |
| Prompt bộ nhớ đệm | "system prompt được lưu trong bộ nhớ đệm" | Bộ nhớ đệm phía nhà cung cấp giảm chi phí gửi lại token gần bằng không |
| HITL checkpoint | "Cổng phê duyệt của con người" | Cần có vòi của con người trước khi thực hiện hành động tốn kém |

## Đọc thêm

- [Anthropic Claude Code Agent SDK — agent loop and budgets](https://code.claude.com/docs/en/agent-sdk/agent-loop) — danh sách cho phép `max_turns`, `max_budget_usd`, công cụ.
- [Microsoft Agent Framework — human-in-the-loop and governance](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — thống đốc chi phí checkpoints.
- [Anthropic — Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — kiểm soát chi phí từ phía nhà cung cấp.
- [Anthropic — Prompt caching (Claude API docs)](https://platform.claude.com/docs/en/prompt-caching) - cơ chế bộ nhớ đệm.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) - hồ sơ chi phí cho agents dài hạn.
