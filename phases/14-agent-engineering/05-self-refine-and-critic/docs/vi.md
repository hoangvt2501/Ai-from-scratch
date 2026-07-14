# Tự tinh chỉnh và PHÊ BÌNH: Cải thiện đầu ra lặp đi lặp lại

> Self-Refine (Madaan et al., 2023) sử dụng một LLM trong ba vai trò - tạo, phản hồi, tinh chỉnh - trong một vòng lặp. Mức tăng trung bình: +20 tuyệt đối trên 7 nhiệm vụ. CRITIC (Gou và cộng sự, 2023) củng cố bước phản hồi bằng cách định tuyến xác minh thông qua các công cụ bên ngoài. Vào năm 2026, mô hình này ships ở mọi framework là "người đánh giá-optimizer" (Anthropic) hoặc vòng lặp guardrail (OpenAI Agents SDK).

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 14 · 01 (Vòng lặp Agent), Giai đoạn 14 · 03 (Phản xạ)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Nêu ba prompts của Tự tinh chỉnh (tạo, phản hồi, tinh chỉnh) và giải thích lý do tại sao lịch sử lại quan trọng đối với các prompt tinh chỉnh.
- Giải thích cái nhìn sâu sắc quan trọng của CRITIC: LLMs không đáng tin cậy trong việc tự xác minh nếu không có grounding bên ngoài.
- Triển khai vòng lặp Tự tinh chỉnh stdlib với lịch sử và trình xác minh bên ngoài tùy chọn.
- Ánh xạ mẫu này với quy trình làm việc "đánh giá optimizer" của Anthropic và guardrails đầu ra của OpenAI Agents SDK.

## Vấn đề

Một agent tạo ra một câu trả lời gần như đúng. Có thể một dòng mã có lỗi cú pháp. Có lẽ một bản tóm tắt quá dài. Có thể một kế hoạch bỏ lỡ một trường hợp cạnh tranh. Điều bạn muốn là: agent phê bình đầu ra của chính nó, sau đó sửa chữa nó.

Tự tinh chỉnh cho thấy điều này hoạt động với một model duy nhất, không có dữ liệu training, không có RL. Nhưng có một điểm mấu chốt: LLMs kém trong việc tự xác minh các sự kiện khó khăn. CRITIC đặt tên cho bản sửa lỗi - định tuyến bước xác minh thông qua các công cụ bên ngoài (tìm kiếm, trình thông dịch mã, máy tính, người chạy thử nghiệm).

Hai bài báo này cùng nhau xác định mặc định năm 2026 cho cải tiến lặp đi lặp lại: tạo, xác minh (bên ngoài khi có thể), tinh chỉnh, dừng khi trình xác minh vượt qua.

## Khái niệm

### Tự tinh chỉnh (Madaan và cộng sự, NeurIPS 2023)

Một LLM, ba vai trò:

```
generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
stop when feedback says "no issues" or budget exhausted.
```

Chi tiết chính: `refine` nhìn thấy toàn bộ lịch sử - tất cả các kết quả và phê bình prior - vì vậy nó không lặp lại sai lầm. Bài báo loại bỏ điều này: lịch sử giảm và chất lượng giảm mạnh.

Tiêu đề: +20 cải thiện tuyệt đối trung bình trên 7 nhiệm vụ (toán học, mã, từ viết tắt, hộp thoại) bao gồm GPT-4. Không training, không có công cụ bên ngoài, model đơn.

### CRITIC (Gou và cộng sự, arXiv: 2305.11738, v4 tháng 2 năm 2024)

Điểm yếu của Self-Refine: bước phản hồi là một LLM tính điểm. Đối với những tuyên bố thực tế, điều này là không đáng tin cậy (ảo giác thường có vẻ thuyết phục đối với model tạo ra nó). CRITIC thay thế `feedback(task, output)` bằng `verify(task, output, tools)` trong đó `tools` bao gồm:

- Một công cụ tìm kiếm cho các tuyên bố thực tế.
- Trình thông dịch mã cho tính chính xác của mã.
- Một máy tính cho số học.
- Trình xác minh miền cụ thể (kiểm tra đơn vị, trình kiểm tra kiểu, trình kiểm tra).

Người xác minh tạo ra một bài phê bình có cấu trúc dựa trên kết quả công cụ. Sau đó, người tinh chỉnh điều kiện về phê bình này.

Tiêu đề: CRITIC vượt trội hơn Self-Refine trong các nhiệm vụ thực tế vì phê bình có cơ sở. Đối với các tác vụ không có trình xác minh bên ngoài (viết sáng tạo, định dạng), CRITIC giảm xuống thành Tự tinh chỉnh.

### Điều kiện dừng

Hai hình dạng phổ biến:

1. **Trình xác minh đạt.** Kiểm tra bên ngoài trả về thành công. Ưu tiên khi có sẵn (kiểm tra đơn vị, kiểm tra loại guardrail xác nhận).
2. **Không có phản hồi nào được đưa ra.** Model nói rằng "đầu ra vẫn ổn." Rẻ hơn nhưng không đáng tin cậy; Ghép nối với giới hạn lặp lại tối đa.

Mặc định năm 2026: Kết hợp chúng. "Dừng lại nếu trình xác minh vượt qua HOẶC model nói tốt VÀ lặp lại >= 2 HOẶC lặp lại >= max_iterations."

### Người đánh giá-Optimizer (Anthropic, 2024)

Bài đăng tháng 12 năm 2024 của Anthropic nêu tên đây là một trong năm mẫu quy trình làm việc. Hai vai trò:

- Người đánh giá: chấm điểm đầu ra và đưa ra một lời phê bình.
- Optimizer: sửa đổi kết quả đưa ra phê bình.

Lặp lại cho đến khi người đánh giá vượt qua. Đây là Self-Refine/CRITIC trong khuôn khổ của Anthropic. Chi tiết kỹ thuật quan trọng Anthropic bổ sung: người đánh giá và optimizer prompts phải khác nhau về cơ bản để model không chỉ đóng dấu cao su.

### OpenAI Agents SDK guardrails đầu ra

OpenAI Agents SDK ships mẫu này là "guardrails đầu ra". guardrail là trình xác thực chạy trên đầu ra cuối cùng của agent. Nếu guardrail ngắt (tăng `OutputGuardrailTripwireTriggered`), đầu ra sẽ bị từ chối và agent có thể thử lại. Guardrails có thể gọi các công cụ (kiểu CRITIC) hoặc là các hàm thuần túy (kiểu Tự tinh chỉnh).

### Cạm bẫy năm 2026

- **Vòng lặp đóng dấu cao su.** Tương tự model việc tạo ra và phê bình với cùng một phong cách prompt hội tụ vào "trông đẹp đối với tôi". Sử dụng các prompts khác về cấu trúc hoặc một model rẻ tiền nhỏ hơn để phê bình.
- **Tinh chỉnh quá mức.** Mỗi lần tinh chỉnh sẽ tăng thêm độ trễ và tokens. Ngân sách 1-3 lượt; Sau đó, chuyển sang xem xét của con người.
- **CRITIC về các nhiệm vụ tầm thường.** Nếu không có trình xác minh bên ngoài, CRITIC sẽ thoái hóa thành Tự tinh chỉnh; Không phải trả độ trễ cho trình xác minh sơ khai.

## Tự xây dựng

`code/main.py` triển khai Self-Refine và CRITIC trên một nhiệm vụ đồ chơi: tạo ra một danh sách gạch đầu dòng ngắn cho một chủ đề. Trình xác minh kiểm tra định dạng (3 dấu đầu dòng, mỗi dấu đầu dòng dưới 60 ký tự). CRITIC thêm một "công cụ xác minh sự thật" bên ngoài để trừng phạt ảo giác đã biết.

Các thành phần:

- `generate` — nhà sản xuất theo kịch bản.
- `feedback` - tự phê bình theo phong cách LLM.
- `verify_external` - Trình xác minh có cơ sở theo phong cách CRITIC.
- `refine` - viết lại đầu ra cho lịch sử.
- Điều kiện dừng - trình xác minh vượt qua hoặc tối đa 4 lần lặp.

Chạy nó:

```
python3 code/main.py
```

So sánh các lần chạy Self-Refine và CRITIC. CRITIC bắt được lỗi thực tế mà Self-Refine đã bỏ lỡ vì trình xác minh bên ngoài đã grounding người tự phê bình thì không.

## Ứng dụng

optimizer đánh giá của Anthropic là mô hình này trong ngôn ngữ thân thiện với Claude. guardrails đầu ra của OpenAI Agents SDK có hình CRITIC (guardrails có thể gọi các công cụ). LangGraph ships một nút phản xạ có nội dung giống như Tự tinh chỉnh. Sử dụng máy tính Gemini 2.5 của Google bổ sung một công cụ đánh giá an toàn cho mỗi bước là một biến thể CRITIC: mọi hành động đều được xác minh trước khi commit.

## Sản phẩm bàn giao

`outputs/skill-refine-loop.md` định cấu hình vòng lặp optimizer đánh giá theo hình dạng tác vụ, tính khả dụng của trình xác minh và ngân sách lặp. Phát ra prompts cho máy phát điện, evaluator/verifier và optimizer, cộng với một policy dừng.

## Bài tập

1. Chạy đồ chơi với max_iterations = 1. CRITIC có còn giúp ích không?
2. Thay thế trình xác minh bên ngoài bằng một công cụ ồn ào (ngẫu nhiên 30% dương tính giả). Vòng lặp làm gì? Đây là thực tế năm 2026 của hầu hết các guardrail stacks.
3. Thực hiện một biến thể "generator-critic trên các models khác nhau": model lớn tạo ra, phê bình model nhỏ. Nó có đánh bại cùng một model không?
4. Đọc PHÊ BÌNH Phần 3 (arXiv: 2305.11738 v4). Đặt tên cho ba danh mục công cụ xác minh và đưa ra một ví dụ cho mỗi loại.
5. Ánh xạ `output_guardrails` của OpenAI Agents SDK với vai trò người xác minh của CRITIC. Điều gì sai SDK và điều gì đúng?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Tự tinh chỉnh | "LLM tự sửa chữa" | Tạo phản hồi -> -> tinh chỉnh vòng lặp trong một model, với lịch sử |
| PHÊ BÌNH | "Xác minh dựa trên công cụ" | Thay thế phản hồi bằng trình xác minh bên ngoài (tìm kiếm, mã, tính toán, kiểm tra) |
| Người đánh giá-Optimizer | "Anthropic mẫu quy trình làm việc" | Hai vai trò - điểm đánh giá optimizer sửa đổi - được lặp lại để hội tụ |
| Đầu ra guardrail | "Kiểm tra sau hoc" | OpenAI Agents SDK xác thực chạy sau khi agent tạo ra đầu ra |
| Bước xác minh | "Giai đoạn phê bình" | Quyết định chịu tải: nối đất hoặc tự đánh giá |
| Tinh chỉnh lịch sử | "Những gì model đã thử" | Prior đầu ra + phê bình được thêm vào trước để tinh chỉnh prompt; giảm và chất lượng sụp đổ |
| Vòng lặp tem cao su | "Tự thỏa thuận thất bại" | Phê bình cùng prompt trả lại "trông tốt"; sửa chữa với các prompts cấu trúc khác nhau |
| Điều kiện dừng | "Kiểm tra hội tụ" | Trình xác minh vượt qua HOẶC không có phản hồi VÀ giới hạn lặp lại; Không bao giờ là một điều kiện |

## Đọc thêm

- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — bài báo kinh điển
- [Gou et al., CRITIC (arXiv:2305.11738)](https://arxiv.org/abs/2305.11738) — xác minh dựa trên công cụ
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — mẫu quy trình làm việc optimizer người đánh giá
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — guardrails đầu ra dưới dạng trình xác minh hình CRITIC
