# Đầu ra có cấu trúc — JSON Schema, Pydantic, Zod, Giải mã hạn chế

> "Yêu cầu model một cách tử tế để trả lại JSON" thất bại từ 5 đến 15 phần trăm thời gian, ngay cả trên models biên giới. Đầu ra có cấu trúc thu hẹp khoảng cách đó bằng cách giải mã hạn chế: model thực sự bị ngăn không phát ra token vi phạm schema. Chế độ nghiêm ngặt của OpenAI, sử dụng công cụ kiểu schema của Anthropic, `responseSchema` của Gemini, `output_type` của Pydantic AI và `.parse` của Zod là năm dạng bề mặt của cùng một ý tưởng. Bài học này xây dựng trình xác thực schema và chế độ hợp đồng nghiêm ngặt mà người học sẽ sử dụng cho mọi pipeline trích xuất production.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib, JSON Schema 2020-12 subset)
**Kiến thức tiên quyết:** Giai đoạn 13 · 02 (function calling lặn sâu)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Viết JSON Schema 2020-12 cho mục tiêu trích xuất bằng cách sử dụng các ràng buộc phù hợp (enum, min/max, bắt buộc, mẫu).
- Giải thích lý do tại sao chế độ nghiêm ngặt và giải mã hạn chế mang lại những đảm bảo khác với "xác thực sau thế hệ".
- Phân biệt ba chế độ lỗi: lỗi phân tích cú pháp, vi phạm schema model từ chối.
- Ship một pipeline trích xuất với sửa chữa đánh máy và xử lý từ chối được đánh máy.

## Vấn đề

Một agent đọc email đơn đặt hàng cần chuyển văn bản miễn phí thành `{customer, line_items, total_usd}`. Ba cách tiếp cận.

**Tiếp cận một: prompt cho JSON.** "Trả lời bằng JSON với các trường khách hàng, line_items, total_usd." Hoạt động 85 đến 95 phần trăm thời gian trên models biên giới. Thất bại theo sáu cách: thiếu dấu ngoặc nhọn, dấu phẩy đuôi, kiểu sai, trường ảo giác, bị cắt bớt ở giới hạn token, văn xuôi bị rò rỉ như "Đây là JSON của bạn:".

**Cách tiếp cận thứ hai: xác thực sau khi tạo.** Tạo tự do, phân tích cú pháp, xác thực dựa trên schema, thử lại khi thất bại. Đáng tin cậy nhưng đắt tiền — bạn phải trả tiền cho mỗi lần thử lại và các lỗi cắt bớt tốn thêm một lượt cho mỗi lần xảy ra.

**Cách tiếp cận thứ ba: giải mã ràng buộc.** Nhà cung cấp thực thi schema tại thời điểm giải mã. Các tokens không hợp lệ sẽ bị che giấu khỏi bản phân phối sampling. Đầu ra được đảm bảo phân tích cú pháp và đảm bảo xác thực. Thất bại sụp đổ xuống một chế độ: từ chối (model quyết định đầu vào không phù hợp với schema).

Mỗi nhà cung cấp biên giới năm 2026 ships một số hình thức tiếp cận ba.

- **OpenAI.** `response_format: {type: "json_schema", strict: true}` cộng `refusal` trong phản hồi nếu model từ chối.
- **Anthropic.** Schema thực thi đầu vào `tool_use`; `stop_reason: "refusal"` không phải là một thứ, nhưng `end_turn` không có cuộc gọi công cụ là tín hiệu.
- **Gemini.** `responseSchema` ở cấp độ yêu cầu; vào năm 2026, các ràng buộc ngữ pháp cấp Gemini ships token đối với các loại được chọn.
- **Pydantic AI.** `output_type=InvoiceModel` phát ra một `RunResult` có cấu trúc được gõ để `InvoiceModel`.
- **Zod (TypeScript).** Runtime trình phân tích cú pháp xác thực đầu ra của nhà cung cấp so với schema Zod; cặp với `beta.chat.completions.parse` của OpenAI.

thread chung: khai báo schema một lần, thực thi từ đầu đến cuối.

## Khái niệm

### JSON Schema 2020-12 — ngôn ngữ chung

Mọi nhà cung cấp đều chấp nhận JSON Schema 2020-12. Các cấu trúc bạn sử dụng nhiều nhất:

- `type`: một trong những `object`, `array`, `string`, `number`, `integer`, `boolean`, `null`.
- `properties`: ánh xạ tên trường với sơ đồ con.
- `required`: danh sách tên trường phải xuất hiện.
- `enum`: tập hợp các giá trị được phép đã đóng.
- `minimum` / `maximum` (số), `minLength` / `maxLength` / `pattern` (chuỗi).
- `items`: lược đồ con được áp dụng cho mọi phần tử mảng.
- `additionalProperties`: `false` cấm các trường bổ sung (mặc định thay đổi tùy theo chế độ).

OpenAI chế độ nghiêm ngặt bổ sung ba yêu cầu: mọi tài sản phải được liệt kê trong `required`, `additionalProperties: false` ở khắp mọi nơi và không có `$ref` chưa được giải quyết. Nếu bạn phá vỡ những thứ này, API trả về 400 tại thời điểm yêu cầu.

### Pydantic, ràng buộc Python

Pydantic v2 tạo JSON Schema từ models hình lớp dữ liệu thông qua `model_json_schema()`. Pydantic AI bao bọc điều này để bạn viết:

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

và agent framework dịch schema thành OpenAI chế độ nghiêm ngặt, Anthropic `input_schema` hoặc Gemini `responseSchema` ở cạnh. Đầu ra của model trở lại dưới dạng phiên bản `Invoice` được nhập. Lỗi xác thực làm tăng `ValidationError` với đường dẫn lỗi được nhập.

### Zod, ràng buộc TypeScript

Zod (`z.object({customer: z.string(), ...})`) tương đương với TS. SDK Node của OpenAI hiển thị `zodResponseFormat(Invoice)` có nghĩa là JSON Schema payload của API.

### Từ chối

Chế độ nghiêm ngặt không thể buộc model phải trả lời. Nếu đầu vào không thể vừa với schema ("email là một bài thơ, không phải hóa đơn"), model sẽ phát ra một trường `refusal` chứa lý do. Mã của bạn phải xử lý điều này như một kết quả class đầu tiên, không phải là một thất bại. Việc từ chối cũng hữu ích như một tín hiệu an toàn: một model được yêu cầu trích xuất số thẻ tín dụng từ một email có nội dung được bảo vệ sẽ trả về một lời từ chối với lý do an toàn đính kèm.

### Giải mã hạn chế trong công khai

Triển khai trọng số mở sử dụng ba kỹ thuật.

1. **Giải mã dựa trên ngữ pháp** (`outlines`, `guidance`, `lm-format-enforcer`): xây dựng một máy tự động hữu hạn xác định từ schema; ở mỗi bước, hãy che giấu logits tokens vi phạm FSM.
2. **Logit tạo mặt nạ bằng trình phân tích cú pháp JSON**: chạy trình phân tích cú pháp streaming JSON cùng với model; Ở mỗi bước, hãy tính tập hợp hợp token tiếp theo.
3. **Giải mã đầu cơ với trình xác minh**: model dự thảo giá rẻ đề xuất tokens, người xác minh thực thi schema.

Các nhà cung cấp thương mại chọn một trong những thứ này đằng sau hậu trường. Hiện đại năm 2026 nhanh hơn so với tạo trơn đối với đầu ra có cấu trúc ngắn và tốc độ gần giống nhau đối với đầu ra dài.

### Ba chế độ thất bại

1. **Lỗi phân tích cú pháp.** Đầu ra không hợp lệ JSON. Không thể xảy ra ở chế độ nghiêm ngặt. Vẫn có thể xảy ra với các nhà cung cấp không nghiêm ngặt.
2. **Schema vi phạm.** Đầu ra phân tích cú pháp nhưng vi phạm schema. Không thể xảy ra ở chế độ nghiêm ngặt. Phổ biến bên ngoài nó.
3. **Từ chối.** model từ chối. Phải được xử lý như một kết quả được nhập.

### Chiến lược thử lại

Khi bạn ở ngoài chế độ nghiêm ngặt (Anthropic sử dụng công cụ, OpenAI không nghiêm ngặt, Gemini cũ hơn), mô hình khôi phục là:

```
generate -> parse -> validate -> if fail, inject error and retry, max 3x
```

Một lần thử lại thường là đủ. Ba lần thử lại bắt các mảnh model yếu. Vượt quá ba là dấu hiệu của một schema xấu: model không thể thỏa mãn nó đối với một số đầu vào, và prompt hoặc schema cần được sửa chữa.

### Hỗ trợ model nhỏ

Giải mã hạn chế hoạt động trên models nhỏ. Một model mở 3B-parameter với thực thi ngữ pháp vượt trội hơn 70B-parameter model với prompting thô về các nhiệm vụ có cấu trúc. Đây là lý do chính khiến đầu ra có cấu trúc quan trọng đối với production: nó tách rời độ tin cậy khỏi kích thước model.

## Ứng dụng

`code/main.py` ships trình xác thực JSON Schema 2020-12 tối thiểu trong stdlib (types, required, enum, min/max, pattern, items, additionalProperties). Nó bao bọc một `Invoice` schema và chạy đầu ra LLM giả mạo thông qua trình xác thực, thể hiện lỗi phân tích cú pháp, vi phạm schema và đường dẫn từ chối. Hoán đổi đầu ra giả mạo cho phản hồi thực của bất kỳ nhà cung cấp nào trong production.

Những gì cần xem:

- Trình xác thực trả về danh sách `[ValidationError]` đã nhập với đường dẫn và thông báo. Đó là hình dạng bạn muốn xuất hiện trong prompt thử lại.
- branch từ chối KHÔNG thử lại. Nó ghi nhật ký và trả về một từ chối đã nhập. Giai đoạn 14 · 09 sử dụng từ chối như một tín hiệu an toàn.
- Kiểm tra `additionalProperties: false` cháy trên đầu vào thử nghiệm đối nghịch, cho thấy lý do tại sao chế độ nghiêm ngặt đóng cửa trên các trường ảo giác.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-structured-output-designer.md`. Với mục tiêu trích xuất văn bản tự do (hóa đơn, vé hỗ trợ, sơ yếu lý lịch, v.v.), skill tạo ra một JSON Schema 2020-12 tương thích với chế độ nghiêm ngặt và một model Pydantic phản chiếu nó, với việc từ chối và xử lý thử lại được nhập vào.

## Bài tập

1. Chạy `code/main.py`. Thêm trường hợp thử nghiệm thứ tư có `total_usd` là số âm. Xác nhận trình xác thực từ chối nó bằng đường dẫn ràng buộc `minimum`.

2. Mở rộng trình xác thực để hỗ trợ `oneOf` với trình phân biệt. Trường hợp phổ biến: `line_item` là một sản phẩm hoặc dịch vụ, được gắn thẻ bởi `kind`. Chế độ nghiêm ngặt có các quy tắc tinh tế ở đây; Kiểm tra hướng dẫn đầu ra có cấu trúc của OpenAI.

3. Viết cùng một schema Hóa đơn như Pydantic BaseModel và so sánh đầu ra `model_json_schema()` với schema cuộn tay của bạn. Xác định một trường mà Pydantic đặt theo mặc định mà phiên bản cuộn tay bỏ qua.

4. Đo lường tỷ lệ từ chối. Xây dựng mười đầu vào không thể trích xuất được (lời bài hát, bằng chứng toán học, email trống) và chạy chúng thông qua một nhà cung cấp thực sự với chế độ nghiêm ngặt. Đếm số lần từ chối so với đầu ra ảo giác. Đây là ground truth của bạn để thử lại nhận biết từ chối.

5. Đọc hướng dẫn đầu ra có cấu trúc của OpenAI từ trên xuống dưới. Xác định một cấu trúc mà nó cấm rõ ràng trong chế độ nghiêm ngặt mà JSON Schema đơn giản cho phép. Sau đó, thiết kế một schema sử dụng cấu trúc bị cấm không cơ bản và tái cấu trúc nó để tương thích nghiêm ngặt.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| JSON Schema 2020-12 | "Thông số kỹ thuật schema" | IETF-draft schema phương ngữ mà mọi nhà cung cấp hiện đại đều nói |
| Chế độ nghiêm ngặt | "Đảm bảo schema" | OpenAI cờ thực thi schema thông qua giải mã ràng buộc |
| Giải mã hạn chế | "Logit mặt nạ" | Thực thi thời gian giải mã che giấu tokens tiếp theo không hợp lệ |
| Từ chối | "Model từ chối" | Kết quả được nhập khi đầu vào không thể vừa với schema |
| Lỗi phân tích cú pháp | "JSON không hợp lệ" | Đầu ra không phân tích cú pháp JSON; không thể dưới sự nghiêm ngặt |
| Schema vi phạm | "Hình dạng sai" | Phân tích cú pháp nhưng vi phạm các loại / bắt buộc / enum / phạm vi |
| `additionalProperties: false` | "Không được phép bổ sung" | Cấm các lĩnh vực không xác định; yêu cầu trong OpenAI nghiêm ngặt |
| Mô hình cơ sở Pydantic | "Đầu ra được nhập" | Python class phát ra và xác nhận JSON Schema |
| Zod schema | "Loại đầu ra TypeScript" | TS runtime schema để xác thực đầu ra của nhà cung cấp |
| Thực thi ngữ pháp | "Giải mã hạn chế trọng số mở" | Mặt nạ logit dựa trên FSM, như trong phác thảo / hướng dẫn |

## Đọc thêm

- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) - chế độ nghiêm ngặt, từ chối và yêu cầu schema
- [OpenAI — Introducing structured outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) - Bài ra mắt tháng 8 năm 2024 giải thích đảm bảo giải mã
- [Pydantic AI — Output](https://ai.pydantic.dev/output/) — các liên kết output_type được nhập tuần tự hóa cho từng nhà cung cấp
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) — thông số kỹ thuật chuẩn
- [Microsoft — Structured outputs in Azure OpenAI](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) — ghi chú triển khai doanh nghiệp và cảnh báo chế độ nghiêm ngặt
