# Function Calling Deep Dive — OpenAI, Anthropic, Gemini

> Ba nhà cung cấp biên giới hội tụ trên cùng một vòng lặp gọi công cụ vào năm 2024 và sau đó phân kỳ về mọi thứ khác. OpenAI sử dụng `tools` và `tool_calls`. Anthropic sử dụng các khối `tool_use` và `tool_result`. Gemini sử dụng tương quan `functionDeclarations` và id duy nhất. Bài học này phân biệt ba cạnh nhau để mã hóa ships trên một nhà cung cấp không bị hỏng khi bạn chuyển nó.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib, schema translators)
**Kiến thức tiên quyết:** Giai đoạn 13 · 01 (giao diện công cụ)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Nêu ba sự khác biệt về hình dạng giữa payloads gọi hàm OpenAI, Anthropic và Gemini (khai báo, gọi, kết quả).
- Dịch một khai báo công cụ trên cả ba định dạng nhà cung cấp và dự đoán các ràng buộc về chế độ nghiêm ngặt sẽ khác nhau ở đâu.
- Sử dụng `tool_choice` trong mỗi nhà cung cấp để bắt buộc, cấm hoặc tự động chọn lệnh gọi công cụ.
- Biết các giới hạn cứng của mỗi nhà cung cấp (số lượng công cụ, độ sâu schema, độ dài đối số) và các chữ ký lỗi mà mỗi nhà cung cấp phát ra khi các giới hạn bị vi phạm.

## Vấn đề

Hình dạng của yêu cầu gọi hàm khác nhau tùy theo nhà cung cấp. Ba ví dụ cụ thể từ năm 2026 production stacks:

**OpenAI Hoàn thành / Phản hồi trò chuyện API.** Bạn vượt qua `tools: [{type: "function", function: {name, description, parameters, strict}}]`. Phản hồi của model chứa `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]` trong đó `arguments` là chuỗi JSON mà bạn phải phân tích cú pháp. Chế độ nghiêm ngặt (`strict: true`) thực thi tuân thủ schema thông qua giải mã ràng buộc.

**Anthropic Tin nhắn API.** Bạn vượt qua `tools: [{name, description, input_schema}]`. Câu trả lời trở lại là `content: [{type: "text"}, {type: "tool_use", id, name, input}]`. `input` đã được phân tích cú pháp (một đối tượng, không phải một chuỗi). Bạn trả lời bằng một tin nhắn `user` mới có chứa khối `{type: "tool_result", tool_use_id, content}`.

**Google Gemini API.** Bạn chuyển `tools: [{functionDeclarations: [{name, description, parameters}]}]` (lồng dưới `functionDeclarations`). Phản hồi đến dưới dạng `candidates[0].content.parts: [{functionCall: {name, args, id}}]` trong đó `id` là duy nhất trong Gemini 3 trở lên cho tương quan cuộc gọi song song. Bạn trả lời bằng `{functionResponse: {name, id, response}}`.

Cùng một vòng lặp. Tên trường khác nhau, lồng nhau khác nhau, quy ước chuỗi và đối tượng khác nhau, cơ chế tương quan khác nhau. Một nhóm viết agent thời tiết trên OpenAI trả một cổng hai ngày cho Anthropic và một ngày khác để Gemini chỉ để làm hệ thống ống nước.

Bài học này xây dựng một trình dịch hợp nhất ba định dạng thành một khai báo công cụ chuẩn và định tuyến ở biên. Giai đoạn 13 · 17 khái quát hóa cùng một mẫu thành một LLM gateway.

## Khái niệm

### Cấu trúc chung

Mỗi nhà cung cấp cần năm điều:

1. **Danh sách công cụ.** Tên công cụ, mô tả và schema đầu vào.
2. **Lựa chọn công cụ.** Buộc một công cụ cụ thể, cấm các công cụ hoặc để model quyết định.
3. **Phát hành cuộc gọi.** Đầu ra có cấu trúc đặt tên cho công cụ và đối số.
4. **ID cuộc gọi.** Tương quan phản hồi với cuộc gọi phù hợp (quan trọng đối với song song).
5. **Chèn kết quả.** Một thông báo hoặc khối liên kết kết quả trở lại cuộc gọi.

### Hình dạng khác biệt, từng trường

| Khía cạnh | OpenAI | Anthropic | Gemini |
|--------|--------|-----------|--------|
| Phong bì khai báo | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| Schema trường | `parameters` | `input_schema` | `parameters` |
| Phản hồi container | `tool_calls[]` trên tin nhắn trợ lý | `content[]` loại `tool_use` | `parts[]` loại `functionCall` |
| Loại đối số | JSON chuỗi | đối tượng phân tích cú pháp | đối tượng phân tích cú pháp |
| Định dạng id | `call_...` (OpenAI tạo) | `toolu_...` (Anthropic) | UUID (Gemini 3+) |
| Khối kết quả | `tool` vai trò, `tool_call_id` | `user` với `tool_result`, `tool_use_id` | `functionResponse` với `id` phù hợp |
| Buộc một công cụ | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| Công cụ cấm | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| schema nghiêm ngặt | `strict: true` | schema-is-schema (luôn được thực thi) | `responseSchema` ở cấp độ yêu cầu |

### Giới hạn bạn sẽ thực sự đạt được

- **OpenAI.** 128 công cụ cho mỗi yêu cầu. Schema độ sâu 5. Chuỗi đối số <= 8192 byte. Chế độ nghiêm ngặt không yêu cầu `$ref`, không có `oneOf`/`anyOf`/`allOf` trùng lặp, mọi thuộc tính được liệt kê trong `required`.
- **Anthropic.** 64 công cụ cho mỗi yêu cầu. Schema độ sâu không giới hạn nhưng thực tế giới hạn 10. Không có cờ chế độ nghiêm ngặt; schema là một hợp đồng và model có xu hướng tuân thủ.
- **Gemini.** 64 chức năng cho mỗi yêu cầu. Schema loại là tập hợp con OpenAPI 3.0 (phân kỳ nhẹ so với JSON Schema 2020-12). Parallel gọi unique-id kể từ Gemini 3.

### `tool_choice` hành vi

Ba chế độ mà mọi người ủng hộ, được đặt tên khác nhau.

- **Tự động.** Model chọn công cụ hoặc văn bản. Mặc định.
- **Bắt buộc / Bất kỳ.** Model phải gọi ít nhất một công cụ.
- **Không có.** Model không được gọi các công cụ.

Cộng với một chế độ duy nhất cho mỗi nhà cung cấp:

- **OpenAI.** Buộc một công cụ cụ thể theo tên.
- **Anthropic.** Buộc một công cụ cụ thể theo tên; Cờ `disable_parallel_tool_use` tách đơn và đa.
- **Gemini.** `mode: "VALIDATED"` định tuyến mọi phản hồi thông qua trình xác thực schema bất kể ý định model.

### Cuộc gọi song song

`parallel_tool_calls: true` của OpenAI (mặc định) phát ra nhiều cuộc gọi trong một tin nhắn trợ lý. Bạn chạy tất cả chúng và trả lời bằng một thông báo vai trò công cụ hàng loạt chứa một mục nhập mỗi `tool_call_id`. Anthropic trong lịch sử đã thực hiện một cuộc gọi; `disable_parallel_tool_use: false` (mặc định kể từ Claude 3.5) cho phép đa. Gemini 2 cho phép gọi song song nhưng không cung cấp id ổn định; Gemini 3 thêm UUID để các phản hồi không theo thứ tự tương quan rõ ràng.

### Streaming

Cả ba đều hỗ trợ các cuộc gọi công cụ được phát trực tuyến. Định dạng dây khác nhau:

- **OpenAI.** Các khối `tool_calls[i].function.arguments` Delta đến dần dần. Bạn tích lũy cho đến `finish_reason: "tool_calls"`.
- **Anthropic.** Các sự kiện bắt đầu chặn / chặn delta / chặn-dừng. `input_json_delta` đoạn mang các đối số từng phần.
- **Gemini.** `streamFunctionCallArguments` (mới trong Gemini 3) phát ra các đoạn có `functionCallId` để nhiều cuộc gọi song song có thể xen kẽ.

Giai đoạn 13 · 03 đi sâu khi lắp ráp lại song song + streaming. Bài học này tập trung vào các hình dạng khai báo và một cuộc gọi.

### Lỗi và sửa chữa

Lỗi đối số không hợp lệ cũng trông khác.

- **OpenAI (không nghiêm ngặt).** Model trả về `arguments: "{bad json}"`, phân tích cú pháp JSON của bạn không thành công, bạn chèn thông báo lỗi và gọi lại.
- **OpenAI (nghiêm ngặt).** Xác thực xảy ra trong quá trình giải mã; JSON không hợp lệ là không thể nhưng `refusal` có thể xuất hiện.
- **Anthropic.** `input` có thể chứa các trường không mong muốn; schema là tư vấn. Xác thực phía server.
- **Gemini.** Quirk OpenAPI 3.0: `enum` trên các trường đối tượng bị bỏ qua một cách âm thầm; xác nhận bản thân.

### Mẫu dịch thuật

Khai báo công cụ chuẩn trong mã của bạn trông như thế này (bạn chọn hình dạng):

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

Ba hàm nhỏ dịch nó thành ba hình dạng nhà cung cấp. Các harness trong `code/main.py` thực hiện chính xác điều này, sau đó thực hiện một cuộc gọi công cụ giả mạo thông qua hình dạng phản hồi của từng nhà cung cấp. Không cần mạng - bài học này dạy các hình dạng, không phải HTTP.

Production nhóm bao bọc trình dịch này bằng `AbstractToolset` (Pydantic AI), `UniversalToolNode` (LangGraph) hoặc `BaseTool` (LlamaIndex). Giai đoạn 13 · 17 ships một gateway để lộ một API hình OpenAI trước mặt bất kỳ ai trong ba người.

## Ứng dụng

`code/main.py` định nghĩa một lớp dữ liệu `Tool` chuẩn và ba trình dịch phát ra JSON khai báo OpenAI, Anthropic và Gemini. Sau đó, nó phân tích phản hồi của nhà cung cấp thủ công của mỗi hình dạng thành cùng một đối tượng gọi chuẩn hóa, chứng minh rằng ngữ nghĩa giống hệt nhau dưới da. Chạy nó và phân biệt ba khai báo cạnh nhau.

Những gì cần xem:

- Ba khối khai báo chỉ khác nhau về tên phong bì và trường.
- Ba khối phản hồi khác nhau về vị trí của cuộc gọi (`tool_calls` cấp cao nhất, khối `content[]` `parts[]` mục nhập).
- Một hàm `canonical_call()` trích xuất `{id, name, args}` từ cả ba hình dạng phản hồi.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-provider-portability-audit.md`. Với tích hợp gọi hàm chống lại một nhà cung cấp, skill tạo ra một kiểm tra tính di động: giới hạn nhà cung cấp nào mà nó dựa vào, trường nào cần đổi tên và trường nào bị hỏng khi chuyển sang nhà cung cấp khác.

## Bài tập

1. Chạy `code/main.py` và xác minh rằng ba JSON khai báo nhà cung cấp đều tuần tự hóa cùng một đối tượng `Tool` cơ bản. Sửa đổi công cụ chuẩn để thêm enum parameter và xác nhận chỉ có Gemini trình dịch cần để xử lý OpenAPI quirk.

2. Thêm trình phân tích cú pháp `ListToolsResponse` cho mỗi nhà cung cấp trích xuất danh sách công cụ mà model trả về sau lệnh gọi `list_tools` hoặc khám phá. OpenAI không có một bản địa; Lưu ý sự bất đối xứng này.

3. Triển khai chuyển đổi `tool_choice`: ánh xạ `ToolChoice(mode="force", tool_name="x")` chuẩn thành cả ba hình dạng nhà cung cấp. Sau đó lập bản đồ `mode="any"` và `mode="none"`. Kiểm tra bảng khác biệt của bài học.

4. Chọn một trong ba nhà cung cấp và đọc hướng dẫn gọi hàm từ đầu đến cuối. Tìm một trường trong thông số kỹ thuật schema của nó mà hai trường còn lại không hỗ trợ. Ứng viên: OpenAI `strict`, Anthropic `disable_parallel_tool_use`, Gemini `function_calling_config.allowed_function_names`.

5. Write a test vector: một lệnh gọi công cụ có đối số vi phạm schema đã khai báo. Chạy nó thông qua trình xác thực của từng nhà cung cấp (trình xác thực stdlib trong Bài 01 sẽ thực hiện như một proxy) và ghi lại lỗi nào kích hoạt. Ghi lại nhà cung cấp nào bạn sẽ sử dụng trong production để đảm bảo tính nghiêm ngặt.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Function calling | "Sử dụng công cụ" | API cấp nhà cung cấp để phát hành lệnh gọi công cụ có cấu trúc |
| Khai báo công cụ | "Thông số kỹ thuật công cụ" | Tên + mô tả + payload đầu vào JSON Schema |
| `tool_choice` | "Buộc / cấm" | Chế độ tự động / bắt buộc / không có / tên cụ thể |
| Chế độ nghiêm ngặt | "Thực thi Schema" | OpenAI cờ hạn chế giải mã để khớp với schema |
| `tool_use` khối | "Hình dạng cuộc gọi của Anthropic" | Khối nội dung nội tuyến với id, tên, đầu vào |
| Phần `functionCall` | "Hình dạng cuộc gọi của Gemini" | Mục `parts[]` chứa tên, đối số và id |
| Đối số-dưới dạng chuỗi | "Chuỗi JSON" | OpenAI trả về args dưới dạng chuỗi JSON, không phải đối tượng |
| Lệnh gọi công cụ song song | "Quạt ra trong một lượt" | Nhiều lệnh gọi công cụ trong một tin nhắn trợ lý |
| Từ chối | "Model từ chối" | Chặn từ chối chỉ ở chế độ nghiêm ngặt thay vì cuộc gọi |
| Tập hợp con OpenAPI 3.0 | "Gemini schema kỳ quặc" | Gemini sử dụng phương ngữ giống JSON Schema với những khác biệt nhỏ |

## Đọc thêm

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — tham chiếu chuẩn bao gồm chế độ nghiêm ngặt và cuộc gọi song song
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — ngữ nghĩa khối `tool_use` và `tool_result`
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) — lệnh gọi song song, id duy nhất và tập hợp con OpenAPI
- [Vertex AI — Function calling reference](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) — Bề mặt doanh nghiệp của Gemini
- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — chi tiết thực thi schema chế độ nghiêm ngặt
