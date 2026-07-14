# Gọi và Streaming công cụ song song với các công cụ

> Ba tra cứu thời tiết độc lập được nối tiếp là ba chuyến khứ hồi. Chạy chúng song song và tổng thời gian thu gọn xuống cuộc gọi đơn lẻ chậm nhất. Mỗi nhà cung cấp biên giới hiện phát ra nhiều lệnh gọi công cụ trong một lượt. Phần thưởng là có thật; hệ thống ống nước rất tinh tế. Bài học này đi theo cả hai nửa: fan-out song song và lắp ráp lại đối số streamed, với trọng tâm là bẫy tương quan id.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib, thread pool + streaming harness)
**Kiến thức tiên quyết:** Giai đoạn 13 · 02 (function calling lặn sâu)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Giải thích lý do tại sao `parallel_tool_calls: true` tồn tại và khi nào nên tắt tính năng này.
- Tương quan các đoạn đối số được phát trực tuyến với id lệnh gọi công cụ phù hợp trong quá trình phân xuất song song.
- Lắp ráp lại các dây `arguments` một phần thành JSON hoàn chỉnh mà không cần phân tích cú pháp sớm.
- Chạy benchmark thời tiết ba thành phố thể hiện độ trễ tuần tự so với song song.

## Vấn đề

Không có các cuộc gọi song song, một agent trả lời "thời tiết ở Bengaluru, Tokyo và Zurich như thế nào" sẽ làm điều này:

```
user -> LLM
LLM -> call get_weather(Bengaluru)
host -> run executor, reply with result
LLM -> call get_weather(Tokyo)
host -> run executor, reply with result
LLM -> call get_weather(Zurich)
host -> run executor, reply with result
LLM -> final text answer
```

Ba chuyến đi khứ hồi LLM, mỗi chuyến cũng trả cho người thực hiện thời gian. Khoảng 4 lần thời gian đồng hồ treo tường lý tưởng.

Với các cuộc gọi song song:

```
user -> LLM
LLM -> call get_weather(Bengaluru); call get_weather(Tokyo); call get_weather(Zurich)
host -> run all three executors concurrently, reply with three results
LLM -> final text answer
```

Một chuyến LLM khứ hồi. Thời gian của người thực thi là tối đa của ba, không phải tổng. Production benchmarks trên OpenAI, Anthropic và Gemini cho thấy đồng hồ treo tường giảm 60 đến 70% khối lượng công việc quạt.

Giá cả là sự phức tạp tương quan. Khi ba cuộc gọi hoàn thành không theo thứ tự, kết quả của bạn phải mang `tool_call_id` phù hợp để model có thể sắp xếp chúng. Khi luồng kết quả, bạn phải tập hợp các đoạn đối số một phần thành JSON hoàn chỉnh trước khi thực thi. Gemini 3 đã thêm các ID duy nhất một phần để giải quyết một vấn đề trong thế giới thực trong đó hai lệnh gọi song song đến cùng một công cụ không thể phân biệt được.

## Khái niệm

### Bật song song

- **OpenAI.** `parallel_tool_calls: true` bật theo mặc định. Đặt `false` để buộc nối tiếp.
- **Anthropic.** Song song qua `disable_parallel_tool_use: false` (mặc định trên Claude 3.5 trở lên). Đặt `true` cho sê-ri.
- **Gemini.** Luôn có khả năng song song; `tool_config.function_calling_config.mode = "AUTO"` để model quyết định.

Tắt song song khi các công cụ có phụ thuộc thứ tự (`create_file` sau đó `write_file`), khi đầu ra của một cuộc gọi thông báo đầu vào của lệnh gọi khác hoặc khi bộ giới hạn tốc độ không thể xử lý phân xuất.

### Tương quan id

Mỗi cuộc gọi mà model phát ra đều có một `id`. Mọi kết quả mà máy chủ trả về phải bao gồm cùng một id. Nếu không có điều này, kết quả sẽ không rõ ràng.

- **OpenAI.** `tool_call_id` trên mỗi thông báo vai trò công cụ.
- **Anthropic.** `tool_use_id` trên mỗi khối `tool_result`.
- **Gemini.** `id` trên mỗi `functionResponse` (Gemini 3 trở lên; Gemini 2 khớp theo tên bị hỏng đối với các cuộc gọi song song cùng tên).

### Chạy cuộc gọi đồng thời

Máy chủ chạy trình thực thi của mỗi lệnh gọi trên thread, coroutine hoặc worker từ xa của riêng nó. harness đơn giản nhất sử dụng hồ bơi thread; production sử dụng asyncio với `asyncio.gather` hoặc đồng thời có cấu trúc. Thứ tự hoàn thành là không thể đoán trước - id là mã định danh.

Một lỗi phổ biến: trả lời với kết quả theo thứ tự danh sách cuộc gọi thay vì thứ tự hoàn thành. Điều này thường hiệu quả vì model chỉ quan tâm đến `tool_call_id`, nhưng nếu kết quả bị loại bỏ hoặc trùng lặp, việc gửi không theo thứ tự sẽ khiến việc gỡ lỗi trở nên khó khăn hơn. Thích trả lời theo thứ tự hoàn thành với id rõ ràng.

### Streaming gọi công cụ

Khi model suối, `arguments` đến từng mảnh. Ba luồng khối riêng biệt cho ba cuộc gọi song song xen kẽ trên dây. Bạn cần một bộ tích lũy cho mỗi id.

Hình dạng theo nhà cung cấp:

- **OpenAI.** Mỗi đoạn là `choices[0].delta.tool_calls[i].function.arguments` (một phần chuỗi). Chunk mang `index` (vị trí trong danh sách cuộc gọi). Bạn tích lũy mỗi chỉ mục, đọc `id` khi nó xuất hiện lần đầu tiên và phân tích cú pháp JSON khi `finish_reason = "tool_calls"`.
- **Anthropic.** Sự kiện luồng được `message_start`, sau đó một `content_block_start` cho mỗi khối với loại `tool_use` (chứa id, tên, đầu vào trống). `content_block_delta` sự kiện mang `input_json_delta` phần lớn. `content_block_stop` đóng từng khối.
- **Gemini.** `streamFunctionCallArguments` (Gemini 3 trở lên) phát ra các khối với một `functionCallId` để các cuộc gọi xen kẽ một cách sạch sẽ. Trước ngày 3 Gemini, streaming đã trả lời từng cuộc gọi hoàn chỉnh một.

### JSON một phần và bẫy phân tích cú pháp sớm

Bạn không thể phân tích cú pháp `arguments` cho đến khi nó hoàn tất. Các JSON một phần như `{"city": "Beng` không hợp lệ và sẽ tăng lên. Cổng chính xác là tín hiệu kết thúc cuộc gọi của nhà cung cấp: `finish_reason = "tool_calls"` của OpenAI, `content_block_stop` của Anthropic hoặc sự kiện kết thúc luồng của Gemini. Chỉ sau đó mới cố gắng `json.loads`. Một cách tiếp cận mạnh mẽ hơn sử dụng trình phân tích cú pháp JSON gia tăng mang lại các sự kiện khi cấu trúc hoàn thành; Hướng dẫn streaming của OpenAI đề xuất điều này cho UX hiển thị chỉ báo "tư duy" trực tiếp. Đếm dấu ngoặc nhọn không đáng tin cậy như một bài kiểm tra tính đầy đủ (dấu ngoặc nhọn bên trong chuỗi được trích dẫn hoặc nội dung thoát gây ra kết quả dương tính giả) và chỉ nên được sử dụng như một phương pháp phỏng đoán gỡ lỗi không chính thức.

### Hoàn thành không theo thứ tự

```
call_A: fast API, returns first
call_B: slow API, returns second
call_C: median API, returns third
```

Câu trả lời của người dẫn chương trình vẫn phải trích dẫn các id:

```
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

Thứ tự trong câu trả lời không quan trọng về tính đúng đắn của OpenAI hoặc Anthropic. Gemini chấp nhận bất kỳ đơn đặt hàng nào miễn là id khớp với nhau.

### Benchmark: tuần tự so với song song

harness trong `code/main.py` mô phỏng ba trình thực thi với độ trễ 400, 600 và 800 ms. Tuần tự chạy nó trong tổng cộng 1800 mili giây. Parallel chạy nó trong tối đa (400, 600, 800) = 800 ms. Sự khác biệt là không đổi, không tỷ lệ, vì vậy khoản tiết kiệm tăng lên theo số lượng công cụ.

Cảnh báo trong thế giới thực: các cuộc gọi song song gây căng thẳng cho APIs xuôi dòng. Quạt ra 10 chiều sang dịch vụ giới hạn tốc độ sẽ không thành công. Giai đoạn 13 · 17 bao gồm áp suất ngược mức gateway; Ngữ nghĩa thử lại được lên kế hoạch cho giai đoạn trong tương lai.

### Đồng hồ treo tường quạt ra Streaming

Nếu bản thân model phát trực tuyến, bạn có thể bắt đầu thực thi ngay sau khi các đối số của một lệnh gọi hoàn tất, thay vì đợi tất cả các lệnh gọi hoàn tất. Đây là một tối ưu hóa OpenAI tài liệu nhưng không phải tất cả SDKs hiển thị. Các harness trong bài học này làm điều đó: ngay sau khi luồng mô phỏng mang lại một đối tượng đối số hoàn chỉnh, máy chủ sẽ bắt đầu cuộc gọi đó.

## Ứng dụng

`code/main.py` có hai nửa. Đầu tiên chạy ba cuộc gọi thời tiết mô phỏng tuần tự và song song bằng cách sử dụng `concurrent.futures.ThreadPoolExecutor` và in thời gian đồng hồ treo tường. Nửa sau phát lại một phản hồi streaming giả - các đoạn `arguments` cho ba cuộc gọi song song xen kẽ trên một luồng - và lắp ráp lại chúng với `StreamAccumulator`. Không có LLM, không có mạng, chỉ có logic lắp ráp lại.

Những gì cần xem:

- Bộ đếm thời gian tuần tự đạt 1,8 giây. Bộ đếm thời gian song song đạt 0,8 giây trên cùng một độ trễ giả.
- Bộ tích lũy xử lý các đoạn đến không theo thứ tự bằng cách đệm mỗi id và chỉ phân tích cú pháp khi JSON của mỗi lệnh gọi hoàn tất.
- Người thực thi bắt đầu ngay sau khi các lập luận của id hoàn tất, không phải sau khi tất cả các luồng kết thúc.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-parallel-call-safety-check.md`. Với một công cụ registry, skill kiểm tra công cụ nào an toàn để song song hóa, công cụ nào có phụ thuộc thứ tự và công cụ nào sẽ lấn át rate limits hạ nguồn - trả về registry sửa đổi với cờ `parallel_safe` cho mỗi công cụ.

## Bài tập

1. Chạy `code/main.py` và thay đổi độ trễ mô phỏng. Xác nhận rằng tỷ lệ song song với tuần tự xấp xỉ `max/sum` (số lần chạy thực hơi lệch so với lý tưởng do lập lịch thread, tuần tự hóa và chi phí harness). Ở phân phối độ trễ nào thì song song không còn quan trọng?

2. Mở rộng bộ tích lũy để xử lý trường hợp "cuộc gọi đã bị hủy giữa dòng" bằng cách thả bộ đệm của nó và phát ra sự kiện `cancelled`. Nhà cung cấp nào ghi lại trường hợp này một cách rõ ràng? Kiểm tra ngữ nghĩa `content_block_stop` của Anthropic và hành vi `finish_reason: "length"` của OpenAI.

3. Thay thế hồ bơi thread bằng `asyncio.gather`. Benchmark cả hai. Bạn sẽ thấy những chiến thắng nhỏ trên async vì chi phí chuyển đổi ngữ cảnh thấp hơn, nhưng chỉ khi người thực thi thực hiện I/O.

4. Chọn hai công cụ KHÔNG nên song song (ví dụ: `create_file` rồi `write_file`). Thêm biểu đồ `ordering_dependency` vào registry và cổng quạt ra song song trên biểu đồ đó. Đây là máy móc tối thiểu để lập lịch nhận biết phụ thuộc, mà giai đoạn kỹ thuật agent trong tương lai chính thức hóa.

5. Đọc phần gọi hàm song song của OpenAI và tài liệu `disable_parallel_tool_use` của Anthropic. Xác định một loại công cụ trong thế giới thực mà Anthropic đề xuất tắt tính song song. (Gợi ý: đột biến do hậu quả trên cùng một tài nguyên.)

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Lệnh gọi công cụ song song | "Quạt ra trong một lượt" | Model phát ra nhiều lệnh gọi công cụ trong một tin nhắn trợ lý |
| `parallel_tool_calls` | "Lá cờ của OpenAI" | Bật hoặc tắt tính năng phát nhiều cuộc gọi |
| `disable_parallel_tool_use` | "Anthropic nghịch đảo" | Cờ chọn không tham gia; Mặc định là bật song song |
| Mã lệnh gọi công cụ | "Xử lý tương quan" | Mã định danh cho mỗi cuộc gọi, thông báo kết quả phải lặp lại |
| Tích lũy | "Bộ đệm luồng" | Bộ đệm chuỗi Per-id cho các đoạn `arguments` một phần |
| Hoàn thành không theo thứ tự | "Nhanh nhất trước" | Các cuộc gọi song song kết thúc theo thứ tự không thể đoán trước; ID là chất keo |
| Biểu đồ phần phụ thuộc | "Hạn chế đặt hàng" | Các công cụ có đầu ra đưa vào đầu vào của các công cụ khác; không thể song song |
| Bẫy phân tích cú pháp sớm | "JSON.parse đã phát nổ" | Cố gắng phân tích cú pháp chuỗi `arguments` chưa hoàn chỉnh |
| `streamFunctionCallArguments` | "Gemini 3 feature" | Các đoạn đối số được phát trực tuyến với id duy nhất cho mỗi lệnh gọi |
| Trả lời theo thứ tự hoàn thành | "Đừng chờ đợi tất cả" | Trả lời với kết quả khi chúng đến, được khóa bằng id |

## Đọc thêm

- [OpenAI — Parallel function calling](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) — hành vi mặc định và cờ chọn không tham gia
- [Anthropic — Tool use: implementing tool use](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implementing-tool-use) — phân lô `disable_parallel_tool_use` và kết quả
- [Google — Gemini function calling parallel section](https://ai.google.dev/gemini-api/docs/function-calling) — các cuộc gọi song song tương quan với id từ Gemini 3
- [OpenAI — Streaming responses with tools](https://platform.openai.com/docs/api-reference/responses-streaming) — cụm lại đối số theo khối cho các luồng OpenAI
- [Anthropic — Streaming messages](https://docs.anthropic.com/en/api/messages-streaming) — `content_block_delta` với `input_json_delta`
