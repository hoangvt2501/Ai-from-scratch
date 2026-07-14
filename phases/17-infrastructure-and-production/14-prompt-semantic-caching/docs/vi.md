# Prompt Bộ nhớ đệm và kinh tế bộ nhớ đệm ngữ nghĩa

> **Ảnh chụp nhanh về giá ngày 2026-04.** Các tuyên bố số bên dưới phản ánh bảng giá của nhà cung cấp được ghi lại tại ấn phẩm của bài học này; Xác minh dựa trên các tài liệu được liên kết trước khi trích dẫn chúng ở hạ lưu.

> Bộ nhớ đệm xảy ra ở hai lớp. L2 (cấp nhà cung cấp) prompt/prefix bộ nhớ đệm sử dụng lại attention KV cho các tiền tố lặp lại — tài liệu bộ nhớ đệm prompt của Anthropic quảng cáo giảm tới 90% chi phí và giảm 85% độ trễ trên prompts dài; đối với Claude 3.5 lần đọc bộ nhớ đệm Sonnet $0.30/M vs $ 3.00/M mới với TTL 5 phút và phí ghi gấp 2 lần cho tùy chọn TTL 1 giờ (docs.anthropic.com, 2026-04). OpenAI prompt bộ nhớ đệm tự động áp dụng cho tokens prompts ≥1024 và giá đầu vào được lưu trong bộ nhớ đệm ở mức chiết khấu khoảng 90% so với Fresh (platform.openai.com, 2026-04); Tỷ lệ chính xác trên mỗi model được lưu trong bộ nhớ đệm phụ thuộc vào thẻ giá trực tiếp. Bộ nhớ đệm ngữ nghĩa L1 (cấp ứng dụng) bỏ qua LLM hoàn toàn dựa trên embedding lần truy cập tương tự. Nhà cung cấp "95% accuracy" đề cập đến tính đúng đắn của trận đấu, không phải tỷ lệ trúng thưởng — tỷ lệ truy cập production được báo cáo dao động từ 10% (trò chuyện mở) đến 70% (Câu hỏi thường gặp có cấu trúc); Cả hai nhà cung cấp đều không công bố đường cơ sở chính thức, vì vậy hãy coi đây là telemetry cộng đồng hơn là đảm bảo. Những cạm bẫy production: song song hóa giết chết bộ nhớ đệm (N yêu cầu song song được đưa ra trước lần ghi bộ nhớ đệm đầu tiên có thể làm tăng chi tiêu gấp nhiều lần) và nội dung động bên trong tiền tố ngăn chặn hoàn toàn việc truy cập bộ nhớ đệm. ProjectDiscovery báo cáo đã chuyển từ tỷ lệ truy cập từ 7% lên 74% (2025-11) bằng cách di chuyển văn bản động ra khỏi tiền tố có thể lưu vào bộ nhớ đệm.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, toy two-layer cache simulator)
**Kiến thức tiên quyết:** Giai đoạn 17 · 04 (vLLM Phục vụ Nội bộ), Giai đoạn 17 · 06 (SGLang RadixAttention)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Phân biệt bộ nhớ đệm prompt/prefix L2 (sử dụng lại KV tại nhà cung cấp) với bộ nhớ đệm ngữ nghĩa L1 (bỏ qua LLM trên prompts tương tự).
- Giải thích `cache_control` đánh dấu rõ ràng của Anthropic và hai tùy chọn TTL (5 phút so với 1 giờ) với hệ số giá của chúng.
- Tính toán khoản tiết kiệm hàng tháng dự kiến với tỷ lệ trúng đích, hỗn hợp prompt/response và giá token.
- Đặt tên cho mô hình chống song song làm tăng hóa đơn lên 5-10 lần và mô hình chống nội dung động làm giảm tỷ lệ trúng đòn.

## Vấn đề

Bạn thêm bộ nhớ đệm prompt vào dịch vụ RAG của mình. Hóa đơn vẫn không thay đổi. Bạn đo tỷ lệ trúng đích; nó là 7%. prompts của bạn trông tĩnh nhưng không phải vậy - system prompt bao gồm ngày hiện tại được định dạng theo phút, ID yêu cầu và sắp xếp lại ví dụ ngẫu nhiên cho sự đa dạng. Mỗi yêu cầu ghi một mục nhập bộ nhớ đệm mới, đọc bằng không.

Riêng biệt, agent của bạn chạy mười lệnh gọi công cụ song song cho mỗi câu hỏi của người dùng. Tất cả mười đều đến nhà cung cấp trước khi quá trình ghi bộ nhớ đệm đầu tiên hoàn tất. Mười ghi, không đọc. Hóa đơn của bạn gấp 5-10 lần so với chi phí "có bộ nhớ đệm".

Bộ nhớ đệm là một giao thức, không phải là một cờ. Hai lớp, hai chế độ hỏng hóc khác nhau.

## Khái niệm

### L2 — bộ nhớ đệm prompt/prefix của nhà cung cấp

Nhà cung cấp lưu trữ KV attention cho tiền tố có thể lưu vào bộ nhớ đệm và sử dụng lại nó trong yêu cầu tiếp theo khớp với tiền tố. Bạn trả chi phí ghi một lần, đọc gần như miễn phí.

**Anthropic (Claude 3.5 / 3.7 / 4 series)**: điểm đánh dấu `cache_control` rõ ràng trong yêu cầu. Bạn gắn thẻ khối nào có thể lưu vào bộ nhớ đệm. TTL: 5 phút (chi phí ghi 1,25 lần cơ sở) hoặc 1 giờ (chi phí ghi 2 lần cơ sở). Bộ nhớ cache viết: $0.30/M on Claude 3.5 Sonnet vs $ 3.00/M tươi - rẻ hơn 10 lần (docs.anthropic.com, tính đến năm 2026-04). Tỷ lệ khác nhau trên mỗi model (Opus/Haiku công bố riêng); Luôn kiểm tra chéo trang giá trực tiếp.

**OpenAI**: Bộ nhớ đệm tự động cho tokens prompts ≥1024 (platform.openai.com, 2026-04). Không có cờ rõ ràng. Đầu vào được lưu trong bộ nhớ cache rẻ hơn khoảng 10 lần so với mới trên thẻ giá gpt-4o/gpt-5 hiện tại. Cả tài liệu và ghi chú phát hành đều không công bố đường cơ sở tỷ lệ truy cập chính thức; Báo cáo cộng đồng tập trung khoảng 30–60% với thiết kế prompt cẩn thận. Theo dõi `usage.cached_tokens` để đo lường của riêng bạn.

**Google (Gemini)**: bộ nhớ đệm ngữ cảnh thông qua API rõ ràng; Bối cảnh 1M-token có nghĩa là bộ nhớ đệm thậm chí còn trả nhiều hơn.

**Tự lưu trữ (vLLM, SGLang)**: Giai đoạn 17 · 06 bao gồm RadixAttention — cùng một mẫu tại máy tính của riêng bạn.

### L1 — bộ nhớ đệm ngữ nghĩa cấp ứng dụng

Trước khi gọi LLM, hãy băm prompt, nhúng nó và tìm một yêu cầu được lưu trong bộ nhớ cache tương tự (độ tương đồng cosine trên ngưỡng, thường là 0,95+). Khi trúng đích, trả về phản hồi đã lưu trong bộ nhớ đệm. Khi trượt, hãy gọi LLM và lưu kết quả vào bộ nhớ đệm.

Mã nguồn mở: Redis Vector Similarity, GPTCache, Qdrant. Thương mại: Portkey Cache, Helicone Cache.

Tuyên bố của nhà cung cấp accuracy đề cập đến tần suất phản hồi được trả về trong bộ nhớ đệm phù hợp về mặt ngữ nghĩa - không phải tần suất bạn đánh. Tỷ lệ truy cập Production:

- Trò chuyện mở: 10-15%.
- Câu hỏi thường gặp / hỗ trợ có cấu trúc: 40-70%.
- Câu hỏi mã: 20-30% (các biến thể nhỏ tiêu diệt lượt truy cập).
- prompts lặp lại agents giọng nói: 50-80% (bộ cố định chuẩn hóa giọng nói).

### Mô hình chống song song

agent của bạn thực hiện 10 lệnh gọi công cụ song song. Tất cả 10 đều có cùng 4K-token system prompt. Anthropic ghi bộ nhớ đệm là theo yêu cầu; Quá trình ghi bộ nhớ cache đầu tiên hoàn thành khoảng 300 mili giây sau khi nhà cung cấp nhìn thấy prompt. Yêu cầu 2-10 đến trong cùng một cửa sổ mili giây và mỗi yêu cầu bị bỏ lỡ bộ nhớ đệm. Bạn trả 10 phí ghi, 0 chiết khấu đọc.

Khắc phục: batch với sequential-first - thực hiện yêu cầu 1 một mình, sau đó kích hoạt 2-10 khi bộ nhớ cache của 1 đã được điền. Thêm 300 ms vào lệnh gọi công cụ đầu tiên; Tiết kiệm 5-10 lần hóa đơn.

### Nội dung động chống mẫu

system prompt của bạn trông như sau:

```
You are a helpful assistant. The current time is 14:32:17.
User ID: abc123. Today is Tuesday...
```

Mỗi yêu cầu là duy nhất. Mọi yêu cầu đều viết. Không có lượt truy cập.

Khắc phục: di chuyển mọi thứ thực sự tĩnh vào tiền tố có thể lưu vào bộ nhớ cache; Thêm nội dung động sau ranh giới bộ nhớ đệm:

```
[cacheable]
You are a helpful assistant. [rules, examples, instructions]
[/cacheable]
[dynamic, not cached]
Current time: 14:32:17. User: abc123.
```

ProjectDiscovery đã chuyển từ 7% lên 74% tỷ lệ truy cập bộ nhớ cache theo cách này và công bố giải phẫu.

### Stack batch + bộ nhớ đệm cho khối lượng công việc qua đêm

Batch APIs (Giai đoạn 17 · 15) giảm giá 50% trong vòng 24 giờ. Đầu vào được lưu trong bộ nhớ cache ở trên cùng giúp bạn ~10 lần trên hết. Khối lượng công việc phân loại, ghi nhãn và tạo báo cáo qua đêm có thể giảm xuống ~10% chi phí đồng bộ-không lưu bộ nhớ đệm bằng cách xếp chồng lên nhau.

### Những con số bạn nên nhớ

Các điểm định giá được ghi lại vào năm 2026-04 từ tài liệu của nhà cung cấp được liên kết và trôi dạt vài tháng một lần - kiểm tra lại trước khi dựa vào chúng.

- Anthropic đọc bộ nhớ cach: $0.30/M trên Claude 3.5 Sonnet, rẻ hơn khoảng 10 lần so với đầu vào mới (tài liệu.anthropic.com).
- Anthropic cao cấp ghi bộ nhớ đệm: 1,25x (TTL 5 phút) hoặc 2x (TTL 1 giờ).
- OpenAI bộ nhớ đệm tự động: áp dụng cho prompts ≥1024 tokens; đầu vào được lưu trong bộ nhớ cache có giá khoảng 10% đầu vào mới trên thẻ giá hiện tại (platform.openai.com).
- Tỷ lệ truy cập bộ nhớ đệm ngữ nghĩa (do cộng đồng báo cáo): ~10% trò chuyện mở; Câu hỏi thường gặp có cấu trúc lên đến ~70%. Không phải là đường cơ sở do nhà cung cấp ghi lại.
- ProjectDiscovery: Tỷ lệ truy cập 7% → 74% bằng cách di chuyển động ra khỏi tiền tố (blog dự án, 2025-11).
- Chống mô hình song song: các báo cáo điển hình về lạm phát hóa đơn gấp 5–10 lần khi N yêu cầu song song bỏ lỡ lần ghi bộ nhớ đệm đầu tiên.

## Ứng dụng

`code/main.py` mô phỏng bộ nhớ đệm L1 + L2 trên khối lượng công việc hỗn hợp. Báo cáo tỷ lệ trúng đích, lập hóa đơn và hiển thị hình phạt song song.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-cache-auditor.md`. Với prompt mẫu và lưu lượng truy cập, kiểm tra khả năng lưu trữ và đề xuất tái cấu trúc.

## Bài tập

1. Chạy `code/main.py`. Chuyển đổi cờ song song. Hóa đơn thay đổi bao nhiêu?
2. system prompt của bạn có một ngày. Di chuyển nó ra. Hiển thị before/after toán tỷ lệ trúng đích.
3. Tính hòa vốn cho TTL 1 giờ (ghi 2 lần) so với TTL 5 phút (ghi 1,25 lần) dựa trên tỷ lệ đến yêu cầu của bạn.
4. Bộ nhớ đệm ngữ nghĩa ở ngưỡng 0,95 đạt 20%. Ở mức 0,85, nó đạt 50% nhưng bạn thấy các phản hồi được lưu trong bộ nhớ cache không chính xác. Chọn đúng ngưỡng và biện minh.
5. Bạn batch 10 truy vấn phụ song song cho mỗi câu hỏi của người dùng. Viết lại để thân thiện với bộ nhớ cache mà không thêm độ trễ từ đầu đến cuối.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Bộ nhớ đệm prompt L2 | "bộ nhớ đệm tiền tố" | Nhà cung cấp lưu trữ KV cho tiền tố lặp lại |
| `cache_control` | "Anthropic điểm đánh dấu bộ nhớ cache" | Đánh dấu thuộc tính rõ ràng các khối có thể lưu trong bộ nhớ đệm |
| Ghi bộ nhớ cache cao cấp | "Viết thuế" | Chi phí bổ sung cho lần bỏ lỡ bộ nhớ đệm đầu tiên (1,25x hoặc 2x) |
| Bộ nhớ đệm ngữ nghĩa L1 | "embedding bộ nhớ đệm" | Hàm băm và nhúng cấp ứng dụng trước khi gọi LLM |
| GPTCache | "LLM lib bộ nhớ đệm" | Thư viện bộ nhớ cache OSS L1 phổ biến |
| Tỷ lệ trúng bộ nhớ đệm | "lượt truy cập / tổng số" | Phần yêu cầu được phục vụ từ bộ nhớ đệm |
| Song song hóa chống mẫu | "bẫy N-write" | N yêu cầu song song bỏ lỡ bộ nhớ đệm N lần |
| Bẫy nội dung động | "Bẫy thời gian trong prompt" | Byte động trong tỷ lệ truy cập tiêu diệt tiền tố |
| RadixChú ý | "bộ nhớ đệm nội bộ" | Triển khai tiền tố-cache của SGLang |

## Đọc thêm

- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — ngữ nghĩa `cache_control` chính thức và TTL.
- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching) — hành vi bộ nhớ đệm tự động và tính đủ điều kiện.
- [TianPan — Semantic Caching for LLMs Production](https://tianpan.co/blog/2026-04-10-semantic-caching-llm-production)
- [ProjectDiscovery — Cut LLM Costs 59% With Prompt Caching](https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching)
- [DigitalOcean / Anthropic — Prompt Caching](https://www.digitalocean.com/blog/prompt-caching-with-digital-ocean)
