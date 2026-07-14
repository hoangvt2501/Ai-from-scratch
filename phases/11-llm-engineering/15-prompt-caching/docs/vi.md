# Prompt Bộ nhớ đệm và bộ nhớ đệm ngữ cảnh

> system prompt của bạn là 4.000 tokens. Ngữ cảnh RAG của bạn là 20.000 tokens. Bạn gửi cả hai với mọi yêu cầu. Bạn cũng phải trả tiền cho cả hai - mọi lúc. Prompt bộ nhớ đệm cho phép nhà cung cấp giữ ấm tiền tố đó và lập hóa đơn cho bạn 10% mức phí bình thường khi sử dụng lại. Nếu được sử dụng đúng cách, nó cắt giảm chi phí inference 50–90% và độ trễ token đầu tiên 40–85%.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 11 · 01 (Kỹ thuật Prompt), Giai đoạn 11 · 05 (Kỹ thuật ngữ cảnh), Giai đoạn 11 · 11 (Bộ nhớ đệm và chi phí)
**Thời lượng:** ~60 phút

## Vấn đề

Một agent mã hóa gửi cùng 15.000 token system prompt cho Claude trên mỗi lượt của cuộc trò chuyện. Hai mươi lượt ở mức $3/M input tokens is $0,90 chỉ trong chi phí đầu vào - trước bất kỳ tin nhắn thực tế nào của người dùng. Nhân với 10.000 cuộc trò chuyện hàng ngày và hóa đơn đạt 9.000/day đô la cho văn bản không bao giờ thay đổi.

Bạn không thể thu nhỏ prompt mà không làm ảnh hưởng đến chất lượng. Bạn không thể tránh gửi nó - model cần nó ở mọi lượt. Động thái duy nhất là ngừng trả giá đầy đủ cho một tiền tố mà nhà cung cấp đã thấy.

Động thái đó là prompt bộ nhớ đệm. Anthropic shipped nó vào tháng 8 năm 2024 (với biến thể TTL kéo dài 1 giờ vào năm 2025), OpenAI tự động hóa nó vào cuối năm đó, Google shipped bộ nhớ đệm ngữ cảnh rõ ràng cùng với Gemini 1.5 và cả ba hiện đều cung cấp nó như một class feature đầu tiên trên models biên giới của họ.

## Khái niệm

![Prompt caching: write once, read cheap](../assets/prompt-caching.svg)

**Thợ máy.** Khi tiền tố của yêu cầu khớp với tiền tố từ yêu cầu gần đây, nhà cung cấp sẽ phục vụ KV-cache từ lần chạy trước thay vì mã hóa lại tokens. Bạn phải trả một khoản phí ghi nhỏ trong lần đầu tiên và chiết khấu đọc lớn mỗi lần sau đó.

**Ba hương vị của nhà cung cấp vào năm 2026.

| Nhà cung cấp | Phong cách API | Giảm giá | Viết phí bảo hiểm | TTL mặc định | Tối thiểu có thể lưu vào bộ nhớ cache |
|---------|-----------|--------------|---------------|-------------|---------------|
| Anthropic | Điểm đánh dấu `cache_control` rõ ràng trên khối nội dung | Giảm giá 90% đầu vào | Phụ phí 25% | 5 phút (có thể kéo dài đến 1 giờ) | 1.024 tokens (Sonnet/Opus), 2.048 (Haiku) |
| OpenAI | Tự động phát hiện tiền tố | Giảm giá 50% đầu vào | Không có | Lên đến 1 giờ (nỗ lực tốt nhất) | 1,024 tokens |
| Google (Gemini) | `CachedContent` API rõ ràng | Lưu trữ hóa đơn; đọc ở mức ~25% so với bình thường | Phí lưu trữ mỗi token· giờ | Người dùng đặt (mặc định 1 giờ) | 4.096 tokens (Flash), 32.768 (Pro) |

**Bất biến.** Chỉ có cả ba tiền tố bộ nhớ đệm. Nếu có bất kỳ token nào khác nhau giữa các yêu cầu, mọi thứ sau token khác nhau đầu tiên đều bị bỏ sót. Đặt các phần * ổn định * ở trên cùng, các phần * có thể thay đổi * ở dưới cùng.

### Bố cục thân thiện với bộ nhớ cache

```
[system prompt]          <-- cache this
[tool definitions]       <-- cache this
[few-shot examples]      <-- cache this
[retrieved documents]    <-- cache if reused, else don't
[conversation history]   <-- cache up to last turn
[current user message]   <-- never cache (different every time)
```

Vi phạm thứ tự - đặt thông báo người dùng phía trên system prompt, xen kẽ truy xuất động giữa vài bức ảnh - và bộ nhớ đệm không bao giờ trúng đích.

### Tính toán hòa vốn

Phí ghi 25% của Anthropic có nghĩa là một khối được lưu trong bộ nhớ cache phải được đọc ít nhất hai lần để tiết kiệm tiền ròng. 1 ghi + 1 lần đọc trung bình 0,675x chi phí cho mỗi yêu cầu (tiết kiệm 32%); 1 ghi + 10 lần đọc trung bình 0,205x (tiết kiệm 80%). Quy tắc chung: lưu vào bộ nhớ đệm bất cứ thứ gì bạn mong muốn sử dụng lại ít nhất 3 lần trong TTL.

## Tự xây dựng

### Bước 1: Anthropic prompt bộ nhớ đệm với các điểm đánh dấu rõ ràng

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "You are a senior Python reviewer. Follow the rubric exactly.\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )
```

Điểm đánh dấu `cache_control` yêu cầu Anthropic lưu trữ khối trong 5 phút. Sử dụng lại trong cửa sổ đó sẽ truy cập; sử dụng lại sau khi hết hạn và ghi lại.

**Các trường sử dụng phản hồi:**

```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # paid at 1.25x
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # paid at 0.1x
```

Kiểm tra cả hai trường trong CI — nếu `cache_read_input_tokens` ở mức 0 trên các yêu cầu, khóa bộ nhớ đệm của bạn đang bị trôi.

### Bước 2: TTL kéo dài một giờ

Đối với các tác vụ batch chạy trong thời gian dài, mặc định 5 phút sẽ hết hạn giữa các tác vụ. Đặt `ttl`:

```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

TTL 1 giờ có giá gấp 2 lần phí ghi (50% so với mức cơ sở thay vì 25%) nhưng hoàn vốn nhanh chóng cho bất kỳ batch nào sử dụng lại tiền tố hơn 5 lần.

### Bước 3: OpenAI bộ nhớ đệm tự động

OpenAI không cung cấp cho bạn gì để cấu hình. Bất kỳ tiền tố nào trên 1.024 tokens khớp với yêu cầu gần đây sẽ tự động được giảm giá 50%.

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # long and stable
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # the discounted portion
```

Quy tắc bố cục thân thiện với bộ nhớ cache tương tự cũng được áp dụng. Hai thứ giết bộ nhớ cache của OpenAI không giết Anthropic: thay đổi trường `user` (được sử dụng làm thành phần khóa bộ nhớ đệm) và sắp xếp lại các công cụ.

### Bước 4: Gemini bộ nhớ đệm ngữ cảnh rõ ràng

Gemini coi bộ nhớ đệm là đối tượng class đầu tiên mà bạn tạo và đặt tên:

```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3-pro",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3-pro",
    contents=["Review this code:\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```

Gemini tính phí lưu trữ mỗi token·giờ miễn là bộ nhớ đệm còn tồn tại và đọc ở mức ~25% tốc độ đầu vào bình thường. Đây là hình dạng phù hợp khi bạn sử dụng lại cùng một prompt khổng lồ trong nhiều sessions trong nhiều ngày.

### Bước 5: đo tỷ lệ trúng trong production

Xem `code/main.py` để biết kế toán ba nhà cung cấp mô phỏng theo dõi số lượng write/read/miss và tính toán chi phí kết hợp trên mỗi 1 nghìn yêu cầu. Gate triển khai với tỷ lệ truy cập mục tiêu — hầu hết các thiết lập production Anthropic sẽ thấy >80% phân số đọc sau khi khởi động.

## Những cạm bẫy vẫn ship vào năm 2026

- **Dấu thời gian động ở trên cùng.** `"Current time: 2026-04-22 15:30:02"` ở đầu system prompt. Mọi yêu cầu đều bị trượt. Di chuyển dấu thời gian bên dưới điểm ngắt bộ nhớ đệm.
- **Sắp xếp lại công cụ.** Tuần tự hóa các công cụ theo thứ tự ổn định — xáo trộn lại giữa các lần triển khai sẽ phá vỡ mọi lần truy cập.
- **Văn bản tự do gần như trùng lặp.** "Bạn hữu ích." vs "Bạn là một trợ lý hữu ích." — chênh lệch một byte = bỏ lỡ hoàn toàn.
- **Các khối quá nhỏ.** Anthropic thực thi mức sàn 1.024 token (2.048 đối với Haiku). Các khối nhỏ hơn âm thầm không lưu vào bộ nhớ đệm.
- **Bảng điều khiển chi phí mù.** Chia "tokens đầu vào" thành bộ nhớ cache và không được lưu trong bộ nhớ đệm. Nếu không, lưu lượng truy cập giảm trông giống như chiến thắng trong bộ nhớ đệm.

## Ứng dụng

Bộ nhớ đệm năm 2026 stack:

| Tình huống | Chọn |
|-----------|------|
| Agent với system prompt 10k+ ổn định, nhiều lượt | Anthropic `cache_control` với TTL 5 phút |
| Batch việc sử dụng lại tiền tố trong 30+ phút | Anthropic với `ttl: "1h"` |
| endpoints phi máy chủ trên GPT-5, không có cơ sở hạ tầng tùy chỉnh | OpenAI tự động (chỉ cần làm cho tiền tố của bạn ổn định và dài) |
| Tái sử dụng kho dữ liệu code/doc khổng lồ trong nhiều ngày | Gemini `CachedContent` rõ ràng |
| Dự phòng giữa các nhà cung cấp | Giữ bố cục tiền tố có thể lưu vào bộ nhớ đệm giống hệt nhau giữa các nhà cung cấp để mọi lần truy cập đều hoạt động |

Kết hợp với bộ nhớ đệm ngữ nghĩa (Giai đoạn 11 · 11) cho lớp thông báo người dùng: prompt xử lý bộ nhớ đệm *token-giống hệt *tái sử dụng, xử lý bộ nhớ đệm ngữ nghĩa *giống hệt ý nghĩa*.

## Sản phẩm bàn giao

Lưu `outputs/skill-prompt-caching-planner.md`:

```markdown
---
name: prompt-caching-planner
description: Design a cache-friendly prompt layout and pick the right provider caching mode.
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

Given a prompt (system + tools + few-shot + retrieval + history + user) and a usage profile (requests per hour, TTL needed, provider), output:

1. Layout. Reordered sections with a single cache breakpoint marked; explain which sections are stable, which are volatile.
2. Provider mode. Anthropic cache_control, OpenAI automatic, or Gemini CachedContent. Justify from TTL and reuse pattern.
3. Break-even. Expected reads per write within TTL; net cost vs no-cache with math.
4. Verification plan. CI assertion that cache_read_input_tokens > 0 on the second identical request; dashboard split by cached vs uncached tokens.
5. Failure modes. List the three most likely reasons the cache will miss in this setup (dynamic timestamp, tool reorder, near-duplicate text) and how you will prevent each.

Refuse to ship a cache plan that places a dynamic field above the breakpoint. Refuse to enable 1h TTL without a reuse count that makes the 2x write premium pay back.
```

## Bài tập

1. **Dễ dàng.** Thực hiện một cuộc trò chuyện 10 lượt với 5.000 token system prompt chống lại Claude. Chạy nó mà không cần `cache_control` và sau đó với. Báo cáo hóa đơn token đầu vào cho từng người.
2. **Trung bình.** Viết một harness thử nghiệm, với mẫu prompt và nhật ký yêu cầu, tính toán tỷ lệ truy cập dự kiến và tiết kiệm đô la cho mỗi nhà cung cấp (Anthropic 5 triệu, Anthropic 1 giờ, OpenAI tự động Gemini rõ ràng).
3. **Khó.** Xây dựng optimizer bố cục: cho một prompt và danh sách các trường được đánh dấu `stable=True/False`, hãy viết lại prompt để đặt một điểm ngắt bộ nhớ cache duy nhất ở vị trí thân thiện với bộ nhớ cache tối đa mà không làm mất thông tin. Xác minh trên một Anthropic endpoint thực.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Prompt bộ nhớ đệm | "Làm cho dài prompts rẻ" | Sử dụng lại bộ nhớ đệm KV phía nhà cung cấp để khớp tiền tố; Giảm giá 50-90% cho tokens nhập lặp lại. |
| `cache_control` | "Điểm đánh dấu Anthropic" | Content-block khai báo "mọi thứ cho đến đây đều có thể lưu vào bộ nhớ đệm"; `{"type": "ephemeral"}`. |
| Ghi bộ nhớ cache | "Trả phí bảo hiểm" | Yêu cầu đầu tiên điền vào bộ nhớ đệm; được tính phí ở mức ~1,25x đầu vào trên Anthropic, miễn phí trên OpenAI. |
| Đọc bộ nhớ cache | "Giảm giá" | Các yêu cầu tiếp theo khớp với tiền tố; được tính phí ở mức 10% (Anthropic), 50% (OpenAI), ~25% (Gemini). |
| TTL | "Nó sống được bao lâu" | Vài giây bộ nhớ đệm vẫn ấm; Anthropic mặc định 5m (có thể mở rộng 1h), OpenAI nỗ lực tốt nhất lên đến 1h, Gemini do người dùng đặt. |
| TTL mở rộng | "Bộ nhớ đệm Anthropic 1 giờ" | `{"type": "ephemeral", "ttl": "1h"}`; 2x viết cao cấp nhưng đáng để batch sử dụng lại. |
| Đối sánh tiền tố | "Tại sao bộ nhớ cache của tôi bị trượt" | Bộ nhớ đệm chỉ đạt được khi mọi token từ khi khởi động đến điểm ngắt giống hệt byte. |
| Bộ nhớ đệm ngữ cảnh (Gemini) | "Rõ ràng" | Đối tượng bộ nhớ đệm được đặt tên của Google; tốt nhất để sử dụng lại kho dữ liệu lớn trong nhiều ngày. |

## Đọc thêm

- [Anthropic — Prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) - `cache_control`, TTL 1h, bàn hòa vốn.
- [OpenAI — Prompt caching](https://platform.openai.com/docs/guides/prompt-caching) - khớp tiền tố tự động.
- [Google — Context caching](https://ai.google.dev/gemini-api/docs/caching) — giá `CachedContent` API và lưu trữ.
- [Anthropic engineering — Prompt caching for long-context workloads](https://www.anthropic.com/news/prompt-caching) — bài ra mắt ban đầu với số độ trễ.
- Giai đoạn 11 · 05 (Kỹ thuật ngữ cảnh) — nơi cắt prompt để bộ nhớ đệm có thể hạ cánh.
- Giai đoạn 11 · 11 (Bộ nhớ đệm và chi phí) - ghép nối bộ nhớ đệm prompt với bộ nhớ đệm ngữ nghĩa trên tin nhắn của người dùng.
- [Pope et al., "Efficiently Scaling Transformer Inference" (2022)](https://arxiv.org/abs/2211.05102) — model bộ nhớ đệm KV mà bộ nhớ đệm prompt hiển thị cho người dùng; giải thích lý do tại sao tiền tố được lưu trong bộ nhớ đệm rẻ hơn ~10× để đọc lại so với tính toán lại.
- [Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" (2023)](https://arxiv.org/abs/2308.16369) - điền trước là giai đoạn prompt các phím tắt bộ nhớ đệm; bài báo này giải thích lý do tại sao TTFT giảm đáng kể khi truy cập vào bộ nhớ đệm trong khi TPOT không bị ảnh hưởng.
- [Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)](https://arxiv.org/abs/2211.17192) - bộ nhớ đệm prompt nằm cùng với giải mã đầu cơ, Flash Attention và MQA/GQA là đòn bẩy bẻ cong đường cong chi phí inference; đọc điều này cho ba người còn lại.
