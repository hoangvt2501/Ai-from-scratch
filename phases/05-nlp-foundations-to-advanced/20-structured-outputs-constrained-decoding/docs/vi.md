# Đầu ra có cấu trúc và giải mã hạn chế

> Yêu cầu một LLM cho JSON. Nhận JSON hầu hết thời gian. Trong production, "hầu hết" là vấn đề. Giải mã hạn chế biến "hầu hết" thành "luôn luôn" bằng cách chỉnh sửa logits trước khi sampling.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 5 · 17 (Chatbots), Giai đoạn 5 · 19 (Từ phụ Tokenization)
**Thời lượng:** ~60 phút

## Vấn đề

Một bộ phân loại prompts một LLM: "Trả về một trong số {tích cực, tiêu cực, trung lập}." Câu model trả về "Tâm lý là tích cực - đánh giá này rất thuận lợi vì khách hàng tuyên bố rõ ràng rằng họ ...". Trình phân tích cú pháp của bạn gặp sự cố. F1 của bộ phân loại của bạn là 0.0.

Tạo dạng tự do không phải là một hợp đồng. Đó là một gợi ý. Một hệ thống production cần một hợp đồng.

Ba lớp tồn tại vào năm 2026.

1. **Prompting.** Hỏi một cách tử tế. "Chỉ trả lại đối tượng JSON." Hoạt động ~80% trên các models biên giới, ít hơn trên những  nhỏ hơn.
2. **Đầu ra có cấu trúc gốc APIs.** OpenAI `response_format`, sử dụng công cụ Anthropic Gemini JSON chế độ. Đáng tin cậy trên các schemas được hỗ trợ. Nhà cung cấp bị khóa.
3. **Giải mã hạn chế.** Sửa đổi logits ở mọi bước tạo để model *không thể* phát ra tokens không hợp lệ. 100% hợp lệ bởi xây dựng. Hoạt động trên bất kỳ model địa phương nào.

Bài học này xây dựng trực giác cho cả ba và đặt tên khi nào cần đạt được cái nào.

## Khái niệm

![Constrained decoding masking invalid tokens at each step](../assets/constrained-decoding.svg)

**Cách thức hoạt động của giải mã hạn chế.** Ở mỗi bước tạo, LLM tạo ra một logit vector trên toàn bộ từ vựng (~100k tokens). Bộ xử lý * logit * nằm giữa model và bộ lấy mẫu. Nó tính toán tokens nào hợp lệ với vị trí hiện tại trong ngữ pháp đích - JSON Schema, regex, ngữ pháp không ngữ cảnh - và đặt logits của tất cả các tokens không hợp lệ thành vô cực âm. softmax trên logits còn lại đặt khối lượng xác suất chỉ trên các tiếp tục hợp lệ.

Triển khai vào năm 2026:

- **Outlines.** Biên dịch JSON Schema hoặc regex vào một máy trạng thái hữu hạn. Mỗi token đều nhận được một tra cứu O (1) hợp lệ token tiếp theo. dựa trên FSM, vì vậy đệ quy schemas cần làm phẳng.
- **XGrammar / llguidance.** Công cụ ngữ pháp không ngữ cảnh. Xử lý JSON Schema đệ quy. Chi phí giải mã gần như bằng không. OpenAI ghi nhận llguidance trong việc triển khai đầu ra có cấu trúc năm 2025 của họ.
- **Giải mã có hướng dẫn vLLM.** Tích hợp `guided_json`, `guided_regex`, `guided_choice`, `guided_grammar` thông qua các phần phụ trợ Outlines, XGrammar hoặc lm-format-enforcer.
- **Người hướng dẫn.** Bao bì dựa trên Pydantic trên bất kỳ LLM nào. Thử lại khi xác thực không thành công. Cross-provider, nhưng không sửa đổi logits — nó dựa vào các lần thử lại + prompts nhận biết đầu ra có cấu trúc.

### Kết quả phản trực giác

Giải mã bị ràng buộc thường *nhanh hơn* so với việc tạo không bị ràng buộc. Hai lý do. Đầu tiên, nó thu hẹp không gian tìm kiếm token tiếp theo. Thứ hai, các triển khai thông minh bỏ qua token thế hệ hoàn toàn để tokens bắt buộc (giàn giáo như `{"name": "` - mỗi byte được xác định).

### Cạm bẫy khiến bạn phải trả giá

Thứ tự thực địa rất quan trọng. Đặt `answer` trước `reasoning`, và model commits đến câu trả lời trước khi nó suy nghĩ. JSON là hợp lệ. Câu trả lời là sai. Không có xác thực nào bắt được nó.

```json
// BAD
{"answer": "yes", "reasoning": "because ..."}

// GOOD
{"reasoning": "... therefore ...", "answer": "yes"}
```

Schema thứ tự trường là logic, không phải định dạng.

## Tự xây dựng

### Bước 1: tạo regex hạn chế từ đầu

Xem `code/main.py` để biết triển khai FSM độc lập. Ý tưởng cốt lõi trong 30 dòng:

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

FSM theo dõi những phần ngữ pháp mà chúng tôi đã thỏa mãn cho đến nay. `valid_tokens(state, tokenizer)` tính toán từ vựng nào tokens có thể nâng cao FSM mà không để lại đường dẫn chấp nhận.

### Bước 2: Đề cương cho JSON Schema

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

Không có lỗi xác thực. Bao giờ. FSM làm cho đầu ra không hợp lệ không thể truy cập được.

### Bước 3: Người hướng dẫn cho Pydantic bất khả tri của nhà cung cấp

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

Cơ chế khác nhau. Người hướng dẫn không chạm vào logits. Nó định dạng schema thành prompt, phân tích cú pháp đầu ra và thử lại khi xác thực không thành công (mặc định 3 lần). Hoạt động với bất kỳ nhà cung cấp nào. Thử lại làm tăng thêm độ trễ và chi phí. Tính di động của nhiều nhà cung cấp là điểm bán hàng.

### Bước 4: APIs nhà cung cấp gốc

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

Giải mã ràng buộc phía Server. Độ tin cậy tương đương với Outlines cho các schemas được hỗ trợ. Không có quản lý model địa phương. Khóa bạn với nhà cung cấp.

## Cạm bẫy

- **schemas đệ quy.** Phác thảo làm phẳng đệ quy đến một độ sâu cố định. Đầu ra có cấu trúc cây (nhận xét lồng nhau, AST) cần XGrammar hoặc llguidance (dựa trên CFG).
- **Số enum khổng lồ.** 10.000 enum tùy chọn biên dịch chậm hoặc hết thời gian chờ. Chuyển sang một người truy lùng: dự đoán top-k ứng cử viên trước, hạn chế những ứng cử viên đó.
- **Ngữ pháp quá nghiêm ngặt.** Buộc `date: "YYYY-MM-DD"` regex và model không thể xuất `"unknown"` cho các ngày bị thiếu. Model bù đắp bằng cách phát minh ra một ngày. Cho phép `null` hoặc lính canh.
- **Cam kết sớm.** Xem cạm bẫy đặt hàng thực địa ở trên. Luôn đặt lý luận lên hàng đầu.
- **Chế độ JSON của nhà cung cấp không có schema.** Chế độ Pure JSON chỉ đảm bảo JSON hợp lệ, không hợp lệ *cho trường hợp sử dụng của bạn*. Luôn cung cấp đầy đủ schema.

## Ứng dụng

stack năm 2026:

| Tình huống | Chọn |
|-----------|------|
| OpenAI/Anthropic/Google model, schema đơn giản | Đầu ra có cấu trúc của nhà cung cấp gốc |
| Bất kỳ nhà cung cấp nào, quy trình làm việc Pydantic, đều có thể chấp nhận thử lại | Người hướng dẫn |
| model cục bộ, cần 100% hiệu lực, schema phẳng | Đề cương (FSM) |
| model cục bộ, schema đệ quy | XGrammar hoặc llhướng dẫn |
| inference server tự lưu trữ | Giải mã có hướng dẫn vLLM |
| Batch xử lý với các lần thử lại có thể chấp nhận được | Người hướng dẫn + model rẻ nhất |

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-structured-output-picker.md`:

```markdown
---
name: structured-output-picker
description: Choose a structured output approach, schema design, and validation plan.
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

Given a use case (provider, latency budget, schema complexity, failure tolerance), output:

1. Mechanism. Native vendor structured output, Instructor retries, Outlines FSM, or XGrammar CFG. One-sentence reason.
2. Schema design. Field order (reasoning first, answer last), nullable fields for "unknown", enum vs regex, required fields.
3. Failure strategy. Max retries, fallback model, graceful `null` handling, out-of-distribution refusal.
4. Validation plan. Schema compliance rate (target 100%), semantic validity (LLM-judge), field-coverage rate, latency p50/p99.

Refuse any design that puts `answer` or `decision` before reasoning fields. Refuse to use bare JSON mode without a schema. Flag recursive schemas behind an FSM-only library.
```

## Bài tập

1. **Dễ dàng.** Prompt một model trọng lượng mở nhỏ (ví dụ: Llama-3.2-3B) mà không cần giải mã hạn chế cho `Review(sentiment, confidence, evidence_span)`. Đo lường phân số phân tích cú pháp là hợp lệ JSON trên 100 bài đánh giá.
2. **Trung bình.** Cùng một kho dữ liệu với chế độ JSON Outlines. So sánh tỷ lệ tuân thủ, độ trễ và accuracy ngữ nghĩa.
3. **Khó.** Triển khai decoder bị hạn chế bởi regex từ đầu cho các số điện thoại (`\d{3}-\d{3}-\d{4}`). Xác minh 0 đầu ra không hợp lệ trên 1000 samples.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Giải mã hạn chế | Buộc đầu ra hợp lệ | Đeo khẩu trang token logits không hợp lệ ở mọi bước thế hệ. |
| Bộ xử lý Logit | Điều ràng buộc | Chức năng: `(logits, state) -> masked_logits`. |
| FSM | Máy trạng thái hữu hạn | Biểu diễn ngữ pháp biên soạn; O(1) tra cứu token tiếp theo hợp lệ. |
| CFG | Ngữ pháp không ngữ cảnh | Ngữ pháp xử lý đệ quy; chậm hơn nhưng biểu cảm hơn FSM. |
| Schema thứ tự trường | Điều đó có quan trọng không? | Có - commits trường đầu tiên; luôn đặt lý luận trước câu trả lời. |
| Giải mã có hướng dẫn | Tên của vLLM cho nó | Cùng một khái niệm, được tích hợp vào inference server. |
| Chế độ JSON | Phiên bản đầu tiên của OpenAI | Bảo lãnh JSON cú pháp; KHÔNG đảm bảo schema phù hợp. |

## Đọc thêm

- [Willard, Louf (2023). Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702) - bài báo Đề cương.
- [XGrammar paper (2024)](https://arxiv.org/abs/2411.15100) - giải mã ràng buộc dựa trên CFG nhanh chóng.
- [vLLM — Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html) — tích hợp inference server.
- [OpenAI — Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs) - API tham khảo + gotchas.
- [Instructor library](https://python.useinstructor.com/) - Pydantic + thử lại giữa các nhà cung cấp.
- [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) - điểm chuẩn 6 frameworks giải mã hạn chế.
