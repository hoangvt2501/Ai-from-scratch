# Chatbots - Dựa trên quy tắc đến thần kinh để LLM Agents

> ELIZA trả lời bằng các mẫu phù hợp. DialogFlow đã ánh xạ ý định. GPT trả lời từ tạ. Claude chạy các công cụ và xác minh. Mỗi thời đại giải quyết thất bại tồi tệ nhất của thời đại trước.

**Loại:** Học
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 5 · 13 (Trả lời câu hỏi), Giai đoạn 5 · 14 (Truy xuất thông tin)
**Thời lượng:** ~75 phút

## Vấn đề

Một người dùng nói "Tôi muốn thay đổi chuyến bay của mình". Hệ thống phải tìm ra những gì họ muốn, thông tin nào còn thiếu, làm thế nào để có được nó và làm thế nào để hoàn thành hành động. Sau đó, người dùng nói "chờ đã, nếu tôi hủy bỏ thay vào đó thì sao?" và hệ thống phải ghi nhớ ngữ cảnh, chuyển đổi tác vụ và giữ nguyên trạng thái.

Cuộc trò chuyện rất khó đối với một hệ thống ML. Đầu vào là kết thúc mở. Đầu ra phải mạch lạc qua nhiều lượt. Hệ thống có thể cần phải hành động trên thế giới (thay đổi chuyến bay, tính phí thẻ). Mọi bước sai đều hiển thị cho người dùng.

Kiến trúc chatbot đã xoay vòng qua bốn mô hình, mỗi mô hình được giới thiệu vì mô hình trước đó thất bại quá rõ ràng. Bài học này hướng dẫn họ theo thứ tự. Bối cảnh production 2026 là sự kết hợp của hai lần trước.

## Khái niệm

![Chatbot evolution: rule-based → retrieval → neural → agent](../assets/chatbot.svg)

**Dựa trên quy tắc (ELIZA, AIML, DialogFlow).** Các mẫu viết tay khớp với đầu vào của người dùng và tạo phản hồi. Bộ phân loại ý định định tuyến đến các luồng được xác định trước. Máy trạng thái lấp đầy khe thu thập thông tin cần thiết. Hoạt động xuất sắc bên trong phạm vi hẹp mà nó được thiết kế. Thất bại ngay bên ngoài nó. Vẫn ships trong các lĩnh vực quan trọng về an toàn (xác thực ngân hàng, đặt vé máy bay) nơi ảo giác không được dung thứ.

**Dựa trên truy xuất.** Một hệ thống kiểu Câu hỏi thường gặp. Mã hóa mọi cặp (lời nói, phản hồi). Tại runtime, mã hóa tin nhắn của người dùng và truy xuất phản hồi được lưu trữ gần nhất. Hãy nghĩ đến "bài viết tương tự" cổ điển của Zendesk feature. Xử lý diễn giải tốt hơn quy tắc. Không có thế hệ, vì vậy không có ảo giác.

**Neural (seq2seq).** Encoder-decoder được huấn luyện về nhật ký hội thoại. Tạo phản hồi từ đầu. Lưu loát nhưng có xu hướng kết quả chung chung ("Tôi không biết") và trôi dạt thực tế. Không bao giờ đáng tin cậy về chủ đề. Lý do Google, Facebook và Microsoft đều có chatbot đáng thất vọng trong năm 2016-2019.

**LLM agents.** Một ngôn ngữ model bao bọc trong một vòng lặp lập kế hoạch, gọi các công cụ và xác minh kết quả. Không phải là chatbot có prompt dài. Vòng lặp agent: lập kế hoạch → công cụ gọi → quan sát kết quả → quyết định bước tiếp theo. Truy xuất đầu tiên grounding (RAG) giữ cho nó không bị ảo giác. Các lệnh gọi công cụ cho phép nó thực sự làm mọi thứ. Đây là kiến trúc năm 2026.

Bốn mô hình không phải là sự thay thế tuần tự. Chatbot production 2026 định tuyến qua cả bốn: dựa trên quy tắc để xác thực và các hành động phá hoại, truy xuất cho Câu hỏi thường gặp, tạo thần kinh cho cụm từ tự nhiên LLM agent cho các truy vấn mở mơ hồ.

## Tự xây dựng

### Bước 1: khớp mẫu dựa trên quy tắc

```python
import re


class RulePattern:
    def __init__(self, pattern, response_template):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = response_template


PATTERNS = [
    RulePattern(r"my name is (\w+)", "Nice to meet you, {0}."),
    RulePattern(r"i (need|want) (.+)", "Why do you {0} {1}?"),
    RulePattern(r"i feel (.+)", "Why do you feel {0}?"),
    RulePattern(r"(.*)", "Tell me more about that."),
]


def rule_based_respond(user_input):
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."
```

ELIZA trong 20 dòng. Thủ thuật phản ánh ("Tôi cảm thấy buồn" → "Tại sao bạn cảm thấy buồn") là bản demo của nhà trị liệu tâm lý kinh điển từ Weizenbaum 1966. Vẫn mang tính hướng dẫn.

### Bước 2: dựa trên truy xuất (Câu hỏi thường gặp)

Đoạn minh họa này yêu cầu `pip install sentence-transformers` (kéo ngọn đuốc). Thay vào đó, `code/main.py` có thể chạy được cho bài học này sử dụng sự tương đồng của Jaccard stdlib, vì vậy bài học chạy mà không có phần phụ thuộc bên ngoài.

```python
from sentence_transformers import SentenceTransformer
import numpy as np


FAQ = [
    ("how do i reset my password", "Go to Settings > Security > Reset Password."),
    ("how do i cancel my order", "Go to Orders, find the order, click Cancel."),
    ("what is your return policy", "30-day returns on unused items, original packaging."),
]


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)


def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None
    return FAQ[best][1]
```

Từ chối dựa trên ngưỡng là lựa chọn thiết kế chính. Nếu trận đấu tốt nhất không đủ gần, hãy quay lại `None` và để hệ thống leo thang.

### Bước 3: tạo thần kinh (đường cơ sở)

Sử dụng encoder-decoder (FLAN-T5) nhỏ được điều chỉnh theo hướng dẫn hoặc model đàm thoại fine-tuned. Production-không thể sử dụng được vào năm 2026 (mâu thuẫn, trôi lạc đề, vô nghĩa thực tế), nhưng ships bên trong các hệ thống lai để diễn đạt tự nhiên. models chỉ decoder kiểu DialoGPT cần dải phân cách rẽ rõ ràng và xử lý EOS để tạo ra các câu trả lời mạch lạc; một FLAN-T5 text2text pipeline hoạt động vượt trội cho một ví dụ giảng dạy.

```python
from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("Respond politely to: Hi there!", max_new_tokens=40)
print(response[0]["generated_text"])
```

### Bước 4: LLM agent vòng lặp

Hình dạng production năm 2026:

```python
def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"error: unknown tool {tool_name!r}"})
                continue
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"error: arguments must be a dict, got {type(args).__name__}"})
                continue
            fn = tools[tool_name]
            result = fn(**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "I could not complete the task in the step budget."
```

Ba điều để kể tên. Công cụ là các chức năng có thể gọi mà LLM có thể gọi. Vòng lặp kết thúc khi LLM trả về câu trả lời cuối cùng thay vì lệnh gọi công cụ. Ngân sách bước ngăn chặn các vòng lặp vô hạn trên các nhiệm vụ mơ hồ.

Real production bổ sung: grounding truy xuất đầu tiên (chèn tài liệu có liên quan trước mỗi cuộc gọi LLM), guardrails (từ chối các hành động phá hoại mà không xác nhận), observability (ghi nhật ký mọi bước) và đánh giá (kiểm tra tự động rằng hành vi agent vẫn phù hợp với thông số kỹ thuật).

### Bước 5: định tuyến kết hợp

```python
def hybrid_chat(user_input):
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    return agent_loop(user_input, tools, llm)


def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)
```

Mô hình: các quy tắc xác định cho bất kỳ thứ gì phá hoại, truy xuất cho các Câu hỏi thường gặp soạn sẵn LLM agents cho mọi thứ khác. Đây là những gì ships trong hệ thống hỗ trợ khách hàng năm 2026.

## Ứng dụng

stack năm 2026:

| Trường hợp sử dụng | Kiến trúc |
|---------|---------------|
| Đặt chỗ, thanh toán, xác thực | Máy trạng thái dựa trên quy tắc + lấp đầy khe cắm |
| Câu hỏi thường gặp về hỗ trợ khách hàng | Truy xuất qua các câu trả lời được tuyển chọn |
| Trò chuyện trợ giúp mở | LLM agent với RAG + lệnh gọi công cụ |
| Công cụ nội bộ / trợ lý IDE | LLM agent với các lệnh gọi công cụ (tìm kiếm, đọc, ghi) |
| Chatbot đồng hành / nhân vật | Điều chỉnh LLM với tính cách system prompt, truy xuất kiến thức |

Luôn sử dụng định tuyến kết hợp trong production. Không có kiến trúc đơn lẻ nào xử lý tốt mọi yêu cầu. Bản thân lớp định tuyến thường là một bộ phân loại ý định nhỏ.

## Các chế độ lỗi vẫn ship

- **Bịa đặt tự tin.** LLM agent tuyên bố rằng họ đã hoàn thành một hành động mà nó không làm. Giảm thiểu: xác minh kết quả, ghi nhật ký các cuộc gọi công cụ, không bao giờ để LLM tuyên bố đã làm điều gì đó mà không trả lại công cụ thành công.
- **Prompt tiêm.** Người dùng chèn văn bản ghi đè system prompt. Xếp hạng LLM01 trong Top 10 OWASP cho các ứng dụng LLM năm 2025. Hai hương vị: chèn trực tiếp (dán vào cuộc trò chuyện) và chèn gián tiếp (ẩn trong tài liệu, email hoặc đầu ra công cụ mà agent đọc).

Tỷ lệ tấn công thay đổi tùy theo kịch bản. Tỷ lệ thành công đo được nằm trong khoảng ~0,5-8,5% trên khắp các models biên giới về benchmarks sử dụng công cụ và mã hóa nói chung. Các thiết lập rủi ro cao cụ thể (các cuộc tấn công thích ứng chống lại agents mã hóa AI, orchestration dễ bị tổn thương) đã đạt ~84%. Production CVE bao gồm EchoLeak (CVE-2025-32711, CVSS 9.3) — lỗ hổng lấy cắp dữ liệu bằng không cần nhấp chuột trong Microsoft 365 Copilot do email do kẻ tấn công kiểm soát.

Giảm thiểu: coi đầu vào của người dùng là không đáng tin cậy trong suốt vòng lặp; vệ sinh trước khi gọi công cụ; cách ly đầu ra công cụ khỏi prompt chính; sử dụng mẫu Kế hoạch-Xác minh-Thực hiện (PVE) trong đó agent lập kế hoạch trước, sau đó xác minh từng hành động so với kế hoạch đó trước khi thực hiện (điều này ngăn chặn kết quả công cụ đưa vào các hành động mới ngoài kế hoạch); yêu cầu xác nhận của người dùng đối với các hành động phá hoại; áp dụng đặc quyền ít nhất cho phạm vi công cụ.

Không có kỹ thuật prompt nào loại bỏ hoàn toàn rủi ro này. Cần có các lớp bảo vệ runtime bên ngoài (LLM Guard, xác thực danh sách cho phép, phát hiện bất thường ngữ nghĩa).
- **Scope creep.** Agent không thực hiện nhiệm vụ vì một lệnh gọi công cụ trả về thông tin liên quan tiếp tuyến. Giảm thiểu: hợp đồng công cụ hẹp; giữ cho system prompt tập trung; Thêm đánh giá cho tỷ lệ ngoài nhiệm vụ.
- **Vòng lặp vô hạn.** Agent tiếp tục gọi cùng một công cụ. Giảm thiểu: ngân sách bước, loại bỏ trùng lặp cuộc gọi công cụ LLM đánh giá về "chúng ta có tiến bộ không".
- **Context window kiệt sức.** Các cuộc trò chuyện dài đẩy những diễn biến sớm nhất ra khỏi ngữ cảnh. Giảm thiểu: tóm tắt các lượt cũ hơn, truy xuất các lượt trong quá khứ có liên quan theo sự tương đồng hoặc sử dụng model ngữ cảnh dài.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-chatbot-architect.md`:

```markdown
---
name: chatbot-architect
description: Design a chatbot stack for a given use case.
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

Given a product context (user need, compliance constraints, available tools, data volume), output:

1. Architecture. Rule-based, retrieval, neural, LLM agent, or hybrid (specify which paths go where).
2. LLM choice if applicable. Name the model family (Claude, GPT-4, Llama-3.1, Mixtral). Match to tool-use quality and cost.
3. Grounding strategy. RAG sources, retrieval method (see lesson 14), tool contracts.
4. Evaluation plan. Task success rate, tool-call correctness, off-task rate, hallucination rate on held-out dialogs.

Refuse to recommend a pure-LLM agent for any destructive action (payments, account deletion, data modification) without a structured confirmation flow. Refuse to skip the prompt-injection audit if the agent has write access to anything.
```

## Bài tập

1. **Dễ dàng.** Triển khai phản hồi dựa trên quy tắc ở trên với 10 mẫu cho bot đặt hàng tại quán cà phê. Các trường hợp cạnh thử nghiệm: đơn đặt hàng kép, sửa đổi, hủy bỏ, ý định không rõ ràng.
2. **Trung bình.** Xây dựng câu hỏi thường gặp kết hợp + dự phòng LLM. 50 mục Câu hỏi thường gặp soạn sẵn cho một sản phẩm SaaS, LLM dự phòng với việc truy xuất qua trang web tài liệu. Đo lường tỷ lệ từ chối và accuracy 100 câu hỏi hỗ trợ thực tế.
3. **Khó.** Thực hiện vòng lặp agent ở trên với ba công cụ (tìm kiếm, đọc-dữ liệu người dùng, gửi-email). Chạy đánh giá với 50 kịch bản thử nghiệm bao gồm prompt lần tiêm. Báo cáo tỷ lệ không thực hiện nhiệm vụ, tỷ lệ tác vụ không thành công và bất kỳ lần tiêm thành công nào.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Ý định | Những gì người dùng muốn | Nhãn phân loại (book_flight, reset_password). Định tuyến đến trình xử lý. |
| Khe cắm | Một phần thông tin | Parameter bot cần (ngày, đích). Lấp đầy vị trí là chuỗi các yêu cầu. |
| RAG | Truy xuất cộng với tạo | Truy xuất các tài liệu có liên quan, sau đó tiếp tục phản hồi của LLM. |
| Lệnh gọi công cụ | Gọi hàm | LLM phát ra một cuộc gọi có cấu trúc với tên + args. Runtime thực thi, trả về kết quả. |
| Agent vòng lặp | Lập kế hoạch, hành động, xác minh | Bộ điều khiển chạy LLM các cuộc gọi xen kẽ với các lệnh gọi công cụ cho đến khi tác vụ hoàn thành. |
| Tiêm Prompt | Tấn công người dùng prompt | Đầu vào độc hại cố gắng ghi đè system prompt. |

## Đọc thêm

- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — tài liệu chatbot dựa trên quy tắc ban đầu.
- [Thoppilan et al. (2022). LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239) - bài báo chatbot thần kinh muộn của Google, ngay trước khi LLM agents tiếp quản.
- [Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — bài báo đặt tên cho mẫu vòng lặp agent.
- [Anthropic's guide on building effective agents](https://www.anthropic.com/research/building-effective-agents) - 2024 production hướng dẫn vẫn giữ nguyên vào năm 2026.
- [Greshake et al. (2023). Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) - giấy tiêm prompt.
- [OWASP Top 10 for LLM Applications 2025 — LLM01 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) - bảng xếp hạng khiến prompt injection trở thành mối quan tâm hàng đầu về bảo mật.
- [AWS — Securing Amazon Bedrock Agents against Indirect Prompt Injections](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/) — các biện pháp phòng thủ orchestration lớp thực tế bao gồm Kế hoạch-Xác minh-Thực hiện và quy trình xác nhận người dùng.
- [EchoLeak (CVE-2025-32711)](https://www.vectra.ai/topics/prompt-injection) — CVE lấy cắp dữ liệu bằng không nhấp chuột chính tắc từ chèn prompt gián tiếp. Trường hợp tham khảo về lý do tại sao agents truy cập ghi cần runtime phòng thủ.
