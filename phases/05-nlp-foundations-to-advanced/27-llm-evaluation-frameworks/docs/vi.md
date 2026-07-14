# Đánh giá LLM - RAGAS, DeepEval, G-Eval

> Khớp chính xác và F1 bỏ lỡ sự tương đương ngữ nghĩa. Đánh giá của con người không mở rộng. LLM với tư cách là thẩm phán là câu trả lời production - với đủ hiệu chuẩn để tin tưởng vào con số.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 5 · 13 (Trả lời câu hỏi), Giai đoạn 5 · 14 (Truy xuất thông tin)
**Thời lượng:** ~75 phút

## Vấn đề

Hệ thống RAG của bạn trả lời: "Ngày 29 tháng 6 năm 2007."
Tham chiếu bằng vàng là: "Ngày 29 tháng 6 năm 2007."
Điểm trận đấu chính xác 0. F1 điểm ~75%. Một con người sẽ đạt điểm 100%.

Bây giờ nhân với 10.000 trường hợp thử nghiệm. Nhân lại với mỗi thay đổi đối với chó tha mồi, chunking, prompt hoặc model. Bạn cần một người đánh giá hiểu ý nghĩa, chạy với giá rẻ trên quy mô lớn, không nói dối về hồi quy và hiển thị các chế độ lỗi phù hợp.

Năm 2026 có ba frameworks sở hữu vấn đề này.

- **RAGAS.** Truy xuất-Tăng cường Thế hệ ASsessment. Bốn chỉ số RAG (độ trung thực, mức độ liên quan của câu trả lời, precision ngữ cảnh, recall ngữ cảnh) với phần phụ trợ NLI + LLM đánh giá. Được hỗ trợ bởi nghiên cứu, nhẹ.
- **DeepEval.** Pytest cho LLMs. G-Eval, hoàn thành nhiệm vụ, ảo giác bias các chỉ số. CI/CD-native.
- **G-Eval.** Một phương pháp (và chỉ số DeepEval): LLM-as-judge với tiêu chí chain-of-thought, tùy chỉnh, điểm 0-1.

Cả ba đều dựa vào LLM với tư cách là thẩm phán. Bài học này xây dựng trực giác cho phương pháp và lớp tin cậy xung quanh nó.

## Khái niệm

![Four evaluation dimensions, LLM-as-judge architecture](../assets/llm-evaluation.svg)

**LLM với tư cách là người đánh giá.** Thay thế chỉ số tĩnh bằng LLM chấm điểm đầu ra cho một tiêu chí. Đưa ra `(query, context, answer)`, prompt một giám khảo LLM: "Điểm 0-1 về sự trung thành." Trả lại điểm.

Tại sao nó hoạt động: LLMs phán đoán gần đúng của con người với một phần nhỏ chi phí. GPT-4o-mini ở ~ $0.003 per scored case enables 1000-sample regression eval runs for under $5.

Tại sao nó thất bại một cách âm thầm:

1. **Giám khảo bias.** Giám khảo thích câu trả lời dài hơn, câu trả lời từ gia đình model của họ, câu trả lời phù hợp với phong cách prompt.
2. **JSON phân tích cú pháp thất bại.** Điểm Bad JSON → NaN → âm thầm loại trừ khỏi tổng hợp. Người dùng RAGAS biết nỗi đau này. Cổng với try/except + chế độ lỗi rõ ràng.
3. **Trôi qua model phiên bản.** Nâng cấp giám khảo thay đổi mọi chỉ số. Đóng băng phiên bản model + của thẩm phán.

**Bộ RAG bốn.

| Số liệu | Câu hỏi | Phần phụ trợ |
|--------|----------|---------|
| Trung thành | Mỗi tuyên bố trong câu trả lời có đến từ ngữ cảnh được truy xuất không? | Bao gồm dựa trên NLI |
| Mức độ liên quan của câu trả lời | Câu trả lời có giải quyết được câu hỏi không? | Tạo các câu hỏi giả định từ câu trả lời; So sánh với câu hỏi thực tế |
| Bối cảnh precision | Trong số các phần được truy xuất, phân số nào có liên quan? | LLM thẩm phán |
| Bối cảnh recall | Truy xuất có trả lại mọi thứ cần thiết không? | LLM-Judge chống lại câu trả lời vàng |

**G-Eval.** Xác định tiêu chí tùy chỉnh: "Câu trả lời có trích dẫn đúng nguồn không?" framework tự động mở rộng thành chain-of-thought bước đánh giá, sau đó điểm 0-1. Tốt cho các kích thước chất lượng theo miền cụ thể mà RAGAS không đề cập.

**Hiệu chuẩn.** Đừng bao giờ tin tưởng vào điểm số của giám khảo thô cho đến khi bạn có mối tương quan với nhãn của con người. Chạy 100 ví dụ được gắn nhãn tay. Thẩm phán cốt truyện vs con người. Tính toán Spearman rho. Nếu rho < 0,7, bảng đánh giá giám khảo của bạn cần được cải thiện.

## Tự xây dựng

### Bước 1: trung thành với NLI (kiểu RAGAS)

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` is any callable: prompt str -> generated str.
# Example: llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

Phân hủy câu trả lời thành các tuyên bố nguyên tử. NLI-kiểm tra từng xác nhận quyền sở hữu dựa trên ngữ cảnh được truy xuất. Faithfulness = phân số được hỗ trợ.

### Bước 2: trả lời mức độ liên quan

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: any model implementing .encode(texts, normalize_embeddings=True) -> ndarray
# e.g., encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

Nếu câu trả lời ngụ ý các câu hỏi khác với câu hỏi được hỏi, mức độ liên quan sẽ giảm xuống.

### Bước 3: Chỉ số tùy chỉnh G-Eval

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

Các bước đánh giá là bảng đánh giá. Các bước rõ ràng ổn định hơn so với prompts "điểm 0-1" ngầm.

### Bước 4: CI cổng

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

Ship dưới dạng tệp pytest. Chạy trên mọi PR. Chặn merges hồi quy.

### Bước 5: đánh giá đồ chơi từ đầu

Xem `code/main.py`. Các xấp xỉ chỉ có Stdlib về độ trung thực (chồng chéo các tuyên bố câu trả lời với ngữ cảnh) và mức độ liên quan (chồng chéo câu trả lời tokens với câu hỏi tokens). Không production. Hiển thị hình dạng.

## Cạm bẫy

- **Không hiệu chuẩn.** Một thẩm phán có mối tương quan 0,3 với nhãn của con người là nhiễu. Yêu cầu chạy hiệu chuẩn trước khi shipping.
- **Tự đánh giá.** Sử dụng cùng một LLM để tạo và đánh giá sẽ làm tăng điểm số lên 10-20%. Sử dụng một gia đình model khác cho thẩm phán.
- **bias vị trí trong đánh giá theo cặp.** Các giám khảo thích tùy chọn đầu tiên được trình bày. Luôn ngẫu nhiên hóa thứ tự và chạy cả hai.
- **Tổng hợp thô che giấu thất bại.** Điểm trung bình 0,85 thường che giấu 5% lỗi thảm khốc. Luôn kiểm tra lượng tử dưới cùng.
- **Golden dataset thối.** Các bộ đánh giá không có phiên bản trôi dạt theo thời gian phá vỡ so sánh theo chiều dọc. Gắn thẻ dataset với mọi thay đổi.
- **Chi phí LLM.** Ở quy mô lớn, các cuộc gọi của thẩm phán chi phối chi phí. Sử dụng model rẻ nhất đáp ứng ngưỡng hiệu chuẩn. GPT-4o-mini, Claude Haiku, Mistral-nhỏ.

## Ứng dụng

stack năm 2026:

| Trường hợp sử dụng | Framework |
|---------|-----------|
| Giám sát chất lượng RAG | RAGAS (4 chỉ số) |
| CI/CD cổng hồi quy | DeepEval + pytest |
| Tiêu chí miền tùy chỉnh | G-Eval trong DeepEval |
| Giám sát lưu lượng truy cập trực tuyến | RAGAS với chế độ không tham chiếu |
| Kiểm tra tại chỗ của con người | LangSmith hoặc Phoenix với giao diện người dùng chú thích |
| Đánh giá đội đỏ / an toàn | Nhắc nhở + DeepEval |

stack điển hình: RAGAS để giám sát, DeepEval cho CI, G-Eval cho các kích thước mới. Chạy cả ba; họ không đồng ý một cách hữu ích.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-eval-architect.md`:

```markdown
---
name: eval-architect
description: Design an LLM evaluation plan with calibrated judge and CI gates.
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

Given a use case (RAG / agent / generative task), output:

1. Metrics. Faithfulness / relevance / context-precision / context-recall + any custom G-Eval metrics with criteria.
2. Judge model. Named model + version, rationale for cost vs accuracy.
3. Calibration. Hand-labeled set size, target Spearman rho vs human > 0.7.
4. Dataset versioning. Tag strategy, change log, stratification.
5. CI gate. Thresholds per metric, regression-window logic, bottom-quantile alert.

Refuse to rely on a judge untested against ≥50 human-labeled examples. Refuse self-evaluation (same model generates + judges). Refuse aggregate-only reporting without bottom-10% surfacing. Flag any pipeline where judge upgrade lands without parallel baseline eval.
```

## Bài tập

1. **Dễ dàng.** Sử dụng RAGAS trên 10 RAG ví dụ có ảo giác đã biết. Xác minh chỉ số độ trung thực nắm bắt từng cái một.
2. **Trung bình.** Nhãn cầm tay 50 QA trả lời 0-1 cho tính chính xác. Ghi điểm với G-Eval. Đo lường Spearman rho giữa thẩm phán và con người.
3. **Khó.** Xây dựng một cổng CI pytest với DeepEval. Cố tình thoái lui tha mồi. Xác minh cổng không thành công. Thêm cảnh báo lượng tử đáy thông qua kiểm tra ngưỡng trên 10% thấp nhất.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| LLM với tư cách là thẩm phán | Ghi bàn bằng LLM | Prompt giám khảo model chấm điểm đầu ra 0-1 với một bảng đánh giá. |
| RAGAS | Thư viện chỉ số RAG | framework đánh giá mã nguồn mở với 4 chỉ số RAG không cần tham chiếu. |
| Trung thành | Câu trả lời có căn cứ không? | Phần yêu cầu câu trả lời được truy xuất bởi ngữ cảnh được truy xuất. |
| Bối cảnh precision | Các đoạn được truy xuất có liên quan không? | Phần nhỏ của top-K phần thực sự quan trọng. |
| Bối cảnh recall | Truy xuất có tìm thấy mọi thứ không? | Phần nhỏ của các tuyên bố trả lời vàng được hỗ trợ bởi các khối được truy xuất. |
| Đánh giá G | Thẩm phán LLM tùy chỉnh | Bảng đánh giá + chain-of-thought bước đánh giá + 0-1 điểm. |
| Hiệu chuẩn | Tin cậy nhưng xác minh | Mối tương quan của Spearman giữa điểm số của giám khảo và điểm số của con người. |

## Đọc thêm

- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) - bài báo RAGAS.
- [Liu et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) - bài báo G-Eval.
- [DeepEval docs](https://deepeval.com/docs/metrics-introduction) - mở production stack.
- [Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) - thành kiến, hiệu chuẩn, giới hạn.
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) - framework thống nhất tích hợp RAGAS, DeepEval, Phoenix.
