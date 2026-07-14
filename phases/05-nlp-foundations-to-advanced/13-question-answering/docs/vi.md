# Hệ thống trả lời câu hỏi

> Ba hệ thống định hình QA hiện đại. Chiết xuất được tìm thấy spans. Tăng cường truy xuất đã đặt chúng vào các tài liệu. Generative tạo ra câu trả lời. Mỗi trợ lý AI hiện đại là sự kết hợp của cả ba.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 5 · 11 (Dịch máy), Giai đoạn 5 · 10 (Cơ chế Attention)
**Thời lượng:** ~75 phút

## Vấn đề

Một người dùng gõ "iPhone đầu tiên ra mắt khi nào?" và mong đợi "29 tháng 6 năm 2007". Không phải "Lịch sử của Apple rất dài và đa dạng". Không phải "2007" ngồi cô lập mà không có câu. Một câu trả lời trực tiếp, có cơ sở, chính xác.

Ba kiến trúc đã thống trị QA trong thập kỷ qua.

- **QA trích xuất.** Đưa ra một câu hỏi và một đoạn văn được biết là chứa câu trả lời, hãy tìm các chỉ số bắt đầu và kết thúc của câu trả lời span trong đoạn văn. SQuAD là benchmark chuẩn.
- **QA miền mở.** Đoạn văn không được đưa ra. Truy xuất đoạn văn có liên quan trước, sau đó trích xuất hoặc tạo câu trả lời. Đây là nền tảng của mọi RAG pipeline ngày nay.
- **Generative / Closed-book QA.** Một ngôn ngữ lớn model câu trả lời từ bộ nhớ tham số của nó. Không truy xuất. Nhanh nhất ở inference, ít đáng tin cậy nhất về sự thật.

Xu hướng vào năm 2026 là kết hợp: lấy một vài đoạn văn tốt nhất, sau đó prompt một model tổng quát để trả lời dựa trên những đoạn văn đó. Đó là RAG, và bài 14 bao gồm một nửa việc truy xuất một cách sâu sắc. Bài học này xây dựng một nửa QA.

## Khái niệm

![QA architectures: extractive, retrieval-augmented, generative](../assets/qa.svg)

**Trích xuất.** Mã hóa câu hỏi và đoạn văn cùng với một transformer (họ BERT). Huấn luyện hai đầu dự đoán bắt đầu và kết thúc token chỉ số của câu trả lời. Loss là entropy chéo trên các vị trí hợp lệ. Đầu ra là một span từ đoạn văn. Không bao giờ ảo giác (bằng cách xây dựng), không bao giờ xử lý các câu hỏi mà đoạn văn không thể trả lời (bằng cách xây dựng).

**Tăng cường truy xuất (RAG).** Hai giai đoạn. Đầu tiên, một tha mồi tìm thấy các đoạn `k` trên cùng từ một kho dữ liệu. Thứ hai, một người đọc (trích xuất hoặc tổng hợp) tạo ra câu trả lời bằng cách sử dụng các đoạn văn đó. Phân chia retriever-reader cho phép mỗi người được huấn luyện và đánh giá độc lập. RAG hiện đại thường thêm một thứ hạng lại giữa chúng.

**Generative.** Câu trả lời chỉ decoder LLM (GPT, Claude, Llama) từ trọng số đã học. Không có bước truy xuất. Xuất sắc về kiến thức chung, thảm họa về những sự kiện hiếm hoi hoặc gần đây. Tỷ lệ ảo giác tương quan nghịch với tần suất thực tế trong dữ liệu pretraining.

## Tự xây dựng

### Bước 1: khai thác QA với pretrained model

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

`deepset/roberta-base-squad2` được huấn luyện về SQuAD 2.0, bao gồm các câu hỏi không thể trả lời. Theo mặc định, `question-answering` pipeline trả về span điểm cao nhất ngay cả khi điểm rỗng của model thắng - nó *không* tự động trả về câu trả lời trống. Để có được hành vi "không có câu trả lời" rõ ràng, hãy chuyển `handle_impossible_answer=True` cho cuộc gọi pipeline: pipeline sau đó chỉ trả về câu trả lời trống khi điểm rỗng vượt quá mỗi điểm span. Luôn kiểm tra trường `score` theo một trong hai cách.

### Bước 2: pipeline tăng cường truy xuất (phác thảo)

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

pipeline hai giai đoạn. Dense retriever (Sentence-BERT) tìm các đoạn văn có liên quan bằng sự tương đồng về ngữ nghĩa. Trình đọc trích xuất (RoBERTa-SQuAD) lấy span câu trả lời từ các đoạn trên cùng được kết hợp. Hoạt động trên kho dữ liệu nhỏ. Đối với kho dữ liệu hàng triệu tài liệu, hãy sử dụng FAISS hoặc cơ sở dữ liệu vector.

### Bước 3: tạo ra với RAG

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

Mô hình prompt quan trọng. Nói rõ ràng với model về nền tảng trong ngữ cảnh và trả về "Tôi không biết" khi ngữ cảnh không đủ sẽ giảm tỷ lệ ảo giác xuống 40-60% so với prompting ngây thơ. Các mẫu phức tạp hơn thêm trích dẫn, điểm tin cậy và trích xuất có cấu trúc.

### Bước 4: đánh giá phản ánh thế giới thực

SQuAD sử dụng **Exact Match (EM)** và **token-level F1**. EM là một kết quả khớp nghiêm ngặt sau khi chuẩn hóa (chữ thường, dấu câu dải, xóa bài viết) - dự đoán khớp chính xác hoặc điểm 0. F1 được tính toán trên token trùng lặp giữa dự đoán và tham chiếu và cung cấp tín dụng một phần. Cả hai diễn giải dưới tín dụng: "29 tháng 6 năm 2007" và "29 tháng 6 năm 2007" thường nhận được 0 EM (chuẩn hóa ngắt thứ tự) nhưng vẫn kiếm được F1 đáng kể từ tokens chồng chéo.

Đối với production QA:

- **Câu trả lời accuracy** (đánh giá LLM hoặc con người, vì số liệu không nắm bắt được sự tương đương về ngữ nghĩa).
- **Trích dẫn accuracy.** Đoạn trích dẫn có thực sự hỗ trợ câu trả lời không? Tầm thường để kiểm tra tự động với sự khớp chuỗi giữa các trích dẫn được tạo và các đoạn văn được truy xuất.
- **Hiệu chỉnh từ chối.** Khi câu trả lời không có trong các đoạn văn được truy xuất, hệ thống có nói chính xác "Tôi không biết" không? Đo lường tỷ lệ tin cậy sai.
- **Truy xuất recall.** Trước khi đánh giá người đọc, hãy đo lường xem chó săn có đúng đoạn vào `k` trên cùng hay không. Người đọc không thể sửa một đoạn văn bị thiếu.

### RAGAS: framework đánh giá production năm 2026

`RAGAS` được xây dựng có mục đích cho các hệ thống RAG và là shipping mặc định vào năm 2026. Nó ghi điểm bốn chiều mà không yêu cầu tham chiếu vàng:

- **Trung thành.** Mỗi tuyên bố trong câu trả lời có đến từ ngữ cảnh được truy xuất không? Được đo lường bằng sự bao gồm dựa trên NLI. Chỉ số ảo giác chính của bạn.
- **Mức độ liên quan của câu trả lời.** Câu trả lời có giải quyết được câu hỏi không? Được đo lường bằng cách tạo ra các câu hỏi giả định từ câu trả lời và so sánh với câu hỏi thực.
- **Ngữ cảnh precision.** Trong số các phần được truy xuất, phần nào thực sự có liên quan? precision thấp = nhiễu tính bằng prompt.
- **Ngữ cảnh recall.** Tập hợp được truy xuất có chứa tất cả thông tin cần thiết không? recall thấp = người đọc không thể thành công.

Tính năng chấm điểm không cần tham khảo cho phép bạn đánh giá lưu lượng truy cập production trực tiếp mà không cần câu trả lời vàng được tuyển chọn. Xếp lớp LLM làm giám khảo lên trên cho các câu hỏi mở trong đó các chỉ số đối sánh chính xác là vô dụng.

`pip install ragas`. Cắm bộ truy xuất + đầu đọc của bạn. Nhận bốn vô hướng cho mỗi truy vấn. Cảnh báo về hồi quy.

## Ứng dụng

Cuộc stack năm 2026.

| Trường hợp sử dụng | Đề xuất |
|---------|-------------|
| Đoạn văn đã cho, hãy tìm câu trả lời span | `deepset/roberta-base-squad2` |
| Trên một kho dữ liệu cố định, sổ đóng không được chấp nhận | RAG: Thickse Retriever + LLM Reader |
| Thời gian thực qua kho tài liệu | RAG với chó săn lai (BM25 + dày đặc) + reranker (bài 14) |
| QA đàm thoại (câu hỏi tiếp theo) | LLM với lịch sử cuộc trò chuyện + RAG trên mỗi lượt |
| Các miền có tính thực tế cao, được quản lý | Trích xuất trên một kho dữ liệu có thẩm quyền; Không bao giờ tạo ra một mình |

Extractive QA không còn hợp thời vào năm 2026 vì RAG với LLMs xử lý nhiều trường hợp hơn. Nó vẫn ships trong bối cảnh yêu cầu trích dẫn theo nghĩa đen: nghiên cứu pháp lý, tuân thủ quy định, công cụ kiểm toán.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-qa-architect.md`:

```markdown
---
name: qa-architect
description: Choose QA architecture, retrieval strategy, and evaluation plan.
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

Given requirements (corpus size, question type, factuality constraint, latency budget), output:

1. Architecture. Extractive, RAG with extractive reader, RAG with generative reader, or closed-book LLM. One-sentence reason.
2. Retriever. None, BM25, dense (name the encoder), or hybrid.
3. Reader. SQuAD-tuned model, LLM by name, or "domain-fine-tuned DistilBERT."
4. Evaluation. EM + F1 for extractive benchmarks; answer accuracy + citation accuracy + refusal calibration for production. Name what you are measuring and how you are measuring it.

Refuse closed-book LLM answers for regulatory or compliance-sensitive questions. Refuse any QA system without a retrieval-recall baseline (you cannot evaluate the reader without knowing the retriever surfaced the right passage). Flag questions that require multi-hop reasoning as needing specialized multi-hop retrievers like HotpotQA-trained systems.
```

## Bài tập

1. **Dễ dàng.** Thiết lập pipeline trích xuất SQuAD ở trên trên 10 đoạn văn Wikipedia. Thủ công 10 câu hỏi. Đo lường tần suất câu trả lời đúng. Bạn sẽ thấy 7-9 đúng nếu đoạn văn và câu hỏi sạch sẽ.
2. **Trung bình.** Thêm công cụ phân loại từ chối. Khi điểm truy xuất cao nhất dưới ngưỡng (giả sử 0,3 cosin), hãy trả về "Tôi không biết" thay vì gọi cho người đọc. Điều chỉnh ngưỡng trên một nhóm bị giữ.
3. **Khó.** Xây dựng một RAG pipeline trên kho dữ liệu 10.000 tài liệu mà bạn chọn. Thực hiện truy xuất lai (BM25 + đậm đặc) với hợp nhất RRF (xem bài 14). Đo lường câu trả lời accuracy có và không có bước kết hợp. Ghi lại loại câu hỏi nào được hưởng lợi nhiều nhất.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| QA khai thác | Tìm câu trả lời span | Dự đoán các chỉ số bắt đầu và kết thúc của câu trả lời trong một đoạn văn nhất định. |
| QA miền mở | QA trên một kho dữ liệu | Không có đoạn văn nhất định; phải truy xuất sau đó trả lời. |
| RAG | Truy xuất sau đó tạo | Thế hệ tăng cường truy xuất. Retriever + người đọc pipeline. |
| SQuAD | benchmark chuẩn | Trả lời câu hỏi của Stanford Dataset. Chỉ số EM + F1. |
| Ảo giác | Câu trả lời bịa đặt | Đầu ra của đầu đọc không được hỗ trợ bởi ngữ cảnh được truy xuất. |
| Hiệu chuẩn từ chối | Biết khi nào nên im lặng | Hệ thống nói chính xác "Tôi không biết" khi không thể trả lời. |

## Đọc thêm

- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) - tờ báo benchmark.
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) - DPR, chó săn mồi dày đặc kinh điển cho QA.
- [Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) - tờ báo đặt tên RAG.
- [Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) - khảo sát RAG toàn diện.
