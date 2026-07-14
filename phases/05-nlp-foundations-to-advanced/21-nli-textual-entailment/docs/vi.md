# Ngôn ngữ tự nhiên Inference - Textual Entailment

> "T đòi hỏi h" có nghĩa là cách đọc của con người t sẽ kết luận h là đúng. NLI là nhiệm vụ dự đoán sự lôi kéo / mâu thuẫn / trung lập. Nhàm chán trên bề mặt, chịu lực production.

**Loại:** Học
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 5 · 05 (Phân tích tâm lý), Giai đoạn 5 · 13 (Trả lời câu hỏi)
**Thời lượng:** ~60 phút

## Vấn đề

Bạn đã xây dựng một trình tóm tắt. Nó đã tạo ra một bản tóm tắt. Làm thế nào để bạn biết bản tóm tắt không chứa ảo giác?

Bạn đã xây dựng một chatbot. Nó trả lời "có." Làm thế nào để bạn biết câu trả lời được hỗ trợ bởi đoạn văn được truy xuất?

Bạn cần phân loại 10.000 bài báo theo chủ đề. Bạn không có nhãn training. Bạn có thể tái sử dụng model không?

Cả ba vấn đề đều giảm xuống Inference ngôn ngữ tự nhiên. NLI đặt câu hỏi: với một tiền đề `t` và một giả thuyết `h`, `h` được bao gồm bởi `t`, mâu thuẫn hay trung lập (không liên quan)?

- **Kiểm tra ảo giác: **`t` = tài liệu nguồn, `h` = yêu cầu tóm tắt. Không bao gồm = ảo giác.
- **QA có cơ sở:** `t` = đoạn văn được truy xuất, `h` = câu trả lời được tạo. Không bao gồm = bịa đặt.
- **Zero-shot phân loại:** `t` = tài liệu, `h` = nhãn bằng lời nói ("Đây là về thể thao"). Entailment = nhãn dự đoán.

Một nhiệm vụ, ba production sử dụng. Đây là lý do tại sao mọi đánh giá RAG đều framework ships một model NLI.

## Khái niệm

![NLI: three-way classification, premise vs hypothesis](../assets/nli.svg)

**Ba nhãn.

- **Đòi hỏi.** `t` → `h`. "Con mèo đang ở trên thảm" đòi hỏi "Có một con mèo."
- **Mâu thuẫn.** `t` → ¬`h`. "Con mèo đang ở trên thảm" mâu thuẫn với "Không có mèo."
- **Trung lập.** Không inference theo cách nào. "Con mèo đang ở trên thảm" là trung lập với "Con mèo đang đói".

**Không phải là sự ràng buộc logic.** NLI là ngôn ngữ * tự nhiên * inference - điều mà một độc giả điển hình sẽ suy luận, không phải logic chặt chẽ. "John dắt chó đi dạo" đòi hỏi "John có một" trong NLI, nhưng logic bậc nhất nghiêm ngặt sẽ chỉ thừa nhận điều đó nếu bạn tiên đề hóa việc sở hữu.

**Datasets.**

- **SNLI** (2015). 570k cặp chú thích con người, chú thích hình ảnh làm tiền đề. Miền hẹp.
- **MultiNLI** (2017). 433k cặp trên 10 thể loại. Tiêu chuẩn training kho dữ liệu vào năm 2026.
- **ANLI** (2019). NLI đối nghịch. Con người đã viết các ví dụ được thiết kế đặc biệt để phá vỡ models hiện có. Khó hơn.
- **DocNLI, ConTRoL** (2020–21). Mặt bằng có độ dài tài liệu. Kiểm tra inference nhiều bước và tầm xa.

**Kiến trúc.** A transformer encoder (BERT, RoBERTa, DeBERTa) đọc `[CLS] premise [SEP] hypothesis [SEP]`. Biểu diễn `[CLS]` cung cấp softmax 3 chiều. Huấn luyện về MNLI, đánh giá trên benchmarks được giữ lại, nhận được 90% + accuracy trên các cặp trong phân phối.

**Zero-shot qua NLI.** Được cung cấp một tài liệu và nhãn ứng cử viên, hãy biến mỗi nhãn thành một giả thuyết ("Văn bản này nói về thể thao"). Tính xác suất bao gồm cho mỗi loại. Chọn mức tối đa. Đây là cơ chế đằng sau `zero-shot-classification` pipeline của Hugging Face.

## Tự xây dựng

### Bước 1: chạy model pretrained NLI

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # return all labels; replaces deprecated return_all_scores=True

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

Đối với production NLI, `facebook/bart-large-mnli` và `microsoft/deberta-v3-large-mnli` là mặc định mở. DeBERTa-v3 đứng đầu bảng xếp hạng.

### Bước 2: Phân loại zero-shot

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

Bản mẫu là "Ví dụ này là về {label}." theo mặc định. Tùy chỉnh với `hypothesis_template`. Không cần dữ liệu training. Không fine-tuning. Hoạt động ra khỏi hộp.

### Bước 3: Kiểm tra độ trung thực của RAG

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

Đây là cốt lõi của sự trung thành của RAGAS. Chia câu trả lời được tạo thành các tuyên bố nguyên tử. Kiểm tra từng thông báo xác nhận quyền sở hữu dựa trên ngữ cảnh đã truy xuất. Báo cáo phân số đòi hỏi.

### Bước 4: Bộ phân loại NLI cuộn tay (khái niệm)

Xem `code/main.py` để biết đồ chơi chỉ có stdlib: tiền đề và giả thuyết được so sánh thông qua sự chồng chéo từ vựng + phát hiện phủ định. Không cạnh tranh với transformer models - nhưng nó cho thấy hình dạng của nhiệm vụ: hai văn bản vào, nhãn 3 chiều ra, loss = entropy chéo trên `{entail, contradict, neutral}`.

## Cạm bẫy

- **Lối tắt chỉ dựa trên giả thuyết.** Models có thể dự đoán nhãn chỉ từ giả thuyết ở mức ~60% trên SNLI vì "không", "không ai", "không bao giờ" tương quan với mâu thuẫn. Đường cơ sở vững chắc để phát hiện rò rỉ nhãn.
- **Heuristic chồng chéo từ vựng.** Heuristic dãy con ("mọi dãy con đều được bao gồm") vượt qua SNLI nhưng không thành công HANS/ANLI. Sử dụng benchmarks đối nghịch.
- **Giảm độ dài tài liệu.** NLI một câu models giảm 20+ F1 trên cơ sở có độ dài tài liệu. Sử dụng models được huấn luyện DocNLI cho ngữ cảnh dài.
- **Zero-shot độ nhạy của bản mẫu.** "Ví dụ này là về {label}" so với "{label}" so với "Chủ đề là {label}" có thể dao động accuracy 10+ điểm. Điều chỉnh mẫu.
- **Tên miền không khớp.** MNLI huấn luyện tiếng Anh tổng quát. Văn bản pháp lý, y tế và khoa học cần models NLI theo lĩnh vực cụ thể (ví dụ: SciNLI, MedNLI).

## Ứng dụng

stack năm 2026:

| Trường hợp sử dụng | Model |
|---------|-------|
| NLI đa năng | `microsoft/deberta-v3-large-mnli` |
| Nhanh / cạnh | `cross-encoder/nli-deberta-v3-base` |
| Phân loại Zero-shot (nhẹ) | `facebook/bart-large-mnli` |
| NLI cấp tài liệu | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| Đa ngôn ngữ | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| Phát hiện ảo giác trong RAG | Lớp NLI bên trong RAGAS / DeepEval |

Mô hình meta năm 2026: NLI là băng keo của việc hiểu văn bản. Bất cứ khi nào bạn cần "A có hỗ trợ B không?" hoặc "A có mâu thuẫn với B không?" - hãy tìm đến NLI trước khi bạn tiếp cận một cuộc gọi LLM khác.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-nli-picker.md`:

```markdown
---
name: nli-picker
description: Pick an NLI model, label template, and evaluation setup for a classification / faithfulness / zero-shot task.
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

Given a use case (faithfulness check, zero-shot classification, document-level inference), output:

1. Model. Named NLI checkpoint. Reason tied to domain, length, language.
2. Template (if zero-shot). Verbalization pattern. Example.
3. Threshold. Entailment cutoff for the decision rule. Reason based on calibration.
4. Evaluation. Accuracy on held-out labeled set, hypothesis-only baseline, adversarial subset.

Refuse to ship zero-shot classification without a 100-example labeled sanity check. Refuse to use a sentence-level NLI model on document-length premises. Flag any claim that NLI solves hallucination — it reduces it; it does not eliminate it.
```

## Bài tập

1. **Dễ dàng.** Chạy `facebook/bart-large-mnli` trên 20 bộ ba thủ công (tiền đề, giả thuyết, nhãn) bao gồm cả ba classes. Đo lường accuracy. Thêm bẫy "heuristic trình tự phụ" đối nghịch ("Tôi không ăn bánh" so với "Tôi đã ăn bánh") và xem nó có bị vỡ không.
2. **Trung bình.** So sánh `"This text is about {label}"` mẫu zero-shot với `"The topic is {label}"` và `"{label}"` trên 100 tiêu đề AG News. Báo cáo accuracy sự thay đổi.
3. **Khó.** Xây dựng một công cụ kiểm tra độ trung thực RAG: phân hủy tuyên bố nguyên tử + NLI cho mỗi tuyên bố. Đánh giá trên 50 câu trả lời do RAG tạo với ngữ cảnh vàng. Đo lường tỷ lệ dương tính giả và âm tính giả so với nhãn tay.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| NLI | Ngôn ngữ tự nhiên Inference | Phân loại 3 chiều của mối quan hệ tiền đề-giả thuyết. |
| RTE | Nhận biết sự liên quan của văn bản | Tên cũ hơn của NLI; cùng một nhiệm vụ. |
| Kéo theo | "t ngụ ý h" | Một độc giả điển hình sẽ kết luận h là đúng với t. |
| Mâu thuẫn | "t loại trừ h" | Một độc giả điển hình sẽ kết luận h là sai cho t. |
| Trung lập | "Chưa quyết định" | Không inference từ t đến h theo cách nào. |
| Phân loại Zero-shot | NLI làm bộ phân loại | Diễn đạt các nhãn như giả thuyết, chọn hàm ý tối đa. |
| Trung thành | Câu trả lời có được hỗ trợ không? | NLI over (truy xuất ngữ cảnh, câu trả lời được tạo). |

## Đọc thêm

- [Bowman et al. (2015). A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326) - SNLI.
- [Williams, Nangia, Bowman (2017). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426) - MultiNLI.
- [Nie et al. (2019). Adversarial NLI](https://arxiv.org/abs/1910.14599) - benchmark ANLI.
- [Yin, Hay, Roth (2019). Benchmarking Zero-shot Text Classification](https://arxiv.org/abs/1909.00161) - NLI như bộ phân loại.
- [He et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) - con ngựa làm việc của NLI năm 2026.
