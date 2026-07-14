# Tóm tắt văn bản

> Hệ thống trích xuất cho bạn biết tài liệu nói gì. Hệ thống trừu tượng cho bạn biết ý của tác giả. Nhiệm vụ khác nhau, cạm bẫy khác nhau.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 5 · 02 (BoW + TF-IDF), Giai đoạn 5 · 11 (Dịch máy)
**Thời lượng:** ~75 phút

## Vấn đề

Một bài viết tin tức dài 2.000 từ sẽ xuất hiện trong nguồn cấp dữ liệu của bạn. Bạn cần 120 từ để nắm bắt nó. Bạn có thể chọn ba câu quan trọng nhất từ bài viết (trích xuất) hoặc viết lại nội dung bằng lời của riêng bạn (trừu tượng). Cả hai đều được gọi là tóm tắt. Chúng là những vấn đề hoàn toàn khác nhau.

Tóm tắt trích xuất là một vấn đề xếp hạng. Chấm điểm từng câu, trả về `k` trên cùng. Đầu ra luôn là ngữ pháp vì nó được nâng lên nguyên văn. Rủi ro là thiếu nội dung được phân phối trên bài viết.

Tóm tắt trừu tượng là một vấn đề thế hệ. Một transformer tạo ra văn bản mới có điều kiện trên đầu vào. Đầu ra trôi chảy và nén nhưng có thể gây ảo giác cho các sự kiện không có trong nguồn. Rủi ro là bịa đặt tự tin.

Bài học này xây dựng cả hai, với chế độ thất bại mà mỗi người sở hữu.

## Khái niệm

![Extractive TextRank vs abstractive transformer](../assets/summarization.svg)

**Trích xuất.** Coi bài viết như một biểu đồ trong đó các nút là câu và các cạnh là những điểm tương đồng. Chạy PageRank (hoặc một cái gì đó tương tự) trên biểu đồ để chấm điểm các câu theo mức độ kết nối của chúng với mọi thứ khác. Những câu có điểm cao nhất là phần tóm tắt. Việc triển khai chuẩn là **TextRank** (Mihalcea và Tarau, 2004).

**Trừu tượng.** Fine-tune transformer encoder-decoder (BART, T5, Pegasus) trên các cặp tóm tắt tài liệu. Tại inference, model đọc tài liệu và tạo bản tóm tắt token token qua cross-attention. Pegasus đặc biệt sử dụng một câu khoảng trống pretraining mục tiêu khiến nó trở nên xuất sắc trong việc tóm tắt mà không cần nhiều fine-tuning.

Đánh giá với **ROUGE** (Nghiên cứu định hướng Recall để đánh giá gisting). ROUGE-1 và ROUGE-2 ghi điểm unigram và bigram chồng chéo. ROUGE-L ghi điểm dãy con phổ biến dài nhất. Cao hơn là tốt hơn nhưng 40 ROUGE-L là "tốt" và 50 là "đặc biệt". Mỗi tờ báo đều báo cáo cả ba. Sử dụng gói `rouge-score`.

## Tự xây dựng

### Bước 1: TextRank (trích xuất)

```python
import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

Hai điều đáng để đặt tên. Hàm tương tự sử dụng chồng chéo từ được chuẩn hóa nhật ký, là biến thể TextRank ban đầu. Cosin của TF-IDF vectors cũng hoạt động. Hệ số giảm chấn 0,85 và số lần lặp lại là giá trị mặc định của PageRank.

### Bước 2: trừu tượng hóa với BART

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(long news article text)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN được fine-tuned trên kho dữ liệu CNN/DailyMail. Nó tạo ra các bản tóm tắt kiểu tin tức ngay lập tức. Đối với các lĩnh vực khác (bài báo khoa học, hộp thoại, pháp lý), hãy sử dụng checkpoint hoặc fine-tune Pegasus tương ứng trên dữ liệu mục tiêu của bạn.

### Bước 3: Đánh giá ROUGE

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

Luôn sử dụng thân cây. Nếu không có nó, "chạy" và "chạy" được tính là các từ khác nhau và ROUGE được tính thấp.

### Beyond ROUGE (đánh giá tóm tắt năm 2026)

ROUGE đã là thước đo tổng kết thống trị trong hai mươi năm và bản thân nó không đủ vào năm 2026. Một phân tích tổng hợp quy mô lớn của các bài báo NLG cho thấy:

- **BERTScore** (tương đồng embedding ngữ cảnh) đã đạt được vị trí trong năm 2023 và hiện được báo cáo cùng với ROUGE trong hầu hết các bài báo tóm tắt.
- **BARTScore** coi đánh giá là tạo ra: chấm điểm tóm tắt theo khả năng pretrained BART chỉ định nó cho nguồn.
- **MoverScore** (Khoảng cách của Earth Mover so với embeddings ngữ cảnh) đạt vị trí hàng đầu trong benchmarks tóm tắt năm 2025 vì nó nắm bắt được sự chồng chéo ngữ nghĩa tốt hơn ROUGE.
- **FactCC** và **Trung thành dựa trên QA** là phổ biến trong giai đoạn 2021-2023, hiện thường được thay thế bằng **G-Eval** (một chuỗi GPT-4 prompt điểm tính mạch lạc, nhất quán, trôi chảy, phù hợp với lý luận chain-of-thought).
- **G-Eval** và các phương pháp tiếp cận LLM-judge tương tự phù hợp với phán đoán của con người ~80% thời gian khi các bảng đánh giá được thiết kế tốt.

Khuyến nghị Production: báo cáo ROUGE-L để so sánh kế thừa, BERTScore để so sánh ngữ nghĩa, G-Eval về tính mạch lạc và tính thực tế. Hiệu chỉnh dựa trên 50-100 bản tóm tắt được gắn nhãn của con người.

### Bước 4: bài toán thực tế

Các bản tóm tắt trừu tượng dễ bị ảo giác. Các bản tóm tắt trích xuất có nguy cơ ảo giác thấp hơn nhiều vì đầu ra được nâng nguyên văn từ nguồn, mặc dù chúng vẫn có thể gây hiểu lầm nếu các câu nguồn bị phi ngữ cảnh, lỗi thời hoặc trích dẫn không theo thứ tự. Đây là lý do lớn nhất duy nhất production các hệ thống vẫn thích các phương pháp trích xuất cho nội dung liền kề tuân thủ.

Các loại ảo giác để đặt tên:

- **Hoán đổi thực thể.** Nguồn nói "John Smith." Tóm tắt nói "John Brown."
- **Số lượng trôi dạt.** Nguồn nói "25.000." Tóm tắt cho biết "25 triệu".
- **Lật cực.** Nguồn tin nói rằng "đã từ chối lời đề nghị." Tóm tắt cho biết "đã chấp nhận lời đề nghị".
- **Phát minh thực tế.** Nguồn không đề cập đến CEO. Tóm tắt cho biết Giám đốc điều hành đã chấp thuận.

Các phương pháp đánh giá hiệu quả:

- **FactCC.** Một bộ phân loại nhị phân được huấn luyện về sự bao gồm giữa câu nguồn và câu tóm tắt. Dự đoán factual/not-factual.
- **Tính thực tế dựa trên QA.** Hỏi QA model câu hỏi có câu trả lời trong nguồn. Nếu bản tóm tắt hỗ trợ các câu trả lời khác nhau, hãy gắn cờ.
- **F1 cấp thực thể.** So sánh các thực thể được đặt tên trong nguồn và tóm tắt. Các thực thể chỉ có mặt trong bản tóm tắt là đáng ngờ.

Đối với bất kỳ điều gì đối mặt với người dùng khi tính thực tế quan trọng (tin tức, y tế, pháp lý, tài chính), trích xuất là mặc định an toàn hơn. Trừu tượng cần kiểm tra tính thực tế trong vòng lặp.

## Ứng dụng

stack năm 2026:

| Trường hợp sử dụng | Đề xuất |
|---------|-------------|
| Tin tức, tóm tắt 3-5 câu, Tiếng Anh | `facebook/bart-large-cnn` |
| Bài báo khoa học | `google/pegasus-pubmed` hoặc T5 được điều chỉnh |
| Nhiều tài liệu, dạng dài | Bất kỳ LLM nào có ngữ cảnh 32k+, được nhắc |
| Tóm tắt hộp thoại | `philschmid/bart-large-cnn-samsum` |
| Khai thác, nguy cơ ảo giác thấp do xây dựng | TextRank hoặc LSA / LexRank của `sumy` |

LLMs có bối cảnh dài thường đánh bại models chuyên ngành vào năm 2026 khi điện toán không phải là một ràng buộc. Sự đánh đổi là chi phí và khả năng tái tạo; models chuyên biệt cho đầu ra nhất quán hơn.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-summary-picker.md`:

```markdown
---
name: summary-picker
description: Pick extractive or abstractive, named library, factuality check.
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

Given a task (document type, compliance requirement, length, compute budget), output:

1. Approach. Extractive or abstractive. Explain in one sentence why.
2. Starting model / library. Name it. `sumy.TextRankSummarizer`, `facebook/bart-large-cnn`, `google/pegasus-pubmed`, or an LLM prompt.
3. Evaluation plan. ROUGE-1, ROUGE-2, ROUGE-L (use rouge-score with stemming). Plus factuality check if abstractive.
4. One failure mode to probe. Entity swap is the most common in abstractive news summarization; flag samples where source entities do not appear in summary.

Refuse abstractive summarization for medical, legal, financial, or regulated content without a factuality gate. Flag input over the model's context window as needing chunked map-reduce summarization (not just truncation).
```

## Bài tập

1. **Dễ dàng.** Chạy TextRank trên 5 bài báo. So sánh 3 câu trên cùng với một bản tóm tắt tài liệu tham khảo. Đo ROUGE-L. Bạn sẽ thấy 30-45 ROUGE-L trên CNN/DailyMail-style bài viết.
2. **Medium.** Thực hiện tính thực tế ở cấp độ thực thể: trích xuất các thực thể được đặt tên từ nguồn và tóm tắt (spaCy), tính toán recall của các thực thể nguồn trong tóm tắt và precision các thực thể tóm tắt so với nguồn. precision cao và recall thấp có nghĩa là an toàn nhưng ngắn gọn; precision thấp có nghĩa là các thực thể bị ảo giác.
3. **Khó.** So sánh BART-large-CNN với một LLM (Claude hoặc GPT-4) trên 50 bài báo CNN/DailyMail. Báo cáo ROUGE-L, tính thực tế (theo thực thể F1) và chi phí cho mỗi bản tóm tắt. Ghi lại nơi mỗi người chiến thắng.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Khai thác | Chọn câu | Trả về các câu nguyên văn từ nguồn. Không bao giờ ảo giác. |
| Trừu tượng | Viết lại | Tạo văn bản mới có điều kiện trên nguồn. Có thể bị ảo giác. |
| ĐỎ | Chỉ số tóm tắt | N-gram / LCS chồng chéo giữa đầu ra hệ thống và tham chiếu. |
| Xếp hạng văn bản | Trích xuất dựa trên đồ thị | PageRank trên biểu đồ tương tự câu. |
| Tính thực tế | Có đúng không | Nguồn có hỗ trợ thông báo xác nhận quyền sở hữu tóm tắt hay không. |
| Ảo giác | Nội dung bịa đặt | Nội dung trong bản tóm tắt mà nguồn không hỗ trợ. |

## Đọc thêm

- [Mihalcea and Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) — bài báo kinh điển trích xuất.
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) - giấy BART.
- [Zhang et al. (2019). PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777) — Pegasus và mục tiêu câu khoảng trống.
- [Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) - Giấy ROUGE.
- [Maynez et al. (2020). On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661) - bài báo cảnh quan thực tế.
