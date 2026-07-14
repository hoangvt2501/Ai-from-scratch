# Tạo văn bản trước Transformers - Ngôn ngữ N-gram Models

> Nếu một từ đáng ngạc nhiên, model là xấu. Perplexity tạo ra một con số bất ngờ. Làm mịn giữ cho nó hữu hạn.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 5 · 01 (Xử lý văn bản), Giai đoạn 2 · 14 (Bayes ngây thơ)
**Thời lượng:** ~45 phút

## Vấn đề

Trước transformers, trước RNN, trước embeddings từ, một ngôn ngữ model dự đoán từ tiếp theo bằng cách đếm tần suất nó theo sau `n-1` từ trước đó. Đếm "con mèo" → "ngồi" 47 lần, "con mèo" → "nhảy" 12 lần, "con mèo" → "tủ lạnh" 0 lần. Chuẩn hóa để có được phân phối xác suất.

Đó là một ngôn ngữ n-gram model. Nó chạy mọi trình nhận dạng giọng nói, mọi trình kiểm tra chính tả và mọi hệ thống dịch máy dựa trên cụm từ từ năm 1980 đến năm 2015. Nó vẫn chạy khi bạn cần mô hình ngôn ngữ trên thiết bị giá rẻ.

Vấn đề thú vị là phải làm gì với n-gram không nhìn thấy. Một model dựa trên số lượng thô gán xác suất bằng không cho bất cứ thứ gì nó chưa thấy, điều này là thảm họa vì các câu dài và hầu hết mọi câu dài đều chứa ít nhất một chuỗi không nhìn thấy. Năm mươi năm nghiên cứu làm mịn đã khắc phục điều đó. Kết quả là làm mịn Kneser-Ney và deep learning hiện đại kế thừa truyền thống thực nghiệm của nó.

## Khái niệm

![N-gram model: count, smooth, generate](../assets/ngram.svg)

**Xác suất N-gram: **`P(w_i | w_{i-n+1}, ..., w_{i-1})`. Cố định `n` (thường là 3 cho tam giác, 4 cho 4 gam). Tính toán từ số đếm:

```text
P(w | context) = count(context, w) / count(context)
```

**Bài toán số không.** Bất kỳ n-gram nào không được nhìn thấy trong training đều có xác suất bằng không. Một nghiên cứu năm 2007 về kho dữ liệu Brown cho thấy ngay cả một model 4 gram cũng có 30% 4 gram không được nhìn thấy trong training. Bạn không thể đánh giá trên bất kỳ văn bản thực nào mà không làm mịn.

**Cách tiếp cận làm mịn, theo thứ tự tinh vi:**

1. **Laplace (add-one).** Thêm 1 vào mỗi lần đếm. Đơn giản, khủng khiếp trong những sự kiện hiếm hoi.
2. **Tốt-Turing.** Phân bổ lại khối lượng xác suất từ các sự kiện tần số cao hơn sang các sự kiện không nhìn thấy dựa trên tần số của tần số.
3. **Nội suy.** Kết hợp ước tính n-gram, (n-1)-gram, v.v. với trọng số có thể điều chỉnh.
4. **Lùi lại.** Nếu n-gram có số không, hãy quay trở lại (n-1) -gram. Katz lùi lại bình thường hóa điều này.
5. **Giảm giá tuyệt đối.** Trừ đi một `D` chiết khấu cố định từ tất cả các số lượng, phân phối lại cho không nhìn thấy.
6. **Kneser-Ney.** Chiết khấu tuyệt đối cộng với một lựa chọn thông minh cho model bậc thấp hơn: sử dụng *xác suất tiếp tục* (có bao nhiêu ngữ cảnh một từ xuất hiện) thay vì tần suất thô.

Cái nhìn sâu sắc của Kneser-Ney rất sâu sắc. "San Francisco" là một bigram phổ biến. Unigram "Francisco" xuất hiện chủ yếu sau "San". Chiết khấu tuyệt đối ngây thơ mang lại cho "Francisco" xác suất unigram cao (vì số lượng cao). Kneser-Ney nhận thấy rằng "Francisco" chỉ xuất hiện trong một bối cảnh và giảm xác suất tiếp tục của nó cho phù hợp. Kết quả: một bigram tiểu thuyết kết thúc bằng "Francisco" có xác suất thấp thích hợp.

**Đánh giá: perplexity.** Số mũ của log-likelihood tiêu cực trung bình trên mỗi từ trên một tập thử nghiệm được giữ lại. Thấp hơn là tốt hơn. perplexity 100 có nghĩa là model bị nhầm lẫn như nó sẽ chọn đồng nhất trong số 100 từ.

```text
perplexity = exp(- (1/N) * Σ log P(w_i | context_i))
```

```figure
ngram-backoff
```

## Tự xây dựng

### Bước 1: đếm trigram

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

Đầu vào là một danh sách các câu được mã hóa. Đầu ra là số lượng n-gram và số lượng ngữ cảnh. `<s>` và `</s>` là ranh giới câu.

### Bước 2: Làm mịn Laplace

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

Thêm 1 vào mỗi lần đếm. Làm mịn nhưng phân bổ quá mức khối lượng cho các sự kiện không nhìn thấy, làm tổn thương các sự kiện hiếm hoi được biết đến.

### Bước 3: Kneser-Ney (bigram, nội suy)

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

Ba bộ phận chuyển động. `continuation_prob` nắm bắt được "từ này xuất hiện trong bao nhiêu ngữ cảnh khác nhau?" (sự đổi mới của Kneser-Ney). `lambda_prev` là khối lượng được giải phóng bởi chiết khấu, được sử dụng để cân nhắc backoff. Xác suất cuối cùng là số hạng chính chiết khấu cộng với số hạng tiếp tục có trọng số.

### Bước 4: tạo văn bản bằng sampling

```python
import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

Sampling tỷ lệ thuận với xác suất. Luôn cho đầu ra khác nhau cho mỗi hạt giống. Đối với đầu ra giống như tìm kiếm chùm tia, hãy chọn argmax ở mỗi bước (tham lam) và thêm một núm ngẫu nhiên nhỏ (temperature).

### Bước 5: perplexity

```python
import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

Thấp hơn là tốt hơn. Đối với kho dữ liệu Brown, một model KN 4 gram được điều chỉnh tốt đạt perplexity khoảng 140. Một transformer LM đạt 15-30 trên cùng một bộ thử nghiệm. Khoảng cách là khoảng 10x. Khoảng cách đó là lý do tại sao lĩnh vực này tiếp tục.

## Ứng dụng

- **Giảng dạy NLP cổ điển.** Tiếp xúc rõ ràng nhất với làm mịn, MLE và perplexity bạn có thể nhận được.
- **KenLM.** Production thư viện n-gram. Được sử dụng như một công cụ ghi điểm lại trong các hệ thống giọng nói và MT, nơi độ trễ thấp quan trọng.
- **Tự động hoàn thành trên thiết bị.** models Trigram trong bàn phím. Tuy nhiên.
- **Đường cơ sở.** Luôn tính toán một perplexity LM n-gram trước khi tuyên bố LM thần kinh của bạn tốt. Nếu transformer của bạn không đánh bại KN với tỷ lệ chênh lệch lớn, có điều gì đó không ổn.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/prompt-lm-baseline.md`:

```markdown
---
name: lm-baseline
description: Build a reproducible n-gram language model baseline before training a neural LM.
phase: 5
lesson: 16
---

Given a corpus and target use (next-word prediction, rescoring, perplexity baseline), output:

1. N-gram order. Trigram for general English, 4-gram if corpus is large, 5-gram for speech rescoring.
2. Smoothing. Modified Kneser-Ney is the default; Laplace only for teaching.
3. Library. `kenlm` for production, `nltk.lm` for teaching, roll your own only to learn.
4. Evaluation. Held-out perplexity with consistent tokenization between train and test sets.

Refuse to report perplexity computed with different tokenization between systems being compared — perplexity numbers are comparable only under identical tokenization. Flag OOV rate in test set; KN handles OOV poorly unless you reserve a special <UNK> token during training.
```

## Bài tập

1. **Dễ dàng.** Huấn luyện một bộ ba LM trên kho ngữ liệu Shakespeare dài 1.000 câu. Tạo 20 câu. Chúng sẽ hợp lý về mặt địa phương nhưng không mạch lạc trên toàn cầu. Đây là bản demo chính tắc.
2. **Trung bình.** Triển khai perplexity cho KN model của bạn về một phần tách Shakespeare được giữ lại. So sánh với Laplace. Bạn sẽ thấy KN thấp hơn perplexity 30-50%.
3. **Khó.** Xây dựng một trình sửa lỗi chính tả tam giác: cho một từ sai chính tả và ngữ cảnh của nó, hãy tạo các chỉnh sửa và xếp hạng theo xác suất ngữ cảnh trong LM. Đánh giá trên kho dữ liệu chính tả Birkbeck (công khai).

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| N-gram | Dãy từ | Trình tự `n` tokens liên tiếp. |
| Làm mịn | Tránh số không | Phân bổ lại khối lượng xác suất để các sự kiện không nhìn thấy có xác suất không bằng không. |
| Perplexity | Chỉ số chất lượng LM | `exp(-average log-prob)` dữ liệu bị giữ lại. Thấp hơn là tốt hơn. |
| Lùi lại | Dự phòng với ngữ cảnh ngắn hơn | Nếu số lượng tam giác bằng không, hãy sử dụng bigram. Katz lùi lại chính thức hóa điều này. |
| Kneser-Ney | Làm mịn tốt nhất cho n-gram | Chiết khấu tuyệt đối + xác suất tiếp tục cho model bậc thấp hơn. |
| Xác suất tiếp tục | KN cụ thể | `P(w)` trọng số theo số ngữ cảnh `w` xuất hiện, không phải theo số lượng thô. |

## Đọc thêm

- [Jurafsky and Martin — Speech and Language Processing, Chapter 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) - điều trị kinh điển của LM n-gram và làm mịn.
- [Chen and Goodman (1998). An Empirical Study of Smoothing Techniques for Language Modeling](https://dash.harvard.edu/handle/1/25104739) - tờ giấy đã giải quyết Kneser-Ney là chất mịn n-gram tốt nhất.
- [Kneser and Ney (1995). Improved Backing-off for M-gram Language Modeling](https://ieeexplore.ieee.org/document/479394) - bài báo KN gốc.
- [KenLM](https://kheafield.com/code/kenlm/) - LM n-gram production nhanh, vẫn được sử dụng vào năm 2026 cho các ứng dụng nhạy cảm với độ trễ.
