# GloVe, FastText và Subword Embeddings

> Word2Vec huấn luyện một embedding cho mỗi từ. GloVe đã phân tích ma trận đồng xuất hiện. FastText đã nhúng các mảnh. BPE nối với transformers.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 5 · 03 (Word2Vec từ đầu)
**Thời lượng:** ~45 phút

## Vấn đề

Word2Vec để lại hai câu hỏi mở.

Đầu tiên, có một dòng nghiên cứu song song phân tích ma trận đồng xuất hiện trực tiếp (LSA, HAL) thay vì thực hiện cập nhật bỏ qua gram trực tuyến. Cách tiếp cận lặp đi lặp lại của Word2Vec về cơ bản có tốt hơn không, hay sự khác biệt là artifact cách hai phương pháp xử lý số lượng? **GloVe** trả lời rằng: phân tích ma trận với một loss được lựa chọn chu đáo phù hợp hoặc đánh bại Word2Vec và chi phí huấn luyện thấp hơn.

Thứ hai, cả hai phương pháp đều không có câu chuyện cho những từ mà nó chưa từng thấy. `Zoomer-approved`, `dogecoin`, bất kỳ danh từ riêng nào được đặt ra vào tuần trước, mọi dạng biến đổi của một gốc hiếm. **FastText** đã khắc phục điều này bằng embedding ký tự n-gram: một từ là tổng các phần của nó, bao gồm cả hình vị, vì vậy ngay cả những từ ngoài từ vựng cũng có vector hợp lý.

Thứ ba, khi transformers đến, câu hỏi lại thay đổi. Từ vựng cấp từ giới hạn khoảng một triệu mục nhập; ngôn ngữ thực sự cởi mở hơn thế. **Mã hóa cặp byte (BPE)** và họ hàng của nó đã giải quyết vấn đề này bằng cách học từ vựng gồm các đơn vị từ phụ thường xuyên bao gồm mọi thứ. Mỗi tokenizer hiện đại cho mọi LLM hiện đại là một từ phụ tokenizer.

Bài học này hướng dẫn cả ba, sau đó giải thích cần tiếp cận khi nào.

## Khái niệm

**GloVe (Global Vectors).** Xây dựng ma trận đồng xuất hiện từ-từ `X` trong đó `X[i][j]` là tần suất `j` từ xuất hiện trong ngữ cảnh của `i` từ. Huấn luyện vectors sao cho `v_i · v_j + b_i + b_j ≈ log(X[i][j])`. Cân loss để các cặp thường xuyên không chiếm ưu thế. Xong.

**FastText.** Một từ là tổng của ký tự n-gram cộng với chính từ. `where` trở nên `<wh, whe, her, ere, re>, <where>`. Từ vector là tổng của các thành phần đó vectors. Huấn luyện như Word2Vec. Lợi ích: các từ không nhìn thấy (`whereupon`) sáng tác từ n-gram đã biết.

**BPE (Mã hóa cặp byte).** Bắt đầu với từ vựng gồm các byte (hoặc ký tự) riêng lẻ. Đếm từng cặp liền kề trong kho dữ liệu. Merge cặp thường xuyên nhất vào một token mới. Lặp lại cho `k` lần lặp lại. Kết quả: một từ vựng `k + 256` tokens trong đó các chuỗi thường xuyên (`ing`, `tion`, `the`) là một tokens và các từ hiếm được chia thành các phần quen thuộc. Mỗi câu mã hóa thành một cái gì đó.

## Tự xây dựng

### GloVe: phân tích ma trận đồng xuất hiện

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

Hai mảnh ghép chuyển động đáng được đặt tên. Chức năng trọng số `f(x) = (x/x_max)^alpha` giảm trọng lượng các cặp rất thường xuyên (như `(the, and)`) để chúng không thống trị loss. Số embedding cuối cùng là tổng của các bảng `W` (giữa) và `W_tilde` (ngữ cảnh). Tổng hợp cả hai là một thủ thuật được công bố có xu hướng vượt trội hơn chỉ khi sử dụng một.

### FastText: embeddings nhận biết từ phụ

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

Mỗi từ được biểu thị bằng tập hợp n gam của nó (thường là 3 đến 6 ký tự). Từ embedding là tổng của embeddings n-gram của nó. Đối với training bỏ qua gram, hãy cắm cái này vào nơi Word2Vec sử dụng một vector duy nhất.

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

Đối với một từ vô hình, bạn vẫn nhận được một vector miễn là một số n-gram của nó được biết đến. `whereupon` chia sẻ `<wh`, `her`, `ere` và `<where` với `where`, vì vậy cả hai hạ cánh gần nhau.

### BPE: từ vựng phụ đã học

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

Lần lặp đầu tiên merges cặp liền kề phổ biến nhất. Sau đủ lần lặp, các chuỗi con thường xuyên (`low`, `est`, `tion`) trở thành một tokens đơn lẻ và các từ hiếm sẽ gặp lỗi hoàn toàn.

Các GPT / BERT / T5 thực sự tokenizers học 30k-100k merges. Kết quả: bất kỳ văn bản nào mã hóa thành một chuỗi có độ dài giới hạn của các ID đã biết, không có OOV nào.

## Ứng dụng

Trong thực tế, bạn hiếm khi tự mình huấn luyện bất kỳ điều nào trong số này. Bạn tải checkpoints được huấn luyện trước.

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

Đối với tokenization subword kiểu BPE trong thời đại transformer:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

Tiền tố `Ġ` đánh dấu ranh giới từ (một quy ước GPT-2). Mỗi tokenizer hiện đại là một biến thể BPE, WordPiece (BERT) hoặc SentencePiece (T5, LLaMA).

### Khi nào nên chọn cái nào

| Tình huống | Chọn |
|-----------|------|
| Pretrained vectors từ có mục đích chung, không cần dung sai OOV | GloVe 300d |
| Pretrained vectors từ có mục đích chung, phải xử lý lỗi chính tả / từ mới / ngôn ngữ giàu hình thái | Văn bản nhanh |
| Bất cứ thứ gì đi vào transformer (training hoặc inference) | Bất cứ điều gì tokenizer model shipped. Không bao giờ hoán đổi. |
| Training model ngôn ngữ của riêng bạn từ đầu | Huấn luyện một BPE hoặc SentencePiece tokenizer trên kho dữ liệu của bạn trước |
| Production phân loại văn bản bằng model tuyến tính | Vẫn là TF-IDF. Bài 02. |

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-embeddings-picker.md`:

```markdown
---
name: tokenizer-picker
description: Pick a tokenization approach for a new language model or text pipeline.
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

Given a task and dataset description, you output:

1. Tokenization strategy (word-level, BPE, WordPiece, SentencePiece, byte-level). One-sentence reason.
2. Vocabulary size target (e.g., 32k for an English-only LM, 64k-100k for multilingual).
3. Library call with the exact training command. Name the library. Quote the arguments.
4. One reproducibility pitfall. Tokenizer-model mismatch is the single most common silent production bug; call out which pair must be used together.

Refuse to recommend training a custom tokenizer when the user is fine-tuning a pretrained LLM. Refuse to recommend word-level tokenization for any model targeting production inference. Flag non-English / multi-script corpora as needing SentencePiece with byte fallback.
```

## Bài tập

1. **Dễ dàng.** Chạy `char_ngrams("playing")` và `char_ngrams("played")`. Tính chồng chéo Jaccard của hai tập n-gram. Bạn sẽ thấy các phần được chia sẻ đáng kể (`pla`, `lay`, `play`), đó là lý do tại sao FastText chuyển tốt qua các biến thể hình thái.
2. **Trung bình.** Mở rộng `learn_bpe` để theo dõi sự phát triển vốn từ vựng. Biểu đồ tokens-per-corpus-character như một hàm của số merges. Lúc đầu, bạn sẽ thấy nén nhanh, tiệm cận gần ~2-3 ký tự mỗi token.
3. **Khó.** Huấn luyện 1k-merge BPE về các tác phẩm hoàn chỉnh của Shakespeare. So sánh tokenization của các từ phổ biến so với danh từ riêng hiếm. Đo tokens trung bình cho mỗi từ trước và sau. Viết ra những gì làm bạn ngạc nhiên.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Ma trận đồng xuất hiện | Bảng tần suất từ-từ | `X[i][j]` = tần suất từ `j` xuất hiện trong cửa sổ xung quanh từ `i`. |
| Từ phụ | Mảnh của một từ | Một ký tự n-gram (FastText) hoặc token đã học (BPE/WordPiece/SentencePiece). |
| BPE | Mã hóa cặp byte | Hợp nhất lặp đi lặp lại các cặp liền kề thường xuyên nhất cho đến khi từ vựng đạt đến kích thước mục tiêu. |
| OOV | Hết từ vựng | Lời mà model chưa từng thấy. Word2Vec/GloVe thất bại. FastText và BPE xử lý nó. |
| BPE cấp byte | BPE trên byte thô | Kế hoạch của GPT-2. Từ vựng bắt đầu với 256 byte, vì vậy không có gì là OOV. |

## Đọc thêm

- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) - bài báo GloVe, bảy trang, vẫn là nguồn gốc tốt nhất của loss.
- [Bojanowski et al. (2017). Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) - Văn bản nhanh.
- [Sennrich, Haddow, Birch (2016). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) - bài báo giới thiệu BPE với NLP hiện đại.
- [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary) - BPE, WordPiece và SentencePiece thực sự khác nhau như thế nào trong thực tế.
