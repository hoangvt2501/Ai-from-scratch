# Word Embeddings — Word2Vec từ đầu

> Một từ là công ty mà nó giữ. Huấn luyện một mạng nông về ý tưởng đó và hình học rơi ra.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 5 · 02 (BoW + TF-IDF), Giai đoạn 3 · 03 (Backpropagation từ đầu)
**Thời lượng:** ~75 phút

## Vấn đề

TF-IDF biết `dog` và `puppy` là những từ khác nhau. Nó không biết chúng có nghĩa gần như giống nhau. Một bộ phân loại được huấn luyện trên `dog` không thể khái quát hóa thành một đánh giá về `puppy`. Bạn có thể liệt kê điều này bằng cách liệt kê các từ đồng nghĩa, nhưng điều đó không thành công với các thuật ngữ hiếm hoi, biệt ngữ tên miền và mọi ngôn ngữ bạn không lường trước được.

Bạn muốn một đại diện nơi `dog` và `puppy` hạ cánh gần nhau trong không gian. Nơi `king - man + woman` hạ cánh gần `queen`. Nơi một model được huấn luyện về `dog` chuyển một số tín hiệu cho `puppy` miễn phí.

Word2Vec đã cho chúng tôi không gian đó. Mạng nơ-ron hai lớp, chạy nghìn tỷ token training, được xuất bản vào năm 2013. Kiến trúc gần như đơn giản một cách đáng xấu hổ. Kết quả đã định hình lại NLP trong một thập kỷ.

## Khái niệm

**Giả thuyết phân phối** (Firth, 1957): "Bạn sẽ biết một từ từ từ công ty mà nó giữ." Nếu hai từ xuất hiện trong ngữ cảnh tương tự, chúng có thể có nghĩa tương tự.

Word2Vec có hai hương vị, cả hai đều khai thác ý tưởng đó.

- **Bỏ qua-gram.** Cho một từ ở giữa, hãy dự đoán các từ xung quanh. `cat -> (the, sat, on)` với kích thước cửa sổ 2.
- **CBOW (túi từ liên tục).** Cho các từ xung quanh, hãy dự đoán trung tâm. `(the, sat, on) -> cat`.

Skip-gram huấn luyện chậm hơn nhưng xử lý các từ hiếm tốt hơn. Nó đã trở thành mặc định.

Mạng có một lớp ẩn không có tính phi tuyến. Đầu vào là một vector một nóng trên từ vựng. Đầu ra là một softmax trên từ vựng. Sau khi training, bạn vứt bỏ layer đầu ra. Trọng lượng lớp ẩn là embeddings.

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          this is the embedding
```

Bí quyết: softmax hơn 100 nghìn từ là quá đắt. Word2Vec sử dụng **sampling âm** để biến nó thành một nhiệm vụ phân loại nhị phân. Dự đoán "từ ngữ cảnh này có xuất hiện gần từ trung tâm này không, có hay không". Lấy mẫu một số từ phủ định (không đồng thời) cho mỗi cặp training thay vì tính toán softmax trên toàn bộ từ vựng.

```figure
word-vector-arithmetic
```

## Tự xây dựng

### Bước 1: training cặp từ kho dữ liệu

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

Mỗi cặp (trung tâm, ngữ cảnh) trong một cửa sổ là một ví dụ training tích cực.

### Bước 2: embedding bảng

Hai ma trận. `W` là từ trung tâm embedding bảng (bảng bạn giữ). `W'` là bảng ngữ cảnh-từ (thường bị loại bỏ, đôi khi được tính trung bình bằng `W`).

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

Khởi tạo ngẫu nhiên nhỏ. Vocab size 10k và dim 100 là thực tế; Để giảng dạy, 50 từ vựng x 16 mờ là đủ để xem hình học.

### Bước 3: tiêu cực sampling mục tiêu

Đối với mỗi cặp dương tính `(center, context)`, lấy mẫu `k` từ ngẫu nhiên từ từ vựng dưới dạng phủ định. Huấn luyện model sao cho `W[center] · W'[context]` tích chấm cao đối với tích cực và thấp đối với tiêu cực.

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

Công thức kỳ diệu: loss logistic trên cặp dương (muốn sigmoid gần 1) cộng với logistic loss trên các cặp âm (muốn sigmoid gần 0). Gradients luồng đến cả hai bàn. Dẫn xuất đầy đủ có trong bài báo gốc; Đi qua nó một lần bằng bút chì và giấy nếu bạn muốn nó dính.

### Bước 4: huấn luyện trên kho đồ chơi

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

Sau khi epochs đủ trên một kho dữ liệu lớn, các từ chia sẻ ngữ cảnh có trung tâm tương tự embeddings. Trên kho đồ chơi, bạn thấy hiệu ứng mờ nhạt. Trên hàng tỷ tokens, bạn thấy điều đó một cách ấn tượng.

### Bước 5: thủ thuật tương tự

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

Trên vectors Google Tin tức 300d được huấn luyện trước:

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`. Không phải vì model biết tiền bản quyền là gì. Bởi vì vector `(king - man)` nắm bắt một cái gì đó giống như "hoàng gia", và thêm nó vào `woman` vùng đất gần khu vực hoàng gia-nữ.

## Ứng dụng

Viết Word2Vec từ đầu là giảng dạy. Production NLP sử dụng `gensim`.

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

Đối với công việc thực sự, bạn hầu như không bao giờ tự huấn luyện Word2Vec. Bạn tải xuống các vectors được huấn luyện trước.

- **GloVe** - Phương pháp tiếp cận phân tích ma trận đồng xuất hiện của Stanford. 50d, 100d, 200d, 300d checkpoints. Bảo hiểm chung tốt. Bài 04 đề cập cụ thể đến GloVe.
- **fastText** — Tiện ích mở rộng Word2Vec của Facebook nhúng ký tự n-gram. Xử lý các từ ngoài từ vựng bằng cách soạn các từ phụ. Bài 04.
- **Pretrained Word2Vec trên Google Tin tức** — 300d, từ vựng 3 triệu từ, xuất bản năm 2013. Vẫn được tải xuống hàng ngày.

### Khi Word2Vec vẫn thắng vào năm 2026

- Truy xuất tên miền cụ thể nhẹ. Huấn luyện về các bản tóm tắt y tế trong một giờ trên máy tính xách tay, nhận chuyên môn vectors không nắm bắt model chung.
- feature engineering theo phong cách tương tự. `gender_vector = mean(man - woman pairs)`. Trừ nó khỏi các từ khác để có được một trục trung lập về giới tính. Vẫn được sử dụng trong nghiên cứu công bằng.
- Khả năng giải thích. 100d đủ nhỏ để vẽ biểu đồ thông qua PCA hoặc t-SNE và thực sự thấy các cụm hình thành.
- Bất cứ nơi nào inference phải chạy trên thiết bị mà không cần GPU. Tra cứu Word2Vec là tìm nạp một hàng.

### Nơi Word2Vec không thành công

Bức tường đa nghĩa. `bank` có một vector. `river bank` và `financial bank` chia sẻ nó. `table` (bảng tính so với đồ nội thất) chia sẻ nó. Một bộ phân loại ở hạ lưu không thể phân biệt các giác quan với vector.

embeddings ngữ cảnh (ELMo, BERT, mỗi transformer kể từ đó) đã giải quyết vấn đề này bằng cách tạo ra một vector khác nhau cho mỗi lần xuất hiện của từ dựa trên ngữ cảnh xung quanh. Đó là bước nhảy từ Word2Vec sang BERT: từ tĩnh sang ngữ cảnh. Giai đoạn 7 bao gồm nửa transformer.

Vấn đề ngoài vốn từ vựng là thất bại khác. Word2Vec chưa bao giờ thấy `Zoomer-approved` nếu nó không có trong dữ liệu training. Không có dự phòng. fastText khắc phục điều này bằng cách thành phần từ phụ (bài 04).

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-embedding-probe.md`:

```markdown
---
name: embedding-probe
description: Inspect a word2vec model. Run analogies, find neighbors, diagnose quality.
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

You probe trained word embeddings to verify they are working. Given a `gensim.models.KeyedVectors` object and a vocabulary, you run:

1. Three canonical analogy tests. `king : man :: queen : woman`. `paris : france :: tokyo : japan`. `walking : walked :: swimming : ?`. Report the top-1 result and its cosine.
2. Five nearest-neighbor tests on domain-specific words the user supplies. Print top-5 neighbors with cosines.
3. One symmetry check. `similarity(a, b) == similarity(b, a)` to within float precision.
4. One degenerate check. If any embedding has a norm below 0.01 or above 100, the model has a training bug. Flag it.

Refuse to declare a model good on analogy accuracy alone. Analogy benchmarks are gameable and do not transfer to downstream tasks. Recommend intrinsic + downstream evaluation together.
```

## Bài tập

1. **Dễ dàng.** Chạy vòng lặp training trên một kho dữ liệu nhỏ (20 câu về chó mèo). Sau 200 epochs, Verify `nearest(vocab, W, W[vocab["cat"]])` trả về `dog` trong top 3. Nếu không, hãy tăng epochs hoặc từ vựng.
2. **Trung bình.** Thêm mẫu phụ của các từ thường gặp. Các từ có tần suất trên `10^-5` được loại khỏi training cặp với xác suất tỷ lệ thuận với tần suất của chúng. Đo lường ảnh hưởng đến sự giống nhau của từ hiếm.
3. **Khó.** Huấn luyện một model trên kho dữ liệu 20 Nhóm tin tức. Tính toán hai trục bias: `he - she` và `doctor - nurse`. Chiếu các từ nghề nghiệp lên cả hai trục. Báo cáo nghề nghiệp nào có khoảng cách bias lớn nhất. Đây là loại công bằng đầu dò mà các nhà nghiên cứu sử dụng.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Từ embedding | Word như một vector | Một biểu diễn dày đặc, ít mờ (thường là 100-300) được học từ ngữ cảnh. |
| Bỏ qua gram | Thủ thuật Word2Vec | Dự đoán các từ ngữ cảnh từ từ trung tâm. Chậm hơn CBOW, tốt hơn cho những từ hiếm. |
| sampling tiêu cực | Training phím tắt | Thay thế softmax trên từ vựng đầy đủ bằng phân loại nhị phân với `k` từ ngẫu nhiên. |
| embedding tĩnh | Một vector mỗi từ | Giống nhau vector bất kể ngữ cảnh. Thất bại về đa nghĩa. |
| embedding theo ngữ cảnh | vector nhạy cảm với ngữ cảnh | Các vector khác nhau cho mỗi lần xuất hiện dựa trên các từ xung quanh. Những gì transformers sản xuất. |
| OOV | Hết từ vựng | Từ không thấy trong training. Word2Vec không thể tạo ra một vector cho những thứ này. |

## Đọc thêm

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) - bài báo sampling âm. Ngắn gọn và dễ đọc.
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) - dẫn xuất rõ ràng nhất của gradients, nếu toán học của bài báo gốc cảm thấy dày đặc.
- [gensim Word2Vec tutorial](https://radimrehurek.com/gensim/models/word2vec.html) — production training cài đặt thực sự hoạt động.
