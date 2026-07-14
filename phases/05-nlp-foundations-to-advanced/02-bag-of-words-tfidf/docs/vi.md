# Túi từ, TF-IDF và biểu diễn văn bản

> Đếm trước, suy nghĩ sau. TF-IDF vẫn đánh bại embeddings trong các nhiệm vụ được xác định rõ ràng vào năm 2026.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 5 · 01 (Xử lý văn bản), Giai đoạn 2 · 02 (Hồi quy tuyến tính từ đầu)
**Thời lượng:** ~75 phút

## Vấn đề

model cần những con số. Bạn có dây.

Mỗi NLP pipeline phải trả lời cùng một câu hỏi. Làm thế nào để chúng ta biến một luồng tokens có độ dài thay đổi thành một vector có kích thước cố định mà bộ phân loại có thể sử dụng. Câu trả lời đầu tiên mà lĩnh vực này đáp xuống là câu trả lời ngu ngốc nhất hoạt động. Đếm các từ. Tạo vector.

vector đó đã mang nhiều production NLP hơn bất kỳ embedding model nào. Bộ lọc thư rác, phân loại chủ đề, phát hiện bất thường nhật ký, xếp hạng tìm kiếm (trước BM25), làn sóng phân tích cảm xúc đầu tiên, thập kỷ đầu tiên của NLP benchmarks học thuật. Năm 2026, các học viên vẫn tiếp cận nó đầu tiên trong các nhiệm vụ phân loại hẹp. Nó nhanh, dễ giải thích và thường không thể phân biệt được với 400 triệu parameter embedding model về các nhiệm vụ mà sự hiện diện của từ là điều quan trọng.

Bài học này xây dựng túi từ, sau đó là TF-IDF, từ đầu. Sau đó hiển thị scikit-learn làm điều tương tự trong ba dòng. Sau đó đặt tên cho chế độ thất bại khiến bạn tiếp cận embeddings.

## Khái niệm

**Túi từ (BoW)** vứt bỏ trật tự. Đối với mỗi tài liệu, hãy đếm số lần mỗi từ vựng xuất hiện. Độ dài Vector là kích thước từ vựng. Vị trí `i` là số lượng từ `i`.

**TF-IDF** cân lại BoW. Một từ xuất hiện trong mọi tài liệu là không có thông tin, vì vậy hãy thu nhỏ nó. Một từ hiếm trong kho dữ liệu nhưng thường xuyên trong một tài liệu là tín hiệu, vì vậy hãy mở rộng quy mô của nó.

```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

Trong đó `TF` là tần suất thuật ngữ trong tài liệu, `df` là tần suất tài liệu (có bao nhiêu tài liệu chứa từ), `N` là tổng số tài liệu. `log` giữ trọng lượng giới hạn cho các từ phổ biến.

Thuộc tính chính: cả hai đều tạo ra các vectors thưa thớt với các trục có thể diễn giải. Bạn có thể xem trọng số của bộ phân loại được huấn luyện và đọc từ nào đẩy tài liệu về phía mỗi class. Bạn không thể làm điều này với BERT embedding 768 chiều.

```figure
bow-tfidf
```

## Tự xây dựng

### Bước 1: xây dựng vốn từ vựng

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

Đầu vào: danh sách các tài liệu được mã hóa (bất kỳ tokenizer cấp từ nào cũng sẽ làm được; `code/main.py` trong bài học này sử dụng một biến thể chữ thường được đơn giản hóa). Đầu ra: `{word: index}` dict. Thứ tự chèn ổn định có nghĩa là chỉ mục từ 0 là từ đầu tiên được nhìn thấy trong tài liệu đầu tiên. Quy ước khác nhau; scikit-learn sắp xếp theo thứ tự bảng chữ cái.

### Bước 2: túi từ

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

Hàng là tài liệu. Cột là chỉ số từ vựng. Mục `[i][j]` là "bao nhiêu lần từ `j` xuất hiện trong tài liệu `i`". Doc 1 đã `cat` hai lần vì nó đã xảy ra. Doc 0 `ran` không lần vì nó không có.

### Bước 3: tần suất thuật ngữ và tần suất tài liệu

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

Hai thủ thuật làm mịn đáng để đặt tên. `(n+1)/(d+1)` tránh `log(x/0)`. Dấu `+1` đuôi đảm bảo một từ trong mỗi tài liệu vẫn có IDF 1 (không phải 0), khớp với mặc định của scikit-learn. Các triển khai khác sử dụng `log(N/df)` thô. Cả hai đều hoạt động; phiên bản làm mịn thân thiện hơn.

### Bước 4: TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

Ba tài liệu, năm từ vựng (`the`, `cat`, `sat`, `dog`, `ran`). `the` xuất hiện trong cả ba, vì vậy IDF của nó thấp. `dog` xuất hiện trong một, vì vậy IDF của nó cao. Các vectors thưa thớt (hầu hết các mục nhỏ) và các từ phân biệt đối xử xuất hiện.

### Bước 5: Chuẩn hóa các hàng L2

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

Nếu không chuẩn hóa, một tài liệu dài hơn sẽ có một vector lớn hơn và thống trị điểm tương tự. Chuẩn hóa L2 đặt mọi tài liệu trên siêu quyển đơn vị. Sự tương đồng cosin giữa các hàng bây giờ chỉ là một tích chấm.

## Ứng dụng

scikit-learn ships phiên bản production.

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer` thực hiện tokenization, từ vựng và BoW trong một cuộc gọi. `TfidfVectorizer` bổ sung trọng số IDF và chuẩn hóa L2. Cả hai đều trả về ma trận thưa thớt. Đối với tài liệu 100k, phiên bản dày đặc không vừa với bộ nhớ; Giữ thưa thớt cho đến khi bộ phân loại yêu cầu dày đặc.

Các núm thay đổi mọi thứ:

| Tranh luận | Hiệu ứng |
|-----|--------|
| `ngram_range=(1, 2)` | Bao gồm bigram. Thường tăng cường phân loại. |
| `min_df=2` | Thả từ vào ít hơn 2 tài liệu. Cắt bớt từ vựng trên dữ liệu nhiễu. |
| `max_df=0.95` | Thả từ vào hơn 95% tài liệu. Xấp xỉ loại bỏ từ dừng mà không cần danh sách được mã hóa cứng. |
| `stop_words="english"` | Danh sách từ dừng tích hợp của scikit-learn. Phụ thuộc vào nhiệm vụ - phân tích tình cảm * không nên * bỏ phủ định. |
| `sublinear_tf=True` | Sử dụng `1 + log(tf)` thay vì `tf` thô. Giúp khi một thuật ngữ lặp lại nhiều lần trong một tài liệu. |

### Khi TF-IDF vẫn thắng (tính đến năm 2026)

- Phát hiện thư rác, gắn nhãn chủ đề, gắn cờ bất thường nhật ký. Sự hiện diện của từ ngữ là điều quan trọng; sắc thái ngữ nghĩa thì không.
- Chế độ dữ liệu thấp (hàng trăm ví dụ được dán nhãn). TF-IDF cộng với hồi quy logistic không có chi phí pretraining.
- Bất cứ nơi nào độ trễ đều quan trọng. TF-IDF cộng với một model tuyến tính trả lời trong micro giây. Embedding tài liệu thông qua transformer mất 10-100ms.
- Các hệ thống phải giải thích dự đoán của họ. Kiểm tra hệ số của bộ phân loại. Những từ tích cực hàng đầu là lý do.

### Khi TF-IDF thất bại

Thất bại mù ngữ nghĩa. Hãy xem xét hai tài liệu sau:

- "Bộ phim không hay chút nào."
- "Bộ phim rất xuất sắc."

Một là đánh giá tiêu cực. Một là tích cực. Sự chồng chéo TF-IDF của họ chính xác là `{the, movie, was}`. Một bộ phân loại túi từ phải ghi nhớ rằng từ `not` gần `good` lật nhãn. Nó có thể học điều này trên đủ dữ liệu, nhưng không bao giờ duyên dáng như một model hiểu cú pháp.

Thất bại khác: những từ ngoài từ vựng ở inference. Một model BoW được huấn luyện về các bài đánh giá IMDb không biết phải làm gì với `Zoomer-approved` nếu token đó không bao giờ xuất hiện trong training. Subword embeddings (bài 04) xử lý điều này. TF-IDF không thể.

### Kết hợp: embeddings trọng số TF-IDF

Mặc định thực dụng năm 2026 cho phân loại dữ liệu trung bình: sử dụng trọng số TF-IDF dưới dạng attention trên embeddings từ.

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

Bạn nhận được năng lực ngữ nghĩa từ embeddings và nhấn mạnh từ hiếm từ TF-IDF. Bộ phân loại huấn luyện trên vector gộp. Điều này vượt trội hơn đối với phân loại cảm xúc, chủ đề và ý định dưới khoảng 50 nghìn ví dụ được gắn nhãn.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/prompt-vectorization-picker.md`:

```markdown
---
name: vectorization-picker
description: Given a text-classification task, recommend BoW, TF-IDF, embeddings, or a hybrid.
phase: 5
lesson: 02
---

You recommend a text-vectorization strategy. Given a task description, output:

1. Representation (BoW, TF-IDF, transformer embeddings, or a hybrid). Explain why in one sentence.
2. Specific vectorizer configuration. Name the library. Quote the arguments (`ngram_range`, `min_df`, `max_df`, `sublinear_tf`, `stop_words`).
3. One failure mode to test before shipping.

Refuse to recommend embeddings when the user has under 500 labeled examples unless they show evidence of semantic failure in a TF-IDF baseline. Refuse to remove stopwords for sentiment analysis (negations carry signal). Flag class imbalance as needing more than a vectorizer change.

Example input: "Classifying 30k customer support tickets into 12 categories. Most tickets are 2-3 sentences. English only. Need explainability for audit logs."

Example output:

- Representation: TF-IDF. 30k examples is not small; explainability requirement rules out dense embeddings.
- Config: `TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`. Keep stopwords because category keywords sometimes are stopwords ("not working" vs "working").
- Failure to test: verify `min_df=3` does not drop rare category keywords. Run `get_feature_names_out` filtered by class and eyeball.
```

## Bài tập

1. **Dễ dàng.** Triển khai `cosine_similarity(doc_vec_a, doc_vec_b)` trên đầu ra TF-IDF chuẩn hóa L2. Xác minh rằng các tài liệu giống hệt nhau đạt điểm 1.0 và tài liệu từ vựng rời rạc đạt điểm 0.0.
2. **Trung bình.** Thêm hỗ trợ `n-gram` vào `bag_of_words`. Parameter `n` tạo ra số lượng trên `n` gam. Kiểm tra xem `n=2` trên `["the", "cat", "sat"]` tạo ra số lượng bigram cho `["the cat", "cat sat"]`.
3. **Khó.** Xây dựng kết hợp TF-IDF-weighted-embedding ở trên bằng cách sử dụng vectors GloVe 100d (tải xuống một lần, bộ nhớ cache). So sánh accuracy phân loại với TF-IDF đơn giản và embeddings tổng hợp trung bình đơn giản trên 20 Nhóm tin tức dataset. Báo cáo cái nào thắng ở đâu.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| BoW | Tần suất từ vector | Số lượng từ vựng trong một tài liệu. Vứt bỏ trật tự. |
| TF | Tần suất kỳ hạn | Số lượng từ trong tài liệu, tùy chọn chuẩn hóa theo độ dài tài liệu. |
| Hậu vệ | Tần suất tài liệu | Đếm tài liệu chứa từ đó ít nhất một lần. |
| IDF | Tần số tài liệu nghịch đảo | `log(N / df)` được làm mịn. Giảm trọng lượng các từ xuất hiện ở khắp mọi nơi. |
| vector thưa thớt | Chủ yếu là số không | Từ vựng thường là 10k-100k từ; hầu hết đều vắng mặt trong bất kỳ tài liệu nhất định nào. |
| Sự tương đồng cosin | Góc Vector | Sản phẩm chấm của vectors chuẩn hóa L2. 1 giống hệt nhau, 0 là trực giao. |

## Đọc thêm

- [scikit-learn — feature extraction from text](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — tham chiếu API chuẩn tắc, cộng với các ghi chú trên mỗi núm.
- [Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval](https://www.sciencedirect.com/science/article/pii/0306457388900210) - tờ báo đã khiến TF-IDF trở thành mặc định trong một thập kỷ.
- ["Why TF-IDF Still Beats Embeddings" — Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) - Năm 2026 diễn ra khi phương pháp cũ chiến thắng và tại sao.
