# Phân tích cảm xúc

> Nhiệm vụ NLP chính tắc. Hầu hết những gì bạn cần biết về phân loại văn bản cổ điển đều hiển thị ở đây.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 5 · 02 (BoW + TF-IDF), Giai đoạn 2 · 14 (Bayes ngây thơ)
**Thời lượng:** ~75 phút

## Vấn đề

"Thức ăn không ngon." Tích cực hay tiêu cực?

Tình cảm nghe có vẻ đơn giản. Một người đánh giá nói rằng họ thích hoặc không thích điều gì đó. Gắn nhãn câu. Lý do nó trở thành nhiệm vụ NLP kinh điển là vì mọi trường hợp trông dễ dàng đều ẩn chứa một trường hợp khó khăn. Phủ định lật ngược ý nghĩa. Sự mỉa mai đảo ngược nó. "Không tệ chút nào" là tích cực mặc dù có hai từ được mã hóa phủ định. Biểu tượng cảm xúc mang nhiều tín hiệu hơn văn bản xung quanh. Từ vựng miền quan trọng (`tight` trong đánh giá âm nhạc so với `tight` trong đánh giá thời trang).

Tình cảm là một phòng thí nghiệm làm việc cho NLP cổ điển. Nếu bạn hiểu tại sao mọi đường cơ sở ngây thơ đều có một chế độ thất bại cụ thể, bạn sẽ hiểu tại sao mọi model giàu hơn lại được phát minh. Bài học này xây dựng đường cơ sở Naive Bayes từ đầu, thêm hồi quy logistic và đặt tên cho những cái bẫy khiến tâm lý production trở thành một vấn đề cấp độ tuân thủ.

## Khái niệm

Tình cảm cổ điển là một công thức gồm hai bước.

1. **Đại diện.** Biến văn bản thành feature vector. BoW, TF-IDF hoặc n-gram.
2. **Phân loại.** Phù hợp với model tuyến tính (Naive Bayes, hồi quy logistic, SVM) trên các ví dụ được gắn nhãn.

Naive Bayes là model ngu ngốc nhất hoạt động. Giả sử mọi feature đều độc lập với nhãn. Ước tính `P(word | positive)` và `P(word | negative)` từ số đếm. Khi inference, nhân xác suất. Giả định độc lập "ngây thơ" là sai lầm một cách buồn cười nhưng kết quả lại mạnh mẽ một cách đáng kinh ngạc. Lý do: với features văn bản thưa thớt và dữ liệu vừa phải, bộ phân loại quan tâm đến việc mỗi từ nghiêng về phía nào hơn là bao nhiêu.

Hồi quy logistic sửa giả định độc lập. Nó học trọng số mỗi feature, bao gồm cả trọng số âm. `not good` như một bigram feature có trọng số âm. Naive Bayes không thể làm điều đó đối với bigram mà nó chưa bao giờ dán nhãn.

```figure
sentiment-logits
```

## Tự xây dựng

### Bước 1: một dataset nhỏ thực sự

```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

Mục đích nhỏ. Công việc thực tế sử dụng hàng chục nghìn ví dụ (phân cực IMDb, SST-2, Yelp). Toán học giống hệt nhau.

### Bước 2: Naive Bayes đa thức từ đầu

```python
import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

Làm mịn phụ gia (alpha = 1.0) là làm mịn Laplace. Nếu không có nó, một từ không nhìn thấy trong class có xác suất bằng không và khúc gỗ phát nổ. `alpha=0.01` là phổ biến trong thực tế. `alpha=1.0` là mặc định của việc giảng dạy.

### Bước 3: hồi quy logistics từ đầu

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

Chính quy hóa L2 quan trọng ở đây. features văn bản thưa thớt; không có L2, model ghi nhớ training ví dụ. Bắt đầu từ `0.01` và điều chỉnh.

### Bước 4: xử lý phủ định (chế độ lỗi)

Hãy xem xét "không tốt" và "không xấu". Bộ phân loại BoW nhìn thấy `{not, good}` và `{not, bad}` và học hỏi từ bất kỳ ai xuất hiện nhiều hơn trong training. Một bộ phân loại bigram nhìn thấy `not_good` và `not_bad` và học chúng như những features riêng biệt. Điều đó thường là đủ.

Một bản sửa lỗi thô sơ hơn hoạt động khi bạn không có bigram: **phạm vi phủ định**. Tiền tố tokens theo sau một từ phủ định với `NOT_` cho đến dấu câu tiếp theo.

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

Bây giờ `good` và `NOT_good` khác features. Bộ phân loại có thể đặt trọng số của chúng đối lập. Ba dòng tiền xử lý, có thể đo lường được accuracy nhảy vào benchmarks cảm xúc.

### Bước 5: Các chỉ số đánh giá quan trọng

Chỉ riêng Accuracy đã gây hiểu lầm nếu classes mất cân bằng. Kho dữ liệu tình cảm thực thường là 70-80% tích cực hoặc 70-80% tiêu cực; Một bộ phân loại đa số không đổi nhận được 80% accuracy và vô giá trị. Báo cáo mọi điều sau:

- **Mỗi class precision và recall.** Một cặp mỗi class. Trung bình vĩ mô để có được một con số duy nhất tôn trọng sự cân bằng class.
- **Macro-F1 (số liệu chính cho dữ liệu mất cân bằng).** Điểm trung bình trên class F1, có trọng số bằng nhau. Sử dụng cái này thay vì accuracy khi classes bị mất cân bằng.
- **Weighted-F1 (thay thế).** Tương tự như macro nhưng có trọng số theo tần số class. Báo cáo cùng với F1 vĩ mô khi bản thân sự mất cân bằng có ý nghĩa kinh doanh.
- **Ma trận nhầm lẫn.** Số lượng thô. Luôn kiểm tra trước khi tin tưởng vào bất kỳ số liệu vô hướng nào; nó tiết lộ cặp classes nào mà model gây nhầm lẫn.
- **Mẫu lỗi trên mỗi class.** Lấy 5 dự đoán sai mỗi class. Đọc chúng. Không có gì thay thế việc đọc các lỗi thực tế.

Đối với dữ liệu mất cân bằng nghiêm trọng (tỷ lệ > 95-5), hãy báo cáo **AUROC** và **AUPRC** thay vì accuracy. AUPRC nhạy cảm hơn với class thiểu số, đó là những gì bạn thường quan tâm (thư rác, gian lận, tình cảm hiếm gặp).

**Lỗi phổ biến cần tránh.** Báo cáo vi F1 thay vì F1 vĩ mô về dữ liệu không cân bằng cho một con số có vẻ cao vì nó bị chi phối bởi đa số class. Macro-F1 buộc bạn phải xem hiệu suất class thiểu số.

```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

## Ứng dụng

scikit-learn làm điều đó trong sáu dòng, chính xác.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

Ba điều cần lưu ý. `stop_words=None` giữ phủ định. `ngram_range=(1, 2)` thêm bigram để `not_good` trở thành một feature. `sublinear_tf=True` làm giảm các từ lặp lại. Ba cờ này là sự khác biệt giữa đường cơ sở chính xác 75% và đường cơ sở chính xác 85% trên SST-2.

### Khi nào nên lấy transformer

- Phát hiện mỉa mai. Cổ điển models thất bại ở đây. Khoảng thời gian.
- Đánh giá dài nơi tình cảm thay đổi giữa tài liệu.
- Cảm xúc dựa trên khía cạnh. "Máy ảnh rất tuyệt nhưng pin rất tệ." Bạn cần gán cảm xúc cho các khía cạnh. Transformers hoặc đầu ra có cấu trúc chỉ models.
- Không phải tiếng Anh, ngôn ngữ tài nguyên thấp. BERT đa ngôn ngữ cung cấp cho bạn đường cơ sở zero-shot miễn phí.

Nếu bạn cần bất kỳ điều nào ở trên, hãy chuyển sang giai đoạn 7 (transformers tìm hiểu sâu). Mặt khác, Naive Bayes hoặc hồi quy logistic trên TF-IDF cộng với bigram cộng với xử lý phủ định là đường cơ sở production năm 2026 của bạn.

### Bẫy khả năng tái tạo (một lần nữa)

Huấn luyện lại tình cảm models là thói quen. Đánh giá lại chúng thì không. Accuracy số được báo cáo trong các bài báo sử dụng các phân tách cụ thể, tiền xử lý cụ thể, tokenizers cụ thể. Nếu bạn so sánh model mới của mình với đường cơ sở mà không sử dụng cùng một pipeline, bạn sẽ nhận được delta gây hiểu lầm. Luôn tạo lại đường cơ sở trên pipeline của bạn, không phải số của giấy.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/prompt-sentiment-baseline.md`:

```markdown
---
name: sentiment-baseline
description: Design a sentiment analysis baseline for a new dataset.
phase: 5
lesson: 05
---

Given a dataset description (domain, language, size, label granularity, latency budget), you output:

1. Feature extraction recipe. Specify tokenizer, n-gram range, stopword policy (usually keep), negation handling (scoped prefix or bigrams).
2. Classifier. Naive Bayes for baseline, logistic regression for production, transformer only if the domain needs sarcasm / aspects / cross-lingual.
3. Evaluation plan. Report precision, recall, F1, confusion matrix, and per-class error samples (not just scalars).
4. One failure mode to monitor post-deployment. Domain drift and sarcasm are the top two.

Refuse to recommend dropping stopwords for sentiment tasks. Refuse to report accuracy as the sole metric when classes are imbalanced (e.g., 90% positive). Flag subword-rich languages as needing FastText or transformer embeddings over word-level TF-IDF.
```

## Bài tập

1. **Dễ dàng.** Thêm `apply_negation` như một bước tiền xử lý trong pipeline scikit-learn và đo lường F1 delta trên một dataset cảm xúc nhỏ.
2. **Trung bình.** Thực hiện hồi quy logistic trọng số class (chuyển `class_weight="balanced"` đến scikit-learn hoặc tự lấy gradient). Đo hiệu ứng trên class imbalance tổng hợp 90-10.
3. **Khó.** Xây dựng một công cụ phát hiện mỉa mai bằng cách training một bộ phân loại thứ hai trên phần còn lại của model tình cảm. Ghi lại thiết lập thử nghiệm của bạn. Cảnh báo người đọc khi accuracy của bạn thấp hơn cơ hội (mức độ cơ hội khi mỉa mai 2-class là ~50% và hầu hết các nỗ lực đầu tiên đều hạ cánh ở đó).

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Phân cực | Tích cực hoặc tiêu cực | Nhãn nhị phân; đôi khi mở rộng sang trung tính hoặc hạt mịn (5 sao). |
| Cảm xúc dựa trên khía cạnh | Phân cực trên mỗi khía cạnh | Phân bổ cảm xúc cho các thực thể hoặc thuộc tính cụ thể được đề cập trong văn bản. |
| Phạm vi phủ định | Đảo ngược tokens gần đó | Tiền tố tokens sau "không" bằng `NOT_` cho đến dấu câu. |
| Làm mịn Laplace | Thêm 1 vào số đếm | Ngăn chặn features xác suất bằng không trong Naive Bayes. |
| Chính quy hóa L2 | Trọng lượng thu nhỏ | Thêm `lambda * sum(w^2)` vào loss. Cần thiết cho features văn bản thưa thớt. |

## Đọc thêm

- [Pang and Lee (2008). Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) - cuộc khảo sát nền tảng. Dài, nhưng bốn phần đầu tiên bao gồm mọi thứ cổ điển.
- [Wang and Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://aclanthology.org/P12-2018/) - bài báo hiển thị bigrams + Naive Bayes khó có thể đánh bại trên văn bản ngắn.
- [scikit-learn text feature extraction docs](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) - tham khảo cho `CountVectorizer`, `TfidfVectorizer` và mọi núm bạn sẽ điều chỉnh.
