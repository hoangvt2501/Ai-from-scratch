# Bayes ngây thơ

> Giả định "ngây thơ" là sai, và dù sao nó cũng hoạt động. Đó là vẻ đẹp của nó.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 2, Bài 01-07 (phân loại, định lý Bayes)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Triển khai Multinomial Naive Bayes từ đầu với tính năng làm mịn Laplace để phân loại văn bản
- Giải thích tại sao giả định độc lập ngây thơ là sai về mặt toán học nhưng lại tạo ra thứ hạng class chính xác trong thực tế
- So sánh các biến thể Đa thức, Bernoulli và Gaussian Naive Bayes và chọn biến thể phù hợp cho một loại feature nhất định
- Đánh giá Naive Bayes so với hồi quy logistic trên high-dimensional dữ liệu thưa thớt và giải thích sự đánh đổi bias-variance tại nơi làm việc

## Vấn đề

Bạn cần phân loại văn bản. Email thành thư rác hoặc không phải thư rác. Đánh giá của khách hàng theo hướng tích cực hoặc tiêu cực. Hỗ trợ ticket thành các danh mục. Bạn có hàng nghìn features (một dữ liệu mỗi từ) và training dữ liệu hạn chế.

Hầu hết các bộ phân loại đều nghẹt thở ở đây. Hồi quy logistic cần đủ mẫu để ước tính hàng nghìn trọng lượng một cách đáng tin cậy. Cây quyết định chia thành từng từ một và quá phù hợp. KNN trong 10.000 chiều là vô nghĩa vì mọi điểm đều cách xa mọi điểm khác như nhau.

Naive Bayes xử lý điều này. Nó đưa ra một giả định sai về mặt toán học (rằng mọi feature đều độc lập với mọi feature khác được đưa ra class), và nó vẫn vượt trội hơn models "thông minh hơn" về phân loại văn bản, đặc biệt là với các bộ training nhỏ. Nó huấn luyện trong một lần chuyển qua dữ liệu. Nó mở rộng đến hàng triệu features. Nó tạo ra các ước tính xác suất (mặc dù thường được hiệu chỉnh kém do giả định độc lập).

Hiểu được lý do tại sao một giả định sai dẫn đến dự đoán tốt sẽ dạy cho bạn một điều cơ bản về máy học: model tốt nhất không phải là đúng nhất, nó là giả định có sự đánh đổi variance bias tốt nhất cho dữ liệu của bạn.

## Khái niệm

### Định lý Bayes (Đánh giá nhanh)

Định lý Bayes lật ngược xác suất có điều kiện:

```
P(class | features) = P(features | class) * P(class) / P(features)
```

Chúng ta muốn `P(class | features)` - xác suất một tài liệu thuộc về một class cho các từ trong đó. Chúng ta có thể tính toán điều này từ:
- `P(features | class)` - likelihood của việc nhìn thấy những từ này trong các tài liệu của class này
- `P(class)` -- xác suất prior của class (thư rác nói chung phổ biến như thế nào?)
- `P(features)` -- bằng chứng, giống nhau đối với tất cả các classes, vì vậy chúng ta có thể bỏ qua nó khi so sánh

class có `P(class | features)` cao nhất sẽ giành chiến thắng.

### Giả định độc lập ngây thơ

Tính toán `P(features | class)` chính xác đòi hỏi phải ước tính xác suất chung của tất cả features với nhau. Với vốn từ vựng 10.000 từ, bạn sẽ cần ước tính sự phân bố trên 2^10.000 kết hợp có thể. Không thể.

Giả định ngây thơ: mọi feature đều độc lập có điều kiện với class.

```
P(w1, w2, ..., wn | class) = P(w1 | class) * P(w2 | class) * ... * P(wn | class)
```

Thay vì một phân phối chung bất khả thi, bạn ước tính n phân phối đơn giản trên mỗi feature. Mỗi người chỉ cần một số đếm.

Giả định này rõ ràng là sai. Các từ "máy móc" và "học tập" không độc lập trong bất kỳ tài liệu nào. Nhưng bộ phân loại không cần ước tính xác suất chính xác. Nó cần xếp hạng chính xác - class nào có xác suất cao nhất. Giả định độc lập đưa ra các lỗi có hệ thống, nhưng những lỗi đó ảnh hưởng đến tất cả classes tương tự, vì vậy xếp hạng vẫn đúng.

### Tại sao nó vẫn hoạt động

Ba lý do:

1. **Xếp hạng trên hiệu chuẩn.** Phân loại chỉ cần class xếp hạng cao nhất là chính xác. Ngay cả khi P (spam) = 0,99999 khi xác suất thực là 0,7, bộ phân loại vẫn chọn thư rác một cách chính xác. Chúng ta không cần xác suất chính xác. Chúng ta cần người chiến thắng chính xác.

2. **bias cao, variance thấp.** Giả định độc lập là một prior mạnh mẽ. Nó hạn chế model rất nhiều, ngăn cản overfitting. Với dữ liệu training hạn chế, một model hơi sai nhưng ổn định sẽ đánh bại một model đúng về mặt lý thuyết nhưng cực kỳ không ổn định. Đây là sự đánh đổi bias-variance đang hoạt động.

3. **Feature sự dư thừa bị hủy bỏ.** Tương quan features cung cấp bằng chứng dư thừa. Bộ phân loại đếm hai lần bằng chứng này, nhưng nó cũng đếm hai lần cho class chính xác. Nếu "máy móc" và "học tập" luôn xuất hiện cùng nhau, cả hai đều cung cấp bằng chứng cho class "công nghệ". NB đếm chúng hai lần, nhưng nó đếm chúng hai lần cho đúng class.

Lý do thứ tư, thực tế: Naive Bayes cực kỳ nhanh. Training là một lần đi qua tần số đếm dữ liệu. Dự đoán là một phép nhân ma trận. Bạn có thể huấn luyện trên một triệu tài liệu trong vài giây. Tốc độ này có nghĩa là bạn có thể lặp lại nhanh hơn, thử nhiều bộ feature hơn và chạy nhiều thử nghiệm hơn so với models chậm hơn.

### Toán học từng bước

Chúng ta hãy trace qua một ví dụ cụ thể. Giả sử chúng ta có hai classes: thư rác và không thư rác. Từ vựng của chúng tôi có ba từ: "miễn phí", "tiền", "gặp gỡ".

Training dữ liệu:
- Email spam đề cập đến "miễn phí" 80 lần, "tiền" 60 lần, "họp" 10 lần (tổng cộng 150 từ)
- Email không spam đề cập đến "miễn phí" 5 lần, "tiền" 10 lần, "họp" 100 lần (tổng cộng 115 từ)
- 40% email là spam, 60% không phải là spam

Với làm mịn Laplace (alpha = 1):

```
P(free | spam)    = (80 + 1) / (150 + 3) = 81/153 = 0.529
P(money | spam)   = (60 + 1) / (150 + 3) = 61/153 = 0.399
P(meeting | spam) = (10 + 1) / (150 + 3) = 11/153 = 0.072

P(free | not-spam)    = (5 + 1) / (115 + 3) = 6/118 = 0.051
P(money | not-spam)   = (10 + 1) / (115 + 3) = 11/118 = 0.093
P(meeting | not-spam) = (100 + 1) / (115 + 3) = 101/118 = 0.856
```

Email mới chứa: "miễn phí" (2 lần), "tiền" (1 lần), "họp" (0 lần).

```
log P(spam | email) = log(0.4) + 2*log(0.529) + 1*log(0.399) + 0*log(0.072)
                    = -0.916 + 2*(-0.637) + (-0.919) + 0
                    = -3.109

log P(not-spam | email) = log(0.6) + 2*log(0.051) + 1*log(0.093) + 0*log(0.856)
                        = -0.511 + 2*(-2.976) + (-2.375) + 0
                        = -8.838
```

Thư rác thắng với tỷ lệ chênh lệch lớn. Từ "miễn phí" xuất hiện hai lần là bằng chứng mạnh mẽ cho thư rác. Lưu ý rằng "meeting" không xuất hiện đóng góp bằng không cho cả hai tổng log (0 * log(P)) -- trong NB đa thức, các từ vắng mặt không có tác dụng. Chính Bernoulli NB đã models rõ ràng sự vắng mặt của từ.

### Ba biến thể

Naive Bayes có ba hương vị. Mỗi models `P(feature | class)` khác nhau.

#### Bayes ngây thơ đa thức

Models mỗi feature dưới dạng số đếm. Tốt nhất cho dữ liệu văn bản trong đó features tần số từ hoặc giá trị TF-IDF.

```
P(word_i | class) = (count of word_i in class + alpha) / (total words in class + alpha * vocab_size)
```

`alpha` là làm mịn Laplace (giải thích bên dưới). Biến thể này là con ngựa làm việc để phân loại văn bản.

#### Bayes ngây thơ của Gaussian

Models mỗi feature dưới dạng phân phối chuẩn. Tốt nhất cho features liên tục.

```
P(x_i | class) = (1 / sqrt(2 * pi * var)) * exp(-(x_i - mean)^2 / (2 * var))
```

Mỗi class có giá trị trung bình và variance riêng trên mỗi feature. Điều này hoạt động tốt khi features thực sự tuân theo một đường cong hình chuông trong mỗi class.

#### Bernoulli ngây thơ Bayes

Models mỗi feature dưới dạng nhị phân (có hoặc vắng mặt). Tốt nhất cho văn bản ngắn hoặc feature vectors nhị phân.

```
P(word_i | class) = (docs in class containing word_i + alpha) / (total docs in class + 2 * alpha)
```

Không giống như Đa thức, Bernoulli rõ ràng phạt việc không có một từ. Nếu "miễn phí" thường xuất hiện trong thư rác nhưng không có trong email này, Bernoulli coi đó là bằng chứng chống lại thư rác.

### Trường hợp sử dụng từng mẫu mã

| Biến thể | Loại Feature | Tốt nhất cho | Ví dụ |
|---------|-------------|----------|---------|
| Đa thức | Số lượng hoặc tần số | Phân loại văn bản, túi từ | Thư rác, phân loại chủ đề |
| Gaussian | Giá trị liên tục | Dữ liệu dạng bảng với features bình thường | Phân loại mống mắt, dữ liệu cảm biến |
| Bernoulli | Nhị phân (0/1) | Văn bản ngắn, feature vectors nhị phân | Thư rác SMS, presence/absence features |

### Làm mịn Laplace

Điều gì xảy ra khi một từ xuất hiện trong dữ liệu thử nghiệm nhưng không bao giờ xuất hiện trong dữ liệu training cho một class cụ thể?

Không làm mịn: `P(word | class) = 0/N = 0`. Một số không nhân qua toàn bộ sản phẩm tạo ra `P(class | features) = 0`, bất kể tất cả các bằng chứng khác. Một từ vô hình duy nhất sẽ phá hủy toàn bộ dự đoán, bất kể có bao nhiêu bằng chứng khác hỗ trợ nó.

Làm mịn Laplace thêm một `alpha` đếm nhỏ (thường là 1) cho mỗi feature đếm:

```
P(word_i | class) = (count(word_i, class) + alpha) / (total_words_in_class + alpha * vocab_size)
```

Với alpha = 1, mỗi từ có ít nhất một xác suất nhỏ. Từ "discombobulate" xuất hiện trong email thử nghiệm không còn giết chết xác suất spam nữa. Làm mịn có cách giải thích Bayes: nó tương đương với việc đặt một prior Dirichlet đồng nhất trên phân phối từ.

Alpha cao hơn có nghĩa là làm mịn mạnh hơn (phân bố đồng đều hơn). Alpha thấp hơn có nghĩa là model tin tưởng dữ liệu hơn. Alpha là một hyperparameter bạn điều chỉnh.

Tác dụng của alpha:

| Alpha | Hiệu ứng | Trường hợp sử dụng |
|-------|--------|-------------|
| 0.001 | Hầu như không làm mịn, tin tưởng vào dữ liệu | Bộ training rất lớn, không có features vô hình nào được mong đợi |
| 0.1 | Làm mịn ánh sáng | Bộ training lớn |
| 1.0 | Làm mịn Laplace tiêu chuẩn | Điểm bắt đầu mặc định |
| 10.0 | Làm mịn nặng, làm phẳng phân phối | Rất nhỏ training bộ, nhiều điều chưa từng thấy features mong đợi |

### Tính toán không gian nhật ký

Nhân hàng trăm xác suất (mỗi xác suất nhỏ hơn 1) gây ra dòng chảy dấu phẩy động. Tích trở thành số không trong dấu phẩy động mặc dù giá trị thực là một số dương rất nhỏ.

Giải pháp: làm việc trong log space. Thay vì nhân xác suất, hãy thêm logarit của chúng:

```
log P(class | x1, x2, ..., xn) = log P(class) + sum_i log P(xi | class)
```

Điều này biến dự đoán thành một sản phẩm chấm:

```
log_scores = X @ log_feature_probs.T + log_class_priors
prediction = argmax(log_scores)
```

Phép nhân ma trận. Đó là lý do tại sao dự đoán Naive Bayes rất nhanh - nó là hoạt động tương tự như một model tuyến tính một lớp.

### Naive Bayes so với hồi quy logistic

Cả hai đều là bộ phân loại tuyến tính cho văn bản. Sự khác biệt là ở những gì họ model.

| Khía cạnh | Bayes ngây thơ | Hồi quy logistic |
|--------|------------|-------------------|
| Kiểu | Tổng quát (models P (X \ | Y)) | Phân biệt đối xử (models P(Y\ | X)) |
| Training | Đếm tần số | Tối ưu hóa chức năng loss |
| Dữ liệu nhỏ | Tốt hơn (mạnh mẽ prior giúp) | Tệ hơn (không đủ để ước tính trọng lượng) |
| Dữ liệu lớn | Tệ hơn (giả định sai sẽ gây tổn thương) | Tốt hơn (ranh giới linh hoạt) |
| Features | Giả định độc lập | Xử lý các mối tương quan |
| Tốc độ | Vượt qua một lần, rất nhanh | Tối ưu hóa lặp lại |
| Hiệu chuẩn | Xác suất kém | Xác suất tốt hơn |

Quy tắc ngón tay cái: bắt đầu với Naive Bayes. Nếu bạn có đủ dữ liệu và ổn định NB, hãy chuyển sang hồi quy logistic.

### Phân loại Pipeline

```mermaid
flowchart LR
    A[Raw Text] --> B[Tokenize]
    B --> C[Build Vocabulary]
    C --> D[Count Word Frequencies]
    D --> E[Apply Smoothing]
    E --> F[Compute Log Probabilities]
    F --> G[Predict: argmax P class given words]

    style A fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333
```

Trong thực tế, chúng tôi làm việc log space để tránh dòng chảy dấu phẩy động. Thay vì nhân nhiều xác suất nhỏ, chúng ta thêm logarit của chúng:

```
log P(class | features) = log P(class) + sum_i log P(feature_i | class)
```

```figure
naive-bayes
```

## Tự xây dựng

Mã trong `code/naive_bayes.py` triển khai cả MultinomialNB và GaussianNB từ đầu.

### Đa thứcNB

Việc triển khai từ đầu:

1. **phù hợp (X, y)**: Đối với mỗi class, hãy đếm tần số của từng feature. Thêm làm mịn Laplace. Tính toán log probabilities. Lưu trữ class priors (nhật ký tần số class).

2. **predict_log_proba(X)**: Đối với mỗi mẫu, tính log P(class) + tổng của log P(feature_i | class) cho tất cả các classes. Đây là phép nhân ma trận: X @ log_probs. T + log_priors.

3. **predict(X)**: Trả về class có log probability cao nhất.

```python
class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]

        self.classes_ = classes
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            counts = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob_[i] = np.log(counts / counts.sum())

        return self
```

Thông tin chi tiết quan trọng: sau khi khớp, dự đoán chỉ là phép nhân ma trận cộng với một bias. Đây là lý do tại sao Naive Bayes rất nhanh.

### GaussianNB

Đối với features liên tục, chúng tôi ước tính trung bình và variance trên class mỗi feature:

```python
class GaussianNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        self.means_ = np.zeros((len(classes), X.shape[1]))
        self.vars_ = np.zeros((len(classes), X.shape[1]))
        self.priors_ = np.zeros(len(classes))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i] = X_c.var(axis=0) + 1e-9
            self.priors_[i] = X_c.shape[0] / X.shape[0]

        return self
```

Dự đoán sử dụng PDF Gaussian trên mỗi feature, nhân với features (được thêm vào log space).

### Demo: Phân loại văn bản

Mã tạo ra dữ liệu túi từ tổng hợp mô phỏng hai classes (bài báo công nghệ và bài báo thể thao). Mỗi class có một phân bố tần suất từ khác nhau. MultinomialNB phân loại chúng bằng cách sử dụng số lượng từ.

Dữ liệu tổng hợp hoạt động như thế này: chúng tôi tạo ra 200 "từ" (feature cột). Từ 0-39 có tần suất cao trong các bài báo công nghệ và thấp trong thể thao. Các từ 80-119 có tần suất cao trong thể thao và thấp trong công nghệ. Các từ 40-79 có tần số trung bình trong cả hai. Điều này tạo ra một kịch bản thực tế trong đó một số từ mạnh class chỉ báo và những từ khác là nhiễu.

### Demo: Features liên tục

Mã tạo ra dữ liệu giống như Iris (3 classes, 4 features, cụm Gaussian). GaussianNB phân loại bằng cách sử dụng giá trị trung bình trên class và variance. Mỗi class có một trung tâm khác nhau (vector trung bình) và độ lan truyền khác nhau (variance), bắt chước dữ liệu trong thế giới thực, nơi các phép đo khác nhau một cách có hệ thống giữa các danh mục.

Mã cũng minh họa:
- **So sánh làm mịn:** Training MultinomialNB với các giá trị alpha khác nhau để thể hiện hiệu quả của độ bền làm mịn đối với accuracy.
- **Training nghiệm kích thước: **NB accuracy cải thiện như thế nào khi dữ liệu training tăng từ 20 lên 1600 mẫu. NB đạt accuracy tốt ngay cả với rất ít mẫu - đây là ưu điểm chính của nó.
- **Ma trận nhầm lẫn:** Mỗi class precision, recall và F1 score để chỉ ra nơi NB mắc lỗi.

### Tốc độ dự đoán

Dự đoán Naive Bayes là một phép nhân ma trận. Đối với n mẫu có d features và k classes:
- Đa thứcNB: một ma trận nhân (n x d) @ (d x k) = O (n * d * k)
- GaussianNB: n * k Đánh giá PDF của Gaussian, mỗi đánh giá trên d features = O (n * d * k)

Cả hai đều tuyến tính trong mọi chiều. So sánh điều này với KNN (yêu cầu tính toán khoảng cách đến tất cả các điểm training) hoặc SVM với hạt nhân RBF (yêu cầu đánh giá hạt nhân dựa trên tất cả các vectors hỗ trợ). NB nhanh hơn theo thứ tự độ lớn tại thời điểm dự đoán.

## Ứng dụng

Với sklearn, cả hai biến thể đều là một dòng:

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB accuracy: {gnb.score(X_test, y_test):.3f}")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)
print(f"MultinomialNB accuracy: {mnb.score(X_test_counts, y_test):.3f}")
```

Đối với phân loại văn bản với sklearn:

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB(alpha=1.0)),
])

text_clf.fit(train_texts, train_labels)
accuracy = text_clf.score(test_texts, test_labels)
```

Mã trong `naive_bayes.py` so sánh các triển khai từ đầu với sklearn trên cùng một dữ liệu để xác minh tính đúng đắn.

### TF-IDF với Naive Bayes

Số lượng từ thô cung cấp cho mỗi từ có trọng lượng bằng nhau cho mỗi lần xuất hiện. Nhưng những từ phổ biến như "the" và "is" xuất hiện thường xuyên trong mọi class - chúng không mang thông tin. TF-IDF (Tần suất thuật ngữ - Tần suất tài liệu nghịch đảo) giảm trọng số các từ phổ biến và tăng trọng số các từ hiếm, phân biệt đối xử.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB(alpha=0.1)),
])
```

Các giá trị TF-IDF không âm, vì vậy chúng hoạt động với MultinomialNB. Sự kết hợp của TF-IDF + MultinomialNB là một trong những đường cơ sở mạnh nhất để phân loại văn bản. Nó thường đánh bại các models phức tạp hơn trên datasets với ít hơn 10.000 mẫu training.

### BernoulliNB cho văn bản ngắn

Đối với văn bản ngắn (tweet, SMS, tin nhắn trò chuyện), BernoulliNB có thể vượt trội hơn MultinomialNB. Văn bản ngắn có số lượng từ thấp, vì vậy thông tin tần số mà MultinomialNB dựa vào là nhiễu. BernoulliNB chỉ quan tâm đến sự hiện diện hay vắng mặt, điều này đáng tin cậy hơn với văn bản ngắn.

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

text_clf = Pipeline([
    ("vectorizer", CountVectorizer(binary=True)),
    ("classifier", BernoulliNB(alpha=1.0)),
])
```

Cờ `binary=True` trong CountVectorizer chuyển đổi tất cả các số đếm thành 0/1. Nếu không có nó, BernoulliNB vẫn hoạt động nhưng đang thấy các số đếm mà nó không được thiết kế.

### Hiệu chỉnh xác suất NB

Xác suất NB được hiệu chỉnh kém. Khi NB nói P (spam) = 0,95, xác suất thực sự có thể là 0,7. Nếu bạn cần ước tính xác suất đáng tin cậy (ví dụ: để đặt ngưỡng hoặc kết hợp với các models khác), hãy sử dụng CalibratedClassifierCV của sklearn:

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_nb = CalibratedClassifierCV(MultinomialNB(), cv=5, method="sigmoid")
calibrated_nb.fit(X_train, y_train)
proba = calibrated_nb.predict_proba(X_test)
```

Điều này phù hợp với hồi quy logistic trên điểm thô của NB bằng cách sử dụng xác thực chéo. Xác suất kết quả gần với tần số class thực hơn nhiều.

### Những sai lầm phổ biến

1. **Giá trị feature âm.** MultinomialNB yêu cầu features không âm. Nếu bạn có giá trị âm (như TF-IDF với một số cài đặt nhất định hoặc features tiêu chuẩn), hãy sử dụng GaussianNB để thay thế hoặc chuyển features thành dương.

2. **Zero variance features.** GaussianNB chia cho variance. Nếu một feature có variance không cho một class (tất cả các giá trị giống hệt nhau), phép tính xác suất sẽ gặp lỗi. Mã thêm một thuật ngữ làm mịn nhỏ (1e-9) cho tất cả các phương sai để ngăn chặn điều này.

3. **Class imbalance.** Nếu 99% email không phải là thư rác, prior P (không phải thư rác) = 0,99 mạnh đến mức nó lấn át bằng chứng likelihood. Bạn có thể đặt class priors theo cách thủ công hoặc sử dụng class_prior parameter trong sklearn.

4. **Feature tỷ lệ.** MultinomialNB không cần chia tỷ lệ (nó hoạt động trên số đếm). GaussianNB cũng không cần mở rộng quy mô (nó ước tính số liệu thống kê trên mỗi feature). Đây là một lợi thế so với hồi quy logistic và SVM, nhạy cảm với quy mô feature.

## Sản phẩm bàn giao

Bài học này tạo ra:
- `outputs/skill-naive-bayes-chooser.md` - một quyết định skill để chọn đúng biến thể NB
- `code/naive_bayes.py` -- MultinomialNB và GaussianNB từ đầu, với so sánh sklearn

### Khi Bayes ngây thơ thất bại

NB không thành công khi giả định độc lập gây ra thứ hạng không chính xác (không chỉ xác suất không chính xác). Điều này xảy ra khi:

1. **Tương tác feature mạnh.** Nếu class phụ thuộc vào sự kết hợp của hai features chứ không phải một trong hai (các mẫu giống XOR), NB sẽ bỏ lỡ nó hoàn toàn. Mỗi feature không cung cấp bằng chứng và NB không thể kết hợp chúng một cách phi tuyến tính.

2. **Tương quan cao features với bằng chứng đối lập.** Nếu feature A nói "spam" và feature B nói "không phải spam", nhưng A và B có mối tương quan hoàn hảo (chúng luôn đồng ý trong thực tế), NB sẽ thấy bằng chứng mâu thuẫn khi không có.

3. **Tập training rất lớn.** Với đủ dữ liệu, các models phân biệt như hồi quy logistic sẽ tìm hiểu ranh giới quyết định thực sự và vượt trội hơn NB. Giả định độc lập đã giúp ích cho dữ liệu nhỏ giờ đây đã kìm hãm model.

Trong thực tế, các chế độ lỗi này rất hiếm khi để phân loại văn bản. Văn bản features rất nhiều, yếu riêng lẻ và các lỗi của giả định độc lập có xu hướng bị loại bỏ. Đối với dữ liệu dạng bảng có ít features tương quan mạnh, trước tiên hãy xem xét hồi quy logistic hoặc models dựa trên cây.

## Bài tập

1. **Thử nghiệm làm mịn.** Huấn luyện MultinomialNB trên dữ liệu văn bản với các giá trị alpha là 0,01, 0,1, 1,0, 10,0 và 100,0. Cốt truyện accuracy vs alpha. Hiệu suất đạt đỉnh ở đâu? Tại sao alpha rất cao lại đau?

2. **Feature kiểm tra tính độc lập.** Lấy một dataset văn bản thực. Chọn hai từ có mối tương quan rõ ràng ("máy móc" và "học tập"). Tính toán P (word1 | class) * P (từ2 | class) và so sánh với P(word1 VÀ word2 | class). Giả định độc lập sai như thế nào? Nó có ảnh hưởng đến việc phân loại accuracy không?

3. **Triển khai Bernoulli.** Mở rộng mã bằng class BernoulliNB. Chuyển đổi túi từ thành nhị phân (present/absent) và so sánh accuracy với MultinomialNB trên dữ liệu văn bản. Khi nào Bernoulli thắng?

4. **NB vs Hồi quy Logistic.** Huấn luyện cả về dữ liệu văn bản. Bắt đầu với 100 mẫu training và tăng lên 10.000. Cốt truyện accuracy so với kích thước training đặt cho cả hai. Hồi quy logistic vượt qua Naive Bayes tại thời điểm nào?

5. **Bộ lọc thư rác.** Xây dựng một bộ phân loại thư rác hoàn chỉnh: mã hóa văn bản email thô, xây dựng từ vựng, tạo features túi từ, huấn luyện MultinomialNB, đánh giá bằng precision và recall (không chỉ accuracy - tại sao?).

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|----------------------|
| Bayes ngây thơ | "Bộ phân loại xác suất đơn giản" | Một bộ phân loại áp dụng định lý Bayes với giả định rằng features độc lập có điều kiện với class |
| Độc lập có điều kiện | "Features không ảnh hưởng lẫn nhau" | P(A, B \ | C) = P (A \ | C) * P (B \ | C) -- biết B không cho bạn biết gì mới về A một khi bạn biết C |
| Làm mịn Laplace | "Làm mịn thêm một" | Thêm một số lượng nhỏ vào mỗi feature để ngăn xác suất bằng không chi phối dự đoán |
| Prior | "Những gì bạn tin tưởng trước khi xem dữ liệu" | P(class) -- xác suất của mỗi class trước khi quan sát bất kỳ features nào |
| Likelihood | "Dữ liệu phù hợp như thế nào" | P (features \ | class) -- xác suất quan sát các features này nếu class được biết |
| Posterior | "Những gì bạn tin sau khi xem dữ liệu" | P (class \ | features) -- xác suất cập nhật của class sau khi quan sát features |
| model tổng quát | "Models cách dữ liệu được tạo ra" | Một model học P(X \ | Y) và P(Y), sau đó sử dụng định lý Bayes để lấy P(Y \ | X) |
| Phân biệt đối xử model | "Models ranh giới quyết định" | Một model trực tiếp học P(Y \ | X) mà không mô hình hóa cách X được tạo ra |
| Log probability | "Tránh tràn ngập" | Làm việc với log P thay vì P để ngăn tích của nhiều số nhỏ trở thành số không trong dấu phẩy động |

## Đọc thêm

- [scikit-learn Naive Bayes docs](https://scikit-learn.org/stable/modules/naive_bayes.html) - cả ba biến thể với các chi tiết toán học
- [McCallum and Nigam, A Comparison of Event Models for Naive Bayes Text Classification (1998)](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf) -- so sánh cổ điển giữa Đa thức và Bernoulli cho văn bản
- [Rennie et al., Tackling the Poor Assumptions of Naive Bayes Text Classifiers (2003)](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf) -- cải tiến NB cho văn bản
- [Ng and Jordan, On Discriminative vs. Generative Classifiers (2001)](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf) - chứng minh NB hội tụ nhanh hơn LR với ít dữ liệu hơn
