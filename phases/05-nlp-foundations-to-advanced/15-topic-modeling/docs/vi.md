# Mô hình hóa chủ đề - LDA và BERTopic

> LDA: tài liệu là hỗn hợp của các chủ đề, chủ đề là sự phân phối trên các từ. BERTopic: cụm tài liệu trong không gian embedding, cụm là chủ đề. Cùng một mục tiêu, sự phân hủy khác nhau.

**Loại:** Học
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 5 · 02 (BoW + TF-IDF), Giai đoạn 5 · 03 (Word2Vec)
**Thời lượng:** ~45 phút

## Vấn đề

Bạn có 10.000 phiếu hỗ trợ khách hàng, 50.000 bài báo hoặc 200.000 tweet. Bạn cần biết bộ sưu tập nói về điều gì mà không cần đọc nó. Bạn không có danh mục được gắn nhãn. Bạn thậm chí không biết có bao nhiêu danh mục.

Mô hình hóa chủ đề trả lời điều đó mà không cần giám sát. Cung cấp cho nó một kho dữ liệu, lấy lại một tập hợp nhỏ các chủ đề mạch lạc và đối với mỗi tài liệu, một sự phân phối trên các chủ đề đó.

Hai họ thuật toán chiếm ưu thế. LDA (2003) coi mỗi tài liệu là một hỗn hợp của các chủ đề tiềm ẩn và mỗi chủ đề như một sự phân phối trên các từ. Inference là Bayes. Nó vẫn ships ở production nơi bạn cần các bài tập chủ đề thành viên hỗn hợp và phân phối xác suất cấp từ có thể giải thích được.

BERTopic (2020) mã hóa tài liệu bằng BERT, giảm kích thước với UMAP, cụm bằng HDBSCAN và trích xuất các từ chủ đề thông qua TF-IDF dựa trên class. Nó giành chiến thắng trên văn bản ngắn, phương tiện truyền thông xã hội và bất cứ thứ gì mà sự tương đồng về ngữ nghĩa quan trọng hơn sự chồng chéo từ ngữ. Một tài liệu có một chủ đề, đây là một giới hạn đối với nội dung dạng dài.

Bài học này xây dựng trực giác cho cả hai và đặt tên nên chọn cái nào cho một kho dữ liệu nhất định.

## Khái niệm

![LDA mixture model vs BERTopic clustering](../assets/topic-modeling.svg)

**Câu chuyện tổng quát LDA.** Mỗi chủ đề là một sự phân phối trên các từ. Mỗi tài liệu là một hỗn hợp của các chủ đề. Để tạo một từ trong tài liệu, hãy lấy mẫu một chủ đề từ hỗn hợp của tài liệu, sau đó lấy mẫu một từ từ phân phối của chủ đề đó. Inference đảo ngược điều này: cho các từ quan sát, suy ra sự phân phối chủ đề trên mỗi tài liệu và phân phối từ trên mỗi chủ đề. Gibbs sụp đổ sampling hoặc Bayes biến thể làm phép toán.

Đầu ra LDA chính:

- `doc_topic`: ma trận `(n_docs, n_topics)`, mỗi hàng tổng bằng 1 (hỗn hợp chủ đề của tài liệu).
- `topic_word`: ma trận `(n_topics, vocab_size)`, mỗi hàng tổng bằng 1 (phân phối từ của chủ đề).

**BERTopic pipeline.**

1. Mã hóa mỗi tài liệu bằng một câu transformer (ví dụ: `all-MiniLM-L6-v2`). vectors 384 mờ.
2. Giảm kích thước với UMAP xuống ~5 chiều. BERT embeddings quá mờ để phân cụm.
3. Cụm với HDBSCAN. Dựa trên mật độ, tạo ra các cụm có kích thước thay đổi và nhãn "ngoại lệ".
4. Đối với mỗi cụm, tính toán TF-IDF dựa trên class trên các tài liệu của cụm để trích xuất các từ hàng đầu.

Đầu ra là một chủ đề cho mỗi tài liệu (cộng với nhãn ngoại lệ -1). Tùy chọn, tư cách thành viên mềm thông qua xác suất của HDBSCAN vector.

## Tự xây dựng

### Bước 1: LDA qua scikit-learn

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names


def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"topic {idx}: {' '.join(words)}")
```

Lưu ý: các từ dừng đã bị xóa, min_df và max_df lọc các thuật ngữ hiếm và phổ biến, CountVectorizer (không phải TfidfVectorizer) vì LDA mong đợi số lượng thô.

### Bước 2: BERTopic (production)

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
```

Bộ lọc trên `Topic != -1` thả vùng lưu trữ ngoại lệ của BERTopic (tài liệu HDBSCAN không thể phân cụm). `min_topic_size` kiểm soát kích thước cụm tối thiểu của HDBSCAN; Mặc định thư viện của BERTopic là 10. Ví dụ này đặt nó thành 15 một cách rõ ràng cho thang điểm của bài học. Đối với kho dữ liệu trên 10.000 tài liệu, tăng lên 50 hoặc 100.

### Bước 3: đánh giá

Cả hai phương pháp đều xuất ra các từ chủ đề. Câu hỏi đặt ra là liệu những từ đó có gắn kết với nhau hay không.

- **Mạch lạc chủ đề (c_v).** Kết hợp NPMI (thông tin lẫn nhau theo điểm chuẩn hóa) của các cặp từ hàng đầu trên ngữ cảnh cửa sổ trượt, tổng hợp điểm số thành vectors chủ đề và so sánh những vectors đó thông qua sự tương đồng cosin. Cao hơn là tốt hơn. Sử dụng `gensim.models.CoherenceModel` với `coherence="c_v"`.
- **Đa dạng chủ đề.** Phần các từ duy nhất trên tất cả các từ hàng đầu của chủ đề. Cao hơn là tốt hơn (chủ đề không trùng lặp).
- **Kiểm tra định tính.** Đọc các từ hàng đầu của mỗi chủ đề. Họ có đặt tên cho một thứ có thật không? Sự phán xét của con người vẫn là tuyến phòng thủ cuối cùng.

## Khi nào nên chọn cái nào

| Tình huống | Chọn |
|-----------|------|
| Văn bản ngắn (tweet, đánh giá, tiêu đề) | BERTopic |
| Tài liệu dài với hỗn hợp chủ đề | LDA |
| Không GPU / điện toán hạn chế | LDA hoặc NMF |
| Cần phân phối nhiều chủ đề cấp tài liệu | LDA |
| Tích hợp LLM để gắn nhãn chủ đề | BERTopic (hỗ trợ trực tiếp) |
| Triển khai biên hạn chế tài nguyên | LDA |
| Sự gắn kết ngữ nghĩa tối đa | BERTopic |

Cân nhắc thực tế lớn nhất là độ dài tài liệu. BERT embeddings cắt bớt; LDA tính công việc trên bất kỳ độ dài nào. Đối với tài liệu dài hơn ngữ cảnh của embedding model, hãy chia đoạn + tổng hợp hoặc sử dụng LDA.

## Ứng dụng

stack năm 2026:

- **BERTopic.** Mặc định cho văn bản ngắn và bất kỳ thứ gì mà ngữ nghĩa quan trọng.
- **`gensim.models.LdaModel`.** LDA cổ điển dành cho production, trưởng thành, đã được thử nghiệm.
- **`sklearn.decomposition.LatentDirichletAllocation`.** LDA dễ dàng cho các thí nghiệm.
- **NMF.** Thừa số ma trận không âm. Thay thế nhanh cho LDA, chất lượng tương đương trên văn bản ngắn.
- **Top2Vec.** Thiết kế tương tự như BERTopic. Cộng đồng nhỏ hơn nhưng tốt ở một số benchmarks.
- **FASTopic.** Mới hơn, nhanh hơn BERTopic trên kho dữ liệu rất lớn.
- **Gắn nhãn dựa trên LLM.** Chạy bất kỳ phân cụm nào, sau đó prompt model để đặt tên cho từng cụm.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-topic-picker.md`:

```markdown
---
name: topic-picker
description: Pick LDA or BERTopic for a corpus. Specify library, knobs, evaluation.
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

Given a corpus description (document count, avg length, domain, language, compute budget), output:

1. Algorithm. LDA / NMF / BERTopic / Top2Vec / FASTopic. One-sentence reason.
2. Configuration. Number of topics: `recommended = max(5, round(sqrt(n_docs)))`, clamped to 200 for corpora under 40,000 docs; permit >200 only when the corpus is genuinely large (>40k) and note the increased compute cost. `min_df` / `max_df` filters and embedding model for neural approaches also belong here.
3. Evaluation. Topic coherence (c_v) via `gensim.models.CoherenceModel`, topic diversity, and a 20-sample human read.
4. Failure mode to probe. For LDA, "junk topics" absorbing stopwords and frequent terms. For BERTopic, the -1 outlier cluster swallowing ambiguous documents.

Refuse BERTopic on documents longer than the embedding model's context window without a chunking strategy. Refuse LDA on very short text (tweets, reviews under 10 tokens) as coherence collapses. Flag any n_topics choice below 5 as likely wrong; flag >200 on corpora under 40k docs as likely over-splitting.
```

## Bài tập

1. **Dễ dàng.** Phù hợp với LDA với 5 chủ đề trên 20 Nhóm tin tức dataset. In 10 từ hàng đầu cho mỗi chủ đề. Gắn nhãn từng chủ đề bằng tay. Thuật toán có tìm thấy các danh mục thực sự không?
2. **Trung bình.** Phù hợp với BERTopic trên cùng một tập hợp con 20 Nhóm tin tức. So sánh số lượng chủ đề được tìm thấy, từ hàng đầu và tính mạch lạc định tính so với LDA. Cái nào làm nổi bật các phạm trù thực sự rõ ràng hơn?
3. **Khó tính.** Tính toán c_v mạch lạc cho cả LDA và BERTopic trên kho dữ liệu của bạn. Chạy mỗi chủ đề với 5, 10, 20, 50 chủ đề. Tính mạch lạc của cốt truyện so với số lượng chủ đề. Báo cáo phương pháp nào ổn định hơn trên số lượng chủ đề.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Chủ đề | Một điều mà kho dữ liệu nói về | Phân phối xác suất trên các từ (LDA) hoặc một cụm tài liệu tương tự (BERTopic). |
| Thành viên hỗn hợp | Doc có nhiều chủ đề | LDA chỉ định mỗi tài liệu phân phối cho tất cả các chủ đề. |
| UMAP | Giảm kích thước | Học tập đa dạng bảo tồn cấu trúc địa phương; được sử dụng trong BERTopic. |
| Quét HDBSCAN | Phân cụm mật độ | Tìm các cụm có kích thước thay đổi; tạo ra nhãn "nhiễu" (-1) cho các giá trị ngoại lệ. |
| c_v mạch lạc | Chỉ số chất lượng chủ đề | Thông tin tương hỗ trung bình theo điểm của các từ chủ đề hàng đầu trong windows trượt. |

## Đọc thêm

- [Blei, Ng, Jordan (2003). Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) - bài báo LDA.
- [Grootendorst (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794) - bài báo BERTopic.
- [Röder, Both, Hinneburg (2015). Exploring the Space of Topic Coherence Measures](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf) - tờ báo giới thiệu c_v và bạn bè.
- [BERTopic documentation](https://maartengr.github.io/BERTopic/) — tài liệu tham khảo production. Ví dụ tuyệt vời.
