# Truy xuất và tìm kiếm thông tin

> BM25 chính xác nhưng giòn. Dense giăng một lưới rộng nhưng bỏ lỡ các từ khóa. Hybrid là mặc định năm 2026. Mọi thứ khác đều đang điều chỉnh.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 5 · 02 (BoW + TF-IDF), Giai đoạn 5 · 04 (GloVe, FastText, Subword)
**Thời lượng:** ~75 phút

## Vấn đề

Người dùng gõ "điều gì sẽ xảy ra nếu ai đó nói dối để lấy tiền" và hy vọng tìm thấy đạo luật thực sự bao gồm điều đó: "Mục 420 IPC". Tìm kiếm từ khóa hoàn toàn bỏ lỡ nó (không có từ vựng chung). Tìm kiếm ngữ nghĩa sẽ bỏ lỡ nếu embeddings không được huấn luyện về văn bản pháp lý. Tìm kiếm thực sự phải xử lý cả hai.

IR là pipeline trong mọi hệ thống RAG, mọi thanh tìm kiếm, tra cứu mờ của mọi trang web tài liệu. Kiến trúc năm 2026 hoạt động ở production không phải là một phương pháp duy nhất. Nó là một chuỗi các phương pháp bổ sung, mỗi phương pháp nắm bắt những thất bại của phương pháp trước đó.

Bài học này xây dựng từng phần và đặt tên những thất bại mà mỗi phần bắt được.

## Khái niệm

![Hybrid retrieval: BM25 + dense + RRF + cross-encoder rerank](../assets/retrieval.svg)

Bốn lớp. Chọn những cái bạn cần.

1. **Truy xuất thưa thớt (BM25).** Nhanh, chính xác trên các kết quả khớp chính xác, khủng khiếp về ngữ nghĩa. Chạy qua một chỉ mục đảo ngược. Dưới 10 mili giây cho mỗi truy vấn trên hàng triệu tài liệu. Mang lại cho bạn tài liệu tham khảo quy chế, mã sản phẩm, thông báo lỗi, đúng các thực thể được đặt tên.
2. **Truy xuất dày đặc.** Mã hóa truy vấn và tài liệu thành vectors. Tìm kiếm hàng xóm gần nhất. Nắm bắt được các diễn giải và sự tương đồng về ngữ nghĩa. Bỏ lỡ các từ khóa khớp chính xác khác nhau bởi một ký tự. 50-200ms cho mỗi truy vấn với FAISS hoặc vector DB.
3. **Fusion.** Merge danh sách được xếp hạng từ thưa thớt và dày đặc. Reciprocal Rank Fusion (RRF) là mặc định dễ dàng vì nó bỏ qua điểm thô (nằm ở các thang điểm khác nhau) và chỉ sử dụng các vị trí xếp hạng. Hợp nhất có trọng số là một tùy chọn khi bạn biết một tín hiệu chiếm ưu thế cho miền của mình.
4. **Cross-encoder xếp hạng lại.** Lấy top 30 từ hợp nhất. Chạy một encoder chéo (truy vấn + tài liệu cùng nhau, chấm điểm từng cặp). Giữ top-5. Cross-encoders chậm hơn trên mỗi cặp so với hai encoders nhưng chính xác hơn nhiều. Bạn khấu hao bằng cách chỉ chạy chúng trong top 30.

Truy xuất ba chiều (BM25 + dày đặc + học thưa như SPLADE) vượt trội hơn hai chiều vào năm 2026 benchmarks nhưng cần cơ sở hạ tầng cho các chỉ số đã học-thưa thớt. Đối với hầu hết các đội, xếp hạng lại hai chiều cộng với encoder chéo là điểm ngọt ngào.

## Tự xây dựng

### Bước 1: BM25 từ đầu

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

Hai parameters đáng biết. `k1=1.5` kiểm soát độ bão hòa tần số hạn; cao hơn có nghĩa là nhiều trọng số hơn về sự lặp lại của thuật ngữ. `b=0.75` kiểm soát chuẩn hóa chiều dài; 0 bỏ qua độ dài tài liệu, 1 chuẩn hóa hoàn toàn. Các giá trị mặc định là các khuyến nghị của Robertson từ bài báo gốc và hiếm khi cần điều chỉnh.

### Bước 2: truy xuất dày đặc với bi-encoder

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

L2-chuẩn hóa embeddings để tích chấm bằng cosin. `all-MiniLM-L6-v2` là 384-mờ, nhanh và đủ mạnh để hầu hết các truy xuất tiếng Anh. Đối với công việc đa ngôn ngữ, hãy sử dụng `paraphrase-multilingual-MiniLM-L12-v2`. Đối với accuracy hàng đầu, `bge-large-en-v1.5` hoặc `e5-large-v2`.

### Bước 3: Hợp nhất xếp hạng đối ứng

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

Hằng số `k=60` đến từ giấy RRF gốc. `k` cao hơn làm phẳng sự đóng góp của sự khác biệt cấp bậc; `k` thấp hơn làm cho các thứ hạng hàng đầu chiếm ưu thế. 60 là mặc định đã xuất bản và hiếm khi cần điều chỉnh.

### Bước 4: tìm kiếm kết hợp + xếp hạng lại

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

Ba giai đoạn được sáng tác. BM25 tìm thấy các kết quả khớp từ vựng. Dense tìm thấy các kết quả khớp ngữ nghĩa. RRF merges hai bảng xếp hạng mà không cần hiệu chỉnh điểm. Cross-encoder chấm điểm lại top 30 bằng cách sử dụng các cặp tài liệu truy vấn với nhau, điều này nắm bắt được mức độ liên quan chi tiết mà hai encoder đã bỏ lỡ. Giữ top-5.

### Bước 5: đánh giá

| Số liệu | Ý nghĩa |
|--------|---------|
| Recall@k | Trong số các truy vấn có tài liệu chính xác, tần suất nó xuất hiện trong top-k? |
| MRR (Xếp hạng đối ứng trung bình) | Trung bình 1/rank của tài liệu liên quan đầu tiên. |
| nDCG@k | Tính đến sự phân cấp liên quan, không chỉ là relevant/not. nhị phân |

Cụ thể RAG, **Recall@k** của săn mồi là con số quan trọng nhất. Người đọc của bạn không thể trả lời nếu đoạn văn đúng không có trong tập đã truy xuất.

Mẹo gỡ lỗi: đối với các truy vấn không thành công, hãy phân biệt thứ hạng thưa thớt và dày đặc. Nếu một tài liệu tìm thấy đúng và tài liệu kia thì không, bạn có từ vựng không khớp (sửa: thêm một nửa còn thiếu) hoặc mơ hồ ngữ nghĩa (sửa chữa: tốt hơn embeddings hoặc xếp hạng lại).

## Ứng dụng

stack năm 2026:

| Quy mô | Stack |
|-------|-------|
| Tài liệu 1K-100K | Trong bộ nhớ BM25 + `all-MiniLM-L6-v2` embeddings + RRF. Không có DB riêng biệt. |
| 100k-10 triệu tài liệu | FAISS hoặc pgvector cho dày đặc + Elasticsearch / OpenSearch cho BM25. Chạy song song. |
| 10 triệu + tài liệu | Qdrant / Weaviate / Vespa / Milvus với hỗ trợ hybrid. Cross-encoder xếp hạng lại trên top 30. |
| Biên giới chất lượng tốt nhất | Ba chiều (BM25 + dày đặc + SPLADE) + Xếp hạng lại tương tác muộn của ColBERT |

Bất cứ điều gì bạn chọn, ngân sách để đánh giá. Benchmark truy xuất recall trước khi đo điểm chuẩn RAG accuracy đầu cuối. Người đọc không thể sửa chữa những gì săn đã bỏ lỡ.

### Những bài học khó có được từ năm 2026 production RAG

- **80% lỗi RAG trace nhập và phân đoạn, không phải model.** Các nhóm dành hàng tuần để hoán đổi LLMs và điều chỉnh prompts trong khi truy xuất lặng lẽ trả về ngữ cảnh sai sau mỗi truy vấn thứ ba. Sửa lỗi phân đoạn trước.
- **Chiến lược phân đoạn quan trọng hơn kích thước chunk.** Phân tách kích thước cố định sẽ phá vỡ bảng, mã và tiêu đề lồng nhau. Nhận biết câu là mặc định; Phân đoạn ngữ nghĩa hoặc dựa trên LLM được đền đáp cho các tài liệu kỹ thuật và hướng dẫn sử dụng sản phẩm.
- **Mẫu phụ huynh-doc.** Truy xuất các đoạn "con" nhỏ cho precision. Khi nhiều con từ cùng một phần cha xuất hiện, hãy hoán đổi trong khối cha để giữ nguyên ngữ cảnh. Điều này liên tục nâng cao chất lượng câu trả lời mà không cần huấn luyện lại.
- **k_rerank=3 thường là tối ưu.** Mỗi đoạn bổ sung vượt qua sẽ làm tăng token chi phí và độ trễ phát điện mà không làm tăng chất lượng câu trả lời. Nếu k=8 vẫn tốt hơn k=3 đối với bạn, thì trình xếp hạng lại đang hoạt động kém hiệu quả.
- **HyDE / mở rộng truy vấn.** Tạo câu trả lời giả định từ truy vấn, nhúng câu trả lời, truy xuất. Thu hẹp khoảng cách cụm từ giữa các câu hỏi ngắn và tài liệu dài. Nâng precision miễn phí không training.
- **Ngân sách ngữ cảnh dưới 8K tokens.** Số lần truy cập nhất quán ở giới hạn đó có nghĩa là ngưỡng xếp hạng lại quá lỏng lẻo.
- **Phiên bản mọi thứ.** Prompts, quy tắc phân đoạn, embedding model, xếp hạng lại. Bất kỳ sự trôi dạt nào cũng âm thầm phá vỡ chất lượng câu trả lời. CI các cổng về độ trung thực, precision ngữ cảnh và tỷ lệ câu hỏi chưa trả lời chặn hồi quy trước khi người dùng nhìn thấy chúng.
- **Truy xuất ba chiều (BM25 + dày đặc + học thưa thớt như SPLADE) vượt trội hơn hai chiều** vào benchmarks 2026, đặc biệt là đối với các truy vấn trộn danh từ riêng với ngữ nghĩa. Ship nó khi cơ sở hạ tầng hỗ trợ các chỉ số SPLADE.

Thiết kế truy xuất thích hợp làm giảm ảo giác từ 70-90% theo các phép đo của ngành năm 2026. Hầu hết các RAG tăng hiệu suất đến từ việc truy xuất tốt hơn chứ không phải model fine-tuning.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-retrieval-picker.md`:

```markdown
---
name: retrieval-picker
description: Pick a retrieval stack for a given corpus and query pattern.
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

Given requirements (corpus size, query pattern, latency budget, quality bar, infra constraints), output:

1. Stack. BM25 only, dense only, hybrid (BM25 + dense + RRF), hybrid + cross-encoder rerank, or three-way (BM25 + dense + learned-sparse).
2. Dense encoder. Name the specific model. Match to language(s), domain, and context length.
3. Reranker. Name the specific cross-encoder model if used. Flag that rerank adds 30-100ms latency on top-30.
4. Evaluation plan. Recall@10 is the primary retriever metric. MRR for multi-answer. Baseline first, incremental improvements measured against it.

Refuse to recommend dense-only for corpora with named entities, error codes, or product SKUs unless the user has evidence dense handles exact matches. Refuse to skip reranking for high-stakes retrieval (legal, medical) where the final top-5 decides the user's answer.
```

## Bài tập

1. **Dễ dàng.** Triển khai `hybrid_search` trên trên kho dữ liệu 500 tài liệu. Kiểm tra 20 truy vấn. So sánh recall ở mức 5 giữa chỉ BM25, chỉ dày đặc và lai.
2. **Trung bình.** Thêm tính toán MRR. Đối với mỗi truy vấn thử nghiệm có một tài liệu chính xác đã biết, hãy tìm thứ hạng của tài liệu chính xác trong bảng xếp hạng BM25, dày đặc và kết hợp. Báo cáo MRR cho từng trường hợp.
3. **Khó.** Fine-tune encoder dày đặc trên miền của bạn bằng cách sử dụng MultipleNegativesRankingLoss (Câu Transformers). Xây dựng một bộ training từ 500 cặp truy vấn-tài liệu. So sánh trước và sau fine-tune recall.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| BM25 | Tìm kiếm từ khóa | Okapi BM25. Chấm điểm tài liệu theo tần suất học kỳ, IDF và độ dài. |
| Truy xuất dày đặc | Tìm kiếm Vector | Mã hóa query + doc thành vectors, tìm hàng xóm gần nhất. |
| Bi-encoder | Embedding model | Mã hóa truy vấn và tài liệu độc lập. Nhanh chóng tại thời điểm truy vấn. |
| encoder chéo | Xếp hạng lại model | Mã hóa truy vấn + tài liệu với nhau. Chậm nhưng chính xác. |
| RRF | Hợp nhất xếp hạng | Kết hợp hai bảng xếp hạng bằng cách tổng `1/(k + rank)`. |
| Recall@k | Chỉ số truy xuất | Phần truy vấn trong đó tài liệu có liên quan nằm trong top-k. |

## Đọc thêm

- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) - phương pháp điều trị BM25 dứt khoát.
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR, encoder lưỡng quy kinh điển.
- [Formal et al. (2021). SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720) - loài chó săn thưa thớt có học hỏi thu hẹp khoảng cách với mật độ dày đặc.
- [Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) - Giấy RRF.
- [Khattab and Zaharia (2020). ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832) — truy xuất tương tác muộn.
