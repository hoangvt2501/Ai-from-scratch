# Embedding Models - Đi sâu vào năm 2026

> Word2Vec cung cấp cho bạn một vector cho mỗi từ. embedding models hiện đại cung cấp cho bạn vector trên mỗi đoạn văn, đa ngôn ngữ, với các chế độ xem thưa thớt, dày đặc và nhiều vector, có kích thước phù hợp với chỉ mục của bạn. Chọn sai và RAG của bạn lấy lại thứ sai.

**Loại:** Học
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 5 · 03 (Word2Vec), Giai đoạn 5 · 14 (Truy xuất thông tin)
**Thời lượng:** ~60 phút

## Vấn đề

Hệ thống RAG của bạn truy xuất đoạn văn sai 40% thời gian. Thủ phạm hiếm khi là cơ sở dữ liệu vector hoặc prompt. Đó là embedding model.

Chọn một embedding vào năm 2026 có nghĩa là chọn trên năm trục:

1. **Dày đặc so với thưa thớt so với nhiều vector.** Một vector mỗi đoạn hoặc một token hoặc một túi từ có trọng lượng thưa thớt.
2. **Phạm vi ngôn ngữ.** models tiếng Anh đơn ngữ vẫn giành chiến thắng trong các nhiệm vụ chỉ bằng tiếng Anh. Đa ngôn ngữ models chiến thắng khi ngữ liệu được trộn lẫn.
3. **Độ dài ngữ cảnh.** 512 tokens so với 8.192 so với 32.768 — và dung lượng hiệu quả thực tế thường là 60-70% mức tối đa được quảng cáo.
4. **Ngân sách thứ nguyên.** 3.072 thả nổi ở precision đầy đủ = 12 KB mỗi vector. Với giá 100 triệu vectors, dung lượng lưu trữ là 1.300/month. đô la cắt ngắn Matryoshka cắt giảm 4× này.
5. **Mở so với lưu trữ.** Trọng lượng mở có nghĩa là bạn kiểm soát stack và dữ liệu. Được lưu trữ có nghĩa là bạn trao đổi quyền kiểm soát cho luôn mới nhất.

Bài học này đặt tên cho sự đánh đổi để bạn có thể chọn bằng chứng, không phải bất cứ thứ gì phổ biến trong quý trước.

## Khái niệm

![Dense, sparse, and multi-vector embeddings](../assets/embedding-modes.svg)

**embeddings dày đặc.** Một vector mỗi đoạn (thường là 384-3.072 chiều). Sự tương đồng cosin xếp hạng các đoạn văn theo khoảng cách ngữ nghĩa. OpenAI `text-embedding-3-large`, chế độ dày đặc BGE-M3, Voyage-3. Lựa chọn mặc định.

**embeddings thưa thớt.** Phong cách SPLADE. Một transformer dự đoán trọng số cho mỗi token từ vựng, sau đó loại bỏ hầu hết các từ vựng. Kết quả là một vector thưa thớt có kích thước |vocab|. Nắm bắt kết hợp từ vựng (như BM25) nhưng với trọng số thuật ngữ đã học. Mạnh mẽ với các truy vấn nặng từ khóa.

**Đa vector (tương tác muộn).** ColBERTv2, Jina-ColBERT. Một vector mỗi token. Chấm điểm bằng MaxSim: đối với mỗi truy vấn token, tìm tài liệu tương tự nhất token, tính tổng điểm. Đắt hơn để lưu trữ và ghi điểm, nhưng chiến thắng trong các truy vấn dài và kho dữ liệu dành riêng cho miền.

**BGE-M3: cả ba cùng một lúc.** Một model xuất đồng thời các biểu diễn dày đặc, thưa thớt và đa vector. Mỗi có thể được truy vấn độc lập; điểm số hợp nhất thông qua tổng trọng số. Mặc định năm 2026 khi bạn muốn linh hoạt từ một checkpoint.

**Matryoshka Representation Learning.** Được huấn luyện để N chiều đầu tiên của vector tạo thành một embedding độc lập hữu ích. Cắt bớt vector 1.536 độ mờ xuống 256 độ mờ và trả ~1% accuracy để tiết kiệm 6× dung lượng lưu trữ. Được hỗ trợ bởi OpenAI text-3, Cohere v4, Voyage-4, Jina v5, Gemini Embedding 2, Nomic v1.5+.

### Bảng xếp hạng MTEB kể một phần câu chuyện

Embedding Benchmark văn bản lớn — 56 tác vụ trên 8 loại tác vụ khi khởi chạy (2022), được mở rộng lên 100+ tác vụ trong MTEB v2. Vào đầu năm 2026, Gemini Embedding 2 đỉnh thu hồi (67,71 MTEB-R). Cohere embed-v4 dẫn đầu chung (65,2 MTEB). BGE-M3 dẫn đầu đa ngôn ngữ có trọng lượng mở (63.0). Bảng xếp hạng là cần thiết nhưng không đủ - luôn benchmark trên miền của bạn.

### Mô hình ba tầng

| Trường hợp sử dụng | Mô hình |
|----------|---------|
| Vượt qua đầu tiên nhanh chóng | Hai encoder dày đặc (BGE-M3, văn bản-3-nhỏ) |
| Tăng Recall | Thưa thớt (SPLADE, BGE-M3 thưa thớt) + cầu chì RRF |
| Precision trên top-50 | Multi-vector (ColBERTv2) hoặc cross-encoder reranker |

Hầu hết production stacks sử dụng cả ba.

## Tự xây dựng

### Bước 1: đường cơ sở - embeddings dày đặc với Sentence-BERT

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True` làm cho tích chấm tương tự cosin bằng nhau. Luôn đặt nó.

### Bước 2: Cắt bớt Matryoshka

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

Bình thường hóa lại sau khi cắt bớt. Nomic v1.5, OpenAI text-3 và Voyage-4 được huấn luyện nên điều này không mất dữ liệu trong vài cấp độ đầu tiên. models không phải Matryoshka (BERT câu gốc) xuống cấp mạnh khi bị cắt bớt.

### Bước 3: Đa chức năng BGE-M3

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays
```

Ba chỉ mục, một inference gọi. Hợp nhất điểm số:

```python
dense_score = ... # cosine over dense_vecs
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

Điều chỉnh trọng số trên miền của bạn.

### Bước 4: Đánh giá MTEB đối với một tác vụ tùy chỉnh

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

Chạy models ứng cử viên của bạn trên một tập hợp con *đại diện*. Đừng chỉ tin tưởng vào thứ hạng trên bảng xếp hạng - lĩnh vực của bạn rất quan trọng.

### Bước 5: cosine cuộn bằng tay từ đầu

Xem `code/main.py`. Thủ thuật băm trung bình embeddings (chỉ stdlib). Không cạnh tranh với transformer embeddings, nhưng thể hiện hình dạng: mã hóa → vector → chuẩn hóa → sản phẩm chấm.

## Cạm bẫy

- **Cùng một model cho truy vấn và tài liệu.** Một số models (Voyage, Jina-ColBERT) sử dụng mã hóa bất đối xứng — truy vấn và tài liệu đi qua các đường dẫn khác nhau. Luôn kiểm tra thẻ model.
- **Thiếu tiền tố.** `bge-*` models cần `"Represent this sentence for searching relevant passages: "` thêm vào trước các truy vấn. Khoảng cách recall 3-5 điểm nếu bạn quên.
- **Cắt tỉa quá mức Matryoshka.** 1.536 → 256 thường an toàn. 1.536 → 64 thì không. Xác thực trên bộ đánh giá của bạn.
- **Cắt bớt ngữ cảnh.** Hầu hết models âm thầm cắt bớt đầu vào trên độ dài tối đa của chúng. Tài liệu dài cần được chia nhỏ (xem bài 23).
- **Bỏ qua đuôi độ trễ.** Điểm MTEB ẩn độ trễ p99. Một model 600 triệu có thể đánh bại một model 335 triệu 2 điểm nhưng có giá cao hơn 3× cho mỗi truy vấn.

## Ứng dụng

stack năm 2026:

| Tình huống | Chọn |
|-----------|------|
| Chỉ tiếng Anh, nhanh API | `text-embedding-3-large` hoặc `voyage-3-large` |
| Trọng lượng mở, tiếng Anh | `BAAI/bge-large-en-v1.5` |
| Trọng lượng mở, đa ngôn ngữ | `BAAI/bge-m3` hoặc `Qwen3-Embedding-8B` |
| Bối cảnh dài (32k+) | Hành trình-3-lớn, Cohere embed-v4, Qwen3-Embedding-8B |
| Triển khai chỉ CPU | Nomic Embed v2 (137 triệu tham số, MoE) |
| Hạn chế lưu trữ | Matryoshka-cắt ngắn + int8 quantization |
| Truy vấn nặng từ khóa | Thêm SPLADE thưa thớt, cầu chì RRF dày đặc |

Mẫu 2026: bắt đầu với BGE-M3 hoặc text-3-large, đánh giá trên miền của bạn với MTEB, hoán đổi nếu model dành riêng cho miền thắng hơn 3 điểm.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-embedding-picker.md`:

```markdown
---
name: embedding-picker
description: Pick embedding model, dimension, and retrieval mode for a given corpus and deployment.
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

Given a corpus (size, languages, domain, avg length), deployment target (cloud / edge / on-prem), latency budget, and storage budget, output:

1. Model. Named checkpoint or API. One-sentence reason.
2. Dimension. Full / Matryoshka-truncated / int8-quantized. Reason tied to storage budget.
3. Mode. Dense / sparse / multi-vector / hybrid. Reason.
4. Query prefix / template if required by the model card.
5. Evaluation plan. MTEB tasks relevant to domain + held-out domain eval with nDCG@10.

Refuse recommendations that truncate Matryoshka to <64 dims without domain validation. Refuse ColBERTv2 for corpora under 10k passages (overhead not justified). Flag long-document corpora (>8k tokens) routed to models with 512-token windows.
```

## Bài tập

1. **Dễ dàng.** Mã hóa 100 câu với `bge-small-en-v1.5` ở độ mờ hoàn toàn (384), sau đó ở Matryoshka 128. Đo lường MRR giảm trên 10 truy vấn.
2. **Trung bình.** So sánh BGE-M3 đậm đặc, thưa thớt và trung bình trên 500 đoạn từ miền của bạn. Cái nào thắng trên recall@10? Nhiệt hạch RRF có đánh bại chế độ đơn tốt nhất không?
3. **Khó.** Chạy MTEB trên ba models ứng viên trên 2 nhiệm vụ miền hàng đầu của bạn. Báo cáo điểm MTEB, độ trễ p99 trên batch 100 truy vấn và truy vấn $/1M. Chọn một trong những tối ưu Pareto.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| embedding đặc | Các vector | Một vector có kích thước cố định cho mỗi văn bản. Sự tương đồng cosin để xếp hạng. |
| embedding thưa thớt | Học BM25 | Một trọng lượng cho mỗi từ vựng token; chủ yếu là số không; được huấn luyện từ đầu đến cuối. |
| Đa vector | Phong cách ColBERT | Một vector mỗi token; Chấm điểm MaxSim; chỉ số lớn hơn, recall tốt hơn. |
| Matryoshka | Thủ thuật búp bê Nga | N mờ đầu tiên là một embedding nhỏ hơn hợp lệ. |
| MTEB | Các benchmark | Embedding Benchmark văn bản khổng lồ — 56 tác vụ khi khởi chạy, 100+ tác vụ trong v2. |
| BEIR | Truy xuất benchmark | 18 nhiệm vụ truy xuất zero-shot; thường được trích dẫn vì tính mạnh mẽ của nhiều miền. |
| Mã hóa bất đối xứng | Truy vấn ≠ đường dẫn tài liệu | Model sử dụng các phép chiếu khác nhau cho các truy vấn và tài liệu. |

## Đọc thêm

- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084) - bài báo hai encoder.
- [Muennighoff et al. (2022). MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) — giấy bảng xếp hạng.
- [Chen et al. (2024). BGE-M3: Multi-lingual, Multi-functionality, Multi-granularity](https://arxiv.org/abs/2402.03216) — model ba chế độ thống nhất.
- [Kusupati et al. (2022). Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) — thang không gian training mục tiêu.
- [Santhanam et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) - tương tác muộn trong production.
- [MTEB leaderboard on Hugging Face](https://huggingface.co/spaces/mteb/leaderboard) - bảng xếp hạng trực tiếp.
