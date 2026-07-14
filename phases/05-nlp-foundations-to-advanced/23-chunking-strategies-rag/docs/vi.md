# Chiến lược phân đoạn cho RAG

> Chunking configuration ảnh hưởng đến chất lượng truy xuất nhiều như việc lựa chọn embedding model (Vectara NAACL 2025). Phân đoạn sai và không có số lần xếp hạng lại nào cứu bạn.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 5 · 14 (Truy xuất thông tin), Giai đoạn 5 · 22 (Embedding Models)
**Thời lượng:** ~60 phút

## Vấn đề

Bạn đưa một hợp đồng dài 50 trang vào một hệ thống RAG. Người dùng hỏi: "Điều khoản chấm dứt là gì?" Retriever trả về trang bìa. Tại sao? Bởi vì model được huấn luyện trên các đoạn 512 token và điều khoản chấm dứt nằm trong 20 trang, được chia thành ngắt trang, không có từ khóa cục bộ nào gắn nó với truy vấn.

Cách khắc phục không phải là "mua một embedding model tốt hơn". Bản sửa lỗi là phân đoạn. Lớn như thế nào? Chồng chéo? Chia ở đâu? Với bối cảnh xung quanh?

Feb 2026 benchmarks cho thấy kết quả đáng ngạc nhiên:

- Nghiên cứu năm 2026 của Vectara: đệ quy 512-token chunking đánh bại phân đoạn ngữ nghĩa 69% → 54% accuracy.
- SPLADE + Mistral-8B về các câu hỏi tự nhiên: chồng chéo không mang lại lợi ích có thể đo lường được.
- Vách đá ngữ cảnh: chất lượng phản hồi giảm mạnh khoảng 2.500 tokens ngữ cảnh.

Câu trả lời "rõ ràng" (phân đoạn ngữ nghĩa, 20% chồng chéo, 1000 tokens) thường sai. Bài học này xây dựng trực giác cho sáu chiến lược và cho bạn biết khi nào nên đạt được chiến lược nào.

## Khái niệm

![Six chunking strategies visualized on one passage](../assets/chunking.svg)

**Đã sửa lỗi phân đoạn.** Tách mỗi N ký tự hoặc tokens. Đường cơ sở đơn giản nhất. Ngắt giữa câu. Nén tốt, mạch lạc kém.

**Đệ quy.** `RecursiveCharacterTextSplitter` của LangChain. Hãy thử tách `\n\n` trước, sau đó `\n`, sau đó `.`, sau đó là khoảng cách. Rơi trở lại sạch sẽ. Mặc định năm 2026.

**Ngữ nghĩa.** Nhúng từng câu. Tính toán sự tương đồng cosin giữa các câu liền kề. Tách khi mức độ tương đồng giảm xuống dưới ngưỡng. Duy trì tính mạch lạc của chủ đề. Chậm hơn; đôi khi tạo ra các mảnh nhỏ 40 token làm tổn thương việc truy xuất.

**Câu.** Chia ranh giới câu. Một câu cho mỗi đoạn hoặc một cửa sổ gồm N câu. Khớp với phân đoạn ngữ nghĩa lên đến ~5k tokens với chi phí thấp.

**Parent-document.** Lưu trữ các chunk con nhỏ để truy xuất *và* chunk cha lớn hơn cho ngữ cảnh. Truy xuất theo trẻ; trả lại cha mẹ. Suy thoái một cách duyên dáng: những đứa trẻ xấu vẫn trả về cha mẹ hợp lý.

**Phân đoạn muộn (2024).** Nhúng toàn bộ tài liệu ở cấp độ token trước, sau đó gộp token embeddings vào embeddings khối. Duy trì ngữ cảnh chéo. Hoạt động với trình nhúng ngữ cảnh dài (BGE-M3, Jina v3). Điện toán cao hơn.

**Truy xuất theo ngữ cảnh (Anthropic, 2024).** Thêm vào trước mỗi đoạn bằng bản tóm tắt do LLM tạo về vị trí của nó trong tài liệu ("Đoạn này là phần 3.2 của các điều khoản chấm dứt..."). Cải thiện 35-50% khả năng truy xuất trong benchmark của chính Anthropic. Tốn kém để lập chỉ mục.

### Quy tắc đánh bại mọi mặc định

Khớp kích thước chunk với loại truy vấn:

| Loại truy vấn | Kích thước khối |
|------------|-----------|
| Factoid ("tên CEO là gì?") | 256-512 tokens |
| Phân tích / nhiều bước nhảy | 512-1024 tokens |
| Hiểu toàn bộ phần | 1024-2048 tokens |

NVIDIA năm 2026 benchmark. Chunk phải đủ lớn để chứa câu trả lời cộng với ngữ cảnh cục bộ, đủ nhỏ để top-K của retriever trả về tập trung vào câu trả lời thay vì nhiễu ngữ cảnh.

## Tự xây dựng

### Bước 1: phân đoạn cố định và đệ quy

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### Bước 2: phân đoạn ngữ nghĩa

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

Điều chỉnh `threshold` trên miền của bạn. Mảnh vỡ → quá cao. Quá thấp → một khối khổng lồ.

### Bước 3: tài liệu cha mẹ

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

Thông tin chi tiết chính: cha mẹ trùng lặp. Nhiều trẻ em có thể ánh xạ đến cùng một cha mẹ; trả lại tất cả sẽ lãng phí bối cảnh.

### Bước 4: truy xuất theo ngữ cảnh (mẫu Anthropic)

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

Lập chỉ mục các đoạn theo ngữ cảnh. Tại thời điểm truy vấn, truy xuất được hưởng lợi từ tín hiệu xung quanh bổ sung.

### Bước 5: Đánh giá

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

Luôn benchmark. Chiến lược "tốt nhất" cho kho dữ liệu của bạn có thể không phù hợp với bất kỳ bài đăng blog nào.

## Cạm bẫy

- **Chunking chỉ được đánh giá trên các truy vấn factoid.** Các truy vấn nhiều bước nhảy tiết lộ những người chiến thắng rất khác nhau. Sử dụng bộ đánh giá phân tầng loại truy vấn.
- **Phân đoạn ngữ nghĩa không có kích thước tối thiểu.** Tạo ra 40 mảnh token gây ảnh hưởng đến việc truy xuất. Luôn thực thi `min_tokens`.
- **Chồng chéo như cargo sự sùng bái.** Các nghiên cứu năm 2026 cho thấy sự chồng chéo thường không mang lại lợi ích và tăng gấp đôi chi phí chỉ số. Đo lường, đừng giả định.
- **Không thực thi min/max.** Các đoạn 5 tokens hoặc 5000 tokens đều phá vỡ truy xuất. Kẹp.
- **Phân đoạn chéo tài liệu.** Không bao giờ để một đoạn span hai tài liệu. Luôn chia từng tài liệu, sau đó merge.

## Ứng dụng

stack năm 2026:

| Tình huống | Chiến lược |
|-----------|----------|
| Bản dựng đầu tiên, kho dữ liệu không xác định | đệ quy, 512 tokens, không chồng chéo |
| QA thực tế | đệ quy, 256-512 tokens |
| Phân tích / nhiều bước nhảy | Đệ quy, 512-1024 tokens + tài liệu mẹ |
| Tham khảo chéo nặng (hợp đồng, giấy tờ) | Phân đoạn muộn hoặc truy xuất theo ngữ cảnh |
| Kho dữ liệu hội thoại / hội thoại | Khối cấp lượt + siêu dữ liệu loa |
| Phát biểu ngắn (tweet, đánh giá) | Một tài liệu = một đoạn |

Bắt đầu với đệ quy 512. Đo lường recall@5 trên tập hợp đánh giá 50 truy vấn. Điều chỉnh từ đó.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-chunker.md`:

```markdown
---
name: chunker
description: Pick a chunking strategy, size, and overlap for a given corpus and query distribution.
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

Given a corpus (document types, avg length, domain) and query distribution (factoid / analytical / multi-hop), output:

1. Strategy. Recursive / sentence / semantic / parent-document / late / contextual. Reason.
2. Chunk size. Token count. Reason tied to query type.
3. Overlap. Default 0; justify if >0.
4. Min/max enforcement. `min_tokens`, `max_tokens` guards.
5. Evaluation plan. Recall@5 on 50-query stratified eval set (factoid, analytical, multi-hop).

Refuse any chunking strategy without min/max chunk size enforcement. Refuse overlap above 20% without an ablation showing it helps. Flag semantic chunking recommendations without a min-token floor.
```

## Bài tập

1. **Dễ dàng.** Cắt một tài liệu 20 trang với cố định (512, 0), đệ quy (512, 0) và đệ quy (512, 100). So sánh số lượng chunk và chất lượng ranh giới.
2. **Trung bình.** Xây dựng bộ đánh giá 30 truy vấn trên 5 tài liệu. Đo lường recall@5 cho đệ quy, ngữ nghĩa và tài liệu mẹ. Cái nào thắng? Nó có khớp với các bài đăng trên blog không?
3. **Khó.** Triển khai truy xuất theo ngữ cảnh. Đo lường sự cải thiện MRR so với đệ quy cơ sở. Báo cáo chi phí chỉ mục (LLM cuộc gọi) so với mức tăng accuracy.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Khối | Một phần của tài liệu | Đơn vị tài liệu phụ được nhúng, lập chỉ mục và truy xuất. |
| Chồng chéo | Biên độ an toàn | N tokens được chia sẻ giữa các khối liền kề; thường vô dụng vào năm 2026 benchmarks. |
| Phân đoạn ngữ nghĩa | Phân đoạn thông minh | Tách nơi câu liền kề embedding sự tương đồng giảm xuống. |
| Tài liệu cha mẹ | Truy xuất hai cấp độ | Lấy trẻ nhỏ, trả lại cha mẹ lớn hơn. |
| Phân đoạn muộn | Khối sau embedding | Nhúng tài liệu đầy đủ ở cấp độ token, gộp vào vectors khối. |
| Truy xuất theo ngữ cảnh | Thủ thuật của Anthropic | Tóm tắt do LLM tạo được thêm vào trước mỗi đoạn trước khi lập chỉ mục. |
| Vách đá bối cảnh | Tường 2500-token | Sự sụt giảm chất lượng được quan sát thấy khoảng 2,5 nghìn tokens ngữ cảnh vào năm RAG (tháng 1 năm 2026). |

## Đọc thêm

- [Yepes et al. / LangChain — Recursive Character Splitting docs](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — mặc định trong production.
- [Vectara (2024, NAACL 2025). Chunking configurations analysis](https://arxiv.org/abs/2410.13070) - việc chia nhỏ cũng quan trọng như sự lựa chọn của embedding.
- [Jina AI — Late Chunking in Long-Context Embedding Models (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) - tờ giấy phân đoạn muộn.
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) - Cải thiện 35-50% truy xuất với tiền tố ngữ cảnh do LLM tạo.
- [NVIDIA 2026 chunk-size benchmark — Premai summary](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) — kích thước chunk theo loại truy vấn.
