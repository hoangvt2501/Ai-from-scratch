# Trích xuất quan hệ & Xây dựng đồ thị tri thức

> NER đã tìm thấy các thực thể. Liên kết thực thể neo họ. Trích xuất quan hệ tìm thấy các cạnh giữa chúng. Biểu đồ tri thức là tổng của các nút, cạnh và nguồn gốc của chúng.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 5 · 06 (NER), Giai đoạn 5 · 25 (Liên kết thực thể)
**Thời lượng:** ~60 phút

## Vấn đề

Một nhà phân tích viết: "Tim Cook trở thành CEO của Apple vào năm 2011". Bốn sự thật:

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

Trích xuất quan hệ (RE) biến văn bản tự do thành bộ ba có cấu trúc `(subject, relation, object)`. Tổng hợp trên một kho dữ liệu và bạn có một biểu đồ tri thức. Tổng hợp và truy vấn và bạn có một nền tảng lý luận cho RAG, phân tích hoặc kiểm tra tuân thủ.

Vấn đề năm 2026: LLMs trích xuất quan hệ một cách nhiệt tình. Quá nhiệt tình. Họ ảo giác ba lần mà văn bản nguồn không hỗ trợ. Nếu không có nguồn gốc, bạn không thể phân biệt bộ ba thực sự với hư cấu hợp lý. Câu trả lời năm 2026 là pipelines neo và xác minh kiểu AEVS.

## Khái niệm

![Text → triples → knowledge graph](../assets/relation-extraction.svg)

**Dạng ba.** `(subject_entity, relation_type, object_entity)`. Các mối quan hệ đến từ một bản thể đóng (thuộc tính Wikidata, FIBO, UMLS) hoặc một tập mở (kiểu OpenIE, bất cứ điều gì cũng được).

**Ba phương pháp chiết xuất.**

1. **Quy tắc / dựa trên mẫu.** Các mẫu Hearst: "X chẳng hạn như Y" → `(Y, isA, X)`. Cộng với regex thủ công. Giòn, chính xác, có thể giải thích được.
2. **Bộ phân loại có giám sát.** Với hai đề cập thực thể trong một câu, hãy dự đoán mối quan hệ từ một tập cố định. Được huấn luyện về TACRED, ACE, KBP. Tiêu chuẩn 2015–2022.
3. **LLM tổng quát.** Prompt model phát ra bộ ba. Hoạt động ra khỏi hộp. Cần nguồn gốc, hoặc ảo giác những thứ rác rưởi trông hợp lý.

**AEVS (Anchor-Extraction-Verification-Supplement, 2026).** Các framework giảm thiểu ảo giác hiện tại:

- **Neo.** Xác định mọi span thực thể và span cụm từ quan hệ với các vị trí chính xác.
- **Trích xuất.** Tạo bộ ba được liên kết với spans neo.
- **Xác minh.** Khớp từng phần tử ba trở lại văn bản nguồn; từ chối bất cứ điều gì không được hỗ trợ.
- **Bổ sung.** Thẻ bảo hiểm đảm bảo không có span neo nào bị rơi.

Ảo giác giảm mạnh. Yêu cầu nhiều điện toán hơn nhưng có thể kiểm tra được.

**Sự đánh đổi giữa mở và đóng.**

- **Bản thể học đóng.** Danh sách thuộc tính cố định (ví dụ: 11.000+ thuộc tính của Wikidata). Có thể dự đoán được. Có thể truy vấn. Khó phát minh.
- **Mở IE.** Bất kỳ cụm từ bằng lời nói nào cũng trở thành một mối quan hệ. recall cao. precision thấp. Lộn xộn để truy vấn.

Production KG thường trộn lẫn: mở IE để khám phá, sau đó chuẩn hóa các mối quan hệ vào một bản thể đóng trước khi hợp nhất vào đồ thị chính.

## Tự xây dựng

### Bước 1: trích xuất dựa trên mẫu

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

Xem `code/main.py` để biết bộ vắt đồ chơi đầy đủ. Các mẫu Hearst vẫn ship trong pipelines miền cụ thể vì chúng có thể gỡ lỗi.

### Bước 2: phân loại quan hệ giám sát

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL là một trình trích xuất quan hệ seq2seq: văn bản vào, gấp ba lần, đã có trong id thuộc tính Wikidata. Fine-tuned dữ liệu giám sát từ xa. Đường cơ sở trọng lượng mở tiêu chuẩn.

### Bước 3: Chiết xuất LLM nhắc bằng neo

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""
```

Xác minh mọi span trả về so với nguồn. Từ chối bất cứ điều gì `text[start:end] != triple_entity`. Đây là bước "xác minh" AEVS ở dạng tối thiểu của nó.

### Bước 4: chuẩn hóa vào một bản thể khép kín

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (inverted subject/object)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # drop unmapped open relations or route to manual review
```

Chuẩn hóa thường chiếm 60-80% công việc kỹ thuật. Ngân sách cho nó.

### Bước 5: xây dựng một biểu đồ nhỏ và truy vấn

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

Đây là nguyên tử của mọi hệ thống RAG trên KG. Mở rộng quy mô với kho ba RDF (Blazegraph, Virtuoso), đồ thị thuộc tính (Neo4j) hoặc kho đồ thị tăng cường vector.

## Cạm bẫy

- **Đồng tham khảo trước RE.** "Anh ấy thành lập Apple" - RE cần biết "anh ấy" là ai. Chạy coref trước (bài 24).
- **Chuẩn hóa thực thể.** "Apple Inc" và "Apple" phải phân giải cho cùng một nút. Liên kết thực thể trước (bài 25).
- **Bộ ba ảo giác.** LLMs phát ra bộ ba mà văn bản không hỗ trợ. Thực thi xác minh span.
- **Trôi dạt chuẩn hóa mối quan hệ.** Các mối quan hệ IE mở không nhất quán ("được sinh ra trong", "đến từ", "là người bản địa của"). Thu gọn thành mã chính tắc hoặc biểu đồ không thể truy vấn được.
- "Tim Cook là CEO của Apple" - đúng bây giờ, sai vào năm 2005. Nhiều mối quan hệ có giới hạn thời gian. Sử dụng từ hạn định (`P580` thời gian bắt đầu, `P582` thời gian kết thúc trong Wikidata).
- **Tên miền không khớp.** REBEL được huấn luyện trên Wikipedia. Văn bản pháp lý, y tế và khoa học thường cần models RE fine-tuned miền.

## Ứng dụng

stack năm 2026:

| Tình huống | Chọn |
|-----------|------|
| production nhanh, tên miền chung | REBEL hoặc LlamaPred với chuẩn hóa Wikidata |
| Miền cụ thể (biomed, hợp pháp) | fine-tune miền kiểu SciREX + bản thể tùy chỉnh |
| Đầu ra được kiểm toán, nhắc nhở LLM | AEVS pipeline: trích xuất → neo → xác minh bổ sung → |
| Tin tức có độ volume cao IE | Dựa trên mẫu + kết hợp được giám sát |
| Xây dựng KG từ đầu | Mở IE + thẻ chuẩn hóa thủ công |
| KG thái dương | Trích xuất với bộ hạn định (start/end thời gian, thời điểm) |

Mô hình tích hợp: NER → coref → thực thể liên kết trích xuất quan hệ → → ánh xạ bản thể học → tải đồ thị. Mỗi giai đoạn là một cánh cổng chất lượng tiềm năng.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-re-designer.md`:

```markdown
---
name: re-designer
description: Design a relation extraction pipeline with provenance and canonicalization.
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

Given a corpus (domain, language, volume) and downstream use (KG-RAG, analytics, compliance), output:

1. Extractor. Pattern-based / supervised / LLM / AEVS hybrid. Reason tied to precision vs recall target.
2. Ontology. Closed property list (Wikidata / domain) or open IE with canonicalization pass.
3. Provenance. Every triple carries source char-span + doc id. Non-negotiable for audit.
4. Merge strategy. Canonical entity id + relation id + temporal qualifiers; dedup policy.
5. Evaluation. Precision / recall on 200 hand-labelled triples + hallucination-rate on LLM-extracted sample.

Refuse any LLM-based RE pipeline without span verification (source provenance). Refuse open-IE output flowing into a production graph without canonicalization. Flag pipelines with no temporal qualifier on time-bounded relations (employer, spouse, position).
```

## Bài tập

1. **Dễ dàng.** Chạy trình trích xuất mẫu trong `code/main.py` trên 5 câu bài báo. Kiểm tra tay precision.
2. **Trung bình.** Sử dụng REBEL (hoặc một LLM nhỏ) trên cùng một câu. So sánh bộ ba. Máy vắt nào có precision cao hơn? recall cao hơn?
3. **Khó.** Xây dựng pipeline AEVS: trích xuất bằng LLM + xác minh spans so với nguồn. Đo tỷ lệ ảo giác trước so với sau bước xác minh trên 50 câu kiểu Wikipedia.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Bộ ba | Đối tượng-quan hệ-chủ thể | `(s, r, o)` bộ là đơn vị nguyên tử của KG. |
| Mở IE | Trích xuất bất cứ thứ gì | Các cụm từ quan hệ từ vựng mở; recall cao, precision thấp. |
| Bản thể học đóng | schema cố định | Tập hợp các kiểu quan hệ có giới hạn (Wikidata, UMLS, FIBO). |
| Chuẩn hóa | Chuẩn hóa mọi thứ | Ánh xạ tên bề mặt / mối quan hệ với id chính tắc. |
| AEVS | Khai thác nối đất | Neo-Extraction-Verification-Supplement pipeline (2026). |
| Xuất xứ | Liên kết nguồn tin cậy | Mỗi bộ ba mang một id tài liệu + ký tự span đến nguồn của nó. |
| Giám sát từ xa | Nhãn giá rẻ | Căn chỉnh văn bản với KG hiện có để tạo dữ liệu training. |

## Đọc thêm

- [Mintz et al. (2009). Distant supervision for relation extraction without labeled data](https://www.aclweb.org/anthology/P09-1113.pdf) — bài báo giám sát từ xa.
- [Huguet Cabot, Navigli (2021). REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204.pdf) - seq2seq RE ngựa làm việc.
- [Wadden et al. (2019). Entity, Relation, and Event Extraction with Contextualized Span Representations (DyGIE++)](https://arxiv.org/abs/1909.03546) — IE chung.
- [AEVS — Anchor-Extraction-Verification-Supplement framework](https://www.mdpi.com/2073-431X/15/3/178) - Thiết kế giảm thiểu ảo giác năm 2026.
- [Wikidata SPARQL tutorial](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) — truy vấn biểu đồ chính tắc.
