# Liên kết thực thể & định hướng

> NER tìm thấy "Paris". Liên kết thực thể quyết định: Paris, Pháp? Paris Hilton? Paris, Texas? Paris (hoàng tử thành Troy)? Nếu không liên kết, biểu đồ tri thức của bạn vẫn mơ hồ.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 5 · 06 (NER), Giai đoạn 5 · 24 (Độ phân giải đồng tham chiếu)
**Thời lượng:** ~60 phút

## Vấn đề

Một câu có nội dung: "Jordan đánh bại báo chí." NER của bạn gắn thẻ "Jordan" là PERSON. Tốt. Nhưng *cái nào* Jordan?

- Michael Jordan (bóng rổ)?
- Michael B. Jordan (diễn viên)?
- Michael I. Jordan (giáo sư ML Berkeley - vâng, sự nhầm lẫn này là có thật trong các bài báo ML)?
- Jordan (đất nước)?
- Jordan (tên tiếng Do Thái)?

Liên kết thực thể (EL) giải quyết mỗi đề cập thành một mục duy nhất trong cơ sở kiến thức: Wikidata, Wikipedia, DBpedia hoặc KB miền của bạn. Hai nhiệm vụ phụ:

1. **Thế hệ ứng cử viên.** Với "Jordan", những mục KB nào là hợp lý?
2. **Định hướng.** Với bối cảnh, ứng cử viên nào là đúng?

Cả hai bước đều có thể học được. Cả hai đều được chuẩn hóa. pipeline kết hợp đã ổn định trong một thập kỷ - điều thay đổi là chất lượng của từ định hướng.

## Khái niệm

![Entity linking pipeline: mention → candidates → disambiguated entity](../assets/entity-linking.svg)

**Tạo ứng viên.** Với dạng bề mặt đề cập ("Jordan"), hãy tra cứu các ứng cử viên trong chỉ mục bí danh. Từ điển bí danh Wikipedia bao gồm hầu hết các thực thể được đặt tên: "JFK" → John F. Kennedy, Jacqueline Kennedy, sân bay JFK, JFK (phim). Chỉ mục điển hình trả về 10-30 ứng viên mỗi lần đề cập.

**Định hướng: ba cách tiếp cận.**

1. **Prior + ngữ cảnh (Milne & Witten, 2008).** `P(entity | mention) × context-similarity(entity, text)`. Hoạt động tốt, nhanh chóng, không training.
2. **Dựa trên Embedding (ESS / REL / Blink).** Mã hóa đề cập + ngữ cảnh. Mã hóa mô tả của từng ứng viên. Chọn cosine tối đa. Mặc định 2020-2024.
3. **Generative (THỂ LOẠI, 2021; Dựa trên LLM, 2023+).** Giải mã tên chính tắc của thực thể token từng token. Bị giới hạn trong một thử tên thực thể hợp lệ để đầu ra được đảm bảo là id KB hợp lệ.

**End-to-end so với pipeline.** Modern models (ELQ, BLINK, ExtEnD, GENRE) chạy NER + tạo ứng cử viên + định hướng trong một lần. Pipeline hệ thống vẫn chiếm ưu thế trong production vì bạn có thể hoán đổi các thành phần.

### Hai phép đo

- **Đề cập đến recall (thế hệ ứng cử viên).** Phần vàng đề cập đến nơi mục KB chính xác xuất hiện trong danh sách ứng cử viên. Sàn cho cả pipeline.
- **Định hướng accuracy / F1.** Cho các ứng cử viên đúng, tần suất top 1 đúng.

Luôn báo cáo cả hai. Một hệ thống có 99% định hướng trên 80% recall ứng cử viên là 80% pipeline.

## Tự xây dựng

### Bước 1: xây dựng chỉ mục bí danh từ chuyển hướng Wikipedia

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

Dữ liệu bí danh Wikipedia: ~18 triệu cặp (bí danh, thực thể). Tải xuống từ kết xuất Wikidata. Lưu trữ dưới dạng chỉ mục đảo ngược.

### Bước 2: định hướng dựa trên ngữ cảnh

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

Sự chồng chéo của Jaccard là một món đồ chơi. Thay thế bằng sự tương tự cosin trên embeddings (xem `code/main.py` bước 2 cho phiên bản transformer).

### Bước 3: Dựa trên embedding (kiểu BLINK)

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

Tại thời điểm lập chỉ mục, hãy nhúng mọi thực thể KB một lần. Tại thời điểm truy vấn, nhúng đề cập + ngữ cảnh một lần, chấm-product vào nhóm ứng viên, chọn tối đa.

### Bước 4: liên kết thực thể tổng quát (khái niệm)

GENRE giải mã tiêu đề Wikipedia của thực thể theo từng ký tự. Giải mã ràng buộc (xem bài 20) đảm bảo chỉ có thể xuất tiêu đề hợp lệ. Tích hợp chặt chẽ với một thử được hỗ trợ bởi KB. Hậu duệ hiện đại là REL-GEN và EL được nhắc nhở LLM với đầu ra có cấu trúc.

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

Kết hợp với danh sách trắng (Outlines `choice`), đây là pipeline EL đơn giản nhất ship vào năm 2026.

### Bước 5: đánh giá trên AIDA-CoNLL

AIDA-CoNLL là benchmark EL tiêu chuẩn: 1.393 bài viết của Reuters, 34k đề cập, các thực thể Wikipedia. Báo cáo tỷ lệ phát hiện NIL trong KB accuracy (`P@1`) và ngoài KB.

## Cạm bẫy

- **Xử lý NIL.** Một số đề cập không có trong KB (thực thể mới nổi, những người ít người biết đến). Các hệ thống phải dự đoán NIL thay vì đoán sai thực thể. Được đo riêng.
- **Đề cập đến lỗi ranh giới.** NER ngược dòng bỏ lỡ một phần spans ("Bank of America" được gắn thẻ chỉ là "Ngân hàng"). EL recall giảm.
- **Mức độ phổ biến bias.** Các hệ thống được huấn luyện dự đoán quá mức các thực thể thường xuyên. Đề cập đến "Michael I. Jordan" trên một tờ báo ML thường liên quan đến bóng rổ Jordan.
- **EL đa ngôn ngữ.** Ánh xạ các đề cập trong văn bản tiếng Trung với các thực thể Wikipedia tiếng Anh. Yêu cầu encoder đa ngôn ngữ hoặc bước dịch.
- **KB cũ kỹ.** Các công ty, sự kiện, con người mới không nằm trong bãi rác Wikipedia năm ngoái. Production pipelines cần một vòng lặp làm mới.

## Ứng dụng

stack năm 2026:

| Tình huống | Chọn |
|-----------|------|
| Tiếng Anh đa năng + Wikipedia | BLINK hoặc REL |
| Đa ngôn ngữ, KB = Wikipedia | THỂ LOẠI |
| Thân thiện với LLM, ít mentions/day | Prompt Claude/GPT-4 với danh sách ứng viên + JSON ràng buộc |
| KB dành riêng cho miền (y tế, pháp lý) | BERT tùy chỉnh với truy xuất nhận biết KB + fine-tune trên bộ kiểu AIDA miền |
| Độ trễ cực thấp | Chỉ prior khớp chính xác (đường cơ sở Milne-Witten) |
| Nghiên cứu SOTA | THỂ LOẠI / ExtEnD / LLM-EL tổng quát |

Production mẫu ships vào năm 2026: NER → coref → EL trên mỗi lần đề cập → thu gọn các cụm thành một thực thể chính tắc trên mỗi cụm. Đầu ra: một ID KB cho mỗi thực thể trong tài liệu, không phải một ID cho mỗi đề cập.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-entity-linker.md`:

```markdown
---
name: entity-linker
description: Design an entity linking pipeline — KB, candidate generator, disambiguator, evaluation.
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

Given a use case (domain KB, language, volume, latency budget), output:

1. Knowledge base. Wikidata / Wikipedia / custom KB. Version date. Refresh cadence.
2. Candidate generator. Alias-index, embedding, or hybrid. Target mention recall @ K.
3. Disambiguator. Prior + context, embedding-based, generative, or LLM-prompted.
4. NIL strategy. Threshold on top score, classifier, or explicit NIL candidate.
5. Evaluation. Mention recall @ 30, top-1 accuracy, NIL-detection F1 on held-out set.

Refuse any EL pipeline without a mention-recall baseline (you cannot evaluate a disambiguator without knowing candidate gen surfaced the right entity). Refuse any pipeline using LLM-prompted EL without constrained output to valid KB ids. Flag systems where popularity bias affects minority entities (e.g. name-clashes) without domain fine-tuning.
```

## Bài tập

1. **Dễ dàng.** Triển khai trình định hướng prior+ngữ cảnh trong `code/main.py` trên 10 đề cập mơ hồ (Paris, Jordan, Apple). Gắn nhãn thủ công cho thực thể chính xác. Đo lường accuracy.
2. **Trung bình.** Mã hóa 50 đề cập mơ hồ bằng một câu transformer. Nhúng mô tả của từng ứng viên. So sánh định hướng dựa trên embedding với chồng chéo ngữ cảnh Jaccard.
3. **Khó.** Xây dựng KB miền 1k thực thể (ví dụ: nhân viên + sản phẩm trong công ty của bạn). Triển khai NER + EL từ đầu đến cuối. Đo lường precision và recall trên 100 câu được giữ lại.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Liên kết thực thể (EL) | Liên kết đến Wikipedia | Ánh xạ đề cập đến một mục nhập KB duy nhất. |
| Tạo ứng viên | Đó có thể là ai? | Trả về danh sách rút gọn các mục KB hợp lý để đề cập. |
| Định hướng | Chọn một trong những phù hợp | Chấm điểm ứng viên bằng ngữ cảnh, chọn người chiến thắng. |
| Chỉ mục bí danh | Bảng tra cứu | Bản đồ từ dạng bề mặt → các thực thể ứng cử viên. |
| KHÔNG | Không tính bằng KB | Dự đoán rõ ràng rằng không có mục nhập KB nào khớp. |
| KB | Cơ sở kiến thức | Wikidata, Wikipedia, DBpedia hoặc KB miền của bạn. |
| AIDA-CoNLL | Các benchmark | 1.393 bài báo của Reuters có liên kết thực thể vàng. |

## Đọc thêm

- [Milne, Witten (2008). Learning to Link with Wikipedia](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) - cách tiếp cận prior + ngữ cảnh cơ bản.
- [Wu et al. (2020). Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)](https://arxiv.org/abs/1911.03814) - con ngựa làm việc dựa trên embedding.
- [De Cao et al. (2021). Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904) — EL tổng quát với giải mã ràng buộc.
- [Hoffart et al. (2011). Robust Disambiguation of Named Entities in Text (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) - tờ báo benchmark.
- [REL: An Entity Linker Standing on the Shoulders of Giants (2020)](https://arxiv.org/abs/2006.01969) - production stack mở.
