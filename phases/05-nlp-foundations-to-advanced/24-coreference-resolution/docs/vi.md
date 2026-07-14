# Độ phân giải đồng tham chiếu

> "Cô ấy đã gọi cho anh ấy. Anh ta không trả lời. Bác sĩ đang ăn trưa." Ba đề cập đến hai người và không ai được nêu tên. Độ phân giải đồng tham chiếu tìm ra ai là ai.

**Loại:** Học
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 5 · 06 (NER), Giai đoạn 5 · 07 (POS & Phân tích cú pháp)
**Thời lượng:** ~60 phút

## Vấn đề

Trích xuất mọi đề cập đến Apple Inc. từ một bài báo dài 300 từ. Dễ dàng khi bài báo nói "Apple". Khó khăn khi nói "công ty", "họ", "gã khổng lồ công nghệ của Cupertino" hoặc "công ty của Jobs". Nếu không giải quyết các đề cập này cho cùng một thực thể, pipeline NER của bạn sẽ bỏ lỡ 60-80% số đề cập.

Độ phân giải đồng tham chiếu liên kết mọi biểu thức đề cập đến cùng một thực thể trong thế giới thực thành một cụm. Nó là chất kết dính giữa NLP cấp bề mặt (NER, phân tích cú pháp) và ngữ nghĩa hạ nguồn (IE, QA, tóm tắt, KG).

Tại sao nó lại quan trọng vào năm 2026:

- Tóm tắt: "Giám đốc điều hành thông báo..." so với "Tim Cook đã thông báo..." — bản tóm tắt nên nêu tên CEO.
- Trả lời câu hỏi: "Cô ấy đã gọi ai?" đòi hỏi phải giải quyết "cô ấy".
- Trích xuất thông tin: một biểu đồ tri thức với "PER1 thành lập Apple" và "Jobs thành lập Apple" là các mục riêng biệt là sai.
- IE nhiều tài liệu: hợp nhất các đề cập trên các bài viết về cùng một sự kiện là đồng tham chiếu tài liệu.

## Khái niệm

![Coreference clustering: mentions → entities](../assets/coref.svg)

**Nhiệm vụ.** Đầu vào: một tài liệu. Đầu ra: một cụm đề cập (spans) trong đó mỗi cụm đề cập đến một thực thể.

**Các loại đề cập.**

- **Thực thể được đặt tên.** "Tim Cook"
- **Danh nghĩa.** "CEO", "công ty"
- **Danh từ.** "anh ấy", "cô ấy", "họ", "nó"
- **"Tim Cook, Giám đốc điều hành của Apple,"

**Kiến trúc.**

1. **Dựa trên quy tắc (Hobbs, 1978).** Độ phân giải đại từ dựa trên cây cú pháp sử dụng các quy tắc ngữ pháp. Đường cơ sở tốt. Khó đánh bại một cách đáng ngạc nhiên về đại từ.
2. **Bộ phân loại cặp đề cập.** Đối với mỗi cặp đề cập (m_i, m_j), hãy dự đoán xem chúng có tham chiếu chính hay không. Cụm bằng cách đóng chuyển tiếp. Tiêu chuẩn trước năm 2016.
3. **Xếp hạng đề cập.** Đối với mỗi đề cập, xếp hạng tiền đề của ứng cử viên (bao gồm cả "không có tiền đề"). Chọn phần trên.
4. **Kết thúc từ đầu đến cuối dựa trên Span (Lee và cộng sự, 2017).** Transformer encoder. Liệt kê tất cả các ứng cử viên spans giới hạn độ dài. Dự đoán điểm đề cập. Dự đoán xác suất tiền đề cho mỗi span. Cụm tham lam. Mặc định hiện đại.
5. **Generative (2024+).** Prompt một LLM: "Liệt kê mọi đại từ trong văn bản này và tiền đề của nó." Hoạt động tốt trên các trường hợp dễ dàng, gặp khó khăn trên các tài liệu dài và các tài liệu tham khảo hiếm hoi.

**Các chỉ số đánh giá.** Năm chỉ số tiêu chuẩn (MUC, B³, CEAF, BLANC, LEA) vì không có chỉ số nào nắm bắt được chất lượng phân cụm. Báo cáo mức trung bình của ba đầu tiên là CoNLL F1. Hiện đại vào năm 2026 trên CoNLL-2012: ~83 F1.

**Các trường hợp khó đã biết.**

- Mô tả xác định đề cập đến các thực thể được giới thiệu các trang trước đó.
- Cầu nối anaphora ("bánh xe" → một chiếc xe đã đề cập trước đó).
- Không có anaphora trong các ngôn ngữ như tiếng Trung và tiếng Nhật.
- Cataphora (đại từ trước tham chiếu): "Khi **cô **bước vào, Mary mỉm cười."

## Tự xây dựng

### Bước 1: pretrained đồng tham chiếu thần kinh (AllenNLP / spaCy-experimental)

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # experimental model
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

Trên một tài liệu dài hơn, bạn sẽ nhận được một cái gì đó như:
- Cụm 1: [Apple, Công ty, họ]
- Cụm 2: [sản phẩm mới]

### Bước 2: trình phân giải đại từ dựa trên quy tắc (giảng dạy)

Xem `code/main.py` để biết cách triển khai chỉ dành cho stdlib:

1. Đề cập trích dẫn: các thực thể được đặt tên (viết hoa spans), đại từ (tra cứu dict), mô tả xác định ("chữ X").
2. Đối với mỗi đại từ, hãy nhìn vào K đề cập trước đó và chấm điểm chúng theo:
   - gender/number thỏa thuận (heuristic)
   - gần đây (gần hơn thắng)
   - vai trò cú pháp (ưu tiên đối tượng)
3. Liên kết tiền đề có điểm cao nhất.

Không cạnh tranh với models thần kinh. Nhưng nó cho thấy không gian tìm kiếm và các quyết định mà model phải đưa ra từ đầu đến cuối.

### Bước 3: sử dụng LLMs để đồng tham khảo

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

Hai chế độ thất bại để xem. Đầu tiên, LLMs quá merge ("anh ấy" và "cô ấy" đề cập đến hai người riêng biệt). Thứ hai, LLMs lặng lẽ bỏ đề cập trong các tài liệu dài. Luôn xác minh bằng kiểm tra bù đắp span.

### Bước 4: đánh giá

Tiêu chuẩn conll-2012 script tính toán MUC, B³, CEAF-φ4 và báo cáo giá trị trung bình. Đối với đánh giá nội bộ, hãy bắt đầu với precision cấp span và recall trên bộ kiểm tra có chú thích của bạn, sau đó thêm F1 liên kết đề cập.

## Cạm bẫy

- **Vụ nổ Singleton.** Một số hệ thống báo cáo mọi đề cập là cụm riêng của nó. B³ là khoan dung. MUC trừng phạt điều này. Luôn kiểm tra cả ba chỉ số.
- **Đại từ trong ngữ cảnh dài.** Hiệu suất giảm ~15 F1 trên tài liệu trên 2.000 tokens. Chunk cẩn thận.
- **Giả định giới tính.** Các quy tắc giới tính được mã hóa cứng vi phạm đối với các đối tượng tham chiếu phi nhị phân, tổ chức, động vật. Sử dụng models đã học hoặc tính điểm trung lập.
- **LLM trôi dạt trên các tài liệu dài.** Một cuộc gọi API duy nhất không thể phân nhóm các đề cập trên 50+ đoạn văn một cách đáng tin cậy. Sử dụng cửa sổ trượt + merge.

## Ứng dụng

stack năm 2026:

| Tình huống | Chọn |
|-----------|------|
| Tiếng Anh, tài liệu đơn lẻ | Lõi thần kinh `en_coreference_web_trf` (spaCy-experimental) hoặc AllenNLP |
| Đa ngôn ngữ | SpanBERT / XLM-R được huấn luyện trên OntoNotes hoặc CoNLL đa ngôn ngữ |
| Tham chiếu sự kiện chéo tài liệu | models đầu cuối chuyên biệt (2025–26 SOTA) |
| Đường cơ sở LLM nhanh | GPT-4o / Claude với prompt coref đầu ra có cấu trúc |
| Production hệ thống hộp thoại | Dự phòng dựa trên quy tắc + sơ cấp thần kinh + đánh giá thủ công cho các vị trí quan trọng |

Mô hình tích hợp ships vào năm 2026: chạy NER trước, chạy coref merge các cụm coref vào các thực thể NER. Các tác vụ xuôi dòng nhìn thấy một thực thể trên mỗi cụm, không phải một thực thể cho mỗi đề cập.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-coref-picker.md`:

```markdown
---
name: coref-picker
description: Pick a coreference approach, evaluation plan, and integration strategy.
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

Given a use case (single-doc / multi-doc, domain, language), output:

1. Approach. Rule-based / neural span-based / LLM-prompted / hybrid. One-sentence reason.
2. Model. Named checkpoint if neural.
3. Integration. Order of operations: tokenize → NER → coref → downstream task.
4. Evaluation. CoNLL F1 (MUC + B³ + CEAF-φ4 average) on held-out set + manual cluster review on 20 documents.

Refuse LLM-only coref for documents over 2,000 tokens without sliding-window merge. Refuse any pipeline that runs coref without a mention-level precision-recall report. Flag gender-heuristic systems deployed in demographically diverse text.
```

## Bài tập

1. **Dễ dàng.** Chạy trình phân giải dựa trên quy tắc trong `code/main.py` trên 5 đoạn văn thủ công. Đo lường accuracy liên kết đề cập so với ground truth.
2. **Phương tiện.** Sử dụng một coref thần kinh pretrained model trên một bài báo. So sánh các cụm với chú thích thủ công của riêng bạn. Nó đã thất bại ở đâu?
3. **Khó khăn.** Xây dựng một pipeline NER tăng cường coref: NER trước, sau đó merge thông qua các cụm coref. Đo lường cải thiện độ bao phủ thực thể so với chỉ NER trên 100 bài viết.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Đề cập | Tài liệu tham khảo | Một span văn bản đề cập đến một thực thể (tên, đại từ, cụm danh từ). |
| Tiền thân | "nó" đề cập đến điều gì | Đề cập trước đó một cái sau đề cập với. |
| Cụm | Đề cập của thực thể | Tập hợp các đề cập đều đề cập đến cùng một thực thể trong thế giới thực. |
| Anaphora | Tham chiếu ngược | Đề cập sau này đề cập đến trước đó ("anh ấy" → "Giăng"). |
| Cataphora | Tham chiếu chuyển tiếp | Đề cập trước đó đề cập đến sau này ("Khi anh ấy đến, John..."). |
| Cầu nối | Tham chiếu ngầm | "Tôi đã mua một chiếc xe hơi. Bánh xe rất tệ." (bánh xe của chiếc xe đó.) |
| CoNLL F1 | Con số trên bảng xếp hạng | Điểm trung bình của MUC, B³, CEAF-φ4 F1. |

## Đọc thêm

- [Jurafsky & Martin, SLP3 Ch. 26 — Coreference Resolution and Entity Linking](https://web.stanford.edu/~jurafsky/slp3/26.pdf) — chương sách giáo khoa kinh điển.
- [Lee et al. (2017). End-to-end Neural Coreference Resolution](https://arxiv.org/abs/1707.07045) - đầu cuối dựa trên span.
- [Joshi et al. (2020). SpanBERT](https://arxiv.org/abs/1907.10529) - pretraining cải thiện coref.
- [Pradhan et al. (2012). CoNLL-2012 Shared Task](https://aclanthology.org/W12-4501/) - benchmark.
- [Hobbs (1978). Resolving Pronoun References](https://www.sciencedirect.com/science/article/pii/0024384178900064) — cổ điển dựa trên quy tắc.
