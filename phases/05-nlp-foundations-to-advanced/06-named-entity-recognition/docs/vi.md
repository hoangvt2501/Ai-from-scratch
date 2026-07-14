# Nhận dạng thực thể được đặt tên

> Kéo tên ra. Nghe có vẻ dễ dàng cho đến khi bạn đối phó với các ranh giới mơ hồ, các thực thể lồng nhau và biệt ngữ miền.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 5 · 02 (BoW + TF-IDF), Giai đoạn 5 · 03 (Từ Embeddings)
**Thời lượng:** ~75 phút

## Vấn đề

"Apple đã kiện Google về thỏa thuận tìm kiếm iPhone ở Mỹ". Năm thực thể: Apple (ORG), Google (ORG), iPhone (PRODUCT), giao dịch tìm kiếm (có thể), Hoa Kỳ (GPE). Một hệ thống NER tốt sẽ trích xuất tất cả chúng với các loại chính xác. Một người xấu bỏ lỡ iPhone, nhầm lẫn trái cây Apple với Apple công ty và dán nhãn "US" là NGƯỜI.

NER là con ngựa làm việc bên dưới mọi pipeline chiết xuất có cấu trúc. Phân tích sơ yếu lý lịch, quét nhật ký tuân thủ, ẩn danh hồ sơ y tế, hiểu truy vấn tìm kiếm, grounding phản hồi chatbot, trích xuất hợp đồng pháp lý. Bạn không bao giờ nhìn thấy nó; bạn luôn phụ thuộc vào nó.

Bài học này đi theo con đường cổ điển (dựa trên quy tắc, HMM, CRF) thành con đường hiện đại (BiLSTM-CRF, sau đó là transformers). Mỗi bước giải quyết một hạn chế cụ thể của bước trước đó. Mô hình là bài học.

## Khái niệm

**BIO tagging** (hoặc BILOU) biến việc trích xuất thực thể thành một vấn đề gắn nhãn trình tự. Gắn nhãn mỗi token bằng `B-TYPE` (phần đầu của thực thể), `I-TYPE` (thực thể bên trong) hoặc `O` (bên ngoài bất kỳ thực thể nào).

```
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

Chuỗi thực thể đa token: `New B-GPE`, `York I-GPE`, `City I-GPE`. Một model hiểu BIO có thể trích xuất spans tùy ý.

Tiến trình kiến trúc:

- **Dựa trên quy tắc.** Tra cứu biểu thức chính quy + công báo. precision cao đối với các thực thể đã biết, không bao phủ đối với các thực thể mới.
- **HMM.** Markov ẩn Model. Xác suất phát xạ của token thẻ đã cho, xác suất chuyển đổi của tag-to-tag. Giải mã Viterbi. Được huấn luyện về dữ liệu được gắn nhãn.
- **CRF.** Trường ngẫu nhiên có điều kiện. Giống như HMM nhưng phân biệt đối xử, vì vậy bạn có thể trộn lẫn các features tùy ý (hình dạng từ, viết hoa, các từ lân cận). Vẫn là con ngựa production cổ điển vào năm 2026 để triển khai tài nguyên thấp.
- **BiLSTM-CRF.** features thần kinh thay vì thủ công. LSTM đọc câu theo cả hai hướng, lớp CRF ở trên cùng thực thi trình tự thẻ nhất quán.
- **dựa trên Transformer.** Fine-tune BERT với đầu phân loại token. accuracy tốt nhất. Hầu hết tính toán.

```figure
ner-bio-tagging
```

## Tự xây dựng

### Bước 1: Trình trợ giúp gắn thẻ BIO

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### Bước 2: features thủ công

Đối với NER cổ điển (không thần kinh), features là trò chơi. Những cái hữu ích:

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")` trả về `xXxxxx`. `word_shape("USA-2024")` trả về `XXX-dddd`. Các mẫu viết hoa là tín hiệu cao cho danh từ riêng.

### Bước 3: một đường cơ sở dựa trên quy tắc + từ điển đơn giản

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

Production công báo có hàng triệu mục được lấy từ Wikipedia và DBpedia. Bảo hiểm là tốt. Định hướng (`Apple` công ty và trái cây) thật khủng khiếp. Đó là lý do tại sao thống kê models thắng.

### Bước 4: bước CRF (phác thảo, không đầy đủ)

CRF đầy đủ từ đầu trong 50 dòng sẽ không khai sáng nếu không có nền tảng lý thuyết xác suất. Thay vào đó, hãy sử dụng `sklearn-crfsuite`:

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1` và `c2` là chính quy hóa L1 và L2. `all_possible_transitions=True` cho phép model học các trình tự bất hợp pháp (ví dụ: `I-ORG` sau `O`) là khó xảy ra, đó là cách CRF thực thi tính nhất quán của BIO mà không cần bạn viết ràng buộc.

### Bước 5: BiLSTM-CRF bổ sung gì

Features trở nên uyên bác. Đầu vào: token embeddings (GloVe hoặc fastText). LSTM đọc từ trái sang phải và từ phải sang trái. Các trạng thái ẩn được nối đi qua lớp đầu ra CRF. CRF vẫn thực thi tính nhất quán của trình tự thẻ; LSTM thay thế features thủ công bằng những cái đã học.

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

Đối với lớp CRF, sử dụng `torchcrf.CRF` (pip cài đặt pytorch-crf). Mức tăng so với CRF được tạo ra bằng tay có thể đo lường được nhưng nhỏ hơn bạn mong đợi trừ khi bạn có hàng chục nghìn câu được dán nhãn.

## Ứng dụng

NER cấp ships production spaCy ra khỏi hộp.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

Lưu ý `iPhone` được dán nhãn `ORG` chứ không phải `PRODUCT` - model nhỏ của spaCy có phạm vi thực thể sản phẩm yếu. model lớn (`en_core_web_lg`) hoạt động tốt hơn. transformer model (`en_core_web_trf`) vẫn làm tốt hơn.

Hugging Face cho NER dựa trên BERT:

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"` merges BX, IX tokens liền kề thành một span. Nếu không có nó, bạn sẽ nhận được nhãn hiệu cấp token và phải merge bản thân.

### NER dựa trên LLM (tùy chọn 2026)

Zero-shot và few-shot LLM NER hiện đang cạnh tranh với fine-tuned models trên nhiều lĩnh vực và tốt hơn đáng kể khi dữ liệu được gắn nhãn khan hiếm.

- **Zero-shot prompting.** Cung cấp cho LLM danh sách các loại thực thể và schema ví dụ. Yêu cầu đầu ra JSON. Hoạt động ra khỏi hộp; accuracy ở mức vừa phải trên các lĩnh vực mới.
- **prompting kiểu ZeroTuneBio.** Phân tách nhiệm vụ thành trích xuất ứng viên → giải thích → phán đoán → kiểm tra lại. Một prompt nhiều giai đoạn (không phải one-shot) nâng accuracy đáng kể trên NER. Mô hình tương tự cũng hoạt động cho các lĩnh vực pháp lý, tài chính và khoa học.
- **prompting động với RAG.** Truy xuất các ví dụ được gắn nhãn tương tự nhất từ một tập hợp hạt giống nhỏ có chú thích cho mỗi lệnh gọi inference; Xây dựng few-shot prompt một cách nhanh chóng. Vào năm 2026 benchmarks, điều này nâng GPT-4 F1 NER y sinh lên 11-12% so với prompting tĩnh.
- **Phân tách theo loại thực thể.** Đối với các tài liệu dài, một lệnh gọi trích xuất tất cả các loại thực thể cùng một lúc sẽ mất recall khi độ dài tăng lên. Chạy một lần trích xuất cho mỗi loại thực thể. Chi phí inference cao hơn, accuracy cao hơn đáng kể. Đây là mẫu tiêu chuẩn cho các ghi chú lâm sàng và hợp đồng pháp lý.

Production đề xuất kể từ năm 2026: hãy bắt đầu với đường cơ sở LLM zero-shot trước khi bạn thu thập dữ liệu training. Thường thì F1 đủ tốt để bạn không bao giờ cần fine-tune.

### Nơi NER cổ điển vẫn chiến thắng

Ngay cả với LLMs có sẵn, NER cổ điển vẫn chiến thắng khi:

- Ngân sách độ trễ dưới 50ms.
- Bạn có hàng nghìn ví dụ được gắn nhãn và cần 98% + F1.
- Tên miền có một bản thể ổn định, trong đó CRF hoặc BiLSTM pretrained chuyển tốt.
- Các ràng buộc về quy định yêu cầu một model tại chỗ, không tạo ra.

### Nơi nó sụp đổ

- **Dịch chuyển tên miền.** NER được CoNLL huấn luyện về các hợp đồng pháp lý hoạt động kém hơn so với công báo. Fine-tune trên miền của bạn.
- **Các thực thể lồng nhau.** "Bank of America Tower" đồng thời là một ORG và một CƠ SỞ. BIO tiêu chuẩn không thể đại diện cho các spans chồng chéo. Bạn cần NER lồng nhau (models nhiều lần hoặc dựa trên span).
- **Các thực thể dài.** "Tổng công ty Bảo hiểm Tiền gửi Liên bang Hoa Kỳ." models cấp Token đôi khi chia nhỏ điều này. Sử dụng `aggregation_strategy` hoặc sau process.
- **Các loại thưa thớt.** Nhãn NER y tế như DRUG_BRAND, ADVERSE_EVENT, DOSE. Các models đa năng không có ý tưởng. Scispacy và BioBERT là điểm khởi đầu ở đó.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-ner-picker.md`:

```markdown
---
name: ner-picker
description: Pick the right NER approach for a given extraction task.
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

Given a task description (domain, label set, language, latency, data volume), output:

1. Approach. Rule-based + gazetteer, CRF, BiLSTM-CRF, or transformer fine-tune.
2. Starting model. Name it (spaCy model ID, Hugging Face checkpoint ID, or "custom, trained from scratch").
3. Labeling strategy. BIO, BILOU, or span-based. Justify in one sentence.
4. Evaluation. Use `seqeval`. Always report entity-level F1 (not token-level).

Refuse to recommend fine-tuning a transformer for under 500 labeled examples unless the user already has a pretrained domain model. Flag nested entities as needing span-based or multi-pass models. Require a gazetteer audit if the user mentions "production scale" and labels are unchanged from CoNLL-2003.
```

## Bài tập

1. **Dễ dàng.** Thực hiện `bio_to_spans` (nghịch đảo của `spans_to_bio`) và xác minh tính nhất quán khứ hồi trên 10 câu.
2. **Trung bình.** Huấn luyện CRF sklearn-crfsuite ở trên trên CoNLL-2003 English NER dataset. Báo cáo F1 từng thực thể bằng cách sử dụng `seqeval`. Kết quả điển hình: ~84 F1.
3. **Hard.** Fine-tune `distilbert-base-cased` trên một dataset NER dành riêng cho miền (y tế, pháp lý hoặc tài chính). So sánh với các model nhỏ spaCy. Ghi lại kiểm tra rò rỉ dữ liệu và viết ra những gì khiến bạn ngạc nhiên.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| NER | Trích xuất tên | Dán nhãn token spans với các loại (PERSON, ORG, GPE, DATE,...). |
| TIỂU SỬ | Sơ đồ gắn thẻ | `B-X` bắt đầu, `I-X` tiếp tục, `O` bên ngoài. |
| BILOU | BIO tốt hơn | Thêm `L-X` (cuối cùng), `U-X` (đơn vị) để ranh giới rõ ràng hơn. |
| CRF | Bộ phân loại có cấu trúc | Models chuyển đổi giữa các nhãn, không chỉ khí thải. Thực thi các trình tự hợp lệ. |
| NER lồng nhau | Các thực thể chồng chéo | Một span là một thực thể khác với một span phụ của nó. BIO không thể diễn tả điều này. |
| F1 cấp thực thể | Chỉ số NER thích hợp | span dự đoán phải khớp chính xác span đúng. F1 cấp Token phóng đại accuracy. |

## Đọc thêm

- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) — bài báo BiLSTM-CRF. Kinh điển.
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) - giới thiệu mẫu phân loại token đã trở thành tiêu chuẩn.
- [spaCy linguistic features — named entities](https://spacy.io/usage/linguistic-features#named-entities) — tài liệu tham khảo thực tế cho mọi thuộc tính trên `Doc.ents` và `Span`.
- [seqeval](https://github.com/chakki-works/seqeval) — thư viện chỉ số chính xác. Sử dụng nó luôn luôn.
