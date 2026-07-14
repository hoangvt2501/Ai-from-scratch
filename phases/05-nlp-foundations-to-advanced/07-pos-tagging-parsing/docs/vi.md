# Gắn thẻ POS và phân tích cú pháp

> Ngữ pháp đã không hợp thời trong một thời gian. Sau đó, mọi LLM pipeline cần xác thực trích xuất có cấu trúc và nó đã quay trở lại.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 5 · 01 (Xử lý văn bản), Giai đoạn 2 · 14 (Bayes ngây thơ)
**Thời lượng:** ~45 phút

## Vấn đề

Bài 01 hứa hẹn rằng lemmatization cần một thẻ một phần của bài phát biểu. Nếu không biết `running` là một động từ, một lemmatizer không thể rút gọn nó thành `run`. Nếu không biết `better` là một tính từ, nó không thể rút gọn thành `good`.

Lời hứa đó đã che giấu cả một lĩnh vực phụ. Gắn thẻ một phần của bài phát biểu chỉ định các danh mục ngữ pháp. Phân tích cú pháp khôi phục cấu trúc cây của câu: từ nào sửa đổi từ nào, động từ nào chi phối đối số nào. Classical NLP đã dành hai mươi năm để tinh chỉnh cả hai. Sau đó, deep learning thu gọn chúng thành một nhiệm vụ phân loại token trên một pretrained transformer và cộng đồng nghiên cứu tiếp tục.

Không phải cộng đồng ứng dụng. Mọi pipeline trích xuất có cấu trúc vẫn sử dụng POS và cây phụ thuộc. JSON do LLM tạo được xác thực dựa trên các ràng buộc ngữ pháp. Hệ thống trả lời câu hỏi phân tách các truy vấn bằng cách sử dụng phân tích cú pháp phần phụ thuộc. Trình đánh giá chất lượng dịch máy kiểm tra alignment của cây phân tích cú pháp.

Đáng biết. Bài học này giới thiệu các bộ thẻ, đường cơ sở và điểm mà bạn ngừng triển khai từ đầu và gọi spaCy.

## Khái niệm

**Gắn thẻ POS** gắn nhãn mỗi token với một danh mục ngữ pháp. Bộ thẻ **Penn Treebank (PTB)** là mặc định bằng tiếng Anh. 36 thẻ với sự khác biệt mà người đọc bình thường thấy cầu kỳ: `NN` danh từ số ít, danh từ số nhiều `NNS`, danh từ riêng `NNP` số ít, `VBD` động từ thì quá khứ, động từ `VBZ` ngôi thứ 3 số ít hiện tại, v.v. Bộ thẻ **Universal Dependencies (UD)** thô hơn (17 thẻ) và bất khả tri về ngôn ngữ; nó đã trở thành mặc định cho công việc đa ngôn ngữ.

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**Phân tích cú pháp** tạo ra một cái cây. Hai phong cách chính:

- **Phân tích cú pháp cử tri.** Cụm danh từ, cụm động từ, cụm giới từ lồng vào nhau. Đầu ra là một cây các thể loại không phải đầu cuối (NP, VP, PP) với các từ là lá.
- **Phân tích cú pháp phụ thuộc.** Mỗi từ có một từ tiêu đề duy nhất mà nó phụ thuộc, được gắn nhãn bằng mối quan hệ ngữ pháp. Đầu ra là một cây trong đó mọi cạnh là một bộ ba (đầu, phụ thuộc, quan hệ).

Phân tích cú pháp phụ thuộc đã giành chiến thắng trong những năm 2010 vì nó khái quát hóa rõ ràng trên các ngôn ngữ, đặc biệt là các ngôn ngữ có trật tự từ.

```
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

## Tự xây dựng

### Bước 1: Đường cơ sở thẻ thường xuyên nhất

Trình gắn thẻ POS ngu ngốc nhất hoạt động. Đối với mỗi từ, hãy dự đoán thẻ mà nó có thường xuyên nhất trong training.

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

Trên kho dữ liệu Brown, đường cơ sở này đạt ~85% accuracy. Không tốt, nhưng sàn dưới đó không có model nghiêm trọng nào rơi xuống.

### Bước 2: trình gắn thẻ bigram HMM

Model xác suất chung của trình tự:

```
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

Hai bảng: xác suất chuyển tiếp (thẻ đã cho thẻ trước đó), xác suất phát xạ (thẻ đã cho từ). Ước tính cả hai từ số đếm với làm mịn Laplace. Giải mã bằng Viterbi (lập trình động trên mạng thẻ).

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

Bigram HMM trên Brown đạt ~93% accuracy. Bước nhảy vọt từ 85% lên 93% chủ yếu là xác suất chuyển đổi - model học được `DET NOUN` phổ biến và `NOUN DET` hiếm gặp.

### Bước 3: tại sao những người gắn thẻ hiện đại đánh bại điều này

Xác suất chuyển đổi + phát thải là cục bộ. Họ không thể nắm bắt được rằng `saw` là một danh từ trong "Tôi đã mua một chiếc cưa" mà là một động từ trong "Tôi đã xem bộ phim". Một CRF có features tùy ý (hậu tố, hình dạng từ, từ trước và sau, chính từ) đạt ~97%. BiLSTM-CRF hoặc transformer đạt ~98% +.

Giới hạn của nhiệm vụ này được thiết lập bởi sự bất đồng của người chú thích. Người chú thích đồng ý khoảng 97% thời gian trên Penn Treebank. Models 98% trong quá khứ có thể overfitting bộ thử nghiệm.

### Bước 4: phác thảo phân tích cú pháp phụ thuộc

Phân tích cú pháp phụ thuộc đầy đủ từ đầu nằm ngoài phạm vi; cách xử lý sách giáo khoa kinh điển là trong Jurafsky và Martin. Hai gia đình cổ điển cần biết:

- **Trình phân tích cú pháp dựa trên chuyển tiếp** (arc-eager, arc-standard) hoạt động giống như một trình phân tích cú pháp giảm dịch chuyển: chúng đọc tokens, chuyển chúng sang stack và áp dụng các hành động giảm tạo vòng cung. Giải mã tham lam rất nhanh. Triển khai cổ điển là MaltParser. Phiên bản thần kinh hiện đại: Trình phân tích cú pháp dựa trên chuyển tiếp của Chen và Manning.
- **Dựa trên đồ thị** trình phân tích cú pháp (thuật toán của Eisner, Dozat-Manning biaffine) chấm điểm mọi cạnh phụ thuộc vào đầu có thể có và chọn cây kéo dài tối đa. Chậm hơn nhưng chính xác hơn.

Đối với hầu hết các công việc được áp dụng, hãy gọi spaCy:

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

Đọc cột `dep` từ dưới lên trên và cấu trúc ngữ pháp của câu sẽ rơi ra.

## Ứng dụng

Mỗi thư viện production NLP ships POS và trình phân tích cú pháp phụ thuộc như một phần của pipeline tiêu chuẩn.

- **spaCy** (`en_core_web_sm` / `md` / `lg` / `trf`). Nhanh chóng, chính xác, tích hợp với tokenization + NER + lemmatization. `token.tag_` (Penn), `token.pos_` (UD), `token.dep_` (quan hệ phụ thuộc).
- **Stanford NLP (khổ thơ)**. Người kế nhiệm CoreNLP của Stanford. Hiện đại trên 60+ ngôn ngữ.
- **trankit**. Dựa trên Transformer, UD accuracy tốt.
- **NLTK**. `pos_tag`. Có thể sử dụng, chậm, cũ hơn. Tốt cho việc giảng dạy.

### Điều này vẫn quan trọng ở đâu vào năm 2026

- **Lemmatization.** Bài 01 cần POS để lemmatization chính xác. Luôn luôn.
- **Trích xuất có cấu trúc từ LLM đầu ra.** Xác nhận rằng một câu được tạo tôn trọng các ràng buộc ngữ pháp (ví dụ: thỏa thuận chủ ngữ-động từ, công cụ sửa đổi bắt buộc).
- **Cảm xúc dựa trên khía cạnh.** Phân tích cú pháp phụ thuộc cho bạn biết tính từ nào sửa đổi danh từ nào.
- **Hiểu truy vấn.** "phim do Wes Anderson đạo diễn với sự tham gia của Bill Murray" phân hủy thành các ràng buộc có cấu trúc thông qua phân tích cú pháp.
- **Chuyển giao đa ngôn ngữ.** Thẻ UD và mối quan hệ phụ thuộc không phụ thuộc vào ngôn ngữ, cho phép phân tích zero-shot cấu trúc các ngôn ngữ mới.
- **pipelines điện toán thấp.** Nếu bạn không thể ship transformer, POS + dependency parse + gazetteer sẽ giúp bạn tiến xa một cách đáng ngạc nhiên.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-grammar-pipeline.md`:

```markdown
---
name: grammar-pipeline
description: Design a classical POS + dependency pipeline for a downstream NLP task.
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

Given a downstream task (information extraction, rewrite validation, query decomposition, lemmatization), you output:

1. Tagset to use. Penn Treebank for English-only legacy pipelines, Universal Dependencies for multilingual or cross-lingual.
2. Library. spaCy for most production, stanza for academic-grade multilingual, trankit for highest UD accuracy. Name the specific model ID.
3. Integration pattern. Show the 3-5 lines that call the library and consume the needed attributes (`.pos_`, `.dep_`, `.head`).
4. Failure mode to test. Noun-verb ambiguity (`saw`, `book`, `can`) and PP-attachment ambiguity are the classical traps. Sample 20 outputs and eyeball.

Refuse to recommend rolling your own parser. Building parsers from scratch is a research project, not an application task. Flag any pipeline that consumes POS tags without handling lowercase/uppercase variants as fragile.
```

## Bài tập

1. **Dễ dàng.** Sử dụng đường cơ sở thẻ thường xuyên nhất trên một kho dữ liệu nhỏ được gắn thẻ (ví dụ: tập con Brown của NLTK), đo lường accuracy trên các câu được giữ lại. Xác minh kết quả ~85%.
2. **Trung bình.** Huấn luyện bigram HMM ở trên và báo cáo mỗi thẻ precision/recall. HMM gây nhầm lẫn nhất cho thẻ nào?
3. **Khó.** Sử dụng phân tích cú pháp phụ thuộc của spaCy để trích xuất bộ ba chủ ngữ-động từ-đối tượng từ một mẫu 1000 câu. Đánh giá trên 50 bộ ba được dán nhãn thủ công. Tài liệu khi trích xuất không thành công (thường là thụ động, phối hợp và đối tượng bị loại bỏ).

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Thẻ POS | Loại từ | Danh mục ngữ pháp. PTB có 36; UD có 17. |
| Ngân hàng cây Penn | Bộ thẻ tiêu chuẩn | Tiếng Anh cụ thể. Thì động từ hạt mịn và số danh từ. |
| Phụ thuộc phổ quát | Bộ thẻ đa ngôn ngữ | Thô hơn PTB; trung lập ngôn ngữ; mặc định cho công việc đa ngôn ngữ. |
| Phân tích cú pháp phần phụ thuộc | Cây câu | Mỗi từ có một đầu, mỗi cạnh có một mối quan hệ ngữ pháp. |
| Viterbi | Lập trình động | Tìm trình tự thẻ có xác suất cao nhất cho phát xạ và chuyển tiếp. |

## Đọc thêm

- [Jurafsky and Martin — Speech and Language Processing, chapters 8 and 18](https://web.stanford.edu/~jurafsky/slp3/) - cách xử lý sách giáo khoa chuẩn về POS và phân tích cú pháp.
- [Universal Dependencies project](https://universaldependencies.org/) — bộ thẻ đa ngôn ngữ và bộ sưu tập ngân hàng cây được sử dụng bởi mọi trình phân tích cú pháp đa ngôn ngữ.
- [spaCy linguistic features guide](https://spacy.io/usage/linguistic-features) — tài liệu tham khảo thực tế cho mọi thuộc tính được hiển thị trên `Token`.
- [Chen and Manning (2014). A Fast and Accurate Dependency Parser using Neural Networks](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) - bài báo đưa trình phân tích cú pháp thần kinh trở thành xu hướng chủ đạo.
