# Xử lý văn bản - Tokenization, gốc, từ ngữ hóa

> Ngôn ngữ là liên tục. Models là rời rạc. Tiền xử lý là cầu nối.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 2 · 14 (Bayes ngây thơ)
**Thời lượng:** ~45 phút

## Vấn đề

Một model không thể đọc "Những con mèo đang chạy". Nó đọc các số nguyên.

Mỗi hệ thống NLP đều mở ra với ba câu hỏi giống nhau. Một từ bắt đầu từ đâu. Gốc của từ là gì. Làm thế nào để chúng ta coi "chạy", "chạy", "chạy" là cùng một thứ khi nó có ích, và là những thứ khác nhau khi nó không hữu ích.

Hiểu sai tokenization và model học hỏi từ rác. Nếu tokenizer của bạn coi `don't` là một token nhưng `do n't` là hai, phân phối training sẽ tách ra. Nếu stemmer của bạn sụp đổ `organization` và `organ` vào cùng một thân cây, mô hình chủ đề sẽ chết. Nếu lemmatizer của bạn cần ngữ cảnh một phần của bài phát biểu nhưng bạn không vượt qua nó, động từ sẽ được coi là danh từ.

Bài học này xây dựng ba bước tiền xử lý từ đầu, sau đó chỉ ra cách NLTK và spaCy thực hiện cùng một công việc để bạn có thể thấy sự đánh đổi.

## Khái niệm

Ba hoạt động. Mỗi người có một công việc và một chế độ thất bại.

**Tokenization** chia một chuỗi thành tokens. "Token" cố tình mơ hồ vì độ chi tiết phù hợp phụ thuộc vào nhiệm vụ. Cấp độ từ cho NLP cổ điển. Tiểu từ cho transformers. Ký tự cho các ngôn ngữ không có khoảng trắng.

**Stemming **cắt hậu tố với các quy tắc. Nhanh, hung hăng, ngu ngốc. `running -> run`. `organization -> organ`. Cái thứ hai đó là chế độ thất bại.

**Lemmatization** giảm một từ xuống dạng từ điển của nó bằng cách sử dụng kiến thức ngữ pháp. Chậm hơn, chính xác, cần có bảng tra cứu hoặc máy phân tích hình thái. `ran -> run` (cần biết "ran" là thì quá khứ của "run"). `better -> good` (cần biết các hình thức so sánh).

Quy tắc ngón tay cái. Thân cây khi tốc độ quan trọng và bạn có thể chịu được nhiễu (lập chỉ mục tìm kiếm, phân loại thô). Lemmatize khi ý nghĩa quan trọng (trả lời câu hỏi, tìm kiếm ngữ nghĩa, bất cứ thứ gì người dùng sẽ đọc).

```figure
edit-distance
```

## Tự xây dựng

### Bước 1: một từ regex tokenizer

Hữu ích đơn giản nhất tokenizer tách các ký tự không phải chữ và số trong khi vẫn giữ dấu câu làm tokens riêng. Không hoàn hảo, không phải là cuối cùng, nhưng nó chạy trong một dòng.

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

Ba mẫu theo thứ tự ưu tiên. Các từ có dấu nháy đơn bên trong tùy chọn (`don't`, `it's`). Những con số thuần túy. Bất kỳ ký tự không phải chữ và số không có khoảng trắng nào dưới dạng token độc lập (dấu câu).

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

Chế độ thất bại để nhận thấy. `3pm` chia thành `['3', 'pm']` vì chúng tôi xen kẽ giữa chạy chữ cái và chạy chữ số. Đủ tốt cho hầu hết các nhiệm vụ. URL, email, hashtag đều bị hỏng. Đối với production, hãy thêm các mẫu trước các mẫu chung.

### Bước 2: Porter stemmer (chỉ bước 1a)

Thuật toán Porter đầy đủ có năm giai đoạn quy tắc. Chỉ riêng bước 1a đã bao gồm các hậu tố tiếng Anh thường gặp nhất và dạy mẫu.

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

Đọc các quy tắc từ trên xuống. Quy tắc `ies -> i` là tại sao `ponies -> poni` chứ không phải `pony`. Real Porter có bước 1b để khắc phục nó. Các quy tắc cạnh tranh. Các quy tắc trước đó sẽ thắng. Thứ tự quan trọng hơn bất kỳ quy tắc đơn lẻ nào.

### Bước 3: một lemmatizer dựa trên tra cứu

Lemmatization thích hợp cần hình thái. Một phiên bản giảng dạy dễ xử lý sử dụng một bảng lemma nhỏ và một dự phòng.

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

Trường hợp cuối cùng là thời điểm giảng dạy quan trọng. `watched` không có trong bảng của chúng tôi và dự phòng của chúng tôi chỉ xử lý `ing`. Từ ngữ thực bao gồm `ed`, động từ bất quy tắc, tính từ so sánh, số nhiều có thay đổi âm thanh (`children -> child`). Đây là lý do tại sao các hệ thống production sử dụng WordNet, bộ hình thái của spaCy hoặc trình phân tích hình thái đầy đủ.

### Bước 4: nối chúng lại với nhau

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

Phần còn thiếu là một trình gắn thẻ POS. Giai đoạn 5 · 07 (POS Tagging) xây dựng một. Hiện tại, hãy mặc định mọi thứ `NOUN` và thừa nhận giới hạn.

## Ứng dụng

NLTK và spaCy ship các phiên bản production. Mỗi một vài dòng.

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize` xử lý các từ rút gọn, Unicode, các trường hợp cạnh mà biểu thức chính quy của bạn bỏ lỡ. `PorterStemmer` chạy cả năm giai đoạn. `WordNetLemmatizer` cần thẻ POS được dịch từ lược đồ Penn Treebank của NLTK sang bộ viết tắt của WordNet. Dây dịch ở trên là phần mà hầu hết các hướng dẫn bỏ qua.

### SpaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy giấu toàn bộ pipeline sau lưng `nlp(text)`. Tokenization, gắn thẻ POS và lemmatization đều chạy. Nhanh hơn NLTK trên quy mô lớn. Chính xác hơn khi ra khỏi hộp. Sự đánh đổi là bạn không thể dễ dàng hoán đổi các thành phần riêng lẻ.

### Khi nào nên chọn cái nào

| Tình huống | Chọn |
|-----------|------|
| Giảng dạy, nghiên cứu, hoán đổi linh kiện | NLTK |
| Production, đa ngôn ngữ, tốc độ rất quan trọng | SpaCy |
| Transformer pipeline (dù sao bạn cũng sẽ mã hóa bằng tokenizer của model) | Sử dụng `tokenizers` / `transformers` và bỏ qua tiền xử lý cổ điển |

### Hai chế độ hỏng hóc không ai cảnh báo bạn

Hầu hết các hướng dẫn dạy các thuật toán và dừng lại. Hai thứ sẽ cắn một pipeline tiền xử lý thực sự và chúng hầu như không bao giờ được bảo hiểm.

**Khả năng tái tạo trôi dạt.** NLTK và spaCy thay đổi tokenization và hành vi của lemmatizer giữa các phiên bản. Những gì tạo ra `['do', "n't"]` trong spaCy 2.x có thể tạo ra `["don't"]` trong 3.x. model của bạn đã được huấn luyện trên một bản phân phối. Inference bây giờ chạy trên một cái khác. Accuracy lặng lẽ xuống cấp và không ai biết tại sao. Ghim các phiên bản thư viện trong `requirements.txt`. Viết một bài kiểm tra hồi quy tiền xử lý đóng băng tokenization dự kiến của 20 câu mẫu. Chạy nó trên mỗi lần nâng cấp.

**Training / inference không khớp.** Huấn luyện với tiền xử lý tích cực (chữ thường, loại bỏ từ dừng, gốc gốc), triển khai trên đầu vào của người dùng thô, xem miệng núi lửa hiệu suất. Đây là lỗi production NLP phổ biến nhất. Nếu bạn xử lý trước trong khi training, bạn phải chạy chức năng giống hệt nhau trong inference. Ship tiền xử lý như một hàm bên trong gói model, không phải là một ô sổ ghi chép mà nhóm phục vụ viết lại.

## Sản phẩm bàn giao

Một prompt có thể tái sử dụng giúp các kỹ sư chọn chiến lược tiền xử lý mà không cần đọc ba sách giáo khoa.

Lưu dưới dạng `outputs/prompt-preprocessing-advisor.md`:

```markdown
---
name: preprocessing-advisor
description: Recommends a tokenization, stemming, and lemmatization setup for an NLP task.
phase: 5
lesson: 01
---

You advise on classical NLP preprocessing. Given a task description, you output:

1. Tokenization choice (regex, NLTK word_tokenize, spaCy, or transformer tokenizer). Explain why.
2. Whether to stem, lemmatize, both, or neither. Explain why.
3. Specific library calls. Name the functions. Quote the POS-tag translation if NLTK is involved.
4. One failure mode the user should test for.

Refuse to recommend stemming for user-visible text. Refuse to recommend lemmatization without POS tags. Flag non-English input as needing a different pipeline.
```

## Bài tập

1. **Dễ dàng.** Mở rộng `tokenize` để giữ URL dưới dạng tokens duy nhất. Kiểm tra: `tokenize("Visit https://example.com today.")` nên tạo một token URL.
2. **Trung bình.** Triển khai Porter bước 1b. Nếu một từ chứa nguyên âm và kết thúc bằng `ed` hoặc `ing`, hãy xóa nó. Xử lý quy tắc phụ âm kép (`hopping -> hop`, không phải `hopp`).
3. **Khó.** Xây dựng một lemmatizer sử dụng WordNet làm bảng tra cứu nhưng quay trở lại Porter stemmer của bạn khi WordNet không có mục nhập. Đo lường accuracy trên một kho dữ liệu được gắn thẻ so với WordNet đơn giản và Porter đơn giản.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Token | Một từ | Bất cứ đơn vị nào mà model tiêu thụ. Có thể là từ, từ phụ, ký tự hoặc byte. |
| Thân cây | Gốc của một từ | Kết quả của việc loại bỏ hậu tố dựa trên quy tắc. Không phải lúc nào cũng là một từ thật. |
| Lemma | Dạng từ điển | Biểu mẫu bạn sẽ tra cứu. Yêu cầu ngữ cảnh ngữ pháp để tính toán chính xác. |
| Thẻ POS | Một phần của bài phát biểu | Danh mục như NOUN, VERB, ADJ. Cần thiết để diễn đạt chính xác. |
| Hình thái học | Quy tắc hình dạng từ | Cách một từ thay đổi hình thức dựa trên thì, số, trường hợp. Lemmatization phụ thuộc vào nó. |

## Đọc thêm

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) - bài báo gốc, năm trang, vẫn là lời giải thích rõ ràng nhất.
- [spaCy 101 — linguistic features](https://spacy.io/usage/linguistic-features) - một pipeline thực sự được kết nối như thế nào.
- [NLTK book, chapter 3](https://www.nltk.org/book/ch03.html) - tokenization trường hợp cạnh mà bạn chưa nghĩ đến.
