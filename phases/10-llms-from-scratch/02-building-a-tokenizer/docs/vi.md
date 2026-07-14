# Xây dựng Tokenizer từ đầu

> Bài 01 cho bạn một món đồ chơi. Bài học này cho bạn một vũ khí.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 10, Bài 01 (Tokenizers: BPE, WordPiece, SentencePiece)
**Thời lượng:** ~90 phút

## Mục tiêu học tập

- Xây dựng BPE tokenizer cấp production xử lý Unicode, chuẩn hóa khoảng trắng và tokens đặc biệt
- Triển khai dự phòng cấp byte để tokenizer có thể mã hóa bất kỳ đầu vào nào (bao gồm biểu tượng cảm xúc, CJK và mã) mà không cần tokens
- Thêm các mẫu biểu thức chính quy tokenization trước để tách văn bản tại ranh giới từ trước khi áp dụng BPE merges
- Huấn luyện một tokenizer tùy chỉnh trên một kho dữ liệu và đánh giá tỷ lệ nén của nó so với tiktoken trên văn bản đa ngôn ngữ

## Vấn đề

Bài BPE tokenizer của bạn từ Bài 01 hoạt động trên văn bản tiếng Anh. Bây giờ hãy ném tiếng Nhật vào nó. Hoặc biểu tượng cảm xúc. Hoặc Python mã với các tab và khoảng trắng hỗn hợp.

Nó bị vỡ.

Không phải vì BPE sai - bởi vì việc triển khai không hoàn chỉnh. Một production tokenizer xử lý các byte thô trong bất kỳ mã hóa nào, chuẩn hóa Unicode trước khi tách, quản lý các tokens đặc biệt không bao giờ được merged, chuỗi trước tokenization với tách từ phụ và thực hiện tất cả những điều này đủ nhanh để không làm tắc nghẽn training pipeline xử lý 15 nghìn tỷ tokens.

tokenizer của GPT-2 có 50.257 tokens. Llama 3 có 128.256. GPT-4 có khoảng 100.000. Đây không phải là những con số đồ chơi. Các bảng merge đằng sau những từ vựng đó được huấn luyện trên hàng trăm gigabyte văn bản và máy móc xung quanh - chuẩn hóa, chuẩn hóa tokenization, chèn token đặc biệt, định dạng mẫu trò chuyện - là thứ phân biệt một tokenizer xử lý "hello world" với một  xử lý toàn bộ internet.

Bạn sẽ chế tạo máy móc đó.

## Khái niệm

### Toàn bộ Pipeline

Một production tokenizer không phải là một thuật toán. Nó là một pipeline gồm năm giai đoạn, mỗi giai đoạn giải quyết một vấn đề khác nhau.

```mermaid
graph LR
    A[Raw Text] --> B[Normalize]
    B --> C[Pre-Tokenize]
    C --> D[BPE Merge]
    D --> E[Special Tokens]
    E --> F[Token IDs]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

Mỗi giai đoạn có một công việc cụ thể:

| Sân khấu | Chức năng của nó | Tại sao nó lại quan trọng |
|-------|-------------|----------------|
| Chuẩn hóa | NFKC Unicode, chữ thường tùy chọn, dải điểm nhấn tùy chọn | Chữ ghép "fi" (U + FB01) trở thành "fi" (hai ký tự). Nếu không có điều này, cùng một từ sẽ có tokens khác nhau. |
| Mã hóa trước | Chia văn bản thành các phần trước khi BPE | Ngăn BPE hợp nhất qua ranh giới từ. "Con mèo" không bao giờ được tạo ra một token "EC". |
| BPE Merge | Áp dụng các quy tắc merge đã học cho chuỗi byte | Nén cốt lõi. Biến byte thô thành tokens từ phụ. |
| Tokens đặc biệt | Chèn [BOS], , [PAD], Điểm đánh dấu mẫu trò chuyện | Những tokens này có ID cố định. Họ không bao giờ tham gia vào BPE merges. Người model cần chúng để cấu trúc. |
| Ánh xạ ID | Chuyển đổi chuỗi token thành ID số nguyên | model nhìn thấy số nguyên, không phải chuỗi. |

### BPE cấp độ byte

Bài học 01 tokenizer hoạt động trên UTF-8 byte. Đó là quyết định đúng. Nhưng chúng tôi đã bỏ qua một điều quan trọng: điều gì sẽ xảy ra khi những byte đó không hợp lệ UTF-8?

BPE cấp độ byte giải quyết vấn đề này bằng cách coi mọi giá trị byte có thể có (0-255) là một token hợp lệ. Từ vựng cơ bản của bạn chính xác là 256 mục. Bất kỳ tệp nào -- văn bản, nhị phân, bị hỏng -- đều có thể được mã hóa mà không tạo ra một token không xác định.

GPT-2 đã thêm một thủ thuật: ánh xạ mỗi byte với một ký tự Unicode có thể in được để từ vựng vẫn có thể đọc được. Byte 0x20 (dấu cách) trở thành ký tự "G" trong ánh xạ của chúng. Đây hoàn toàn là thẩm mỹ. Thuật toán không quan tâm.

Sức mạnh thực sự: BPE cấp byte xử lý mọi ngôn ngữ trên trái đất. Các ký tự Trung Quốc là 3 UTF-8 byte mỗi ký tự. Tiếng Nhật có thể là 3-4 byte. Tiếng Ả Rập, Devanagari, biểu tượng cảm xúc - tất cả chỉ là chuỗi byte. Thuật toán BPE tìm các mẫu trong các chuỗi byte này giống hệt như cách nó tìm các mẫu trong các byte ASCII tiếng Anh.

### Tokenization trước

Trước khi BPE chạm vào văn bản của mình, bạn cần chia nó thành nhiều phần. Điều này ngăn thuật toán merge tạo tokens span ranh giới từ.

GPT-2 sử dụng mẫu biểu thức chính quy để tách văn bản:

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

Mẫu này tách ra theo các từ rút gọn ("don't" trở thành "don" + "'t"), các từ có khoảng trắng, số, dấu câu và khoảng trắng tùy chọn. Khoảng trắng đầu được gắn vào từ -- vì vậy "the cat" trở thành [" the", " cat"], không phải ["the", ", "cat"].

Llama sử dụng SentencePiece, bỏ qua hoàn toàn regex. Nó coi luồng byte thô như một chuỗi dài và cho phép thuật toán BPE tìm ra ranh giới. Điều này đơn giản hơn nhưng cho phép BPE tự do hơn để tạo tokens ô chữ.

Sự lựa chọn quan trọng. Biểu thức chính quy của GPT-2 ngăn tokenizer học rằng "the" ở cuối một từ và "the" ở đầu từ tiếp theo nên merge. SentencePiece cho phép điều đó, đôi khi tạo ra nén hiệu quả hơn nhưng ít tokens dễ giải thích hơn.

### Tokens đặc biệt

Mỗi production tokenizer dành token ID cho các điểm đánh dấu cấu trúc:

| Token | Mục đích | Được sử dụng bởi |
|-------|---------|---------|
| `[BOS]` / `<s>` | Bắt đầu trình tự | Llama 3, GPT |
| `[EOS]` / `</s>` | Kết thúc trình tự | Tất cả models |
| `[PAD]` | Đệm cho batch alignment | BERT, T5 |
| `[UNK]` | token không xác định (BPE cấp byte loại bỏ điều này) | BERT, WordPiece |
| `<\ | im_start\ | >` | Bắt đầu ranh giới tin nhắn trò chuyện | ChatGPT, Qwen |
| `<\ | im_end\ | >` | Kết thúc ranh giới tin nhắn trò chuyện | ChatGPT, Qwen |
| `<\ | người dùng\ | >` | Điểm đánh dấu lượt người dùng | Llama 3 |
| `<\ | Trợ lý\ | >` | Điểm đánh dấu rẽ trợ lý | Llama 3 |

Các tokens đặc biệt không bao giờ được chia theo BPE. Chúng được khớp chính xác trước khi thuật toán merge chạy, được thay thế bằng ID cố định của chúng và văn bản xung quanh được mã hóa bình thường.

### Mẫu trò chuyện

Đây là nơi hầu hết mọi người bị nhầm lẫn và hầu hết các triển khai bị hỏng.

Khi bạn gửi tin nhắn đến model trò chuyện, API chấp nhận danh sách tin nhắn:

```
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

model không nhìn thấy JSON. Nó thấy một chuỗi token phẳng. Mẫu trò chuyện chuyển đổi tin nhắn thành chuỗi phẳng đó bằng cách sử dụng tokens đặc biệt. Mỗi model làm điều này khác nhau:

```
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT:
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

Làm sai bản mẫu và model tạo ra rác. Nó được huấn luyện trên một định dạng chính xác. Bất kỳ sai lệch nào -- thiếu dòng mới, token hoán đổi, thêm khoảng trắng -- đặt đầu vào bên ngoài phân phối training.

### Tốc độ

Python quá chậm đối với production tokenization.

TikToken (OpenAI) được viết bằng Rust với Python ràng buộc. HuggingFace tokenizers cũng Rust. SentencePiece là C++. Chúng đạt được tốc độ gấp 10-100 lần so với Python thuần túy.

Đối với góc nhìn: mã hóa 15 nghìn tỷ tokens cho Llama 3 training trước ở mức 1 triệu tokens mỗi giây (Python nhanh) sẽ mất 174 ngày. Ở mức 100 triệu tokens mỗi giây (Rust), mất 1,7 ngày.

Bạn đang xây dựng Python để hiểu thuật toán. Trong production, bạn sẽ sử dụng một triển khai đã biên dịch và chỉ chạm vào trình bao bọc Python.

```figure
weight-tying
```

## Tự xây dựng

### Bước 1: Mã hóa cấp độ byte

Nền tảng. Chuyển đổi bất kỳ chuỗi nào thành một chuỗi byte, ánh xạ mỗi byte thành một ký tự có thể in được để hiển thị và đảo ngược process.

```python
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
```

Kiểm tra trên văn bản đa ngôn ngữ để xem số byte:

```python
texts = [
    ("English", "hello"),
    ("Chinese", "你好"),
    ("Emoji", "🔥"),
    ("Mixed", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)} chars -> {len(b)} bytes -> {b}")
```

"Xin chào" là 5 byte. "你好" là 6 byte (3 mỗi ký tự). Biểu tượng cảm xúc lửa là 4 byte. tokenizer cấp byte không quan tâm đó là ngôn ngữ nào. Byte là byte.

### Bước 2: Tokenizer trước với Regex

Chia văn bản thành các phần bằng cách sử dụng mẫu biểu thức chính quy GPT-2. Mỗi đoạn được mã hóa độc lập bởi BPE.

```python
import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]
```

Mô-đun `regex` hỗ trợ thoát thuộc tính Unicode (`\p{L}` cho chữ cái `\p{N}` cho số). Thư viện tiêu chuẩn `re` mô-đun thì không, vì vậy chúng ta quay trở lại classes ký tự ASCII. Đối với production tokenizers đa ngôn ngữ, hãy cài đặt `regex`.

Hãy thử nó:

```python
print(pre_tokenize("Hello, world! Don't stop."))
# [' Hello', ',', ' world', '!', " Don", "'t", ' stop', '.']
```

Khoảng trống đầu vẫn gắn liền với từ. Các dấu rút gọn được tách ra ở dấu nháy đơn. Dấu câu trở thành phần riêng của nó. BPE sẽ không bao giờ merge tokens vượt qua các ranh giới này.

### Bước 3: BPE trên chuỗi byte

Thuật toán cốt lõi từ Bài học 01, nhưng hiện đang hoạt động trên các khối được mã hóa trước một cách độc lập.

```python
from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged
```

### Bước 4: Xử lý Token đặc biệt

tokens đặc biệt cần khớp chính xác và ID cố định. Họ bỏ qua hoàn toàn BPE.

```python
class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts
```

### Bước 5: Tokenizer Class đầy đủ

Xâu chuỗi mọi thứ lại với nhau: chuẩn hóa, phân tách trên các tokens đặc biệt, mã hóa trước, BPE merge, ánh xạ thành ID.

```python
import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)
```

### Bước 6: Kiểm tra đa ngôn ngữ

Bài kiểm tra thực sự. Ném tiếng Anh, tiếng Trung, biểu tượng cảm xúc và mã vào đó.

```python
corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Input:   {text}")
    print(f"Tokens:  {len(ids)} ids")
    print(f"Decoded: {decoded}")
    print()
```

Các ký tự Trung Quốc tạo ra 3 byte mỗi ký tự. Biểu tượng cảm xúc tạo ra 4 byte. Không có ký tự nào trong số này làm hỏng tokenizer. Không có ký tự nào tạo ra tokens không xác định. Đó là sức mạnh của BPE cấp byte.

## Ứng dụng

### So sánh Tokenizers thực tế

Tải tokenizers thực tế từ Llama 3, GPT-4 và Mistral. Xem cách mỗi đoạn xử lý cùng một đoạn văn đa ngôn ngữ.

```python
import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4 ({len(tokens)} tokens): {pieces}")
```

```python
from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name} ({len(tokens)} tokens): {pieces[:20]}...")
```

Bạn sẽ thấy số lượng token khác nhau cho cùng một văn bản. Llama 3 với từ vựng 128K tích cực hơn trong việc hợp nhất các mẫu phổ biến. GPT-4 với 100K nằm ở giữa. Mistral với 32K tạo ra nhiều tokens hơn nhưng có lớp embedding nhỏ hơn.

Sự đánh đổi luôn giống nhau: vốn từ vựng lớn hơn có nghĩa là trình tự ngắn hơn nhưng parameters hơn.

## Sản phẩm bàn giao

Bài học này tạo ra một prompt để xây dựng và gỡ lỗi production tokenizers. Xem `outputs/prompt-tokenizer-builder.md`.

## Bài tập

1. **Dễ dàng:** Thêm một phương thức `get_token_bytes(id)` hiển thị các byte thô cho bất kỳ ID token nào. Sử dụng nó để kiểm tra những gì merged tokens phổ biến nhất của bạn thực sự đại diện.
2. **Trung bình:** Thực hiện pre-tokenizer kiểu Llama phân chia trên khoảng trắng và chữ số nhưng vẫn giữ khoảng trắng ở đầu. So sánh từ vựng của nó với cách tiếp cận regex GPT-2 trên cùng một kho dữ liệu.
3. **Khó:** Thêm một phương thức mẫu trò chuyện lấy danh sách các tin nhắn `{"role": ..., "content": ...}` và tạo trình tự token chính xác cho định dạng trò chuyện Llama 3. Kiểm tra nó với việc triển khai HuggingFace.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|----------------------|
| BPE cấp byte | "Tokenizer hoạt động trên byte" | BPE với từ vựng cơ bản là 256 giá trị byte -- xử lý bất kỳ đầu vào nào mà không có tokens không xác định |
| tokenization trước | "Chia tay trước khi BPE" | Biểu thức chính quy hoặc phân tách dựa trên quy tắc ngăn BPE hợp nhất qua ranh giới từ |
| Chuẩn hóa NFKC | "Dọn dẹp Unicode" | Phân rã chuẩn theo sau là thành phần tương thích -- chữ ghép "fi" trở thành "fi", chiều rộng đầy đủ "A" trở thành "A" |
| Mẫu chat | "Làm thế nào thông điệp trở nên tokens" | Định dạng chính xác để chuyển đổi danh sách các thư role/content thành một chuỗi token phẳng -- dành riêng cho model và phải khớp với định dạng training |
| tokens đặc biệt | "Kiểm soát tokens" | ID token dành riêng bỏ qua BPE -- [BOS], , [pad], điểm đánh dấu trò chuyện -- khớp chính xác trước merge |
| Khả năng sinh sản | "Tokens mỗi từ" | Tỷ lệ tokens đầu ra so với từ nhập - 1,3 đối với tiếng Anh trong GPT-4, 2-3 đối với tiếng Hàn, cao hơn có nghĩa là lãng phí ngữ cảnh |
| tiktoken | "OpenAI tokenizer" | Triển khai Rust BPE với các ràng buộc Python - nhanh hơn 10-100 lần so với Python thuần túy |
| Merge bảng | "Từ vựng" | Danh sách theo thứ tự của cặp byte merges học trong quá trình training - đây là kiến thức đã học của tokenizer |

## Đọc thêm

- [OpenAI tiktoken source](https://github.com/openai/tiktoken) -- Rust BPE triển khai được sử dụng bởi GPT-3.5/4
- [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) -- Rust tokenizer thư viện hỗ trợ BPE, WordPiece, Unigram
- [Llama 3 paper (Meta, 2024)](https://arxiv.org/abs/2407.21783) -- chi tiết về từ vựng và tokenizer training 128K
- [SentencePiece (Kudo & Richardson, 2018)](https://arxiv.org/abs/1808.06226) -- tokenization bất khả tri ngôn ngữ
- [GPT-2 tokenizer source](https://github.com/openai/gpt-2/blob/master/src/encoder.py) -- ánh xạ byte-to-Unicode ban đầu
