# Tokenization từ phụ — BPE, WordPiece, Unigram, SentencePiece

> Lời nói tokenizers nghẹn ngào những từ vô hình. Nhân vật tokenizers thổi bay độ dài trình tự. Subword tokenizers phân chia sự khác biệt. Mọi LLM ships hiện đại đều có trên một.

**Loại:** Học
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 5 · 01 (Xử lý văn bản), Giai đoạn 5 · 04 (GloVe / FastText / Subword)
**Thời lượng:** ~60 phút

## Vấn đề

Từ vựng của bạn có 50.000 từ. Người dùng gõ "untokenizable". tokenizer của bạn trả về `[UNK]`. model bây giờ không có tín hiệu về từ này. Tệ hơn: tài liệu phân vị thứ 90 trong kho dữ liệu của bạn có 40 từ hiếm, có nghĩa là 40 bit thông tin bị loại bỏ cho mỗi tài liệu.

Subword tokenization giải quyết vấn đề này. Các từ phổ biến vẫn tokens đơn. Những từ hiếm phân hủy thành những mảnh có ý nghĩa: `untokenizable` → `un`, `token`, `izable`. Training dữ liệu bao gồm mọi thứ vì bất kỳ chuỗi nào cuối cùng đều là một chuỗi byte.

Mọi biên giới LLM vào năm 2026 đều ships trên một trong ba thuật toán (BPE, Unigram, WordPiece), được bao bọc trong một trong ba thư viện (tiktoken, SentencePiece, HF Tokenizers). Bạn không thể ship một model ngôn ngữ mà không chọn một ngôn ngữ.

## Khái niệm

![BPE vs Unigram vs WordPiece, character-by-character](../assets/subword-tokenization.svg)

**BPE (Mã hóa cặp byte).** Bắt đầu với từ vựng cấp ký tự. Đếm từng cặp liền kề. Merge cặp thường xuyên nhất vào một token mới. Lặp lại cho đến khi bạn đạt được kích thước từ vựng mục tiêu. Thuật toán thống trị: GPT-2/3/4, Llama, Gemma, Qwen2, Mistral.

**BPE cấp độ byte.** Cùng một thuật toán nhưng trên các byte thô (256 cơ sở tokens) thay vì các ký tự Unicode. Đảm bảo không `[UNK]` tokens - bất kỳ chuỗi byte nào được mã hóa. GPT-2 sử dụng 50.257 tokens (256 byte + 50.000 merges + 1 đặc biệt).

**Unigram.** Bắt đầu với một vốn từ vựng khổng lồ. Gán cho mỗi token một xác suất unigram. Cắt tỉa lặp đi lặp lại tokens loại bỏ ít nhất làm tăng log-likelihood. Xác suất ở inference: có thể lấy mẫu mã hóa (hữu ích cho việc tăng cường dữ liệu thông qua chính quy hóa từ phụ). Được sử dụng bởi T5, mBART, ALBERT, XLNet, Gemma.

**WordPiece.** Merge cặp tối đa hóa likelihood của kho dữ liệu training thay vì tần số thô. Được sử dụng bởi BERT, DistilBERT, ELECTRA.

**SentencePiece vs tiktoken.** SentencePiece là thư viện * huấn luyện * từ vựng (BPE hoặc Unigram) trực tiếp trên văn bản Unicode thô, mã hóa khoảng trắng dưới dạng `▁`. TikToken là * encoder * nhanh của OpenAI chống lại các từ vựng được tạo sẵn; nó không huấn luyện.

Quy tắc ngón tay cái:

- **Training một từ vựng mới:** SentencePiece (đa ngôn ngữ, không có tokenization trước) hoặc HF Tokenizers.
- **inference nhanh chống lại từ vựng GPT:** tiktoken (cl100k_base, o200k_base).
- **Cả hai: **HF Tokenizers - một thư viện, training + phục vụ.

```figure
bpe-merge
```

## Tự xây dựng

### Bước 1: BPE lại từ đầu

Xem `code/main.py`. Vòng lặp:

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

Ba sự kiện mà thuật toán mã hóa. `</w>` đánh dấu kết thúc từ để "thấp" (hậu tố) và "thấp hơn" (tiền tố) vẫn khác biệt. Trọng số tần số làm cho các cặp tần số cao giành chiến thắng sớm. Danh sách merge được sắp xếp theo thứ tự — inference áp dụng merges theo thứ tự training.

### Bước 2: mã hóa với merges đã học

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

Ngây thơ O(n·|merges|). Triển khai Production (tiktoken, HF Tokenizers) sử dụng tra cứu xếp hạng merge với hàng đợi ưu tiên và chạy trong thời gian gần tuyến tính.

### Bước 3: SentencePiece vào thực tế

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # or "unigram"
    character_coverage=0.9995, # lower for CJK (e.g. 0.9995 for English, 0.995 for Japanese)
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

Lưu ý: không cần tokenization trước, khoảng trống được mã hóa dưới dạng `▁` `character_coverage` kiểm soát mức độ tích cực của các ký tự hiếm được bảo tồn so với ánh xạ đến `<unk>`.

### Bước 4: tiktoken cho từ vựng tương thích với OpenAI

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

Chỉ mã hóa. Nhanh (Rust phần phụ trợ). Đối sánh chính xác với GPT-4/5 tokenization để đếm byte, ước tính chi phí, lập ngân sách theo khung thời gian ngữ cảnh.

## Những cạm bẫy vẫn ship vào năm 2026

- **Tokenizer trôi dạt.** Training trên từ vựng A, triển khai đối với từ vựng B. Token ID khác nhau; model xuất ra rác. Kiểm tra hàm băm `tokenizer.json` trong CI.
- **Sự mơ hồ của khoảng trắng.** BPE "xin chào" và "xin chào" tạo ra các tokens khác nhau. Luôn chỉ định `add_special_tokens` và `add_prefix_space` rõ ràng.
- **Thiếu huấn luyện đa ngôn ngữ.** Kho ngữ liệu nặng tiếng Anh tạo ra các từ vựng chia các scripts không phải tiếng Latinh thành tokens nhiều hơn 5-10 lần. Cùng một prompt có giá cao hơn 5-10 lần trong Japanese/Arabic GPT-3.5. o200k_base đã khắc phục một phần điều này.
- **Biểu tượng cảm xúc tách ra.** Một biểu tượng cảm xúc có thể mất 5 tokens. Checkpoint xử lý biểu tượng cảm xúc khi lập ngân sách theo ngữ cảnh.

## Ứng dụng

stack năm 2026:

| Tình huống | Chọn |
|-----------|------|
| Training một model đơn ngữ từ đầu | HF Tokenizers (BPE) |
| Training model đa ngôn ngữ | SentencePiece (Unigram, `character_coverage=0.9995`) |
| Phục vụ API tương thích với OpenAI | TikToken (`o200k_base` cho GPT-4+) |
| Từ vựng miền cụ thể (mã, toán học, protein) | Huấn luyện BPE tùy chỉnh trên kho dữ liệu tên miền, merge với từ vựng cơ bản |
| inference cạnh, model nhỏ | Unigram (từ vựng nhỏ hơn hoạt động tốt hơn) |

Kích thước từ vựng là một quyết định mở rộng quy mô, không phải là một hằng số. Heuristic thô: 32k cho tham số <1B, 50-100k cho 1-10B, 200k + cho multilingual/frontier.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-bpe-vs-wordpiece.md`:

```markdown
---
name: tokenizer-picker
description: Pick tokenizer algorithm, vocab size, library for a given corpus and deployment target.
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

Given a corpus (size, languages, domain) and deployment target (training from scratch / fine-tuning / API-compatible inference), output:

1. Algorithm. BPE, Unigram, or WordPiece. One-sentence reason.
2. Library. SentencePiece, HF Tokenizers, or tiktoken. Reason.
3. Vocab size. Rounded to nearest 1k. Reason tied to model size and language coverage.
4. Coverage settings. `character_coverage`, `byte_fallback`, special-token list.
5. Validation plan. Average tokens-per-word on held-out set, OOV rate, compression ratio, round-trip decode equality.

Refuse to train a character-coverage <0.995 tokenizer on corpora with rare-script content. Refuse to ship a vocab without a frozen `tokenizer.json` hash check in CI. Flag any monolingual tokenizer under 16k vocab as likely under-spec.
```

## Bài tập

1. **Dễ dàng.** Huấn luyện 500 merge BPE trên kho dữ liệu nhỏ bé của `code/main.py`. Mã hóa ba từ bị giữ lại. Có bao nhiêu sản xuất chính xác 1 token so với >1 token?
2. **Trung bình.** So sánh số lượng token trên 100 câu Wikipedia tiếng Anh giữa `cl100k_base`, `o200k_base` và một SentencePiece BPE bạn luyện với vocab = 32k. Báo cáo tỷ lệ nén của từng loại.
3. **Khó.** Huấn luyện cùng một kho dữ liệu với BPE, Unigram và WordPiece. Đo lường accuracy xuôi dòng khi sử dụng từng  trên một bộ phân loại cảm xúc nhỏ. Sự lựa chọn có di chuyển kim hơn 1 điểm F1 không?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| BPE | Mã hóa cặp byte | Tham lam merge các cặp ký tự thường xuyên nhất cho đến khi kích thước từ vựng mục tiêu đạt được. |
| BPE cấp byte | Không có tokens chưa từng được biết đến | BPE trên 256 byte thô; GPT-2 / Llama sử dụng cái này. |
| Unigram | Xác suất tokenizer | Mận khô từ một bộ ứng cử viên lớn bằng cách sử dụng log-likelihood; được sử dụng bởi T5, Gemma. |
| SentencePiece | Khoảng trắng | Thư viện huấn luyện BPE/Unigram trên văn bản thô; không gian được mã hóa dưới dạng `▁`. |
| tiktoken | Cái nhanh | OpenAI BPE encoder được hỗ trợ bởi Rust cho các từ vựng dựng sẵn. Không training. |
| Danh sách Merge | Những con số kỳ diệu | Danh sách `(a, b) → ab` merges theo thứ tự; inference áp dụng theo thứ tự. |
| Phạm vi nhân vật | Làm thế nào hiếm là quá hiếm? | Phần ký tự trong training ngữ liệu mà tokenizer phải bao gồm; ~0,9995 điển hình. |

## Đọc thêm

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) - tờ báo BPE.
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) — bài báo Unigram.
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) - thư viện.
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) - tài liệu tham khảo ngắn gọn.
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) - sách dạy nấu ăn + danh sách mã hóa.
