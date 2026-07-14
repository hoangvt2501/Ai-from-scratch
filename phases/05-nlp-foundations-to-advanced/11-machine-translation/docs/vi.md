# Dịch máy

> Dịch thuật là nhiệm vụ đã trả tiền cho NLP nghiên cứu trong ba mươi năm và tiếp tục được trả tiền cho đến bây giờ.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 5 · 10 (Cơ chế Attention), Giai đoạn 5 · 04 (GloVe, FastText, Subword)
**Thời lượng:** ~75 phút

## Vấn đề

Một model đọc một câu bằng một ngôn ngữ và tạo ra một câu bằng ngôn ngữ khác. Độ dài khác nhau. Thứ tự từ khác nhau. Một số từ nguồn ánh xạ đến nhiều từ đích và ngược lại. Thành ngữ từ chối ánh xạ một-một. "Tôi nhớ bạn" trong tiếng Pháp là "tu me manques" - nghĩa đen là "bạn đang thiếu tôi". Không có alignment cấp độ từ nào tồn tại được điều đó.

Dịch máy là nhiệm vụ buộc NLP phải phát minh ra encoder-decoders, attention, transformers và cuối cùng là toàn bộ mô hình LLM. Mỗi bước tiến đều đến vì chất lượng bản dịch có thể đo lường được và khoảng cách giữa con người và máy móc rất cứng đầu.

Bài học này bỏ qua bài học lịch sử và dạy pipeline làm việc của năm 2026: pretrained encoder-decoder đa ngôn ngữ (NLLB-200 hoặc mBART), đánh giá tokenization subword, beam search, BLEU và chrF, và một số chế độ thất bại vẫn ship production không bị bắt.

## Khái niệm

![MT pipeline: tokenize → encode → decode with attention → detokenize](../assets/mt-pipeline.svg)

MT hiện đại là một decoder transformer encoder được huấn luyện trên văn bản song song. encoder đọc nguồn bằng tokenization ngôn ngữ của nó. decoder tạo mục tiêu, từng từ phụ một, sử dụng đầu ra của encoder qua cross-attention (bài 10). Giải mã sử dụng beam search để tránh bẫy giải mã tham lam. Đầu ra được detokenized, detruecase và chấm điểm dựa trên một tham chiếu.

Ba lựa chọn hoạt động thúc đẩy chất lượng MT trong thế giới thực.

- **Tokenizer.** SentencePiece BPE được huấn luyện trên một kho ngữ liệu hỗn hợp. Từ vựng được chia sẻ giữa các ngôn ngữ là điều cho phép zero-shot cặp trong NLLB.
- **Model kích thước.** NLLB-200 chưng cất 600M phù hợp với máy tính xách tay. NLLB-200 3.3B là mặc định production đã xuất bản. 54,5 tỷ là trần nghiên cứu.
- **Giải mã.** Chiều rộng chùm tia 4-5 cho nội dung chung. Hình phạt chiều dài để tránh đầu ra quá ngắn. Giải mã hạn chế khi bạn cần tính nhất quán của thuật ngữ.

```figure
seq2seq-alignment
```

## Tự xây dựng

### Bước 1: Cuộc gọi MT pretrained

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

```text
Les chats courent.
```

Ba điều quan trọng ở đây. `src_lang` cho tokenizer biết script và phân đoạn nào sẽ áp dụng. `forced_bos_token_id` cho người decoder biết nên tạo ngôn ngữ nào. Cả hai đều là thủ thuật dành riêng cho NLLB; mBART và M2M-100 sử dụng các quy ước riêng và chúng không thể hoán đổi cho nhau.

### Bước 2: BLEU và chrF

BLEU đo sự chồng chéo n-gram giữa đầu ra và tham chiếu. Bốn kích thước n-gram tham chiếu (1-4), trung bình hình học của độ chính xác, hình phạt ngắn gọn cho đầu ra quá ngắn. Điểm số nằm trong [0, 100]. Thường được sử dụng. Bực bội khi giải thích: 30 BLEU là "có thể sử dụng được"; 40 là "tốt"; 50 là "ngoại lệ"; sự khác biệt dưới 1 BLEU là nhiễu.

chrF đo điểm F cấp ký tự. Nhạy cảm hơn với các ngôn ngữ giàu hình thái mà BLEU đếm thấp các kết quả phù hợp. Thường được báo cáo cùng với BLEU.

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

Luôn sử dụng `sacrebleu`. Nó chuẩn hóa tokenization để điểm số có thể so sánh được giữa các bài báo. Tính toán BLEU của riêng bạn là cách gây hiểu lầm benchmarks xảy ra.

### Hệ thống phân cấp đánh giá ba cấp (2026)

Đánh giá MT hiện đại sử dụng ba họ số liệu bổ sung. Ship với ít nhất hai.

- **Heuristic** (BLEU, chrF). Nhanh, dựa trên tài liệu tham khảo, có thể diễn giải, không nhạy cảm với diễn giải. Sử dụng để so sánh kế thừa và phát hiện hồi quy.
- **Đã học** (COMET, BLEURT, BERTScore). models thần kinh được huấn luyện dựa trên phán đoán của con người; So sánh sự tương đồng về ngữ nghĩa của bản dịch với nguồn và tài liệu tham khảo. COMET có mối liên hệ cao nhất với nghiên cứu MT kể từ năm 2023 và là mặc định production năm 2026 khi chất lượng quan trọng.
- **LLM-as-judge** (không có tài liệu tham khảo). Prompt một model lớn để chấm điểm các bản dịch về sự trôi chảy, đầy đủ, giọng điệu, sự phù hợp về văn hóa. GPT-4 với tư cách là giám khảo phù hợp với thỏa thuận của con người ~80% thời gian khi bảng đánh giá được thiết kế tốt. Sử dụng cho nội dung kết thúc mở không có tài liệu tham khảo.

Thực tế 2026 stack: `sacrebleu` cho BLEU và chrF, `unbabel-comet` cho COMET và LLM nhắc nhở cho tín hiệu cuối cùng đối mặt với con người. Hiệu chỉnh mọi chỉ số dựa trên 50-100 ví dụ được gắn nhãn của con người trước khi tin tưởng vào dữ liệu production.

Các chỉ số không có tham chiếu (COMET-QE, BLEURT-QE, LLM-as-judge) cho phép bạn đánh giá các bản dịch mà không cần tham chiếu, điều này rất quan trọng đối với các cặp ngôn ngữ đuôi dài không tồn tại bản dịch tham chiếu.

### Bước 3: những gì phá vỡ production

Các pipeline làm việc trên sẽ dịch trôi chảy 80% thời gian và âm thầm thất bại 20% còn lại. Các chế độ lỗi được đặt tên:

- **Ảo giác.** Model phát minh ra nội dung không có trong nguồn. Phổ biến trong từ vựng miền không quen thuộc. Triệu chứng: đầu ra trôi chảy nhưng tuyên bố sự thật mà nguồn không nêu rõ. Giảm thiểu: giải mã hạn chế về các thuật ngữ miền, đánh giá của con người về nội dung được quy định, giám sát đầu ra lâu hơn nhiều so với đầu vào.
- **Tạo ra ngoài mục tiêu.** Model dịch sang sai ngôn ngữ. NLLB dễ mắc phải điều này một cách đáng ngạc nhiên trên các cặp ngôn ngữ hiếm. Giảm thiểu: xác minh `forced_bos_token_id` và luôn giải mã bằng ID ngôn ngữ model kiểm tra đầu ra.
- **Trôi dạt thuật ngữ.** "Đăng ký" trở thành "s'inscrire" trong tài liệu 1 và "créer un compte" trong tài liệu 2. Đối với văn bản giao diện người dùng và chuỗi hướng tới người dùng, tính nhất quán quan trọng hơn chất lượng thô. Giảm thiểu: giải mã hạn chế bảng thuật ngữ hoặc từ điển sau chỉnh sửa.
- **Hình thức không phù hợp.** Tiếng Pháp "tu" vs "vous", mức độ lịch sự của Nhật Bản. Các model chọn bất kỳ hình thức nào phổ biến hơn trong training. Đối với nội dung hướng đến khách hàng, điều này thường sai. Giảm thiểu: prompt tiền tố với một token hình thức nếu model hỗ trợ nó hoặc fine-tune một model nhỏ trên kho dữ liệu chỉ chính thức.
- **Sự bùng nổ độ dài khi đầu vào ngắn.** Các câu đầu vào rất ngắn thường tạo ra các bản dịch quá dài vì hình phạt độ dài rơi khỏi vách đá dưới ~5 tokens nguồn. Giảm thiểu: nắp có chiều dài tối đa cứng tỷ lệ thuận với chiều dài nguồn.

### Bước 4: fine-tuning cho một miền

Pretrained models là những người tổng quát. Dịch thuật pháp lý, y tế hoặc trò chơi được hưởng lợi từ việc fine-tuning dữ liệu song song trên miền. Công thức không kỳ lạ:

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)


def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()
```

Vài nghìn ví dụ song song chất lượng cao đánh bại vài trăm nghìn ví dụ nhiễu trên web. Chất lượng của dữ liệu training là đòn bẩy production lớn nhất.

## Ứng dụng

production stack năm 2026 cho MT:

| Trường hợp sử dụng | Điểm bắt đầu được đề xuất |
|---------|---------------------------|
| Bất kỳ-bất kỳ, 200 ngôn ngữ | `facebook/nllb-200-distilled-600M` (máy tính xách tay) hoặc `nllb-200-3.3B` (production) |
| Lấy tiếng Anh làm trung tâm, chất lượng cao, 50 ngôn ngữ | `facebook/mbart-large-50-many-to-many-mmt` |
| Chạy ngắn, inference giá rẻ, English-French/German/Spanish | Helsinki-NLP / Marian models |
| Phía trình duyệt quan trọng về độ trễ | Marian lượng tử hóa ONNX (~50 MB) |
| Chất lượng tối đa, sẵn sàng trả tiền | GPT-4 / Claude / Gemini với prompts dịch thuật |

LLMs hiện vượt trội hơn MT models chuyên biệt trên một số cặp ngôn ngữ vào năm 2026, đặc biệt là về nội dung thành ngữ và ngữ cảnh dài. Sự đánh đổi là chi phí và độ trễ trên mỗi token. Chọn một LLM khi độ dài ngữ cảnh, tính nhất quán về phong cách hoặc điều chỉnh miền thông qua prompting quan trọng hơn thông lượng.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-mt-evaluator.md`:

```markdown
---
name: mt-evaluator
description: Evaluate a machine translation output for shipping.
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

Given a source text and a candidate translation, output:

1. Automatic score estimate. BLEU and chrF ranges you would expect. State whether a reference is available.
2. Five-point human-verifiable check list: (a) content preservation (no hallucinations), (b) correct language, (c) register / formality match, (d) terminology consistency with glossary if provided, (e) no truncation or length explosion.
3. One domain-specific issue to probe. E.g., for legal: named entities and statute citations. For medical: drug names and dosages. For UI: placeholder variables `{name}`.
4. Confidence flag. "Ship" / "Ship with review" / "Do not ship". Tie to the severity of issues found in step 2.

Refuse to ship a translation without a language-ID check on output. Refuse to evaluate without a reference unless the user explicitly opts in to reference-free scoring (COMET-QE, BLEURT-QE). Flag any content over 1000 tokens as likely needing chunked translation.
```

## Bài tập

1. **Dễ dàng.** Dịch một đoạn văn tiếng Anh dài 5 câu sang tiếng Pháp và trở lại tiếng Anh bằng `nllb-200-distilled-600M`. Đo mức độ gần gũi của chuyến đi khứ hồi với bản gốc. Bạn sẽ thấy sự bảo tồn ngữ nghĩa với sự trôi dạt lựa chọn từ.
2. **Trung bình.** Thực hiện kiểm tra ID ngôn ngữ trên đầu ra bản dịch bằng `fasttext lid.176` hoặc `langdetect`. Tích hợp vào cuộc gọi MT để các thế hệ ngoài mục tiêu được phát hiện trước khi quay trở lại.
3. **Khó.** Fine-tune `nllb-200-distilled-600M` trên kho dữ liệu tên miền 5.000 cặp mà bạn chọn. Đo BLEU trên một bộ trước và sau fine-tuning. Báo cáo loại câu nào được cải thiện và loại câu nào thụt lùi.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| BLEU | Điểm dịch thuật | N-gram precision với hình phạt ngắn gọn. [0, 100]. |
| chrF | Điểm F của nhân vật | Điểm F cấp nhân vật. Nhạy cảm hơn đối với các ngôn ngữ giàu hình thái. |
| NMT | MT thần kinh | Transformer encoder-decoder được huấn luyện về văn bản song song. Mặc định 2017+. |
| NLLB | Không có ngôn ngữ nào bị bỏ lại phía sau | Gia đình MT model 200 ngôn ngữ của Meta. |
| Giải mã hạn chế | Đầu ra được kiểm soát | Buộc các tokens hoặc n-gram cụ thể xuất hiện / không xuất hiện trong đầu ra. |
| Ảo giác | Nội dung được phát minh | Model đầu ra không được nguồn hỗ trợ. |

## Đọc thêm

- [Costa-jussà et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) - bài báo NLLB.
- [Post (2018). A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/) - tại sao `sacrebleu` là cách chính xác duy nhất để báo cáo BLEU.
- [Popović (2015). chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/) - giấy chrF.
- [Hugging Face MT guide](https://huggingface.co/docs/transformers/tasks/translation) - hướng dẫn fine-tuning thực tế.
