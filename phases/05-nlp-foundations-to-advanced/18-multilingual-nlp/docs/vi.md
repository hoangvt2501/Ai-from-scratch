# NLP đa ngôn ngữ

> Một model, 100+ ngôn ngữ, không training dữ liệu cho hầu hết chúng. Chuyển giao đa ngôn ngữ là phép màu thực tế của những năm 2020.

**Loại:** Học
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 5 · 04 (GloVe, FastText, Subword), Giai đoạn 5 · 11 (Dịch máy)
**Thời lượng:** ~45 phút

## Vấn đề

Tiếng Anh có hàng tỷ ví dụ được dán nhãn. Tiếng Urdu có hàng nghìn. Maithili hầu như không có. Bất kỳ hệ thống NLP thực tế nào phục vụ khán giả toàn cầu đều phải làm việc trên các ngôn ngữ mà dữ liệu training cụ thể của nhiệm vụ không tồn tại.

models đa ngôn ngữ giải quyết vấn đề này bằng cách training một model trên nhiều ngôn ngữ cùng một lúc. Biểu diễn được chia sẻ cho phép model chuyển skills đã học bằng các ngôn ngữ có tài nguyên cao sang các ngôn ngữ có tài nguyên thấp. Fine-tune model về phân tích tình cảm tiếng Anh, và nó tạo ra những dự đoán tình cảm tốt đáng ngạc nhiên về tiếng Urdu. Đó là zero-shot chuyển giao đa ngôn ngữ, và nó đã định hình lại cách NLP ships thế giới với thế giới.

Bài học này nêu tên sự đánh đổi, models chuẩn và một quyết định khiến các nhóm mới làm việc đa ngôn ngữ vấp ngã: chọn ngôn ngữ nguồn để chuyển tiếp.

## Khái niệm

![Cross-lingual transfer via shared multilingual embedding space](../assets/multilingual.svg)

**Từ vựng được chia sẻ.** Đa ngôn ngữ models sử dụng SentencePiece hoặc WordPiece tokenizer được huấn luyện về văn bản từ tất cả các ngôn ngữ đích. Từ vựng được chia sẻ: cùng một đơn vị từ phụ đại diện cho cùng một hình vị trên các ngôn ngữ liên quan. `anti-` bằng tiếng Anh và tiếng Ý nhận được token giống nhau.

**Đại diện được chia sẻ.** Một transformer pretrained về mô hình hóa ngôn ngữ được che giấu trên nhiều ngôn ngữ học được rằng các câu tương tự về mặt ngữ nghĩa trong các ngôn ngữ khác nhau tạo ra các trạng thái ẩn tương tự. mBERT, XLM-R và NLLB đều thể hiện điều này. Embeddings cho "cat" trong cụm tiếng Anh gần "chat" trong tiếng Pháp và "gato" trong tiếng Tây Ban Nha, và embeddings đầy đủ câu cũng vậy.

**Zero-shot chuyển.** Fine-tune model trên dữ liệu được gắn nhãn bằng một ngôn ngữ (thường là tiếng Anh). Tại inference, hãy chạy nó trên bất kỳ ngôn ngữ nào khác mà model hỗ trợ. Không cần nhãn ngôn ngữ đích. Kết quả là mạnh đối với các ngôn ngữ liên quan đến kiểu chữ và yếu hơn đối với các ngôn ngữ ở xa.

**Few-shot fine-tuning.** Thêm 100-500 ví dụ được gắn nhãn bằng ngôn ngữ đích. Accuracy nhảy lên 95-98% mức cơ sở của tiếng Anh về các nhiệm vụ phân loại. Đây là đòn bẩy tiết kiệm chi phí nhất trong NLP đa ngôn ngữ.

## Các models

| Model | Năm | Phạm vi bảo hiểm | Ghi chú |
|-------|------|----------|-------|
| mBERT | 2018 | 104 ngôn ngữ | Được huấn luyện trên Wikipedia. LM đa ngôn ngữ thực tế đầu tiên. Yếu về tài nguyên thấp. |
| XLM-R | 2019 | 100 ngôn ngữ | Được huấn luyện trên CommonCrawl (lớn hơn nhiều so với Wikipedia). Đặt đường cơ sở đa ngôn ngữ. Cơ sở 270M, Lớn 550M. |
| XLM-V | 2023 | 100 ngôn ngữ | XLM-R với từ vựng 1M-token (so với 250k). Tốt hơn với tài nguyên thấp. |
| mT5 | 2020 | 101 ngôn ngữ | Kiến trúc T5 để tạo đa ngôn ngữ. |
| NLLB-200 · | 2022 | 200 ngôn ngữ | model dịch thuật của Meta; bao gồm 55 ngôn ngữ tài nguyên thấp. |
| NỞ HOA | 2022 | 46 ngôn ngữ + 13 lập trình | Mở 176B LLM huấn luyện đa ngôn ngữ. |
| Aya-23 · | 2024 | 23 ngôn ngữ | Cohere LLM đa ngôn ngữ. Mạnh về tiếng Ả Rập, tiếng Hindi, tiếng Swahili. |

Chọn theo trường hợp sử dụng. Phân loại hoạt động tốt với XLM-R-base làm mặc định lành mạnh. Các tác vụ tạo yêu cầu mT5 hoặc NLLB tùy thuộc vào bản dịch và thế hệ mở. Làm việc theo phong cách LLM kết hợp với Aya-23 hoặc Claude sử dụng prompting đa ngôn ngữ rõ ràng.

## Quyết định ngôn ngữ nguồn (nghiên cứu năm 2026)

Hầu hết các đội mặc định sử dụng tiếng Anh làm nguồn fine-tuning. Nghiên cứu gần đây (2026) cho thấy điều này thường sai.

Sự tương đồng về ngôn ngữ dự đoán chất lượng truyền tốt hơn kích thước kho dữ liệu thô. Đối với các mục tiêu Slav, tiếng Đức hoặc tiếng Nga thường đánh bại tiếng Anh. Đối với các mục tiêu Ấn Độ, tiếng Hindi thường đánh bại tiếng Anh. Chỉ số tương tự **qWALS** (2026, dựa trên World Atlas of Language Structures features) định lượng điều này. **LANGRANK** (Lin và cộng sự, ACL 2019) là một phương pháp riêng biệt, sớm hơn để xếp hạng các ngôn ngữ nguồn ứng cử viên từ sự kết hợp của sự tương đồng ngôn ngữ, kích thước kho dữ liệu và mối quan hệ di truyền.

Quy tắc thực tế: nếu ngôn ngữ mục tiêu của bạn có một ngôn ngữ có tài nguyên cao gần gũi về mặt kiểu chữ, hãy thử sử fine-tuning ngôn ngữ đó trước, sau đó so sánh với fine-tune tiếng Anh.

## Tự xây dựng

### Bước 1: zero-shot phân loại đa ngôn ngữ

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

Một model, ba ngôn ngữ, cùng một API. XLM-R được huấn luyện về dữ liệu NLI truyền tốt đến phân loại thông qua thủ thuật bao gồm.

### Bước 2: Không gian embedding đa ngôn ngữ

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

Các bản dịch hạ cánh gần trong không gian embedding. Một câu tiếng Anh khác hạ cánh xa hơn. Đây là điều làm cho việc truy xuất, phân cụm và tương tự đa ngôn ngữ hoạt động.

### Bước 3: few-shot fine-tuning chiến lược

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

Đối với 100-500 ví dụ về ngôn ngữ đích, `num_train_epochs=5` và `learning_rate=2e-5` là mặc định an toàn. Tỷ lệ học cao hơn khiến alignment đa ngôn ngữ sụp đổ và bạn nhận được model chỉ bằng tiếng Anh.

## Đánh giá thực sự hiệu quả

- **accuracy theo ngôn ngữ trên các học phần bị giữ.** Không tổng hợp. Cốt liệu che giấu đuôi dài.
- **Benchmark so với đường cơ sở đơn ngữ.** Đối với các ngôn ngữ có đủ dữ liệu, model đơn ngữ được huấn luyện từ đầu đôi khi đánh bại  đa ngôn ngữ. Kiểm tra.
- **Kiểm thử cấp thực thể.** Các thực thể được đặt tên bằng ngôn ngữ đích. models đa ngôn ngữ thường có tokenization yếu đối với scripts cách xa tiếng Latinh.
- **Tính nhất quán giữa các ngôn ngữ.** Cùng một nghĩa trong hai ngôn ngữ sẽ tạo ra cùng một dự đoán. Đo khoảng cách.

## Ứng dụng

stack năm 2026:

| Nhiệm vụ | Đề xuất |
|-----|-------------|
| Phân loại, 100 ngôn ngữ | XLM-R-base (~270M) fine-tuned |
| Zero-shot phân loại văn bản | `joeddav/xlm-roberta-large-xnli` |
| Câu đa ngôn ngữ embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Bản dịch, 200 ngôn ngữ | `facebook/nllb-200-distilled-600M` (xem bài 11) |
| Đa ngôn ngữ tổng quát | Claude, GPT-4, Aya-23, mT5-XXL |
| Ngôn ngữ tài nguyên thấp NLP | XLM-V hoặc một fine-tune dành riêng cho miền trên ngôn ngữ tài nguyên cao có liên quan |

Luôn lập ngân sách cho fine-tuning bằng ngôn ngữ mục tiêu nếu hiệu suất quan trọng. Zero-shot là điểm khởi đầu, không phải là câu trả lời cuối cùng.

### Thuế tokenization (điều gì xảy ra đối với các ngôn ngữ tài nguyên thấp)

models đa ngôn ngữ chia sẻ một tokenizer trên tất cả các ngôn ngữ của họ. Từ vựng đó được huấn luyện trên một kho dữ liệu bị chi phối bởi tiếng Anh, tiếng Pháp, tiếng Tây Ban Nha, tiếng Trung, tiếng Đức. Đối với bất kỳ ngôn ngữ nào nằm ngoài tập hợp thống trị, ba loại thuế kết hợp âm thầm:

- **Thuế sinh sản.** Văn bản ngôn ngữ tài nguyên thấp mã hóa thành nhiều tokens trên mỗi từ so với tiếng Anh. Một câu tiếng Hindi có thể cần gấp 3-5 lần tokens của một câu tiếng Anh tương đương. 3-5 lần đó ăn context window, hiệu quả training và độ trễ của bạn.
- **Thuế khôi phục biến thể.** Mọi lỗi chính tả, biến thể dấu phụ, chuẩn hóa Unicode không khớp hoặc biến thể chữ hoa chữ thường sẽ trở thành một chuỗi không liên quan bắt đầu nguội trong không gian embedding. Người model không thể học các tương ứng chính tả mà người bản ngữ coi là hiển nhiên.
- **Thuế tràn dung lượng.** Thuế 1 và 2 tiêu thụ vị trí ngữ cảnh, độ sâu lớp và kích thước embedding. Những gì còn lại cho lý luận thực tế nhỏ hơn một cách có hệ thống so với những gì một ngôn ngữ có tài nguyên cao nhận được từ cùng một model.

Triệu chứng thực tế: model của bạn huấn luyện bình thường bằng tiếng Hindi, đường cong loss có vẻ đúng, đánh giá perplexity có vẻ hợp lý và production đầu ra sai một cách tinh tế. Hình thái sụp đổ giữa câu. Các biến dạng hiếm gặp vẫn không thể phục hồi. **Bạn không thể mở rộng dữ liệu theo cách thoát khỏi tokenizer bị hỏng.**

Giảm thiểu: chọn một tokenizer có phạm vi phủ sóng tốt cho ngôn ngữ mục tiêu của bạn (từ vựng 1M-token của XLM-V là một cách khắc phục trực tiếp); xác minh khả năng sinh sản tokenization trên văn bản mục tiêu trước khi training; sử dụng dự phòng cấp byte (SentencePiece `byte_fallback=True`, BPE cấp độ byte kiểu GPT-2) cho scripts đuôi dài thực sự để không có gì là OOV.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-multilingual-picker.md`:

```markdown
---
name: multilingual-picker
description: Pick source language, target model, and evaluation plan for a multilingual NLP task.
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

Given requirements (target languages, task type, available labeled data per language), output:

1. Source language for fine-tuning. Default English; check LANGRANK or qWALS if target language has a typologically close high-resource language.
2. Base model. XLM-R (classification), mT5 (generation), NLLB (translation), Aya-23 (generative LLM).
3. Few-shot budget. Start with 100-500 target-language examples if available. Zero-shot only if labeling is infeasible.
4. Evaluation plan. Per-language accuracy (not aggregate), cross-lingual consistency, entity-level F1 on non-Latin scripts.

Refuse to ship a multilingual model without per-language evaluation — aggregate metrics hide long-tail failures. Flag scripts with low tokenization coverage (Amharic, Tigrinya, many African languages) as needing a model with byte-fallback (SentencePiece with byte_fallback=True, or byte-level tokenizer like GPT-2).
```

## Bài tập

1. **Dễ dàng.** Chạy pipeline phân loại zero-shot trên 10 câu cho mỗi ngôn ngữ trên tiếng Anh, tiếng Pháp, tiếng Hindi và tiếng Ả Rập. Báo cáo accuracy về từng loại. Bạn sẽ thấy tiếng Pháp mạnh, tiếng Hindi đàng hoàng, tiếng Ả Rập thay đổi.
2. **Trung bình.** Sử dụng `paraphrase-multilingual-MiniLM-L12-v2` để xây dựng một trình truy xuất đa ngôn ngữ trên một kho dữ liệu hỗn hợp ngôn ngữ nhỏ. Truy vấn bằng tiếng Anh, truy xuất tài liệu bằng bất kỳ ngôn ngữ nào. Đo lường recall@5.
3. **Khó.** So sánh fine-tuning nguồn tiếng Anh và nguồn tiếng Hindi cho nhiệm vụ phân loại tiếng Hindi. Sử dụng 500 ví dụ về ngôn ngữ mục tiêu cho few-shot fine-tuning dưới cả hai chế độ. Báo cáo nguồn nào tạo ra accuracy tiếng Hindi tốt hơn và bao nhiêu. Đây là luận án thu nhỏ của LANGRANK.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| model đa ngôn ngữ | Một model, nhiều ngôn ngữ | Chia sẻ từ vựng và parameters giữa các ngôn ngữ. |
| Chuyển giao đa ngôn ngữ | Huấn luyện một ngôn ngữ, chạy trên một ngôn ngữ khác | Fine-tune trên nguồn, đánh giá trên mục tiêu mà không có nhãn ngôn ngữ đích. |
| Zero-shot | Không có nhãn ngôn ngữ đích | Chuyển mà không cần fine-tuning trên ngôn ngữ đích. |
| Few-shot | Nhãn mục tiêu nhỏ | 100-500 ví dụ ngôn ngữ mục tiêu được sử dụng cho fine-tuning. |
| mBERT | LM đa ngôn ngữ đầu tiên | BERT pretrained 104 ngôn ngữ trên Wikipedia. |
| XLM-R | Đường cơ sở đa ngôn ngữ tiêu chuẩn | RoBERTa 100 ngôn ngữ pretrained trên CommonCrawl. |
| NLLB | MT 200 ngôn ngữ của Meta | Không có ngôn ngữ nào bị bỏ lại phía sau. Bao gồm 55 ngôn ngữ tài nguyên thấp. |

## Đọc thêm

- [Conneau et al. (2019). Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) — giấy XLM-R.
- [Pires, Schlinger, Garrette (2019). How Multilingual is Multilingual BERT?](https://arxiv.org/abs/1906.01502) - bài báo phân tích bắt đầu dòng nghiên cứu chuyển giao đa ngôn ngữ.
- [Costa-jussà et al. (2022). No Language Left Behind](https://arxiv.org/abs/2207.04672) - Giấy NLLB-200.
- [Üstün et al. (2024). Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model](https://arxiv.org/abs/2402.07827) - Aya, LLM đa ngôn ngữ của Cohere.
- [Language Similarity Predicts Cross-Lingual Transfer Learning Performance (2026)](https://www.mdpi.com/2504-4990/8/3/65) — bài báo ngôn ngữ nguồn qWALS / LANGRANK.
