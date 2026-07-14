# BERT — Mô hình ngôn ngữ mặt nạ

> GPT dự đoán từ tiếp theo. BERT dự đoán một từ bị thiếu. Một câu khác biệt - và nửa thập kỷ của mọi thứ embedding hình dạng.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 7 · 05 (Transformer đầy đủ), Giai đoạn 5 · 02 (Biểu diễn văn bản)
**Thời lượng:** ~45 phút

## Vấn đề

Vào năm 2018, mọi nhiệm vụ NLP - tình cảm, NER, QA, liên quan - đều huấn luyện model của riêng mình từ đầu trên dữ liệu được gắn nhãn của riêng nó. Không có "hiểu tiếng Anh" được huấn luyện trước checkpoint bạn có thể fine-tune. ELMo (2018) cho thấy bạn có thể huấn luyện trước embeddings ngữ cảnh với LSTM hai chiều; nó giúp nhưng không khái quát hóa.

BERT (Devlin et al. 2018) đặt câu hỏi: điều gì sẽ xảy ra nếu chúng ta lấy một transformer encoder, huấn luyện nó trên mọi câu trên internet và buộc nó dự đoán các từ bị thiếu từ ngữ cảnh ở cả hai bên? Sau đó, bạn fine-tune một đầu vào nhiệm vụ xuôi dòng của mình. Hiệu quả Parameter là một sự tiết lộ.

Kết quả: trong vòng 18 tháng, BERT và các biến thể của nó (RoBERTa, ALBERT, ELECTRA) thống trị mọi bảng xếp hạng NLP tồn tại. Đến năm 2020, mọi công cụ tìm kiếm, pipeline kiểm duyệt nội dung và hệ thống tìm kiếm ngữ nghĩa trên trái đất đều có BERT bên trong.

Vào năm 2026, models chỉ dành cho encoder vẫn là công cụ phù hợp để phân loại, truy xuất và trích xuất có cấu trúc - chúng chạy nhanh hơn 5–10× mỗi token so với decoders và embeddings của chúng là xương sống của mọi stack truy xuất hiện đại. ModernBERT (tháng 12 năm 2024) đã đẩy kiến trúc lên bối cảnh 8K với Flash Attention + RoPE + GeGLU.

## Khái niệm

![Masked language modeling: pick tokens, mask them, predict originals](../assets/bert-mlm.svg)

### Tín hiệu training

Lấy một câu: `the quick brown fox jumps over the lazy dog`.

Mặt nạ 15% tokens ngẫu nhiên:

```
input:  the [MASK] brown fox jumps [MASK] the lazy dog
target: the  quick brown fox jumps  over  the lazy dog
```

Huấn luyện model để dự đoán tokens ban đầu ở các vị trí đeo mặt nạ. Bởi vì encoder là hai chiều, dự đoán `[MASK]` ở vị trí 1 có thể sử dụng `brown fox jumps` ở vị trí 2+. Đó là điều GPT không thể làm được.

### Các quy tắc BERT mặt nạ

Trong số 15% tokens được chọn để dự đoán:

- 80% được thay thế bằng `[MASK]`.
- 10% được thay thế bằng một token ngẫu nhiên.
- 10% không thay đổi.

Tại sao không phải lúc nào cũng `[MASK]`? Bởi vì `[MASK]` không bao giờ xuất hiện vào thời điểm inference. Training model mong đợi `[MASK]` ở 100% vị trí đeo mặt nạ sẽ tạo ra sự thay đổi phân phối giữa pretraining và fine-tuning. 10% ngẫu nhiên + 10% không thay đổi giữ cho model trung thực.

### Dự đoán câu tiếp theo (NSP) - và lý do tại sao nó bị loại bỏ

BERT ban đầu cũng được huấn luyện về NSP: cho hai câu A và B, dự đoán nếu B tuân theo A. RoBERTa (2019) đã cắt bỏ nó và cho thấy NSP bị tổn thương, không được giúp đỡ. Hiện đại encoders bỏ qua nó.

### Điều gì đã thay đổi vào năm 2026: ModernBERT

Bài báo ModernBERT năm 2024 đã xây dựng lại khối vào năm 2026 primitives:

| Thành phần | Bản gốc BERT (2018) | BERT hiện đại (2024) |
|-----------|----------------------|-------------------|
| Vị trí | Học tuyệt đối | Dây thừng |
| Kích hoạt | GELU | GeGLU |
| Chuẩn hóa | LayerNorm | RMSNorm trước định mức |
| Attention | Dày đặc đầy đủ | Luân phiên địa phương (128) + toàn cầu |
| Độ dài ngữ cảnh | 512 | 8192 |
| Tokenizer | WordPiece | BPE |

Và không giống như stack 2018, nó là Flash-Attention-native. Inference nhanh hơn 2–3× ở độ dài trình tự 8K so với DeBERTa-v3 với điểm GLUE tốt hơn.

### Các trường hợp sử dụng vẫn chọn encoder vào năm 2026

| Nhiệm vụ | Tại sao encoder đánh bại decoder |
|------|---------------------------|
| Truy xuất / tìm kiếm ngữ nghĩa embeddings | Ngữ cảnh hai chiều = chất lượng embedding tốt hơn trên mỗi token |
| Phân loại (tình cảm, ý định, độc tính) | Một forward pass; không có chi phí phát điện |
| Ghi nhãn NER / token | Đầu ra trên mỗi vị trí, nguyên bản là hai chiều |
| Zero-shot bao gồm (NLI) | Đầu phân loại trên đầu encoder |
| Reranker cho RAG | Ghi điểm chéo encoder, nhanh hơn 10 lần so với LLM người xếp hạng lại |

```figure
transformer-residual
```

## Tự xây dựng

### Bước 1: logic mặt nạ

Xem `code/main.py`. Hàm `create_mlm_batch` nhận danh sách các ID token, kích thước từ vựng và xác suất mặt nạ. Trả về ID đầu vào (có áp dụng mặt nạ) và nhãn (chỉ ở các vị trí được che mặt, -100 ở những nơi khác — quy ước chỉ mục bỏ qua của PyTorch).

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # else: keep original
    return input_ids, labels
```

### Bước 2: chạy dự đoán MLM trên một kho dữ liệu nhỏ

Rèn luyện đầu encoder + MLM 2 lớp trên vốn từ vựng 20 từ, 200 câu. Không gradient - chúng tôi thực hiện kiểm tra sự tỉnh táo của đường chuyền trước. Cần training đầy đủ PyTorch.

### Bước 3: So sánh các loại khẩu trang

Cho thấy quy tắc ba chiều giữ cho model có thể sử dụng được mà không cần `[MASK]` như thế nào. Dự đoán về một câu không đeo mặt nạ và một câu đeo mặt nạ. Cả hai nên tạo ra phân phối token hợp lý vì model nhìn thấy cả hai mô hình trong training.

### Bước 4: fine-tune đầu

Thay thế đầu MLM bằng đầu phân loại trên dataset tình cảm đồ chơi. Chỉ có người đứng đầu huấn luyện; encoder bị đóng băng. Đây là mô hình mà mọi ứng dụng BERT đều tuân theo.

## Ứng dụng

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**Embedding models fine-tuned BERT.** `sentence-transformers` models giống như `all-MiniLM-L6-v2` BERT được huấn luyện với loss tương phản. encoder cũng vậy. loss đã thay đổi.

**Người xếp hạng lại encoder chéo cũng được fine-tuned BERT.** Phân loại cặp trên `[CLS] query [SEP] doc [SEP]`. Sự attention hai chiều giữa truy vấn và doc chính xác là những gì mang lại lợi thế chất lượng encoders chéo so với bộ mã hóa biencoder.

**Khi nào không nên chọn BERT vào năm 2026.** Bất cứ điều gì tạo ra. encoder không có cách hợp lý để tự hồi quy tokens. Ngoài ra: bất cứ thứ gì dưới tham số 1B mà một decoder nhỏ có thể phù hợp với chất lượng với tính linh hoạt hơn (Phi-3-Mini, Qwen2-1.5B).

## Sản phẩm bàn giao

Xem `outputs/skill-bert-finetuner.md`. skill phạm vi một BERT fine-tune (lựa chọn xương sống, thông số kỹ thuật đầu, dữ liệu, đánh giá, dừng) cho một nhiệm vụ phân loại hoặc trích xuất mới.

## Bài tập

1. **Dễ dàng.** Chạy `code/main.py` và in phân phối khẩu trang trên 10.000 tokens. Xác nhận ~15% được chọn và trong số đó ~80% trở thành `[MASK]`.
2. **Trung bình.** Thực hiện mặt nạ toàn bộ từ: nếu một từ được mã hóa thành các từ phụ, hãy che tất cả các từ phụ cùng nhau hoặc không có từ phụ. Đo lường xem điều này có cải thiện accuracy MLM trên kho dữ liệu 500 câu hay không.
3. **Khó.** Huấn luyện một BERT nhỏ (2 lớp, d = 64) trên 10.000 câu từ một dataset công cộng. Fine-tune `[CLS]` token cho tâm lý SST-2. So sánh với đường cơ sở chỉ decoder ở các tham số phù hợp - cái nào thắng?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| MLM | "Mô hình ngôn ngữ mặt nạ" | Training tín hiệu: thay ngẫu nhiên 15% tokens bằng `[MASK]`, dự đoán bản gốc. |
| Hai chiều | "Nhìn cả hai chiều" | Encoder attention không có mặt nạ nhân quả - mọi vị trí đều nhìn thấy mọi vị trí khác. |
| `[CLS]` | "Người bơi bơi token" | Một token đặc biệt được thêm vào đầu mỗi chuỗi; embedding cuối cùng của nó được sử dụng làm biểu diễn ở cấp độ câu. |
| `[SEP]` | "Dải phân đoạn" | Tách các chuỗi được ghép nối (ví dụ: query/doc, câu A/B). |
| NSP | "Dự đoán câu tiếp theo" | Nhiệm vụ pretraining thứ hai của BERT; được chứng minh là vô dụng trong RoBERTa, bị loại bỏ sau năm 2019. |
| Fine-tuning | "Thích ứng với một nhiệm vụ" | Giữ encoder chủ yếu đông lạnh; Huấn luyện một cái đầu nhỏ trên đầu cho nhiệm vụ hạ lưu. |
| encoder chéo | "Một người xếp hạng lại" | Một BERT lấy cả truy vấn và tài liệu làm đầu vào, xuất ra điểm liên quan. |
| BERT hiện đại | "Làm mới năm 2024" | Encoder được xây dựng lại với RoPE, RMSNorm, GeGLU, local/global attention xen kẽ, bối cảnh 8K. |

## Đọc thêm

- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — giấy gốc.
- [Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) - làm thế nào để rèn luyện BERT đúng; giết NSP.
- [Clark et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) - phát hiện token thay thế đánh bại MLM ở điện toán phù hợp.
- [Warner et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663) - Giấy ModernBERT.
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) — tài liệu tham khảo encoder chuẩn.
