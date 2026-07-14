# T5, BART - Encoder-Decoder Models

> Encoders hiểu. Decoders tạo. Đặt chúng lại với nhau và bạn sẽ có một model được xây dựng cho các tác vụ đầu vào → đầu ra: dịch, tóm tắt, viết lại, phiên âm.

**Loại:** Học
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 7 · 05 (Transformer đầy đủ), Giai đoạn 7 · 06 (BERT), Giai đoạn 7 · 07 (GPT)
**Thời lượng:** ~45 phút

## Vấn đề

Decoder chỉ có GPT và chỉ encoder BERT đều loại bỏ kiến trúc năm 2017 cho một mục tiêu khác nhau. Nhưng nhiều nhiệm vụ là đầu vào-đầu ra một cách tự nhiên:

- Bản dịch: Tiếng Anh → tiếng Pháp.
- Tóm tắt: Bài viết 5.000 token → tóm tắt 200 token.
- Nhận dạng giọng nói: tokens âm thanh tokens → văn bản.
- Trích xuất có cấu trúc: văn xuôi → JSON.

Đối với những thứ này, encoder-decoder tạo nên sự vừa vặn sạch sẽ nhất. encoder tạo ra một đại diện dày đặc của nguồn. decoder tạo ra đầu ra, tham gia chéo vào biểu diễn đó ở mọi bước. Training được dịch chuyển từng cái một ở phía đầu ra. Tương tự loss như GPT, chỉ điều kiện vào đầu ra encoder.

Hai bài báo đã xác định cẩm nang hiện đại:

1. **T5** (Raffel và cộng sự 2019). "Chuyển văn bản thành văn bản Transformer." Mọi tác vụ NLP đều được định hình lại dưới dạng văn bản vào, văn bản ra. Kiến trúc đơn, từ vựng đơn, loss đơn. Pretrained trên dự đoán span được che giấu (spans bị hỏng trong đầu vào, giải mã chúng trong đầu ra).
2. **BART** (Lewis và cộng sự, 2019). "Transformer hai chiều và tự thoái lui." Bộ mã hóa tự động khử nhiễu: đầu vào bị hỏng theo nhiều cách (xáo trộn, mặt nạ, xóa, xoay), yêu cầu decoder dựng lại bản gốc.

Vào năm 2026, định dạng encoder-decoder tồn tại ở nơi cấu trúc đầu vào quan trọng:

- Thì thầm (lời nói → văn bản).
- Bản dịch của Google stack.
- Một số models hoàn thành / sửa chữa mã có cấu trúc ngữ cảnh và chỉnh sửa riêng biệt.
- Flan-T5 và các biến thể cho các tác vụ suy luận có cấu trúc.

Decoder-only giành được ánh đèn sân khấu, nhưng encoder-decoder không bao giờ biến mất.

## Khái niệm

![Encoder-decoder with cross-attention](../assets/encoder-decoder.svg)

### Vòng lặp chuyển tiếp

```
source tokens ─▶ encoder ─▶ (N_src, d_model)  ──┐
                                                 │
target tokens ─▶ decoder block                   │
                 ├─▶ masked self-attention       │
                 ├─▶ cross-attention ◀───────────┘
                 └─▶ FFN
                ↓
              next-token logits
```

Điều quan trọng là encoder chạy một lần cho mỗi đầu vào. decoder chạy tự hồi quy nhưng tham gia chéo đến đầu ra encoder * giống nhau * ở mỗi bước. Lưu vào bộ nhớ đệm đầu ra encoder là một cách tăng tốc miễn phí cho các đầu vào dài.

### T5 pretraining — tham nhũng span

Chọn spans ngẫu nhiên của đầu vào (độ dài trung bình 3 tokens, tổng cộng 15%). Thay thế mỗi span bằng một lính canh duy nhất: `<extra_id_0>`, `<extra_id_1>`, v.v. decoder chỉ xuất ra spans bị hỏng với tiền tố lính gác của chúng:

```
source: The quick <extra_id_0> fox jumps <extra_id_1> dog
target: <extra_id_0> brown <extra_id_1> over the lazy
```

Tín hiệu rẻ hơn so với dự đoán toàn bộ chuỗi. Cạnh tranh với MLM (BERT) và tiền tố-LM (UniLM) trong quá trình cắt bỏ của bài báo T5.

### BART pretraining - khử nhiễu đa nhiễu

BART thử năm chức năng gây nhiễu:

1. Token mặt nạ.
2. Token xóa.
3. Điền văn bản (che span decoder chèn độ dài phù hợp).
4. Hoán vị câu.
5. Xoay tài liệu.

Kết hợp điền văn bản + hoán vị câu tạo ra những con số xuôi dòng tốt nhất. The decoder luôn tái tạo lại bản gốc. Đầu ra của BART là chuỗi đầy đủ, không chỉ spans bị hỏng - vì vậy tính toán pretraining cao hơn T5.

### Inference

Cùng một thế hệ tự hồi quy như GPT. Tham lam / chùm tia / top-p sampling áp dụng. Beam search (chiều rộng 4–5) là tiêu chuẩn để dịch và tóm tắt vì phân phối đầu ra hẹp hơn chat.

### Khi nào nên chọn từng mẫu mã vào năm 2026

| Nhiệm vụ | Encoder-decoder? | Tại sao |
|------|------------------|-----|
| Dịch thuật | Có, thường | Trình tự nguồn rõ ràng; phân phối đầu ra cố định; beam search tác phẩm |
| Chuyển giọng nói thành văn bản | Có (Thì thầm) | Phương thức đầu vào khác với đầu ra; encoder hình dạng âm thanh features |
| Trò chuyện / lý luận | Không, chỉ dành cho decoder | Không có "đầu vào" liên tục - cuộc trò chuyện là trình tự |
| Hoàn thành mã | Thường không | Chỉ Decoder với bối cảnh dài sẽ thắng; models mã như Qwen 2.5 Coder chỉ dành cho decoder |
| Tóm tắt | Một trong hai hoạt động | BART, PEGASUS đánh bại các đường cơ sở chỉ decoder trước đó; LLMs chỉ decoder hiện đại phù hợp với chúng |
| Trích xuất có cấu trúc | Hoặc | T5 sạch sẽ vì "văn bản → văn bản" hấp thụ bất kỳ định dạng đầu ra nào |

Xu hướng kể từ ~ năm 2022: chỉ decoder tiếp quản các tác vụ mà encoder decoder từng sở hữu vì (a) chỉ decoder được điều chỉnh theo lệnh LLMs khái quát hóa bất kỳ thứ gì thông qua prompting, (b) một kiến trúc mở rộng dễ dàng hơn hai, (c) RLHF giả định một decoder. Encoder-decoder nắm giữ nơi phương thức đầu vào khác nhau (lời nói, hình ảnh) hoặc beam search chất lượng quan trọng.

## Tự xây dựng

Xem `code/main.py`. Chúng ta triển khai tham nhũng span kiểu T5 cho kho đồ chơi - phần hữu ích nhất của bài học này vì nó xuất hiện trong mọi công thức encoder decoder pretraining kể từ đó.

### Bước 1: span tham nhũng

```python
def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """Pick spans summing to ~mask_rate of tokens. Return (corrupted_input, target)."""
    n = len(tokens)
    n_mask = max(1, int(n * mask_rate))
    n_spans = max(1, int(round(n_mask / mean_span)))
    ...
```

Định dạng mục tiêu là quy ước T5: `<sent0> span0 <sent1> span1 ...`. Đầu vào bị hỏng xen kẽ tokens không thay đổi với tokens lính gác tại span vị trí.

### Bước 2: xác minh khứ hồi

Với đầu vào và mục tiêu bị hỏng, hãy xây dựng lại câu gốc. Nếu sự tham nhũng của bạn có thể đảo ngược, forward pass đã được xác định rõ ràng. Đây là một kiểm tra sự tỉnh táo - thực tế training không bao giờ làm điều này, nhưng bài kiểm tra rẻ và phát hiện từng lỗi trong sổ sách kế toán span của bạn.

### Bước 3: Nhiễu BART

Năm chức năng: `token_mask`, `token_delete`, `text_infill`, `sentence_permute`, `document_rotate`. Soạn hai trong số chúng và hiển thị kết quả.

## Ứng dụng

Tham khảo HuggingFace:

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
```

Thủ thuật T5: tên nhiệm vụ đi vào văn bản đầu vào. Cùng một model xử lý hàng chục tác vụ vì mỗi tác vụ là văn bản vào, văn bản ra. Vào năm 2026, mô hình này đã được khái quát hóa bằng models chỉ decoder được điều chỉnh theo lệnh, nhưng T5 đã hệ thống hóa nó trước.

## Sản phẩm bàn giao

Xem `outputs/skill-seq2seq-picker.md`. skill chọn giữa encoder-decoder và chỉ decoder cho một nhiệm vụ mới với cấu trúc đầu vào-đầu ra, độ trễ và mục tiêu chất lượng.

## Bài tập

1. **Dễ dàng.** Chạy `code/main.py`, áp dụng span hỏng cho câu dài 30 token, xác minh rằng việc nối tokens nguồn không phải sentinel với mục tiêu đã giải mã spans tái tạo bản gốc.
2. **Trung bình.** Thực hiện nhiễu `text_infill` của BART: thay thế spans ngẫu nhiên bằng một `<mask>` token duy nhất và decoder phải suy ra độ dài span chính xác cộng với nội dung. Cho thấy một ví dụ.
3. **Khó.** Fine-tune `flan-t5-small` trên một kho ngữ liệu tiếng Anh → lợn-Latinh nhỏ (200 cặp). Đo BLEU trên một bộ 50 cặp được giữ ra. So sánh với fine-tuning `Llama-3.2-1B` trên cùng một dữ liệu với cùng một điện toán.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Encoder-decoder | "Seq2seq transformer" | Hai stacks: encoder hai chiều cho đầu vào, decoder nhân quả với cross-attention cho đầu ra. |
| Cross-attention | "Nơi nguồn nói chuyện với mục tiêu" | Decoder's Q × encoder's K/V. Nơi duy nhất encoder thông tin đi vào decoder. |
| Span tham nhũng | "Thủ thuật pretraining của T5" | Thay thế spans ngẫu nhiên bằng tokens lính gác; decoder xuất ra spans. |
| Mục tiêu khử nhiễu | "Trò chơi của BART" | Áp dụng chức năng nhiễu cho đầu vào, huấn luyện decoder để tái tạo trình tự sạch. |
| Lính gác token | "Trình giữ chỗ `<extra_id_N>`" | tokens đặc biệt gắn thẻ spans bị hỏng trong nguồn và gắn lại chúng trong đích. |
| Bánh flan | "T5 được điều chỉnh theo hướng dẫn" | T5 fine-tuned >1.800 nhiệm vụ; làm cho encoder-decoder cạnh tranh trong việc tuân theo hướng dẫn. |
| Beam search | "Chiến lược giải mã" | Giữ top-k trình tự từng phần ở mỗi bước; Tiêu chuẩn cho translation/summarization. |
| Giáo viên ép buộc | "Nhập Training lần" | Trong quá trình training, hãy cung cấp token đầu ra trước đó thực sự cho decoder, không phải sampled một. |

## Đọc thêm

- [Raffel et al. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) - T5.
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) - BART.
- [Chung et al. (2022). Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) - Flan-T5.
- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) - Whisper, encoder-decoder 2026 kinh điển.
- [HuggingFace `modeling_t5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) — triển khai tham khảo.
