# Toàn bộ Transformer — Encoder + Decoder

> Attention là ngôi sao. Mọi thứ khác - dư lượng, chuẩn hóa, chuyển tiếp cross-attention - là giàn giáo cho phép bạn stack nó sâu.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 7 · 02 (Self-Attention), Giai đoạn 7 · 03 (Multi-Head Attention), Giai đoạn 7 · 04 (Mã hóa vị trí)
**Thời lượng:** ~75 phút

## Vấn đề

Một lớp attention duy nhất là một bộ chiết xuất feature, không phải là một model. Một matmul mỗi lớp là không đủ dung lượng cho ngôn ngữ. Bạn cần độ sâu - và độ sâu mà không cần hệ thống ống nước phù hợp.

Bài báo Vaswani năm 2017 đã đóng gói sáu quyết định thiết kế biến một lớp attention thành một khối có thể xếp chồng lên nhau. Mọi transformer kể từ đó - chỉ encoder (BERT), chỉ decoder (GPT), encoder decoder (T5) - đều thừa hưởng cùng một bộ xương. Vào năm 2026, các khối đã được tinh chế (RMSNorm, SwiGLU, pre-norm, RoPE) nhưng khung xương giống hệt nhau.

Bài học này là bộ xương. Các bài học tiếp theo chuyên về nó - 06 cho encoders, 07 cho decoders, 08 cho encoder-decoder.

## Khái niệm

![Encoder and decoder block internals, wired](../assets/full-transformer.svg)

### Sáu mảnh

1. **Embedding + tín hiệu vị trí.** Tokens → vectors. Vị trí được tiêm qua RoPE (hiện đại) hoặc hình sin (cổ điển).
2. **Self-attention.** Mọi vị trí đều quan tâm đến nhau. Đeo mặt nạ trong decoders.
3. **Mạng chuyển tiếp nguồn cấp dữ liệu (FFN).** MLP hai lớp theo vị trí: `W_2 · activation(W_1 · x)`. Tỷ lệ mở rộng 4× theo mặc định.
4. **Kết nối còn lại.** `x + sublayer(x)`. Nếu không có điều này, gradients biến mất qua ~6 lớp.
5. **Chuẩn hóa lớp.** `LayerNorm` hoặc `RMSNorm` (hiện đại). Ổn định dòng dư.
6. **Cross-attention (chỉ decoder).** Truy vấn đến từ decoder, khóa và giá trị từ đầu ra encoder.

Quan sát một vector chảy qua một khối: attention trộn qua các vị trí, phần dư mang nó về phía trước, FFN biến đổi nó và định mức giữ cho luồng ổn định.

```figure
transformer-block
```

### Khối Encoder (được sử dụng bởi BERT, T5 encoder)

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

Encoder là hai chiều. Không đeo mặt nạ. Tất cả các vị trí đều xem tất cả các vị trí.

### Khối Decoder (được sử dụng bởi GPT, T5 decoder)

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

Decoder có ba lớp con trên mỗi khối. Cái giữa - cross-attention - là nơi duy nhất thông tin chảy từ encoder này sang decoder khác. Trong kiến trúc chỉ decoder thuần túy (GPT), cross-attention bị bỏ qua và bạn chỉ có self-attention + FFN được che mặt.

### Trước định mức so với sau định mức

Giấy gốc: `x + sublayer(LN(x))` vs `LN(x + sublayer(x))`. Post-norm mất đi sự ưu ái vào khoảng năm 2019 - sẽ khó tập luyện sâu hơn nếu không khởi động cẩn thận. Pre-norm (`LN` *before* sublayer) là mặc định năm 2026: Llama, Qwen, GPT-3+, Mistral đều sử dụng nó.

### Khối hiện đại hóa năm 2026

Vaswani 2017 shipped LayerNorm + ReLU. Các stacks hiện đại đã thay thế cả hai. Các khối production thực sự trông như thế nào:

| Thành phần | 2017 | 2026 |
|-----------|------|------|
| Chuẩn hóa | LayerNorm | RMSNorm |
| Kích hoạt FFN | ReLU | SwiGLU |
| Mở rộng FFN | 4× | 2.6× (SwiGLU sử dụng ba ma trận, tổng số tham số khớp nhau) |
| Chức vụ | Hình sin tuyệt đối | Dây thừng |
| Attention | MHA đầy đủ | GQA (hoặc MLA) |
| Bias thuật ngữ | Có | Không |

RMSNorm loại bỏ căn giữa trung bình của LayerNorm (trừ ít hơn một lần), giúp tiết kiệm tính toán và ít nhất là ổn định về mặt kinh nghiệm. SwiGLU (`Swish(W1 x) ⊙ W3 x`) liên tục vượt trội hơn ReLU/GELU FFN ~0,5 điểm ppl trong các bài báo Llama, PaLM và Qwen.

### Số lượng Parameter

Đối với một khối có `r` mở rộng `d_model = d` và FFN:

- MHA: `4 · d²` (phép chiếu Q, K, V, O)
- FFN (SwiGLU): `3 · d · (r · d)` ≈ `3rd²`
- Định mức: không đáng kể

Tại `d = 4096, r = 2.6, layers = 32` (khoảng Llama 3 8B), tổng: `32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B parameters per layer × 32 ≈ 7B` (cộng với embeddings và đầu). Khớp với số lượng đã công bố.

## Tự xây dựng

### Bước 1: các khối xây dựng

Sử dụng `Matrix` class nhỏ từ Bài 03 (được sao chép vào file này để độc lập):

- `layer_norm(x, eps=1e-5)` - trừ trung bình, chia cho std.
- `rms_norm(x, eps=1e-6)` - chia cho RMS. Không có phép trừ trung bình.
- `gelu(x)` và `silu(x) * W3 x` (SwiGLU).
- `ffn_swiglu(x, W1, W2, W3)`.
- `encoder_block(x, params)` và `decoder_block(x, enc_out, params)`.

Xem `code/main.py` để biết hệ thống dây điện đầy đủ.

### Bước 2: nối dây encoder 2 lớp và decoder 2 lớp

Stack họ. Truyền đầu ra encoder vào mọi decoder cross-attention. Thêm LN cuối cùng trước phép chiếu đầu ra.

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### Bước 3: chạy về phía trước trên một ví dụ về đồ chơi

Cung cấp nguồn 6 token và mục tiêu 5 token. Xác minh hình dạng đầu ra là `(5, vocab)`. Không training - bài học này là về kiến trúc, không phải loss.

### Bước 4: hoán đổi trong RMSNorm + SwiGLU

Thay thế LayerNorm và ReLU-FFN bằng RMSNorm và SwiGLU. Xác nhận các hình dạng vẫn khớp. Đây là lần hiện đại hóa năm 2026 với một chức năng thay thế.

## Ứng dụng

Các triển khai tham khảo PyTorch/TF: `nn.TransformerEncoderLayer`, `nn.TransformerDecoderLayer`. Nhưng hầu hết mã production 2026 cuộn khối riêng của nó vì:

- Flash Attention được gọi bên trong attention chứ không phải qua `nn.MultiheadAttention`.
- GQA / MLA không có trong tài liệu tham khảo stdlib.
- RoPE, RMSNorm SwiGLU không phải là PyTorch mặc định.

HF `transformers` có các khối tham chiếu rõ ràng mà bạn nên đọc: `modeling_llama.py` là khối chỉ dành cho decoder năm 2026 chính tắc. Đó là ~500 dòng và đáng để đi bộ qua một lần.

**Encoder vs decoder vs encoder-decoder — khi nào nên chọn:**

| Nhu cầu | Chọn | Ví dụ |
|------|------|---------|
| Phân loại, embeddings, QA qua văn bản | Chỉ Encoder | BERT, DeBERTa, BERT hiện đại |
| Tạo văn bản, trò chuyện, mã, lập luận | Chỉ Decoder | GPT, Llama, Claude, Qwen |
| Đầu vào có cấu trúc → đầu ra có cấu trúc (dịch, tóm tắt) | Encoder-decoder | T5, BART, thì thầm |

Decoder chỉ giành được ngôn ngữ vì nó mở rộng quy mô rõ ràng nhất và xử lý cả khả năng hiểu và tạo ra. Encoder-decoder vẫn tốt nhất khi đầu vào có nhận dạng "chuỗi nguồn" rõ ràng (dịch, nhận dạng giọng nói, tác vụ có cấu trúc).

## Sản phẩm bàn giao

Xem `outputs/skill-transformer-block-reviewer.md`. skill xem xét việc triển khai khối transformer mới đối với các vỡ nợ năm 2026 và gắn cờ các phần bị thiếu (tỷ lệ mở rộng trước định mức, RoPE, RMSNorm, GQA, FFN).

## Bài tập

1. **Dễ dàng.** Đếm parameters trong encoder_block của bạn ở `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True`. Xác thực bằng cách triển khai khối và sử dụng `sum(p.numel() for p in block.parameters())`.
2. **Trung bình.** Chuyển từ hậu chuẩn sang trước định mức. Khởi tạo cả hai và đo định mức kích hoạt sau 12 lớp xếp chồng lên nhau trên đầu vào ngẫu nhiên. Các hoạt động của Post-norm sẽ bùng nổ; pre-norm nên có giới hạn.
3. **Khó.** Thực hiện decoder encoder 4 lớp trên tác vụ sao chép đồ chơi (sao chép `x` đảo ngược). Huấn luyện 100 bước. Báo cáo loss. Hoán đổi bằng RMSNorm + SwiGLU + RoPE — loss có giảm không?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Khối | "Một lớp transformer" | Stack định mức + attention + định mức + FFN, được bọc trong các kết nối dư. |
| Còn lại | "Bỏ qua kết nối" | `x + f(x)` đầu ra; cho phép gradient chảy qua stacks sâu. |
| Định mức trước | "Bình thường hóa trước, không phải sau" | Hiện đại: `x + sublayer(LN(x))`. Tập luyện sâu hơn mà không cần thể dục dụng cụ khởi động. |
| RMSNorm | "LayerNorm mà không có ý nghĩa" | Chia cho RMS; ít hơn một OP, cùng một sự ổn định thực nghiệm. |
| SwiGLU | "FFN mọi người đã chuyển sang" | `Swish(W1 x) ⊙ W3 x → W2`. Beats ReLU/GELU trên LM ppl. |
| Cross-attention | "Làm thế nào decoder nhìn thấy encoder" | MHA với Q từ decoder, K/V từ đầu ra encoder. |
| Mở rộng FFN | "MLP giữa rộng bao nhiêu" | Tỷ lệ kích thước ẩn trên d_model, thường là 4 (LayerNorm) hoặc 2,6 (SwiGLU). |
| Không Bias | "Bỏ các điều khoản +b" | Các stacks hiện đại bỏ qua các thành kiến trong các lớp tuyến tính; cải thiện ppl nhẹ, model nhỏ hơn. |

## Đọc thêm

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) - thông số khối ban đầu.
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) - tại sao tiền chuẩn lại đánh bại hậu chuẩn một cách sâu sắc.
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — RMSNorm.
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) - tờ báo SwiGLU.
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — khối chỉ dành cho decoder chuẩn năm 2026.
