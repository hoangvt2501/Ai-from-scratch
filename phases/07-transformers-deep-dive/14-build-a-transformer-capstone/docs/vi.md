# Xây dựng Transformer từ đầu — The Capstone

> Mười ba bài học. Một model. Không có phím tắt.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 7 · 01 đến 13. Đừng bỏ qua.
**Thời lượng:** ~120 phút

## Vấn đề

Bạn đã đọc mọi bài báo. Bạn đã triển khai attention, phân chia nhiều đầu, mã hóa vị trí, khối encoder và decoder, thua lỗ BERT và GPT, MoE KV cache. Bây giờ hãy làm cho họ làm việc cùng nhau trong một nhiệm vụ thực sự.

Điểm mấu chốt: huấn luyện một transformer nhỏ chỉ dành cho decoder từ đầu đến cuối trong một tác vụ mô hình hóa ngôn ngữ cấp ký tự. Nó đọc Shakespeare. Nó tạo ra Shakespeare mới. Nó đủ nhỏ để luyện tập trên máy tính xách tay trong vòng chưa đầy 10 phút. Nó đủ chính xác để hoán đổi trong một dataset lớn hơn và training dài hơn sẽ giúp bạn có được một LM thực sự.

Đây là "nanoGPT" của khóa học. Nó không phải là bản gốc — hướng dẫn nanoGPT năm 2023 của Karpathy là cách triển khai tham khảo mà mọi sinh viên viết ít nhất một lần. Chúng ta nâng hình dạng và trang bị lại nó xung quanh những gì chúng tôi đã đề cập.

## Khái niệm

![Transformer-from-scratch block diagram](../assets/capstone.svg)

Kiến trúc, chú thích:

```
input tokens (B, N)
   │
   ▼
token embedding + positional embedding  ◀── Lesson 04 (RoPE option)
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── Lesson 05
│  MultiHeadAttention (causal)      │  ◀── Lesson 03 + 07 (causal mask)
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── Lesson 05
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head (tied to token embedding)
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy            ◀── Lesson 07
```

### Những gì chúng tôi ship

- `GPTConfig` - một nơi để định cấu hình tất cả hyperparameters.
- `MultiHeadAttention` — nhân quả, hàng loạt, với đường dẫn kiểu Flash tùy chọn (`scaled_dot_product_attention` của PyTorch).
- `SwiGLUFFN` — FFN hiện đại.
- `Block` - chuẩn trước, attention bọc dư + FFN.
- `GPT` — embeddings, khối xếp chồng, đầu LM, generate().
- Training vòng lặp với AdamW, cosin LR gradient cắt.
- tokenizer cấp độ ký tự trên văn bản Shakespeare.

### Những gì chúng tôi không ship

- RoPE — được thực hiện theo khái niệm trong Bài 04. Ở đây chúng tôi sử dụng embeddings vị trí đã học để đơn giản. Các bài tập yêu cầu bạn hoán đổi trong RoPE.
- KV cache trong quá trình tạo — mỗi bước thế hệ sẽ tính toán lại attention trên tiền tố đầy đủ. Chậm hơn nhưng đơn giản hơn. Các bài tập yêu cầu bạn thêm một KV cache.
- Flash Attention - PyTorch 2.0+ tự động gửi nếu đầu vào khớp; chúng tôi sử dụng `F.scaled_dot_product_attention`.
- MoE — FFN duy nhất trên mỗi khối. Bạn đã thấy MoE trong Bài 11.

### Chỉ số mục tiêu

Trên máy tính xách tay Mac M2, 4 lớp, 4 đầu, d_model=128 GPT được huấn luyện 2.000 bước trên `tinyshakespeare.txt`:

- Training loss hội tụ từ ~4,2 (ngẫu nhiên) đến ~1,5 trong khoảng 6 phút.
- Đầu ra được lấy mẫu trông có hình dạng Shakespeare: các từ cổ xưa, ngắt dòng, tên riêng như "ROMEO:" xuất hiện.
- Val loss (giữ 10% văn bản cuối cùng) theo dõi training loss chặt chẽ; không có overfitting vào size/budget. này

## Tự xây dựng

Bài học này sử dụng PyTorch. Cài đặt `torch` (CPU bản dựng là được). Xem `code/main.py`. Các script xử lý:

- Tải xuống `tinyshakespeare.txt` nếu thiếu (hoặc đọc bản sao cục bộ).
- tokenizer ký tự cấp byte.
- Train/val tách tại 90/10.
- Training vòng lặp với tính năng tự động truyền BF16 trên phần cứng được hỗ trợ.
- Sampling sau khi training hoàn tất.

### Bước 1: dữ liệu

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65 nhân vật độc đáo. Từ vựng nhỏ. Phù hợp với vocab_size 4 byte. Không BPE, không có tokenizer kịch tính.

### Bước 2: model

Xem `code/main.py`. Khối này là sách giáo khoa từ Bài 05 - tiền chuẩn mực, RMSNorm, SwiGLU, MHA nhân quả. Số lượng Parameter cho 4/4/128: ~800K.

### Bước 3: training vòng lặp

Nhận một batch ngẫu nhiên có chiều dài-256 token windows. Chuyển tiếp. Entropy chéo dịch chuyển từng một. Lùi. Bước AdamW. Nhật ký. Lặp lại.

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### Bước 4: lấy mẫu

Cho một prompt, liên tục chuyển tiếp, lấy mẫu từ top-p logits, thêm và tiếp tục. Dừng lại sau 500 tokens.

### Bước 5: đọc đầu ra

Sau 2.000 bước:

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

Không phải Shakespeare. Nhưng hình dạng của Shakespeare. Một chiến thắng rõ ràng với ~800K parameters và 6 phút trên máy tính xách tay.

## Ứng dụng

Capstone này là một kiến trúc tham chiếu. Ba phần mở rộng để ship nó thành một cái gì đó thực tế:

1. **Hoán đổi tokenizer.** Sử dụng BPE (ví dụ: `tiktoken.get_encoding("cl100k_base")`). Kích thước từ vựng nhảy từ 65 lên ~50.000. Dung lượng Model cần phải mở rộng quy mô để bù đắp.
2. **Huấn luyện trên một kho dữ liệu lớn hơn.** Sử dụng `OpenWebText` hoặc `fineweb-edu` (HuggingFace). 10B tokens trên một máy bay A100 mất ~24 giờ đối với GPT 125M tham số.
3. **Thêm RoPE + KV cache + Flash Attention.** Các bài tập dưới đây sẽ hướng dẫn bạn từng bài tập.

Điều này kết thúc như một parameter GPT 125 triệu tạo ra tiếng Anh trôi chảy. Không phải là một model biên giới. Nhưng cùng một đường dẫn mã - chỉ lớn hơn - là những gì Karpathy, EleutherAI và Viện Allen sử dụng để huấn luyện checkpoints nghiên cứu vào năm 2026.

## Sản phẩm bàn giao

Xem `outputs/skill-transformer-review.md`. skill xem xét việc triển khai transformer từ đầu để đảm bảo tính chính xác trên tất cả 13 bài học prior.

## Bài tập

1. **Dễ dàng.** Chạy `code/main.py`. Xác minh loss xác thực bước cuối cùng của model được huấn luyện của bạn dưới 2.0. Thay đổi `max_steps` từ 2.000 lên 5.000 - val loss có tiếp tục cải thiện không?
2. **Trung bình.** Thay thế embeddings vị trí đã học bằng RoPE. Áp dụng vòng quay cho Q và K bên trong `MultiHeadAttention`. Huấn luyện và xác minh giá loss ít nhất là thấp.
3. **Trung bình.** Triển khai một KV cache trong vòng lặp sampling. Tạo 500 tokens có và không có bộ nhớ cache. Đồng hồ treo tường sẽ được cải thiện từ 5–20× trên máy tính xách tay.
4. **Khó.** Thêm đầu thứ hai vào model dự đoán token cộng một tiếp theo (MTP - Dự đoán đa Token từ DeepSeek-V3). Cùng huấn luyện. Nó có giúp ích không?
5. **Cứng.** Thay thế FFN duy nhất trên mỗi khối bằng một MoE 4 chuyên gia. Bộ định tuyến + định tuyến 2 hàng đầu. Xem val loss thay đổi như thế nào ở parameters hoạt động phù hợp.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| nanoGPT | "Hướng dẫn của Karpathy repo" | Mã transformer training chỉ dành cho decoder tối thiểu, ~300 LOC; tài liệu tham khảo kinh điển. |
| Shakespeare tí hon | "Kho đồ chơi tiêu chuẩn" | ~1,1 MB văn bản; mọi hướng dẫn character-LM kể từ năm 2015 đều sử dụng nó. |
| Ràng buộc embeddings | "Chia sẻ ma trận input/output" | Trọng lượng đầu LM = chuyển vị của ma trận token embedding; tiết kiệm parameters, nâng cao chất lượng. |
| BF16 tự động | "Training precision mánh khóe" | Chạy forward/back trong bf16, giữ trạng thái optimizer ở fp32; tiêu chuẩn từ năm 2021. |
| Gradient cắt | "Ngăn chặn gai" | Giới hạn tiêu chuẩn tốt nghiệp toàn cầu ở mức 1.0; ngăn chặn training nổ tung. |
| Lịch trình Cosine LR | "Mặc định năm 2020+" | LR tăng tuyến tính (khởi động) sau đó phân rã hình cosin đến 10% đỉnh. |
| MFU | "Model Sử dụng FLOP" | Đạt đỉnh FLOPs / lý thuyết; 40% mật độ, 30% MoE mạnh vào năm 2026. |
| Val loss | "Những loss bị giam giữ" | Entropy chéo trên dữ liệu mà model chưa bao giờ nhìn thấy; Máy dò quá khớp. |

## Đọc thêm

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) — triển khai có chú thích cổ điển.
