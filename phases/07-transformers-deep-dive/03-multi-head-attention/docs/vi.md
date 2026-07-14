# Multi-Head Attention

> Một người đứng đầu attention học từng mối quan hệ một. Tám đầu học tám. Đầu là tự do. Lấy nhiều hơn trong số họ.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 7 · 02 (Self-Attention từ đầu)
**Thời lượng:** ~75 phút

## Vấn đề

Một đầu self-attention duy nhất tính toán một ma trận attention. Ma trận đó nắm bắt một loại mối quan hệ - thường là mối quan hệ giảm thiểu loss trên bất kỳ tín hiệu training nào. Nếu dữ liệu của bạn có sự đồng ý chủ ngữ-động từ, đồng tham chiếu, diễn ngôn tầm xa và phân đoạn cú pháp tất cả đều rối rắm với nhau, một đầu duy nhất sẽ bôi chúng thành một phân phối mềm-tối đa duy nhất và mất một nửa tín hiệu.

Bản sửa lỗi từ bài báo Vaswani năm 2017: chạy song song một số hàm attention, mỗi hàm có phép chiếu Q, K, V riêng và nối các đầu ra. Mỗi đầu hoạt động trong một không gian con nhỏ hơn của `d_model / n_heads` chiều. Tổng parameters giữ nguyên. Sức mạnh biểu cảm tăng lên.

Multi-head attention là mặc định mỗi transformer vào năm 2026 ships. Lập luận duy nhất là về *bao nhiêu* đầu và liệu các khóa và giá trị có chia sẻ phép chiếu hay không (Grouped-Query Attention, Multi-Query Attention, Multi-head Latent Attention).

## Khái niệm

![Multi-head attention splits, attends, concatenates](../assets/multi-head-attention.svg)

**Tách.** Lấy `X` hình dạng `(N, d_model)`. Chiếu đến Q, K, V mỗi hình dạng `(N, d_model)`. Định hình lại để `(N, n_heads, d_head)` nơi `d_head = d_model / n_heads`. Chuyển sang `(n_heads, N, d_head)`.

**Tham dự song song.** Chạy attention sản phẩm chấm theo tỷ lệ bên trong mỗi đầu. Mỗi đầu sản xuất `(N, d_head)`. Các đầu hoạt động trên các không gian con khác nhau của embedding và không bao giờ nói chuyện trong quá trình tính toán attention.

**Nối và chụp.** Stack quay trở lại `(N, d_model)` và nhân với ma trận đầu ra đã học `W_o` `(d_model, d_model)` hình dạng. `W_o` là nơi những cái đầu có thể trộn lẫn.

**Tại sao nó hoạt động.** Mỗi người đứng đầu có thể chuyên môn hóa mà không cạnh tranh với những người khác về ngân sách đại diện. Các nghiên cứu thăm dò từ năm 2019–2024 cho thấy vai trò của người đứng đầu riêng biệt: người đứng đầu theo vị trí, người đứng đầu theo token trước, đầu sao chép, người đứng đầu thực thể được đặt tên, người đứng đầu cảm ứng (làm nền tảng cho việc học theo ngữ cảnh).

**Dòng dõi các biến thể năm 2026:**

| Biến thể | Đầu Q | K/V đầu | Được sử dụng bởi |
|---------|---------|-----------|---------|
| Nhiều đầu (MHA) | N | N | GPT-2, BERT, T5 |
| Đa truy vấn (MQA) | N | 1 | PaLM, Chim ưng |
| Truy vấn nhóm (GQA) | N | G (ví dụ: N/8) | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| Tiềm ẩn nhiều đầu (MLA) | N | nén xuống cấp thấp | Tìm kiếm sâu-V2, V3 |

GQA là mặc định hiện đại vì nó cắt giảm bộ nhớ đệm KV theo hệ số `N/G` trong khi vẫn giữ được chất lượng gần như đầy đủ. MLA đi xa hơn bằng cách nén K/V vào một không gian tiềm ẩn, sau đó chiếu lại thời gian tính toán - chi phí FLOPs, tiết kiệm bộ nhớ hơn rất nhiều.

```figure
multihead-split
```

## Tự xây dựng

### Bước 1: tách đầu từ attention một đầu mà chúng ta đã có

Lấy `SelfAttention` từ Bài 02 và quấn nó bằng một cặp split/concat. Xem `code/main.py` để biết cách triển khai numpy; Logic là:

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

Một lần định hình lại và một lần chuyển vị. Không có vòng lặp. Đây chính xác là những gì PyTorch làm dưới `nn.MultiheadAttention`.

### Bước 2: chạy attention sản phẩm chấm tỷ lệ trên mỗi đầu

Mỗi đầu có một lát Q, K, V riêng của nó. Attention trở thành một matmul hàng loạt:

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

Trên phần cứng thực tế `Qh @ Kh.transpose(...)` là một `bmm`. GPU thấy một mẻ duy nhất có hình dạng `(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)`. Thêm đầu là miễn phí.

### Bước 3: Biến thể Attention truy vấn nhóm

Chỉ có các dự báo khóa và giá trị thay đổi. Q nhận `n_heads` nhóm; K và V nhận được `n_kv_heads < n_heads` nhóm và được lặp lại để khớp:

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

Tại inference điều này giúp tiết kiệm bộ nhớ vì chỉ có `n_kv_heads` bản sao sống trong KV cache chứ không phải `n_heads`. Llama 3 70B sử dụng 64 đầu truy vấn với 8 đầu KV - thu nhỏ bộ nhớ đệm 8×.

### Bước 4: thăm dò những gì mỗi đầu đã học

Chạy MHA trên một câu ngắn với 4 đầu. Đối với mỗi đầu, in ma trận `(N, N)` attention. Bạn sẽ thấy các đầu khác nhau chọn ra cấu trúc khác nhau ngay cả với khởi tạo ngẫu nhiên - đó là một phần tín hiệu, một phần đối xứng quay trong các không gian con.

## Ứng dụng

Trong PyTorch, phiên bản một dòng:

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

GQA tính đến PyTorch 2.5+:

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention auto-dispatches Flash Attention on CUDA.
# For GQA, pass Q of shape (B, n_heads, N, d_head) and K,V of shape
# (B, n_kv_heads, N, d_head). PyTorch handles the repeat.
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**Có bao nhiêu đầu?** Quy tắc ngón tay cái từ production models vào năm 2026:

| Kích thước Model | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| Nhỏ (~125 triệu) | 768 | 12 | 64 |
| Cơ sở (~350M) | 1024 | 16 | 64 |
| Lớn (~1B) | 2048 | 16 | 128 |
| Biên giới (~70B) | 8192 | 64 | 128 |

`d_head` hầu như luôn hạ cánh ở mức 64 hoặc 128. Nó là đơn vị của mức độ một cái đầu có thể "nhìn thấy". Giảm xuống dưới 32 và những người đứng đầu bắt đầu chiến đấu với hệ số tỷ lệ `sqrt(d_head)`; Vượt quá 256 và bạn sẽ mất quyền lợi "nhiều bác sĩ chuyên khoa nhỏ".

## Sản phẩm bàn giao

Xem `outputs/skill-mha-configurator.md`. skill đề xuất số lượng nhân viên, số lượng kv và chiến lược dự báo cho một transformer mới dựa parameter ngân sách, độ dài trình tự và mục tiêu triển khai.

## Bài tập

1. **Dễ dàng.** Lấy MHA từ `code/main.py` và thay đổi `n_heads` từ 1 đến 16 với `d_model=64` cố định. Vẽ loss của một model một lớp nhỏ trên một tác vụ sao chép tổng hợp. Nhiều đầu hơn có giúp ích, ổn định hay tổn thương?
2. **Medium.** Triển khai MQA (một đầu KV được chia sẻ trên tất cả các đầu truy vấn). Đo lường số lượng parameter giảm so với MHA đầy đủ. Tính toán kích thước bộ nhớ đệm KV thu nhỏ ở inference cho N=2048.
3. **Khó.** Triển khai một phiên bản nhỏ của Attention tiềm ẩn nhiều đầu: nén K, V thành tiềm ẩn `r` cấp bậc, lưu trữ tiềm ẩn trong KV cache, giải nén tại attention thời điểm. Bộ nhớ đệm vượt qua 1/8 MHA đầy đủ ở `r` nào trong khi chất lượng vẫn nằm trong khoảng 1 bit của ppl xác thực?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Cái đầu | "Một mạch attention duy nhất" | Một phép chiếu Q/K/V của kích thước `d_head = d_model / n_heads` với ma trận attention riêng của nó. |
| d_head | "Kích thước đầu" | Chiều rộng ẩn trên mỗi đầu; hầu như luôn luôn là 64 hoặc 128 trong production. |
| Tách / kết hợp | "Định hình lại thủ thuật" | `(N, d_model) ↔ (n_heads, N, d_head)` định hình lại + chuyển vị xung quanh attention. |
| W_o | "Chiếu đầu ra" | `(d_model, d_model)` trận được áp dụng sau khi nối các đầu; nơi những cái đầu trộn lẫn. |
| MQA | "Một đầu KV" | Multi-Query Attention: phép chiếu K/V được chia sẻ duy nhất. KV cache nhỏ nhất, một số loss chất lượng. |
| GQA | "Mặc định kể từ Llama 2" | Grouped-Query Attention với `n_kv_heads < n_heads`; lặp lại để khớp với Q. |
| MLA | "Thủ thuật của DeepSeek" | Attention tiềm ẩn nhiều đầu: K, V nén thành tiềm ẩn cấp thấp, giải nén tại thời điểm tham dự. |
| Đầu cảm ứng | "Mạch đằng sau học tập trong ngữ cảnh" | Một cặp đầu phát hiện những lần xuất hiện trước đó và sao chép những gì xảy ra sau đó. |

## Đọc thêm

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) — thông số kỹ thuật nhiều đầu ban đầu.
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) - bài báo MQA.
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) - cách chuyển đổi MHA sang GQA sau training.
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) - MLA và lý do tại sao nó đánh bại MHA/GQA về bộ nhớ đệm.
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) - cái nhìn máy móc về những gì đầu thực sự làm.
