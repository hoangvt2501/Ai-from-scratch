# Mã hóa vị trí - hình sin, RoPE, ALiBi

> Attention là hoán vị-bất biến. "Con mèo ngồi trên thảm" và "con mèo ngồi trên thảm" tạo ra cùng một đầu ra mà không có tín hiệu vị trí. Ba thuật toán khắc phục nó - mỗi thuật toán có một đặt cược khác nhau về ý nghĩa của "vị trí".

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 7 · 02 (Self-Attention), Giai đoạn 7 · 03 (Multi-Head Attention)
**Thời lượng:** ~45 phút

## Vấn đề

Các attention tích chấm theo tỷ lệ là mù trật tự. Ma trận attention `softmax(Q K^T / √d) V` được tính toán từ các điểm tương đồng theo cặp. Xáo trộn các hàng `X`, xáo trộn các hàng đầu ra theo cách tương tự. Không có gì bên trong attention quan tâm đến vị trí.

Đó không phải là một lỗi trong một túi model. Đối với ngôn ngữ, mã, âm thanh, video - bất cứ thứ gì mà trật tự mang ý nghĩa - đều gây tử vong.

Cách khắc phục là tiêm vị trí vào embeddings bằng cách nào đó. Ba thời đại của câu trả lời:

1. **Xoang tuyệt đối** (Vaswani 2017). Thêm `sin/cos` vị trí vào embedding. Đơn giản, không có thể học được, ngoại suy kém vượt quá độ dài đã được huấn luyện.
2. **RoPE - Vị trí quay Embeddings **(Su 2021). Xoay Q và K vectors theo một góc tỷ lệ thuận với vị trí. Mã hóa vị trí * tương đối * trực tiếp trong sản phẩm chấm. Thống trị vào năm 2026.
3. **ALiBi - Attention với các thành kiến tuyến tính **(Nhấn 2022). Bỏ qua hoàn toàn embeddings; Thêm hình phạt tuyến tính trên mỗi đầu để attention điểm dựa trên khoảng cách. Ngoại suy độ dài tuyệt vời.

Tính đến năm 2026, về cơ bản mọi model mở biên giới đều sử dụng RoPE: Llama 2/3/4, Qwen 2/3, Mistral, Mixtral, DeepSeek-V3, Kimi. Một số models ngữ cảnh dài sử dụng ALiBi hoặc các biến thể hiện đại của nó. Hình sin tuyệt đối là lịch sử.

## Khái niệm

![Sinusoidal absolute vs RoPE rotations vs ALiBi distance bias](../assets/positional-encoding.svg)

### Hình sin tuyệt đối

Tính toán trước ma trận cố định `PE` hình dạng `(max_len, d_model)`:

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

Sau đó `X' = X + PE[:N]` trước attention. Mỗi chiều là một hình sin ở một tần số khác nhau. model học cách đọc vị trí từ mẫu pha. Thất bại vượt quá `max_len`: không có gì cho model biết điều gì xảy ra ở vị trí 2048 khi nó chỉ nhìn thấy các vị trí 0–2047.

### Dây thừng

Xoay vectors Q và K (không phải embeddings). Đối với một cặp kích thước `(2i, 2i+1)`:

```
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base = 10000 by default
```

Áp dụng cùng một vòng xoay cho các phím có vị trí `pos_k`. Sản phẩm chấm `q'_m · k'_n` trở thành một hàm của `(m - n)` một mình. Đó là: **điểm số attention chỉ phụ thuộc vào khoảng cách tương đối**, mặc dù vòng quay được khóa khỏi các vị trí tuyệt đối. Thủ thuật đẹp.

Mở rộng RoPE: `base` có thể được chia tỷ lệ (nhận biết NTK, YaRN, LongRoPE) để ngoại suy đến các bối cảnh dài hơn mà không cần huấn luyện lại. Llama 3 mở rộng ngữ cảnh từ 8K lên 128K theo cách này.

### ALiBi

Bỏ qua thủ thuật embedding. Bias điểm attention trực tiếp:

```
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

Trong đó `m_h` là độ dốc dành riêng cho đầu (ví dụ: `1 / 2^(8·h/H)`). Gần hơn tokens được tăng cường; tokens bị phạt. Không tốn chi phí training lần. Bài báo cho thấy phép ngoại suy chiều dài đánh bại hình sin và phù hợp với RoPE trên chiều dài được huấn luyện ban đầu của nó.

### Chọn gì vào năm 2026

| Biến thể | Ngoại suy | Chi phí Training | Được sử dụng bởi |
|---------|---------------|---------------|---------|
| Hình sin tuyệt đối | Nghèo | Miễn phí | transformer ban đầu, đầu BERT |
| Học tuyệt đối | Không có | tí hon | GPT-2, GPT-3 |
| Dây thừng | Tốt với việc mở rộng quy mô | Miễn phí | Llama 2/3/4, Qwen 2/3, Mistral, DeepSeek-V3, Kimi |
| Dây + YaRN | xuất sắc | Giai đoạn fine-tune | Qwen2-1M, Llama 3.1 128K |
| ALiBi | xuất sắc | Miễn phí | BLOOM, MPT, Bạch Xuyên |

RoPE đã giành chiến thắng vì nó cắm vào attention mà không thay đổi kiến trúc, mã hóa vị trí tương đối và `base` hyperparameter của nó mang lại một núm gọn gàng cho fine-tuning ngữ cảnh dài.

```figure
rope-explorer
```

## Tự xây dựng

### Bước 1: mã hóa hình sin

Xem `code/main.py`. Tính toán 4 dòng:

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

Thêm điều này vào ma trận embedding trước layer attention đầu tiên.

### Bước 2: Áp dụng RoPE cho Q, K

RoPE hoạt động tại chỗ trên Q và K. Đối với mỗi cặp mờ:

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

Quan trọng: áp dụng chức năng tương tự cho Q ở vị trí `m` và K ở vị trí `n`. Sản phẩm chấm của họ chiếm một hệ số `cos((m-n)·θ_i)` trên mỗi cặp tọa độ. Attention học vị trí tương đối miễn phí.

### Bước 3: Độ dốc và bias ALiBi

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # add to attention scores before softmax
```

Thêm `bias[h]` vào ma trận điểm `(seq_len, seq_len)` attention của `h` đầu, sau đó softmax.

### Bước 4: xác minh thuộc tính khoảng cách tương đối của RoPE

Chọn hai vectors `a, b` ngẫu nhiên. Xoay theo `(pos_a, pos_b)`. Sau đó là `(pos_a + k, pos_b + k)`. Cả hai sản phẩm dấu chấm phải khớp trong lỗi dấu phẩy động. Thuộc tính đó là toàn bộ điểm của RoPE - nó bất biến với độ lệch tuyệt đối, chỉ có khoảng cách tương đối mới quan trọng.

## Ứng dụng

PyTorch 2.5+ ships tiện ích RoPE tại `torch.nn.functional`. Hầu hết mã production sử dụng `flash_attn` hoặc `xformers` nơi RoPE được áp dụng bên trong hạt nhân attention.

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**Thủ thuật ngữ cảnh dài vào năm 2026:**

- **Nội suy nhận biết NTK.** Thay đổi tỷ lệ `base` thành `base * (scale_factor)^(d/(d-2))` khi mở rộng từ 4K lên 16K+.
- **YaRN.** Nội suy thông minh hơn giúp bảo toàn entropy attention trong ngữ cảnh dài. Llama 3.1 128K sử dụng nó.
- **LongRoPE.** Phương pháp năm 2024 của Microsoft sử dụng tìm kiếm tiến hóa để chọn các hệ số tỷ lệ trên mỗi chiều. Phi-3-Long sử dụng nó.
- **Nội suy vị trí + fine-tuning.** Chỉ cần thu nhỏ vị trí theo hệ số mở rộng và fine-tune cho tokens 1–5B. Hiệu quả đáng ngạc nhiên.

## Sản phẩm bàn giao

Xem `outputs/skill-positional-encoding-picker.md`. skill chọn một chiến lược mã hóa cho một model mới nhất định với độ dài ngữ cảnh mục tiêu, nhu cầu ngoại suy và ngân sách training.

## Bài tập

1. **Dễ dàng.** Vẽ ma trận `PE` hình sin làm bản đồ nhiệt cho `max_len=512, d=128`. Xác nhận mô hình "sọc rộng hơn khi chỉ số kích thước tăng lên".
2. **Trung bình.** Triển khai mở rộng quy mô RoPE nhận biết NTK. Huấn luyện một LM nhỏ trên các dãy có độ dài 256, sau đó kiểm tra độ dài 1024 có và không có tỷ lệ. Đo lường perplexity.
3. **Khó.** Triển khai ALiBi và RoPE trong cùng một mô-đun attention. Huấn luyện một transformer 4 lớp trên một tác vụ sao chép với các chuỗi có độ dài 512. Ngoại suy đến 2048 tại thời điểm thử nghiệm. So sánh sự xuống cấp.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Mã hóa vị trí | "Nói với attention về trật tự" | Bất kỳ tín hiệu nào được thêm vào embeddings hoặc attention mã hóa vị trí. |
| Hình sin | "Cái gốc" | `sin/cos` ở tần số hình học được thêm vào embeddings; không ngoại suy. |
| Dây thừng | "embeddings quay" | Xoay Q, K theo góc phụ thuộc vào vị trí; Sản phẩm chấm mã hóa khoảng cách tương đối. |
| ALiBi | "Thủ thuật bias tuyến tính" | Thêm '-m·\ | i-j\ | ' để attention điểm; không cần embedding, ngoại suy tuyệt vời. |
| Căn cứ | "Núm của RoPE" | Bộ chia tỷ lệ tần số trong RoPE; tăng để mở rộng ngữ cảnh ở inference. |
| Nhận biết NTK | "Thủ thuật mở rộng quy mô RoPE" | Thay đổi `base` để độ mờ tần số cao không bị ép khi ngữ cảnh mở rộng. |
| YaRN | "Cái ưa thích" | Nội suy trên mỗi chiều + ngoại suy bảo toàn entropy attention. |
| Ngoại suy | "Làm việc vượt quá độ dài được huấn luyện" | Sơ đồ vị thế có thể phục vụ đầu ra chính xác trong quá khứ `max_len` thấy trong training không? |

## Đọc thêm

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) — hình sin ban đầu.
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) - Giấy RoPE.
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) - ALiBi.
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) - mở rộng quy mô RoPE hiện đại.
- [Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) — Bài báo ngữ cảnh dài Llama 2 của Meta.
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) — phương pháp của Microsoft được sử dụng bởi Phi-3-Long và được trích dẫn trong phần Sử dụng nó.
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) - triển khai cấp production của mọi sơ đồ mở rộng RoPE (mặc định, tuyến tính, động, YaRN, LongRoPE, Llama-3).
