# Attention Biến thể — Cửa sổ trượt, thưa thớt, vi sai

> attention đầy đủ là một vòng tròn. Mọi token nhìn thấy mọi token, và ký ức phải trả giá. Bốn biến thể bẻ cong hình dạng của hình tròn và thu hồi một nửa chi phí.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 7 · 02 (Self-Attention), Giai đoạn 7 · 03 (Nhiều đầu), Giai đoạn 7 · 12 (KV Cache / Flash Attention)
**Thời lượng:** ~60 phút

## Vấn đề

attention đầy đủ tốn bộ nhớ `O(N²)` và tính toán `O(N²)` theo độ dài trình tự. Đối với ngữ cảnh 128K Llama 3 70B là 16 tỷ attention mục nhập trên mỗi lớp, nhân với 80 lớp. Flash Attention (Bài 12) ẩn bộ nhớ kích hoạt `O(N²)` nhưng không thay đổi chi phí số học - mọi token vẫn tham gia vào mọi token khác.

Ba classes biến thể thay đổi cấu trúc liên kết của chính ma trận attention:

1. **Cửa sổ trượt attention (SWA).** Mỗi token chú ý đến một cửa sổ cố định của hàng xóm, không phải tiền tố đầy đủ. Bộ nhớ và điện toán giảm xuống `O(N · W)` nơi `W` là cửa sổ. Gemma 2/3, lớp đầu tiên của Mistral 7B, Phi-3-Long.
2. **attention thưa thớt / khối.** Chỉ các cặp được chọn mới `(i, j)` được ghi bàn; các rest buộc phải có trọng lượng bằng không. Longformer, BigBird OpenAI transformer thưa thớt.
3. **attention vi sai.** Tính toán hai bản đồ attention với các phép chiếu Q/K riêng biệt, trừ một bản đồ này với cái kia. Giết chết "bồn rửa attention" chảy máu trọng lượng trong vài tokens đầu tiên. Transformer DIFF của Microsoft (2024).

Những điều này cùng tồn tại. Một model biên giới năm 2026 thường trộn lẫn chúng: hầu hết các lớp là SWA-1024, cứ một phần năm là attention đầy toàn cầu và một số ít là các đầu vi sai giúp dọn dẹp truy xuất. Tỷ lệ SWA trên toàn cầu 5: 1 của Gemma 3 là mặc định trong sách giáo khoa hiện tại.

## Khái niệm

### Cửa sổ trượt Attention (SWA)

Mỗi truy vấn tại vị trí `i` chỉ quan tâm đến các vị trí trong `[i - W, i]` (SWA nhân quả) hoặc `[i - W/2, i + W/2]` (hai chiều). Tokens bên ngoài cửa sổ nhận `-inf` trong ma trận điểm số.

```
full causal:           sliding window (W=4):
positions 0-7          positions 0-7, W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x
```

Đối với `N = 8192` và `W = 1024`, ma trận điểm số có 1024 × 8192 hàng không phải số không được kỳ vọng - giảm 8×.

**KV cache thu nhỏ với SWA.** Chỉ cần giữ `W` tokens cuối cùng của K và V trên mỗi lớp. Đối với một config kiểu Gemma-3 (cửa sổ 1024, ngữ cảnh 128K), KV cache giảm 128×.

**Chi phí chất lượng.** Chỉ có SWA transformers gặp khó khăn với việc truy xuất tầm xa. Cách khắc phục: xen kẽ các lớp SWA với các lớp attention đầy đủ. Gemma 3 sử dụng 5:1 SWA:global. Mistral 7B đã sử dụng một stack SWA nhân quả, trong đó thông tin "chảy về phía trước" thông qua các windows chồng chéo - mỗi lớp mở rộng trường tiếp nhận hiệu quả lên `W` và sau `L` lớp, model có thể tham gia `L × W` tokens trở lại.

### Thưa thớt / Khối Attention

Chọn trước một mẫu `N × N` thưa thớt. Ba hình dạng chuẩn:

- **Địa phương + sải bước (OpenAI transformer thưa thớt).** Tham dự `W` tokens cuối cùng cộng với mỗi token thứ `stride` trước đó. Ghi lại cả cục bộ và tầm xa tại `O(N · sqrt(N))` tính toán.
- **Longformer / BigBird.** Cửa sổ cục bộ + một tập hợp nhỏ các tokens toàn cầu (ví dụ: `[CLS]`) tham gia cho tất cả mọi người và có sự tham gia của tất cả mọi người + các liên kết thưa thớt ngẫu nhiên. Bối cảnh thực nghiệm 2× ở chất lượng phù hợp.
- **Attention thưa thớt gốc (DeepSeek, 2025).** Tìm hiểu khối `(Q, K)` nào quan trọng; Bỏ qua các khối không ở cấp hạt nhân. Tương thích với FlashAttention.

Sparse attention là một câu chuyện kỹ thuật hạt nhân. Toán học rất đơn giản (che ma trận điểm); chiến thắng đến từ việc không bao giờ tải các mục nhập bằng không vào SRAM. FlashAttention-3 và FlexAttention 2026 API tạo ra các mẫu thưa thớt tùy chỉnh class đầu tiên trong PyTorch.

### Attention vi sai (DIFF Transformer, 2024)

attention thông thường có một vấn đề "attention chìm": softmax buộc mọi hàng phải tổng bằng 1, vì vậy tokens không muốn chú ý đến bất cứ điều gì cụ thể trọng lượng kết xuất vào token đầu tiên (hoặc một vài lần đầu tiên). Điều này đánh cắp dung lượng đáng lẽ phải đi vào nội dung thực.

attention vi phân khắc phục điều này bằng cách tính **hai** attention bản đồ và trừ:

```
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

trong đó `λ` là một vô hướng đã học (thường là 0,5–0,8). A1 nắm bắt trọng số nội dung thực; A2 chụp bồn rửa. Phép trừ hủy bồn rửa, phân bổ lại trọng lượng cho các tokens có liên quan.

Kết quả được báo cáo (Microsoft 2024): perplexity thấp hơn 5–10%, ngữ cảnh hiệu quả dài hơn 1,5–2× ở cùng độ dài được huấn luyện, truy xuất kim trong đống cỏ khô sắc nét hơn.

### So sánh biến thể

| Biến thể | Điện toán | KV cache | Chất lượng so với đầy đủ | Production sử dụng |
|---------|---------|----------|-----------------|----------------|
| attention đầy đủ | O (N²) | O (N) mỗi lớp | Đường cơ sở | mỗi lớp mặc định của model |
| SWA (cửa sổ 1024) | O (N · W) | O (W) mỗi lớp | -0,1 ppl, tốt với các lớp toàn cầu | Gemma 2/3, Phi-3-Long |
| Địa phương + sải bước thưa thớt | O (N · √N) | hỗn hợp | Tương tự với SWA | OpenAI transformer thưa thớt, Longformer |
| BigBird (cục bộ + toàn cầu + ngẫu nhiên) | Xấp xỉ O (N) | hỗn hợp | Các trận đấu đầy đủ ở 2× ngữ cảnh | BERT bối cảnh dài ban đầu |
| Thưa thớt gốc (DeepSeek-V3.2) | O (N · phân số hoạt động) | O (N) | trong vòng 0,05 ppl | Tìm kiếm sâu-V3.2, 2025 |
| Sự khác biệt | O (2 · N²) | O (2N) | -5 đến -10% mỗi người | DIFF Transformer, đầu năm 2026 models |

```figure
gqa-kv-sharing
```

## Tự xây dựng

Xem `code/main.py`. Chúng ta triển khai công cụ so sánh mặt nạ nhân quả hiển thị attention đầy đủ, SWA, cục bộ + sải bước và vi sai cạnh nhau trên một chuỗi đồ chơi.

### Bước 1: mặt nạ nhân quả đầy đủ (đường cơ sở)

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

Đường cơ sở từ Bài 07. Hình tam giác dưới; trọng lượng bằng không trên đường chéo.

### Bước 2: mặt nạ nhân quả cửa sổ trượt

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

Một parameter - `window`. Đối với `window >= n`, bạn phục hồi đầy đủ attention nhân quả. Đối với `window = 1`, mỗi token chỉ quan tâm đến chính nó.

### Bước 3: mặt nạ thưa thớt cục bộ + sải bước

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

Cửa sổ cục bộ dày đặc cộng với mỗi `stride` token quay trở lại phần đầu của chuỗi. Trường tiếp nhận phát triển theo các bước nhật ký với các lớp bổ sung.

### Bước 4: attention vi sai

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

Hai attention vé, trừ với hệ số trộn đã học. Trong mã, chúng tôi so sánh bản đồ nhiệt attention chìm của đơn và vi sai và xem bồn rửa sụp đổ.

### Bước 5: KV cache kích thước

In kích thước bộ nhớ đệm trên mỗi lớp ở mức `N = 131072` cho mỗi biến thể. SWA và các biến thể thưa thớt giảm 10–100×. Nhân đôi chênh lệch. Thanh toán hóa đơn bộ nhớ của bạn một cách có ý thức.

## Ứng dụng

Mô hình production năm 2026:

```python
from transformers import AutoModelForCausalLM
# Gemma 3 mixes SWA (window=1024) and global layers at 5:1.
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

FlexAttention trong PyTorch 2.5+ chấp nhận chức năng mặt nạ:

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

Điều này biên dịch thành một hạt nhân Triton tùy chỉnh. Trong vòng 10% tốc độ FlashAttention-3 đối với các mẫu phổ biến và chức năng mặt nạ là một Python có thể gọi được.

**Khi nào nên chọn mỗi:**

- **Pure full attention** — mọi lớp lên đến ~16K ngữ cảnh hoặc khi chất lượng truy xuất là tối quan trọng.
- **SWA + hỗn hợp toàn cầu** — ngữ cảnh dài (>32K), training và inference bị ràng buộc bộ nhớ. Mặc định năm 2026 trên 32K.
- **attention khối thưa thớt **- hạt nhân tùy chỉnh, mẫu tùy chỉnh. Dành riêng cho khối lượng công việc chuyên biệt (truy xuất, âm thanh).
- **attention khác biệt **— bất kỳ khối lượng công việc nào mà ô nhiễm bồn rửa attention gây tổn thương (RAG ngữ cảnh dài, đống cỏ khô).

## Sản phẩm bàn giao

Xem `outputs/skill-attention-variant-picker.md`. skill chọn cấu trúc liên kết attention cho một model mới với độ dài ngữ cảnh mục tiêu, nhu cầu truy xuất và cấu hình điện toán training/inference.

## Bài tập

1. **Dễ dàng.** Chạy `code/main.py`. Xác minh SWA ở `window=4` không mọi thứ bên ngoài 4 tokens cuối cùng mỗi hàng. Xác minh `window=n` tái tạo đầy đủ attention nhân quả giống hệt nhau.
2. **Trung bình.** Thực hiện SWA nhân quả với `window=1024` trên đỉnh Bài 07. Huấn luyện 1.000 bước trên tinyshakespeare. Val loss thoái lui bao nhiêu so với attention đầy đủ? Bộ nhớ đỉnh giảm bao nhiêu?
3. **Khó.** Triển khai hỗn hợp lớp 5: 1 kiểu Gemma-3 (5 SWA, 1 toàn cầu) trong model capstone. So sánh chất lượng loss, bộ nhớ và thế hệ với các đường cơ sở SWA thuần túy và toàn cầu thuần túy ở parameters phù hợp.
4. **Khó.** Thực hiện attention vi sai với một `λ` đã học trên mỗi đầu. Huấn luyện về một nhiệm vụ truy xuất tổng hợp (một kim, 2.000 chất gây xao nhãng). Đo lường accuracy truy xuất so với đường cơ sở một attention ở parameters phù hợp.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Cửa sổ trượt attention (SWA) | "attention địa phương" | Mỗi truy vấn tham gia vào `W` tokens cuối cùng của nó; KV cache thu nhỏ xuống `O(W)`. |
| Trường tiếp nhận hiệu quả | "model nhìn thấy bao xa" | Trong stack SWA `L` lớp với `W` cửa sổ, tối đa `L × W` tokens. |
| Máy kéo dài / BigBird | "Địa phương + toàn cầu + ngẫu nhiên" | Các mô hình thưa thớt với một số tokens toàn cầu luôn tham dự; cách tiếp cận bối cảnh dài sớm. |
| Attention thưa thớt bản địa | "Thủ thuật hạt nhân của DeepSeek" | Tìm hiểu độ thưa thớt ở cấp độ khối; bỏ qua không khối ở cấp hạt nhân trong khi vẫn giữ chất lượng. |
| attention vi sai | "Hai bản đồ, một trừ" | Transformer DIFF: trừ một `λ` đã học với bản đồ attention giây từ bản đồ đầu tiên để hủy attention bồn rửa. |
| Attention bồn rửa | "Cân nặng chảy token 0" | Chuẩn hóa Softmax buộc các hàng phải có tổng bằng 1; truy vấn không thông tin đổ trọng lượng vào vị trí 0. |
| Chú ý linh hoạt | "Mặt nạ như Python" | PyTorch API 2.5+ biên dịch các hàm mặt nạ tùy ý thành hạt nhân hình FlashAttention. |
| Hỗn hợp loại lớp | "5:1 SWA-to-global" | Xen kẽ các lớp attention thưa thớt và đầy đủ trong một stack để giữ chất lượng ở bộ nhớ thấp hơn. |

## Đọc thêm

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — cửa sổ trượt chuẩn + giấy token toàn cầu.
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) - cục bộ + toàn cầu + ngẫu nhiên.
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) - OpenAI mô hình cục bộ + sải bước.
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) - hỗn hợp SWA: toàn cầu 1: 1.
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786) - hỗn hợp 5: 1 với window = 1024 hiện là mặc định trong sách giáo khoa.
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258) — DIFF Transformer bài báo.
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) - attention thưa thớt đã học của DeepSeek-V3.2.
- [PyTorch — FlexAttention blog and docs](https://pytorch.org/blog/flexattention/) — API tham chiếu cho mẫu mask-as-callable trong Use It.
