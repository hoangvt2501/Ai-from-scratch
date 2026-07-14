# Tối ưu hóa KV Cache, Flash Attention và Inference

> Training song song và bị ràng buộc bởi FLOP. Inference nối tiếp và ràng buộc bộ nhớ. Nút thắt cổ chai khác nhau, thủ thuật khác nhau.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 7 · 02 (Self-Attention), Giai đoạn 7 · 05 (Transformer đầy đủ), Giai đoạn 7 · 07 (GPT)
**Thời lượng:** ~75 phút

## Vấn đề

Một decoder tự hồi quy ngây thơ `O(N²)` hoạt động để tạo ra `N` tokens: ở mỗi bước, nó tính toán lại attention trên tiền tố đầy đủ. Đối với phản hồi 4K-token là 16 triệu attention hoạt động, hầu hết chúng đều dư thừa. Mọi trạng thái ẩn của tiền tố token đều có tính xác định sau khi được tính toán - bạn chỉ cần chạy truy vấn của token mới đối với các khóa và giá trị được lưu trong bộ nhớ cache của mọi thứ trước đó.

Trên hết, bản thân attention di chuyển rất nhiều dữ liệu. attention tiêu chuẩn cụ thể hóa ma trận điểm N×N, đầu ra softmax N×d, đầu ra cuối cùng ×d - quá nhiều lần đọc và ghi vào HBM. Đối với N≥2K, attention trở thành giới hạn bộ nhớ trước khi nó trở thành FLOP ràng buộc. Hạt attention cổ điển sử dụng ít GPUs hiện đại từ 4–10×.

Hai tối ưu hóa, cả hai đều từ Dao và cộng sự, đã đẩy inference biên giới từ "chậm" sang "nhanh":

1. **KV cache.** Lưu trữ vectors K và V của mọi tiền tố token. Mỗi attention của token mới là một truy vấn đối với các khóa được lưu trong bộ nhớ đệm. Inference giảm từ `O(N²)` xuống `O(N)` mỗi bước phát điện.
2. **Flash Attention.** Xếp lớp tính toán attention để ma trận N×N đầy đủ không bao giờ chạm HBM. Tất cả softmax + matmul đều xảy ra trong SRAM. Tăng tốc đồng hồ treo tường 2–4× trên A100; 5–10× trên H100 với FP8.

Đến năm 2026, cả hai đều phổ biến. Mọi production inference stack (vLLM, TensorRT-LLM, SGLang, llama.cpp) đều giả định chúng. Mọi biên giới đều model ships bật Flash Attention.

## Khái niệm

![KV cache growth and Flash Attention tiling](../assets/kv-cache-flash-attn.svg)

### KV cache toán học

Mỗi decoder lớp, mỗi token, mỗi con:

```
bytes_per_token_per_layer = 2 * d_head * dtype_size
                          ^
                          K and V
```

Đối với model 7B có 32 lớp, 32 đầu, d_head = 128, fp16:

```
per token per layer = 2 * 128 * 2 = 512 bytes
per token (32 layers) = 16 KB
per 32K context = 512 MB
```

Đối với Llama 3 70B (80 lớp, d_head = 128, GQA với 8 đầu KV):

```
per token per layer = 2 * 8 * 128 * 2 = 4096 bytes (4 KB)
per 32K context = 10.4 GB
```

10 GB đó là lý do tại sao Llama 3 70B ở ngữ cảnh 128K cần hầu hết A100 40 GB chỉ để KV cache ở batch kích thước 1.

**GQA là chiến thắng KV-cache.** MHA với 64 đầu sẽ là 32 GB. MLA thậm chí còn nén hơn nữa.

Kéo kích thước và xem kích thước bộ nhớ đệm di chuyển. Đẩy độ dài trình tự hoặc batch lên và xem nó thổi qua một GPU nhanh như thế nào:

```figure
kv-cache-sizer
```

### Flash Attention — thủ thuật lát gạch

attention tiêu chuẩn:

```
S = Q @ K^T          (HBM read, N×N, HBM write)
P = softmax(S)       (HBM read, HBM write)
O = P @ V            (HBM read, HBM write)
```

Ba chuyến khứ hồi HBM. Trên H100, băng thông HBM là 3 TB/s; SRAM là 30 TB/s. Mỗi chuyến đi HBM là một yếu tố 10 chậm lại so với việc giữ mọi thứ trên chip.

Flash Attention:

```
for each block of Q (tile size ~128 × 128):
    load Q_tile into SRAM
    for each block of K, V:
        load K_tile, V_tile into SRAM
        compute S_tile = Q_tile @ K_tile^T     (SRAM)
        running softmax aggregation             (SRAM)
        accumulate into O_tile                  (SRAM)
    write O_tile to HBM
```

Một chuyến đi HBM cho mỗi ô. Tổng dung lượng bộ nhớ giảm từ `O(N²)` xuống `O(N)`. Backward pass tính toán lại một số giá trị từ forward pass thay vì lưu trữ chúng - một chiến thắng bộ nhớ khác.

**Thủ thuật số.** Chạy softmax duy trì `(max, sum)` trên các ô để chuẩn hóa cuối cùng là chính xác. Không phải là gần đúng - Flash Attention tính toán đầu ra giống hệt bit với attention tiêu chuẩn (modulo fp16 không liên kết).

**Sự phát triển của phiên bản:**

| Phiên bản | Năm | Thay đổi chính | Tăng tốc trên phần cứng tham chiếu |
|---------|------|-----------|-------------------------------|
| Đèn flash 1 | 2022 | Hạt nhân SRAM lát gạch | 2× trên A100 |
| Đèn flash 2 | 2023 | Tính song song tốt hơn, thứ tự nhân quả đầu tiên | 3× trên A100 |
| Đèn flash 3 | 2024 | Phễu không đồng bộ, FP8 | 1.5–2× trên H100 (~740 TFLOPs FP16) |
| Đèn flash 4 | 2026 | Blackwell 5-stage pipeline, phần mềm exp2 | Inference đầu tiên (ban đầu chỉ chuyển tiếp) |

Flash 4 chỉ chuyển tiếp khi khởi chạy. Training vẫn sử dụng Flash 3. Hỗ trợ GQA và varlen cho Flash 4 đang chờ xử lý (giữa năm 2026).

### Giải mã đầu cơ - chiến thắng độ trễ khác

model giá rẻ đề xuất N tokens. Big model xác minh tất cả N song song. Nếu xác minh chấp nhận k tokens, bạn đã trả 1 model forward pass lớn cho k thế hệ. Điển hình k = 3–5 trên mã và văn xuôi.

Mặc định năm 2026:
- **EAGLE 2 / Medusa.** Các đầu nháp tích hợp chia sẻ trạng thái ẩn của người xác minh. Tăng tốc 2–3× mà không có loss chất lượng.
- **Giải mã suy đoán với model nháp.** Tăng tốc 2–4× trên phần cứng tiêu dùng.
- **Giải mã Lookahead.** Lặp lại Jacobi; Không cần model nháp. Thích hợp nhưng miễn phí.

### Lô liên tục

inference hàng loạt cổ điển: đợi trình tự chậm nhất kết thúc, sau đó bắt đầu một batch mới. Lãng phí GPU khi các câu trả lời ngắn kết thúc sớm.

Batching liên tục (shipped đầu tiên trong Orca, bây giờ là vLLM, TensorRT-LLM, SGLang): hoán đổi yêu cầu mới vào batch ngay sau khi yêu cầu cũ kết thúc. Tăng thông lượng 5–10× cho khối lượng công việc trò chuyện thông thường.

### PagedAttention — KV cache dưới dạng bộ nhớ ảo

Tiêu đề của vLLM feature. KV cache được phân bổ theo khối 16-token; Bảng trang ánh xạ các vị trí logic với các khối vật lý. Cho phép bạn chia sẻ KV trên các mẫu song song (beam search, sampling song song), tiền tố hoán đổi nóng cho bộ nhớ đệm prompt và bộ nhớ chống phân mảnh. 4× Cải thiện thông lượng so với phân bổ liền kề ngây thơ.

```figure
flash-attention-memory
```

## Tự xây dựng

Xem `code/main.py`. Chúng ta thực hiện:

1. Một decoder ngây thơ `O(N²)` gia tăng.
2. Một decoder bộ nhớ đệm KV `O(N)`.
3. Một softmax xếp gạch mô phỏng thuật toán chạy tối đa của Flash Attention.

### Bước 1: KV cache

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

Đơn giản: tiếp tục phát triển trên mỗi token K, V vectors trong danh sách mỗi lớp, mỗi đầu.

### Bước 2: lát gạch softmax

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash-attention-style softmax(qK^T)V with running max/sum."""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

Đầu ra giống hệt bit với `softmax(qK) V` trong một lần chụp, nhưng bất cứ lúc nào tập làm việc là một khối `tile × d_head`, không phải là `N × d_head` đầy đủ.

### Bước 3: so sánh giải mã naïve và cache trên thế hệ 100-token

Đếm attention hoạt động. Ngây thơ: `O(N²)` = 5050. Bộ nhớ đệm: `O(N)` = 100. Mã in cả hai.

## Ứng dụng

```python
# HuggingFace transformers auto-enables KV cache on decoder-only generate().
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # use FA3 if Hopper
    torch_dtype="bfloat16",
)
# generate() uses KV cache automatically
```

vLLM production:

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

Bộ nhớ đệm tiền tố trên các yêu cầu là một chiến thắng lớn vào năm 2026 — cùng một system prompt, ví dụ few-shot hoặc tài liệu ngữ cảnh dài sử dụng lại KV trong các cuộc gọi. Đối với khối lượng công việc agent có prompts công cụ lặp đi lặp lại, bộ nhớ đệm tiền tố thường tăng thông lượng 5×.

## Sản phẩm bàn giao

Xem `outputs/skill-inference-optimizer.md`. skill chọn triển khai attention, chiến lược KV cache, quantization và giải mã đầu cơ cho việc triển khai inference mới.

## Bài tập

1. **Dễ dàng.** Chạy `code/main.py`. Xác nhận decoders ngây thơ và được lưu trong bộ nhớ đệm tạo ra cùng một đầu ra; lưu ý sự khác biệt về số lượng op.
2. **Trung bình.** Triển khai bộ nhớ đệm tiền tố: cho một prompt P và một số lần hoàn thành, hãy chạy một forward pass trên P để lấp đầy KV cache, sau đó branch mỗi lần hoàn thành. Đo tốc độ so với mã hóa lại P cho từng loại.
3. **Khó.** Triển khai đồ chơi PagedAttention: KV cache trong các khối 16 token cố định với danh sách miễn phí. Khi một trình tự kết thúc, hãy trả lại các khối của nó vào nhóm. Mô phỏng 1.000 lần hoàn thành trò chuyện với độ dài khác nhau. So sánh phân mảnh bộ nhớ và phân bổ liền kề.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| KV cache | "Thủ thuật giúp giải mã nhanh chóng" | Lưu trữ K và V từ mọi tiền tố token; các truy vấn mới tham gia vào chúng thay vì tính toán lại. |
| HBM | "GPU ký ức chính" | bộ nhớ băng thông cao; 80 GB trên H100, 192 GB trên B200. ~3 TB/s băng thông. |
| SRAM | "Bộ nhớ trên chip" | Bộ nhớ nhanh trên mỗi SM, ~256 KB mỗi SM trên H100. ~30 TB/s băng thông. |
| Flash Attention | "Hạt nhân attention lát" | Tính toán attention mà không cụ thể hóa N×N trong HBM. |
| Lô liên tục | "Không chờ đợi" | Hoán đổi các trình tự đã hoàn thành ra, những trình tự mới vào mà không làm cạn kiệt batch. |
| PagedChú ý | "Tiêu đề của vLLM" | KV cache được phân bổ trong các khối cố định với bảng trang; loại bỏ sự phân mảnh. |
| Bộ nhớ đệm tiền tố | "Tái sử dụng prompts dài" | Bộ nhớ cache KV cho tiền tố được chia sẻ trên các yêu cầu; cắt giảm chi phí lớn cho agents. |
| Giải mã suy đoán | "Soạn thảo + xác minh" | model dự thảo giá rẻ đề xuất tokens; Big model xác minh K trong một lần vượt qua. |

## Đọc thêm

- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) - Đèn flash 1.
- [Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) - Đèn flash 2.
- [Shah et al. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) - Đèn flash 3.
- [FlashAttention-4 release notes (Dao-AILab, 2026)](https://github.com/Dao-AILab/flash-attention) - pipeline 5 giai đoạn Blackwell và thủ thuật phần mềm-exp2; đọc repo README để biết các cảnh báo chỉ khởi chạy chuyển tiếp mà bài học này đề cập.
- [Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — giấy vLLM.
- [Leviathan et al. (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) - giải mã thông số kỹ thuật.
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) - EAGLE-1/2 bài báo cho cách tiếp cận dự thảo tích hợp mà bài học trích dẫn.
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) - cách tiếp cận Medusa được tham chiếu cùng với EAGLE.
- [vLLM docs — PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) - đi sâu vào thiết kế khối 16 token và bảng trang.
