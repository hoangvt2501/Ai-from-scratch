# Gradient Tính toán lại điểm kiểm tra và kích hoạt

> Backprop giữ mọi kích hoạt trung gian. Ở 70B parameters và ngữ cảnh 128K, đó là 3 TB kích hoạt cho mỗi cấp bậc. Điểm kiểm tra đánh đổi FLOPs cho bộ nhớ: tính toán lại thay vì lưu. Câu hỏi đặt ra là phân đoạn nào nên bỏ và câu trả lời không phải là "tất cả chúng".

**Loại:** Xây dựng
**Ngôn ngữ:** Python (with numpy, optional torch)
**Kiến thức tiên quyết:** Giai đoạn 10 Bài 04 (Pre-Training Mini-GPT), Giai đoạn 10 Bài 05 (Mở rộng quy mô & Phân tán)
**Thời lượng:** ~70 phút

## Vấn đề

Training một transformer lưu trữ, đối với mỗi lớp, các đầu vào cho mọi op được phân biệt ngược: đầu vào attention, phép chiếu Q/K/V, đầu ra softmax, đầu vào FFN, đầu ra định mức và luồng dư. Đối với một lớp có kích thước ẩn `d`, độ dài trình tự `L` batch `B`, điều này theo thứ tự `12 * B * L * d` float trên mỗi lớp.

Đối với `d=8192, L=8192, B=1`, đó là 800 MB/layer trong BF16. Một model 64 lớp là 51 GB kích hoạt - và đó là trước khi bạn nhân với kích thước microbatch, trước khi bạn thêm attention softmax trung gian (`L^2` mỗi người) và trước khi bạn phân tích các bản sao một phần song song tensor.

Hóa đơn hai mặt: trọng số BF16 cộng với trạng thái optimizer có thể vừa với 80GB, nhưng các kích hoạt sẽ đẩy bạn qua. Gradient điểm kiểm tra (hay còn gọi là tính toán lại kích hoạt) là bản sửa lỗi tiêu chuẩn. Bỏ hầu hết các kích hoạt; làm lại chuyển tiếp trong quá trình lùi để lấy lại chúng. Chi phí: thêm FLOPs. Lợi ích: bộ nhớ giảm theo tỷ lệ checkpoint phân đoạn so với tổng số lớp.

Được thực hiện một cách ngây thơ, điểm kiểm tra tốn thêm khoảng 33% FLOPs chuyển tiếp cho mỗi bước. Thực hiện tốt - điểm kiểm tra có chọn lọc theo "lựa chọn thông minh" của Korthikanti et al. - bạn tiết kiệm bộ nhớ gấp 5 lần với chi phí FLOP dưới 5%. Và với matmul FP8, giảm tải FSDP và MoE song song chuyên gia, điều này thực sự quan trọng: bạn không thể mua bộ nhớ hoặc điện toán lãng phí.

## Khái niệm

### Những gì Backward thực sự cần

`output = layer(input)`. Backward muốn `grad_input` và `grad_params`. Để tính toán chúng, nó cần:

- `input` (để tính toán `grad_params = input.T @ grad_output` cho các lớp tuyến tính)
- một số dẫn xuất trung gian hoạt hóa (dẫn xuất của ReLU/GELU/softmax phụ thuộc vào giá trị hoạt hóa)

forward pass tự động lưu trữ chúng trong biểu đồ autograd. Mỗi `tensor.retain_grad()` và mọi hoạt động cần đầu vào của nó đều giữ lại một tham chiếu.

### Điểm kiểm tra đầy đủ ngây thơ

Chia mạng thành `N` phân đoạn. Trong quá trình chuyển tiếp, chỉ lưu trữ *đầu vào* cho từng phân đoạn. Khi cần lùi trung gian, hãy chạy lại forward pass của phân đoạn để cụ thể hóa chúng, sau đó phân biệt.

Ví dụ: transformer 32 lớp được chia thành 32 đoạn, mỗi phân đoạn 1 lớp.

- Bộ nhớ: 32 đầu vào lớp (nhỏ) so với 32 * (kích hoạt volume trên mỗi lớp) (lớn).
- Tính toán bổ sung: 1 chuyển tiếp bổ sung cho mỗi phân đoạn, tức là chuyển tiếp thêm ~33% FLOPs tổng số (vì lùi là 2 lần về phía trước, bước đầy đủ trở thành 1 + 1 + 2 = 4 đơn vị thay vì 1 + 2 = 3).

Đây là công thức ban đầu của Chen et al. 2016: một checkpoint mỗi `sqrt(L)` lớp để cân bằng bộ nhớ và tính toán. Đối với L = 64, đó là 8 checkpoints.

### Điểm kiểm tra có chọn lọc (Korthikanti 2022)

Không phải tất cả các kích hoạt đều có chi phí như nhau. Đầu ra attention softmax là `B*L*L*heads` và tăng * bậc hai * với độ dài trình tự. Kích hoạt ẩn FFN là `B*L*4d` và phát triển tuyến tính. Đối với các chuỗi dài, softmax chiếm ưu thế.

Điểm kiểm tra chọn lọc giữ các kích hoạt rẻ để lưu trữ (phép chiếu tuyến tính, số dư) và chỉ tính toán lại những kích hoạt đắt tiền (attention). Bạn trả FLOPs tối thiểu để tính toán lại nhưng tiết kiệm bộ nhớ O(L^2).

Megatron-Core thực hiện điều này dưới dạng tính toán lại kích hoạt "có chọn lọc". Được sử dụng trong hầu hết các lần chạy training biên giới 2024+.

### Giảm tải

Thay thế cho tính toán lại: ship kích hoạt để CPU RAM giữa tiến và lùi. Yêu cầu băng thông PCIe; có lợi khi băng thông nhàn rỗi vượt quá chi phí tái vật chất. Các chiến lược hỗn hợp là phổ biến: checkpoint một số lớp, giảm tải cho những lớp khác.

FSDP2 ships giảm tải như một tùy chọn class đầu. Giảm tải tỏa sáng khi GPU bị tắc nghẽn bộ nhớ nhưng truyền CPU-GPU có khoảng trống.

### Tính toán lại chi phí Model

FLOPs mỗi bước với điểm kiểm tra ngây thơ từng `k` lớp ra khỏi `L`:

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # one extra forward per layer in the segment
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

Với checkpoint chọn lọc, bạn chỉ tính toán lại hạt nhân attention, không phải toàn bộ lớp:

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### Tiết kiệm bộ nhớ Model

volume kích hoạt trên mỗi lớp: `A`. Đối với `L` lớp, tổng bộ nhớ kích hoạt: `L * A`.

checkpoint đầy đủ (kích thước phân đoạn 1): chỉ lưu trữ `L * input_volume` (~`L * 1/10 A` cho transformer tiêu chuẩn). Lưu ~`9 * L * A * 1/10`.

Checkpoint mỗi `k` lớp: lưu trữ `L/k * A` cộng với giá trị của `k-1` lớp trong phân đoạn đang hoạt động.

Ở `k = sqrt(L)`, chi phí bộ nhớ và điện toán lại đều mở rộng quy mô với `sqrt(L)` - sự cân bằng tối ưu cho các lớp chi phí đồng nhất.

### Khi nào không nên Checkpoint

- Các lớp trong cùng của giai đoạn pipeline đã được bay. Dù sao thì họ cũng phải kết thúc.
- Các lớp đầu tiên và cuối cùng nếu chúng thống trị tính toán của giai đoạn (hiếm gặp ở transformers).
- Attention nhân đã sử dụng FlashAttention - Flash đã tính toán lại softmax nhanh chóng, vì vậy điểm kiểm tra cấp lớp bổ sung bổ sung thêm rất ít.

### Các mẫu triển khai

1. **Trình bao bọc hàm: **bao bọc một phân đoạn trong `torch.utils.checkpoint.checkpoint(fn, input)`. PyTorch chỉ lưu trữ `input`, tính toán lại mọi thứ khác ở chế độ ngược.

2. **Dựa trên trình trang trí:** gắn nhãn các lớp là có thể kiểm tra; huấn luyện viên quyết định tại config thời điểm phân đoạn nào được bọc.

3. **Tính toán lại rõ ràng thủ công:** tự viết backward pass, gọi một `recompute_forward` tùy chỉnh sao chép chuyển tiếp với đầu vào được lưu trữ.

Cả ba đều cho cùng một kết quả chức năng. Wrappers là thành ngữ tiêu chuẩn.

### Tương tác với TP / PP / FP8

- **Tensor song song:** checkpoint đầu vào phải được thu thập hoặc phân tán lại trên máy tính lại; xử lý chi phí giao tiếp.
- **Pipeline song song: **Mô hình điển hình là checkpoint chuyển tiếp của từng giai đoạn pipeline để các vi lô thứ tự ngược có thể sử dụng lại bộ nhớ kích hoạt.
- **Tính toán lại FP8: **lịch sử amax được cập nhật trong quá trình tính toán lại phải khớp với lịch sử chuyển tiếp ban đầu hoặc tỷ lệ FP8 trôi dạt. Hầu hết frameworks chụp nhanh thang đo.

## Tự xây dựng

### Bước 1: Một Model đồ chơi với các phân đoạn

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### Bước 2: Ngây thơ lùi cần tất cả các kích hoạt

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### Bước 3: Bộ nhớ Checkpoint-Every-k

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### Bước 4: Chi phí Model

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### Bước 5: Công cụ ước tính bộ nhớ

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### Bước 6: Kích thước phân đoạn tối ưu

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### Bước 7: Quyết định Checkpoint chọn lọc

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## Ứng dụng

- **torch.utils.checkpoint**: `from torch.utils.checkpoint import checkpoint` — trình bao bọc chuẩn trong PyTorch. Bao bọc một hàm; chỉ lưu trữ đầu vào, tính toán lại khi ngược.
- **Tính toán lại kích hoạt Megatron-Core**: hỗ trợ các chế độ `selective`, `full` và `block`. Tiêu chuẩn trong training biên giới 2024+.
- **Giảm tải FSDP2**: `module.to_empty(device="cpu")` với `offload_policy` trong FSDP2 kích hoạt phân đoạn để CPU thay vì tính toán lại.
- **DeepSpeed ZeRO-Offload**: Giảm tải CPU cho các trạng thái và kích hoạt optimizer, bổ sung cho điểm kiểm tra.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/prompt-activation-recompute-policy.md` - một prompt lấy model config của bạn (lớp, ẩn, sêu, batch) và bộ nhớ có sẵn GPU và phát ra policy tính toán lại trên mỗi lớp (không có / chọn lọc / đầy đủ / giảm tải).

## Bài tập

1. Xác minh tính chính xác. Chạy `model_forward` + `model_backward` (kích hoạt đầy đủ) so với `model_forward_checkpointed` + `model_backward_checkpointed` (phân đoạn). Parameter gradients phải giống với precision máy.

2. Kích thước đoạn quét `k` từ 1 đến `L`. Vẽ FLOP trên đầu và bộ nhớ. Tìm đầu gối của đường cong.

3. Thực hiện điểm kiểm tra chọn lọc: lưu trữ đầu vào mô-đun attention nhưng không lưu trữ trung gian của nó. Đo lường chi phí FLOP so với điểm kiểm tra toàn lớp cho model 32 lớp tại seq = 8192.

4. Thêm giảm tải. Lưu đầu vào phân đoạn vào "bộ đệm CPU" mô phỏng (một danh sách riêng biệt). Đo "băng thông PCIe" dưới dạng bytes/time và tìm điểm hòa vốn giữa giảm tải và tính toán lại.

5. Benchmark một PyTorch transformer thực sự có và không có `torch.utils.checkpoint`. Đo bộ nhớ (thông qua `torch.cuda.max_memory_allocated`) và thời gian bước.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|----------------------|
| Gradient trạm kiểm soát | "Tiết kiệm bộ nhớ bằng cách làm lại về phía trước" | Chỉ lưu trữ đầu vào phân đoạn; tính toán lại các trung gian trong quá trình lùi để nhận tensors hỗ trợ gradient |
| Tính toán lại kích hoạt | "Giống như kiểm tra" | Tên hương vị HPC cho cùng một kỹ thuật |
| Kích thước phân đoạn (k) | "Có bao nhiêu lớp mỗi checkpoint" | Số lớp có chất trung gian bị loại bỏ và tái vật chất hóa cùng nhau |
| Điểm kiểm tra chọn lọc | "Thủ thuật của Korthikanti" | Chỉ tính toán lại các kích hoạt tốn kém để lưu trữ (attention softmax); giữ những cái rẻ tiền |
| Kiểm tra đầy đủ | "Phiên bản ngây thơ" | Tính toán lại các trung gian của mọi lớp trong mọi phân đoạn |
| Chặn điểm kiểm tra | "Hạt thô" | Checkpoint toàn bộ khối transformer; độ chi tiết lớn nhất |
| Chi phí FLOP | "Thuế tính toán" | Thêm FLOPs mỗi bước = (tính toán lại FLOPs) / (fwd + bwd FLOPs); 33% ngây thơ, 5% chọn lọc |
| Giảm tải kích hoạt | "Ship CPU" | Di chuyển các kích hoạt đến CPU RAM trên tiến->lùi; thay thế cho tính toán lại |
| Quy tắc sqrt-L | "Tối ưu cổ điển" | Đối với các lớp có chi phí đồng đều, khoảng cách checkpoint tối ưu là các lớp sqrt (L) |
| Attention-softmax volume | "Vấn đề O (L ^ 2)" | L^2 * đầu * batch nổi; thống trị bộ nhớ kích hoạt ở ngữ cảnh dài |

## Đọc thêm

- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) -- bài báo gốc chính thức hóa gradient trạm kiểm soát
- [Korthikanti et al., 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) - tính toán lại kích hoạt có chọn lọc và phân tích chi phí chính thức
- [Pudipeddi et al., 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645) - cách tiếp cận bộ nhớ không đổi thay thế thông qua tái vật chất hóa chế độ đảo ngược
- [Ren et al., 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840) - giảm tải kích hoạt trên quy mô lớn
- [PyTorch torch.utils.checkpoint docs](https://pytorch.org/docs/stable/checkpoint.html) - API tiêu chuẩn
- [Megatron-Core activation recomputation documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html) - chế độ chọn lọc, đầy đủ và khối
