# Hỗn hợp các chuyên gia (MoE)

> Một transformer 70B dày đặc kích hoạt mọi parameter cho mỗi token. Một MoE 671B chỉ kích hoạt 37B mỗi token và đánh bại nó trên mỗi benchmark. Sự thưa thớt là ý tưởng mở rộng quan trọng nhất của thập kỷ.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 7 · 05 (Transformer đầy đủ), Giai đoạn 7 · 07 (GPT)
**Thời lượng:** ~45 phút

## Vấn đề

FLOPs của một transformer dày đặc tại inference bằng số parameter của nó (nhân với 2 cho forward pass). Mở rộng quy mô model mật độ và mọi token thanh toán toàn bộ hóa đơn. Đến năm 2024, biên giới đã chạm vào bức tường điện toán: để trở nên thông minh hơn một cách có ý nghĩa, bạn cần nhiều FLOPs hơn theo cấp số nhân mỗi token.

Hỗn hợp các chuyên gia phá vỡ liên kết này. Thay thế mỗi FFN bằng `E` chuyên gia độc lập + một bộ định tuyến chọn `k` chuyên gia mỗi token. Tổng parameters = `E × FFN_size`. parameters hoạt động trên mỗi token = `k × FFN_size`. Điển hình configuration 2026: `E=256`, `k=8`. Quy mô lưu trữ với `E`, quy mô điện toán với `k`.

Biên giới năm 2026 gần như hoàn toàn MoE: DeepSeek-V3 (tổng cộng 671 tỷ / 37 tỷ đang hoạt động), Mixtral 8×22B, Qwen2.5-MoE, Llama 4, Kimi K2 gpt-oss. Trên bảng xếp hạng độc lập của Phân tích nhân tạo, 10 models mã nguồn mở hàng đầu đều MoE.

## Khái niệm

![MoE layer: router selects k of E experts per token](../assets/moe.svg)

### Hoán đổi FFN

Khối transformer dày đặc:

```
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

MoE khối:

```
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # pick k of E per token
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

Mỗi chuyên gia là một FFN độc lập (thường là SwiGLU). Bộ định tuyến là một lớp tuyến tính duy nhất. Mỗi token chọn các chuyên gia `k` của riêng mình và nhận được một hỗn hợp kết quả đầu ra của họ.

### Bài toán cân bằng tải

Nếu bộ định tuyến đưa 90% tokens thông qua expert 3, các chuyên gia khác sẽ chết đói. Ba bản sửa lỗi đã được thử:

1. **loss cân bằng tải phụ **(Switch Transformer, Mixtral). Thêm một hình phạt tỷ lệ thuận với variance trong cách sử dụng của chuyên gia. Hoạt động, nhưng thêm tín hiệu hyperparameter và gradient thứ hai.
2. **Dung lượng chuyên gia + token giọt **(Chuyển đổi sớm). Mỗi chuyên gia processes nhiều nhất `C × N/E` tokens; tràn tokens bỏ qua lớp. Làm tổn thương chất lượng.
3. **Cân bằng không loss phụ trợ **(DeepSeek-V3). Thêm một bias chuyên gia đã học để thay đổi lựa chọn top-k của bộ định tuyến. Bias được cập nhật bên ngoài training loss. Không có hình phạt cho mục tiêu chính. Mở khóa lớn của năm 2024.

Cách tiếp cận của DeepSeek-V3: sau mỗi bước training, đối với mọi chuyên gia, hãy kiểm tra xem việc sử dụng nó cao hơn hay thấp hơn mục tiêu. Đẩy bias bằng `±γ`. Lựa chọn sử dụng `scores + bias`. Xác suất chuyên gia được sử dụng để kiểm soát là `scores` thô không thay đổi. Tách định tuyến khỏi biểu cảm.

### Chuyên gia chia sẻ

DeepSeek-V2/V3 cũng chia các chuyên gia thành *chia sẻ* và *định tuyến*. Mỗi token đều đi qua tất cả các chuyên gia được chia sẻ. Các chuyên gia được định tuyến được chọn qua top-k. Các chuyên gia được chia sẻ nắm bắt kiến thức chung; Chuyên gia định tuyến chuyên môn. V3 chạy 1 chuyên gia được chia sẻ cộng với top 8 trong số 256 định tuyến.

### Chuyên gia chi tiết

MoE cổ điển (GShard, Switch): mỗi chuyên gia rộng như một FFN đầy đủ. `E` nhỏ (8–64), `k` nhỏ (1–2).

MoE hạt mịn hiện đại (DeepSeek-V3, Qwen-MoE): mỗi chuyên gia hẹp hơn (kích thước FFN 1/8). `E` lớn (256+), `k` lớn hơn (8+). Tổng parameters giống nhau, nhưng kết hợp mở rộng nhanh hơn nhiều. `C(256, 8) = 400 trillion` "chuyên gia" có thể có mỗi token. Chất lượng tăng lên, độ trễ không đổi.

### Hồ sơ chi phí

Mỗi token, mỗi lớp:

| Config | Tham số / token hoạt động | Tổng số tham số |
|--------|-----------------------|--------------|
| Mixtral 8×22B | ~39 tỷ | 141 tỷ |
| Llama 3 70B (dày đặc) | 70 tỷ | 70 tỷ |
| Tìm kiếm sâu-V3 | 37 tỷ | 671 tỷ |
| Kimi K2 (MoE) | ~32 tỷ | 1 tấn |

DeepSeek-V3 đánh bại Llama 3 70B (dày đặc) trên hầu hết các benchmark trong khi thực hiện **ít FLOPs hoạt động hơn mỗi token**. Nhiều parameters hơn = nhiều kiến thức hơn. Nhiều FLOPs hoạt động hơn = nhiều điện toán hơn trên mỗi token. MoE tách rời chúng.

### Điểm mấu chốt: ký ức

Tất cả các chuyên gia đều sống trên GPU bất kể cái nào cháy. Một model 671B cần ~1,3 TB VRAM cho trọng lượng fp16. Việc triển khai Frontier MoE đòi hỏi sự song song của chuyên gia - các chuyên gia phân đoạn trên GPUs, định tuyến tokens trên mạng. Độ trễ bị chi phối bởi giao tiếp tất cả mọi người, không phải matmul.

## Tự xây dựng

Xem `code/main.py`. Một lớp MoE nhỏ gọn trong stdlib thuần túy với:

- Các chuyên gia `n_experts=8` SwiGLU (mỗi chuyên gia một tuyến tính, để minh họa)
- top-k = 2 định tuyến
- Trọng lượng cổng chuẩn hóa softmax
- Cân bằng phụ trợ không loss thông qua bias từng chuyên gia

### Bước 1: bộ định tuyến

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # softmax over ORIGINAL scores of the chosen experts
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

Bias ảnh hưởng đến việc lựa chọn, không ảnh hưởng đến trọng lượng cổng. Đó là thủ thuật DeepSeek-V3 - bias điều chỉnh sự mất cân bằng tải mà không cần điều khiển dự đoán của model.

### Bước 2: chạy 100 tokens qua bộ định tuyến

Theo dõi chuyên gia nào sa thải tần suất. Nếu không có bias, việc sử dụng bị lệch. Với vòng lặp cập nhật bias (`-γ` dành cho các chuyên gia sử dụng quá mức `+γ` dành cho các chuyên gia không được sử dụng nhiều), việc sử dụng hội tụ thành một phân phối đồng nhất qua một vài lần lặp.

### Bước 3: so sánh số lượng tham số

In "tương đương dày đặc" của một MoE config. Hình dạng DeepSeek-V3: 256 định tuyến + 1 chia sẻ, 8 đang hoạt động, d_model = 7168. Tổng số parameter thật đáng kinh ngạc. Số lượng hoạt động là một phần bảy của Llama dày đặc 3 70B.

## Ứng dụng

ÔmFace tải:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

production inference 2026: vLLM hỗ trợ định tuyến MoE nguyên bản. SGLang có con đường song song chuyên gia nhanh nhất. Cả hai đều tự động xử lý lựa chọn top-k và song song chuyên nghiệp.

**Khi nào nên chọn MoE:**
- Bạn muốn chất lượng biên giới với chi phí inference mỗi token thấp hơn.
- Bạn có cơ sở hạ tầng song song VRAM / chuyên gia.
- Khối lượng công việc của bạn nặng về token (trò chuyện, mã) chứ không phải nặng ngữ cảnh (tài liệu dài).

**Khi nào KHÔNG nên chọn MoE:**
- Triển khai biên — bạn trả toàn bộ dung lượng lưu trữ cho bất kỳ FLOP đang hoạt động nào.
- Phân phối một người dùng quan trọng về độ trễ — định tuyến chuyên gia làm tăng chi phí.
- models nhỏ (<7B) — Lợi thế chất lượng của MoE chỉ xuất hiện trên ngưỡng điện toán (~6B tham số hoạt động).

## Sản phẩm bàn giao

Xem `outputs/skill-moe-configurator.md`. skill chọn bố cục E, k và chuyên gia chia sẻ cho một MoE mới với ngân sách, training tokens và mục tiêu triển khai parameter.

## Bài tập

1. **Dễ dàng.** Chạy `code/main.py`. Xem cách bản cập nhật bias không có loss phụ trợ giúp chuyên gia sử dụng hơn 50 lần lặp lại.
2. **Trung bình.** Thay thế bộ định tuyến đã học bằng bộ định tuyến dựa trên băm (xác định, không học). So sánh chất lượng và sự cân bằng. Tại sao bộ định tuyến đã học lại tốt hơn?
3. **Khó khăn.** Triển khai "định tuyến phù hợp rollout" kiểu GRPO (thủ thuật DeepSeek-V3.2): ghi nhật ký mà các chuyên gia kích hoạt trong quá trình inference, buộc định tuyến tương tự trong quá trình tính toán gradient. Đo lường ảnh hưởng đến thiết lập gradient policy đồ chơi.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Chuyên gia | "Một FFN trong số rất nhiều" | Một mạng chuyển tiếp nguồn cấp dữ liệu độc lập; parameters dành riêng cho một lát cắt thưa thớt của tính toán FFN. |
| Bộ định tuyến | "Cánh cổng" | Một lớp tuyến tính nhỏ chấm điểm mỗi token so với từng chuyên gia; top-k lựa chọn. |
| Định tuyến Top-k | "K chuyên gia hoạt động trên mỗi token" | Tính toán FFN của mỗi token thông qua chính xác k chuyên gia, có trọng số theo cổng. |
| loss phụ trợ | "Hình phạt cân bằng tải" | Thêm loss thuật ngữ phạt việc sử dụng sai lệch của chuyên gia. |
| Phụ trợ không loss | "Thủ thuật của DeepSeek-V3" | Cân bằng thông qua bias của từng chuyên gia chỉ dựa trên lựa chọn của bộ định tuyến; không có thêm gradient. |
| Chuyên gia chia sẻ | "Luôn bật" | Chuyên gia bổ sung mà mọi token đều đi qua; nắm bắt được kiến thức chung. |
| Tính song song của chuyên gia | "Mảnh vỡ của chuyên gia" | Phân phối các chuyên gia khác nhau cho các GPUs khác nhau; định tuyến tokens trên mạng. |
| Thưa thớt | "Tham số hoạt động < tổng tham số" | Tỷ lệ `k × expert_size / (E × expert_size)`; 37/671 ≈ 5,5% cho DeepSeek-V3. |

## Đọc thêm

- [Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) - ý tưởng.
- [Fedus, Zoph, Shazeer (2022). Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) - Switch, MoE cổ điển.
- [Jiang et al. (2024). Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Mixtral 8×7B.
- [DeepSeek-AI (2024). DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) - MLA + MoE không có loss phụ trợ + MTP.
- [Wang et al. (2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) - giấy cân bằng dựa trên bias.
- [Dai et al. (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) — phân chia chi tiết + chia sẻ chuyên gia mà bộ định tuyến của bài học này sử dụng.
- [Kim et al. (2022). DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training](https://arxiv.org/abs/2201.05596) — bài báo chuyên gia được chia sẻ ban đầu.
