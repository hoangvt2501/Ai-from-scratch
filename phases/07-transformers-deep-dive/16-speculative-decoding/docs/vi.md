# Giải mã suy đoán - Soạn thảo, Xác minh, Lặp lại

> Giải mã tự hồi quy là nối tiếp. Mỗi token chờ đợi cái trước. Giải mã đầu cơ phá vỡ chuỗi: một model rẻ tiền nháp N tokens, model đắt tiền xác minh tất cả N trong một forward pass. Khi bản nháp đúng, bạn đã trả một tiền lớn cho N thế hệ.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 7 · 07 (GPT LM nhân quả), Giai đoạn 7 · 12 (Attention KV Cache & Flash)
**Thời lượng:** ~60 phút

## Vấn đề

Một LLM sampling 70B token mất ~30 ms trên H100. Một model nháp 3B mất ~3 ms. Nếu chúng ta để bản nháp 3B 5 tokens trước, sau đó chạy 70B * một lần * để xác minh tất cả 5, tổng số là `5×3 + 30 = 45 ms` cho tối đa 5 tokens được chấp nhận - so với `5×30 = 150 ms` cho việc tạo đường thẳng. Đó là chiêu hàng giải mã đầu cơ đầy đủ: đổi một lượng nhỏ bộ nhớ GPU bổ sung (model dự thảo) để có độ trễ giải mã thấp hơn 2–4×.

Thủ thuật phải bảo toàn sự phân phối. sampling đầu cơ, được giới thiệu bởi Leviathan et al. (2023) và Chen et al. đồng thời, đảm bảo rằng trình tự đầu ra được phân phối **giống hệt nhau **với những gì các model lớn sẽ tự tạo ra. Không có sự đánh đổi về chất lượng. Chỉ cần nhanh hơn.

Bốn gia đình gồm các cặp người xác minh dự thảo thống trị inference năm 2026:

1. **Đầu cơ vani (Leviathan 2023).** model dự thảo riêng biệt (ví dụ: Llama 3 1B) + người xác minh (ví dụ: Llama 3 70B).
2. **Medusa (Cai 2024).** Nhiều đầu giải mã trên trình xác minh dự đoán các vị trí `t+1..t+k` song song. Không có model dự thảo riêng biệt.
3. **Gia đình EAGLE (Li 2024, 2025).** Bản nháp nhẹ sử dụng lại các trạng thái ẩn của trình xác minh; tỷ lệ chấp nhận gần hơn vani; 3–4× điển hình.
4. **Giải mã Lookahead (Fu 2024).** Lặp lại Jacobi; không cần model bản nháp. Tự suy đoán. Thích hợp nhưng không phụ thuộc.

Mỗi production inference stack vào năm 2026 ships giải mã đầu cơ theo mặc định. vLLM, TensorRT-LLM, SGLang và llama.cpp đều hỗ trợ ít nhất vanilla + EAGLE-2.

## Khái niệm

### Thuật toán cốt lõi

Với `M_q` xác minh và `M_p` dự thảo rẻ hơn:

1. Hãy `x_1..x_k` là tiền tố đã được giải mã.
2. **Bản nháp**: sử dụng `M_p` để tự động đề xuất `d_{k+1}, d_{k+2}, ..., d_{k+N}` với xác suất dự thảo `p_1..p_N`.
3. **Xác minh song song**: chạy `M_q` một lần trên `x_1..x_k, d_{k+1}, ..., d_{k+N}`, nhận xác suất xác minh `q_1..q_{N+1}` cho các vị trí `k+1..k+N+1`.
4. **Accept/reject mỗi bản nháp token từ trái sang phải**: đối với mỗi `i`, chấp nhận với xác suất `min(1, q_i(d_i) / p_i(d_i))`.
5. Khi loại bỏ đầu tiên ở vị trí `j`: `t_j` mẫu từ phân phối "dư" `(q_j - p_j)_+` chuẩn hóa. Tất cả các bản nháp sau `j` đều bị loại bỏ.
6. Khi chấp nhận tất cả các `N`: lấy mẫu thêm một token `t_{N+1}` từ `q_{N+1}` (tiền thưởng miễn phí token).

Thủ thuật phân phối dư là cái nhìn sâu sắc về toán học giữ cho đầu ra được phân phối chính xác như thể `M_q` đã lấy mẫu từ đầu.

### Điều gì quyết định tăng tốc

Giả sử `α` = tỷ lệ chấp nhận dự kiến cho mỗi token nháp. Giả sử `c` = tỷ lệ chi phí nháp trên người xác minh. Mỗi bước:

- Thế hệ ngây thơ thực hiện 1 cuộc gọi model lớn mỗi token.
- Đầu cơ thực hiện 1 cuộc gọi model lớn mỗi `(1 - α^{N+1}) / (1 - α) ≈ 1/(1-α)` tokens khi `α` cao.

Nguyên tắc ngón tay cái điển hình ở `α = 0.75` và `N = 5`: 3× ít cuộc gọi model lớn hơn. Chi phí dự thảo rẻ 5×. Tổng số đồng hồ treo tường giảm ~2,5×.

**α phụ thuộc vào:**

- Bản nháp xấp xỉ người xác minh tốt như thế nào. Dữ liệu cùng một gia đình / cùng training giúp tăng đáng kể α.
- Giải mã chiến lược. Dự thảo tham lam chống lại người xác minh tham lam: α cao Temperature sampling: khó khớp hơn; sự chấp nhận giảm.
- Loại nhiệm vụ. Mã và đầu ra có cấu trúc chấp nhận nhiều hơn (có thể dự đoán được); viết sáng tạo dạng tự do chấp nhận ít hơn.

### Medusa — bản nháp không có bản nháp model

Medusa thay thế model nháp bằng các đầu đầu ra bổ sung trên trình xác minh. Tại vị trí `t`:

```
shared trunk → hidden h_t
    ├── head_0: predict token at t+1  (standard LM head)
    ├── head_1: predict token at t+2
    ├── head_2: predict token at t+3
    ├── head_3: predict token at t+4
```

Mỗi đầu xuất ra logits riêng. Tại inference, bạn lấy mẫu từ mỗi đầu để có được một trình tự ứng viên, sau đó xác minh bằng một forward pass bằng cách sử dụng sơ đồ attention cây xem xét tất cả các phần tiếp theo của ứng cử viên cùng một lúc.

Ưu điểm: không có model thứ hai. Nhược điểm: thêm parameters có thể huấn luyện; cần một giai đoạn fine-tuning có giám sát (~1B tokens); Tỷ lệ chấp nhận thấp hơn một chút so với đầu cơ vani với bản nháp tốt.

### EAGLE — bản nháp tốt hơn bằng cách sử dụng lại các trạng thái ẩn

EAGLE-1/2/3 (Li và cộng sự, 2024–2025) làm cho bản nháp model một transformer nhỏ (thường là 1 lớp) thu nạp các trạng thái ẩn lớp cuối cùng của trình xác minh. Bởi vì dự thảo nhìn thấy đại diện feature của người xác minh, các dự đoán của nó tương quan chặt chẽ với phân phối đầu ra của người xác minh. Tỷ lệ chấp nhận tăng từ ~0,6 (vani) lên 0,85+.

EAGLE-3 (2025) đã thêm tìm kiếm cây trên các ứng cử viên tiếp tục. vLLM và SGLang ship EAGLE-2/3 làm lộ trình thông số kỹ thuật mặc định cho Llama 3/4 và Qwen 3.

### Điệu nhảy KV cache

Nguồn cấp dữ liệu xác minh `N` tokens nháp vào trình xác minh trong một forward pass. Điều này mở rộng KV cache của người xác minh bằng `N` mục nhập. Nếu một số bản nháp bị từ chối, bạn phải quay bộ nhớ đệm trở lại độ dài tiền tố được chấp nhận.

Production triển khai (`--speculative-model` của vLLM, LookaheadDecoder của TensorRT-LLM) xử lý điều này với bộ đệm KV cào. Viết trước, commit về sự chấp nhận. Nó không khó về mặt khái niệm, nhưng nó rất khó khăn.

## Tự xây dựng

Xem `code/main.py`. Chúng ta triển khai thuật toán sampling suy đoán cốt lõi (bước loại bỏ + phân phối dư) với:

- Một "model lớn" là một softmax xác định trên phân phối được mã hóa bằng tay (vì vậy chúng ta có thể xác minh toán học chấp nhận một cách phân tích).
- Một "model nháp" là một sự xáo trộn của model lớn.
- Một vòng chấp nhận / từ chối tạo ra phân phối cận biên giống như sampling trực tiếp.

### Bước 1: bước từ chối

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u` là một số ngẫu nhiên đồng nhất. `q_prob` là xác suất của người xác minh đối với token được soạn thảo. `p_prob` là xác suất của model dự thảo. Định lý Leviathan là quyết định Bernoulli này, tiếp theo là sampling từ số dư khi bị từ chối, bảo toàn chính xác sự phân phối của người xác minh.

### Bước 2: phân phối dư

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

Trừ `p` từ `q` phần tử khôn ngoan, kẹp các giá trị âm về không, chuẩn hóa lại. Mẫu từ điều này trên bất kỳ sự từ chối nào.

### Bước 3: một bước đầu cơ

```python
def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix
```

Năm người chấp nhận → một phần thưởng → sáu tokens được tạo ra trong một lần vượt qua người xác minh.

### Bước 4: đo lường tỷ lệ chấp nhận

Chạy 10.000 bước đầu cơ ở các mức chất lượng nháp khác nhau. Tỷ lệ chấp nhận lô so với sự phân kỳ KL giữa phân phối bản nháp và người xác minh. Bạn sẽ thấy một mối quan hệ đơn điệu rõ ràng.

### Bước 5: xác minh tương đương phân phối

Về mặt kinh nghiệm: biểu đồ của tokens được tạo ra bởi vòng lặp suy đoán phải khớp với biểu đồ do sampling tạo ra trực tiếp từ người xác minh. Đây là định lý Leviathan trong thực tế. Thử nghiệm chi-square xác nhận trong sampling lỗi.

## Ứng dụng

Production:

```bash
# vLLM with EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model /models/llama-3.1-eagle-70b \
    --speculative-draft-tensor-parallel-size 1 \
    --num-speculative-tokens 5

# vLLM with vanilla draft model
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5
```

TensorRT-LLM có đường dẫn Medusa nhanh nhất tính đến giữa năm 2026. `faster-whisper` bao bọc giải mã đầu cơ cho Whisper-large với một bản nháp nhỏ.

**Chọn một bản nháp:**

| Chiến lược | Khi nào nên chọn | Tăng tốc |
|----------|--------------|---------|
| Bản nháp vani (họ 1B/3B Llama) | Nguyên mẫu nhanh, không training | 1.8–2.3× |
| Đầu Medusa | Bạn có thể fine-tune trình xác minh | 2–3× |
| ĐẠI BÀNG-2 / 3 | Production, tốc độ tối đa | 3–4× |
| Lookahead | Không có bản nháp, không training, không có tham số bổ sung | 1,3–1,6× |

**Khi nào KHÔNG giải mã thông số kỹ thuật:**

- Thế hệ trình tự đơn từ 1–5 tokens. Trên cao chiếm ưu thế.
- Sáng tạo hoang dã / temperature sampling cao (giọt α).
- Triển khai hạn chế bộ nhớ (bản nháp model thêm VRAM).

## Sản phẩm bàn giao

Xem `outputs/skill-spec-decode-picker.md`. skill chọn một chiến lược giải mã đầu cơ (vani / Medusa / EAGLE / lookahead) và điều chỉnh parameters (N, bản nháp temperature) cho khối lượng công việc inference mới.

## Bài tập

1. **Dễ dàng.** Chạy `code/main.py`. Xác nhận phân phối token suy đoán khớp với phân phối mẫu trực tiếp của người xác minh trên 50.000 tokens trong chi bình phương p > 0,05.
2. **Trung bình.** Tăng tốc cốt truyện (tokens trên mỗi model lớn về phía trước) như một hàm của `N` cho `α = 0.5, 0.7, 0.85`. Xác định `N` tối ưu cho từng α. (Gợi ý: tokens dự kiến cho mỗi cuộc gọi xác minh = `(1 - α^{N+1}) / (1 - α)`.)
3. **Khó.** Thực hiện một Medusa nhỏ: lấy GPT capstone từ Bài 14, thêm 3 đầu LM dự đoán vị trí t+2, t+3, t+4. Huấn luyện trên tinyshakespeare với một loss nhiều đầu chung. So sánh tỷ lệ chấp nhận so với bản nháp vani được thực hiện bằng cách cắt bớt cùng một model.
4. **Khó.** Thực hiện rollback: bắt đầu với tiền tố 10 token KV cache, nạp 5 tokens nháp, mô phỏng từ chối ở vị trí 3. Xác minh rằng bộ nhớ đệm của bạn đọc chính xác khớp với "tiền tố + 2 bản nháp được chấp nhận đầu tiên" ở lần lặp tiếp theo.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Dự thảo model | "Cái rẻ" | Một model nhỏ hơn đề xuất ứng cử viên tokens; thường rẻ hơn 10–50× so với người xác minh. |
| Người xác minh | "Cái lớn" | Mục tiêu model chúng tôi bảo tồn sự phân phối của ai; chạy một lần cho mỗi bước đầu cơ. |
| Tỷ lệ chấp nhận (α) | "Tần suất bản nháp đúng" | Xác suất mỗi token người xác minh chấp nhận bản nháp. 0,7–0,9 điển hình. |
| Phân phối dư | "Dự phòng từ chối" | `(q - p)_+` chuẩn hóa; sampling từ việc từ chối sẽ bảo toàn sự phân phối của người xác minh. |
| Tiền thưởng token | "Người tự do" | Khi tất cả N bản nháp được chấp nhận, hãy lấy mẫu thêm một bản nháp từ bản phân phối bước tiếp theo của trình xác minh. |
| Medusa | "Suy đoán không có bản nháp" | Nhiều đầu LM trên trình xác minh dự đoán song song các vị trí t + 1..t + k. |
| ĐẠI BÀNG | "Dự thảo trạng thái ẩn" | Bản nháp transformer nhỏ có điều kiện dựa trên các trạng thái ẩn lớp cuối cùng của trình xác minh. |
| Giải mã Lookahead | "Lặp lại Jacobi" | Tự suy đoán bằng cách sử dụng một lần lặp điểm cố định; không có model dự thảo. |
| Cây attention | "Xác minh nhiều ứng viên cùng một lúc" | Xác minh phân nhánh xem xét một số dự thảo tiếp tục đồng thời. |
| KV rollback | "Hoàn tác bản nháp bị từ chối" | Bộ đệm KV cào; commit khi chấp nhận, loại bỏ khi từ chối. |

## Đọc thêm

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) - thuật toán cốt lõi và định lý tương đương.
- [Chen et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — giới thiệu đồng thời; bằng chứng từ chối Bernoulli sạch.
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — Giấy Medusa; Xác minh attention cây.
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — ĐẠI BÀNG-1; dự thảo có điều kiện trạng thái ẩn.
- [Li et al. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) — ĐẠI BÀNG-2; độ sâu cây động.
- [Li et al. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) — ĐẠI BÀNG-3.
- [Fu et al. (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) - nhìn về phía trước, cách tiếp cận không dự thảo.
- [vLLM docs — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) — tài liệu tham khảo production kinh điển với cả bốn chiến lược được kết nối.
- [SafeAILab / EAGLE reference implementation](https://github.com/SafeAILab/EAGLE) — mã tham chiếu cho EAGLE-1/2/3.
