# Giải mã đầu cơ và EAGLE

> Một biên giới LLM tạo ra một token yêu cầu một forward pass đầy đủ trên hàng tỷ parameters. forward pass đó được cung cấp quá mức: hầu hết thời gian một model nhỏ hơn nhiều có thể đoán chính xác 3-5 tokens tiếp theo và các model lớn chỉ cần *xác minh* dự đoán. Khi đoán đúng, bạn nhận được 5 tokens cho giá của một. Giải mã đầu cơ (Leviathan et al. 2023) đã làm cho điều này chính xác và EAGLE-3 (2025) đã đẩy tỷ lệ chấp nhận lên ~4,5 tokens cho mỗi lần xác minh - tăng tốc gấp 4-5 lần ở phân phối đầu ra phù hợp.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (with numpy)
**Kiến thức tiên quyết:** Giai đoạn 10 Bài 12 (Tối ưu hóa Inference), Giai đoạn 10 Bài 04 (Pre-training Mini-GPT)
**Thời lượng:** ~75 phút

## Vấn đề

Giải mã thông lượng cho 70B-class model trên H100 thường là 40-80 tokens/second. Mỗi token yêu cầu đọc đầy đủ forward pass tất cả các trọng số model từ HBM. Bạn không thể làm cho model nhỏ hơn mà không thay đổi đầu ra của nó. Bạn không thể tăng kích thước batch ngoài bộ nhớ. Bạn đang gặp khó khăn - trừ khi bạn có thể để model xuất ra nhiều hơn một token mỗi forward pass.

Tạo tự hồi quy vốn dĩ trông nối tiếp: `x_{t+1} = sample(p(· | x_{1:t}))`. Nhưng có một cơ hội đồng thời. Nếu bạn có một công cụ dự đoán rẻ tiền nói rằng "4 tokens tiếp theo có thể là [a, b, c, d]", bạn có thể xác minh tất cả 5 vị trí trong một **forward pass duy nhất của model lớn** và chấp nhận tiền tố khớp dài nhất.

Leviathan, Kalai, Matias (2023, "Inference nhanh từ Transformers thông qua Giải mã đầu cơ") đã thực hiện điều này chính xác thông qua một quy tắc accept/reject thông minh để bảo toàn phân phối sampling của model mục tiêu. Phân phối đầu ra giống nhau, nhanh hơn 2-4×.

## Khái niệm

### Thiết lập hai Model

- **model mục tiêu **`M_p`: model lớn, chậm, chất lượng cao mà bạn thực sự muốn lấy mẫu. Phân phối: `p(x)`.
- **model nháp **`M_q`: một model nhỏ, nhanh, chất lượng thấp hơn. Phân phối: `q(x)`. Nhỏ hơn 5-30×.

Mỗi bước:

1. Dự thảo model đề xuất `K` tokens tự hồi quy: `x_1, x_2, ..., x_K ~ q`.
2. Target model chạy MỘT forward pass trên tất cả các vị trí `K+1` song song, tạo ra `p(x_k)` cho mỗi token được đề xuất.
3. Accept/reject từng token từ trái sang phải thông qua quy tắc sampling từ chối đã sửa đổi bên dưới. Chấp nhận tiền tố khớp dài nhất.
4. Nếu bất kỳ token nào bị từ chối, hãy lấy mẫu thay thế từ bản phân phối đã sửa và dừng lại. Nếu không, hãy lấy mẫu một token thưởng từ `p(· | x_1...x_K)`.

Nếu bản nháp khớp với mục tiêu một cách hoàn hảo, bạn sẽ nhận được K + 1 tokens cho mỗi mục tiêu tiền đạo. Nếu bản nháp sai ở vị trí 1, bạn chỉ nhận được 1 token.

### Quy tắc chính xác

Giải mã suy đoán **có thể chứng minh được trong phân phối với sampling từ p**. Quy tắc từ chối:

```
For each drafted token x_t:
    r ~ Uniform(0, 1)
    if r < p(x_t) / q(x_t):
        accept x_t
    else:
        sample replacement from residual: (p - q)+ / ||(p - q)+||_1
        stop
```

trong đó `(p - q)+` biểu thị phần dương của sự khác biệt theo điểm. Khi bản nháp và mục tiêu đồng ý (`p ≈ q`) chấp nhận là gần 1. Khi họ không đồng ý, phân phối dư được xây dựng sao cho mẫu tổng thể vẫn chính xác `p`.

**Trường hợp tham lam.** Đối với temperature = 0 sampling chỉ cần kiểm tra `argmax(p) == x_t`. Nếu có, chấp nhận; nếu không, xuất `argmax(p)` và dừng.

### Tăng tốc dự kiến

Nếu tỷ lệ chấp nhận cấp token của dự thảo model là `α` thì tokens dự kiến được tạo ra trên mỗi forward pass mục tiêu là:

```
E[tokens] = (1 - α^{K+1}) / (1 - α)        # K = draft length, α in [0, 1]
```

Ở `α = 0.8, K = 4`: `(1 - 0.8^5)/(1 - 0.8) = 3.36` tokens mỗi chuyển tiếp. Một mục tiêu chuyển tiếp có giá khoảng `cost_q * K + cost_p` (K bước nháp cộng với một xác minh mục tiêu). Nếu `cost_p >> cost_q * K` tỷ lệ tăng tốc `3.36× / 1 = 3.36×` về thông lượng.

Sự parameter thực sự duy nhất là `α`, điều này phụ thuộc hoàn toàn vào alignment mục tiêu dự thảo. Một bản nháp tốt là tất cả.

### Training bản nháp: Distillation

Một model nhỏ ngẫu nhiên tạo ra một bản nháp kém. Công thức tiêu chuẩn là chưng cất từ mục tiêu:

1. Chọn một kiến trúc nhỏ (~1B cho mục tiêu 70B, ~500M cho mục tiêu 7B).
2. Chạy model mục tiêu trên một kho dữ liệu văn bản lớn; lưu trữ các bản phân phối token tiếp theo của nó.
3. Huấn luyện bản nháp với sự phân kỳ KL so với sự phân phối của mục tiêu (không chống lại tokens sự thật cơ sở).

Kết quả: `α` thường là 0,6-0,8 trên mã hóa, 0,7-0,85 khi trò chuyện bằng ngôn ngữ tự nhiên. Tăng tốc 2-3× trong production.

### EAGLE: Soạn thảo cây + Tái sử dụng Feature

Li, Wei, Zhang, Zhang (2024, "EAGLE: Suy đoán Sampling yêu cầu suy nghĩ lại Feature sự không chắc chắn") đã quan sát thấy hai sự kém hiệu quả trong giải mã suy đoán tiêu chuẩn:

1. Bản nháp thực hiện K các bước nối tiếp, mỗi bước đầy đủ stack. Nhưng bản nháp có thể sử dụng lại features (trạng thái ẩn) của mục tiêu từ lần xác minh gần đây nhất - mục tiêu đã tính toán các đại diện phong phú mà bản dự thảo đang lấy lại từ đầu.
2. Bản nháp xuất ra một chuỗi tuyến tính. Nếu bản nháp có thể xuất ra một *cây* gồm các ứng cử viên (mỗi nút có nhiều lần đoán), forward pass duy nhất của mục tiêu có thể xác minh song song nhiều đường dẫn ứng cử viên thông qua mặt nạ attention cây và chọn branch được chấp nhận dài nhất.

Thay đổi EAGLE-1:
- Đầu vào dự thảo = trạng thái ẩn cuối cùng của mục tiêu tại vị trí t, không phải tokens thô.
- Kiến trúc nháp = 1 lớp transformer decoder (không phải một model nhỏ riêng biệt).
- Đầu ra = cây K = 4-8 ứng cử viên trên mỗi độ sâu, độ sâu 4-6.

EAGLE-2 (2024) bổ sung cấu trúc liên kết cây động: cây phát triển rộng hơn ở nơi gió lùa không chắc chắn và hẹp ở nơi nó tự tin. Tăng `α_effective` mà không làm tăng chi phí xác minh.

EAGLE-3 (Li et al. 2025, "EAGLE-3: Mở rộng quy mô Inference Tăng tốc của Models ngôn ngữ lớn thông qua Kiểm tra Training thời gian") loại bỏ sự phụ thuộc feature lớp trên cùng cố định và huấn luyện bản nháp với loss "mô phỏng thời gian kiểm tra" mới - bản nháp được huấn luyện trên các đầu ra phù hợp với phân phối thời gian kiểm tra của mục tiêu thay vì phân phối training bắt buộc của giáo viên. Tỷ lệ chấp nhận tăng từ 0,75 (EAGLE-2) lên 0,82 (EAGLE-3) và tokens/verify trung bình từ 3,0 lên 4,5.

### Xác minh Attention cây

Khi bản nháp xuất ra một cây, mục tiêu model xác minh nó trong một forward pass duy nhất bằng cách sử dụng **mặt nạ attention cây** - một mặt nạ nhân quả mã hóa cấu trúc liên kết cây chứ không phải một dòng thuần túy. Mỗi token chỉ quan tâm đến tổ tiên của nó trong cây. Vượt qua xác minh vẫn là một chuyển tiếp, một matmul; mặt nạ tô pô chỉ tốn thêm một vài mục KV.

```
        root
       /    \
      a      b
     / \    / \
    c  d   e   f
```

Nếu `a, b` là ứng cử viên token nhất cạnh tranh và `c, d, e, f` là ứng viên token thứ hai, tất cả sáu vị trí đều được xác minh trong một forward pass. Đầu ra là tiền tố dài nhất dọc theo bất kỳ con đường nào được chấp nhận.

### Khi nó thắng, khi nó không

**Chiến thắng:**
- Trò chuyện / hoàn thành với văn bản có thể dự đoán được (mã, tiếng Anh phổ biến, đầu ra có cấu trúc). `α` cao.
- Cài đặt có GPU điện toán không sử dụng trong quá trình giải mã (giai đoạn liên kết bộ nhớ). Soạn thảo cây sử dụng FLOPs có sẵn.

**Thua / không thắng:**
- Đầu ra ngẫu nhiên cao (viết sáng tạo ở temperature cao). `α` giảm về phía `1/|vocab|`.
- Batch phục vụ với tính đồng thời rất cao - việc phân phối đã lấp đầy FLOPs, rất ít chỗ để xác minh cây.
- Mục tiêu rất nhỏ models khi bản nháp không nhỏ hơn nhiều.

Các cửa hàng Production thường báo cáo tăng tốc 2-3× đồng hồ treo tường khi trò chuyện, 3-5× khi tạo mã và gần như bằng không khi viết sáng tạo.

```figure
speculative-decoding
```

## Tự xây dựng

`code/main.py`:

- Một `speculative_decode(target, draft, prompt, K, temperature)` tham chiếu thực hiện quy tắc từ chối chính xác và xác minh rằng nó bảo toàn phân phối của mục tiêu (KL thực nghiệm < 0,01 so với sampling mục tiêu thuần túy).
- Một máy phác thảo cây kiểu EAGLE xây dựng một cây sâu K với top-p phân nhánh.
- Trình tạo mặt nạ attention cây tạo ra mẫu nhân quả phù hợp cho người xác minh.
- Một harness tỷ lệ chấp nhận chạy cả trên một LM nhỏ (chưng cất một mục tiêu GPT-2-nhỏ từ mục tiêu GPT-2-trung bình).

```python
def speculative_step(p_target, q_draft, K, temperature=1.0):
    """One round of speculative decoding. Returns list of accepted tokens."""
    # 1. Draft K tokens
    draft_tokens = []
    q_probs = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_probs.append(probs[t])
        state = draft_step(state, t)

    # 2. Target computes p at every drafted position + 1 extra
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. Accept/reject left-to-right
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_probs[k]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_probs[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
            return accepted
    # 4. All K accepted → sample bonus token from target
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted
```

## Ứng dụng

- **vLLM** và **SGLang** ship giải mã suy đoán class đầu tiên. Cờ: `--speculative_model`, `--num_speculative_tokens`. EAGLE-2/3 hỗ trợ thông qua cờ `--spec_decoding_algorithm eagle`.
- **NVIDIA TensorRT-LLM** hỗ trợ cây Medusa và EAGLE nguyên bản.
- **Dự thảo tham khảo models**: `Qwen/Qwen3-0.6B-spec` (bản nháp cho Qwen3-32B), `meta-llama/Llama-3.2-1B-Instruct-spec` (bản nháp cho 70B).
- **Đầu Medusa** (Cai và cộng sự 2024, "Medusa: Framework tăng tốc LLM Inference đơn giản với nhiều đầu giải mã"): thay vì model nháp, hãy thêm đầu dự đoán song song K vào chính mục tiêu. Triển khai đơn giản hơn, chấp nhận thấp hơn một chút so với EAGLE.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-speculative-tuning.md` - một skill lập hồ sơ khối lượng công việc của model mục tiêu và chọn: model nháp, K (chiều dài bản nháp), chiều rộng cây, temperature và thời điểm quay trở lại giải mã đơn giản.

## Bài tập

1. Thực hiện quy tắc loại bỏ chính xác và xác minh theo kinh nghiệm. Chạy 10K samples qua `speculative_decode` và qua sampling mục tiêu đơn giản; tính toán khoảng cách TV giữa hai phân phối đầu ra. Nên < 0.01.

2. Tính công thức tăng tốc. Cho `α` và `K` cố định, biểu đồ dự kiến tokens cho mỗi mục tiêu chuyển tiếp. Tìm K tối ưu cho α ∈ {0,5, 0,7, 0,9}.

3. Huấn luyện một bản nháp nhỏ. Lấy mục tiêu GPT-2 124 triệu và chắt lọc bản nháp 30 triệu GPT-2 trên 100 triệu tokens với KL loss. Đo lường `α` trên văn bản bị giữ lại. Dự kiến: 0.6-0.7.

4. Triển khai phác thảo cây kiểu EAGLE. Thay vì chuỗi, hãy đặt branches đầu ra nháp 3 ở mỗi độ sâu. Xây dựng mặt nạ attention cây. Xác minh mục tiêu chấp nhận branch chính xác dài nhất.

5. Đo lường các chế độ lỗi. Chạy giải mã suy đoán ở temperature = 1,5 (ngẫu nhiên cao). Hiển thị α sụp đổ và thuật toán chậm hơn giải mã đơn giản do chi phí dự thảo.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| Mục tiêu model | "Người model lớn" | model chậm, chất lượng cao mà bạn muốn lấy mẫu (phân phối p) |
| Dự thảo model | "Nhà đầu cơ" | Yếu tố dự đoán nhỏ, nhanh (phân phối q); nhỏ hơn 5-30 lần |
| K / chiều dài dự thảo | "Nhìn về phía trước" | Số lượng tokens được suy đoán trên mỗi lần xác minh |
| α / tỷ lệ chấp nhận | "Tỷ lệ trúng đích" | Xác suất mỗi token đề xuất của dự thảo được chấp nhận |
| Quy tắc từ chối chính xác | "Bài kiểm tra chấp nhận" | r < p/q so sánh rằng bảo toàn phân phối của mục tiêu |
| Phân phối dư | "Đã sửa pq" | (p - q)+ / || (P - Q)+ || _1, phân phối cho mẫu từ khi từ chối |
| Soạn thảo cây | "Đầu cơ phân nhánh" | Bản nháp xuất ra một cây ứng viên, được xác minh trong một lần qua với mặt nạ attention có cấu trúc cây |
| Mặt nạ attention cây | "Mặt nạ tô pô" | Mặt nạ nhân quả mã hóa cấu trúc liên kết cây để mỗi nút chỉ quan tâm đến tổ tiên của nó |
| Đầu Medusa | "Đầu song song" | K dự đoán bổ sung trên chính mục tiêu; không có model dự thảo riêng biệt |
| EAGLE feature tái sử dụng | "Dự thảo trạng thái ẩn" | Đầu vào bản nháp là trạng thái ẩn cuối cùng của mục tiêu, không phải tokens thô, thu nhỏ bản nháp |
| loss mô phỏng thời gian thử nghiệm | "EAGLE-3 training" | Huấn luyện bản nháp về đầu ra phù hợp với phân phối thời gian kiểm tra của mục tiêu, không phải ép giáo viên |

## Đọc thêm

- [Leviathan, Kalai, Matias, 2023 — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) - quy tắc từ chối chính xác và phân tích tăng tốc lý thuyết
- [Chen, Borgeaud, Irving et al., 2023 — "Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318) — bài báo sampling suy đoán đồng thời tại DeepMind
- [Cai, Li, Geng, Wang, Wang, Zhu, Dao, 2024 — "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"](https://arxiv.org/abs/2401.10774) — đầu song song thay thế cho model dự thảo
- [Li, Wei, Zhang, Zhang, 2024 — "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"](https://arxiv.org/abs/2401.15077) — feature tái sử dụng và soạn thảo cây
- [Li et al., 2024 — "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"](https://arxiv.org/abs/2406.16858) — cấu trúc liên kết cây động
- [Li et al., 2025 — "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"](https://arxiv.org/abs/2503.01840) — khớp thời gian kiểm tra tàu
- [Fu, Haotian, Peng et al., 2024 — "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"](https://arxiv.org/abs/2402.02057) — giải mã Jacobi/lookahead, một giải pháp thay thế không có đầu cơ
