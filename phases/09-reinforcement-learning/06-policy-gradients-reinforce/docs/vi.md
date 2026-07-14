# Policy Gradient - CỦNG CỐ TỪ ĐẦU

> Ngừng ước tính giá trị. Tham số hóa policy trực tiếp, tính toán gradient lợi nhuận kỳ vọng, bước lên dốc. Williams (1992) đã viết nó trong một định lý. Đó là lý do tại sao PPO, GRPO và mọi vòng lặp LLM RL tồn tại.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 3 · 03 (Backpropagation), Giai đoạn 9 · 03 (Monte Carlo), Giai đoạn 9 · 04 (TD Learning)
**Thời lượng:** ~75 phút

## Vấn đề

Q-learning và DQN tham số hóa hàm *value*. Bạn chọn hành động theo `argmax Q`. Điều đó tốt cho các hành động rời rạc và trạng thái rời rạc. Nó gặp lỗi khi các hành động liên tục (`argmax` trên mô-men xoắn 10 chiều?) hoặc khi bạn muốn một policy ngẫu nhiên (`argmax` là xác định theo cấu trúc).

Thay vào đó, Policy gradients tham số hóa *policy*. `π_θ(a | s)` là một mạng nơ-ron xuất ra phân phối trên các hành động. Lấy mẫu từ nó để hành động. Tính toán gradient lợi nhuận kỳ vọng đối với `θ`. Bước lên dốc. Không có `argmax`. Không có đệ quy Bellman. Chỉ cần gradient đi lên trên `J(θ) = E_{π_θ}[G]`.

Định lý REINFORCE (Williams 1992) cho bạn biết gradient này có thể tính toán được: `∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`. Chạy một episode. Tính toán lợi nhuận. Nhân với `∇ log π_θ(a | s)` ở mỗi bước. Trung bình. Gradient đi lên. Xong.

Mọi thuật toán LLM-RL vào năm 2026 - PPO, DPO, GRPO - đều là một sự tinh chỉnh của REINFORCE. Hiểu nó trong ngón tay của bạn là điều kiện tiên quyết để rest giai đoạn này và cho Giai đoạn 10 · 07 (triển khai RLHF) và Giai đoạn 10 · 08 (DPO).

## Khái niệm

![Policy gradient: softmax policy, log-π gradient, return-weighted update](../assets/policy-gradient.svg)

**Định lý policy gradient.** Đối với bất kỳ policy `π_θ` nào được tham số hóa bởi `θ`:

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

trong đó `G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}` là lợi nhuận chiết khấu từ bước `t`. Kỳ vọng là trên quỹ đạo đầy đủ `τ` lấy mẫu từ `π_θ`.

**Chứng minh ngắn.** Phân biệt `J(θ) = Σ_τ P(τ; θ) G(τ)` theo kỳ vọng. Sử dụng `∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)` (thủ thuật đạo hàm log). Yếu tố `log P(τ; θ) = Σ log π_θ(a_t | s_t) + environment terms that do not depend on θ`. Các thuật ngữ môi trường biến mất. Hai dòng đại số cho bạn định lý.

**Variance thủ thuật giảm thiểu.** Vanilla REINFORCE có variance giết người - trả lại ồn ào, `∇ log π` ồn ào, sản phẩm của họ rất ồn. Hai bản sửa lỗi tiêu chuẩn:

1. **Trừ đường cơ sở.** Thay thế `G_t` bằng `G_t - b(s_t)` cho bất kỳ `b(s_t)` cơ sở nào không phụ thuộc vào `a_t`. Không thiên vị vì `E[b(s_t) · ∇ log π(a_t | s_t)] = 0`. Lựa chọn điển hình: `b(s_t) = V̂(s_t)` học được bởi một nhà phê bình → diễn viên-nhà phê bình (Bài 07).
2. **Phần thưởng để đi.** Thay thế `Σ_t G_t · ∇ log π_θ(a_t | s_t)` bằng `Σ_t G_t^{from t} · ∇ log π_θ(a_t | s_t)`. Chỉ có lợi nhuận trong tương lai mới quan trọng đối với một hành động nhất định - phần thưởng trong quá khứ đóng góp nhiễu không trung bình.

Kết hợp, bạn nhận được:

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

đó là REINFORCE với đường cơ sở - tổ tiên trực tiếp của A2C (Bài 07) và PPO (Bài 08).

**Softmax policy tham số hóa.** Đối với các hành động rời rạc, lựa chọn tiêu chuẩn:

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

trong đó `f_θ` là bất kỳ mạng nơ-ron nào xuất ra điểm số cho mỗi hành động. gradient có dạng sạch:

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

tức là điểm của hành động đã thực hiện trừ đi giá trị kỳ vọng của nó theo policy.

**Gaussian policy cho các hành động liên tục.** `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`. `∇ log N(a; μ, σ)` có dạng đóng. Đó là tất cả những gì Giai đoạn 9 · SAC của 07 cần.

```figure
policy-gradient-landscape
```

## Tự xây dựng

### Bước 1: softmax policy mạng

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

Sử dụng policy tuyến tính (một trọng số vector cho mỗi hành động) cho môi trường dạng bảng. Đối với Atari, hoán đổi CNN và giữ đầu softmax.

### Bước 2: sampling và xác suất log

```python
def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)
```

### Bước 3: rollout với log-probs được chụp

```python
def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory
```

### Bước 4: TĂNG CƯỜNG cập nhật

```python
def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]
```

gradient `∇ log π(a|s) = e_a - π(·|s)` (mộthot của `a` trừ xác suất) là trái tim của softmax policy gradients. Đốt cháy nó vào trí nhớ cơ bắp.

### Bước 5: đường cơ sở

Giá trị trung bình chạy là `G` trong episodes gần đây là đủ variance giảm để có được GridWorld 4×4 chạy; phải mất ~500 episodes để hội tụ. Nâng cấp đường cơ sở lên một `V̂(s)` đã học và bạn sẽ nhận được diễn viên-nhà phê bình.

## Cạm bẫy

- **Bùng nổ gradients.** Lợi nhuận có thể rất lớn. Luôn chuẩn hóa `G` để `~N(0, 1)` trên batch trước khi nhân với `∇ log π`.
- **Entropy sụp đổ.** policy hội tụ đến một hành động gần như xác định quá sớm, ngừng khám phá, bị mắc kẹt. Khắc phục: thêm `β · H(π(·|s))` phần thưởng entropy vào mục tiêu.
- **variance cao.** Vanilla REINFORCE cần hàng nghìn episodes. Đường cơ sở của nhà phê bình (Bài 07) hoặc vùng tin cậy của TRPO/PPO (Bài 08) là bản sửa lỗi tiêu chuẩn.
- **Lấy mẫu kém hiệu quả.** Bật policy có nghĩa là bạn vứt bỏ mọi chuyển tiếp sau một lần cập nhật. Hiệu chỉnh ngoài policy thông qua tầm quan trọng sampling mang lại dữ liệu, với chi phí là độ variance (tỷ lệ của PPO là trọng số IS bị cắt).
- **gradients không cố định.** gradient tương tự từ 100 episodes trước sử dụng `π` cũ. Các phương thức policy cập nhật vài rollouts một lần vì lý do này.
- **Chỉ định tín dụng. **Nếu không có phần thưởng để đi, phần thưởng trong quá khứ sẽ góp phần gây ồn ào. Luôn sử dụng phần thưởng để đi.

## Ứng dụng

Vào năm 2026, REINFORCE hiếm khi được vận hành trực tiếp nhưng công thức gradient của nó ở khắp mọi nơi:

| Trường hợp sử dụng | Phương pháp dẫn xuất |
|----------|---------------|
| Kiểm soát liên tục | PPO / SAC với Gaussian policy |
| LLM RLHF | PPO với hình phạt KL, chạy trên policy cấp token |
| Lý luận LLM (DeepSeek) | GRPO - CỦNG CỐ VỚI ĐƯỜNG CƠ SỞ TƯƠNG ĐỐI NHÓM, không có người chỉ trích |
| Đa agent | Nhà phê bình tập trung REINFORCE (MADDPG, COMA) |
| Robot hành động rời rạc | A2C, A3C, PPO |
| Cài đặt chỉ tùy chọn | DPO — REINFORCE được viết lại thành likelihood loss ưu tiên, không sampling |

Khi bạn đọc `loss = -advantage * log_prob` trong một training script năm 2026, đó là CỦNG CỐ với đường cơ sở. Toàn bộ bài báo (DPO, GRPO, RLOO) là thủ thuật giảm variance trên một dòng này.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-policy-gradient-trainer.md`:

```markdown
---
name: policy-gradient-trainer
description: Produce a REINFORCE / actor-critic / PPO training config for a given task and diagnose variance issues.
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

Given an environment (discrete / continuous actions, horizon, reward stats), output:

1. Policy head. Softmax (discrete) or Gaussian (continuous) with parameter counts.
2. Baseline. None (vanilla), running mean, learned `V̂(s)`, or A2C critic.
3. Variance controls. Reward-to-go on by default, return normalization, gradient clip value.
4. Entropy bonus. Coefficient β and decay schedule.
5. Batch size. Episodes per update; on-policy data freshness contract.

Refuse REINFORCE-no-baseline on horizons > 500 steps. Refuse continuous-action control with a softmax head. Flag any run with `β = 0` and observed policy entropy < 0.1 as entropy-collapsed.
```

## Bài tập

1. **Dễ dàng.** Triển khai REINFORCE trên 4×4 GridWorld với softmax policy tuyến tính. Huấn luyện 1.000 episodes mà không có đường cơ sở. Vẽ đường cong học tập; đo lường variance (chuẩn lợi nhuận).
2. **Trung bình.** Thêm đường cơ sở trung bình chạy. Huấn luyện lại. So sánh hiệu quả và variance mẫu với đường chạy vani. Đường cơ sở giảm các bước hội tụ xuống bao nhiêu?
3. **Khó.** Thêm phần thưởng entropy `β · H(π)`. Quét `β ∈ {0, 0.01, 0.1, 1.0}`. Vẽ sơ đồ trở lại cuối cùng và policy entropy. Điểm ngọt ngào trong nhiệm vụ này ở đâu?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Policy gradient | "Huấn luyện policy trực tiếp" | '∇J(θ) = E[G · ∇ log π_θ(a\ | s)]`; bắt nguồn từ thủ thuật phái sinh log. |
| CỦNG cố | "Thuật toán PG ban đầu" | Williams (1992); Monte Carlo trở lại nhân với log-policy gradient. |
| Thủ thuật log-đạo hàm | "Công cụ ước tính hàm điểm" | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`; làm cho gradients kỳ vọng trở nên dễ dàng. |
| Đường cơ sở | "Giảm Variance" | Bất kỳ `b(s)` nào trừ đi `G`; không thiên vị vì `E[b · ∇ log π] = 0`. |
| Phần thưởng để đi | "Chỉ tính lợi nhuận trong tương lai" | `G_t^{from t}` thay vì `G_0` đầy đủ; chính xác và variance thấp hơn. |
| Tiền thưởng entropy | "Khuyến khích khám phá" | `+β · H(π(·\ | s))' giữ cho policy không bị sụp đổ. |
| Trên policy | "Rèn luyện những gì bạn vừa thấy" | Gradient kỳ vọng là w.r.t. policy hiện tại - không thể sử dụng lại dữ liệu cũ trực tiếp. |
| Lợi thế | "Tốt hơn mức trung bình bao nhiêu" | `A(s, a) = G(s, a) - V(s)`; số lượng đã ký REINFORCE-with-baseline nhân lên. |

## Đọc thêm

- [Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) — bài báo REINFORCE gốc.
- [Sutton et al. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) — định lý policy-gradient hiện đại với xấp xỉ hàm.
- [Sutton & Barto (2018). Ch. 13 — Policy Gradient Methods](http://incompleteideas.net/book/RLbook2020.pdf) - trình bày sách giáo khoa.
- [OpenAI Spinning Up — VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) - giải thích sư phạm rõ ràng với mã PyTorch.
- [Peters & Schaal (2008). Reinforcement Learning of Motor Skills with Policy Gradients](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf) - Giảm variance và quan điểm gradient tự nhiên kết nối REINFORCE với gia đình khu vực tin cậy (TRPO, PPO).
