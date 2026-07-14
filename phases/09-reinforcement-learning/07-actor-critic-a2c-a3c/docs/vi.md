# Diễn viên kiêm nhà phê bình - A2C và A3C

> REINFORCE ồn ào. Thêm một nhà phê bình học được `V̂(s)`, trừ nó khỏi lợi nhuận và bạn sẽ nhận được lợi thế có cùng kỳ vọng nhưng variance thấp hơn nhiều. Đó là diễn viên-nhà phê bình. A2C chạy nó đồng bộ; A3C chạy nó trên threads. Cả hai đều là model tinh thần cho mọi phương pháp RL sâu hiện đại.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 9 · 04 (TD Learning), Giai đoạn 9 · 06 (CỦNG cố)
**Thời lượng:** ~75 phút

## Vấn đề

Vanilla REINFORCE hoạt động, nhưng variance của nó rất khủng khiếp. Monte Carlo trở lại `G_t` có thể dao động trên hệ số 10 giữa episodes. Nhân nhiễu đó với `∇ log π` và tính trung bình tạo ra một công cụ ước tính gradient mất hàng nghìn episodes để di chuyển policy cùng một khoảng cách mà bạn có thể di chuyển nó với ít cập nhật DQN hơn nhiều.

variance đến từ việc sử dụng lợi nhuận thô. Nếu bạn trừ đi một `b(s_t)` đường cơ sở - bất kỳ hàm nào của trạng thái, bao gồm cả giá trị đã học - kỳ vọng không thay đổi và variance giảm xuống. Đường cơ sở có thể xử lý tốt nhất là `V̂(s_t)`. Bây giờ số lượng nhân `∇ log π` là *lợi thế*:

`A(s, a) = G - V̂(s)`

Một hành động là tốt nếu nó tạo ra lợi nhuận trên mức trung bình; xấu nếu thấp hơn. CỦNG cố với một nhà phê bình uyên bác là * diễn viên-nhà phê bình *. Nhà phê bình cho diễn viên một giáo viên variance thấp. Đây là mọi phương pháp policy sâu sau năm 2015 (A2C, A3C, PPO, SAC, IMPALA).

## Khái niệm

![Actor-critic: policy net plus value net, TD residual as advantage](../assets/actor-critic.svg)

**Hai mạng, một loss dùng chung:**

- **Diễn viên **`π_θ(a | s)`: policy. Được lấy mẫu để diễn xuất. Được huấn luyện với policy gradient.
- **Phê bình **`V_φ(s)`: ước tính lợi nhuận dự kiến từ tiểu bang. Được huấn luyện để giảm thiểu `(V_φ(s) - target)²`.

**Ưu điểm.** Hai hình thức tiêu chuẩn:

- *MC advantage:* `A_t = G_t - V_φ(s_t)`. Không thiên vị, variance cao hơn.
- *TD advantage:* `A_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)`. Thiên vị (sử dụng `V_φ`), variance thấp hơn nhiều. Còn được gọi là *TD residual* `δ_t`.

**Lợi thế n-bước.** Nội suy giữa hai:

`A_t^{(n)} = r_{t+1} + γ r_{t+2} + … + γ^{n-1} r_{t+n} + γ^n V_φ(s_{t+n}) - V_φ(s_t)`

`n = 1` là TD thuần túy. `n = ∞` là MC. Hầu hết các triển khai đều sử dụng `n = 5` cho Atari, `n = 2048` cho PPO trên MuJoCo.

**Ước tính lợi thế tổng quát (GAE).** Schulman và cộng sự (2016) đã đề xuất một mức trung bình có trọng số theo cấp số nhân trên tất cả các lợi thế n bước:

`A_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`

với `λ ∈ [0, 1]`. `λ = 0` là TD (variance thấp, bias cao). `λ = 1` là MC (variance cao, không thiên vị). `λ = 0.95` là mặc định năm 2026 - điều chỉnh cho đến khi mặt số bias/variance ở vị trí bạn muốn.

**A2C: lợi thế đồng bộ giữa diễn viên-nhà phê bình.** Thu thập `T` bước trên `N` môi trường song song. Lợi thế điện toán cho từng bước. Cập nhật diễn viên và nhà phê bình về batch kết hợp. Lặp lại. Người anh em đơn giản hơn, có khả năng mở rộng hơn của A3C.

**A3C: lợi thế không đồng bộ diễn viên-nhà phê bình.** Mnih et al. (2016). Spawn `N` worker threads, mỗi người chạy một env. Mỗi worker tính toán gradients cục bộ trên rollout riêng của nó, sau đó áp dụng chúng không đồng bộ cho một parameter server dùng chung. Không cần bộ đệm phát lại — workers khử tương quan bằng cách chạy các quỹ đạo khác nhau. A3C đã chứng minh bạn có thể huấn luyện trên CPUs trên quy mô lớn. Vào năm 2026, A2C dựa trên GPU (env song song hàng loạt) chiếm ưu thế vì GPUs muốn batches lớn.

**Kết hợp loss.**

`L(θ, φ) = -E[ A_t · log π_θ(a_t | s_t) ]  +  c_v · E[(V_φ(s_t) - G_t)²]  -  c_e · E[H(π_θ(·|s_t))]`

Ba thuật ngữ: policy-gradient loss, hồi quy giá trị, phần thưởng entropy. `c_v ~0.5`, `c_e ~0.01` là điểm khởi đầu chuẩn.

## Tự xây dựng

### Bước 1: một nhà phê bình

Nhà phê bình tuyến tính `V_φ(s) = w · features(s)` cập nhật với MSE:

```python
def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

Trên một môi trường dạng bảng, nhà phê bình hội tụ trong vài trăm episodes. Trên Atari, thay thế nhà phê bình tuyến tính bằng một trung kế CNN được chia sẻ + đầu giá trị.

### Bước 2: lợi thế n-step

Cho một rollout độ dài `T` và một `V(s_T)` cuối cùng được khởi động:

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

`returns` là mục tiêu của các nhà phê bình. `advantages` là những gì nhân lên `∇ log π`.

### Bước 3: Cập nhật kết hợp

```python
for step_i, (x, a, _r, probs) in enumerate(traj):
    adv = advantages[step_i]
    target_v = returns[step_i]

    # critic
    critic_update(w, x, target_v, lr_v)

    # actor
    for i in range(N_ACTIONS):
        grad_logpi = (1.0 if i == a else 0.0) - probs[i]
        for j in range(N_FEAT):
            theta[i][j] += lr_a * adv * grad_logpi * x[j]
```

On-policy, một rollout cho mỗi bản cập nhật, tỷ lệ học tập riêng biệt cho diễn viên và nhà phê bình.

### Bước 4: song song hóa (A3C so với A2C)

- **A3C: **quay lên `N` threads. Mỗi người chạy môi trường riêng và forward pass riêng. Định kỳ đẩy các bản cập nhật gradient đến một chính được chia sẻ. Không có khóa trên chính - các cuộc đua vẫn ổn, chúng chỉ thêm nhiễu.
- **A2C:** chạy `N` các phiên bản env trong một process duy nhất, stack các quan sát thành một backward pass `[N, obs_dim]` batch, forward pass hàng loạt, hàng loạt. Sử dụng GPU cao hơn, xác định, dễ suy luận hơn. Mặc định vào năm 2026.

Mã đồ chơi của chúng tôi là một luồng để rõ ràng; viết lại thành A2C hàng loạt là ba dòng numpy.

## Cạm bẫy

- **Nhà phê bình bias trước diễn viên gradient.** Nếu nhà phê bình là ngẫu nhiên, đường cơ sở của nó là không có thông tin và bạn đang training với nhiễu thuần túy. Làm nóng nhà phê bình trong vài trăm bước trước khi bật policy gradient hoặc sử dụng một diễn viên chậm learning rate.
- **Chuẩn hóa lợi thế. **Chuẩn hóa lợi thế lên zero-mean/unit-std mỗi batch. Ổn định training ồ ạt với chi phí gần như bằng không.
- **Thân cây dùng chung.** Sử dụng trình trích xuất feature dùng chung cho diễn viên và nhà phê bình về đầu vào hình ảnh. Đầu riêng biệt. Chia sẻ features miễn phí cho cả hai tổn thất.
- **Hợp đồng policy.** A2C sử dụng lại dữ liệu cho chính xác một bản cập nhật. Nhiều hơn và gradient của bạn bị thiên vị (hiệu chỉnh sampling quan trọng là những gì PPO thêm vào).
- **Sự sụp đổ entropy.** Nếu không có `c_e > 0`, policy trở nên gần như xác định trong vài trăm bản cập nhật và ngừng khám phá.
- **Thang điểm thưởng.** Mức độ lợi thế phụ thuộc vào thang điểm thưởng. Chuẩn hóa phần thưởng (ví dụ: phân chia chạy-tiêu chuẩn) để có gradient độ nhất quán giữa các nhiệm vụ.

## Ứng dụng

A2C/A3C hiếm khi là lựa chọn cuối cùng vào năm 2026 nhưng chúng là kiến trúc mà mọi thứ sau này tinh chỉnh:

| Phương pháp | Liên quan đến A2C |
|--------|----------------|
| PPO | A2C + tỷ lệ quan trọng bị cắt cho các bản cập nhật nhiều epoch |
| IMPALA | Hiệu chỉnh tắt policy A3C + V-trace |
| SAC (Giai đoạn 9 · 07) | Off-policy A2C với một nhà phê bình giá trị mềm (bài học tiếp theo) |
| GRPO (Giai đoạn 9 · 12) | A2C không có nhà phê bình - lợi thế tương đối nhóm |
| DPO | A2C sụp đổ thành loss xếp hạng ưu tiên, không sampling |
| AlphaStar / OpenAI Năm | A2C với giải đấu training + giả trước khi training |

Nếu bạn thấy "lợi thế" trong một bài báo năm 2026, hãy nghĩ đến diễn viên kiêm nhà phê bình.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-actor-critic-trainer.md`:

```markdown
---
name: actor-critic-trainer
description: Produce an A2C / A3C / GAE configuration for a given environment, with advantage estimation and loss weights specified.
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

Given an environment and compute budget, output:

1. Parallelism. A2C (GPU batched) vs A3C (CPU async) and the number of workers.
2. Rollout length T. Steps per env per update.
3. Advantage estimator. n-step or GAE(λ); specify λ.
4. Loss weights. `c_v` (value), `c_e` (entropy), gradient clip.
5. Learning rates. Actor and critic (separate if using).

Refuse single-worker A2C on environments with horizon > 1000 (too on-policy, too slow). Refuse to ship without advantage normalization. Flag any run with `c_e = 0` and observed entropy < 0.1 as entropy-collapsed.
```

## Bài tập

1. **Dễ dàng.** Huấn luyện diễn viên-nhà phê bình với lợi thế MC (`G_t - V(s_t)`) trên 4×4 GridWorld. So sánh hiệu quả của mẫu với REINFORCE-with-running-mean-baseline từ Bài 06.
2. **Trung bình.** Chuyển sang lợi thế dư TD (`r + γ V(s') - V(s)`). Đo lường variance lợi thế batches. Nó giảm bao nhiêu?
3. **Khó.** Triển khai GAE (λ). Quét `λ ∈ {0, 0.5, 0.9, 0.95, 1.0}`. Vẽ biểu đồ trả lại cuối cùng so với hiệu quả mẫu. Điểm ngọt ngào bias/variance nhất cho nhiệm vụ này là gì?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Diễn viên | "Mạng lưới policy" | 'π_θ(a\ | s)', được cập nhật bởi policy gradient. |
| Nhà phê bình | "Giá trị ròng" | `V_φ(s)`, được cập nhật bởi hồi quy MSE thành mục tiêu lợi nhuận / TD. |
| Lợi thế | "Tốt hơn mức trung bình bao nhiêu" | `A(s, a) = Q(s, a) - V(s)` hoặc các công cụ ước tính của nó. Hệ số nhân cho `∇ log π`. |
| TD dư | "δ" | `δ_t = r + γ V(s') - V(s)`; ước tính lợi thế một bước. |
| GAE | "Núm nội suy" | Tổng các ưu điểm n bước có trọng số theo cấp số nhân, được tham số hóa bằng `λ`. |
| A2C | "Diễn viên-nhà phê bình đồng bộ" | Lô trên các môi trường; một bước gradient mỗi rollout. |
| Đáp 3C | "Diễn viên-nhà phê bình không đồng bộ" | Worker threads đẩy gradients đến một tham số chung server. Giấy gốc; ít phổ biến hơn vào năm 2026. |
| Dây khởi động | "Sử dụng chữ V ở đường chân trời" | Cắt bớt rollout, thêm `γ^n V(s_{t+n})` để đóng tổng. |

## Đọc thêm

- [Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — A3C, bài báo của diễn viên-nhà phê bình không đồng bộ ban đầu.
- [Schulman et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) - GAE.
- [Sutton & Barto (2018). Ch. 13 — Actor-Critic Methods](http://incompleteideas.net/book/RLbook2020.pdf) - nền tảng; ghép nối điều này với Ch. 9 về xấp xỉ hàm khi nhà phê bình là một mạng nơ-ron.
- [Espeholt et al. (2018). IMPALA](https://arxiv.org/abs/1802.01561) - diễn viên-nhà phê bình phân tán có thể mở rộng với hiệu chỉnh ngoài policy V-trace.
- [OpenAI Baselines / Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - production A2C/PPO triển khai đáng đọc.
- [Konda & Tsitsiklis (2000). Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms) - kết quả hội tụ cơ bản cho sự phân rã diễn viên-nhà phê bình hai thời gian.
