# Phương pháp Monte Carlo - Học hỏi từ Episodes hoàn chỉnh

> Lập trình động cần một model. Monte Carlo không cần gì ngoài episodes. Chạy policy, theo dõi lợi nhuận, tính trung bình chúng. Ý tưởng đơn giản nhất trong RL - và là ý tưởng mở khóa mọi thứ ở hạ lưu.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 9 · 01 (MDP), Giai đoạn 9 · 02 (Lập trình động)
**Thời lượng:** ~75 phút

## Vấn đề

Lập trình động rất thanh lịch, nhưng nó giả định bạn có thể truy vấn `P(s' | s, a)` cho mọi trạng thái và hành động. Hầu như không có gì trong thế giới thực hoạt động theo cách đó. Một robot không thể tính toán phân tích sự phân bố trên các pixel của máy ảnh sau một mô-men xoắn chung. Thuật toán định giá không thể tích hợp trên mọi phản ứng có thể có của khách hàng. Một LLM không thể liệt kê tất cả các lần tiếp tục có thể xảy ra sau một token.

Bạn cần một phương pháp chỉ cần khả năng *lấy mẫu* từ môi trường. Chạy policy. Nhận một `s_0, a_0, r_1, s_1, a_1, r_2, …, s_T` quỹ đạo. Sử dụng nó để ước tính các giá trị. Đó là Monte Carlo.

Sự chuyển đổi từ DP sang MC rất quan trọng về mặt triết học: chúng ta chuyển từ *model đã biết + sao lưu chính xác* sang *rollouts lấy mẫu + lợi nhuận trung bình*. Các variance nhảy vọt, nhưng khả năng ứng dụng bùng nổ. Mọi thuật toán RL sau bài học này - TD, Q-learning, REINFORCE, PPO, GRPO - đều là một công cụ ước tính Monte Carlo trong tim, đôi khi với bootstrapping được xếp lớp trên cùng.

## Khái niệm

![Monte Carlo: rollout, compute returns, average; first-visit vs every-visit](../assets/monte-carlo.svg)

**Ý tưởng cốt lõi, trong một dòng: **`V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)` nơi `G^{(i)}(s)` được quan sát trở lại sau các chuyến thăm đến `s` dưới policy `π`.

**Chuyến thăm đầu tiên so với mỗi lần ghé thăm MC.** Cho một episode đến thăm tiểu bang `s` nhiều lần, MC lần thăm đầu tiên chỉ tính số tiền trở lại từ lần thăm đầu tiên; mỗi lần truy cập MC tính tất cả các lượt truy cập. Cả hai đều không thiên vị về giới hạn. Lần thăm đầu tiên đơn giản hơn để phân tích (mẫu iid). Mỗi lần truy cập sử dụng nhiều dữ liệu hơn mỗi episode và thường hội tụ nhanh hơn trong thực tế.

**Giá trị trung bình gia tăng.** Thay vì lưu trữ tất cả lợi nhuận, hãy cập nhật mức trung bình đang chạy:

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

Tổ chức lại: `V_new = V_old + α · (target - V_old)` với `α = 1/n`. Hoán đổi `1/n` cho một `α ∈ (0, 1)` kích thước bước không đổi và bạn sẽ nhận được một công cụ ước tính MC không cố định theo dõi các thay đổi trong `π`. Động thái đó là toàn bộ bước nhảy từ MC sang TD đến mọi thuật toán RL hiện đại.

**Khám phá bây giờ là một vấn đề. **DP chạm vào mọi trạng thái bằng cách liệt kê. MC chỉ nhìn thấy các tiểu bang policy lượt truy cập. Nếu `π` là xác định, toàn bộ các vùng của không gian trạng thái không bao giờ được lấy mẫu và ước tính giá trị của chúng ở mức không mãi mãi. Ba bản sửa lỗi, theo thứ tự lịch sử:

1. **Bắt đầu khám phá. **Bắt đầu mỗi episode từ một cặp (s, a) ngẫu nhiên. Đảm bảo phạm vi bảo hiểm; không thực tế trong thực tế (bạn không thể "đặt lại" rô bốt về trạng thái tùy ý).
2. **ε tham lam.** Hành động tham lam w.r.t. Q hiện tại, nhưng với xác suất `ε` chọn một hành động ngẫu nhiên. Tất cả các cặp trạng thái-hành động đều được lấy mẫu tiệm cận.
3. **Off-policy MC.** Thu thập dữ liệu theo một policy `μ` hành vi, tìm hiểu về policy `π` mục tiêu thông qua sampling quan trọng. variance cao, nhưng đó là cầu nối cho các phương pháp đệm phát lại như DQN.

**Kiểm soát Monte Carlo.** Đánh giá → cải thiện → đánh giá, giống như policy lần lặp lại, nhưng đánh giá dựa trên sampling:

1. Chạy `π`, nhận episode.
2. Cập nhật `Q(s, a)` từ lợi nhuận quan sát được.
3. Làm `Q` wrt tham lam `π` ε.
4. Lặp lại.

Hội tụ đến `Q*` và `π*` với xác suất 1 trong điều kiện nhẹ (mỗi cặp đến thăm thường xuyên vô hạn, `α` thỏa mãn Robbins-Monro).

```figure
epsilon-greedy
```

## Tự xây dựng

### Bước 1: rollout → danh sách (s, a, r)

```python
def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

Không có model, chỉ có `env.reset()` và `env.step(s, a)`. Giao diện tương tự như môi trường phòng tập thể dục nhưng bị tước bỏ.

### Bước 2: tính toán lợi nhuận (quét ngược)

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

Một lần vượt qua, `O(T)`. Sự lặp lại ngược `G_t = r_{t+1} + γ G_{t+1}` tránh được việc tổng lại.

### Bước 3: Đánh giá MC lần đầu

```python
def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

Ba dòng thực hiện công việc: đánh dấu trạng thái như đã thấy trong lần truy cập đầu tiên, số lượng gia tăng, cập nhật trung bình chạy.

### Bước 4: Điều khiển MC tham lam ε (bật-policy)

```python
def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

### Bước 5: so sánh với tiêu chuẩn vàng DP

Ước tính MC của bạn về `V^π` phải phù hợp với kết quả DP từ Bài 02 là episodes → ∞. Trong thực tế: 50.000 episodes trên 4×4 GridWorld giúp bạn chỉ còn `~0.1` câu trả lời DP.

## Cạm bẫy

- **episodes vô hạn. **MC yêu cầu episodes để * chấm dứt*. Nếu policy của bạn có thể lặp lại mãi mãi, hãy giới hạn `max_steps` và coi giới hạn là lỗi ngầm. GridWorld với một policy ngẫu nhiên thường hết thời gian chờ - đó là điều bình thường, chỉ cần đảm bảo rằng bạn đếm nó một cách chính xác.
- **Variance.** MC sử dụng lợi nhuận đầy đủ. Trong episodes dài, sự variance là rất lớn - một phần thưởng không may mắn ở cuối cùng sẽ thay đổi `V(s_0)` với cùng một số tiền. Phương pháp TD (Bài 04) cắt giảm điều này bằng cách khởi động.
- **Phạm vi bảo hiểm của tiểu bang. **MC tham lam trên một Q mới có mối quan hệ sẽ chỉ thử một hành động. Bạn * phải* khám phá (tham lam ε, khám phá bắt đầu, UCB).
- **policies không cố định.** Nếu `π` thay đổi (như trong điều khiển MC), các khoản trả về cũ sẽ đến từ một policy khác. MC α không đổi xử lý việc này; MC trung bình mẫu thì không.
- **Tầm quan trọng của policy không sampling.** Trọng số `π(a|s)/μ(a|s)` nhân lên trên một quỹ đạo. Variance bùng nổ với đường chân trời. Giới hạn với IS có trọng số cho mỗi quyết định hoặc chuyển sang TD.

## Ứng dụng

Vai trò năm 2026 của các phương pháp Monte Carlo:

| Trường hợp sử dụng | Tại sao chọn MC |
|----------|--------|
| Trò chơi chân trời ngắn (blackjack, poker) | Episodes chấm dứt một cách tự nhiên; lợi nhuận sạch sẽ. |
| Đánh giá ngoại tuyến của policy đã ghi | Lợi nhuận chiết khấu trung bình so với quỹ đạo được lưu trữ. |
| Tìm kiếm cây Monte Carlo (AlphaZero) | MC rollouts từ hướng dẫn lựa chọn lá cây. |
| Đánh giá LLM RL | Tính toán phần thưởng trung bình trên số lần hoàn thành được lấy mẫu cho một policy nhất định. |
| Ước tính cơ sở trong PPO | Mục tiêu lợi thế `A_t = G_t - V(s_t)` sử dụng MC `G_t`. |
| Giảng dạy RL | Thuật toán đơn giản nhất thực sự hoạt động - thoát khỏi bootstrapping để xem lõi. |

Các thuật toán RL sâu hiện đại (PPO, SAC) nội suy giữa MC thuần túy (trả về đầy đủ) và TD thuần túy (khởi động một bước) thông qua trả về `n` bước hoặc GAE. Cả hai endpoints đều là các trường hợp của cùng một ước tính.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-mc-evaluator.md`:

```markdown
---
name: mc-evaluator
description: Evaluate a policy via Monte Carlo rollouts and produce a convergence report with DP-comparison if available.
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

Given an environment (episodic, with reset+step API) and a policy, output:

1. Method. First-visit vs every-visit MC. Reason.
2. Episode budget. Target number, variance diagnostic, expected standard error.
3. Exploration plan. ε schedule (if needed) or exploring starts.
4. Gold-standard comparison. DP-optimal V* if tabular; otherwise a bound from a Q-learning / PPO baseline.
5. Termination check. Max-step cap, timeouts, handling of non-terminating trajectories.

Refuse to run MC on non-episodic tasks without a finite horizon cap. Refuse to report V^π estimates from fewer than 100 episodes per state for tabular tasks. Flag any policy with zero-variance actions as an exploration risk.
```

## Bài tập

1. **Dễ dàng.** Thực hiện đánh giá MC lần đầu tiên về policy ngẫu nhiên đồng nhất trên 4×4 GridWorld. Chạy 10,000 episodes. Vẽ `V(0,0)` như một hàm của episode tính vào câu trả lời DP.
2. **Trung bình.** Thực hiện kiểm soát MC tham lam ε với `ε ∈ {0.01, 0.1, 0.3}`. So sánh lợi nhuận trung bình sau 20.000 episodes. Đường cong trông như thế nào? Sự đánh đổi bias-variance tồn tại ở đâu?
3. **Khó.** Thực hiện MC *off-policy* với tầm quan trọng sampling: thu thập dữ liệu theo policy `μ` ngẫu nhiên đồng nhất, ước tính `V^π` cho policy `π` tối ưu xác định. So sánh IS thuần túy so với IS theo quyết định so với IS có trọng số. Cái nào có variance thấp nhất?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Monte Carlo | "sampling ngẫu nhiên" | Ước tính kỳ vọng bằng cách tính trung bình trên các mẫu iid từ phân phối. |
| Trả `G_t` | "Phần thưởng trong tương lai" | Tổng phần thưởng giảm giá từ bước `t` đến cuối episode: `Σ_{k≥0} γ^k r_{t+k+1}`. |
| MC thăm khám đầu tiên | "Đếm mỗi trạng thái một lần" | Chỉ lần đầu tiên trong một episode mới góp phần vào ước tính giá trị. |
| MC mỗi lần ghé thăm | "Sử dụng tất cả các lượt truy cập" | Mỗi lượt truy cập đều đóng góp; hơi thiên vị nhưng hiệu quả hơn về mẫu. |
| ε tham lam | "Nhiễu thăm dò" | Chọn hành động tham lam với prob `1-ε`; hành động ngẫu nhiên với prob `ε`. |
| Tầm quan trọng sampling | "Sửa sampling từ phân phối sai" | Cân lại trả về theo 'π(a\ | s)/μ(a\ | s)` products to estimate `V^π` from `μ' dữ liệu. |
| Trên policy | "Học hỏi từ dữ liệu của chính mình" | Mục tiêu policy = hành vi policy. MC vani, PPO, SARSA. |
| Tắt policy | "Học hỏi từ dữ liệu của người khác" | Nhắm mục tiêu hành vi policy ≠ policy. MC lấy mẫu quan trọng, Q-learning, DQN. |

## Đọc thêm

- [Sutton & Barto (2018). Ch. 5 — Monte Carlo Methods](http://incompleteideas.net/book/RLbook2020.pdf) - cách xử lý kinh điển.
- [Singh & Sutton (1996). Reinforcement Learning with Replacing Eligibility Traces](https://link.springer.com/article/10.1007/BF00114726) - phân tích lần đầu tiên so với mỗi lần ghé thăm.
- [Precup, Sutton, Singh (2000). Eligibility Traces for Off-Policy Policy Evaluation](http://incompleteideas.net/papers/PSS-00.pdf) - điều khiển MC và variance ngoài policy.
- [Mahmood et al. (2014). Weighted Importance Sampling for Off-Policy Learning](https://arxiv.org/abs/1404.6362) - các ước tính IS variance thấp hiện đại.
- [Tesauro (1995). TD-Gammon, A Self-Teaching Backgammon Program](https://dl.acm.org/doi/10.1145/203330.203343) - minh chứng thực nghiệm quy mô lớn đầu tiên về MC/TD tự chơi hội tụ với trò chơi siêu phàm; tiền thân khái niệm cho mọi bài học trong nửa sau của giai đoạn này.
