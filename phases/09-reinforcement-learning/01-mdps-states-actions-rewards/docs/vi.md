# MDP, tiểu bang, hành động và phần thưởng

> Một Process của Markov Decision là năm điều: trạng thái, hành động, chuyển tiếp, phần thưởng, chiết khấu. Mọi thứ trong RL - Q-learning, PPO, DPO, GRPO - đều tối ưu hóa trên hình dạng này. Học nó một lần, đọc rest của học tăng cường miễn phí.

**Loại:** Học
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 1 · 06 (Xác suất & Phân phối), Giai đoạn 2 · 01 (ML Phân loại)
**Thời lượng:** ~45 phút

## Vấn đề

Bạn đang viết một bot cờ vua. Hoặc một công cụ lập kế hoạch kiểm kê. Hoặc một agent giao dịch. Hoặc vòng lặp PPO huấn luyện một model lý luận. Bốn lĩnh vực khác nhau, một sự thật đáng ngạc nhiên: cả bốn đều sụp đổ thành cùng một đối tượng toán học.

Học có giám sát cung cấp cho bạn `(x, y)` cặp và yêu cầu bạn phù hợp với một chức năng. Học tăng cường không cung cấp cho bạn nhãn - chỉ có một luồng trạng thái, hành động bạn đã thực hiện và phần thưởng vô hướng. Nước đi có thắng trò chơi không? Quyết định bổ sung có tiết kiệm tiền không? Giao dịch có kiếm được lợi nhuận không? Sự token LLM vừa tạo ra có dẫn đến phần thưởng cao hơn từ giám khảo không?

Bạn không thể học hỏi từ luồng này cho đến khi bạn chính thức hóa nó. "Những gì tôi đã thấy", "những gì tôi đã làm", "những gì đã xảy ra tiếp theo", "điều đó tốt như thế nào" - mỗi thứ phải trở thành một đối tượng mà bạn có thể lý luận. Chính thức hóa đó là một Process của Markov Decision. Mọi thuật toán RL trong giai đoạn này, bao gồm cả vòng lặp RLHF và GRPO ở cuối, đều tối ưu hóa hình dạng này.

## Khái niệm

![Markov decision process: states, actions, transitions, rewards, discount](../assets/mdp.svg)

**Năm đối tượng.**

- **Trạng thái** `S`. Mọi thứ agent cần quyết định. Trong GridWorld, ô. Trong cờ vua, bàn cờ. Trong một LLM, context window cộng với bất kỳ bộ nhớ nào.
- **Hành động** `A`. Các lựa chọn. Di chuyển up/down/left/right. chơi một nước đi. Phát ra một token.
- **Chuyển tiếp** `P(s' | s, a)`. Cho trạng thái `s` và hành động `a`, phân phối trên trạng thái tiếp theo. Xác định trong cờ vua, ngẫu nhiên trong kiểm kê, gần như xác định trong giải mã LLM.
- **Phần thưởng** `R(s, a, s')`. Tín hiệu vô hướng. Thắng = +1, loss = -1. Doanh thu trừ chi phí. Thuật ngữ tỷ lệ log-likelihood tính bằng GRPO.
- **Giảm giá** `γ ∈ [0, 1)`. Phần thưởng trong tương lai được tính bao nhiêu so với hiện tại. `γ = 0.99` mua chân trời ~100 bước; `γ = 0.9` mua ~10.

**Thuộc tính Markov **`P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, …, s_t, a_t)`. Tương lai chỉ phụ thuộc vào trạng thái hiện tại. Nếu không, biểu diễn trạng thái là không đầy đủ - không phải là một thất bại của phương pháp, một thất bại của trạng thái.

**Policies và trả về.** Một policy `π(a | s)` ánh xạ các trạng thái với phân phối hành động. Lợi nhuận `G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …` là tổng chiết khấu của phần thưởng trong tương lai. Giá trị `V^π(s) = E[G_t | s_t = s]` là lợi nhuận dự kiến bắt đầu từ `s` dưới policy `π`. `Q^π(s, a) = E[G_t | s_t = s, a_t = a]` giá trị Q là lợi nhuận dự kiến bắt đầu bằng một hành động cụ thể. Mỗi thuật toán RL ước tính một trong hai thuật toán này, sau đó cải thiện `π` cho phù hợp.

**Phương trình Bellman.** Các phương trình điểm cố định mà mọi thứ trong giai đoạn này sử dụng:

`V^π(s) = Σ_a π(a|s) Σ_{s', r} P(s', r | s, a) [r + γ V^π(s')]`
`Q^π(s, a) = Σ_{s', r} P(s', r | s, a) [r + γ Σ_{a'} π(a'|s') Q^π(s', a')]`

Những điều này chia lợi nhuận kỳ vọng thành "phần thưởng của bước này" cộng với "giá trị chiết khấu của nơi bạn hạ cánh". Đệ quy. Mọi thuật toán trong Giai đoạn 9 đều lặp lại phương trình này thành hội tụ (lập trình động), lấy mẫu từ nó (Monte Carlo) hoặc khởi động nó một bước (chênh lệch thời gian).

```figure
discount-horizon
```

## Tự xây dựng

### Bước 1: một MDP xác định nhỏ

Thế giới lưới 4×4. Agent bắt đầu từ trên cùng bên trái, thiết bị đầu cuối ở dưới cùng bên phải, phần thưởng -1 mỗi bước, hành động `{up, down, left, right}`. Xem `code/main.py`.

```python
GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL
```

Năm dòng. Đó là toàn bộ môi trường. Chuyển tiếp xác định, hình phạt bước không đổi, hấp thụ trạng thái cuối.

### Bước 2: triển khai policy

Một policy là một hàm từ trạng thái đến phân phối hành động. Đơn giản nhất: ngẫu nhiên đồng nhất.

```python
def uniform_policy(state):
    return {a: 0.25 for a in ACTIONS}

def rollout(policy, max_steps=200):
    s, total, steps = (0, 0), 0.0, 0
    for _ in range(max_steps):
        a = sample(policy(s))
        s, r, done = step(s, a)
        total += r
        steps += 1
        if done:
            break
    return total, steps
```

Chạy policy ngẫu nhiên 1000 lần. Lợi nhuận trung bình là khoảng -60 đến -80 cho bảng 4×4 này. Lợi nhuận tối ưu là -6 (đường thẳng xuống bên phải). Thu hẹp khoảng cách đó là tất cả mọi thứ trong Giai đoạn 9.

### Bước 3: tính `V^π` chính xác thông qua phương trình Bellman

Đối với các MDP nhỏ, phương trình Bellman là một hệ thống tuyến tính. Liệt kê các trạng thái, áp dụng kỳ vọng, lặp lại cho đến khi các giá trị ngừng thay đổi.

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in all_states()}
    while True:
        delta = 0.0
        for s in all_states():
            if s == TERMINAL:
                continue
            v = 0.0
            for a, pi_a in policy(s).items():
                s_next, r, _ = step(s, a)
                v += pi_a * (r + gamma * V[s_next])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

Đây là đánh giá policy lặp đi lặp lại. Đây là thuật toán đầu tiên trong Sutton & Barto và là nền tảng lý thuyết của mọi phương pháp RL tiếp theo.

### Bước 4: `γ` là một hyperparameter có ý nghĩa vật lý

Chân trời hiệu quả là khoảng `1 / (1 - γ)`. `γ = 0.9` → 10 bước. `γ = 0.99` → 100 bước. `γ = 0.999` → 1000 bước.

Quá thấp và agent hành động thiển cận. Quá cao và việc phân bổ tín dụng trở nên ồn ào, bởi vì nhiều bước đầu chia sẻ trách nhiệm về phần thưởng trong tương lai xa. LLM RLHF thường sử dụng `γ = 1` vì episodes ngắn và có giới hạn. Kiểm soát nhiệm vụ sử dụng `0.95–0.99`. Trò chơi chiến lược dài hạn sử dụng `0.999`.

## Cạm bẫy

- **Trạng thái không phải Markovian.** Nếu bạn cần ba quan sát cuối cùng để quyết định, "trạng thái" không chỉ là quan sát hiện tại. Khắc phục: stack khung hình (DQN trên Atari stacks 4) hoặc sử dụng trạng thái lặp lại (LSTM/GRU trên các quan sát).
- **Phần thưởng thưa thớt.** Phần thưởng chỉ giành chiến thắng khiến việc học gần như không thể trong không gian trạng thái lớn. Phần thưởng định hình (tín hiệu trung gian) hoặc bootstrap với bắt chước (Giai đoạn 9 · 09).
- **Hack phần thưởng.** Tối ưu hóa phần thưởng proxy thường tạo ra hành vi bệnh hoạn. agent đua thuyền của OpenAI quay vòng tròn thu thập sức mạnh mãi mãi thay vì kết thúc cuộc đua. Luôn xác định phần thưởng từ kết quả mục tiêu, không phải proxy.
- **Giảm giá sai thông số kỹ thuật.** `γ = 1` vào một nhiệm vụ chân trời vô hạn làm cho mọi giá trị trở nên vô hạn. Luôn giới hạn bằng đường chân trời hữu hạn hoặc `γ < 1`.
- **Thang phần thưởng.** Phần thưởng {+100, -100} so với {+1, -1} cho policies tối ưu giống hệt nhau nhưng độ lớn gradient rất khác nhau. Chuẩn hóa thành `[-1, 1]` trước khi cắm vào PPO/DQN.

## Ứng dụng

stack 2026 giảm mỗi RL pipeline xuống MDP trước khi chạm vào mã:

| Tình huống | Tiểu bang | Hoạt động | Phần thưởng | γ |
|-----------|-------|--------|--------|---|
| Điều khiển (di chuyển, thao tác) | Góc khớp + vận tốc | Mô-men xoắn liên tục | Định hình cụ thể cho nhiệm vụ | 0.99 |
| Trò chơi (cờ vua, cờ vây, poker) | Hội đồng quản trị + lịch sử | Di chuyển hợp pháp | Thắng = + 1 / loss = -1 | 1.0 (hữu hạn) |
| Hàng tồn kho / giá cả | Tồn kho + nhu cầu | Số lượng đặt hàng | Doanh thu - chi phí | 0.95 |
| RLHF cho LLMs | Bối cảnh tokens | token tiếp theo | Điểm model phần thưởng khi kết thúc | 1.0 (episode ~200 tokens) |
| GRPO lý luận | Prompt + phản hồi một phần | token tiếp theo | Trình xác minh 0/1 ở cuối | 1.0 |

Viết năm bộ trước khi viết bất kỳ vòng lặp training nào. Hầu hết các báo cáo lỗi "RL không hoạt động" trace trở lại công thức MDP đã bị hỏng trên giấy.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-mdp-modeler.md`:

```markdown
---
name: mdp-modeler
description: Given a task description, produce a Markov Decision Process spec and flag formulation risks before training.
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

Given a task (control / game / recommendation / LLM fine-tuning), output:

1. State. Exact feature vector or tensor spec. Justify Markov property.
2. Action. Discrete set or continuous range. Dimensionality.
3. Transition. Deterministic, stochastic-with-known-model, or sample-only.
4. Reward. Function and source. Sparse vs shaped. Terminal vs per-step.
5. Discount. Value and horizon justification.

Refuse to ship any MDP where the state is non-Markovian without explicit mention of frame-stacking or recurrent state. Refuse any reward that was not defined in terms of the target outcome. Flag any `γ ≥ 1.0` on an infinite-horizon task. Flag any reward range >100x the typical step reward as a likely gradient-explosion source.
```

## Bài tập

1. **Dễ dàng.** Triển khai 4×4 GridWorld và policy rollout ngẫu nhiên trong `code/main.py`. Chạy 10.000 episodes. Báo cáo giá trị trung bình và chuẩn lợi nhuận. So sánh với lợi nhuận tối ưu (-6).
2. **Trung bình.** Chạy `policy_evaluation` với `γ ∈ {0.5, 0.9, 0.99}` cho policy ngẫu nhiên đồng nhất. In `V` dưới dạng lưới 4×4 cho mỗi cái. Giải thích lý do tại sao các giá trị trạng thái gần thiết bị đầu cuối tăng nhanh hơn với `γ` lớn hơn.
3. **Khó.** Biến ngẫu nhiên GridWorld: mỗi hành động trượt sang một hướng liền kề với xác suất `p = 0.1`. Đánh giá lại policy đồng phục. `V[start]` trở nên tốt hơn hay tệ hơn? Tại sao?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| MDP | "Thiết lập học tăng cường" | Tuple `(S, A, P, R, γ)` thỏa mãn tài sản Markov. |
| Tiểu bang | "Những gì agent nhìn thấy" | Thống kê đầy đủ cho các động lực trong tương lai theo policy class đã chọn. |
| Policy | "Hành vi của Agent" | Phân phối có điều kiện 'π(a \ | s) ` or deterministic map `s → a'. |
| Trở về | "Tổng phần thưởng" | Số tiền chiết khấu `Σ γ^t r_t` từ bước hiện tại. |
| Giá trị | "Một nhà nước tốt như thế nào" | Lợi nhuận dự kiến dưới `π` bắt đầu từ `s`. |
| Giá trị Q | "Một hành động tốt như thế nào" | Lợi nhuận dự kiến dưới `π` bắt đầu từ `s` với hành động đầu tiên `a`. |
| Phương trình Bellman | "Đệ quy lập trình động" | Phân tách điểm cố định của giá trị / Q thành phần thưởng một bước cộng với giá trị kế thừa chiết khấu. |
| Giảm giá `γ` | "Tương lai vs hiện tại" | Trọng lượng hình học về phần thưởng trong tương lai xa; `~1/(1-γ)` chân trời hiệu quả. |

## Đọc thêm

- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf) - sách giáo khoa. Ch. 3 bao gồm MDP và phương trình Bellman; Ch. 1 thúc đẩy giả thuyết phần thưởng làm nền tảng cho mỗi bài học tiếp theo.
- [Bellman (1957). Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) — nguồn gốc của phương trình Bellman.
- [OpenAI Spinning Up — Part 1: Key Concepts](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) - sơn lót MDP ngắn gọn từ góc RL sâu.
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — tài liệu tham khảo nghiên cứu hoạt động về MDP và các phương pháp giải pháp chính xác.
- [Littman (1996). Algorithms for Sequential Decision Making (PhD thesis)](https://www.cs.rutgers.edu/~mlittman/papers/thesis-main.pdf) — dẫn xuất rõ ràng nhất của MDP như một chuyên ngành lập trình động.
