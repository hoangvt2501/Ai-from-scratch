# Lập trình động - Lặp lại Policy và lặp lại giá trị

> Lập trình động RL với gian lận. Bạn đã biết các chức năng chuyển đổi và phần thưởng; bạn chỉ cần lặp lại phương trình Bellman cho đến khi `V` hoặc `π` ngừng di chuyển. Đó là benchmark mà mọi phương pháp dựa trên sampling đều cố gắng tiếp cận.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 9 · 01 (MDP)
**Thời lượng:** ~75 phút

## Vấn đề

Bạn có một MDP với một model đã biết: bạn có thể truy vấn `P(s' | s, a)` và `R(s, a, s')` cho bất kỳ cặp hành động trạng thái nào. Một người quản lý hàng tồn kho biết phân phối nhu cầu. Một trò chơi trên bàn cờ có các chuyển đổi xác định. Một thế giới lưới là bốn dòng Python. Bạn có một *model*.

RL không có Model (Q-learning, PPO, REINFORCE) được phát minh cho trường hợp bạn không có model - bạn chỉ có thể lấy mẫu từ môi trường. Nhưng khi bạn có một phương pháp, có những phương pháp nhanh hơn, tốt hơn: lập trình động. Bellman đã thiết kế chúng vào năm 1957. Họ vẫn định nghĩa tính đúng đắn: khi mọi người nói "policy tối ưu cho MDP này", họ có nghĩa là DP policy sẽ quay trở lại.

Bạn cần chúng vào năm 2026 vì ba lý do. Thứ nhất, mọi môi trường dạng bảng trong nghiên cứu RL (GridWorld, FrozenLake, CliffWalking) đều được giải quyết bằng DP để tạo ra policy tiêu chuẩn vàng. Thứ hai, các giá trị chính xác cho phép bạn *gỡ lỗi* các phương thức sampling: nếu ước tính của Q-learning cho `V*(s_0)` không đồng ý với câu trả lời DP 30%, Q-learning của bạn có lỗi. Thứ ba, các phương pháp lập kế hoạch và RL ngoại tuyến hiện đại (MCTS, tìm kiếm của AlphaZero, RL dựa trên model trong Giai đoạn 9 · 10) tất cả đều lặp lại một bản sao lưu Bellman trên một model đã học hoặc đã cho.

## Khái niệm

![Policy iteration and value iteration, side by side](../assets/dp.svg)

**Hai thuật toán, cả hai đều lặp lại điểm cố định trên Bellman.

**Policy lần lặp.** Xen kẽ hai bước cho đến khi policy ngừng thay đổi.

1. *Evaluation:* đưa ra policy `π`, hãy tính toán `V^π` bằng cách áp dụng nhiều lần `V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ V(s')]` cho đến khi hội tụ.
2. *Improvement:* được `V^π`, hãy làm cho `π` w.r.t. tham lam `V^π`: `π(s) ← argmax_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`.

Sự hội tụ được đảm bảo bởi vì (a) mỗi bước cải tiến giữ nguyên `π` hoặc tăng nghiêm ngặt `V^π` đối với một số trạng thái, (b) không gian của policies xác định là hữu hạn. Thường hội tụ trong ~5–20 lần lặp bên ngoài ngay cả đối với không gian trạng thái lớn.

**Lặp lại giá trị.** Thu gọn đánh giá và cải tiến thành một lần quét. Áp dụng phương trình Bellman *tối ưu*:

`V(s) ← max_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`

Lặp lại cho đến khi `max_s |V_{new}(s) - V(s)| < ε`. Trích xuất policy ở cuối bằng cách thực hiện hành động tham lam. Nhanh hơn cho mỗi lần lặp lại - không có vòng lặp đánh giá bên trong - nhưng thường cần nhiều lần lặp lại hơn để hội tụ.

**Vòng lặp policy tổng quát (GPI).** Khung thống nhất. Chức năng giá trị và policy bị khóa trong vòng lặp cải tiến hai chiều; bất kỳ phương pháp nào thúc đẩy cả hai hướng tới tính nhất quán lẫn nhau (lặp lại giá trị không đồng bộ, lặp lại policy sửa đổi, Q-learning, actor-critic PPO) đều là một ví dụ của GPI.

**Tại sao `γ < 1` lại quan trọng.** Toán tử Bellman là một sự rút gọn `γ` trong định mức trên: `||T V - T V'||_∞ ≤ γ ||V - V'||_∞`. Sự co lại ngụ ý điểm cố định duy nhất và sự hội tụ hình học. Thả `γ < 1` và bạn mất đảm bảo - bạn cần một đường chân trời hữu hạn hoặc một trạng thái đầu cuối hấp thụ.

```figure
value-iteration-gamma
```

## Tự xây dựng

### Bước 1: xây dựng GridWorld MDP model

Sử dụng cùng một GridWorld 4×4 từ Bài 01. Chúng ta thêm một biến thể ngẫu nhiên: với xác suất `0.1` agent trượt theo một hướng vuông góc ngẫu nhiên.

```python
SLIP = 0.1

def transitions(state, action):
    if state == TERMINAL:
        return [(state, 0.0, 1.0)]
    outcomes = []
    for direction, prob in action_probs(action):
        outcomes.append((apply_move(state, direction), -1.0, prob))
    return outcomes
```

`transitions(s, a)` trả về danh sách `(s', r, p)`. Đây là toàn bộ model.

### Bước 2: Đánh giá policy

Cho một policy `π(s) = {action: prob}`, lặp lại phương trình Bellman cho đến khi `V` ngừng di chuyển:

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = sum(pi_a * sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a))
                   for a, pi_a in policy(s).items())
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

### Bước 3: Cải tiến policy

Thay thế `π` bằng `V` policy tham lam. Nếu `π` không thay đổi, hãy quay lại - chúng tôi đang ở mức tối ưu.

```python
def policy_improvement(V, gamma=0.99):
    new_policy = {}
    for s in states():
        best_a = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a)),
        )
        new_policy[s] = best_a
    return new_policy
```

### Bước 4: khâu chúng lại với nhau

```python
def policy_iteration(gamma=0.99):
    policy = {s: "up" for s in states()}   # arbitrary start
    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy
```

Hội tụ điển hình trên 4×4: 4–6 lần lặp bên ngoài. Đầu ra `V*(0,0) ≈ -6` và một policy giảm nghiêm ngặt số bước.

### Bước 5: lặp lại giá trị (phiên bản một vòng lặp)

```python
def value_iteration(gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = max(sum(p * (r + gamma * V[s_prime])
                       for s_prime, r, p in transitions(s, a))
                   for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    policy = policy_improvement(V, gamma)
    return V, policy
```

Cùng một điểm cố định, ít dòng mã hơn.

## Cạm bẫy

- **Quên xử lý các thiết bị đầu cuối.** Nếu bạn áp dụng Bellman ở trạng thái hấp thụ, nó vẫn chọn một "hành động tốt nhất" mà không thay đổi gì. Bảo vệ với `if s == terminal: V[s] = 0`.
- **Sup-norm so với hội tụ L2.** Sử dụng `max |V_new - V|`, không phải trung bình. Đảm bảo lý thuyết nằm trên sup-norm.
- **Cập nhật tại chỗ và đồng bộ.** Cập nhật `V[s]` tại chỗ (Gauss-Seidel) hội tụ nhanh hơn một `V_new` riêng biệt (Jacobi). Mã Production sử dụng tại chỗ.
- **Policy hòa.** Nếu hai hành động có giá trị Q bằng nhau, `argmax` có thể phá vỡ các mối quan hệ khác nhau mỗi lần lặp, khiến kiểm tra "policy ổn định" dao động. Sử dụng tie-break ổn định (hành động đầu tiên theo thứ tự cố định).
- **Vụ nổ không gian trạng thái.** DP là `O(|S| · |A|)` mỗi lần quét. Hoạt động lên đến ~10⁷ trạng thái. Ngoài ra, bạn cần xấp xỉ hàm (Giai đoạn 9 · 05 trở đi).

## Ứng dụng

Vào năm 2026, DP là đường cơ sở về tính đúng đắn và vòng lặp bên trong của các nhà lập kế hoạch:

| Trường hợp sử dụng | Phương pháp |
|----------|--------|
| Giải chính xác một MDP dạng bảng nhỏ | Lặp lại giá trị (đơn giản hơn) hoặc lặp lại policy (ít bước bên ngoài hơn) |
| Xác minh việc triển khai Q-learning / PPO | So sánh với DP-tối ưu V* trên môi trường đồ chơi |
| RL dựa trên Model (Giai đoạn 9 · 10) | Bellman sao lưu trên một model chuyển tiếp đã học |
| Lập kế hoạch trong AlphaZero / MuZero | Monte Carlo Tree Search = sao lưu Bellman không đồng bộ |
| RL ngoại tuyến (CQL, IQL) | Lặp lại Q thận trọng — DP với hình phạt đối với các hành động OOD |

Mỗi khi ai đó nói "hàm giá trị tối ưu", họ có nghĩa là "điểm cố định DP". Khi bạn nhìn thấy `V*` hoặc `Q*` trong một bài báo, hãy hình dung vòng lặp này.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-dp-solver.md`:

```markdown
---
name: dp-solver
description: Solve a small tabular MDP exactly via policy iteration or value iteration. Report convergence behavior.
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

Given an MDP with a known model, output:

1. Choice. Policy iteration vs value iteration. Reason tied to |S|, |A|, γ.
2. Initialization. V_0, starting policy. Convergence sensitivity.
3. Stopping. Sup-norm tolerance ε. Expected number of sweeps.
4. Verification. V*(s_0) computed exactly. Greedy policy extracted.
5. Use. How this baseline will be used to debug/evaluate sampling-based methods.

Refuse to run DP on state spaces > 10⁷. Refuse to claim convergence without a sup-norm check. Flag any γ ≥ 1 on an infinite-horizon task as a guarantee violation.
```

## Bài tập

1. **Dễ dàng.** Chạy lặp lại giá trị trên 4×4 GridWorld với `γ ∈ {0.9, 0.99}`. Bao nhiêu lần quét cho đến `max |ΔV| < 1e-6`? In `V*` dưới dạng lưới 4×4.
2. **Trung bình.** So sánh policy lặp lại so với lặp lại giá trị trên *stochastic* GridWorld (xác suất trượt `0.1`). Đếm: quét, thời gian đồng hồ treo tường, `V*(0,0)` cuối cùng. Cái nào hội tụ nhanh hơn trong các lần lặp? Trong đồng hồ treo tường?
3. **Khó.** Xây dựng sửa đổi policy lặp lại: trong bước đánh giá, chỉ chạy `k` lần quét thay vì hội tụ. Vẽ lỗi `V*(0,0)` so với `k` cho `k ∈ {1, 2, 5, 10, 50}`. Đường cong cho bạn biết gì về sự đánh đổi evaluation/improvement?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Policy lặp lại | "Thuật toán DP" | Đánh giá xen kẽ (`V^π`) và cải tiến (tham lam `π` wrt `V^π`) cho đến khi policy ngừng thay đổi. |
| Lặp lại giá trị | "DP nhanh hơn" | Dự phòng tối ưu Bellman được áp dụng trong một lần quét; hội tụ để `V*` về mặt hình học. |
| Nhà điều hành Bellman | "Đệ quy" | `(T V)(s) = max_a Σ P (r + γ V(s'))`; một sự co lại `γ` trong định mức trên. |
| Sự co lại | "Tại sao DP hội tụ" | Bất kỳ toán tử nào `T` với '\ | \ | T x - T y \ | \ | ≤ γ \ | \ | x - y\ | \ | ' có một điểm cố định duy nhất. |
| Điểm GPI | "Mọi thứ đều là DP" | Vòng lặp Policy tổng quát: bất kỳ phương pháp nào thúc đẩy `V` và `π` đến sự nhất quán lẫn nhau. |
| Cập nhật đồng bộ | "Phong cách Jacobi" | Sử dụng `V` cũ trong suốt quá trình quét; có thể phân tích rõ ràng nhưng chậm hơn. |
| Cập nhật tại chỗ | "Kiểu Gauss-Seidel" | Sử dụng `V` khi nó đang được cập nhật; hội tụ nhanh hơn trong thực tế. |

## Đọc thêm

- [Sutton & Barto (2018). Ch. 4 — Dynamic Programming](http://incompleteideas.net/book/RLbook2020.pdf) — trình bày chuẩn mực của policy lặp lại và lặp lại giá trị.
- [Bertsekas (2019). Reinforcement Learning and Optimal Control](http://www.athenasc.com/rlbook.html) - xử lý nghiêm ngặt các lập luận lập bản đồ co lại.
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — sửa đổi policy lặp lại và phân tích hội tụ của nó.
- [Howard (1960). Dynamic Programming and Markov Processes](https://mitpress.mit.edu/9780262582300/dynamic-programming-and-markov-processes/) — bài báo lặp lại policy ban đầu.
- [Bertsekas & Tsitsiklis (1996). Neuro-Dynamic Programming](http://www.athenasc.com/ndpbook.html) — cầu nối từ DP đến DP xấp xỉ / RL sâu được sử dụng bởi mọi bài học tiếp theo.
