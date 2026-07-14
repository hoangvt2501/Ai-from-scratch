# Sự khác biệt về thời gian - Q-Learning & SARSA

> Monte Carlo đợi cho đến khi episode kết thúc. TD cập nhật sau mỗi bước bằng cách khởi động ước tính giá trị tiếp theo. Q-learning không policy và lạc quan; SARSA policy và thận trọng. Cả hai đều là một dòng mã. Cả hai đều củng cố mọi phương pháp RL sâu trong giai đoạn này.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 9 · 01 (MDP), Giai đoạn 9 · 02 (Lập trình động), Giai đoạn 9 · 03 (Monte Carlo)
**Thời lượng:** ~75 phút

## Vấn đề

Monte Carlo hoạt động nhưng nó có hai yêu cầu đắt đỏ. Nó cần episodes chấm dứt và nó chỉ cập nhật sau khi hoàn trả cuối cùng. Nếu episode của bạn là 1.000 bước, MC đợi 1.000 bước để cập nhật bất cứ thứ gì. Nó có variance cao, bias thấp và chậm trong thực tế.

Lập trình động có cấu hình ngược lại - các bản sao lưu khởi động variance không - nhưng yêu cầu một model đã biết.

Học sự khác biệt thời gian (TD) phân chia sự khác biệt. Từ một `(s, a, r, s')` chuyển tiếp duy nhất, hãy hình thành một `r + γ V(s')` mục tiêu một bước và thúc đẩy `V(s)` hướng tới nó. Không có model. Không có episodes hoàn chỉnh. Bias sử dụng `V` gần đúng trên RHS, nhưng variance thấp hơn đáng kể so với MC và cập nhật trực tuyến từ bước một.

Đây là trục mà tất cả các RL hiện đại - DQN, A2C, PPO, SAC - xoay quanh. Điểm rest của Giai đoạn 9 là các lớp xấp xỉ hàm và thủ thuật được xây dựng dựa trên bản cập nhật TD một bước mà bạn sẽ viết trong bài học này.

## Khái niệm

![Q-learning vs SARSA: off-policy max vs on-policy Q(s', a')](../assets/td.svg)

**Bản cập nhật TD (0) cho V:**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

Số lượng trong ngoặc đơn là lỗi TD `δ = r + γ V(s') - V(s)`. Nó là tương tự trực tuyến của `G_t - V(s_t)` trong MC. Hội tụ đòi hỏi `α` thỏa mãn Robbins-Monro (`Σ α = ∞`, `Σ α² < ∞`) và tất cả các tiểu bang đều ghé thăm vô hạn thường xuyên.

**Q-learning.** Một phương pháp TD ngoài policy để điều khiển:

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

`max` giả định policy * tham lam * sẽ được theo dõi từ `s'` trở đi, bất kể agent thực sự thực hiện hành động nào. Sự tách rời đó làm cho Q-learning học `Q*` trong khi agent khám phá thông qua ε tham lam. Mnih et al. (2015) đã chuyển đổi điều này thành Q-learning sâu trên Atari (Bài 05).

**SARSA.** Phương pháp TD trên policy:

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

Tên là bộ `(s, a, r, s', a')`. SARSA sử dụng hành động `a'` agent *thực sự* thực hiện tiếp theo, không phải `argmax` tham lam. Hội tụ để `Q^π` cho bất kỳ `π` tham lam ε nào đang chạy, trong giới hạn `ε → 0` trở thành `Q*`.

**Sự khác biệt đi bộ trên vách đá.** Trong nhiệm vụ đi bộ trên vách đá cổ điển (rơi khỏi vách đá = phần thưởng -100), Q-learning học con đường tối ưu dọc theo mép vách đá nhưng đôi khi bị phạt trong quá trình khám phá. SARSA học một con đường an toàn hơn cách vách đá một bước vì nó yếu tố nhiễu thăm dò vào giá trị Q của nó. Với training, cả hai đều đạt mức tối ưu ở `ε → 0`. Trong thực tế, điều quan trọng: khi khám phá thực sự diễn ra khi triển khai, hành vi của SARSA thận trọng hơn.

**SARSA dự kiến.** Thay thế `Q(s', a')` bằng giá trị dự kiến của nó dưới `π`:

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

variance thấp hơn SARSA (không có mẫu `a'`), cùng mục tiêu trên policy. Thường là mặc định trong sách giáo khoa hiện đại.

**TD và TD (λ) bước n.** Nội suy giữa TD (0) và MC bằng cách đợi `n` bước trước khi khởi động. `n=1` là TD, `n=∞` là MC. TD (λ) trung bình trên tất cả các `n` với trọng số hình học `(1-λ)λ^{n-1}`. Hầu hết các RL sâu sử dụng `n` từ 3 đến 20.

```figure
qlearning-gridworld
```

## Tự xây dựng

### Bước 1: SARSA đối với policy tham lam ε

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

Tám dòng. Sự khác biệt *duy nhất* so với Q-learning là đường mục tiêu.

### Bước 2: Q-learning

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

Các `max` tách mục tiêu khỏi hành vi. Một biểu tượng đó là sự khác biệt giữa on-policy và off-policy.

### Bước 3: đường cong học tập

Theo dõi lợi nhuận trung bình trên 100 episodes. Q-learning hội tụ nhanh hơn trên GridWorld xác định đơn giản; SARSA bảo thủ hơn trong việc đi bộ trên vách đá. Trên 4×4 GridWorld ở `code/main.py`, cả hai đều gần như tối ưu sau ~2.000 episodes với `α=0.1, ε=0.1`.

### Bước 4: so sánh với sự thật của DP

Chạy lặp lại giá trị (Bài 02) để nhận `Q*`. Kiểm tra `max_{s,a} |Q_learned(s,a) - Q*(s,a)|`. Một TD agent dạng bảng khỏe mạnh sẽ hạ cánh trong vòng `~0.5` trên GridWorld 4×4 sau 10.000 episodes.

## Cạm bẫy

- **Giá trị Q ban đầu rất quan trọng.** Khởi đầu lạc quan (`Q = 0` cho một nhiệm vụ phần thưởng tiêu cực) khuyến khích khám phá. Khởi đầu bi quan có thể bẫy một policy tham lam mãi mãi.
- **α lịch trình. **`α` không đổi là tốt cho các vấn đề không tĩnh. Phân rã `α_n = 1/n` mang lại sự hội tụ về lý thuyết nhưng quá chậm trong thực tế - ghim `α` trong `[0.05, 0.3]` và theo dõi đường cong học tập.
- **ε lịch trình. **Bắt đầu cao (`ε=1.0`), phân rã đến `ε=0.05`. "GLIE" (tham lam trong giới hạn với khám phá vô hạn) là điều kiện hội tụ.
- **Tối đa bias trong Q-learning.** Toán tử `max` bị lệch lên trên khi `Q` bị ồn. Dẫn đến đánh giá quá cao - Double Q-learning của Hasselt (được DDQN sử dụng trong Bài 05) khắc phục điều này với hai bảng Q.
- **episodes không kết thúc. **TD có thể học mà không cần thiết bị đầu cuối, nhưng bạn cần phải giới hạn các bước hoặc xử lý dây đeo khởi động một cách chính xác ở nắp. Tiêu chuẩn: coi nắp là không đầu cuối, tiếp tục khởi động.
- **Trạng thái băm.** Nếu trạng thái tuples/tensors, hãy sử dụng khóa có thể băm (bộ dữ liệu, không phải danh sách; bộ số float được làm tròn, không phải thô).

## Ứng dụng

Bối cảnh TD năm 2026:

| Nhiệm vụ | Phương pháp | Lý do |
|------|--------|--------|
| Môi trường dạng bảng nhỏ | Học Q | Học policy tối ưu trực tiếp. |
| An toàn trên policy quan trọng | SARSA / SARSA dự kiến | Bảo thủ trong quá trình khám phá. |
| High-dimensional trạng thái | DQN (Giai đoạn 9 · 05) | Chức năng Neural-net Q với phát lại và mạng mục tiêu. |
| Hành động liên tục | SAC / TD3 (Giai đoạn 9 · 07) | Cập nhật TD trên Q-network; policy net phát ra hành động. |
| LLM RL (dựa trên model phần thưởng) | PPO / GRPO (Giai đoạn 9 · 08, 12) | Diễn viên-nhà phê bình với lợi thế theo phong cách TD thông qua GAE. |
| RL ngoại tuyến | CQL / IQL (Giai đoạn 9 · 08) | Q-learning với quy tắc hóa bảo thủ. |

Chín mươi phần trăm "RL" bạn đọc trong các bài báo năm 2026 là một số công phu của Q-learning hoặc SARSA. Hiểu bản cập nhật dạng bảng trong ngón tay của bạn trước khi đọc sâu hơn.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-td-agent.md`:

```markdown
---
name: td-agent
description: Pick between Q-learning, SARSA, Expected SARSA for a tabular or small-feature RL task.
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

Given a tabular or small-feature environment, output:

1. Algorithm. Q-learning / SARSA / Expected SARSA / n-step variant. One-sentence reason tied to on-policy vs off-policy and variance.
2. Hyperparameters. α, γ, ε, decay schedule.
3. Initialization. Q_0 value (optimistic vs zero) and justification.
4. Convergence diagnostic. Target learning curve, `|Q - Q*|` check if DP is possible.
5. Deployment caveat. How will exploration behave at inference? Is SARSA's conservatism needed?

Refuse to apply tabular TD to state spaces > 10⁶. Refuse to ship a Q-learning agent without a max-bias caveat. Flag any agent trained with ε held at 1.0 throughout (no exploitation phase).
```

## Bài tập

1. **Dễ dàng.** Triển khai Q-learning và SARSA trên GridWorld 4×4. Vẽ đường cong học tập (lợi nhuận trung bình trên 100 episodes) cho 2.000 episodes. Ai hội tụ nhanh hơn?
2. **Trung bình.** Xây dựng môi trường đi bộ trên vách đá (4×12, hàng cuối cùng là vách đá với phần thưởng -100 và đặt lại để bắt đầu). So sánh Q-learning và SARSA policies cuối cùng. Chụp màn hình các con đường mà mỗi người đi. Cái nào gần vách đá hơn?
3. **Khó.** Thực hiện Double Q-learning. Trên GridWorld phần thưởng ồn ào (nhiễu Gaussian σ = 5 được thêm vào phần thưởng mỗi bước), cho thấy Q-learning đánh giá quá cao `V*(0,0)` một lượng có ý nghĩa trong khi Double Q-learning thì không.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Lỗi TD | "Tín hiệu cập nhật" | `δ = r + γ V(s') - V(s)`, phần còn lại đã khởi động. |
| TD(0) | "TD một bước" | Cập nhật sau mỗi lần chuyển đổi chỉ bằng cách sử dụng ước tính của trạng thái tiếp theo. |
| Học Q | "Off-policy RL 101" | Cập nhật TD với `max` qua các hành động trạng thái tiếp theo; học `Q*` bất kể hành vi policy. |
| SARSA | "Học Q trên policy" | Cập nhật TD bằng cách sử dụng hành động tiếp theo thực tế; học `Q^π` cho π tham lam ε hiện tại. |
| SARSA dự kiến | "SARSA variance thấp" | Thay thế `a'` lấy mẫu với kỳ vọng của nó dưới π. |
| GLIE | "Lịch trình thăm dò chính xác" | Tham lam trong giới hạn với khám phá vô hạn; cần thiết cho sự hội tụ Q-learning. |
| Khởi động | "Sử dụng ước tính hiện tại trong mục tiêu" | Điều gì phân biệt TD với MC. Nguồn bias nhưng giảm variance lớn. |
| Tối đa hóa bias | "Q-learning đánh giá quá cao" | `max` trên các ước tính ồn ào là thiên vị đi lên; được cố định bằng Double Q-learning. |

## Đọc thêm

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — giấy gốc và bằng chứng hội tụ.
- [Sutton & Barto (2018). Ch. 6 — Temporal-Difference Learning](http://incompleteideas.net/book/RLbook2020.pdf) - TD (0), SARSA, Q-learning, SARSA dự kiến.
- [Hasselt (2010). Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) - sửa chữa để tối đa hóa bias.
- [Seijen, Hasselt, Whiteson, Wiering (2009). A Theoretical and Empirical Analysis of Expected SARSA](https://ieeexplore.ieee.org/document/4927542) - động lực SARSA dự kiến.
- [Rummery & Niranjan (1994). On-line Q-learning using connectionist systems](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) - bài báo đặt ra SARSA (sau đó được gọi là "học Q-learning kết nối sửa đổi").
- [Sutton & Barto (2018). Ch. 7 — n-step Bootstrapping](http://incompleteideas.net/book/RLbook2020.pdf) - khái quát hóa TD (0) thành TD (n), con đường từ Q-learning đến tính đủ điều kiện traces và sau đó là GAE trong PPO.
