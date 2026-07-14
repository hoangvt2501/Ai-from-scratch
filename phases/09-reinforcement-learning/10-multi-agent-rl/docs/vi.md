# Đa Agent RL

> Single-agent RL giả định môi trường đứng yên. Đặt hai agents học tập trong cùng một thế giới và giả định đó gặp lỗi: mỗi agent là một phần của môi trường kia và cả hai đều đang thay đổi. Multi-agent RL là tập hợp các thủ thuật để làm cho việc học hội tụ khi giả định Markov không còn giữ nguyên nữa.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 9 · 04 (Q-learning), Giai đoạn 9 · 06 (CỦNG cố), Giai đoạn 9 · 07 (Diễn viên-Nhà phê bình)
**Thời lượng:** ~45 phút

## Vấn đề

Một robot học cách điều hướng một căn phòng là một vấn đề agent RL. Một đội bóng đá thì không. Đối thủ của AlphaStar vs StarCraft thì không. Một thị trường đấu thầu agents thì không. Hai chiếc xe đàm phán một điểm dừng bốn chiều thì không. Nhiều vấn đề trong thế giới thực thì không.

Trong mọi bối cảnh đa agent, từ quan điểm của bất kỳ agent nào, agents người kia *là* một phần của môi trường. Khi họ học hỏi và thay đổi hành vi của mình, môi trường trở nên không đứng yên. Thuộc tính Markov - "trạng thái tiếp theo chỉ phụ thuộc vào trạng thái hiện tại và hành động của tôi" - bị vi phạm bởi vì trạng thái tiếp theo cũng phụ thuộc vào những gì * người khác * agents chọn và policies của họ là các mục tiêu chuyển động.

Điều này phá vỡ các bằng chứng hội tụ dạng bảng (đảm bảo của Q-learning giả định một môi trường tĩnh). Nó cũng phá vỡ RL sâu ngây thơ: agents đuổi theo nhau trong các vòng lặp, không bao giờ hội tụ đến một policy ổn định. Bạn cần các kỹ thuật đa agent cụ thể: training tập trung / thực thi phi tập trung, đường cơ sở phản thực tế, chơi giải đấu, tự chơi.

Ứng dụng năm 2026: swarms robot, định tuyến giao thông, đội xe tự hành, mô phỏng thị trường, hệ thống đa agent LLM (Giai đoạn 16) và bất kỳ trò chơi nào có nhiều hơn một người chơi thông minh.

## Khái niệm

![Four MARL regimes: indep, centralized critic, self-play, league](../assets/marl.svg)

**Chủ nghĩa hình thức: Trò chơi Markov.** Khái quát hóa MDP: các trạng thái `S`, một `a = (a_1, …, a_n)` hành động chung, `P(s' | s, a)` chuyển tiếp và phần thưởng mỗi agent `R_i(s, a, s')`. Mỗi agent `i` tối đa hóa lợi nhuận của chính mình dưới policy `π_i` của riêng mình. Nếu phần thưởng giống hệt nhau, nó là **hoàn toàn hợp tác**. Nếu tổng bằng không, nó là **đối nghịch**. Nếu trộn lẫn, nó là **tổng chung**.

**Thách thức cốt lõi:**

- **Không đứng yên.** `P(s' | s, a_i)` theo quan điểm của agent `i` phụ thuộc vào `π_{-i}`, đang thay đổi.
- **Chỉ định tín dụng.** Với phần thưởng được chia sẻ, agent nào gây ra nó?
- **Phối hợp khám phá.** Agents phải khám phá các chiến lược bổ sung, không phải khám phá dư thừa cùng một trạng thái.
- **Khả năng mở rộng.** Không gian hành động chung phát triển theo cấp số nhân trong `n`.
- **observability một phần.** Mỗi agent chỉ nhìn thấy quan sát của riêng mình; trạng thái toàn cầu bị ẩn.

**Bốn chế độ thống trị:**

**1. Học Q độc lập / PPO độc lập (IQL, IPPO).** Mỗi agent học Q hoặc policy của riêng mình, coi người khác như một phần của môi trường. Đơn giản, đôi khi nó hoạt động (đặc biệt là với trải nghiệm phát lại hoạt động như một thủ thuật mô hình hóa agent làm mịn). Hội tụ lý thuyết: không có. Trong thực tế: tốt cho các nhiệm vụ được ghép nối lỏng lẻo, không tốt cho các nhiệm vụ được ghép nối chặt chẽ.

**2. training tập trung, thực thi phi tập trung (CTDE).** Mô hình hiện đại phổ biến nhất. Mỗi agent có *policy* `π_i` điều kiện `o_i` quan sát cục bộ - thực thi phi tập trung tiêu chuẩn khi triển khai. Trong *training*, một nhà phê bình tập trung `Q(s, a_1, …, a_n)` các điều kiện về trạng thái toàn cầu đầy đủ và hành động chung. Ví dụ:
- **MADDPG** (Lowe et al. 2017): DDPG với một nhà phê bình tập trung mỗi agent.
- **COMA **(Foerster et al. 2017): đường cơ sở phản thực tế - hãy hỏi "phần thưởng của tôi sẽ là gì nếu tôi hành động `a'` thay thế?" - cô lập đóng góp của tôi.
- **MAPPO** / **IPPO** với nhà phê bình chia sẻ (Yu et al. 2022): PPO với chức năng giá trị tập trung. Thống trị vào năm 2026 cho hợp tác xã MARL.
- **QMIX **(Rashid et al. 2018): phân hủy giá trị - `Q_tot(s, a) = f(Q_1(s, a_1), …, Q_n(s, a_n))` với trộn đơn điệu.

**3. Tự phát. **Hai bản sao của cùng một agent chơi với nhau. policy của đối thủ *là* policy của tôi từ ảnh chụp nhanh trong quá khứ. AlphaGo / AlphaZero / MuZero. OpenAI năm. Hoạt động tốt nhất cho các trò chơi có tổng bằng không; tín hiệu training là đối xứng.

**4. Chơi giải đấu.** Mở rộng khả năng tự chơi sang môi trường tổng chung / đối thủ: giữ một quần thể policies trong quá khứ và hiện tại, lấy mẫu đối thủ từ giải đấu, tập luyện chống lại họ. Thêm những kẻ bóc lột (chuyên đánh bại những người giỏi nhất hiện tại) và những kẻ khai thác chính (chuyên đánh bại những kẻ bóc lột). AlphaStar (StarCraft II). Cần thiết khi trò chơi thừa nhận các chu kỳ chiến lược "kéo-téo".

**Giao tiếp.** Cho phép agents gửi các thông điệp đã học `m_i` cho nhau. Hoạt động trong môi trường hợp tác. Foerster et al. (2016) đã chỉ ra rằng giao tiếp giữa các agent có thể phân biệt có thể được huấn luyện từ đầu đến cuối. Các hệ thống đa agent dựa trên LLM ngày nay (Giai đoạn 16) về cơ bản giao tiếp bằng ngôn ngữ tự nhiên.

## Tự xây dựng

Bài học này sử dụng GridWorld 6×6 với hai agents hợp tác. Họ bắt đầu ở các góc đối diện và phải đạt được mục tiêu chung. Phần thưởng được chia sẻ: `-1` mỗi bước trong khi một trong hai agent vẫn đang di chuyển, `+10` khi cả hai đều đến. Xem `code/main.py`.

### Bước 1: môi trường đa agent

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # two agents

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

Không gian hành động chung * là `|A|² = 16`. Trạng thái toàn cầu là hai vị trí.

### Bước 2: Q-learning độc lập

Mỗi agent chạy bảng Q của riêng mình được khóa trên trạng thái chung. Ở mỗi bước: cả hai chọn các hành động tham lam ε, thu thập chuyển đổi chung, mỗi người cập nhật Q của riêng mình với phần thưởng được chia sẻ.

```python
def independent_q(env, episodes, alpha, gamma, epsilon):
    Q1, Q2 = defaultdict(default_q), defaultdict(default_q)
    for _ in range(episodes):
        s = env.reset()
        while not done:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next
```

Làm việc với nhiệm vụ này vì phần thưởng dày đặc và liên kết. Thất bại trong các nhiệm vụ được kết hợp chặt chẽ (ví dụ: khi một người agent phải *đợi* cho người kia).

### Bước 3: Q tập trung với cập nhật giá trị phân hủy

Sử dụng một Q thay vì các hành động chung `Q(s, a_1, a_2)`. Cập nhật từ phần thưởng được chia sẻ. Phân cấp khi thực hiện bằng cách gạt ra bên lề: `π_i(s) = argmax_{a_i} max_{a_{-i}} Q(s, a_1, a_2)`. Giao dịch không gian hành động chung theo cấp số nhân để có một cái nhìn toàn cầu *chính xác*.

### Bước 4: tự chơi đơn giản (đối nghịch 2-agent)

Cùng agent, hai vai trò. Huấn luyện agent A chống lại agent B; sau `K` episodes, sao chép trọng số của A vào B. training đối xứng, tiến trình nhất quán. Công thức AlphaZero thu nhỏ.

## Cạm bẫy

- **Phát lại không tĩnh.** Trải nghiệm phát lại với agents độc lập kém hơn agent đơn vì các chuyển tiếp cũ được tạo ra bởi các đối thủ hiện đã lỗi thời. Khắc phục: gắn nhãn lại hoặc trọng số theo thời gian gần đây.
- **Sự mơ hồ về việc gán tín dụng.** Phần thưởng được chia sẻ sau một episode dài; không có cách nào rõ ràng để nói agent nào đã đóng góp. Khắc phục: đường cơ sở phản thực tế (COMA) hoặc định hình phần thưởng cho mỗi agent.
- **Policy trôi dạt / rượt đuổi. **Phản hồi tốt nhất của mỗi agent thay đổi theo bản cập nhật của nhau. Khắc phục: phê bình tập trung, tốc độ học chậm hoặc đóng băng từng người một.
- **Phần thưởng hack thông qua phối hợp.** Agents tìm thấy các khai thác phối hợp mà nhà thiết kế không lường trước được. Đấu giá agents hội tụ để trả giá bằng không. Khắc phục: thiết kế phần thưởng cẩn thận, ràng buộc hành vi.
- **Dự phòng khám phá.** Cả hai đều agents khám phá cùng một cặp hành động trạng thái. Khắc phục: tiền thưởng entropy trên mỗi agent hoặc điều kiện vai trò.
- **Chu kỳ giải đấu. **Tự chơi thuần túy có thể bị mắc kẹt trong chu kỳ thống trị. Khắc phục: chơi giải đấu với các đối thủ đa dạng.
- **Vụ nổ mẫu.** `n` agents × không gian trạng thái × các hành động chung. Xấp xỉ với xấp xỉ hàm; không gian hành động được phân tích (một policy đầu ra trên mỗi agent).

## Ứng dụng

Bản đồ ứng dụng MARL 2026:

| Tên miền | Phương pháp | Ghi chú |
|--------|--------|-------|
| Điều hướng / thao tác hợp tác | MAPPO / QMIX | CTDE; nhà phê bình chia sẻ + các tác nhân phi tập trung. |
| Trò chơi hai người chơi (cờ vua, cờ vây, poker) | Tự phát với MCTS (AlphaZero) | Tổng bằng không; training đối xứng. |
| Nhiều người chơi phức tạp (Dota, StarCraft) | Chơi giải đấu + pretraining bắt chước | OpenAI năm, AlphaStar. |
| Đội xe tự hành | CTDE MAPPO / PPO với attention | Một phần obs; quy mô đội thay đổi. |
| Thị trường đấu giá | Cân bằng lý thuyết trò chơi + RL | RL trường trung bình khi `n` → ∞. |
| LLM hệ thống đa agent (Giai đoạn 16) | Giao tiếp ngôn ngữ tự nhiên + điều kiện vai trò | RL vòng lặp ở lớp lập kế hoạch agent. |

Vào năm 2026, lĩnh vực tăng trưởng lớn nhất của MARL là dựa trên LLM: swarms đàm phán, tranh luận, xây dựng phần mềm model agents ngôn ngữ. RL hiển thị dưới dạng tối ưu hóa ưu tiên trên đầu ra *cấp quỹ đạo*, không phải cấp token (Giai đoạn 16 · 03).

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-marl-architect.md`:

```markdown
---
name: marl-architect
description: Pick the right multi-agent RL regime (IPPO, CTDE, self-play, league) for a given task.
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

Given a task with `n` agents, output:

1. Regime classification. Cooperative / adversarial / general-sum. Justify.
2. Algorithm. IPPO / MAPPO / QMIX / self-play / league. Reason tied to coupling tightness and reward structure.
3. Information access. Centralized training (what global info goes to the critic)? Decentralized execution?
4. Credit assignment. Counterfactual baseline, value decomposition, or reward shaping.
5. Exploration plan. Per-agent entropy, population-based training, or league.

Refuse independent Q-learning on tightly-coupled cooperative tasks. Refuse to recommend self-play for general-sum with cycle risks. Flag any MARL pipeline without a fixed-opponent eval (cherry-picked self-play numbers are common).
```

## Bài tập

1. **Dễ dàng.** Huấn luyện Q-learning độc lập trên GridWorld hợp tác 2 agent. Bao nhiêu episodes cho đến khi trung bình trở về > 0? Vẽ đường cong học tập chung.
2. **Trung bình.** Thêm một nhiệm vụ "phối hợp": mục tiêu chỉ đạt được khi cả hai agents bước vào nó trong cùng một lượt. Q độc lập có còn hội tụ không? Cái gì phá vỡ?
3. **Khó.** Triển khai một nhà phê bình tập trung cho training kiểu MAPPO và so sánh tốc độ hội tụ với PPO độc lập trong nhiệm vụ điều phối.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Trò chơi Markov | "MDP đa agent" | `(S, A_1, …, A_n, P, R_1, …, R_n)`; mỗi agent đều có phần thưởng riêng. |
| CTDE | "training tập trung, thực thi phi tập trung" | Nhà phê bình chung tại thời điểm training; policy của mỗi agent chỉ sử dụng các ob cục bộ. |
| IPPO | "PPO độc lập" | Mỗi agent chạy PPO riêng biệt. Đường cơ sở đơn giản; thường bị đánh giá thấp. |
| BẢN ĐỒ | "Nhiều agent PPO" | PPO với hàm giá trị tập trung có điều kiện trên trạng thái toàn cầu. |
| QMIX | "Phân hủy giá trị đơn điệu" | `Q_tot = f_monotone(Q_1, …, Q_n)` cho phép argmax phi tập trung. |
| COMA | "Đa agent phản thực tế" | Lợi thế = Q của tôi trừ đi Q dự kiến bị gạt ra ngoài lề hành động của tôi. |
| Tự chơi | "Agent so với bản thân trong quá khứ" | Một agent, hai vai trò; tiêu chuẩn cho các trò chơi có tổng bằng không. |
| Chơi giải đấu | "Dân số training" | Lưu trữ policies trong quá khứ, lấy mẫu đối thủ từ nhóm; xử lý các chu kỳ chiến lược. |

## Đọc thêm

- [Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) - CTDE với một nhà phê bình tập trung.
- [Foerster et al. (2017). Counterfactual Multi-Agent Policy Gradients (COMA)](https://arxiv.org/abs/1705.08926) - các đường cơ sở phản thực tế để phân bổ tín chỉ.
- [Rashid et al. (2018). QMIX: Monotonic Value Function Factorisation](https://arxiv.org/abs/1803.11485) - phân hủy giá trị với sự đơn điệu.
- [Yu et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO)](https://arxiv.org/abs/2103.01955) - PPO mạnh mẽ một cách đáng ngạc nhiên đối với MARL.
- [Vinyals et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z) - giải đấu chơi trên quy mô lớn.
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270) - tự chơi thuần túy trong các trò chơi có tổng bằng không.
- [Sutton & Barto (2018). Ch. 15 — Neuroscience & Ch. 17 — Frontiers](http://incompleteideas.net/book/RLbook2020.pdf) - bao gồm cách xử lý ngắn gọn của sách giáo khoa về cài đặt đa agent và vấn đề không đứng yên mà CTDE được thiết kế để giải quyết.
- [Zhang, Yang & Başar (2021). Multi-Agent Reinforcement Learning: A Selective Overview](https://arxiv.org/abs/1911.10635) - khảo sát bao gồm MARL hợp tác, cạnh tranh và hỗn hợp với kết quả hội tụ.
