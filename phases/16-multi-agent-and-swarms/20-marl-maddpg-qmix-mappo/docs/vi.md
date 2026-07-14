# MARL - MADDPG, QMIX, MAPPO

> Di sản học tăng cường của phối hợp đa agent, vẫn cung cấp thông tin cho các hệ thống LLM-agent vào năm 2026. **MADDPG** (Lowe và cộng sự, NeurIPS 2017, arXiv:1706.02275) đã giới thiệu Training tập trung, Thực thi phi tập trung (CTDE): mỗi nhà phê bình nhìn thấy tất cả các trạng thái và hành động của agents trong quá trình training; Tại thời điểm thử nghiệm, chỉ có các tác nhân cục bộ chạy. Hoạt động cho các môi trường hợp tác, cạnh tranh và hỗn hợp. **QMIX** (Rashid và cộng sự, ICML 2018, arXiv: 1803.11485) là phân hủy giá trị với mạng trộn đơn điệu; mỗi agent Q kết hợp thành Q chung để `argmax` phân phối sạch sẽ - chiếm ưu thế trong Thử thách đa Agent StarCraft (SMAC). **MAPPO** (Yu và cộng sự, NeurIPS 2022, arXiv:2103.01955) được PPO với hàm giá trị tập trung; "hiệu quả đáng ngạc nhiên" trên thế giới hạt, SMAC, Google Research Football, Hanabi với điều chỉnh tối thiểu. Những điều này củng cố training policies cho các nhóm agent phải hành động phi tập trung. MAPPO là **đường cơ sở hợp tác xã-MARL mặc định năm 2026**. Bài học này xây dựng mỗi ý tưởng từ một món đồ chơi thế giới lưới nhỏ và đưa ba ý tưởng vào trí nhớ cơ bắp trước khi chạm vào LLM-agent training.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, small NumPy-free implementations)
**Kiến thức tiên quyết:** Giai đoạn 09 (Học tăng cường), Giai đoạn 16 · 09 (Mạng Swarm song song)
**Thời lượng:** ~90 phút

## Vấn đề

Hệ thống LLM-agent ngày càng huấn luyện policies phối hợp giữa các agent: khi nào trì hoãn, khi nào hành động, gọi ngang hàng nào. Tài liệu cho bạn biết cách huấn luyện policies như vậy là Multi-Agent Reinforcement Learning (MARL), có trước làn sóng LLM và có một tập hợp nhỏ các thuật toán thống trị.

Đọc các bài báo MARL mà không có từ vựng mẫu rất đau đớn. Các training tập trung với thực thi phi tập trung (CTDE), phân tách giá trị và các nhà phê bình tập trung không phải là những từ thông dụng - chúng là câu trả lời cụ thể cho các vấn đề cụ thể:

- RL độc lập (mỗi agent học một mình) là không cố định từ quan điểm của mỗi agent. Tệ.
- RL tập trung (một agent kiểm soát tất cả) không mở rộng quy mô và vi phạm các ràng buộc thực thi.
- CTDE tận dụng tốt nhất cả hai: huấn luyện với thông tin toàn cầu, triển khai với policies địa phương.

## Khái niệm

### Ba môi trường mà các tờ báo sử dụng

- **Thế giới hạt (môi trường hạt đa agent).** Vật lý 2D đơn giản với các nhiệm vụ cooperative/competitive. Giường thử nghiệm ban đầu của MADDPG.
- **Thử thách đa Agent StarCraft (SMAC).** Quản lý vi mô hợp tác, quan sát từng phần. Thử nghiệm của QMIX. Hành động rời rạc, trạng thái liên tục.
- **Google Research Football, Hanabi, MPE.** Đường cơ sở MAPPO.

Các môi trường khác nhau có các loại action/observation khác nhau. Các thuật toán chọn cho phù hợp.

### MADDPG (2017) — mô hình CTDE

Mỗi agent `i` có một `mu_i(o_i)` diễn viên ánh xạ quan sát của riêng nó với hành động. Mỗi agent cũng có một `Q_i(x, a_1, ..., a_n)` phê bình xem tất cả các quan sát và mọi hành động trong quá trình training. Nam diễn viên được cập nhật bởi policy gradient chống lại đánh giá của nhà phê bình.

```
actor update:    grad_theta_i J = E[grad_theta mu_i(o_i) * grad_a_i Q_i(x, a_1..n) at a_i=mu_i(o_i)]
critic update:   TD on Q_i(x, a_1..n) given next-state joint estimate
```

Tại sao CTDE: tại thời điểm training, chúng tôi biết hành động của mọi người; Chúng ta sử dụng điều đó để giảm bớt variance trong mỗi nhà phê bình. Tại thời điểm triển khai, mỗi agent chỉ nhìn thấy `o_i` và gọi `mu_i(o_i)`.

Chế độ thất bại: các nhà phê bình phát triển với N agents (đầu vào bao gồm tất cả các hành động). Không vượt quá ~10 agents mà không có xấp xỉ.

### QMIX (2018) — phân hủy giá trị

Chỉ hợp tác. Phần thưởng toàn cầu là tổng của một hàm đơn điệu của mỗi agent giá trị Q:

```
Q_tot(tau, a) = f(Q_1(tau_1, a_1), ..., Q_n(tau_n, a_n)),   df/dQ_i >= 0
```

Sự đơn điệu đảm bảo `argmax_a Q_tot` có thể được tính toán bởi mỗi agent chọn `argmax_{a_i} Q_i` một cách độc lập. Đó là **chính xác thuộc tính thực thi phi tập trung** bạn cần. Tại thời điểm training, một mạng trộn tạo ra `Q_tot` từ các Q mỗi agent.

Tại sao QMIX chiến thắng trên SMAC: quản lý vi mô hợp tác của StarCraft có agents đồng nhất, quan sát địa phương, phần thưởng toàn cầu - hoàn toàn phù hợp để phân hủy giá trị.

Chế độ thất bại: hạn chế tính đơn điệu là hạn chế; Một số nhiệm vụ có cấu trúc phần thưởng không thể phân hủy đơn điệu (một nhiệm vụ agent hy sinh cho nhóm). Tiện ích mở rộng (QTRAN, QPLEX) nới lỏng điều này.

### MAPPO (2022) — mặc định bị bỏ qua

Đa Agent PPO: PPO với chức năng giá trị tập trung. Mỗi agent có policy riêng; Tất cả agents đều chia sẻ (hoặc có mỗi agent) hàm giá trị nhìn thấy trạng thái đầy đủ. Yu et al. 2022 đã so sánh MAPPO với MADDPG, QMIX và các tiện ích mở rộng của chúng trên năm benchmarks và nhận thấy:

- MAPPO phù hợp hoặc đánh bại các phương pháp MARL ngoài policy trên thế giới hạt, SMAC, Google Research Football, Hanabi, MPE.
- Yêu cầu điều chỉnh hyperparameter tối thiểu.
- training ổn định; có thể tái tạo trên các hạt.

Cộng đồng đã đánh giá thấp MARL trên policy cho đến bài báo này. Vào năm 2026, MAPPO là đường cơ sở mặc định cho MARL hợp tác; bất kỳ phương pháp mới nào cũng phải đánh bại nó.

### Tại sao các kỹ sư agent LLM nên quan tâm

Ba mục đích sử dụng trực tiếp:

1. **Bộ định tuyến training.** Meta-agent chọn agent phụ nào xử lý một tác vụ. Đây là vấn đề MARL với N agents phụ phi tập trung và một bộ định tuyến tập trung. MAPPO phù hợp.
2. **Vai trò xuất hiện.** Trong mô phỏng agent tổng hợp, training agents áp dụng các vai trò bổ sung theo thời gian là một vấn đề MARL ngụy trang. Sự phân rã giá trị kiểu QMIX buộc tính bổ sung bằng cách xây dựng.
3. **Sử dụng công cụ đa agent.** Khi agents chia sẻ công cụ và cạnh tranh về ngân sách, training chúng thông qua CTDE sẽ tạo ra các policies địa phương có thể triển khai và tôn trọng các hạn chế về nguồn lực.

Cảnh báo thực tế: vào năm 2026, hầu hết các hệ thống production LLM agent prompt policies của họ hơn là huấn luyện chúng. MARL xuất hiện khi bạn có (a) nhiều dữ liệu tương tác, (b) tín hiệu phần thưởng rõ ràng và (c) sẵn sàng đầu tư vào cơ sở hạ tầng training.

### CTDE như một mẫu thiết kế vượt ra ngoài RL

Ngay cả khi không có training, CTDE là một mô hình kiến trúc hữu ích:

- Trong quá trình *thiết kế*, giả sử khả năng hiển thị toàn đội.
- Tại *runtime*, thực thi thực thi phi tập trung: mỗi agent chỉ nhìn thấy `o_i`.

Mô hình buộc bạn phải giữ cho trạng thái mỗi agent rõ ràng và suy nghĩ về observability một phần trước. Nhiều hệ thống đa agent production âm thầm giả định trạng thái chung ở khắp mọi nơi - kỷ luật CTDE ngăn chặn điều đó.

### Vấn đề không đứng yên

Khi nhiều agents học đồng thời, môi trường của mỗi agent (bao gồm cả policies của người khác) là không đứng yên. Chứng minh một agent RL cổ điển gặp lỗi. Các thuật toán MARL trong bài học này đều giải quyết vấn đề này:

- MADDPG: nhà phê bình toàn cầu nhìn thấy tất cả các hành động, vì vậy ước tính giá trị của nó là cố định.
- QMIX: phân tách giá trị di chuyển việc học đến một không gian Q chung, nơi tính tối ưu được xác định rõ ràng.
- MAPPO: hàm giá trị tập trung làm giảm variance từ những thay đổi policy của người khác.

Trong các hệ thống LLM-agent, sự không đứng yên biểu hiện như "agent của tôi đã hoạt động vào tháng trước, bây giờ các agent ngược dòng khác đã thay đổi, của tôi hoạt động sai trái." Training MARL với CTDE là cách khắc phục có nguyên tắc; Các bản sửa lỗi cấp prompt nhanh hơn nhưng kém bền hơn.

### Bài học này KHÔNG đề cập đến những gì

Training mạng thực tế là một chủ đề Giai đoạn 09. Bài học này xây dựng các phiên bản policy tập lệnh minh họa các mẫu CTDE, phân tách giá trị và giá trị tập trung mà không cần cập nhật gradient. Mục tiêu là nội bộ hóa các mẫu trước khi bạn chọn một thư viện MARL đầy đủ (PyMARL, MARLlib, RLlib multi-agent).

## Tự xây dựng

`code/main.py` thực hiện ba trình diễn mẫu, tất cả đều trên một thế giới lưới hợp tác 2 agent nhỏ:

- Môi trường: 2 agents trên lưới 4x4, một viên thưởng. Phần thưởng = 1 nếu bất kỳ agent nào đạt đến viên nén; nhiệm vụ kết thúc.
- `IndependentAgents` - mỗi agent đối xử với người khác như môi trường. Đường cơ sở.
- `MADDPGStyle` - nhà phê bình tập trung tính toán một giá trị chung; actor policies cập nhật từ đó. Kịch bản policy cải tiến.
- `QMIXStyle` - phân hủy giá trị bằng máy trộn đơn sắc.
- `MAPPOStyle` — chức năng giá trị tập trung; policies cập nhật dựa trên đường cơ sở được chia sẻ.

Cả bốn đều chạy cùng một episodes và báo cáo các bước trung bình đến mục tiêu. Các biến thể CTDE hội tụ đến các đường dẫn ngắn hơn đường cơ sở độc lập.

Chạy:

```
python3 code/main.py
```

Đầu ra dự kiến: độc lập agents thực hiện trung bình ~6 bước; Các biến thể CTDE hội tụ về ~3,5 bước (tối ưu cho lưới 4x4 là 3). Sự khác biệt của mẫu xuất hiện mặc dù có policies theo kịch bản.

## Ứng dụng

`outputs/skill-marl-picker.md` là một skill chọn thuật toán MARL cho một nhiệm vụ đa agent nhất định: hợp tác vs cạnh tranh, đồng nhất vs không đồng nhất, loại không gian hành động, quy mô, tín hiệu phần thưởng.

## Sản phẩm bàn giao

MARL ở production rất hiếm. Khi bạn sử dụng nó:

- **Bắt đầu với MAPPO.** Bài báo năm 2022 đã xác định đây là đường cơ sở; Việc tái tạo nó trước tiên giúp tiết kiệm hàng tuần theo đuổi các phương pháp lạ mắt hơn.
- **Ghi lại luồng hành động và quan sát của mọi agent.** Gỡ lỗi MARL mà không có mỗi agent traces là vô vọng.
- **Tách mã training khỏi mã thực thi.** CTDE là một kỷ luật; Hãy để đường dẫn thực thi thực sự chỉ nhìn thấy `o_i`.
- **Cảnh báo định hình phần thưởng.** MARL cực kỳ nhạy cảm với thiết kế phần thưởng. Một lỗi phối hợp trong việc tạo hình và agents học cách khai thác nó. Chạy thử nghiệm đối nghịch.
- **Đối với LLM agents**, hãy xem xét policies cấp prompt trước. Chỉ đầu tư vào training MARL khi dữ liệu tương tác + tín hiệu phần thưởng + hạ tầng đều có mặt.

## Bài tập

1. Chạy `code/main.py`. Đo lường khoảng cách giữa các bước đến mục tiêu giữa agents độc lập và kiểu MAPPO. Khoảng cách tăng hay thu hẹp trên lưới 6x6?
2. Triển khai một biến thể cạnh tranh: hai agents, một viên, chỉ người đầu tiên đạt được phần thưởng. Mô hình nào xử lý cạnh tranh một cách sạch sẽ? MADDPG trong lịch sử.
3. Đọc MADDPG (arXiv:1706.02275) Phần 3. Thực hiện quy tắc cập nhật chính xác của nhà phê bình một cách tượng trưng bằng mã giả bằng lời của riêng bạn.
4. Đọc MAPPO (arXiv: 2103.01955). Tại sao các tác giả lập luận về giá trị tập trung + PPO đánh bại MARL ngoài policy trên benchmarks của họ? Liệt kê ba tuyên bố mạnh mẽ nhất.
5. Áp dụng CTDE như một mẫu thiết kế cho một hệ thống LLM-agent giả định (ví dụ: nghiên cứu agent + trình tóm tắt + mã hóa). Thông tin chung có sẵn tại thời điểm thiết kế mà không có sẵn tại runtime là gì?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| MARL | "Nhiều Agent RL" | Học tăng cường cho hệ thống đa agent. |
| CTDE | "Training tập trung, thực thi phi tập trung" | Huấn luyện với thông tin toàn cầu; triển khai với policies địa phương. |
| ĐIÊN | "DDPG đa Agent" | CTDE với mỗi agent phê bình xem tất cả các quan sát + hành động. |
| QMIX | "Phân hủy giá trị" | Trộn đơn điệu của mỗi agent Qs. Hợp tác. |
| BẢN ĐỒ | "Nhiều Agent PPO" | PPO với chức năng giá trị tập trung. Đường cơ sở mặc định năm 2026. |
| Phân hủy giá trị | "Tổng các Q riêng lẻ" | Q khớp được biểu thị dưới dạng một hàm đơn điệu của mỗi agent Qs. |
| Không đứng yên | "Mục tiêu di chuyển" | Môi trường của mỗi agent thay đổi khi những người khác học. Vấn đề cốt lõi của MARL. |
| Bật-policy / tắt-policy | "Học hỏi từ hiện tại / phát lại" | PPO đang policy (MAPPO); DDPG và Q-learning không policy. |
| SMAC | "Thử thách đa Agent StarCraft" | benchmark quản lý vi mô hợp tác; Mảnh đất cây nhà lá vườn của QMIX. |

## Đọc thêm

- [Lowe et al. — Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275) — MADDPG; NeurIPS 2017
- [Rashid et al. — QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485) - QMIX; ICML 2018
- [Yu et al. — The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955) - MAPPO; Thần kinh IPS 2022
- [BAIR blog post on MAPPO](https://bair.berkeley.edu/blog/2021/07/14/mappo/) - khung có thể đọc được của kết quả MAPPO
- [SMAC repository](https://github.com/oxwhirl/smac) - Thử thách đa Agent StarCraft
