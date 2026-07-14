# Chuyển từ Sim sang thực

> Một policy được huấn luyện trong trình mô phỏng bị lỗi trên phần cứng là một policy ghi nhớ trình mô phỏng. Ngẫu nhiên miền, thích ứng miền và nhận dạng hệ thống là ba công cụ giúp các bộ điều khiển đã học vượt qua khoảng cách thực tế.

**Loại:** Học
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 9 · 08 (PPO), Giai đoạn 2 · 10 (Bias/Variance)
**Thời lượng:** ~45 phút

## Vấn đề

Training một robot thực sự chậm, nguy hiểm và tốn kém. Một con hai chân mất hàng triệu training episodes để học cách đi bộ; một con hai chân thực sự bị ngã ngay cả khi bị hỏng phần cứng. Mô phỏng cung cấp cho bạn thiết lập lại không giới hạn, khả năng tái tạo xác định, môi trường song song và không có thiệt hại vật lý.

Nhưng trình mô phỏng đã sai. Vòng bi có nhiều ma sát hơn MuJoCo models. Máy ảnh có biến dạng ống kính mà trình mô phỏng không bao gồm. Động cơ có độ trễ, phản ứng dữ dội và bão hòa mà 99% models sim bỏ qua. Gió, bụi và ánh sáng thay đổi phá hoại một policy được huấn luyện về kết xuất vô trùng. **Khoảng cách thực tế** - sự khác biệt có hệ thống giữa phân phối sim và phân phối thực - là vấn đề trung tâm của RL triển khai cho robot.

Bạn cần một policy * mạnh mẽ để chuyển đổi phân phối từ mô phỏng sang thực *. Ba cách tiếp cận lịch sử: ngẫu nhiên hóa trình mô phỏng (ngẫu nhiên hóa miền), điều chỉnh policy với một ít dữ liệu thực (thích ứng miền / fine-tuning) hoặc xác định parameters của hệ thống thực và khớp chúng (nhận dạng hệ thống). Vào năm 2026, công thức thống trị kết hợp cả ba với mô phỏng song song lớn (Isaac Sim, Isaac Lab, Mujoco MJX trên GPU).

## Khái niệm

![Three sim-to-real regimes: domain randomization, adaptation, system identification](../assets/sim-to-real.svg)

**Ngẫu nhiên hóa miền (DR).** Tobin et al. 2017, Peng et al. 2018. Trong quá trình training, hãy ngẫu nhiên hóa mọi parameter sim có thể khác nhau trên robot thực: khối lượng, hệ số ma sát, độ lợi PD của động cơ, nhiễu cảm biến, vị trí camera, ánh sáng, kết cấu, models tiếp xúc. policy học phân phối có điều kiện về "nó đang ở trong sim nào ngày nay" và khái quát hóa trên toàn bộ span. Nếu robot thực nằm trong phong bì training, policy sẽ hoạt động.

- **Ưu điểm: **không cần dữ liệu thực. Một công thức, nhiều robot.
- **Nhược điểm: **training ngẫu nhiên quá mức tạo ra một policy "phổ quát" nhưng quá thận trọng. Quá nhiều nhiễu ≈ quá nhiều chính quy hóa.

**Nhận dạng hệ thống (SI).** Lắp parameters của trình mô phỏng với dữ liệu trong thế giới thực trước khi training. Nếu bạn có thể đo ma sát khớp cánh tay trên rô-bốt thật, hãy cắm nó vào mô phỏng. Sau đó, huấn luyện một policy mong đợi những giá trị đó. Cần truy cập vào hệ thống thực nhưng giảm khoảng cách thực tế trực tiếp.

- **Ưu điểm: **mục tiêu training chính xác, nhiễu thấp.
- **Nhược điểm: **lỗi model còn lại không thể nhìn thấy đối với policy; các hiệu ứng nhỏ không xác định được (ví dụ: dải chết động cơ) vẫn phá vỡ việc triển khai.

**Thích ứng miền.** Huấn luyện trong sim, fine-tune với một lượng nhỏ dữ liệu thực. Hai hương vị:

- **Real2Sim2Real:** Tìm hiểu một trình mô phỏng còn lại `f(s, a, z) - f_sim(s, a)` sử dụng rollouts thực, huấn luyện trong sim đã sửa chữa. Thu hẹp khoảng cách mà không cần nhiều dữ liệu thực.
- **Thích ứng quan sát: **huấn luyện một policy ánh xạ các quan sát thực → các quan sát giống sim thông qua trình trích xuất feature đã học (ví dụ: GAN pixel-to-pixel). Bộ điều khiển vẫn ở chế độ sim.

**Học tập đặc quyền / giáo viên-học sinh. **Miki et al. 2022 (ANYmal bốn chân). Huấn luyện một * giáo viên * mô phỏng có quyền truy cập vào thông tin đặc quyền (ma sát ground truth, chiều cao địa hình, trôi IMU). Chắt lọc một * sinh viên * chỉ nhìn thấy các quan sát cảm biến thực. Học sinh học cách suy ra các features đặc quyền từ lịch sử, mạnh mẽ trên các parameters vật lý.

**Mô phỏng song song hàng loạt.** 2024–2026. Isaac Lab, Mujoco MJX, Brax đều chạy hàng nghìn robot song song trên một GPU. PPO với 4.096 hình người song song thu thập nhiều năm kinh nghiệm tính bằng giờ. "Khoảng cách thực tế" thu hẹp khi training phân bố mở rộng; DR trở nên gần như tự do khi mỗi trong số 4.096 env đó có các parameters ngẫu nhiên khác nhau.

**Công thức 2026 trong thế giới thực (ví dụ đi bộ bốn chân):**

1. Sim song song hàng loạt với trọng lực ngẫu nhiên miền, ma sát, tăng động cơ payload.
2. Giáo viên policy huấn luyện với thông tin đặc quyền (bản đồ địa hình, vận tốc cơ thể ground truth).
3. Học sinh policy chắt lọc từ giáo viên chỉ bằng cách sử dụng cảm giác tự nhiên (khớp chân encoders).
4. Thích ứng quan sát tùy chọn thông qua bộ mã hóa tự động trên IMU thực.
5. Triển khai. Zero-shot trên 10+ môi trường. Nếu không thành công, hãy thực hiện vài phút fine-tuning trong thế giới thực với PPO hạn chế về an toàn.

## Tự xây dựng

Mã của bài học này là một minh chứng nhỏ về ngẫu nhiên miền trên GridWorld với các chuyển đổi *nhiễu*. Chúng ta huấn luyện một policy trải nghiệm xác suất trượt ngẫu nhiên trong "sim" và đánh giá trên "thực" với mức trượt mà nó chưa từng thấy trong quá trình training. Hình dạng ánh xạ trực tiếp đến chuyển từ MuJoCo sang phần cứng.

### Bước 1: sim được tham số hóa

```python
def step(state, action, slip):
    if rng.random() < slip:
        action = random_perpendicular(action)
    ...
```

`slip` là một parameter trình mô phỏng phơi bày. Trong robot thực sự, nó có thể là ma sát, khối lượng, độ lợi động cơ - bất cứ thứ gì chuyển đổi giữa sim và thực.

### Bước 2: huấn luyện với DR

Khi bắt đầu mỗi episode, hãy lấy mẫu `slip ~ Uniform[0.0, 0.4]`. Huấn luyện PPO / Q-learning / bất cứ thứ gì. Làm điều này cho nhiều episodes.

### Bước 3: Đánh giá zero-shot trên phiếu "thật"

Đánh giá trên `slip ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}`. Bốn người đầu tiên nằm trong training hỗ trợ; `0.5` và `0.7` ở bên ngoài. Một policy được huấn luyện DR phải ở mức hỗ trợ bên trong gần như tối ưu và xuống cấp một cách duyên dáng bên ngoài. Một policy được huấn luyện trượt cố định sẽ giòn bên ngoài training trượt của nó.

### Bước 4: so sánh với training hẹp

Huấn luyện policy thứ hai chỉ với `slip = 0.0`. Đánh giá trên cùng một `slip` quét. Bạn sẽ thấy một sự sụt giảm thảm khốc ngay khi trượt thực sự > 0.

## Cạm bẫy

- **Quá nhiều ngẫu nhiên. **Huấn luyện trên `slip ∈ [0, 0.9]` và policy của bạn không thích rủi ro đến mức không bao giờ thử con đường tối ưu. Phù hợp với phân phối trong thế giới thực *mong đợi*, không phải "bất cứ điều gì có thể xảy ra".
- **Quá ít ngẫu nhiên. **Huấn luyện trên một lát mỏng và policy hoàn toàn không thể khái quát hóa. Sử dụng chương trình giảng dạy thích ứng (Ngẫu nhiên miền tự động) để mở rộng phân phối khi policy cải thiện.
- **Xác định sai parameter không gian. **Ngẫu nhiên hóa sai thứ (màu sắc của máy ảnh khi khoảng cách thực sự là độ trễ của động cơ) và DR không giúp ích gì. Hồ sơ robot thực sự trước.
- **Rò rỉ thông tin đặc quyền.** Một giáo viên sử dụng trạng thái toàn cầu cho các hành động, không chỉ quan sát, có thể tạo ra một học sinh không thể bắt kịp. Đảm bảo policy của giáo viên có thể nhận ra bởi học sinh có lịch sử quan sát.
- **Lỗi chuyển sim sang sim.** Nếu policy của bạn không mạnh mẽ đối với biến thể sim khó hơn, nó cũng sẽ không mạnh mẽ trong thế giới thực. Luôn kiểm tra trên biến thể sim bị tạm dừng trước khi triển khai.
- **Không có phong bì an toàn trong thế giới thực.** Một policy hoạt động trong sim và "hoạt động thật" mà không có tấm chắn an toàn cấp thấp vẫn có thể phá vỡ phần cứng. Thêm rate limits, giới hạn mô-men xoắn, giới hạn khớp trong bộ điều khiển chưa học.

## Ứng dụng

stack mô phỏng thành thực tế năm 2026:

| Tên miền | Stack |
|--------|-------|
| Chuyển động bằng chân (ANYmal, Spot, hình người) | Isaac Lab + DR + giáo viên / sinh viên đặc quyền |
| Thao tác (tay khéo léo, gắp và đặt) | Phòng thí nghiệm Isaac + DR + DR-GAN cho thị lực |
| Lái xe tự động | Sim CARLA / NVIDIA DRIVE + DR + fine-tune thực |
| Đua máy bay không người lái | RotorS / Flightmare + DR + thích ứng trực tuyến |
| Thao tác Finger/in-hand | OpenAI Dactyl (DR ở quy mô chưa từng có) |
| Cánh tay công nghiệp | MuJoCo-Warp + SI + fine-tune thực nhỏ |

Để kiểm soát ở mọi quy mô, quy trình làm việc nhất quán: lắp sim tốt nhất có thể, ngẫu nhiên hóa những gì bạn không thể lắp vào, huấn luyện policies khổng lồ, chưng cất, triển khai với tấm chắn an toàn.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-sim2real-planner.md`:

```markdown
---
name: sim2real-planner
description: Plan a sim-to-real transfer pipeline for a given robot + task, covering DR, SI, and safety.
version: 1.0.0
phase: 9
lesson: 11
tags: [rl, sim2real, robotics, domain-randomization]
---

Given a robot platform, a task, and access to real hardware time, output:

1. Reality gap inventory. Suspected sources ranked by expected impact (contact, sensing, actuation delay, vision).
2. DR parameters. Exact list, ranges, distribution. Justify each range against real measurements.
3. SI steps. Which parameters to measure; measurement method.
4. Teacher/student split. What privileged info the teacher uses; what obs the student uses.
5. Safety envelope. Low-level limits, emergency stops, backup controller.

Refuse to deploy without (a) a zero-shot sim-variant test, (b) a safety shield, (c) a rollback plan. Flag any DR range wider than 3× measured real variability as likely over-randomized.
```

## Bài tập

1. **Dễ dàng.** Huấn luyện một agent Q-learning trên GridWorld trượt cố định (trượt = 0.0). Đánh giá trên ∈ trượt {0.0, 0.1, 0.3, 0.5}. Hoàn trả biểu đồ so với trượt.
2. **Trung bình.** Huấn luyện một agent sampling `slip ~ Uniform[0, 0.3]` học DR Q. Đánh giá cùng một lần quét. DR mua bao nhiêu với giá trượt = 0,5 (ngoài phân phối)?
3. **Khó.** Thực hiện chương trình giảng dạy: bắt đầu với trượt = 0,0, mở rộng phạm vi DR mỗi khi policy đạt 90% mức tối ưu. Đo tổng số bước môi trường để đạt được trượt = 0,3 zero-shot so với đường cơ sở DR cố định.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Khoảng cách thực tế | "Sự khác biệt giữa sim với thực tế" | Chuyển đổi phân phối giữa training và triển khai physics/sensing. |
| Ngẫu nhiên miền (DR) | "Huấn luyện qua các sim ngẫu nhiên" | Ngẫu nhiên hóa parameters sim trong quá trình training để policy khái quát hóa. |
| Nhận dạng hệ thống (SI) | "Đo sim thật và vừa vặn" | Ước tính parameters vật lý thực tế; đặt SIM để phù hợp. |
| Thích ứng miền | "Fine-tune trên dữ liệu thực" | Các fine-tune nhỏ trong thế giới thực sau khi training sim; có thể thích ứng với OBS hoặc động lực. |
| Thông tin đặc quyền | "Ground truth cho giáo viên" | Thông tin chỉ có sim có; học sinh phải suy ra nó từ lịch sử obs. |
| Teacher/student | "Chắt lọc đặc quyền -> quan sát được" | Giáo viên được huấn luyện với các phím tắt; học sinh học cách bắt chước mà không cần chúng. |
| ADR | "Ngẫu nhiên tên miền tự động" | Chương trình giảng dạy mở rộng phạm vi DR khi policy được cải thiện. |
| Real2Sim | "Thu hẹp khoảng cách bằng dữ liệu thực" | Tìm hiểu phần dư để làm cho sim bắt chước rollouts thật. |

## Đọc thêm

- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) — bài báo DR gốc (tầm nhìn cho robot).
- [Peng et al. (2018). Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537) - DR cho động lực, chuyển động bốn chân.
- [OpenAI et al. (2019). Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113) - Dactyl, ADR trên quy mô lớn.
- [Miki et al. (2022). Learning robust perceptive locomotion for quadrupedal robots in the wild](https://www.science.org/doi/10.1126/scirobotics.abk2822) — giáo viên-học sinh cho ANYmal.
- [Makoviychuk et al. (2021). Isaac Gym: High Performance GPU Based Physics Simulation for Robot Learning](https://arxiv.org/abs/2108.10470) — SIM song song khổng lồ thúc đẩy việc triển khai giai đoạn 2025–2026.
- [Akkaya et al. (2019). Automatic Domain Randomization](https://arxiv.org/abs/1910.07113) - Phương pháp giảng dạy ADR.
- [Sutton & Barto (2018). Ch. 8 — Planning and Learning with Tabular Methods](http://incompleteideas.net/book/RLbook2020.pdf) - khung Dyna (sử dụng model để lập kế hoạch + rollouts) làm nền tảng cho pipelines sim thành thực hiện đại.
- [Zhao, Queralta & Westerlund (2020). Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey](https://arxiv.org/abs/2009.13303) — phân loại các phương pháp sim thành thực với kết quả benchmark.
