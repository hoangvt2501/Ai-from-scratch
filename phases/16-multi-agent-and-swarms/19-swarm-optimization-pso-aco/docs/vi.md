# Swarm Tối ưu hóa cho LLMs (PSO, ACO)

> Tối ưu hóa lấy cảm hứng từ sinh học đang trở lại LLM. **LMPSO** (arXiv:2504.09247) sử dụng PSO trong đó vận tốc của mỗi hạt là một prompt và LLM tạo ra ứng cử viên tiếp theo; Hoạt động tốt trên các đầu ra theo trình tự có cấu trúc (biểu thức toán học, chương trình). **Model Swarms** (arXiv:2410.11163) coi mỗi chuyên gia LLM như một hạt PSO trên đa tạp có trọng lượng model và báo cáo **13,3% mức tăng trung bình** trên 12 đường cơ sở trên 9 datasets chỉ với 200 trường hợp. **SwarmPrompt** (ICAART 2025) lai PSO + Sói xám để tối ưu hóa prompt. **AMRO-S** (arXiv:2603.12933) là chuyên gia pheromone lấy cảm hứng từ ACO để định tuyến đa agent LLM — **Tăng tốc gấp 4,7 lần**, bằng chứng định tuyến có thể giải thích, cập nhật không đồng bộ được kiểm soát chất lượng giúp tách inference khỏi việc học. Bài học này triển khai PSO trên không gian prompt parameter và ACO trên định tuyến agent, đo lường lý do tại sao các thuật toán cổ điển này phù hợp với thời đại LLM và khi nào chúng không phù hợp.

**Loại:** Tìm hiểu + Xây dựng
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 16 · 09 (Mạng Swarm song song), Giai đoạn 16 · 14 (Đồng thuận và BFT)
**Thời lượng:** ~75 phút

## Vấn đề

Bạn có một prompt đạt điểm 62% trong việc đánh giá nhiệm vụ của mình. Bạn muốn cải thiện nó. Động thái ngây thơ là tinh chỉnh thủ công không cần gradient, quy mô xấu. Học tăng cường cần tín hiệu phần thưởng và đủ rollouts để huấn luyện. Backprop thông qua prompts là không thực sự khả thi - prompt là một chuỗi rời rạc, không phải là một parameter có thể vi phân.

Tối ưu hóa lấy cảm hứng từ sinh học cổ điển - PSO cho không gian tìm kiếm liên tục, ACO để chọn đường dẫn - được thiết kế chính xác cho chế độ này: không gradient, dựa trên dân số, rẻ cho mỗi đánh giá. Ghép nối chúng với LLMs để có bước tìm kiếm không gradient và bạn sẽ nhận được một optimizer thiết thực đáng ngạc nhiên.

Các mẫu tương tự áp dụng cho agent *định tuyến* trong các hệ thống đa agent. Đường mòn pheromone kiểu ACO ghi lại agent nào hoạt động tốt nhất trên loại tác vụ nào, cho phép bộ định tuyến khai thác đường mòn và phân rã pheromone để có thể khám phá lại các tuyến đường.

## Khái niệm

### Bồi dưỡng PSO (Kennedy & Eberhart 1995)

Tối ưu hóa Swarm hạt: tổng thể các hạt trong không gian tìm kiếm liên tục. Mỗi hạt có vị trí `x_i` và vận tốc `v_i`. Mỗi lần lặp:

```
v_i <- w * v_i + c1 * r1 * (p_best_i - x_i) + c2 * r2 * (g_best - x_i)
x_i <- x_i + v_i
evaluate fitness(x_i)
update p_best_i if improved
update g_best if global best
```

Trong đó `p_best` là tốt nhất của hạt, `g_best` là tốt nhất của swarm, `w, c1, c2` là quán tính + nhận thức + trọng lượng xã hội, `r1, r2` là các yếu tố ngẫu nhiên.

### PSO trên đầu ra LLM - LMPSO

arXiv:2504.09247 điều chỉnh PSO cho các đầu ra có cấu trúc do LLM tạo ra (biểu thức toán học, chương trình). Mỗi hạt là một đầu ra ứng cử viên. Vận tốc là một * prompt * mô tả cách sửa đổi đầu ra hiện tại theo hướng tốt nhất personal/global. LLM tạo ra đầu ra mới từ vận tốc prompt. "Quán tính" của vận tốc là một prompt giống như "thực hiện các thay đổi gia tăng nhỏ".

Điều này hoạt động tốt khi:
- Đầu ra có cấu trúc (có thể phân tích cú pháp, có thể đánh giá).
- Thể dục là tự động (chạy thử, đánh giá số học).
- Dân số nhỏ (~10-30 hạt) nên tổng số cuộc gọi LLM vẫn có thể quản lý được.

Nó không hoạt động tốt khi thể dục cần sự đánh giá của con người - chi phí cho mỗi lần lặp lại trở nên quá cao.

### Model Swarms

arXiv: 2410.11163 đưa PSO ra khỏi lớp đầu ra và vào lớp * model *. Mỗi "hạt" là một chuyên gia LLM (parameters). swarm đưa parameters hướng tới tốt nhất tập thể thông qua bản cập nhật không cần gradient. Báo cáo: Mức tăng trung bình 13,3% trên 12 đường cơ sở trên 9 datasets, chỉ với 200 phiên bản mỗi lần lặp.

Thông tin chi tiết quan trọng là LLM chuyên gia models đã ở gần trong một đa tạp parameter dùng chung (trọng lượng bộ chuyển đổi, LoRA delta). PSO trên không gian con chiều thấp này rẻ và hiệu quả.

### Bồi dưỡng ACO (Dorigo 1992)

Tối ưu hóa đàn kiến: kiến đi qua biểu đồ; Mỗi con đường có một vệt pheromone. Xác suất di chuyển kiến trọng lượng theo cường độ pheromone. Kiến hoàn thành nhiệm vụ lắng đọng pheromone tỷ lệ thuận với chất lượng dung dịch. Pheromone phân rã theo thời gian.

### AMRO-S — ACO cho định tuyến agent

arXiv:2603.12933 sử dụng ACO để định tuyến nhiều agent. Mỗi loại nhiệm vụ là một "điểm đến"; mỗi agent là một con đường khả thi. Pheromone tăng cường các tuyến đường tạo ra đầu ra tốt. Những đóng góp chính:

- **Bằng chứng định tuyến có thể giải thích.** Cường độ pheromone là một tín hiệu mà con người có thể đọc được.
- **Cập nhật không đồng bộ được kiểm soát chất lượng.** Pheromone chỉ cập nhật sau khi vượt qua kiểm tra chất lượng, tách inference khỏi việc học.
- **Tăng tốc gấp 4,7 lần** trên benchmark định tuyến đa agent.

Cổng chất lượng rất quan trọng: nếu không có nó, agents nhanh nhưng sai sẽ tích lũy pheromone và hệ thống khóa vào các tuyến đường xấu.

### Khi nào nên sử dụng PSO / ACO cho LLMs

**Sử dụng PSO khi:**
- Không gian tìm kiếm liên tục hoặc ánh xạ đến parameters liên tục (prompt embeddings, trọng số LoRA, tạo số parameters).
- Thể dục rẻ và tự động.
- Dân số có thể nhỏ (10-30).

**Sử dụng ACO khi:**
- Bạn gặp vấn đề về định tuyến hoặc chọn đường dẫn.
- Các quyết định được củng cố theo thời gian (các loại nhiệm vụ tương tự quay trở lại).
- Bạn cần bằng chứng có thể giải thích cho các quyết định định tuyến.

**Không sử dụng khi:**
- Thể dục yêu cầu sự đánh giá của con người (quá đắt cho mỗi lần lặp).
- Không gian tìm kiếm là rời rạc và kết hợp theo cách mà PSO không bao gồm (thay vào đó sử dụng các thuật toán di truyền).
- Các quyết định theo thời gian thực cần độ trễ nghiêm ngặt (PSO/ACO hội tụ chậm so với phỏng đoán một lần).

### Tại sao lấy cảm hứng từ sinh học vẫn chiến thắng

Các phương pháp dựa trên Gradient cần các tín hiệu có thể phân biệt được. LLM ra và quyết định định tuyến không thể phân biệt được. Các phương pháp giả gradient (bộ định tuyến học tăng cường, bộ dò prompt kiểu DPO) hoạt động nhưng cần training đắt tiền.

PSO và ACO chỉ cần một chức năng *đánh giá*. Nếu bạn có thể ghi điểm đầu ra của ứng viên hoặc quyết định định tuyến, bạn có thể tối ưu hóa trên không gian. Điều đó làm cho tiêu chuẩn về khả năng áp dụng thấp hơn nhiều.

### Giới hạn thực tế

- **Ngân sách dân số.** N hạt × lặp lại T × chi phí cho mỗi lần đánh giá. Đối với LLM đánh giá ở ~ $0.02 / call, a 20-particle PSO running 50 iterations costs ~$20. Lập kế hoạch cho phù hợp.
- **Thăm dò và khai thác.** Tỷ lệ phân rã pheromone và quán tính PSO đánh đổi; phân rã quá nhanh → quên dung dịch; Quá chậm → bị mắc kẹt trên Optima cục bộ sớm.
- **Trôi dạt thảm khốc.** Cả hai thuật toán đều có thể hội tụ và sau đó phân kỳ nếu bối cảnh thể dục thay đổi (phân phối dữ liệu mới). Theo dõi sự ổn định của thể lực tốt nhất.

## Tự xây dựng

`code/main.py` thực hiện:

- `LMPSO` - PSO trên prompt parameters số (trọng số temperature, top_k). "Thế hệ LLM" của mỗi hạt được mô phỏng như một chức năng thể dục theo kịch bản. Chạy thuật toán trong 30 lần lặp lại và hiển thị g_best hội tụ.
- `AMRO_S` - Định tuyến kiểu ACO. 3 agents, 4 loại nhiệm vụ, ma trận pheromone, 100 nhiệm vụ định tuyến. Bản in (task_type → agent lựa chọn) phân phối theo thời gian để hiển thị sự hình thành đường mòn.
- So sánh: định tuyến ngẫu nhiên so với định tuyến ACO trên cùng một luồng tác vụ. Đo lường chất lượng và độ trễ.

Chạy:

```
python3 code/main.py
```

Sản lượng dự kiến:
- LMPSO: g_best thể lực cải thiện từ ngẫu nhiên đến gần tối ưu hơn 30 lần lặp.
- AMRO-S: bảng pheromone ổn định ở agent bên phải cho mỗi loại nhiệm vụ; Định tuyến ACO đánh bại ngẫu nhiên ~30-40% về chất lượng và cũng giảm độ trễ (ít thử lại hơn).

## Ứng dụng

`outputs/skill-swarm-optimizer.md` giúp lựa chọn giữa PSO, ACO, thuật toán di truyền và optimizers dựa trên gradient cho các bài toán tối ưu hóa LLM / agent.

## Sản phẩm bàn giao

- **Bắt đầu nhỏ.** 10-20 hạt, 20-50 lần lặp. Chỉ tăng quy mô nếu đường cong hội tụ cho thấy mức tăng rõ ràng.
- **Ghi nhật ký pheromone hoặc g_best mỗi lần lặp.** Gỡ lỗi swarm optimizers mà không có dấu vết rất khó khăn.
- **Cập nhật cổng chất lượng.** Đặc biệt đối với định tuyến ACO: agents nhanh và sai không được tích lũy pheromone.
- **Đặt lại sự phân rã trên ca phân phối.** Khi phân phối đánh giá của bạn thay đổi, pheromone già sẽ cũ; đặt lại hoặc tăng gấp đôi tốc độ phân rã tạm thời.
- **Giới hạn chi phí cho mỗi lần lặp.** Đưa ra chỉ số chi phí cho mỗi lần lặp. PSO có giá 500 đô la / lần lặp lại và tăng 0,5% không thể vận chuyển được.

## Bài tập

1. Chạy `code/main.py`. Quan sát sự hội tụ LMPSO. Thay đổi quy mô dân số 5, 10, 20, 50. Thời gian hội tụ bão hòa ở kích thước nào?
2. Thực hiện thí nghiệm "trôi dạt thảm họa": sau lần lặp 30, thay đổi chức năng thể dục. PSO thích ứng nhanh như thế nào? Đặt lại `p_best` có giúp ích không?
3. Thêm một cổng chất lượng vào AMRO-S: chỉ gửi pheromone trên các lần chạy với điểm đánh giá > 0,7. Điều này thay đổi sự hội tụ so với phiên bản không được kiểm soát như thế nào?
4. Đọc LMPSO (arXiv:2504.09247). Ánh xạ "vận tốc như một prompt" của tờ báo trở lại vận tốc số của bạn. Những gì bị mất trong mô phỏng và những gì được bảo tồn?
5. Đọc AMRO-S (arXiv: 2603.12933). Triển khai "đường dẫn nhanh inference" tách rời với cập nhật pheromone không đồng bộ. Điều này thay đổi độ trễ hệ thống như thế nào khi tải liên tục?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| PSO | "Tối ưu hóa Swarm hạt" | Kennedy-Eberhart 1995. Dựa trên dân số gradient không optimizer. |
| ACO | "Tối ưu hóa đàn kiến" | Dorigo 1992. Path/route tối ưu hóa thông qua các vệt pheromone. |
| LMPSO | "PSO với thế hệ LLM" | arXiv: 2504.09247 (bằng tiếng Anh). Vận tốc là một prompt; LLM tạo ra các ứng cử viên. |
| Model Swarms | "PSO về trọng lượng chuyên gia" | arXiv:2410.11163. Cập nhật miễn phí Gradient trên không gian con model parameter. |
| AMRO-S | "ACO cho định tuyến agent" | arXiv: 2603.12933 (bằng tiếng Anh). Ma trận pheromone trên × agent loại nhiệm vụ. |
| p_best / g_best | "Cá nhân / toàn cầu tốt nhất" | Các giải pháp tốt nhất trên mỗi hạt và trên toàn swarm được tìm thấy cho đến nay. |
| Pheromone | "Định tuyến bộ nhớ" | Sức mạnh trên một cạnh; phân rã theo thời gian; tiền gửi về chất lượng. |
| Cập nhật kiểm soát chất lượng | "Chỉ học hỏi từ những lần chạy tốt" | Tiền gửi pheromone có điều kiện kiểm tra chất lượng. |
| Trôi dạt thảm khốc | "Ca phân phối" | Cảnh quan thể dục thay đổi; p_best cũ và pheromone trở nên cũ. |

## Đọc thêm

- [Kennedy & Eberhart — Particle Swarm Optimization](https://ieeexplore.ieee.org/document/488968) — bài báo PSO năm 1995
- [Dorigo — Ant Colony Optimization](https://www.aco-metaheuristic.org/about.html) - Quỹ ACO năm 1992
- [LMPSO — Language Model Particle Swarm Optimization](https://arxiv.org/abs/2504.09247) — PSO cho đầu ra LLM có cấu trúc
- [Model Swarms — gradient-free LLM expert optimization](https://arxiv.org/abs/2410.11163) — PSO trên không gian con trọng lượng model
- [AMRO-S — ant-colony multi-agent routing](https://arxiv.org/abs/2603.12933) - định tuyến điều khiển pheromone với cổng chất lượng
