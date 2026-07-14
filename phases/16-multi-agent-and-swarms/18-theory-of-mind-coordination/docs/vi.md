# Lý thuyết về tâm trí và sự phối hợp nổi lên

> Li et al. (arXiv: 2310.10701) đã chỉ ra rằng LLM agents trong một trò chơi văn bản hợp tác triển lãm **Lý thuyết tâm trí bậc cao mới nổi **(ToM) - lý luận về những gì agent khác tin tưởng về niềm tin của một phần ba agent - nhưng thất bại trong việc lập kế hoạch dài hạn do quản lý bối cảnh và ảo giác. Riedl (arXiv: 2510.05174) đã đo sức mạnh tổng hợp bậc cao hơn trên một quần thể và phát hiện ra rằng **chỉ **điều kiện ToM-prompt tạo ra sự khác biệt liên quan đến bản sắc và tính bổ sung theo định hướng mục tiêu; LLMs dung lượng thấp hơn chỉ cho thấy sự xuất hiện giả. Nghĩa là, sự xuất hiện phối hợp là prompt điều kiện và phụ thuộc model, không tự do. Bài học này thực hiện một agent nhận biết ToM tối thiểu, chạy một nhiệm vụ hợp tác có và không có prompting ToM và đo lường delta phối hợp so với giao thức Riedl 2025.

**Loại:** Tìm hiểu + Xây dựng
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 16 · 07 (Xã hội Tâm trí và Tranh luận), Giai đoạn 16 · 17 (Agents tổng quát)
**Thời lượng:** ~75 phút

## Vấn đề

Sự phối hợp đa agent thường trông kỳ diệu: agents phân chia lao động, dự đoán lẫn nhau, tránh dư thừa. Thông thường "sự xuất hiện" này là một artifact của kỹ thuật prompt - ai đó bảo agents "phối hợp". Tháo prompt, loại bỏ sự phối hợp.

Phát hiện năm 2025 của Riedl nghiêm ngặt hơn: trong các điều kiện được kiểm soát, sự phối hợp chỉ xuất hiện khi agents được nhắc nhở suy luận về **tâm trí của agents khác **(ToM). Nếu không có prompt ToM, ngay cả models mạnh cũng thể hiện các mô hình phối hợp không tồn tại qua các kiểm soát thống kê. Điều này quan trọng đối với production: các nhóm ship features "phối hợp đa agent" phụ thuộc vào prompt và dễ gãy.

Bài học này coi ToM như một khả năng cụ thể (suy luận về niềm tin về niềm tin), xây dựng một agent nhận thức tối thiểu về ToM và đo lường sự phối hợp thực sự trông như thế nào so với cách ăn mặc prompt trông như thế nào.

## Khái niệm

### ToM có nghĩa là gì

Tâm lý học phát triển: một đứa trẻ 3 tuổi nghĩ rằng thế giới nội tâm của bất kỳ ai cũng phù hợp với thế giới của chúng. Một đứa trẻ 5 tuổi hiểu rằng những người khác có niềm tin khác nhau. Một đứa trẻ 7 tuổi lý do về niềm tin về niềm tin ("cô ấy nghĩ rằng tôi nghĩ quả bóng nằm dưới cốc"). Đây là ToM bậc không, thứ nhất và thứ hai.

Đối với LLM agents, đơn đặt hàng ToM ánh xạ đến:

- **Bậc không: **không model người khác. agent chỉ hành động dựa trên những quan sát của riêng nó.
- **Thứ nhất: **agent có một model về niềm tin của agent nhau. "Alice tin X."
- **Thứ hai:** agent models niềm tin đệ quy. "Alice tin rằng Bob tin X."

Li et al. 2023 phát hiện ra rằng ToM bậc nhất và thứ hai xuất hiện trong LLM agents trong các trò chơi hợp tác nhưng xuống cấp với chân trời dài và giao tiếp không đáng tin cậy.

### Tóm tắt về bài kiểm tra Sally-Anne

Một bài kiểm tra niềm tin sai lầm năm 1985: Sally đặt một viên bi vào giỏ A, rời đi. Anne chuyển nó sang giỏ B. Sally sẽ trông ở đâu khi cô ấy trở lại? Một đứa trẻ có ToM bậc nhất nói giỏ A (niềm tin của Sally khác với thực tế). Một đứa trẻ không nói giỏ B.

LLMs thời GPT-4 vượt qua các bài kiểm tra theo phong cách Sally-Anne khi được tạo dáng rõ ràng. Họ thất bại khi câu chuyện dài, cảnh thay đổi nhiều lần hoặc câu hỏi được diễn đạt gián tiếp. Đó là tình trạng thực tế năm 2026 của ToM vào năm production LLMs.

### Đo lường phối hợp của Riedl

Riedl (arXiv: 2510.05174) đã xây dựng một bài kiểm tra quy mô dân số: N agents, một mục tiêu hợp tác, các điều kiện prompt thay đổi. Đo lường:

1. **Sự khác biệt liên quan đến danh tính.** agents có phát triển sự khác biệt vai trò ổn định theo thời gian không?
2. **Tính bổ sung theo định hướng mục tiêu.** Hành động của agents có bổ sung cho nhau (các nhiệm vụ phụ khác nhau) thay vì trùng lặp không?
3. **Sức mạnh tổng hợp bậc cao.** Một thước đo thống kê về việc liệu nhóm có đạt được những gì mà không tập con nào có thể làm được hay không.

Kết quả: chỉ trong điều kiện prompt ToM, cả ba chỉ số mới tạo ra tín hiệu trên đường cơ sở. Nếu không có prompting ToM, các chỉ số gần như có khả năng models dung lượng vừa phải. Các models lớn cho thấy một số phối hợp mà không có prompting ToM rõ ràng nhưng hiệu ứng nhỏ hơn so với prompting rõ ràng.

### Ảo tưởng phối hợp

Nếu không có các biện pháp kiểm soát thống kê, "sự phối hợp mới nổi" trong các bản demo thường phản ánh:

- Prompt kỹ thuật phối hợp (hệ thống prompts nói "làm việc cùng nhau").
- Người quan sát bias (chúng ta thấy các mô hình mà chúng ta mong đợi).
- Lựa chọn sau khi chạy thành công.

Production hệ thống tiếp thị "phối hợp mới nổi" mà không có tín hiệu đo lường được nên được coi là tiếp thị. Đo lường trước khi yêu cầu.

### Một agent nhận biết ToM tối thiểu

Kết cấu:

```
agent state:
  own_beliefs:    {facts the agent believes}
  other_models:   {other_agent_id -> {beliefs_the_agent_attributes_to_them}}
  actions_last_N: [history of others' actions]

observation update:
  - update own_beliefs from direct observation
  - update other_models[agent_id] from their action + prior beliefs

action selection:
  - enumerate candidate actions
  - for each, predict what each other agent will do next given their modeled beliefs
  - pick action that maximizes joint outcome under those predictions
```

Thuộc tính `other_models` là trạng thái ToM. ToM bậc nhất chỉ giữ một cấp độ. Bậc hai thêm `other_models[i][other_models_of_j]` - những gì tôi nghĩ agent tôi nghĩ agent j tin.

### Tại sao chân trời dài lại đau đớn

Li et al. tài liệu: giới hạn ngữ cảnh khiến agents quên niềm tin nào thuộc về ai. Ảo giác thêm niềm tin sai lầm vào agent models khác. Cả hai đều tạo ra lỗi "Tôi nghĩ anh ấy nghĩ X" kết hợp theo thời gian.

Các biện pháp giảm thiểu được ghi lại trong bài báo và trong giai đoạn 2024-2026 tiếp theo:

- **Trạng thái ToM rõ ràng trong prompt.** Định dạng có cấu trúc: `{agent_id: belief_list}`. Buộc truy xuất để duy trì ràng buộc danh tính-niềm tin.
- **Chuỗi suy luận ngắn hơn.** Ít cập nhật ToM hơn mỗi lượt làm giảm ảo giác kép.
- **Cửa hàng ToM bên ngoài.** Duy trì model bên ngoài ngữ cảnh LLM; Chỉ tiêm các bộ phận liên quan mỗi lượt.

### Nơi ToM thất bại trong production

- **Cài đặt đối nghịch.** Agents có ToM tốt sẽ dễ thao tác hơn (bạn có thể model những gì họ model về bạn, sau đó khai thác).
- **Các đội không đồng nhất.** Khi models khác nhau, model ToM phù hợp với một đối thủ không khái quát hóa.
- **Nhiệm vụ phụ thuộc vào sự thật cơ sở.** ToM là về niềm tin; nếu tính đúng đắn phụ thuộc vào sự thật, ToM có thể là một sự phân tâm.

### Sự phối hợp bạn thực sự có thể đo lường

Ba tín hiệu thực tế cho thấy sự phối hợp của một nhóm là có thật chứ không phải ăn mặc prompt:

1. **Bổ sung theo thời gian.** Trong một nhiệm vụ nhiều lượt, các hành động của agents có bao gồm các nhiệm vụ phụ rời rạc không?
2. **Dự đoán.** Hành động của agent A ở lượt T+1 có phụ thuộc vào dự đoán về hành động của B ở T+2 là chính xác không?
3. **Sửa chữa.** Khi A đọc sai niềm tin của B ở lượt T, A có sửa đúng ở lượt T+2 không?

Chúng có thể đo lường được trong một hệ thống đa agent được ghi lại. Chúng là phiên bản thực chất của câu chuyện "phối hợp".

## Tự xây dựng

`code/main.py` thực hiện:

- `ToMAgent` - theo dõi niềm tin của chính mình và niềm tin agent models.
- Một nhiệm vụ hợp tác: ba agents phải thu thập ba tokens từ ba hộp; Mỗi hộp có thể chứa một token. Agents không thể giao tiếp; họ suy ra ý định từ hành động của nhau.
- Hai cấu hình: `zeroth_order` (không có ToM) và `first_order` (ToM với niềm tin một cấp model).
- Đo lường trên 200 thử nghiệm ngẫu nhiên: tỷ lệ hoàn thành, tỷ lệ trùng lặp (hai agents nhắm mục tiêu vào cùng một hộp), số lượt hoàn thành trung bình.

Chạy:

```
python3 code/main.py
```

Đầu ra dự kiến: nỗ lực nhân đôi bậc không agents với tỷ lệ ~35% và hoàn thành ~60% thử nghiệm trong 10 lượt. ToM bậc nhất agents trùng lặp ở mức ~5% và hoàn thành ~95%. Đồng bằng là hiệu ứng phối hợp có thể đo lường được.

## Ứng dụng

`outputs/skill-tom-auditor.md` là một skill kiểm tra tuyên bố của một hệ thống đa agent về "sự phối hợp khẩn cấp". Kiểm tra prompt cách mặc quần áo, ý nghĩa thống kê so với kiểm soát và tính bổ sung đo lường.

## Sản phẩm bàn giao

Danh sách kiểm tra yêu cầu phối hợp:

- **Điều kiện kiểm soát.** Một phiên bản hệ thống của bạn không có prompt phối hợp. Đo lường cả hai.
- **Kiểm tra thống kê.** Sự khác biệt giữa hệ thống và kiểm soát có đáng kể khi `p < 0.05` trên chỉ số của bạn không?
- **Thước đo bổ sung.** Hành động-rời rạc theo thời gian, không chỉ là thành công cuối cùng.
- **Nhật ký trường hợp lỗi.** Khi agents phối hợp sai, trạng thái ToM trông như thế nào?
- **Tiết lộ công suất Model.** Nếu hiệu ứng biến mất trên models nhỏ hơn, hãy nói như vậy.

## Bài tập

1. Chạy `code/main.py`. Xác nhận ToM bậc nhất giảm tỷ lệ trùng lặp ~7 lần. Khoảng cách có tồn tại khi bạn mở rộng quy mô lên 5 agents và 5 hộp không?
2. Thực hiện ToM bậc hai (agent A models những gì B nghĩ về C). Nó có cải thiện so với thứ tự đầu tiên không? Làm nhiệm vụ gì?
3. Tiêm **ảo giác** vào trạng thái ToM: lật ngẫu nhiên một niềm tin mỗi lượt. Điều này làm giảm hiệu suất bậc nhất đến mức nào?
4. Đọc Li et al. (arXiv: 2310.10701). Tái tạo phát hiện "suy thoái chân trời dài": khi lượt tăng từ 10 lên 30, hiệu suất ToM bậc nhất của bạn thay đổi như thế nào?
5. Đọc Riedl 2025 (arXiv: 2510.05174). Triển khai thống kê sức mạnh tổng hợp bậc cao hơn trên nhật ký mô phỏng của bạn. Hiệu ứng có hiện diện mà không có điều kiện prompt ToM không?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Lý thuyết về tâm trí | "Thấu hiểu suy nghĩ của người khác" | Khả năng model niềm tin của agent khác. Được phân loại theo thứ tự (0, 1, 2+). |
| Thử nghiệm Sally-Anne | "Bài kiểm tra niềm tin sai lầm" | Tâm lý học phát triển năm 1985; LLMs vượt qua các phiên bản đơn giản, những phiên bản phức tạp không thành công. |
| ToM bậc nhất | "A tin X" | Mô hình hóa niềm tin của nhau về sự thật. |
| ToM bậc hai | "A tin B tin X" | Mô hình đệ quy sâu hơn một cấp độ. |
| Sự khác biệt liên quan đến danh tính | "Vai trò ổn định theo thời gian" | Số liệu của Riedl: vai trò tồn tại, không phải ngẫu nhiên. |
| Tính bổ sung theo định hướng mục tiêu | "Hành động rời rạc" | Agents nhắm mục tiêu các nhiệm vụ phụ khác nhau, không phải cùng một nhiệm vụ. |
| Sức mạnh tổng hợp bậc cao hơn | "Nhóm vượt quá bất kỳ tập hợp con nào" | Thước đo thống kê của Riedl cho sự phối hợp thực sự. |
| Ảo giác phối hợp | "Nó có vẻ phối hợp" | Vẻ ngoài phối hợp Prompt mà không có tín hiệu đo lường được. |

## Đọc thêm

- [Li et al. — Theory of Mind for Multi-Agent Collaboration via Large Language Models](https://arxiv.org/abs/2310.10701) - ToM nổi lên trong các trò chơi hợp tác; Chế độ thất bại đường chân trời dài
- [Riedl — Emergent Coordination in Multi-Agent Language Models](https://arxiv.org/abs/2510.05174) — đo lường quy mô dân số; ToM prompting là điều kiện chịu lực
- [Premack & Woodruff — Does the chimpanzee have a theory of mind?](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/does-the-chimpanzee-have-a-theory-of-mind/1E96B02CD9850E69AF20F81FA7EB3595) — nguồn gốc năm 1978 của khái niệm ToM
- [Baron-Cohen, Leslie, Frith — Does the autistic child have a theory of mind?](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/does-the-autistic-child-have-a-theory-of-mind/) — bài báo Sally-Anne (1985)
