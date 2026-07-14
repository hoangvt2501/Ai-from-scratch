# Tree of Thoughts và LATS: Tìm kiếm có chủ ý

> Một quỹ đạo chain-of-thought duy nhất không có chỗ để quay trở lại. ToT (Yao và cộng sự, 2023) biến lý luận thành một cái cây với khả năng tự đánh giá trên mỗi nút. LATS (Zhou và cộng sự, 2024) hợp nhất ToT với ReAct và Reflexion theo Monte Carlo Tree Search. Trò chơi 24 đi từ 4% (CoT) lên 74%(ToT); LATS đạt 92.7% pass@1 trên HumanEval.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 14 · 01 (Vòng lặp Agent), Giai đoạn 14 · 03 (Phản xạ)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Lý luận khung như tìm kiếm: các nút là "suy nghĩ", các cạnh là "mở rộng", giá trị là "hứa hẹn như thế nào".
- Triển khai tìm kiếm cây BFS kiểu ToT stdlib với tính điểm tự đánh giá.
- Mở rộng đến vòng lặp LATS MCTS đồ chơi với chọn / mở rộng / mô phỏng / lan truyền ngược.
- Quyết định khi nào tìm kiếm xứng đáng với hệ số nhân token (Game of 24, tạo mã) và khi nào một quỹ đạo duy nhất là đủ (Hỏi & Đáp đơn giản).

## Vấn đề

Chain-of-thought là một cuộc đi bộ tuyến tính. Nếu bước đầu tiên sai, mọi bước tiếp theo đều hoạt động trên một tiền đề xấu. Trong Ván 24 (sử dụng bốn chữ số với + − × ÷ để tạo thành 24), GPT-4 CoT đạt 4% accuracy. model chọn sai biểu thức phụ sớm và không thể khôi phục.

Những gì lý luận cần là khả năng đề xuất nhiều ứng viên, đánh giá họ, chọn những ứng cử viên đầy hứa hẹn và quay trở lại khi ngõ cụt xuất hiện. Đó là tìm kiếm. Tree of Thoughts và LATS là hai công thức kinh điển.

## Khái niệm

### Cây suy nghĩ (Yao và cộng sự, NeurIPS 2023)

Mỗi nút là một bước trung gian mạch lạc ("một suy nghĩ"). Mỗi nút có thể mở rộng thành K suy nghĩ con. LLM tự đánh giá từng nút với prompt tính điểm. Tìm kiếm khám phá cây - BFS, DFS hoặc chùm tia.

```
                     (root: "find 24 from 4 6 4 1")
                    /               |            \
           ("6 - 4 = 2")    ("4 + 1 = 5")    ("4 * 6 = 24")  <- Score: HIGH
              /   \              |                  |
          ...    ...          ...                finish
```

Tự đánh giá là phần chịu lực. Bài báo cho thấy ba biến thể: phân loại `sure / likely / impossible`, điểm số `1..10` và bỏ phiếu giữa các ứng cử viên. Cả ba đều đánh bại CoT đáng kể trong Game of 24 (4% -> 74% với GPT-4).

### LATS (Chu và cộng sự, ICML 2024)

LATS hợp nhất ToT, ReAct và Reflexion theo MCTS. LLM đóng ba vai trò:

- **Policy**: đề xuất các hành động tiếp theo của ứng cử viên (kiểu ReAct).
- **Chức năng giá trị**: ghi điểm một quỹ đạo một phần (tự đánh giá kiểu ToT).
- **Tự phản ánh**: khi thất bại, hãy viết một phản ánh bằng ngôn ngữ tự nhiên (Kiểu phản xạ) và sử dụng nó để gieo lại rollouts trong tương lai.

Phản hồi môi trường (quan sát) trộn lẫn vào hàm giá trị để tìm kiếm được thông báo bởi kết quả công cụ thực, không chỉ model ý kiến. Kết quả tại thời điểm giấy: HumanEval pass@1 92,7% với GPT-4 (SOTA), WebShop trung bình 75,9 với GPT-3.5 (tiếp cận fine-tuning dựa trên gradient).

### MCTS, tối thiểu

Bốn giai đoạn cho mỗi lần lặp:

1. **Chọn **- đi bộ từ gốc đến lá bằng cách sử dụng UCT (độ tin cậy trên ràng buộc với cây).
2. **Mở rộng** - tạo K trẻ em thông qua policy.
3. **Mô phỏng** - rollout từ một đứa trẻ sử dụng policy, chấm điểm chiếc lá bằng hàm giá trị (hoặc phần thưởng môi trường).
4. **Backpropagate** — cập nhật số lượt truy cập và ước tính giá trị trên đường dẫn.

Công thức UCT: `Q(s, a) + c * sqrt(ln N(s) / N(s, a))`. Thuật ngữ đầu tiên là bóc lột; thứ hai là khám phá. Điều chỉnh `c` cho mỗi tác vụ.

### Thực tế chi phí

Tìm kiếm bùng nổ tokens. ToT trong Game of 24 sử dụng 100–1000 lần tokens của CoT. LATS cũng tương tự. Điều này không miễn phí; Tìm kiếm dự trữ cho:

- Các nhiệm vụ mà một quỹ đạo duy nhất rõ ràng là không đủ (Game of 24, mã phức tạp).
- Các nhiệm vụ mà đồng hồ treo tường ít quan trọng hơn tính chính xác.
- Các tác vụ có hàm giá trị rẻ, đáng tin cậy (kiểm tra đơn vị cho mã, mục tiêu rõ ràng cho toán học).

Nếu nhiệm vụ của bạn có một câu trả lời đúng duy nhất và một người đánh giá ồn ào, tìm kiếm thường làm cho mọi thứ trở nên tồi tệ hơn - nó tìm thấy một câu trả lời sai "điểm cao".

### Định vị năm 2026

Hầu hết production agents không chạy LATS. Họ chạy ReAct với xác minh dựa trên công cụ (CRITIC, Bài 05). Tìm kiếm hiển thị trong các ngách chuyên biệt:

- Mã hóa agents chạy thử nghiệm dưới dạng hàm giá trị (kiểu HumanEval).
- Nghiên cứu sâu agents khám phá nhiều đường dẫn truy vấn.
- Quy trình làm việc nặng về lập kế hoạch bên trong biểu đồ con LangGraph.

AlphaEvolve (Bài học 11) là năm 2025: tìm kiếm tiến hóa qua mã, tính phù hợp có thể kiểm tra bằng máy, lợi nhuận biên giới (cải tiến matmul 4x4 đầu tiên trong 56 năm).

## Tự xây dựng

`code/main.py` thực hiện:

- Một ToT BFS nhỏ bé trong một nhiệm vụ "chọn hoạt động số học" cách điệu.
- Một vòng lặp LATS MCTS đồ chơi trên cùng một tác vụ (Chọn / Mở rộng / Mô phỏng / Backpropagate) với lựa chọn UCT.
- Một hàm giá trị bao gồm một điểm tượng trưng cộng với một điểm tự đánh giá.

Chạy nó:

```
python3 code/main.py
```

trace cho thấy ToT mở rộng ba ứng cử viên trên mỗi nút với BFS, so với LATS hội tụ trên rollout tốt nhất thông qua MCTS. Token số đếm được in cho cả hai.

## Ứng dụng

LangGraph ships khám phá kiểu ToT dưới dạng mẫu đồ thị con; blog của nhóm LangChain về LATS (Tháng Năm 2024) là hướng dẫn tham khảo. LlamaIndex ships một `TreeOfThoughts` agent. Đối với hầu hết năm 2026 production agents mô hình này tồn tại sau một cổng `if task_complexity > threshold: use_search()` - xem mô hình đánh giá optimizer trong Bài 05.

## Sản phẩm bàn giao

`outputs/skill-search-policy.md` chọn giữa ReAct tuyến tính, ToT, LATS và tìm kiếm tiến hóa với hình dạng nhiệm vụ, ngân sách và độ trung thực của người đánh giá.

## Bài tập

1. Chạy đồ chơi LATS với UCT c = 0.1 so với c = 2.0. Những thay đổi nào trong trace?
2. Hoán đổi hàm giá trị cho người ghi bàn ồn ào hơn (thêm jitter ngẫu nhiên). MCTS có còn tìm thấy lá tốt nhất không? Tín hiệu trên nhiễu tối thiểu mà nó chịu được là bao nhiêu?
3. Triển khai ToT tìm kiếm chùm tia (giữ top-k ở mỗi cấp độ) và so sánh với BFS. Cái nào tốt hơn với ngân sách token eo hẹp?
4. Đọc LATS Phần 5.1. Tái tạo số lượng quỹ đạo HumanEval: mất bao nhiêu rollouts để đạt được pass@1 được báo cáo?
5. Đọc cuộc thảo luận của bài báo LATS về "khi LATS giúp ích ít hơn". Viết hình dạng nhiệm vụ ánh xạ quy tắc quyết định một đoạn vào chiến lược tìm kiếm.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Cây tư tưởng | "CoT phân nhánh" | Yao et al. — cây của các nút suy nghĩ với khả năng tự đánh giá |
| LÁT | "MCTS cho LLMs" | Zhou và cộng sự - thống nhất ToT + ReAct + Reflexion theo MCTS |
| UCT | "Giới hạn độ tin cậy trên" | Lựa chọn công thức, cân bằng, khai thác (Q) và thăm dò (ln N/n) |
| Hàm giá trị | "Trạng thái này tốt như thế nào" | Được nhắc điểm LLM hoặc phần thưởng môi trường; nguồn cấp dữ liệu backprop |
| Policy | "Người đề xuất hành động" | Máy phát điện kiểu ReAct; Phát hành ứng cử viên vào thoughts/actions tới |
| Rollout | "Quỹ đạo mô phỏng" | Đi từ nút này sang lá khác bằng cách sử dụng policy, ghi điểm với giá trị |
| Tuyên truyền ngược | "Cập nhật tổ tiên" | Đẩy phần thưởng của chiếc lá lên đường, cập nhật số lượt truy cập và Q |
| Chi phí tìm kiếm | "Token vụ nổ" | 100-1000x CoT trong Game of 24; ngân sách trước khi bạn nhận nuôi |

## Đọc thêm

- [Yao et al., Tree of Thoughts (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) — bài báo kinh điển
- [Zhou et al., LATS (arXiv:2310.04406)](https://arxiv.org/abs/2310.04406) - MCTS với phản hồi Phản xạ
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — các mẫu biểu đồ con để tìm kiếm
- [AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) - tìm kiếm tiến hóa với các công cụ đánh giá lập trình
