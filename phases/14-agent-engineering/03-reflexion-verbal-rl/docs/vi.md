# Phản xạ: Học tăng cường bằng lời nói

> RL dựa trên Gradient cần hàng nghìn bản thử nghiệm và một cụm GPU để khắc phục chế độ lỗi. Reflexion (Shinn và cộng sự, NeurIPS 2023) thực hiện điều đó bằng ngôn ngữ tự nhiên: sau mỗi lần thử nghiệm thất bại, agent viết một phản ánh, lưu trữ nó trong bộ nhớ từng đợt và điều kiện thử nghiệm tiếp theo trên bộ nhớ đó. Đây là mô hình đằng sau tính toán thời gian ngủ của Letta, các bài học CLAUDE.md của Claude Code và quy tắc học tập của quy trình làm việc chuyên nghiệp.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 14 · 01 (Vòng lặp Agent), Giai đoạn 14 · 02 (Quay lại)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Kể tên ba thành phần của Reflexion (Actor, Evaluator, Self-Reflector) và vai trò của trí nhớ theo từng tập.
- Triển khai vòng lặp phản xạ stdlib với trình đánh giá nhị phân, bộ đệm phản xạ và các lần thử lại mới.
- Chọn giữa các nguồn phản hồi vô hướng, phỏng đoán và tự đánh giá cho một nhiệm vụ nhất định.
- Giải thích lý do tại sao củng cố bằng lời nói bắt được lỗi mà RL dựa trên gradient sẽ cần hàng nghìn lần thử nghiệm để sửa chữa.

## Vấn đề

Một agent thất bại trong một nhiệm vụ. Trong RL tiêu chuẩn, bạn sẽ chạy thêm hàng nghìn bản dùng thử, tính toán gradients, cập nhật trọng số. Đắt tiền, chậm chạp và hầu hết các production agents không có ngân sách training cho mỗi thất bại.

Reflexion (Shinn et al., arXiv:2303.11366) đặt ra một câu hỏi khác: điều gì sẽ xảy ra nếu agent chỉ nghĩ về lý do tại sao nó thất bại và thử lại với suy nghĩ đó trong prompt? Không cập nhật trọng lượng. Không gradient. Chỉ là ngôn ngữ tự nhiên được lưu trữ giữa các thử nghiệm.

Kết quả: trên ALFWorld, nó đánh bại ReAct và các đường cơ sở không fine-tuned khác. Trên HotpotQA, nó được cải thiện so với ReAct. Khi tạo mã (HumanEval/MBPP), nó thiết lập hiện đại tại thời điểm đó. Tất cả không có một bước gradient nào.

## Khái niệm

### Ba thành phần

```
Actor         : generates a trajectory (ReAct-style loop)
Evaluator     : scores the trajectory — binary, heuristic, or self-eval
Self-Reflector: writes a natural-language reflection on the failure
```

Cộng với một cấu trúc dữ liệu:

```
Episodic memory: list of prior reflections, prepended to the next trial's prompt
```

Một thử nghiệm chạy Diễn viên. Người đánh giá chấm điểm nó. Nếu điểm thấp, Self-Reflector sẽ tạo ra phản xạ ("Tôi đã chọn nhầm công cụ vì tôi đọc nhầm câu hỏi là hỏi về X khi nó hỏi về Y"). Sự phản chiếu đi vào ký ức từng tập. Phiên tòa tiếp theo bắt đầu lại nhưng nhìn thấy sự phản chiếu.

### Ba loại đánh giá

1. **Vô hướng** — một tín hiệu nhị phân bên ngoài. ALFWorld thành công hoặc thất bại. Các bài kiểm tra HumanEval đạt hoặc không đạt. Đơn giản nhất, tín hiệu cao nhất.
2. **Heuristic** — chữ ký lỗi được xác định trước. "Nếu agent tạo ra cùng một hành động hai lần liên tiếp, hãy đánh dấu là bị kẹt." "Nếu quỹ đạo vượt quá 50 bước, hãy đánh dấu là không hiệu quả."
3. **Tự đánh giá** - LLM ghi điểm quỹ đạo của riêng nó. Cần thiết khi không có ground truth. Tín hiệu yếu hơn; kết hợp tốt với xác minh dựa trên công cụ (Bài 05 - PHÊ BÌNH).

Mặc định năm 2026 là một sự kết hợp: vô hướng khi có sẵn, tự đánh giá khi không, phỏng đoán như đường ray an toàn.

### Tại sao điều này khái quát hóa

Phản xạ không phải là một thuật toán mới mà là một mô hình được đặt tên. Hầu hết mọi agent "tự phục hồi" production đều chạy một số biến thể:

- Tính toán thời gian ngủ của Letta (Bài 08): một agent riêng biệt phản ánh các cuộc trò chuyện trong quá khứ và ghi vào các khối bộ nhớ.
- Mô hình `CLAUDE.md` / "lưu bộ nhớ" của Claude Code: phản ánh được ghi lại dưới dạng bài học, được thêm vào sessions trong tương lai.
- Lệnh `/learn-rule` của Pro-Workflow: Các chỉnh sửa được ghi lại dưới dạng quy tắc rõ ràng.
- Các nút phản xạ của LangGraph: một nút chấm điểm đầu ra và các tuyến để tinh chỉnh nếu cần.

Tất cả đều xuất phát từ cùng một cái nhìn sâu sắc: ngôn ngữ tự nhiên là một phương tiện đủ phong phú để mang "những gì tôi học được từ thất bại" giữa các lần chạy.

### Khi nào nó hoạt động và khi nào nó không

Phản xạ hoạt động khi:

- Có tín hiệu thất bại rõ ràng (kiểm tra thất bại, lỗi công cụ, đáp án sai).
- Nhiệm vụ class có thể lặp lại (có thể hỏi lại cùng một loại câu hỏi).
- Sự phản ánh có chỗ để cải thiện quỹ đạo (đủ ngân sách hành động).

Phản xạ không giúp ích gì khi:

- agent đã thành công trong lần thử đầu tiên.
- Lỗi là bên ngoài (mạng ngừng hoạt động, công cụ bị hỏng) - phản ánh về "mạng đã ngừng hoạt động" không giúp ích gì cho việc chạy trong tương lai.
- Sự phản ánh biến thành mê tín dị đoan - lưu trữ một câu chuyện về một cuộc chạy trốn bong tróc một lần.

Cạm bẫy năm 2026: Ký ức thối rữa. Phản xạ tích lũy; một số lỗi thời hoặc sai; chạy lại chậm hơn khi bộ đệm theo từng tập tăng lên. Giảm thiểu: nén chặt định kỳ (Bài 06), TTL về phản xạ hoặc agent dọn dẹp thời gian ngủ riêng biệt (Letta).

```figure
react-trace
```

## Tự xây dựng

`code/main.py` triển khai Reflexion trên một câu đố đồ chơi: tạo ra một danh sách 3 yếu tố tổng cho một mục tiêu. Diễn viên phát ra danh sách ứng cử viên; Người đánh giá kiểm tra tổng; Người tự phản xạ viết một dòng về những gì đã xảy ra. Sự phản ánh đi vào ký ức từng giai đoạn cho phiên tòa tiếp theo.

Các thành phần:

- `Actor` - một policy có kịch bản cải thiện khi nhìn thấy phản chiếu.
- `Evaluator.binary()` — pass/fail trên tổng mục tiêu.
- `SelfReflector` - tạo ra chẩn đoán một dòng về lỗi.
- `EpisodicMemory` — một danh sách có giới hạn với ngữ nghĩa TTL.

Chạy nó:

```
python3 code/main.py
```

trace cho thấy ba phiên tòa. Thử nghiệm 1 thất bại, phản ánh được lưu trữ, thử nghiệm 2 nhìn thấy phản ánh và cải thiện nhưng vẫn thất bại, thử nghiệm 3 thành công. So sánh với một lần chạy cơ bản (không phản ánh) - nó vẫn bị mắc kẹt ở câu trả lời của thử nghiệm 1.

## Ứng dụng

LangGraph ships phản xạ dưới dạng một mẫu nút. Lệnh `/memory` của Claude Code và `/learn-rule` của quy trình làm việc chuyên nghiệp bên ngoài bộ đệm theo từng tập dưới dạng tệp đánh dấu. Tính toán thời gian ngủ của Letta chạy Self-Reflector trong thời gian ngừng hoạt động để agent chính vẫn bị giới hạn về độ trễ. OpenAI Agents SDK không ship Phản xạ trực tiếp; Bạn xây dựng nó với một Guardrail tùy chỉnh từ chối quỹ đạo theo điểm số và `Session` bộ nhớ tồn tại qua các lần chạy.

## Sản phẩm bàn giao

`outputs/skill-reflexion-buffer.md` tạo và duy trì bộ đệm theo từng tập với tính năng chụp phản chiếu, TTL và loại bỏ trùng lặp. Cho một nhiệm vụ class và một thất bại, nó phát ra một phản ánh thực sự giúp ích cho thử nghiệm tiếp theo (không phải là một câu nói chung chung "hãy cẩn thận hơn").

## Bài tập

1. Chuyển từ công cụ đánh giá nhị phân sang vô hướng trả về số liệu khoảng cách (khoảng cách từ mục tiêu). Nó có hội tụ nhanh hơn không?
2. Thêm TTL gồm 10 thử nghiệm vào phản ánh. Những suy nghĩ cũ hơn có gây tổn thương hay giúp ích sau thời điểm đó không?
3. Triển khai trình đánh giá heuristic: đánh dấu thử nghiệm là bị kẹt nếu hành động tương tự lặp lại. Điều này tương tác với Self-Reflector như thế nào?
4. Chạy Reflexion với một Actor đối nghịch bỏ qua phản xạ. Sự phản ánh tối thiểu prompt kỹ thuật buộc Actor phải chú ý đến chúng là gì?
5. Đọc Phần 4 của bài báo Reflexion trên AlfWorld. Tái tạo cải thiện tỷ lệ thành công 130% theo khái niệm: Delta vs vanilla ReAct là gì?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Phản xạ | "Tự sửa" | Shinn et al. 2023 — Diễn viên, Người đánh giá, Tự phản xạ cộng với trí nhớ theo từng tập |
| Củng cố bằng lời nói | "Học mà không cần gradients" | Phản ánh ngôn ngữ tự nhiên được đặt trước cho prompt thử nghiệm tiếp theo |
| Trí nhớ theo từng giai đoạn | "Phản ánh cho mỗi nhiệm vụ" | Bộ đệm có giới hạn của phản xạ prior cho một tác vụ class |
| Công cụ đánh giá vô hướng | "Tín hiệu thành công nhị phân" | Điểm Pass/fail hoặc số từ ground truth |
| Công cụ đánh giá heuristic | "Máy dò dựa trên mẫu" | Chữ ký lỗi được xác định trước (ví dụ: vòng lặp bị kẹt, quá nhiều bước) |
| Người tự đánh giá | "LLM với tư cách là thẩm phán trên chính trace" | Dự phòng tín hiệu thấp hơn khi không có ground truth - ghép nối với xác minh dựa trên công cụ |
| Ký ức thối rữa | "Phản ánh cũ" | Bộ đệm theo từng đợt chứa đầy các mục lỗi thời; Sửa chữa bằng compaction/TTL |
| Phản ánh thời gian ngủ | "Tự phản ánh không đồng bộ" | Chạy Self-Reflector off the hot path để agent chính luôn nhanh |

## Đọc thêm

- [Shinn et al., Reflexion: Language Agents with Verbal Reinforcement Learning (arXiv:2303.11366)](https://arxiv.org/abs/2303.11366) — bài báo kinh điển
- [Letta, Sleep-time Compute](https://www.letta.com/blog/sleep-time-compute) — phản xạ không đồng bộ trong production
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) - quản lý bộ đệm theo từng tập như một phần của ngữ cảnh
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — mẫu nút phản xạ
