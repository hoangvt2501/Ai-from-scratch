# Capstone 17 — Gia sư AI cá nhân (Thích ứng, Đa phương thức, có bộ nhớ)

> Khanmigo (Khan Academy), Duolingo Max, Google LearnLM / Gemini for Education, Quizlet Q-Chat và Synthesis Tutor đều shipped dạy kèm đa phương thức thích ứng trên quy mô lớn vào năm 2026. Hình dạng phổ biến là một policy Socrates (không bao giờ chỉ bỏ câu trả lời), một model người học cập nhật sau mỗi lần tương tác (phong cách theo dõi kiến thức Bayes), giọng nói + văn bản + đầu vào toán học, truy xuất biểu đồ chương trình giảng dạy, lập lịch lặp lại cách nhau và các bộ lọc an toàn cứng cho nội dung phù hợp với lứa tuổi. Điểm mấu chốt là ship một gia sư theo chủ đề cụ thể (đại số K-12 hoặc Python giới thiệu), thực hiện một nghiên cứu hiệu quả kéo dài hai tuần với 10 người học và vượt qua cuộc kiểm tra an toàn nội dung.

**Loại:** Đá đỉnh
**Ngôn ngữ:** Python (backend, learner model), TypeScript (web app), SQL (curriculum graph via Postgres + Neo4j)
**Kiến thức tiên quyết:** Giai đoạn 5 (NLP), Giai đoạn 6 (phát biểu), Giai đoạn 11 (LLM kỹ thuật), Giai đoạn 12 (đa phương thức), Giai đoạn 14 (agents), Giai đoạn 17 (cơ sở hạ tầng), Giai đoạn 18 (an toàn)
**Các giai đoạn thực hiện: **P5 · P6 · P11 · P12 · P14 · P17 · Trang 18
**Thời lượng:** 30 giờ

## Vấn đề

Dạy kèm thích ứng từng là một lĩnh vực nghiên cứu công nghệ giáo dục. Đến năm 2026, nó là một sản phẩm tiêu dùng. Khanmigo được triển khai trên hầu hết các khu học chánh của Hoa Kỳ. Duolingo Max đạt hàng chục triệu MAU. LearnLM / Gemini for Education của Google hỗ trợ dạy kèm trong Google Classroom. Quizlet Q-Chat nằm bên cạnh thẻ ghi nhớ. Synthesis Tutor đã trở nên lan truyền với gia sư dành cho những đứa trẻ tò mò. Các yếu tố phổ biến: đầu vào đa phương thức (phương trình nhập, nói, ảnh), sư phạm Socrates (hỏi trước, giải thích sau), model người học cập nhật sau mỗi tương tác và an toàn nghiêm ngặt phù hợp với lứa tuổi.

Bạn sẽ xây dựng một trong những thứ này cho một nhóm cụ thể. Thanh đo lường là một nghiên cứu hiệu quả thực tế: điểm trước và sau kiểm tra trong hai tuần với 10 người học. Vòng lặp giọng nói phải cảm thấy tự nhiên (capstone 03 sub-stack). Bộ nhớ phải tôn trọng quyền riêng tư. Bộ lọc an toàn phải vượt qua đội đỏ nhận biết COPPA cho K-12.

## Khái niệm

Bốn thành phần. **Gia sư policy **là một vòng lặp Socrates: khi người học hỏi câu trả lời, policy hỏi một câu hỏi dẫn dắt; khi họ làm đúng, nó sẽ chuyển sang khái niệm tiếp theo; Khi chúng bị mắc kẹt, nó cung cấp một gợi ý giàn giáo. **Learner model** là truy tìm kiến thức Bayes (hoặc một biến thể đơn giản) cập nhật xác suất thành thạo trên mỗi nút chương trình giảng dạy sau mỗi tương tác. **Biểu đồ chương trình giảng dạy** là một Neo4j của các khái niệm với các cạnh tiên quyết; policy đi trên biểu đồ để chọn khái niệm tiếp theo. **Bộ nhớ** là một kho lưu trữ ngữ nghĩa + theo từng giai đoạn (kiểu bộ nhớ tác nhân) chứa các tương tác, sai lầm và sở thích trong quá khứ.

UX là đa phương thức. Nhập văn bản cho câu trả lời đã nhập. Nhập liệu bằng giọng nói qua LiveKit + Whisper (sử dụng lại capstone 03). Nhập ảnh cho các bài toán qua dấu chấm. ocr hoặc PaliGemma 2. Đầu ra giọng nói qua Cartesia Sonic-2. An toàn sử dụng Llama Guard 4 cộng với bộ lọc phù hợp với lứa tuổi (chặn nội dung người lớn, bạo lực, tự làm hại bản thân) và policy lưu giữ trí nhớ nhận biết COPPA.

Nghiên cứu hiệu quả là sản phẩm được phân phối. 10 học viên, trước và sau kiểm tra, hai tuần. Báo cáo học tập đạt được delta và khoảng tin cậy. So sánh với đường cơ sở không thích ứng (cùng một nội dung được phân phối tuyến tính mà không có policy gia sư).

## Kiến trúc

```
learner device
  |
  +-- text         -> web app
  +-- voice        -> LiveKit Agents (ASR + TTS)
  +-- photo math   -> dots.ocr / PaliGemma 2
       |
       v
  tutor policy (LangGraph)
       - Socratic decision head
       - next-concept chooser (curriculum graph walk)
       - hint scaffolder
       - mastery update
       |
       v
  learner model (BKT / item-response theory)
       - per-concept mastery probability
       - spaced-repetition scheduler (SM-2 or FSRS)
       |
       v
  memory (agentmemory-style)
       - episodic: every interaction
       - semantic: learned mistakes, preferences
       - retention policy: COPPA / GDPR aware
       |
       v
  curriculum graph (Neo4j)
       - prerequisite edges
       - OER content attached
       |
       v
  safety:
    Llama Guard 4 + age-appropriate filter
    memory access guarded by learner ID scope
```

## Stack

- Lựa chọn môn học: Đại số K-12 hoặc Python giới thiệu (chọn một môn cho chiều sâu)
- Gia sư policy: LangGraph qua Claude Sonnet 4.7 (với bộ nhớ đệm prompt)
- model người học: Truy tìm kiến thức Bayes (cổ điển) hoặc FSRS cho khoảng cách
- Biểu đồ chương trình giảng dạy: Neo4j của các khái niệm + các cạnh tiên quyết + nội dung OER
- Bộ nhớ: vector liên tục kiểu agentmemory + tập + kho ngữ nghĩa
- Lồng tiếng: LiveKit Agents 1.0 + Cartesia Sonic-2 (tái sử dụng capstone 03 sub-stack)
- Toán ảnh: dấu chấm. ocr hoặc PaliGemma 2 để nhận dạng phương trình
- An toàn: Llama Guard 4 + bộ lọc tùy chỉnh phù hợp với lứa tuổi
- Đánh giá: Tạo câu hỏi cấp độ nở hoa, harness kiểm tra pre/post, công cụ nghiên cứu hiệu quả

## Tự xây dựng

1. **Biểu đồ chương trình giảng dạy.** Xây dựng một Neo4j gồm 50-150 nút khái niệm (ví dụ: đại số K-12 từ "đường số" đến "công thức bậc hai") với các cạnh tiên quyết. Đính kèm nội dung OER trên mỗi nút (Open Textbook, OpenStax).

2. **Người học model.** Khởi tạo truy tìm kiến thức Bayes với priors: đoán, trượt, tốc độ học. Cập nhật thông thạo từng khái niệm sau mỗi lần tương tác. Kiên trì cho mỗi người học.

3. **Gia sư policy.** LangGraph với các nút: `read_signal` (câu trả lời của người học có đúng / một phần / bị mắc kẹt không?), `select_concept` (biểu đồ chương trình giảng dạy đi bộ chọn khái niệm ưu tiên cao nhất), `scaffold` (prompt Socrates), `update_mastery`.

4. **Ký ức.** Mọi tương tác đều ghi vào một kho lưu trữ theo tập. Sai lầm và sở thích thúc đẩy trí nhớ ngữ nghĩa. policy lưu giữ nhận biết COPPA: tự động xóa sau 1 năm, phụ huynh có thể truy cập.

5. **Đường dẫn thoại.** LiveKit Agents worker đính kèm với policy gia sư. ASR qua Whisper-v3-turbo. TTS qua Cartesia Sonic-2. Hỗ trợ sà lan (tái sử dụng cơ chế capstone 03).

6. **Đường dẫn toán ảnh.** Tải lên hoặc chụp ảnh; Chạy dấu chấm. ocr hoặc PaliGemma 2 để nhận ra phương trình; Nguồn cấp dữ liệu cho gia sư dưới dạng đầu vào có cấu trúc.

7. **An toàn.** Mỗi đầu ra model đều đi Llama Guard 4 + bộ lọc phù hợp với lứa tuổi (chặn tự làm hại bản thân, nội dung người lớn, bạo lực). Quyền truy cập bộ nhớ được giới hạn theo ID người học; Bề mặt truy cập của phụ huynh để xóa.

8. **Nghiên cứu hiệu quả.** 10 người học, bài kiểm tra trước (tiêu chuẩn hóa 30 câu hỏi cơ sở), hai tuần tương tác với gia sư (3 sessions/week), sau bài kiểm tra. So sánh với nhóm thuần tập cơ sở không thích ứng gồm 10 người học trên cùng một nội dung.

9. **Báo cáo tiến độ hàng tuần.** Mỗi người học, tự động tạo bản tóm tắt PDF về các chủ đề đã khám phá, quỹ đạo thành thạo và các bước tiếp theo được đề xuất.

## Ứng dụng

```
learner: "I don't understand why 3x + 6 = 12 means x = 2"
[signal]   stuck
[concept]  'isolating variables' (prerequisite: addition-subtraction-equality)
[scaffold] "what number would you subtract from both sides to start?"
learner: "6"
[signal]   correct
[mastery]  addition-subtraction-equality: 0.62 -> 0.77
[concept]  continue 'isolating variables'
[scaffold] "great. now what is 3x / 3 equal to?"
```

## Sản phẩm bàn giao

`outputs/skill-ai-tutor.md` là sản phẩm được giao. Một gia sư thích ứng theo chủ đề cụ thể với đầu vào đa phương thức, model của người học, trí nhớ, độ an toàn và hiệu quả đo lường.

| Trọng lượng | Tiêu chí | Cách đo lường |
| :-: | --- | --- |
| 25 | Học tập đạt được delta | Pre/post-test delta trong một nghiên cứu kéo dài hai tuần với 10 người học |
| 20 | Sự trung thành của Socrates | Điểm đánh giá trên các mẫu bảng điểm |
| 20 | UX đa phương thức | Mạch lạc giọng nói + ảnh + văn bản từ đầu đến cuối |
| 20 | Tư thế an toàn + quyền riêng tư | Tỷ lệ vượt qua Llama Guard 4 + lưu nhận COPPA |
| 15 | Độ rộng chương trình giảng dạy và chất lượng đồ thị | Phạm vi khái niệm + tính nhất quán của biểu đồ điều kiện tiên quyết |
| **100** |||

## Bài tập

1. Chạy nghiên cứu hiệu quả có và không có model người học thích ứng (thứ tự khái niệm ngẫu nhiên). Báo cáo delta. Mong đợi thích ứng để giành chiến thắng, nhưng kích thước là con số thú vị.

2. Thêm một đầu dò đa phương thức: cùng một câu hỏi khái niệm được cung cấp dưới dạng văn bản, giọng nói và ảnh. Đo lường xem người học có hội tụ nhanh hơn với phương thức họ thích hay không.

3. Xây dựng bảng điều khiển dành cho mẹ: các chủ đề đã thực hành, quỹ đạo thành thạo, các khái niệm sắp tới, các sự kiện an toàn (bất kỳ lần truy cập guardrail nào). Căn chỉnh COPPA.

4. Thêm chế độ chuyển đổi ngôn ngữ: gia sư chấp nhận đầu vào tiếng Tây Ban Nha và dạy bằng tiếng Tây Ban Nha. Đo độ phủ sóng X-Guard.

5. Nhấn mạnh quyền riêng tư của bộ nhớ: xác minh rằng người học A không thể xem dữ liệu của người học B ngay cả thông qua một cuộc tấn công nhập lại clip thoại. Ghi lại việc cố gắng truy cập và cảnh báo.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| policy Socrates | "Hỏi, đừng vứt" | Gia sư hỏi một câu hỏi dẫn dắt thay vì đưa ra câu trả lời |
| Truy tìm kiến thức Bayes | "BKT" | Phương trình model người học cổ điển cho xác suất thành thạo cho mỗi khái niệm |
| FSRS | "Bộ lập lịch lặp lại cách nhau miễn phí" | Bộ lập lịch lặp lại cách nhau 2024, tốt hơn SM-2 |
| Biểu đồ chương trình giảng dạy | "Khái niệm DAG" | Neo4j của các khái niệm với các cạnh tiên quyết |
| Trí nhớ theo từng giai đoạn | "Nhật ký mỗi tương tác" | Mọi tương tác được lưu trữ để truy xuất sau này |
| Bộ nhớ ngữ nghĩa | "Cửa hàng mẫu đã học" | Những sai lầm và sở thích được thu gọn được thúc đẩy từ nhiều tập |
| COPPA | "Luật bảo mật trẻ em" | Luật pháp Hoa Kỳ hạn chế thu thập dữ liệu từ trẻ em dưới 13 tuổi |

## Đọc thêm

- [Khanmigo (Khan Academy)](https://www.khanmigo.ai) — gia sư K-12 dành cho người tiêu dùng tham khảo
- [Duolingo Max](https://blog.duolingo.com/duolingo-max/) - Gia sư học ngôn ngữ tham khảo
- [Google LearnLM / Gemini for Education](https://blog.google/technology/google-deepmind/learnlm) — model tham chiếu được lưu trữ
- [Quizlet Q-Chat](https://quizlet.com) — tham chiếu thay thế
- [Synthesis Tutor](https://www.synthesis.com) — tham khảo khởi động
- [FSRS algorithm](https://github.com/open-spaced-repetition/fsrs4anki) — bộ lập lịch lặp lại cách nhau
- [Bayesian Knowledge Tracing](https://en.wikipedia.org/wiki/Bayesian_knowledge_tracing) - cổ điển dành cho người học model
- [LiveKit Agents](https://github.com/livekit/agents) — stack giọng nói
