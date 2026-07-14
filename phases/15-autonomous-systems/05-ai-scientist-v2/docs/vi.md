# AI Scientist v2 — Nghiên cứu tự trị cấp hội thảo

> AI Scientist v2 của Sakana (Yamada et al., arXiv:2504.08066) chạy vòng lặp nghiên cứu đầy đủ: giả thuyết, mã, thí nghiệm, số liệu, viết, đệ trình. Đây là hệ thống đầu tiên có đánh giá ngang hàng trên giấy tại hội thảo ICLR 2025. Đánh giá độc lập (Beel và cộng sự) cho thấy 42% thí nghiệm thất bại do lỗi mã hóa và đánh giá tài liệu thường xuyên dán nhãn sai các khái niệm đã được thiết lập là mới. Tài liệu riêng của Sakana cảnh báo rằng cơ sở mã thực thi mã được viết LLM và khuyến nghị cách ly Docker. Cả hai nửa của bức tranh đó đều là điểm mấu chốt.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, research-loop state-machine toy)
**Kiến thức tiên quyết:** Giai đoạn 15 · 03 (AlphaEvolve), Giai đoạn 15 · 04 (Tổng giám đốc)
**Thời lượng:** ~60 phút

## Vấn đề

Nghiên cứu là một nhiệm vụ mở. Không giống như tìm kiếm thuật toán của AlphaEvolve hoặc tự sửa đổi giới hạn benchmark của DGM, kết quả nghiên cứu không có tiêu chí chính xác có thể kiểm tra bằng máy. Một bài báo được đánh giá bởi người phản biện, không phải kiểm tra đơn vị. Điều đó làm cho vòng lặp khó đóng hơn - và có giá trị hơn nếu đóng lại, bởi vì nghiên cứu là nơi tồn tại sự tiến bộ kép.

AI Scientist v1 (Sakana, 2024) đã khép lại vòng lặp bằng cách bắt đầu từ các mẫu do con người tạo ra. Các LLM lấp đầy các thí nghiệm trong một giàn giáo cố định. AI Scientist v2 (Yamada et al., 2025) loại bỏ yêu cầu bản mẫu bằng cách sử dụng tìm kiếm cây agentic với vòng lặp phê bình model ngôn ngữ tầm nhìn. Hệ thống tạo ra ý tưởng, thực hiện thử nghiệm, tạo số liệu, viết bài báo và lặp lại phản hồi của người phản biện.

Phán quyết bình duyệt: một bài báo được tạo ra bởi v2 đã được chấp nhận tại hội thảo ICLR 2025 (có tiết lộ). Phán quyết đánh giá độc lập: hệ thống không đáng tin cậy. Cả hai đều đúng.

## Khái niệm

### Kiến trúc

1. **Tạo ý tưởng.** LLM đề xuất các ý tưởng nghiên cứu có điều kiện trên một chủ đề và prior tài liệu. v1 đã sử dụng các mẫu; V2 sử dụng agentic tìm kiếm trên một không gian của các giả thuyết.
2. **Kiểm tra tính mới.** Bước truy xuất tài liệu kiểm tra xem ý tưởng đã được xuất bản hay chưa. Đây là bước mà đánh giá của Beel và cộng sự phát hiện ra việc dán nhãn sai - các phương pháp đã được thiết lập thường được phân loại là mới.
3. **Kế hoạch thử nghiệm.** agent soạn thảo một giao thức thử nghiệm và viết mã.
4. **Thực thi.** Mã chạy trong một sandbox. Các lỗi được đưa trở lại vòng lặp thử lại. Trong các phép đo của Beel và cộng sự, 42% thí nghiệm thất bại do lỗi mã hóa ở giai đoạn này.
5. **Tạo hình.** Một ngôn ngữ thị giác model đọc các số liệu được tạo ra và viết lại chúng cho rõ ràng. Đây là bổ sung kỹ thuật quan trọng của v2.
6. **Viết.** Người LLM soạn thảo một bài báo, lặp lại với một người phản biện nội bộ.
7. **Tùy chọn: nộp.** Bài báo được gửi đến một địa điểm.

### Ý nghĩa của kết quả nghiệm thu hội thảo

Một bài báo do v2 tạo ra đã vượt qua đánh giá ngang hàng tại hội thảo ICLR 2025. Các tác giả đã tiết lộ nguồn gốc của bài báo cho ủy ban chương trình. Sự chấp nhận là một điểm dữ liệu; nó không phải là giấy phép để tuyên bố hệ thống "thực hiện nghiên cứu".

Bối cảnh quan trọng: các bài báo hội thảo có tiêu chuẩn thấp hơn các bài báo hội nghị chính. Đánh giá ngang hàng ồn ào; Một phần nhỏ các bài dự thi được chấp nhận vào bất kỳ ngày nào. Một thành công là bằng chứng về khái niệm, không phải là tuyên bố về độ tin cậy. Bài báo của Nature 2026 ghi lại vòng lặp từ đầu đến cuối và bản thân nó là đồng tác giả của các nhà nghiên cứu con người; nó không phải là "hệ thống đã viết một bài báo trên Nature".

### Những gì đánh giá độc lập tìm thấy

Beel et al. (arXiv: 2502.14297) đã thực hiện một đánh giá bên ngoài. Phát hiện tiêu đề:

- **Thử nghiệm thất bại.** 42% thử nghiệm thất bại do lỗi mã hóa (imports xấu, hình dạng không khớp, biến không xác định). Vòng lặp thử lại đã bắt được một số, không phải tất cả.
- **Dán nhãn sai mới lạ.** Bước truy xuất văn học thường gắn cờ các khái niệm đã được thiết lập là mới lạ. Đây là nghiên cứu tương đương với ảo giác.
- **Khoảng cách chất lượng trình bày.** Phê bình nhân vật ngôn ngữ thị giác đã tạo ra hình ảnh cấp xuất bản, che giấu những điểm yếu thực nghiệm tiềm ẩn.

Phát hiện cuối cùng là phát hiện quan trọng cho giai đoạn này. Một hệ thống tạo ra kết quả thuyết phục mà không thực hiện nghiên cứu thuyết phục sẽ nguy hiểm hơn, không an toàn hơn so với một hệ thống thất bại rõ ràng. Đánh giá phải đạt được các tuyên bố cơ bản, không dừng lại ở con số.

### Mối quan tâm thoát sandbox

README repository của Sakana cảnh báo:

> Do bản chất của phần mềm này thực thi mã do LLM tạo nên chúng tôi không thể đảm bảo an toàn. Có rủi ro về các gói nguy hiểm, truy cập web không kiểm soát và sinh ra các processes ngoài ý muốn. Tự chịu rủi ro khi sử dụng và cân nhắc cách ly Docker.

Đây là hình thức hoạt động của quyền tự chủ trong một miền chưa được xác minh. LLM viết mã; mã chạy; Mã có thể làm bất cứ điều gì mà process được phép làm. Nếu không có sandbox giới hạn cứng các hành động của hệ thống tệp, mạng và process, bất kỳ agent nghiên cứu tự định hướng nào cũng có thể trích xuất dữ liệu, ghi điện toán hoặc tự viết lại.

Câu chuyện sandbox của AlphaEvolve dễ dàng hơn vì người đánh giá của nó chặt chẽ. Vòng lặp của AI Scientist v2 chạy mã mở với các mục tiêu mở. Đó là lý do tại sao nó cần cách ly mạnh hơn (tối thiểu Docker; ưu tiên seccomp / gVisor) và xem xét thủ công mọi đệ trình trước khi nó rời khỏi hệ thống.

### Vị trí của v2 ở biên giới stack

| Hệ thống | Mục tiêu | Loại đầu ra | Người đánh giá | Lỗi đã biết |
|---|---|---|---|---|
| AlphaEvolve | Thuật toán | Mã | đơn vị + benchmark | bị giới hạn bởi sự nghiêm ngặt của người đánh giá |
| DGM | agent giàn giáo | Mã | SWE-băng ghế | Hack phần thưởng |
| AI Nhà khoa học v2 | Bài nghiên cứu | văn bản + mã + số liệu | Đánh giá ngang hàng (yếu) | Thí nghiệm thất bại, dán nhãn sai, đánh bóng điểm yếu che giấu |

V2 có công cụ đánh giá tự động yếu nhất trong ba, bề mặt đầu ra rộng nhất và đường dẫn đến artifacts công cộng ngắn nhất. Các kiểm soát hoạt động (sandbox, xem xét, tiết lộ) đang thực hiện hầu hết các công việc an toàn.

## Ứng dụng

`code/main.py` mô phỏng vòng lặp v2 như một cỗ máy trạng thái: kiểm tra ý tưởng → tính mới → thử nghiệm → hình → viết → xem xét → chấp nhận hoặc lặp lại. Mỗi trạng thái có xác suất thất bại có thể định cấu hình được lấy từ Beel et al. phát hiện. Chạy trình mô phỏng cho N vòng lặp và đếm:

- Có bao nhiêu ý tưởng đạt được sự phục tùng.
- Có bao nhiêu bài dự thi sẽ có một lỗ hổng thử nghiệm nghiêm trọng mà tờ giấy đánh bóng che giấu.
- Ngân sách thử lại đánh đổi chất lượng so với năng suất như thế nào.

## Sản phẩm bàn giao

`outputs/skill-ai-scientist-sandbox-review.md` là một danh sách kiểm tra đánh giá hai cổng cho bất kỳ thứ gì được tạo ra bởi một agent vòng lặp nghiên cứu trước khi nó rời khỏi sandbox.

## Bài tập

1. Chạy `code/main.py` với parameters mặc định. Phần nào của vòng lặp chạy tạo ra một tờ giấy "sạch"? Phần nào tạo ra một bài báo có lỗ hổng thất bại trong thí nghiệm mà nhân vật phê bình đã đánh bóng?

2. Các mặc định đã sử dụng 42% / 25% của Beel et al. Chạy lại với `--experiment-failure 0.20 --novelty-mislabel 0.10` và sau đó với `--experiment-failure 0.60 --novelty-mislabel 0.40`. Làm thế nào để chia sẻ bóng bẩy nhưng thiếu sót thay đổi giữa hai lần chạy?

3. Đọc AI Scientist v2 của Sakana repo README về các yêu cầu sandbox. Kể tên hai hạn chế bổ sung (ngoài Docker) mà bạn sẽ áp dụng cho một cuộc chạy tự động kéo dài nhiều ngày.

4. Đọc Beel và cộng sự. Phần 4 về khoảng cách chất lượng trình bày. Thiết kế thêm một công cụ đánh giá có thể bắt các bài báo trông bóng bẩy nhưng có sai sót về mặt thực nghiệm.

5. Đề xuất một giao thức đánh giá của con người cho các kết quả nghiên cứu agent có quy mô tốt hơn so với "một tiến sĩ đọc mọi bài báo". Xác định nút thắt cổ chai và thiết kế xung quanh nó.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|---|---|---|
| AI Nhà khoa học v1 | "Nghiên cứu theo khuôn mẫu của Sakana agent" | Điền các thí nghiệm vào một giàn giáo cố định |
| AI Nhà khoa học v2 | "Nghiên cứu không có mẫu agent" | Agentic tìm kiếm cây với VLM phê bình hình |
| Agentic tìm kiếm cây | "Nghiên cứu phân nhánh agent" | Mở rộng song song nhiều kế hoạch thử nghiệm; mận khô của nhà phê bình nội bộ |
| Phê bình ngôn ngữ tầm nhìn | "VLM đánh bóng các con số" | model đa phương thức đọc số liệu và viết lại chúng để rõ ràng |
| Truy xuất tài liệu | "Kiểm tra tính mới" | Tìm kiếm prior làm việc để xác nhận tính mới của ý tưởng - được ghi lại để dán nhãn sai |
| Mặt nạ đánh bóng | "Giấy đẹp, nghiên cứu hỏng" | Chất lượng trình bày vượt quá chất lượng thí nghiệm; che giấu điểm yếu |
| Sandbox trốn thoát | "Mã LLM gặp lỗi" | Mã được thực thi Agent thực hiện những điều mà nhà thiết kế vòng lặp không có ý định |

## Đọc thêm

- [Yamada et al. (2025). The AI Scientist-v2](https://arxiv.org/abs/2504.08066) - giấy.
- [Sakana blog on the Nature 2026 publication](https://sakana.ai/ai-scientist-nature/) — tóm tắt nhà cung cấp với ngữ cảnh đánh giá ngang hàng.
- [Beel et al. (2025). Independent evaluation of The AI Scientist](https://arxiv.org/abs/2502.14297) - số đánh giá bên ngoài.
- [Sakana AI Scientist v1 paper](https://arxiv.org/abs/2408.06292) — người tiền nhiệm được tạo mẫu.
- [Anthropic — Measuring AI agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) - khuôn khổ rộng hơn của agents nghiên cứu mở.
