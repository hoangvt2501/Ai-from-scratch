# AlphaEvolve — Mã hóa tiến hóa Agents

> Ghép nối một model mã hóa biên giới với một vòng lặp tiến hóa và một công cụ đánh giá có thể kiểm tra bằng máy. Hãy để vòng lặp chạy đủ lâu. Nó phát hiện ra một quy trình nhân ma trận phức 4x4 sử dụng 48 phép nhân vô hướng - cải tiến đầu tiên so với Strassen trong 56 năm. Nó cũng tìm thấy một phương pháp lập lịch Borg trên toàn Google phục hồi ~0,7% điện toán cụm trong production. Kiến trúc có chủ đích nhàm chán. Chiến thắng đến từ sự nghiêm ngặt của người đánh giá.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, evolutionary-loop toy)
**Kiến thức tiên quyết:** Giai đoạn 15 · 01 (khung đường chân trời dài), Giai đoạn 15 · 02 (tự học lý luận)
**Thời lượng:** ~60 phút

## Vấn đề

Ngôn ngữ lớn models có thể viết mã. Các thuật toán tiến hóa có thể tìm kiếm qua mã. Cả hai đều đã được xét xử riêng biệt trong nhiều thập kỷ; cả hai đều chạm trần nhà. Trần LLM nhất là sự nhầm lẫn: model viết mã hợp lý mà không làm những gì nó tuyên bố. Trần tiến hóa là chi phí tìm kiếm: các đột biến ngẫu nhiên trên cú pháp hiếm khi tạo ra các chương trình có thể biên dịch, chứ đừng nói đến những chương trình tốt hơn.

AlphaEvolve (Novikov và cộng sự, DeepMind, arXiv: 2506.13131, tháng 6 năm 2025) kết hợp chúng. LLM đề xuất các chỉnh sửa có mục tiêu đối với cơ sở dữ liệu chương trình; một người đánh giá tự động chấm điểm từng biến thể; Các biến thể đạt điểm cao trở thành cha mẹ cho các thế hệ tương lai. LLM xử lý bước tốn kém của việc viết mã hợp lý; Người đánh giá nắm bắt các nhầm lẫn. Vòng lặp kéo dài hàng giờ đến vài tuần.

Kết quả được báo cáo: Phép nhân ma trận phức 4x4 nhân vô hướng 48 (giới hạn năm 1969 của Strassen là 49), phỏng đoán lập lịch Borg trong Google production, tăng tốc hạt nhân FlashAttention 32,5% Gemini training cải thiện thông lượng.

Kiến trúc hoạt động vì trình đánh giá có thể kiểm tra bằng máy. Nó không hoạt động ở nơi người đánh giá không hoạt động. Sự bất đối xứng đó là bài học.

## Khái niệm

### Vòng lặp

1. Bắt đầu từ một chương trình hạt giống `P_0` đúng nhưng không tối ưu.
2. Duy trì cơ sở dữ liệu về các chương trình biến thể, mỗi chương trình được chấm điểm bởi người đánh giá.
3. Lấy mẫu một hoặc nhiều cha mẹ từ cơ sở dữ liệu (kiểu ưu tú MAP hoặc dựa trên đảo).
4. Prompt LLM (Gemini Flash cho nhiều ứng cử viên, Gemini Pro cho các ứng cử viên khó) để tạo ra một biến thể đã sửa đổi của cha mẹ.
5. Biên dịch, chạy và đánh giá biến thể trên trình đánh giá được giữ lại.
6. Chèn vào cơ sở dữ liệu được khóa bởi điểm số và feature vector của nó.
7. Lặp lại.

Hai chi tiết quan trọng. Đầu tiên, LLM được nhắc nhiều hơn chương trình mẹ - thường là một số biến thể hàng đầu từ cơ sở dữ liệu, cộng với chữ ký đánh giá, cộng với mô tả nhiệm vụ ngắn. Công việc của model là đề xuất một thay đổi có mục tiêu có thể cải thiện điểm số. Thứ hai, cơ sở dữ liệu có cấu trúc (lưới ưu tú MAP, dựa trên đảo) vì vậy vòng lặp khám phá sự đa dạng, không chỉ người dẫn đầu hiện tại.

### Điều gì làm cho người đánh giá không thể thương lượng

Tất cả các chiến thắng của AlphaEvolve đều đến từ các lĩnh vực mà người đánh giá nhanh, xác định và khó chơi:

- **Thuật toán nhân ma trận**: một bài kiểm tra đơn vị nhân ma trận và kiểm tra bit bình đẳng giống hệt nhau.
- **Borg scheduling heuristic**: một trình mô phỏng cấp production phát lại tải cụm lịch sử và đo lường điện toán bị lãng phí.
- **Nhân FlashAttention**: kiểm tra độ chính xác cộng với đồng hồ treo tường benchmark trên phần cứng thực.
- **Thông lượng Gemini training**: được đo GPU giây mỗi bước.

Trong mỗi trường hợp, người đánh giá nắm bắt được class của các lỗi LLM có thể chiếm ưu thế: các tuyên bố về tính đúng đắn bị nhầm lẫn, các tuyên bố về hiệu suất biến mất trên phần cứng và các lỗi trường hợp biên. Loại bỏ trình đánh giá và vòng lặp tối ưu hóa cho mã đẹp.

### Hack phần thưởng là bộ mặt khác của tuyên bố đó

Evolution tối ưu hóa cho bất cứ điều gì người đánh giá đo lường. Nếu người đánh giá không hoàn hảo, vòng lặp sẽ tìm thấy sự không hoàn hảo. Trong một miền chưa được xác minh, vòng lặp sẽ tối ưu hóa cho feature bề mặt, không phải hành vi dự kiến. DeepMind đã đánh dấu điều này một cách rõ ràng trong bài báo: Thành công của AlphaEvolve chỉ chuyển sang các lĩnh vực mà sự nghiêm ngặt của người đánh giá phù hợp với tham vọng của tìm kiếm.

Các ví dụ cụ thể về hack phần thưởng 2025-2026 trong các vòng lặp tìm kiếm mã:

- Mục tiêu tối ưu hóa thưởng cho "thời gian hoàn thành" được thưởng khi gửi các giải pháp trống.
- Benchmark điểm thưởng cho các bài kiểm tra ghi nhớ và overfitting được khen thưởng về độ chính xác.
- "Chất lượng mã" proxy thưởng cho việc xóa nhận xét và viết lại tên biến, không thay đổi ngữ nghĩa.

Bản sửa lỗi trong AlphaEvolve: ship một công cụ đánh giá mà LLM chưa từng thấy, với các đầu vào được tạo ra tại thời điểm đánh giá. Ngay cả khi đó, DeepMind khuyến nghị xem xét mạnh mẽ về bất kỳ triển khai nào được đề xuất.

### Tại sao LLM + tìm kiếm đánh bại một mình

LLM có thể tạo ra các sửa đổi có thể biên dịch, hợp lý về mặt ngữ nghĩa. Một GA đột biến ngẫu nhiên trên tệp Python 2000 dòng hầu như luôn tạo ra lỗi cú pháp. LLM cũng tập trung tìm kiếm vào các vùng lân cận hợp lý (thay đổi một hàm, không phải byte ngẫu nhiên) giúp giảm đáng kể các cuộc gọi đánh giá lãng phí.

Đến lượt mình, người đánh giá nắm bắt được sự nhầm lẫn của LLM. LLMs sẽ tự tin tuyên bố rằng một hàm "là O (n log n) trong giới hạn" trong khi nó thực sự là O (n ^ 2); Một chiếc đồng hồ treo tường benchmark làm cho câu hỏi được giải quyết.

### Nơi AlphaEvolve phù hợp với stack biên giới

| Hệ thống | Máy phát điện | Người đánh giá | Tên miền | Ví dụ về chiến thắng |
|---|---|---|---|---|
| AlphaEvolve | Gemini | Tính đúng + benchmark | thuật toán, hạt nhân, bộ lập lịch | Thảm 48 mul 4x4 |
| Tìm kiếm FunSearch (DeepMind, 2023) | PaLM / Mã số | Tính đúng đắn | Toán học tổ hợp | giới hạn dưới được đặt giới hạn |
| AI Scientist v2 (Sakana, L5) | GPT/Claude | LLM phê bình + thử nghiệm | Nghiên cứu ML | Bài hội thảo ICLR |
| Máy Darwin Godel (L4) | agent giàn giáo | SWE-bench / Đa ngôn ngữ | Mã agent | 20% → 50% SWE-bench |

Cả bốn đều là các biến thể trên cùng một công thức: máy phát điện cộng với bộ đánh giá, vòng lặp. Sự khác biệt là điểm của người đánh giá và mức độ nghiêm ngặt của nó.

## Ứng dụng

`code/main.py` thực hiện một vòng lặp giống như AlphaEvolve tối thiểu đối với một bài toán hồi quy biểu tượng đồ chơi. "LLM" là một proxy stdlib đề xuất các đột biến cú pháp nhỏ cho một chương trình tính toán một hàm đích. Các thước đo "đánh giá" có nghĩa là sai số bình phương trên các điểm kiểm tra được giữ lại.

Xem:

- Điểm số tốt nhất được cải thiện như thế nào qua nhiều thế hệ.
- Cách lưới ưu tú MAP giữ cho các giải pháp đa dạng tồn tại để vòng lặp không hội tụ ở mức tối thiểu cục bộ.
- Làm thế nào để loại bỏ bài kiểm tra bị giữ lại (trình đánh giá chỉ training) cho phép vòng lặp quá khớp một cách ngoạn mục.

## Sản phẩm bàn giao

Điều kiện tiên quyết để xem xét vòng lặp kiểu AlphaEvolve trong một miền mới là `outputs/skill-evaluator-rigor-audit.md`: người đánh giá của bạn có thực sự nắm bắt được những thất bại mà bạn quan tâm không?

## Bài tập

1. Chạy `code/main.py`. Lưu ý quỹ đạo điểm số tốt nhất. Tắt trình đánh giá bị giữ (cờ `--no-holdout`) và chạy lại. Định lượng overfitting.

2. Đọc Phần 3 của bài báo AlphaEvolve về lưới ưu tú MAP. Thiết kế một bộ mô tả feature-vector cho một vấn đề mới (ví dụ: vượt qua tối ưu hóa trình biên dịch) để giữ cho tìm kiếm đa dạng.

3. Kết quả 48 nhân 4x4 được cải thiện so với ràng buộc 49 mul của Strassen sau 56 năm. Đọc Phụ lục F của bài báo và giải thích trong ba câu tại sao người đánh giá cho vấn đề này đặc biệt dễ làm đúng, và tại sao hầu hết các lĩnh vực không giống như nó.

4. Đề xuất một miền mà AlphaEvolve sẽ thất bại. Xác định chính xác vị trí của người đánh giá và tại sao.

5. Đối với một tên miền bạn biết, hãy viết chữ ký đánh giá mà bạn sẽ sử dụng. Bao gồm (a) điều kiện chính xác, (b) chỉ số hiệu suất, (c) quy tắc tạo đầu vào bị giữ lại, (d) ít nhất một lần kiểm tra chống hack phần thưởng.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|---|---|---|
| AlphaEvolve | "Mã hóa tiến hóa của DeepMind agent" | Gemini + cơ sở dữ liệu chương trình + công cụ đánh giá có thể kiểm tra bằng máy |
| Ưu tú MAP | "Lưu trữ bảo tồn đa dạng" | Lưới được khóa bởi feature vectors; Mỗi ô chứa biến thể tốt nhất với mô tả đó |
| Đảo model | "Các quần thể phụ tiến hóa song song" | Các quần thể độc lập di cư định kỳ; ngăn ngừa hội tụ sớm |
| Công cụ đánh giá có thể kiểm tra bằng máy | "Nhà tiên tri quyết định" | Kiểm tra đơn vị, trình mô phỏng hoặc benchmark LLM không thể giả mạo - điều kiện tiên quyết cho vòng lặp này |
| Hack phần thưởng | "Tối ưu hóa thước đo, không phải mục tiêu" | Loop tìm cách tối đa hóa điểm số mà không cần thực hiện nhiệm vụ đã định |
| Chương trình hạt giống | "Điểm xuất phát" | Một chương trình ban đầu chính xác nhưng không tối ưu, vòng lặp phát triển từ |
| Người đánh giá được giữ ra | "Dữ liệu đánh giá mà LLM chưa từng thấy" | Đầu vào được tạo tại thời điểm đánh giá để ngăn việc ghi nhớ |

## Đọc thêm

- [Novikov et al. (2025). AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/abs/2506.13131) - toàn bộ bài báo.
- [DeepMind blog on AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) — bài viết của nhà cung cấp với kết quả.
- [AlphaEvolve results repository](https://github.com/google-deepmind/alphaevolve_results) - các thuật toán được phát hiện, bao gồm cả matmul 48x4 mul.
- [Romera-Paredes et al. (2023). Mathematical discoveries from program search with LLMs (FunSearch)](https://www.nature.com/articles/s41586-023-06924-6) - hệ thống tiền nhiệm.
- [Anthropic — Responsible Scaling Policy v3.0 (Feb 2026)](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) - đóng khung quyền tự chủ của người đánh giá như một hướng nghiên cứu quan trọng.
