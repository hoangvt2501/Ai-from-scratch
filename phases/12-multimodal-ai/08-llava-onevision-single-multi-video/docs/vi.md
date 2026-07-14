# LLaVA-OneVision: Một hình ảnh, nhiều hình ảnh, video trong một Model

> Trước LLaVA-OneVision (Li và cộng sự, tháng 8 năm 2024), thế giới VLM mở có các dòng riêng biệt: LLaVA-1.5 cho hình ảnh đơn lẻ, models đa hình ảnh như Mantis và VILA, models video như Video-LLaVA và Video-LLaMA. Mỗi người đều thắng benchmark của mình và thất bại ở những người khác. LLaVA-OneVision lập luận rằng một chương trình giảng dạy duy nhất có thể huấn luyện một model thống trị cả ba kịch bản và các hiệu ứng chuyển giao nhiệm vụ nổi lên (skills một hình ảnh được xuất sang video, suy luận nhiều hình ảnh được xuất sang một hình ảnh) đánh bại tổng số các chuyên gia. Công thức này rất đơn giản: ngân sách token hình ảnh không đổi trong các tình huống, cộng với chương trình giảng dạy rõ ràng chuyển từ hình ảnh đơn sang OneVision (nhiều hình ảnh) đến video. Bài học này đọc ngân sách, chương trình giảng dạy và các hành vi mới nổi.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib, token budget solver + curriculum planner)
**Kiến thức tiên quyết:** Giai đoạn 12 · 05 (LLaVA), Giai đoạn 12 · 06 (bất kỳ độ phân giải nào)
**Thời lượng:** ~180 phút

## Mục tiêu học tập

- Thiết kế ngân sách token hình ảnh không đổi trên các đầu vào một hình ảnh, nhiều hình ảnh và video.
- Đặt hàng một chương trình giảng dạy training chuyển skills từ một hình ảnh sang video mà không bị lãng quên thảm khốc.
- Giải thích lý do tại sao một model đánh bại các chuyên gia cùng một parameter được tính khi chương trình giảng dạy được thực hiện đúng.
- Kể tên ba khả năng nổi lên được báo cáo bởi LLaVA-OneVision: suy luận nhiều camera, prompting đặt dấu hiệu, agent ảnh chụp màn hình iPhone.

## Vấn đề

Hình ảnh, nhiều hình ảnh và video đều nhấn mạnh một model khác nhau.

Hình ảnh đơn cần tokens độ phân giải cao (AnyRes, ~2880 visual tokens) để nắm bắt chi tiết OCR và chi tiết. Ngân sách cho mỗi mẫu: một hình ảnh, 2880 tokens.

Nhiều hình ảnh muốn một số hình ảnh ở độ phân giải vừa phải (~576 tokens mỗi ảnh) để suy luận giữa các hình ảnh phù hợp với ngữ cảnh. Ngân sách cho mỗi mẫu: 4-8 hình ảnh, 576 hình ảnh, 2300-4600 tokens.

Video muốn nhiều khung hình ở độ phân giải thấp (~196 tokens mỗi khung hình sau khi gộp) để ghi lại động lực thời gian. Ngân sách cho mỗi mẫu: 8-32 khung hình, 196 khung hình, 1600-6200 tokens.

Nếu bạn huấn luyện models riêng biệt, bạn chọn một ngân sách. Nếu bạn huấn luyện một model, bạn cần ngân sách để mở rộng quy mô hợp lý trên các tình huống mà không làm tăng ngữ cảnh.

Trước OneVision, câu trả lời mặc định là "huấn luyện một kịch bản, bỏ qua các kịch bản khác". Video-LLaVA đã trang bị thêm video vào một model hình ảnh với các giai đoạn training bổ sung. LLaVA-NeXT đã thêm hỗ trợ nhiều hình ảnh với lát gạch. Không ai xử lý cả ba một cách sạch sẽ.

## Khái niệm

### Ngân sách OneVision token

LLaVA-OneVision chọn ngân sách token hình ảnh thống nhất khoảng 3000-4000 tokens cho mỗi mẫu, được phân bổ khác nhau cho mỗi tình huống:

- Hình ảnh đơn: AnyRes-9 (ô 3x3 + hình thu nhỏ), mỗi ô ở mức 384 với 729 bản vá, gộp hai tuyến tính tích cực 2x2 → 182 mỗi ô. Tổng: 9 * 182 + 182 = 1820 tokens. Hoặc AnyRes-4 ở mức 729 mỗi ô = 2916 + 729.
- Đa hình ảnh: mỗi hình ảnh ở độ phân giải trung bình (384, không lát gạch), 729 tokens không gộp. Ngân sách 6 hình ảnh → 4374 tokens.
- Video: 32 khung hình ở độ phân giải 384 với nhóm bili tuyến 3x3 mạnh mẽ → 81 tokens mỗi khung hình. Tổng: 32 * 81 = 2592 tokens.

Việc phân bổ duy trì tổng tokens gần như không đổi. The LLM không bao giờ nhìn thấy một batch thổi bay bối cảnh của nó. Các encoder tạo ra các hình học khác nhau cho mỗi kịch bản, nhưng LLM tiêu thụ cùng một ngân sách.

### Chương trình giảng dạy ba giai đoạn

LLaVA-OneVision huấn luyện theo ba giai đoạn:

1. SFT một hình ảnh (giai đoạn SI). Tất cả dữ liệu là một hình ảnh cộng với văn bản. Huấn luyện về đầu vào AnyRes có độ phân giải cao. Điều này dạy nhận thức, OCR và hiểu biết chi tiết. Sử dụng dữ liệu LLaVA-NeXT cộng với dữ liệu hình ảnh đơn dành riêng cho OneVision.
2. OneVision SFT (giai đoạn OV). Kết hợp hình ảnh đơn + nhiều hình ảnh + video (khung hình được lấy mẫu đồng nhất). Huấn luyện về ngân sách token thống nhất. Điều này dạy model xử lý các hình dạng batch không đồng nhất. Không đặt lại trọng lượng - tiếp tục từ giai đoạn SI.
3. Chuyển giao nhiệm vụ (giai đoạn TT). Tiếp tục với hỗn hợp tác vụ mục tiêu, thường nặng hơn trên nhiều hình ảnh hoặc video tùy thuộc vào sản phẩm. fine-tune tùy chọn để triển khai.

Quan trọng: thứ tự chương trình giảng dạy rất quan trọng. Training ưu tiên video hoặc ưu tiên nhiều hình ảnh tạo ra hiệu suất hình ảnh kém hơn so với ưu tiên hình ảnh đơn, ngay cả với cùng một dữ liệu. Bài báo loại bỏ điều này một cách rõ ràng.

### Tại sao chương trình giảng dạy hoạt động

training hình ảnh đơn xây dựng cơ sở nhận thức. Bản vá tokens mang features hình ảnh chi tiết; LLM học cách tích hợp chúng với văn bản. Nhiều hình ảnh và video giới thiệu những thách thức về cấu trúc (hình ảnh nào là hình ảnh nào, điều gì xảy ra trước) khó học nếu không có cơ sở nhận thức vững chắc.

Nếu bạn huấn luyện tất cả các kịch bản từ đầu cùng nhau, model không phù hợp với nhận thức (dữ liệu hình ảnh đơn hạn chế trên mỗi batch) và cấu trúc quá phù hợp (nhiều dữ liệu nhiều hình ảnh / video). Kết quả: một model tuân theo các mẫu suy luận hình ảnh chéo nhưng nông về mặt hình ảnh.

Thứ tự chương trình giảng dạy cung cấp cho bạn sức mạnh nhận thức từ giai đoạn SI, sau đó compositional/temporal lý luận từ giai đoạn OV mà không làm mất cả hai.

### skills kịch bản chéo mới nổi

Báo cáo LLaVA-OneVision báo cáo ba khả năng mới nổi:

1. Lý luận nhiều camera. Được huấn luyện về nhiều hình ảnh + video riêng biệt; tại inference, được hỏi lý do về cảnh lái xe nhiều camera. model tích hợp chính xác các chế độ xem mặc dù chưa bao giờ thấy định dạng chính xác đó trong training.
2. Bộ đánh dấu prompting. Người dùng chú thích các đối tượng trong hình ảnh bằng các dấu được đánh số; model lý do về "Mác 3 đang làm gì so với Mác 7". Không được huấn luyện về nhãn hiệu cũng như chú thích; Học hỏi từ sự kết hợp giữa grounding không gian + tham chiếu nhiều hình ảnh.
3. agent chụp màn hình iPhone. Người dùng cung cấp ảnh chụp màn hình iPhone và yêu cầu lập kế hoạch nhấp chuột tiếp theo. Được huấn luyện về ảnh chụp màn hình giao diện người dùng, video về quy trình làm việc của người dùng và các cặp before/after nhiều hình ảnh. Khái quát hóa cho trường hợp sử dụng agent.

Đây không phải là những nhiệm vụ được huấn luyện; chúng xuất hiện từ cấu trúc thành phần của chương trình giảng dạy.

### Gộp token trực quan

Ngân sách token yêu cầu gộp lại. OneVision sử dụng nội suy hai tuyến tính trên lưới bản vá 2D: 24x24 = 576 bản vá trở thành 12x12 = 144 (hệ số 2x) hoặc 8x8 = 64 (hệ số 3x). Việc gộp lại được thực hiện trong không gian lưới vá, không token không gian, để bảo tồn địa phương.

Bản thân việc lựa chọn hệ số gộp cho mỗi kịch bản là một hyperparameter. Gộp ít hơn = nhiều hơn tokens = đại diện phong phú hơn. Gộp nhiều hơn = ít tokens hơn = nhiều khung hình / hình ảnh phù hợp hơn.

### LLaVA-OneVision-1.5

Phiên bản theo dõi năm 2025 (LLaVA-OneVision-1.5, arXiv 2509.23661) "hoàn toàn mở" trong dữ liệu training, trọng số model và mã. Phù hợp với khoảng cách độc quyền trên một số benchmarks và dân chủ hóa công thức. Cùng một chương trình giảng dạy, nhiều dữ liệu hơn, LLM cơ sở tốt hơn. Không thay đổi kiến trúc.

### Tương phản với Qwen2.5-VL

Qwen2.5-VL (Bài 12.09) đưa ra các lựa chọn khác nhau. Nó sử dụng M-RoPE và FPS động thay vì gộp cố định. Ngân sách của nó mở rộng theo đầu vào - video 1 phút sử dụng nhiều tokens hơn video 5 giây. LLaVA-OneVision cố định ngân sách và mở rộng quy mô gộp. Cả hai đều hoạt động; họ đánh đổi khả năng cấu hình để có thể dự đoán.

## Ứng dụng

`code/main.py` là một công cụ lập kế hoạch chương trình giảng dạy và ngân sách cho một VLM kiểu OneVision. Với ngân sách token cho mỗi mẫu và hỗn hợp kịch bản mục tiêu (giả sử 40% hình ảnh đơn, 30% nhiều hình ảnh, 30% video), nó:

- Phân bổ độ phân giải, hệ số gộp và khung hình cho mỗi kịch bản.
- Kiểm tra xem mọi kịch bản có phù hợp với ngân sách dùng chung hay không.
- Các báo cáo dự kiến token đếm, LLM FLOPs và kịch bản nào được mã hóa dưới mức.
- In lịch trình training từng giai đoạn.

Sử dụng nó để lập kế hoạch fine-tune OneVision hoặc để kiểm tra chi phí cho mỗi yêu cầu của triển khai VLM.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-onevision-budget-planner.md`. Với phân phối nhiệm vụ mục tiêu và ngân sách cho mỗi mẫu, nó phát ra hệ số AnyRes, tổng hợp trên mỗi khung hình, số khung hình video và trọng số giai đoạn chương trình giảng dạy. Sử dụng điều này bất cứ khi nào bạn huấn luyện hoặc fine-tune một VLM kịch bản thống nhất.

## Bài tập

1. Sản phẩm của bạn hỗ trợ 80% hình ảnh đơn, 10% đa hình ảnh (2-4 hình ảnh), 10% video (8-16 khung hình). Thiết kế ngân sách token. Bạn sẽ đặt ngân sách bổ sung mà bạn tiết kiệm được từ việc không thực hiện nhiều hình ảnh ở đâu?

2. Đọc LLaVA-OneVision Phần 4.3 (khả năng mới nổi). Đề xuất mới nổi thứ tư skill chương trình giảng dạy có thể sẽ mở khóa nhưng tờ báo không đưa tin.

3. Hoán đổi thứ tự chương trình giảng dạy - huấn luyện nhiều hình ảnh trước, sau đó là hình ảnh đơn, sau đó là video. Dự đoán benchmarks nào xuống cấp và tại sao.

4. Bài báo báo cáo video benchmarks huấn luyện chỉ trên 8 khung hình cho mỗi mẫu. Điều đó có khái quát hóa cho các video dài 30 giây ở inference không? Điều gì phá vỡ đầu tiên - ngân sách token hay lý luận thời gian?

5. Gộp hai tuyến tính của các bản vá 24x24 thành 12x12 là giảm 4 lần cho mỗi độ mờ. Triển khai gộp trong stdlib Python và xác minh rằng giá trị trung bình trên mỗi khối 2x2 khớp với đầu ra hai tuyến.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| Kịch bản OneVision | "Một hình ảnh, nhiều hình ảnh hoặc video" | Một trong ba hình dạng đầu vào mà VLM thống nhất xử lý; ngân sách không đổi trên |
| Token ngân sách | "Có bao nhiêu tokens cho mỗi mẫu" | Tổng tokens hình ảnh mà LLM nhìn thấy trên mỗi training / inference mẫu, thường là 3000-4000 |
| Chương trình giảng dạy | "Training lệnh" | Thứ tự giai đoạn (một hình ảnh → nhiều hình ảnh → video) được chọn để chuyển khẩn cấp |
| Gộp hai tuyến | "Token thu nhỏ" | Áp dụng nội suy hai tuyến cho lưới vá (2D) để giảm số lượng token trong khi vẫn giữ được vị trí |
| skill mới nổi | "Không được huấn luyện, vẫn làm việc" | Khả năng xuất hiện ở inference mà không khớp với dữ liệu training, do thành phần chương trình giảng dạy |
| AnyRes-k | "Thiết lập k-tile" | k ô phụ có độ phân giải cố định cộng với một hình thu nhỏ, k điển hình ∈ {4, 9} |
| Chuyển giao nhiệm vụ | "Tổng quát hóa kịch bản chéo" | Skills học trên một hình ảnh áp dụng cho video (và ngược lại) thông qua đường trục được chia sẻ |

## Đọc thêm

- [Li et al. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326)
- [LLaVA-OneVision-1.5: Fully Open Framework (arXiv:2509.23661)](https://arxiv.org/abs/2509.23661)
- [Lin et al. — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Lin et al. — VILA (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
