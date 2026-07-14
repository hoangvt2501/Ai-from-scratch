# Frameworks an toàn biên giới - RSP, PF, FSF

> Ba phòng thí nghiệm chính frameworks xác định quản trị ngành công nghiệp năm 2026 về năng lực biên giới. Anthropic Responsible Scaling Policy v3.0 (Tháng 2 năm 2026) giới thiệu Mức độ an toàn AI theo cấp độ (ASL-1 đến ASL-5+), được mô hình hóa theo mức an toàn sinh học, với ASL-3 được kích hoạt vào tháng 5 năm 2025 cho các models liên quan đến CBRN. OpenAI Chuẩn bị Framework phiên bản 2 (Tháng Tư 2025) xác định năm tiêu chí cho các khả năng được theo dõi và tách Báo cáo Khả năng khỏi Báo cáo Bảo vệ. DeepMind Frontier Safety Framework phiên bản 3.0 (Tháng 9 năm 2025) giới thiệu Mức khả năng quan trọng bao gồm CCL thao tác có hại mới. Cả ba hiện đều bao gồm các điều khoản điều chỉnh đối thủ cạnh tranh cho phép trì hoãn nếu các phòng thí nghiệm ngang hàng ship không có các biện pháp bảo vệ tương đương. alignment phòng thí nghiệm chéo vẫn là cấu trúc chứ không phải thuật ngữ: "Ngưỡng khả năng", "Ngưỡng năng lực cao" và "Mức độ khả năng quan trọng" biểu thị các cấu trúc tương tự.

**Loại:** Học
**Ngôn ngữ:** none
**Kiến thức tiên quyết:** Giai đoạn 18 · 17 (WMDP), Giai đoạn 18 · 07-09 (lừa dối thất bại)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Mô tả cấu trúc bậc ASL của Anthropic và điều gì đã kích hoạt ASL-3.
- Đặt tên cho năm tiêu chí Chuẩn bị OpenAI Framework phiên bản 2 cho các khả năng được theo dõi.
- Mô tả cấu trúc cấp độ khả năng quan trọng của DeepMind và CCL thao túng có hại.
- Giải thích các điều khoản điều chỉnh đối thủ cạnh tranh và lý do tại sao chúng lại quan trọng đối với động lực cuộc đua.
- Xác định một trường hợp an toàn và mô tả cấu trúc ba trụ cột (giám sát, không đọc được, không có khả năng).

## Vấn đề

Bài 7-17 xác định rằng lừa dối là có thể, khả năng sử dụng kép tồn tại và đánh giá có giới hạn. Một phòng thí nghiệm có model có khả năng biên giới cần một cấu trúc quản trị nội bộ:
- Xác định ngưỡng khi cần các biện pháp bảo vệ mới.
- Xác định các đánh giá bắt buộc trước khi mở rộng quy mô.
- Mô tả trường hợp an toàn trông như thế nào.
- Xử lý vấn đề động lực chủng tộc (nếu đối thủ cạnh tranh ship không có biện pháp bảo vệ, bạn sẽ làm gì?).

Ba frameworks 2025-2026 là hiện đại - không hoàn hảo, phát triển và đủ liên kết giữa các phòng thí nghiệm mà câu hỏi quản trị bây giờ là liệu các frameworks có đầy đủ hay không, chứ không phải liệu chúng có tồn tại hay không.

## Khái niệm

### Anthropic Responsible Scaling Policy v3.0 (Tháng Hai 2026)

Cấu trúc ASL:
- ASL-1: không phải là model biên giới (được gộp bởi đường cơ sở yếu hơn biên giới).
- ASL-2: đường cơ sở biên giới hiện tại; được triển khai với các biện pháp bảo vệ thông thường.
- ASL-3: nguy cơ lạm dụng thảm khốc cao hơn đáng kể; Các khả năng liên quan đến CBRN. Kích hoạt Tháng Năm 2025.
- ASL-4: AI ngưỡng vượt qua R&D-2; models có thể tự động hóa nghiên cứu AI cấp độ đầu vào.
- ASL-5+: models nghiên cứu và phát triển AI tiên tiến giúp tăng tốc đáng kể việc mở rộng quy mô hiệu quả.

Tính năng mới trong phiên bản 3.0:
- Lộ trình An toàn Biên giới (công khai ở dạng đã được biên tập).
- Báo cáo rủi ro (hàng quý, một số được xem xét bên ngoài).
- AI R&D được chia thành AI R&D-2 và AI R&D-4.
- Một khi AI R&D-4 được vượt qua, cần phải có một trường hợp an toàn khẳng định, xác định các rủi ro sai lệch từ việc models theo đuổi các mục tiêu sai lệch.

### OpenAI Chuẩn bị Framework v2 (15 Tháng Tư, 2025)

Năm tiêu chí cho các khả năng được theo dõi:
- **Hợp lý.** Mối đe dọa hợp lý model tồn tại.
- **Có thể đo lường được.** Có thể đánh giá thực nghiệm.
- **Nghiêm trọng.** Tác hại là lớn.
- **Net-new.** Không phải rủi ro đã có từ trước được mở rộng quy mô.
- **Ngay lập tức hoặc không thể khắc phục được.** Tác hại xảy ra nhanh chóng hoặc không thể hoàn tác.

Các khả năng đáp ứng cả năm khả năng được theo dõi. Những người khác thì không.

Cấu trúc PF v2 khác:
- Báo cáo Khả năng riêng biệt (những gì model có thể làm) khỏi Báo cáo Bảo vệ (những biện pháp kiểm soát tồn tại).
- Đánh giá của Nhóm Tư vấn An toàn.
- Lãnh đạo phê duyệt; Ủy ban An toàn và An ninh của Hội đồng quản trị giám sát.
- "Điều khoản điều chỉnh": OpenAI có thể giảm yêu cầu nếu một phòng thí nghiệm khác ships mà không có các biện pháp bảo vệ tương đương.

### DeepMind Frontier Safety Framework v3.0 (Tháng Chín 2025)

Mức độ khả năng quan trọng (CCL) theo miền:
- Nâng cao vũ khí sinh học
- Nâng cao không gian mạng
- ML Tăng tốc R&D
- Thao túng có hại (mới trong phiên bản 3.0): models có thể thay đổi đáng kể beliefs/behavior trong bối cảnh có nguy cơ cao.

phiên bản 2.0 (tháng 2 năm 2025) đã thêm phần Alignment lừa đảo và các cấp độ bảo mật cao hơn cho các CCL R&D ML.

### alignment phòng thí nghiệm chéo

- Anthropic "Ngưỡng năng lực".
- DeepMind "Mức độ khả năng quan trọng".
- OpenAI "Ngưỡng năng lực cao".

Không có thuật ngữ tiêu chuẩn ngành. Phù hợp về mặt cấu trúc: ba cấp năng lực biên giới với các tiêu chí đánh giá đã công bố. Cả ba đều bao gồm các điều khoản điều chỉnh đối thủ cạnh tranh (2025 trở đi).

### Trường hợp an toàn

Trường hợp an toàn là một lập luận bằng văn bản rằng việc triển khai là an toàn có thể chấp nhận được trong các giả định trong trường hợp xấu nhất. Cấu trúc tiêu chuẩn nhắm đến ba trụ cột:

- **Giám sát.** Chúng ta có thể phát hiện hành vi xấu nếu nó xảy ra không?
- **Không đọc được.** Người model có thiếu khả năng thực hiện một kế hoạch mạch lạc để gây hại không?
- **Không có khả năng.** model có thiếu khả năng gây hại được đề cập không?

Các trường hợp an toàn khác nhau nhắm vào các trụ cột khác nhau. Đối với trường hợp ASL-3 CBRN, không có khả năng (thông qua việc bỏ học) là mục tiêu chính. Đối với alignment lừa đảo, giám sát và không đọc được là mục tiêu. Đối với việc nâng cao không gian mạng, cả ba đều có liên quan.

### Vấn đề động lực chủng tộc

Các điều khoản điều chỉnh đối thủ cạnh tranh đang gây tranh cãi. Các nhà phê bình cho rằng họ tạo ra một cuộc chạy đua xuống đáy: nếu cả ba phòng thí nghiệm sẽ giảm yêu cầu khi một đối thủ cạnh tranh đào tẩu, sự cân bằng sẽ chuyển sang đào tẩu. Những người bảo vệ lập luận rằng giải pháp thay thế (các biện pháp bảo vệ đơn phương) tạo ra kết quả tồi tệ hơn nếu phòng thí nghiệm đào tẩu ít ý thức về an toàn hơn.

AISI của Vương quốc Anh, CAISI của Hoa Kỳ và Văn phòng AI EU (Bài 24) là các đối tác quản trị bên ngoài. Các frameworks phòng thí nghiệm là tự nguyện; Các frameworks pháp lý đang nổi lên.

### Điều này phù hợp với Giai đoạn 18 ở đâu

Bài 17-18 là lớp đo lường và quản trị bên trên các phân tích lừa dối và đội đỏ. Các bài học 19-24 bao gồm phúc lợi, bias, quyền riêng tư, hình mờ và cấu trúc quy định. Bài 28 lập bản đồ hệ sinh thái nghiên cứu (MATS, Redwood, Apollo, METR) vận hành các đánh giá.

## Ứng dụng

Không có mã cho bài học này. Đọc ba nguồn chính: RSP v3.0, PF v2, FSF v3.0. Ánh xạ cấu trúc cấp của mỗi phòng thí nghiệm với các phòng thí nghiệm khác và xác định một ngưỡng mà mỗi phòng thí nghiệm xác định mà các phòng thí nghiệm khác không xác định.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-framework-diff.md`. Đưa ra framework an toàn hoặc ghi chú phát hành, nó so sánh các định nghĩa ngưỡng của framework, đánh giá bắt buộc và cấu trúc trường hợp an toàn so với RSP v3.0, PF v2, FSF v3.0 và gắn cờ các khoảng trống giữa các phòng thí nghiệm.

## Bài tập

1. Đọc RSP v3.0, PF v2 và FSF v3.0. Biên soạn một bảng ngưỡng CBRN của mỗi phòng thí nghiệm, ngưỡng R&D AI của mỗi phòng thí nghiệm và đánh giá trước khi triển khai bắt buộc của từng phòng thí nghiệm.

2. Điều khoản điều chỉnh đối thủ cạnh tranh có trong cả ba frameworks (2025+). Viết một đoạn tranh luận cho nó; viết một đoạn tranh luận chống lại. Xác định giả định mà mỗi vị trí phụ thuộc vào.

3. Thiết kế hộp an toàn cho model vượt qua ngưỡng R&D-4 AI của Anthropic. Kể tên bằng chứng mà mỗi trụ cột trong số ba trụ cột (giám sát, không đọc được, không có khả năng) yêu cầu.

4. FSF v3.0 của DeepMind giới thiệu CCL thao tác có hại. Đề xuất ba phép đo thực nghiệm cho thấy một model đã vượt qua ngưỡng này.

5. Đọc "Các yếu tố chung của Policies an toàn AI biên giới" (2025) của METR. Kể tên ba sự hội tụ giữa các phòng thí nghiệm mạnh nhất và hai sự phân kỳ lớn nhất.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| RSP | "Anthropic framework" | Policy mở rộng quy mô có trách nhiệm; Cấp ASL; phiên bản 3.0 Tháng Hai 2026 |
| PF | "OpenAI framework" | Chuẩn bị Framework; năm tiêu chí; v2 Tháng Tư 2025 |
| FSF | "framework của DeepMind" | Framework An toàn Biên giới; CCL; phiên bản 3.0 Tháng Chín 2025 |
| ASL-3 · | "An toàn sinh học cấp độ 3-tương tự" | Anthropic cấp cho các khả năng liên quan đến CBRN; kích hoạt Tháng Năm 2025 |
| CCL | "Mức năng lực quan trọng" | Cấu trúc ngưỡng của DeepMind; mỗi tên miền |
| Trường hợp an toàn | "Lập luận chính thức" | Lập luận bằng văn bản rằng việc triển khai là an toàn có thể chấp nhận được trong trường hợp xấu nhất U |
| Điều khoản điều chỉnh | "Trợ cấp đào ngũ của đối thủ cạnh tranh" | Framework điều khoản để giảm yêu cầu nếu đối thủ cạnh tranh ship không có biện pháp bảo vệ tương đương |

## Đọc thêm

- [Anthropic — Responsible Scaling Policy v3.0 (February 2026)](https://www.anthropic.com/responsible-scaling-policy) — Cấp ASL, lộ trình AI phân tách R&D
- [OpenAI — Updating the Preparedness Framework (April 15, 2025)](https://openai.com/index/updating-our-preparedness-framework/) — năm tiêu chí, điều khoản điều chỉnh
- [DeepMind — Strengthening our Frontier Safety Framework (September 2025)](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — CCL v3.0, Thao túng có hại
- [METR — Common Elements of Frontier AI Safety Policies (2025)](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) - so sánh giữa các phòng thí nghiệm
