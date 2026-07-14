# CAIS, CAISI và rủi ro quy mô xã hội

> Trung tâm An toàn AI (CAIS, San Francisco, được thành lập năm 2022 bởi Hendrycks và Zhang) công bố framework bốn rủi ro - sử dụng độc hại, chủng tộc AI, rủi ro tổ chức, AI giả mạo - và tuyên bố tháng 5 năm 2023 về nguy cơ tuyệt chủng do hàng trăm giáo sư và lãnh đạo công ty ký. Các bản phát hành năm 2026 từ CAIS: Bảng điều khiển AI để đánh giá model biên giới, Chỉ số Lao động Từ xa (với Thang điểm AI), Báo cáo Chiến lược Siêu trí tuệ AI bản tin Frontiers. Một thực thể riêng biệt: Trung tâm Tiêu chuẩn và Đổi mới AI NIST (CAISI) - các thỏa thuận tự nguyện đối mặt với chính phủ Hoa Kỳ và đánh giá năng lực không được phân loại tập trung vào các rủi ro về vũ khí mạng, sinh học và hóa học. CAIS đánh dấu rủi ro tổ chức là một trong bốn rủi ro cấp cao nhất: văn hóa an toàn, kiểm tra nghiêm ngặt, phòng thủ nhiều lớp và bảo mật thông tin là nền tảng nhưng thường được đánh đổi với tốc độ triển khai. California SB-53, nếu được ký kết, sẽ là quy định rủi ro thảm họa cấp tiểu bang đầu tiên của Hoa Kỳ.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, four-risk inventory and mitigation matcher)
**Kiến thức tiên quyết:** Giai đoạn 15 · 19 (RSP), Giai đoạn 15 · 20 (PF + FSF)
**Thời lượng:** ~45 phút

## Vấn đề

Bài 19 và 20 đề cập đến policies mở rộng quy mô bên trong phòng thí nghiệm. Bài 21 bao gồm đánh giá năng lực độc lập. Bài học này bao gồm quan điểm thứ ba: các tổ chức xã hội dân sự và chính phủ định hình cuộc thảo luận công khai và cơ sở quy định cho rủi ro AI thảm họa.

Hai thực thể riêng biệt quan trọng. CAIS là một tổ chức nghiên cứu phi lợi nhuận xuất bản frameworks suy nghĩ về rủi ro AI và điều phối các tuyên bố công khai. CAISI là một trung tâm của chính phủ Hoa Kỳ trong NIST điều hành các thỏa thuận tự nguyện với các phòng thí nghiệm và đánh giá năng lực chưa được phân loại. Những cái tên có vần; các nhiệm vụ không trùng lặp. Một học viên nên biết cả hai.

Nội dung thực tế: framework bốn rủi ro của CAIS là phân loại rủi ro quy mô xã hội được trích dẫn rộng rãi nhất trong tài liệu. Văn hóa an toàn và rủi ro tổ chức là một trong bốn điều đó, và đây là điều trực tiếp nhất dưới sự kiểm soát của người hành nghề. SB-53 (California) sẽ là quy định rủi ro thảm họa cấp tiểu bang đầu tiên của Hoa Kỳ nếu được ký kết; Khung của dự luật quan trọng vì quy định cấp tiểu bang trong lịch sử đã dẫn dắt hành động liên bang trong policy công nghệ Hoa Kỳ.

## Khái niệm

### CAIS - Trung tâm An toàn AI

- Được thành lập: 2022 tại San Francisco, bởi Dan Hendrycks và các đồng nghiệp (tên "Zhang" đề cập đến một cộng tác viên ban đầu, không phải là người đồng sáng lập hiện tại; xem trang web của CAIS để biết ban lãnh đạo hiện tại).
- Tình trạng: 501 (c) (3) phi lợi nhuận.
- Kết quả đáng chú ý năm 2023: tuyên bố về nguy cơ tuyệt chủng, có chữ ký của hàng trăm nhà nghiên cứu và CEO. Tuyên bố: "Giảm thiểu nguy cơ tuyệt chủng từ AI nên là ưu tiên toàn cầu cùng với các rủi ro quy mô xã hội khác như đại dịch và chiến tranh hạt nhân".
- Kết quả năm 2026: Bảng điều khiển AI để đánh giá model biên giới, Chỉ số lao động từ xa (kết hợp với Thang đo AI), Báo cáo Chiến lược Siêu trí tuệ AI bản tin Frontiers.

### framework bốn rủi ro

framework của CAIS nhóm rủi ro AI thảm khốc thành bốn loại cấp cao nhất:

1. **Sử dụng độc hại**: kẻ xấu sử dụng AI để gây hại (tổng hợp vũ khí sinh học, thông tin sai lệch, tấn công mạng).
2. **AI cuộc đua**: áp lực cạnh tranh giữa các phòng thí nghiệm, công ty hoặc quốc gia đẩy việc triển khai vượt qua điểm an toàn.
3. **Rủi ro tổ chức**: động lực phòng thí nghiệm nội bộ (lỗi văn hóa an toàn, kiểm tra không đủ, bảo mật thiếu nguồn lực) tạo ra việc triển khai kém.
4. **AI giả mạo**: một AI đủ năng lực theo đuổi các mục tiêu mâu thuẫn với phúc lợi của con người.

Đây không phải là phân loại duy nhất; nó được trích dẫn nhiều nhất. Các hạng mục không loại trừ lẫn nhau - một AI giả mạo được sản xuất bởi một tổ chức đánh đổi kiểm toán để lấy tốc độ trong một cuộc đua là cả bốn.

### Nơi rủi ro của tổ chức tồn tại

Trong bốn loại, rủi ro tổ chức là có thể hành động tốt nhất đối với các học viên. Văn hóa an toàn, kiểm tra nghiêm ngặt, phân lớp phòng thủ và bảo mật thông tin của phòng thí nghiệm quyết định liệu model ships của họ với các biện pháp kiểm soát của Bài học 10-18 có thực sự được áp dụng hay không, hay liệu những biện pháp kiểm soát đó có phải là các mục trong danh sách kiểm tra mà không ai xác minh hay không.

Các đòn bẩy rủi ro tổ chức cụ thể:

- **Văn hóa an toàn**: các thành viên trong nhóm có cảm thấy có thể leo thang mối quan tâm mà không phải trả chi phí nghề nghiệp không? Các cuộc khảo sát của CAIS cho thấy đây là một yếu tố dự đoán mạnh mẽ về các đòn bẩy khác.
- **Kiểm toán nghiêm ngặt**: bên ngoài và nội bộ. Đánh giá chỉ nội bộ tạo ra các báo cáo lạc quan.
- **Phòng thủ nhiều lớp**: không có lớp nào là đủ (chủ đề chạy của Giai đoạn 15).
- **Bảo mật thông tin**: Trọng lượng model bị rò rỉ, rò rỉ dữ liệu đánh giá, kỹ thuật bỏ qua màn hình bị rò rỉ. RAND SL-4 trong Bài 19 là một tiêu chuẩn cụ thể.

### CAISI - Trung tâm Tiêu chuẩn và Đổi mới AI

- Hoạt động trong NIST.
- Điều hành các thỏa thuận tự nguyện với các phòng thí nghiệm biên giới.
- Công bố các đánh giá năng lực chưa được phân loại tập trung vào các rủi ro về vũ khí mạng, sinh học và hóa học.
- Khác với CAIS; các từ viết tắt va chạm; kiểm tra URL (nist.gov) để xác nhận bạn đang đọc.

Vai trò của CAISI là đối tác công khai, đối mặt với chính phủ đối với các cam kết phòng thí nghiệm tư nhân của METR (Bài học 21). Các báo cáo của CAISI không được phân loại; Các báo cáo METR thường được kiểm soát NDA. Một học viên đọc cả hai sẽ có một bức tranh đầy đủ hơn.

### California SB-53

Dự luật Thượng viện California (session 2025–2026) giải quyết rủi ro thảm khốc từ models biên giới. Các điều khoản chính như dự thảo:

- Ngưỡng năng lực cụ thể trigger nghĩa vụ cấp tiểu bang.
- Bảo vệ người tố giác cho nhân viên phòng thí nghiệm AI.
- Yêu cầu báo cáo sự cố đối với các hỏng hóc thảm khốc.

Nếu được ký kết, đây sẽ là quy định rủi ro thảm họa cấp tiểu bang đầu tiên của Hoa Kỳ. Bất kể tình trạng ký kết, khung của dự luật định hình cách các cơ quan lập pháp tiểu bang khác tiếp cận vấn đề. Các học viên ở California nên theo dõi tình trạng của dự luật; các học viên ở những nơi khác nên đọc nó để hiểu quy định cấp tiểu bang của Hoa Kỳ có thể sẽ như thế nào.

### Rủi ro quy mô xã hội không phải là vấn đề đơn tầng

Chủ đề chạy của Giai đoạn 15 - phòng thủ theo chiều sâu - cũng áp dụng ở tầng xã hội. Không có tổ chức, quy định hoặc framework đơn lẻ nào có thể đóng được rủi ro thảm khốc. Hệ sinh thái chỉ hoạt động khi:

- Phòng thí nghiệm ship mở rộng quy mô policies (Bài 19, 20).
- Người đánh giá bên ngoài đưa ra các phép đo (Bài 21).
- Theo dõi và công khai xã hội dân sự (CAIS).
- Chính phủ điều hành các chương trình tự nguyện và quy định cơ bản (CAISI, SB-53).
- Các học viên xây dựng các điều khiển nhiều lớp (Bài 10–18).

Đây là tổng hợp cuối cùng cho giai đoạn: mỗi bài học trước là một lớp trong một stack mà tính hoàn chỉnh của nó quan trọng hơn bất kỳ sức mạnh của bất kỳ lớp đơn lẻ nào.

## Ứng dụng

`code/main.py` triển khai một công cụ kiểm kê rủi ro nhỏ. Với một triển khai được đề xuất, nó gắn thẻ triển khai dựa trên bốn danh mục rủi ro và trả về danh sách kiểm tra giảm thiểu. Nó là một công cụ hỗ trợ đọc cho framework, không thay thế cho phán đoán của con người.

## Sản phẩm bàn giao

`outputs/skill-societal-risk-review.md` xem xét việc triển khai tư thế rủi ro quy mô xã hội: nó chạm vào loại nào trong bốn loại, những biện pháp giảm thiểu nào được áp dụng, mức độ rủi ro của tổ chức là gì.

## Bài tập

1. Chạy `code/main.py`. Cung cấp trong ba triển khai tổng hợp ở các quy mô khác nhau. Xác nhận bốn thẻ rủi ro phù hợp với những gì bạn mong đợi; Xác định một trường hợp công cụ bị thẻ dưới hoặc thẻ.

2. Đọc đầy đủ tài liệu bốn rủi ro của CAIS. Chọn một danh mục rủi ro và viết hai đoạn về những gì bạn tin là sự phát triển quan trọng nhất của năm 2026 trong danh mục đó.

3. Đọc dự thảo hiện tại của California SB-53. Xác định một điều khoản mà bạn tin rằng củng cố tình trạng rủi ro thảm khốc và một điều khoản bạn tin rằng làm suy yếu nó. Biện minh cho cả hai.

4. Chọn một triển khai production AI mà bạn biết (của bạn hoặc một triển khai đã xuất bản). Chấm điểm nó dựa trên các đòn bẩy phụ rủi ro của tổ chức: văn hóa an toàn, kiểm toán nghiêm ngặt, phòng thủ nhiều lớp, bảo mật thông tin. Cái nào yếu nhất? Chi phí để đưa nó đến ngang bằng là bao nhiêu?

5. Phác thảo phiên bản 2028 của framework bốn rủi ro phản ánh một năm năng lực bổ sung và một năm kinh nghiệm triển khai bổ sung. Bạn sẽ thêm, xóa hoặc nhóm lại những gì?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|---|---|---|
| CAIS | "Trung tâm An toàn AI" | Phi lợi nhuận; framework bốn rủi ro; Tuyên bố tuyệt chủng năm 2023 |
| CAISI | "Chính phủ Hoa Kỳ AI an toàn" | Trung tâm NIST; thỏa thuận tự nguyện; Đánh giá chưa được phân loại |
| framework bốn rủi ro | "Phân loại của CAIS" | sử dụng độc hại, chạy đua AI, rủi ro tổ chức, AI giả mạo |
| Sử dụng độc hại | "Kẻ xấu sử dụng AI" | Vũ khí sinh học, thông tin sai lệch, tấn công mạng |
| AI cuộc đua | "Áp lực cạnh tranh" | Labs/companies/nations đẩy triển khai vượt quá an toàn |
| Rủi ro tổ chức | "Thất bại nội bộ phòng thí nghiệm" | Văn hóa an toàn, kiểm toán, phòng thủ, an ninh thông tin |
| Rogue AI | "Sai lệch agent" | Có khả năng AI theo đuổi các mục tiêu mâu thuẫn với phúc lợi của con người |
| California SB-53 | "Quy định cấp tiểu bang" | Dự luật 2025–2026; quy định rủi ro thảm họa đầu tiên của tiểu bang Hoa Kỳ nếu được ký kết |

## Đọc thêm

- [Center for AI Safety](https://safe.ai/) - ngôi nhà tổ chức của framework bốn rủi ro.
- [CAIS — AI Risks that Could Lead to Catastrophe](https://safe.ai/ai-risk) - bài báo bốn rủi ro.
- [CAIS — May 2023 statement on extinction risk](https://safe.ai/statement-on-ai-risk) — tuyên bố chung ngắn.
- [NIST CAISI](https://www.nist.gov/caisi) - trung tâm đổi mới và tiêu chuẩn AI hướng đến chính phủ.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) - kết nối các cam kết cấp phòng thí nghiệm với khung quy mô xã hội.
