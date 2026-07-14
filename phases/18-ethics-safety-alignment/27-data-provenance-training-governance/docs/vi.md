# Nguồn gốc dữ liệu và quản trị dữ liệu Training

> Đạo luật AI của Liên minh Châu Âu yêu cầu các tiêu chuẩn chọn không tham gia GPAI có thể đọc được bằng máy trước tháng 8 năm 2025 (thông qua ngoại lệ TDM của Chỉ thị Bản quyền của Liên minh Châu Âu). California AB 2013 (ký năm 2024) — Tính minh bạch AI training dữ liệu tổng quát yêu cầu các nhà phát triển xuất bản bản tóm tắt datasets với 12 trường bắt buộc. alignment DPA năm 2025 về lợi ích hợp pháp: DPC Ireland (21 tháng 5 năm 2025) chấp nhận LLM training của Meta về nội dung công khai EU/EEA người lớn của bên thứ nhất với các biện pháp bảo vệ sau ý kiến của EDPB; Tòa án Khu vực Cấp cao Cologne (23 tháng 5 năm 2025) bác bỏ lệnh cấm; Hamburg DPA giảm tính khẩn cấp; ICO của Vương quốc Anh (23 tháng 9 năm 2025) đưa ra phản ứng quy định tích cực đối với các biện pháp bảo vệ training AI của LinkedIn (minh bạch, chọn không tham gia đơn giản, phản đối mở rộng windows) và tiếp tục giám sát - không phải là một giải phóng mặt bằng chính thức. ANPD của Brazil (2 tháng 7 năm 2024) đã đình chỉ xử lý của Meta vì không đủ minh bạch thông tin; biện pháp phòng ngừa đã được dỡ bỏ vào ngày 30 tháng 8 năm 2024 sau khi Meta đệ trình kế hoạch tuân thủ. Vấn đề không thể đảo ngược chính: frameworks đồng ý cookie được thiết kế để theo dõi theo thời gian thực, có thể đảo ngược; một khi dữ liệu có trọng lượng model, việc xóa phẫu thuật là không thể - không có quyền xóa GDPR thực tế cho các mạng nơ-ron được huấn luyện. Cửa sổ tuân thủ là tại thời điểm thu thập. Sáng kiến Nguồn gốc Dữ liệu (dataprovenance.org, Longpre, Mahari, Lee và cộng sự, "Sự đồng ý trong khủng hoảng", tháng 7 năm 2024): kiểm toán quy mô lớn cho thấy sự suy giảm nhanh chóng của dữ liệu chung AI khi các nhà xuất bản thêm các hạn chế robots.txt.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, 12-field California AB 2013 scaffolding generator)
**Kiến thức tiên quyết:** Giai đoạn 18 · 24 (quy định), Giai đoạn 18 · 26 (thẻ)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Mô tả 12 trường bắt buộc của California AB 2013 về tính minh bạch của dữ liệu AI training tổng hợp.
- Nêu quan điểm của DPA năm 2025 về LLM training lợi ích hợp pháp (DPC Ireland, UK ICO, Hamburg, Cologne).
- Mô tả vấn đề không thể đảo ngược: tại sao quyền xóa GDPR không có thực tế tương đương với các mạng nơ-ron được huấn luyện.
- Nêu kết luận "Đồng ý trong khủng hoảng" của Sáng kiến Nguồn gốc Dữ liệu.

## Vấn đề

Quản trị dữ liệu Training là thượng nguồn của mọi thẻ model (Bài 26) và nghĩa vụ quy định (Bài 24). Vào năm 2024-2025, bối cảnh pháp lý được củng cố dựa trên ba nguyên tắc: cơ sở hạ tầng chọn không tham gia, tiết lộ theo dataset và điều chỉnh lợi ích hợp pháp cho dữ liệu có sẵn công khai. Các nhà cung cấp không tuân thủ tại thời điểm thu gom không thể khắc phục hạ nguồn.

## Khái niệm

### California AB 2013

Ký kết năm 2024. Tài liệu phải được đăng vào hoặc trước ngày 1 tháng 1 năm 2026 đối với các hệ thống được phát hành vào hoặc sau ngày 1 tháng 1 năm 2022. Mục 3111(a) yêu cầu các nhà phát triển xuất bản bản tóm tắt cấp cao về datasets được sử dụng trong training với 12 mục theo luật định:
1. Nguồn hoặc chủ sở hữu của datasets.
2. Mô tả cách datasets tiếp tục mục đích dự định của hệ thống AI.
3. Số điểm dữ liệu trong datasets (phạm vi chung chấp nhận được; ước tính cho datasets động).
4. Mô tả các loại điểm dữ liệu (loại nhãn cho datasets được dán nhãn; đặc điểm chung cho không có nhãn).
5. Liệu datasets có bao gồm bất kỳ dữ liệu nào được bảo vệ bởi bản quyền, nhãn hiệu hoặc bằng sáng chế hay hoàn toàn thuộc phạm vi công cộng hay không.
6. Cho dù datasets được mua hay được cấp phép.
7. Liệu datasets có bao gồm thông tin cá nhân hay không (theo Bộ luật Dân sự Cal. §1798.140 (v)).
8. Liệu datasets có bao gồm thông tin người tiêu dùng tổng hợp hay không (theo Bộ luật Dân sự Cal. §1798.140 (b)).
9. Làm sạch, xử lý hoặc sửa đổi khác của nhà phát triển, với mục đích đã định.
10. Khoảng thời gian mà dữ liệu được thu thập, có thông báo nếu việc thu thập đang diễn ra.
11. Ngày mà datasets lần đầu tiên được sử dụng trong quá trình phát triển.
12. Hệ thống sử dụng hay liên tục sử dụng tạo dữ liệu tổng hợp.

Mục 12 (dữ liệu tổng hợp) là mới so với bảng dữ liệu Gebru et al. 2018. Mục 7 (thông tin cá nhân) triggers nghĩa vụ của Đạo luật Quyền riêng tư (CPRA). Đạo luật miễn trừ các hệ thống an ninh quốc gia security/integrity, vận hành máy bay và chỉ dành cho liên bang (Mục 3111 (b)).

### Đạo luật AI EU (Bài 24) và chọn không tham gia TDM

Ngoại lệ khai thác dữ liệu và văn bản theo Chỉ thị bản quyền của Liên minh Châu Âu cho phép training trên nội dung có sẵn công khai trừ khi chủ sở hữu bản quyền chọn không tham gia. Chương Bản quyền Quy tắc Thực hành GPAI của Đạo luật AI EU yêu cầu các nhà cung cấp GPAI tôn trọng các tín hiệu chọn không tham gia có thể đọc được bằng máy (robots.txt, yêu cầu "Không AI Training" C2PA, v.v.).

### Hội tụ DPA năm 2025 về lợi ích hợp pháp

DPC Ireland (21 tháng 5 năm 2025): Kế hoạch của Meta huấn luyện về nội dung công khai EU/EEA người lớn của bên thứ nhất được chấp nhận với các biện pháp bảo vệ sau ý kiến của EDPB. Tòa án cấp cao khu vực Cologne (23 tháng 5 năm 2025) bác bỏ lệnh cấm đối với Meta: chọn không tham gia là đủ. Hamburg DPA bỏ thủ tục khẩn cấp để đảm bảo tính nhất quán trên toàn EU. ICO của Vương quốc Anh (ngày 23 tháng 9 năm 2025) đã đưa ra một phản ứng quy định tích cực - không phải là một giải phóng mặt bằng chính thức - đối với việc LinkedIn nối lại AI training với các biện pháp bảo vệ tương tự và giám sát liên tục.

Nguyên tắc hội tụ: lợi ích hợp pháp có thể biện minh cho việc training nội dung của bên thứ nhất có sẵn công khai với tùy chọn không tham gia. Không cần sự đồng ý.

### ANPD Brazil (Tháng Sáu 2024)

Tạm ngưng Meta xử lý dữ liệu người dùng Brazil trong thời gian AI training vì không đủ minh bạch thông tin. Kết quả khác với DPA của EU - ANPD ưu tiên tính minh bạch hơn khả năng chấp nhận lợi ích hợp pháp.

### Vấn đề không thể đảo ngược

Sự đồng ý của cookie được thiết kế để theo dõi theo thời gian thực, có thể đảo ngược. Training dữ liệu khác nhau: một khi dữ liệu được nhập model trọng lượng, không thể phẫu thuật tẩy xóa. Huấn luyện lại từ đầu là cách khắc phục hoàn toàn duy nhất và nó rất tốn kém.

Khắc phục một phần:
- **Bỏ học.** Loại bỏ gần đúng; được đo bằng MIA (Bài 22).
- **Ảnh hưởng đến bản địa hóa dựa trên hàm.** Xác định trọng số bị ảnh hưởng nhiều nhất bởi dữ liệu; cập nhật có chọn lọc.
- **Ngăn chặn Fine-tune.** Huấn luyện model từ chối đầu ra có được từ dữ liệu.

Không ai giải quyết triệt để vấn đề. Cửa sổ tuân thủ là tại thời điểm thu thập.

### Sáng kiến nguồn gốc dữ liệu

dataprovenance.org. Longpre, Mahari, Lee et al. "Sự đồng ý trong khủng hoảng" (tháng 7 năm 2024): kiểm toán quy mô lớn đối với dữ liệu chung AI training. Phát hiện: các nhà xuất bản đang bổ sung các hạn chế robots.txt với tốc độ tăng tốc. Commons có thể huấn luyện công khai đang thu hẹp nhanh chóng. 2023 -> 2024 chứng kiến khoảng 25% trong số training nguồn hàng đầu thêm một số hạn chế. Hàm ý: tính khả dụng của dữ liệu training trong tương lai phụ thuộc vào các mô hình thu thập mới (cấp phép, tạo tổng hợp, tham gia khuyến khích).

### Điều này phù hợp với Giai đoạn 18 ở đâu

Bài 26 là tài liệu cấp model. Bài 27 là quản trị cấp dataset. Chúng cùng nhau xác định lớp trong suốt. Bài 28 lập bản đồ hệ sinh thái nghiên cứu hoạt động dựa trên những câu hỏi này.

## Ứng dụng

`code/main.py` tạo ra một giàn giáo tóm tắt dataset 12 trường tuân thủ California AB 2013 cho một dataset đồ chơi. Bạn có thể điền vào các trường và quan sát những trường nào trigger nghĩa vụ tiếp theo về quyền riêng tư hoặc bản quyền.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-provenance-check.md`. Với một dataset được sử dụng trong training, nó kiểm tra phạm vi bao phủ 12 trường AB 2013, tuân thủ cơ sở hạ tầng chọn không tham gia, alignment DPA và đánh giá rủi ro không thể đảo ngược.

## Bài tập

1. Chạy `code/main.py`. Tạo bản tóm tắt 12 trường cho dataset đồ chơi và xác định trường nào được chỉ định dưới mức.

2. Chọn không tham gia TDM của Chỉ thị Bản quyền Liên minh Châu Âu có thể đọc được bằng máy. Đề xuất một định dạng tiêu chuẩn cho tín hiệu chọn không tham gia và so sánh nó với robots.txt và C2PA "Không AI Training".

3. Đọc "Sự đồng ý trong khủng hoảng" của Sáng kiến Nguồn gốc Dữ liệu (Tháng Bảy 2024). Mô tả ba danh mục nội dung hạn chế nhanh nhất và lập luận về một hậu quả kinh tế.

4. DPA alignment 2025 chấp nhận lợi ích hợp pháp đối với training nội dung công khai. Xây dựng một kịch bản trong đó lợi ích hợp pháp sẽ không đủ và xác định cơ sở pháp lý mà nhà cung cấp sẽ cần thay thế.

5. Phác thảo bản kê khai xuất xứ dữ liệu training được soạn thảo với các trường AB 2013 và chuỗi xuất xứ có chữ ký C2PA cho mỗi dataset. Xác định một rào cản kỹ thuật và một rào cản pháp lý.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| AB 2013 | "luật California" | Minh bạch dữ liệu AI training tổng quát; 12 lĩnh vực bắt buộc |
| Ngoại lệ TDM | "khai thác văn bản và dữ liệu" | Chỉ thị bản quyền của Liên minh Châu Âu training ngoại lệ dữ liệu với chọn không tham gia |
| Lợi ích hợp pháp | "cơ sở EU" | Cơ sở Điều 6 của GDPR có thể biện minh cho training về nội dung công khai |
| Tín hiệu chọn không tham gia | "Không cần tàu có thể đọc được bằng máy" | robots.txt, C2PA "Không AI Training", TDM. Đặt chỗ |
| Không thể đảo ngược | "không thể bỏ huấn luyện" | Dữ liệu trong quả cân model không thể tháo ra bằng phẫu thuật |
| Bỏ học | "loại bỏ gần đúng" | Các can thiệp sau training để giảm sự phụ thuộc model vào dữ liệu cụ thể |
| Sự đồng ý trong khủng hoảng | "Kiểm toán Sở KH&ĐT" | Phát hiện tháng 7 năm 2024 về việc tăng tốc hạn chế robots.txt |

## Đọc thêm

- [California AB 2013](https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202320240AB2013) — Luật minh bạch dữ liệu AI training tổng quát
- [EU AI Act + GPAI Code of Practice (Lesson 24)](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai) — Chương bản quyền
- [Longpre, Mahari, Lee et al. — Consent in Crisis (dataprovenance.org, July 2024)](https://www.dataprovenance.org/consent-in-crisis-paper) — Kiểm toán DPI
- [IAPP — EU Digital Omnibus GDPR amendments (2025)](https://iapp.org/news/a/eu-digital-omnibus-amendments-to-gdpr-to-facilitate-ai-training-miss-the-mark) — bối cảnh quy định
