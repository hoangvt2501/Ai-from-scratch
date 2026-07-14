# Tuân thủ - SOC 2, HIPAA, GDPR, PCI-DSS, Đạo luật AI của EU, ISO 42001

> Phạm vi bảo hiểm đa framework là đặt cược cho các giao dịch doanh nghiệp năm 2026. **Đạo luật AI của EU**: có hiệu lực từ ngày 1 tháng 8 năm 2024. Hầu hết các yêu cầu về rủi ro cao đều có hiệu lực từ ngày 2 tháng 8 năm 2026. Phạt tiền lên đến 15 triệu euro hoặc 3% doanh thu hàng năm toàn cầu đối với các nghĩa vụ hệ thống rủi ro cao (Điều 99(4)); lên đến 35 triệu euro hoặc 7% đối với các hành vi AI bị cấm (Điều 99(3)). Áp dụng trên toàn cầu nếu phục vụ người dùng ở Liên minh Châu Âu. **Đạo luật AI Colorado**: có hiệu lực từ ngày 30 tháng 6 năm 2026 (bị trì hoãn từ tháng 2 năm 2026 bởi SB25B-004) — đánh giá tác động đối với các hệ thống có rủi ro cao, quyền kháng cáo AI quyết định. Virginia tương tự cho credit/employment/housing/education. **SOC 2 Loại II**: yêu cầu AI B2B trên thực tế (Loại II, không phải Loại I, đối với fintech). **GDPR**: mức phạt cụ thể AI lớn nhất được ghi nhận là 30,5 triệu euro đối với Clearview AI (DPA Hà Lan, tháng 9 năm 2024); Garante của Ý đã phát hành 15 triệu euro đối với OpenAI vào tháng 12 năm 2024 (sau đó bị lật ngược khi kháng cáo vào tháng 3 năm 2026). Biên tập PII theo thời gian thực tại inference là tiêu chuẩn có thể bảo vệ được; Dọn dẹp sau xử lý là không đủ. **HIPAA**: ràng buộc chăm sóc sức khỏe — không thể gửi PHI đến các dịch vụ AI bên ngoài mà không có BAA. **PCI-DSS**: Phạm vi lớp tương tác AI yêu cầu configuration + thỏa thuận hợp đồng, không tự động. **ISO 42001**: tiêu chuẩn quản trị AI mới nổi, yêu cầu mua sắm ngày càng tăng cùng với ISO 27001. Hồ sơ tham khảo: OpenAI duy trì SOC 2 Loại 2, ISO/IEC 27001:2022, ISO/IEC 27701:2019, GDPR/CCPA/HIPAA (BAA)/FERPA, PCI-DSS cho các thành phần thanh toán ChatGPT. Lập bản đồ framework chéo giúp giảm mệt mỏi trong kiểm toán: bản đồ kiểm soát truy cập trên ISO 27001 A.5.15-5.18, GDPR Điều 32, HIPAA §164.312(a).

**Loại:** Học
**Ngôn ngữ:** (Python optional — compliance is policy + process, not code)
**Kiến thức tiên quyết:** Giai đoạn 17 · 25 (Bảo mật), Giai đoạn 17 · 13 (Observability)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Liệt kê bảy frameworks 2026 có liên quan đến LLM sản phẩm và khớp từng sản phẩm với một phân khúc khách hàng.
- Trích dẫn lịch trình thực thi Đạo luật AI của EU (có hiệu lực vào tháng 8 năm 2024; thực thi rủi ro cao vào tháng 8 năm 2026) và mức trần phạt hai cấp (15 triệu euro / 3% đối với các nghĩa vụ rủi ro cao, 35 triệu euro / 7% đối với các hành vi bị cấm).
- Giải thích lý do tại sao dọn dẹp PII sau xử lý là không đủ cho GDPR và đặt tên biên tập lớp inference theo thời gian thực làm tiêu chuẩn có thể bảo vệ.
- Mô tả ánh xạ kiểm soát framework chéo (ví dụ: bản đồ kiểm soát truy cập theo ISO 27001 A.5.15-5.18 + GDPR Điều 32 + HIPAA §164.312(a)).

## Vấn đề

Mua sắm của khách hàng doanh nghiệp yêu cầu SOC 2 Loại II, GDPR, HIPAA BAA, ISO 27001 và "Tuyên bố tuân thủ Đạo luật AI của EU". Nhóm của bạn có SOC 2 Loại I. Bạn còn sáu tháng nữa là đến Loại II và chưa bắt đầu ghi lại Điều 30 của GDPR.

Phạm vi đa framework không phải là vấn đề LLM - đó là vấn đề SaaS của doanh nghiệp, với các lớp phủ dành riêng cho LLM. Các nhóm mua sắm vào năm 2026 muốn có một ma trận với một hàng trên mỗi framework và một cột trên mỗi điều khiển, không phải PDF.

## Khái niệm

### Bảy frameworks

| Framework | Phạm vi | Yêu cầu cụ thể của LLM |
|-----------|-------|--------------------------|
| SOC 2 Loại II | Đường cơ sở SaaS B2B | Kiểm soát Process được kiểm toán trong 6-12 tháng |
| HIPAA | Chăm sóc sức khỏe Hoa Kỳ | BAA yêu cầu; PHI không thể rời khỏi cơ sở hạ tầng mà không có thỏa thuận đã ký |
| GDPR | Người dùng EU | Biên tập PII theo thời gian thực; quyền của chủ thể dữ liệu; Hồ sơ Điều 30 |
| PCI-DSS | Dữ liệu thanh toán | Configuration + hợp đồng thanh toán AI chạm |
| Đạo luật AI EU | Phục vụ người dùng EU | Phân loại cấp rủi ro; Hệ thống rủi ro cao: đánh giá sự phù hợp, tài liệu, ghi nhật ký |
| Đạo luật AI Colorado | Phục vụ cư dân CO | Đánh giá tác động; Quyền kháng cáo |
| Tiêu chuẩn ISO 42001 | Quản trị AI | Mới nổi; cặp với ISO 27001 |

### Dòng thời gian của Đạo luật AI EU

- Ngày 1 tháng 8 năm 2024: có hiệu lực.
- Ngày 2 tháng 2 năm 2025: các hành vi AI bị cấm được thực thi.
- Ngày 2 tháng 8 năm 2026: thực thi các hệ thống rủi ro cao (đánh giá sự phù hợp, tài liệu, ghi nhật ký).
- Tháng 8 năm 2027: các hệ thống rủi ro cao trong các sản phẩm theo luật hài hòa.

Cấp độ rủi ro: Không thể chấp nhận được (bị cấm), Rủi ro cao (tuân thủ + ghi nhật ký), Rủi ro hạn chế (minh bạch), Rủi ro tối thiểu (không ràng buộc). Hầu hết B2B LLM SaaS đều có rủi ro hạn chế; rủi ro cao bắt đầu cho việc làm, tín dụng, giáo dục, thực thi pháp luật, di cư, các dịch vụ thiết yếu.

Tiền phạt (Điều 99): lên đến 15 triệu euro hoặc 3% doanh thu hàng năm toàn cầu do vi phạm các nghĩa vụ hệ thống rủi ro cao (Điều 99(4)); lên đến 35 triệu euro hoặc 7% đối với các hành vi AI bị cấm (Điều 99(3)); tùy theo mức nào cao hơn được áp dụng.

### GDPR — biên tập theo thời gian thực là tiêu chuẩn

Dọn dẹp hậu xử lý (biên tập PII sau khi LLM nhìn thấy) không phải là một tư thế có thể bảo vệ được - model đã xem dữ liệu. Biên tập lớp inference theo thời gian thực là tiêu chuẩn năm 2026:

- Nhận dạng thực thể trước cuộc gọi LLM.
- tokenization nhất quán (Phương pháp tiếp cận lưới) bảo toàn ngữ nghĩa.
- Chỉ lưu trữ prompts đã biên tập + chọn tham gia được đồng ý thô.

Việc thực thi gần đây: 30,5 triệu euro đối với Clearview AI (DPA Hà Lan, tháng 9 năm 2024) là khoản tiền phạt GDPR cụ thể AI lớn nhất được ghi nhận cho đến nay; 15 triệu euro so với OpenAI (Garante của Ý, tháng 12 năm 2024) là khoản tiền phạt cụ thể LLM lớn nhất, mặc dù nó đã bị lật ngược khi kháng cáo vào tháng 3 năm 2026 và phán quyết vẫn đang được xem xét thêm. Các yêu cầu bồi thường sau xử lý đã không thành công trong quá trình kiểm toán.

### HIPAA - BAA không phải là tùy chọn

Bạn không thể gửi PHI đến các dịch vụ AI bên ngoài mà không có Thỏa thuận liên kết kinh doanh đã ký. Cả ba nền tảng LLM siêu quy mô (Bedrock, Azure OpenAI, Vertex) đều cung cấp BAA. OpenAI API trực tiếp cung cấp BAA. Anthropic API trực tiếp cung cấp BAA. Xác nhận trước khi gửi PHI.

### SOC 2 Loại II

Loại I: điều khiển được thiết kế và ghi lại.
Loại II: kiểm soát hoạt động hiệu quả trong 6-12 tháng.

Mua sắm B2B vào năm 2026 mặc định là Loại II. Loại I là người khởi động; Loại II là cổng.

Các trình điều khiển kiểm tra phổ biến: nhật ký truy cập (ai đã xem gì), quản lý thay đổi (nó được triển khai như thế nào), đánh giá rủi ro (hàng quý), ứng phó sự cố (đã được kiểm tra?). Nhật ký đánh giá từ Giai đoạn 17 · 25 có thể tái sử dụng trực tiếp.

### Lập bản đồ framework chéo

Một policy kiểm soát truy cập đáp ứng nhiều kiểm soát framework:

| Điều khiển | Frameworks |
|---------|-----------|
| Ghi nhật ký truy cập | ISO 27001 A.5.15-5.18, GDPR Điều 32, HIPAA §164.312 (a) |
| Quản lý thay đổi | ISO 27001 A.8.32, PCI DSS Yêu cầu 6, phạm vi thông báo vi phạm HIPAA |
| Mã hóa trong quá trình truyền | ISO 27001 A.8.24, GDPR Điều 32, HIPAA §164.312 (e) |
| Quản lý bí mật | ISO 27001 A.8.19, PCI DSS Yêu cầu 8, SOC 2 CC6.1 |

Các công cụ tuân thủ (Drata, Vanta, Secureframe) tự động hóa ánh xạ này. Xứng đáng với chi phí trên quy mô lớn.

### ISO 42001 — mới nổi

Xuất bản cuối năm 2023. Yêu cầu mua sắm ngày càng tăng cùng với ISO 27001. Framework quản trị AI bao gồm quản lý rủi ro, chất lượng dữ liệu, tính minh bạch, giám sát của con người.

### Hồ sơ tham khảo của OpenAI

OpenAI duy trì SOC 2 Loại 2, ISO/IEC 27001:2022, ISO/IEC 27701:2019, GDPR/CCPA/HIPAA (BAA)/FERPA, PCI-DSS cho các thành phần thanh toán ChatGPT. Đó là gần bằng số tiền đặt cược của doanh nghiệp vào năm 2026.

### Những con số bạn nên nhớ

- Tiền phạt theo Đạo luật AI của EU: lên đến 15 triệu euro / 3% (nghĩa vụ rủi ro cao, Điều 99 (4)); lên đến 35 triệu euro / 7% (các hành vi bị cấm, Điều 99(3)).
- Thực thi Đạo luật AI EU có rủi ro cao: Ngày 2 tháng 8 năm 2026.
- Khoản tiền phạt GDPR dành riêng cho AI lớn nhất được ghi nhận: 30,5 triệu euro, Clearview AI (DPA Hà Lan, tháng 9 năm 2024).
- Khoản tiền phạt GDPR lớn nhất dành riêng cho LLM: 15 triệu euro, OpenAI (Garante của Ý, tháng 12 năm 2024; bị lật ngược khi kháng cáo tháng 3 năm 2026).
- Cửa sổ SOC 2 Loại II: 6-12 tháng điều khiển hoạt động.
- Ngày có hiệu lực của Đạo luật AI Colorado: 30 tháng 6 năm 2026 (bị trì hoãn từ tháng 2 năm 2026 bởi SB25B-004).

## Ứng dụng

`code/main.py` là một bảng tính ánh xạ tuân thủ trong Python - được cung cấp một kiểm soát, liệt kê frameworks nó thỏa mãn.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-compliance-matrix.md`. Cho phân khúc khách hàng và địa lý, chỉ định các frameworks và kiểm soát bắt buộc.

## Bài tập

1. Khách hàng doanh nghiệp đầu tiên của bạn yêu cầu tuyên bố SOC 2 Loại II, HIPAA BAA, EU AI Đạo luật. Tư thế tuân thủ khả thi tối thiểu để giành được thỏa thuận là gì?
2. Phân loại ba sản phẩm LLM giả định theo các cấp độ rủi ro của Đạo luật AI EU. Những thay đổi nào có nguy cơ cao?
3. Bạn vô tình gửi PHI đến một nhà cung cấp không có BAA. Xem qua phản ứng sự cố.
4. Tranh luận liệu ISO 42001 có "cần thiết vào năm 2026" đối với một nhà cung cấp AI thị trường tầm trung hay không.
5. Ánh xạ các trường nhật ký kiểm tra LLM của bạn (Giai đoạn 17 · 25) với ít nhất ba framework kiểm soát.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| SOC 2 Loại II | "Kiểm soát đã được kiểm toán" | Kiểm soát hoạt động trong 6-12 tháng, được chứng thực độc lập |
| HIPAA BAA | "Hợp đồng chăm sóc sức khỏe" | Thỏa thuận liên kết kinh doanh; bắt buộc đối với PHI |
| GDPR | "Quyền riêng tư của Liên minh Châu Âu" | Biên tập PII theo thời gian thực là tiêu chuẩn có thể bảo vệ được năm 2026 |
| Đạo luật AI EU | "Quy tắc AI của EU" | Thực thi rủi ro cao Tháng Tám 2026; 15 triệu euro / 3% (nghĩa vụ rủi ro cao) — 35 triệu euro / 7% (các hành vi bị cấm) |
| Đạo luật AI Colorado | "Luật tiểu bang AI Hoa Kỳ" | Ngày 30 tháng 6 năm 2026 có hiệu lực (bị trì hoãn bởi SB25B-004); Đánh giá tác động |
| Tiêu chuẩn ISO 42001 | "Quản trị AI" | Các framework mới nổi về rủi ro AI + minh bạch |
| Tiêu chuẩn ISO 27001 | "ISMS bảo mật" | Cơ sở hệ thống quản lý an toàn thông tin |
| Đánh giá sự phù hợp | "Gói tài liệu AI EU" | Yêu cầu rủi ro cao: tài liệu, thử nghiệm, ghi nhật ký |
| Lập bản đồ framework chéo | "Một điều khiển, nhiều khung hình" | Một policy thỏa mãn nhiều điều khiển framework |

## Đọc thêm

- [OpenAI Security and Privacy](https://openai.com/security-and-privacy/) — hồ sơ tuân thủ tham khảo.
- [GuardionAI — LLM Compliance 2026: ISO 42001, EU AI Act, SOC 2, GDPR](https://guardion.ai/blog/llm-compliance-guide-iso-42001-eu-ai-act-soc2-gdpr-2026)
- [Dsalta — SOC 2 Type 2 Audit Guide 2026: 10 AI Controls](https://www.dsalta.com/resources/ai-compliance/soc-2-type-2-audit-guide-2026-10-ai-powered-controls-every-saas-team-needs)
- [EU AI Act official text](https://eur-lex.europa.eu/eli/reg/2024/1689/oj) — nguồn chính.
- [Colorado AI Act](https://leg.colorado.gov/bills/sb24-205) — nguồn chính.
- [ISO/IEC 42001:2023](https://www.iso.org/standard/81230.html) - AI tiêu chuẩn hệ thống quản lý.
