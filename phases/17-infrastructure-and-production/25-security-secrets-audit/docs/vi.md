# Bảo mật — Bí mật, Xoay vòng khóa API, Nhật ký kiểm tra Guardrails

> Loại bỏ sự lan rộng bí mật thông qua các vault tập trung (HashiCorp Vault, AWS Secrets Manager, Azure Key Vault). Không bao giờ lưu trữ thông tin đăng nhập trong tệp config, tệp env trong VCS, bảng tính. Sử dụng vai trò IAM thay vì khóa tĩnh; OIDC cho CI/CD. Mô hình AI-gateway là giải pháp năm 2026: ứng dụng → gateway → model nhà cung cấp, với gateway lấy thông tin đăng nhập từ vault tại runtime. Xoay vòng trong vault và tất cả các ứng dụng sẽ bắt đầu trong vài phút - không cần triển khai lại, không có thông báo Slack "ai có khóa mới". Vòng quay policy ≤90 ngày; quét bằng TruffleHog / GitGuardian / Gitleaks trên mỗi commit. Không tin cậy: MFA, SSO, RBAC/ABAC, tokens tồn tại trong thời gian ngắn, tư thế thiết bị. Quét PII sử dụng nhận dạng thực thể để che PHI/PII trước khi chuyển tiếp; tokenization nhất quán (Phương pháp tiếp cận lưới) ánh xạ các giá trị nhạy cảm đến các trình giữ chỗ ổn định để LLM duy trì ngữ nghĩa code/relationship. Đầu ra mạng: LLM các dịch vụ trong danh sách trắng mạng con VPC/VNet chuyên dụng chỉ `api.openai.com`, `api.anthropic.com`, v.v.; chặn tất cả các chuyến đi khác. Trình điều khiển sự cố năm 2026: Cuộc tấn công chuỗi cung ứng Vercel thông qua thông tin đăng nhập CI/CD bị xâm phạm đã đánh cắp môi trường trên hàng nghìn lần triển khai của khách hàng.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, toy PII-scrubber + audit-log writer)
**Kiến thức tiên quyết:** Giai đoạn 17 · 19 (AI Gateways), Giai đoạn 17 · 13 (Observability)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Liệt kê bốn mẫu chống quản lý bí mật (tệp config trong VCS, môi trường mã hóa cứng, bảng tính, khóa tĩnh) và đặt tên cho các mẫu thay thế của chúng.
- Giải thích mô hình AI-gateway-pulls-from-vault là tiêu chuẩn production năm 2026.
- Triển khai trình lọc PII với tokenization nhất quán (cùng giá trị → cùng trình giữ chỗ) để ngữ nghĩa tồn tại.
- Kể tên sự cố chuỗi cung ứng Vercel năm 2026 và những gì nó dạy về vệ sinh thông tin xác thực CI/CD.

## Vấn đề

Một thực tập sinh commits `.env` với chìa khóa API. Họ xóa nó nhanh chóng. Các khóa đã có trong lịch sử git - quét GitGuardian bắt được nó, process xoay vòng của bạn là "Làm lỏng nhóm, cập nhật 40 tệp config, triển khai lại tất cả các dịch vụ". 8 giờ sau, một nửa dịch vụ của bạn hoạt động và một nửa đang chờ triển khai windows.

Riêng prompts người dùng bao gồm "SSN của tôi là 123-45-6789". Prompt đến OpenAI. Bạn có BAA nhưng policy nội bộ của bạn là che PII trước khi chuyển tiếp. Bạn đã không.

Riêng biệt, LLM pod của cụm EKS có thể truy cập bất kỳ máy chủ internet nào. Ai đó lấy cắp dữ liệu thông qua tra cứu DNS đến miền do kẻ tấn công kiểm soát. Không có gì ngăn cản nó.

Bảo mật cho các dịch vụ LLM phải giải quyết cả ba vectors. Thông tin đăng nhập được hỗ trợ bởi Vault. Chà PII. Lọc đầu ra mạng. Nhật ký kiểm tra.

## Khái niệm

### Vault tập trung + kéo vai trò IAM

**Vault**: HashiCorp Vault, AWS Secrets Manager, Azure Key Vault, GCP Secret Manager. Một nguồn sự thật.

**Vai trò IAM**: app/gateway xác thực thông qua danh tính IAM, không phải khóa tĩnh. Vault trả lại bí mật cho cuộc đời của token.

**Mô hình AI-gateway**: gateway kéo `OPENAI_API_KEY` từ kho tiền vào thời điểm yêu cầu. Xoay trong hầm; yêu cầu tiếp theo sẽ nhận được khóa mới. Không triển khai lại.

### Vòng quay policy ≤ 90 ngày

Tất cả các khóa API, tokens gốc kho tiền CI/CD thông tin đăng nhập. Xoay vòng tự động nếu có thể. Xoay thủ công được ghi lại và theo dõi.

### Quét bí mật

- **TruffleHog** — regex + entropy trên commits.
- **GitGuardian** — thương mại, accuracy cao.
- **Gitleaks** — OSS, chạy trong CI.

Chạy trên mọi commit. Chặn PR nếu phát hiện bí mật mới.

### Tư thế không tin cậy

- Yêu cầu MFA trên tất cả các tài khoản.
- SSO qua SAML/OIDC.
- RBAC (dựa trên vai trò) hoặc ABAC (dựa trên thuộc tính) để truy cập chi tiết.
- tokens tồn tại trong thời gian ngắn (giờ, không phải ngày).
- Vị trí thiết bị — chỉ các thiết bị công ty có mã hóa đĩa.

### Chà PII / PHI

Trước khi prompt rời khỏi cơ sở hạ tầng của bạn:

1. Nhận dạng thực thể (spaCy NER, Presidio, thương mại).
2. Mặt nạ các thực thể phù hợp: `"My SSN is 123-45-6789"` → `"My SSN is [SSN_TOKEN_A3F]"`.
3. tokenization nhất quán (Phương pháp tiếp cận lưới): cùng một giá trị ánh xạ đến cùng một trình giữ chỗ để LLM duy trì các mối quan hệ.
4. Ánh xạ ngược tùy chọn cho phản hồi LLM.

Bộ lọc biểu thức chính quy tĩnh bắt các mẫu cơ bản; NER bắt được nhiều hơn. Sử dụng cả hai.

### Đầu vào + đầu ra guardrails

Đầu vào: chặn các bài bẻ khóa đã biết, các chủ đề bị cấm; giới hạn tốc độ cho mỗi người dùng.

Đầu ra: xóa biểu thức chính quy để tìm bí mật bị rò rỉ (API mẫu chính, mẫu email trong ngữ cảnh từ chối), phân loại cho các vi phạm policy.

### Danh sách trắng đầu ra mạng

LLM dịch vụ trong một mạng con chuyên dụng:
- Danh sách trắng: `api.openai.com`, `api.anthropic.com`, vector DB endpoints, vault endpoints.
- Mọi thứ khác: thả.
- DNS thông qua trình phân giải chỉ trong danh sách cho phép (tránh lọc đường hầm DNS).

### Nhật ký kiểm tra

Nhật ký bất biến của mọi cuộc gọi LLM với:
- Dấu thời gian.
- Người dùng / tenant.
- Prompt băm (không phải prompt thô để bảo mật).
- Model + phiên bản.
- Token có giá trị.
- Giá cả.
- Hàm băm phản hồi.
- Bất kỳ chuyến đi guardrail nào.

Giữ lại theo yêu cầu quy định (SOC 2 1 năm, HIPAA 6 năm).

### Sự cố Vercel năm 2026

Tấn công chuỗi cung ứng: thông tin đăng nhập CI/CD bị xâm phạm đã đánh cắp môi trường trên hàng nghìn lần triển khai của khách hàng. Bài học: CI/CD thông tin đăng nhập tương đương với sản phẩm. Lưu trữ trong hầm. Phạm vi hẹp. Xoay mạnh mẽ.

### Những con số bạn nên nhớ

- policy vòng quay: ≤ 90 ngày.
- Quét trên mọi commit: TruffleHog / GitGuardian / Gitleaks.
- Vercel 2026: CI/CD tín nhiệm bị xâm phạm → hàng nghìn môi trường khách hàng bị rò rỉ.
- Lưu giữ nhật ký kiểm toán: SOC 2 = 1 năm, HIPAA = 6 năm.

## Ứng dụng

`code/main.py` triển khai máy lọc PII đồ chơi với tokenization nhất quán và nhật ký kiểm tra chỉ nối thêm.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-llm-security-plan.md`. Với phạm vi quy định và trạng thái hiện tại, lập kế hoạch di chuyển, dọn dẹp, đầu ra, nhật ký kiểm tra kho tiền.

## Bài tập

1. Chạy `code/main.py`. Gửi hai prompts tham chiếu đến cùng một SSN. Xác nhận cả hai đều nhận được cùng một trình giữ chỗ.
2. Thiết kế policy đầu ra mạng cho triển khai vLLM-on-EKS gọi OpenAI + Anthropic + Weaviate.
3. Bạn khám phá ra một chìa khóa trong lịch sử git (2 tuổi). Phản hồi chính xác là gì - xoay phím, quét lịch sử hoặc cả hai? Biện minh.
4. Nhật ký kiểm tra của bạn tăng 10 bậc lưu giữ thiết kế GB/day. (nóng 30 ngày, ấm 12 tháng, lạnh 6 năm).
5. Tranh luận liệu tokenization ngược (thay thế các giá trị thực trở lại phản hồi LLM) có xứng đáng với sự phức tạp so với việc giữ cho các trình giữ chỗ hiển thị.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Hầm | "Cửa hàng bí mật" | Dịch vụ quản lý thông tin xác thực tập trung |
| Vai trò IAM | "Xác thực dựa trên danh tính" | Vai trò do ứng dụng đảm nhận; Trả lại tín nhiệm tồn tại trong thời gian ngắn |
| OIDC cho CI/CD | "tokens phát hành cloud" | Không có khóa tĩnh trong CI - nhận dạng qua OIDC |
| TruffleHog / GitGuardian / Gitleaks | "Máy quét bí mật" | Phát hiện bí mật Commit lần |
| RBAC / ABAC | "Kiểm soát truy cập" | Dựa trên vai trò so với dựa trên thuộc tính |
| Chà PII | "Mặt nạ dữ liệu" | Xóa hoặc mã hóa các thực thể nhạy cảm |
| tokenization nhất quán | "Trình giữ chỗ ổn định" | Cùng một giá trị → cùng một token mỗi lần |
| Phương pháp tiếp cận lưới | "Lưới tokenization" | Mẫu tokenization bảo tồn ngữ nghĩa |
| Danh sách trắng lối ra | "Danh sách cho phép gửi đi" | Chỉ những miền được phép truy cập được |
| Nhật ký kiểm tra | "Lịch sử bất biến" | Bản ghi chỉ thêm để tuân thủ |

## Đọc thêm

- [Doppler — Advanced LLM Security](https://www.doppler.com/blog/advanced-llm-security)
- [Portkey — Manage LLM API keys with secret references](https://portkey.ai/blog/secret-references-ai-api-key-management/)
- [Datadog — LLM Guardrails Best Practices](https://www.datadoghq.com/blog/llm-guardrails-best-practices/)
- [JumpServer — Secrets Management Best Practices 2026](https://www.jumpserver.com/blog/secret-management-best-practices-2026)
- [Microsoft Presidio](https://github.com/microsoft/presidio) — Phát hiện và ẩn danh PII.
- [HashiCorp Vault docs](https://developer.hashicorp.com/vault/docs)
