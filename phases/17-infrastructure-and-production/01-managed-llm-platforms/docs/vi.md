# Nền tảng LLM được quản lý — Bedrock, Vertex AI, Azure OpenAI

> Ba siêu quy mô, ba chiến lược riêng biệt. AWS Bedrock là một thị trường model — Claude, Llama, Titan, Stability, Cohere đằng sau một API. Azure OpenAI là quan hệ đối tác OpenAI độc quyền cùng với Đơn vị thông lượng được cung cấp (PTU) cho dung lượng chuyên dụng. Vertex AI đứng đầu Gemini với câu chuyện dài và đa phương thức hay nhất. Vào năm 2026, Phân tích nhân tạo đo lường Azure OpenAI ở mức trung bình ~50 ms và Bedrock ở ~75 ms trên Llama tương đương 3.1 405B — PTU giải thích khoảng cách vì dung lượng chuyên dụng vượt trội so với chia sẻ theo yêu cầu. Quy tắc quyết định không phải là "cái nào nhanh nhất" mà là "danh mục nào model và bề mặt FinOps phù hợp với sản phẩm của tôi". Bài học này dạy bạn chọn với sự đánh đổi được viết ra, không phải rung cảm.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, toy cost-and-latency comparator)
**Kiến thức tiên quyết:** Giai đoạn 11 (Kỹ thuật LLM), Giai đoạn 13 (Công cụ & Giao thức)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Đặt tên cho ba chiến lược nền tảng (thị trường so với độc quyền so với Gemini đầu tiên) và khớp từng chiến lược với một trường hợp sử dụng sản phẩm.
- Giải thích Đơn vị thông lượng được cung cấp (PTU) mua cho bạn trong Azure OpenAI và lý do tại sao Bedrock theo yêu cầu thường đọc chậm hơn ~25 mili giây ở thang đo 405B.
- Lập sơ đồ bề mặt phân bổ FinOps cho từng nền tảng (Hồ sơ ứng dụng Inference Bedrock so với dự án Vertex cho mỗi nhóm so với phạm vi Azure + đặt trước PTU).
- Viết ra policy "tối thiểu hai nhà cung cấp" và giải thích lý do tại sao khóa một nhà cung cấp là sai lầm đắt giá vào năm 2026.

## Vấn đề

Bạn đã chọn Claude 3.7 Sonnet cho sản phẩm của mình. Bây giờ bạn cần phải phục vụ nó. Bạn có thể gọi trực tiếp cho Anthropic API hoặc bạn có thể gọi thông qua AWS Bedrock hoặc bạn có thể thông qua một gateway. API trực tiếp là đơn giản nhất; Bedrock bổ sung BAA, VPC endpoints, IAM và phân bổ CloudWatch. gateway bổ sung failover, thanh toán hợp nhất và rate limits giữa các nhà cung cấp.

Câu hỏi sâu sắc hơn là danh mục. Nếu bạn cần Claude và Llama và Gemini trong cùng một sản phẩm, bạn không thể mua tất cả chúng từ một nơi trừ khi nơi đó là Bedrock cộng với Vertex cộng với Azure OpenAI đồng thời. Các siêu quy mô không thể hoán đổi cho nhau - mỗi người đã đặt cược khác nhau vào việc ai sở hữu lớp model.

Bài học này lập bản đồ ba cược, khoảng cách độ trễ, khoảng cách FinOps và rủi ro khóa.

## Khái niệm

### Ba chiến lược

**AWS Bedrock** — thị trường. Claude (Anthropic), Llama (Meta), Titan (bên thứ nhất của AWS), Stability (hình ảnh), Cohere (embeddings), Mistral, cùng với các danh mục phụ hình ảnh và embedding. Một API, một nền tảng IAM, một xuất CloudWatch. Cá cược của Bedrock là khách hàng muốn tùy chọn hơn là họ muốn một model.

**Azure OpenAI** — quan hệ đối tác độc quyền. Bạn nhận được GPT-4 / 4o / 5 / o-series, DALL · E, Whisper và fine-tuning của OpenAI models trong trung tâm dữ liệu Azure. Không có OpenAI models nào không phải trong danh mục "Azure OpenAI Service" — những thứ đó được chuyển đến Azure AI Foundry (sản phẩm riêng biệt). Đặt cược của Azure là OpenAI vẫn là biên giới và khách hàng muốn doanh nghiệp kiểm soát mối quan hệ cụ thể đó.

**Vertex AI** - Gemini đầu tiên, mọi thứ khác thứ hai. Gemini 1.5 / 2.0 / 2.5 Flash và Pro, cộng với Model Garden (bên thứ ba). Đặt cược của Vertex là bối cảnh dài đa phương thức - ngữ cảnh 1 triệu token Gemini là yếu tố khác biệt.

### Khoảng cách độ trễ trên quy mô lớn

Phân tích nhân tạo chạy liên tục benchmarks. Tương đương Llama 3.1 Triển khai 405B (chia sẻ theo yêu cầu), Azure OpenAI trung bình đầu tiên-token độ trễ khoảng 50 ms; Đá nền là khoảng 75 ms. Khoảng cách không phải là lỗi AWS mà là dung lượng model sự khác biệt. Azure bán PTU (Đơn vị thông lượng được cung cấp), dự trữ GPU năng lực cho tenant. Tương đương của Bedrock (Thông lượng được cung cấp) tồn tại nhưng bắt đầu khoảng $21/hour mỗi đơn vị và hầu hết khách hàng vẫn sử dụng chia sẻ theo yêu cầu.

Dung lượng chia sẻ theo yêu cầu cạnh tranh với mọi lưu lượng truy cập của khách hàng khác. Dung lượng chuyên dụng thì không. Nếu SLA sản phẩm của bạn là TTFT < 100 ms ở P99, bạn có thể mua PTU trên Azure, mua Thông lượng được cung cấp Bedrock hoặc chấp nhận variance mặc định.

### Kinh tế thông lượng được cung cấp

Azure PTU: một khối điện toán inference dành riêng. Tiết kiệm tới ~70% so với theo yêu cầu đối với khối lượng công việc có thể dự đoán được. Chi phí cố định mỗi giờ bất kể lưu lượng truy cập - bạn trả tiền cho đặt trước ngay cả khi không hoạt động. Mức hòa vốn thường là khoảng 40-60% sử dụng bền vững.

Thông lượng được cung cấp Bedrock: $21-$50 mỗi giờ tùy thuộc vào model và khu vực. Toán học tương tự - hòa vốn là khoảng một nửa mức sử dụng đỉnh. Yêu cầu cam kết hàng tháng.

Dung lượng được cung cấp của Vertex được bán cho mỗi Gemini SKU; Giá cả khác nhau tùy theo model và khu vực và ít được quảng cáo công khai hơn.

### Bề mặt FinOps - điểm khác biệt thực sự

**Hồ sơ Inference ứng dụng Bedrock** là phân bổ rõ ràng nhất trên thị trường. Gắn thẻ hồ sơ bằng `team`, `product`, `feature`; định tuyến tất cả model lời cầu nguyện qua nó; CloudWatch chia nhỏ chi phí trên mỗi cấu hình mà không cần xử lý hậu kỳ. Đã thêm vào năm 2025, vẫn là bản địa hyperscaler chi tiết nhất.

**Vertex** phân bổ là dự án cho mỗi nhóm cộng với nhãn ở mọi nơi. Bạn model từng nhóm dưới dạng một dự án GCP, đặt nhãn trên mọi tài nguyên và sử dụng BigQuery Billing Export + DataStudio cho tổng số. Nhiều công việc hơn, nhưng BigQuery cung cấp cho bạn SQL tùy ý về dữ liệu chi phí.

**Azure** dựa trên phạm vi subscription/resource-group cộng với thẻ, với dự trữ PTU là đối tượng chi phí class đầu tiên. Thẻ được kế thừa từ các nhóm tài nguyên, không phải yêu cầu, vì vậy phân bổ cho mỗi yêu cầu yêu cầu các chỉ số tùy chỉnh của Application Insights hoặc gateway đóng dấu tiêu đề.

Mẫu: Bedrock là bản gốc sạch nhất, Vertex linh hoạt nhất thông qua BigQuery, Azure mờ đục nhất trừ khi bạn đo lường.

### Khóa là rủi ro năm 2026

Cam kết siêu quy mô đơn lẻ là tốt khi một model thống trị. Vào năm 2026, biên giới di chuyển hàng tháng - Claude 3,7 trong một quý, Gemini 2,5 vào quý tiếp theo GPT-5 quý sau đó. Khóa vào một nền tảng sẽ khóa bạn ra khỏi hai phần ba biên giới.

Các nhóm làm việc mẫu áp dụng: tối thiểu hai nhà cung cấp cho bất kỳ cuộc gọi LLM quan trọng nào về sản phẩm. Đá nền cộng với Azure OpenAI là cặp chung - Claude từ cái này, GPT từ cái kia, failover giữa chúng, cùng một gateway. Tăng chi phí là không đáng kể vì gateway tuyến đường tối ưu; Mức tăng tính khả dụng trong thời gian ngừng hoạt động (như sự cố Azure OpenAI tháng 1 năm 2025, sự cố ngừng hoạt động US-East-1 của AWS) là quyết định.

### Nơi lưu trữ dữ liệu, BAA và các ngành được quản lý

Nền tảng: BAA ở hầu hết các khu vực; VPC endpoints; guardrails. Mặc định fintech phổ biến.
Azure OpenAI: HIPAA, SOC 2, ISO 27001; Nơi lưu trữ dữ liệu của EU; vỡ nợ do doanh nghiệp quy định.
Đỉnh: HIPAA, GDPR, nơi lưu trữ dữ liệu trên mỗi khu vực; stack tuân thủ của Google Cloud.

Cả ba đều đáp ứng hộp kiểm cơ bản. Sự khác biệt là policies lưu giữ dữ liệu, cách xử lý nhật ký và liệu giám sát lạm dụng có đọc lưu lượng truy cập của bạn hay không (chọn tham gia mặc định trên hầu hết; chọn không tham gia có sẵn cho doanh nghiệp).

### Những con số bạn nên nhớ

- Azure OpenAI TTFT trung bình trên Llama tương đương 3.1 405B: ~50 ms (với PTU).
- TTFT trung bình nền tảng theo yêu cầu: ~75 ms.
- Thông lượng cung cấp nền tảng: $21-$ 50/hr mỗi đơn vị.
- Azure PTU hòa vốn: ~40-60% sử dụng bền vững.
- Tiết kiệm PTU so với theo yêu cầu khi sử dụng cao: lên đến 70%.

## Ứng dụng

`code/main.py` so sánh ba nền tảng trên khối lượng công việc tổng hợp — nó models kinh tế theo yêu cầu so với PTU, variance TTFT và độ trung thực phân bổ chi phí. Chạy nó để xem PTU mang lại lợi nhuận ở đâu và chiều rộng model của thị trường lớn hơn khoảng cách TTFT ở đâu.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-managed-platform-picker.md`. Với hồ sơ khối lượng công việc (models cần thiết, SLA TTFT, volume hàng ngày, yêu cầu tuân thủ), nó đề xuất một nền tảng chính, dự phòng và kế hoạch đo lường FinOps.

## Bài tập

1. Chạy `code/main.py`. Azure PTU đánh bại mức sử dụng bền vững nào đối với class model 70B? Tính hòa vốn và so sánh với dải 40-60% được quảng cáo.
2. Sản phẩm của bạn cần Claude Sonnet 3.7 và Thiết kế GPT-4o. triển khai hai nhà cung cấp — dành cho siêu quy mô nào, gateway nào ở phía trước, failover policy là gì?
3. Khách hàng chăm sóc sức khỏe được quản lý yêu cầu BAA, nơi lưu trữ dữ liệu ở Miền Đông Hoa Kỳ và TTFT P99 dưới 100 mili giây. Chọn một nền tảng và biện minh với ba features cụ thể.
4. Bạn phát hiện ra hóa đơn Bedrock của mình đã tăng gấp 4 lần trong tháng này mà không có thay đổi về lưu lượng truy cập. Nếu không có Hồ sơ Inference ứng dụng, bạn sẽ tìm ra thủ phạm như thế nào? Với hồ sơ, mất bao lâu?
5. Đọc các trang định giá Azure OpenAI và Bedrock. Đối với khối lượng công việc 100M-token/month Claude, loại nào rẻ hơn — Anthropic API trực tiếp, Bedrock theo yêu cầu hay Thông lượng được cung cấp Bedrock?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Nền tảng | "Dịch vụ AWS LLM" | Model thị trường trên Claude, Llama, Titan, Mistral, Cohere |
| Azure OpenAI | "ChatGPT của Azure" | OpenAI models độc quyền trong trung tâm dữ liệu Azure với các biện pháp kiểm soát doanh nghiệp |
| Đỉnh AI | "LLM của Google" | Nền tảng đầu tiên Gemini với Model Garden cho models của bên thứ ba |
| PTU | "Năng lực chuyên dụng" | Đơn vị thông lượng được cung cấp — inference GPUs dự trữ, định giá theo giờ |
| Hồ sơ Inference ứng dụng | "Gắn thẻ nền tảng" | Hồ sơ cost/usage cho mỗi sản phẩm có thẻ, CloudWatch-native |
| Vườn Model | "Danh mục đỉnh" | Phần model của bên thứ ba của Vertex AI, tách biệt với Gemini |
| Tối thiểu hai nhà cung cấp | "LLM dư thừa" | Policy chạy mọi đường dẫn LLM quan trọng trên các siêu quy mô ≥2 |
| BAA | "Giấy tờ HIPAA" | Thỏa thuận liên kết kinh doanh; bắt buộc đối với PHI; được cung cấp bởi cả ba |
| Giám sát lạm dụng | "Người theo dõi nhật ký" | Quét an toàn phía nhà cung cấp trên prompts/outputs; Chọn không tham gia trong Enterprise |

## Đọc thêm

- [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/) — thẻ giá có thẩm quyền và định giá Thông lượng được cung cấp.
- [Azure OpenAI Service Pricing](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/) - Kinh tế PTU và thẻ tỷ lệ.
- [Vertex AI Generative AI Pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing) - Gemini bậc và phụ phí Model Garden.
- [Artificial Analysis LLM Leaderboard](https://artificialanalysis.ai/) — độ trễ và thông lượng liên tục benchmarks giữa các nhà cung cấp.
- [The AI Journal — AWS Bedrock vs Azure OpenAI CTO Guide 2026](https://theaijournal.co/2026/03/aws-bedrock-vs-azure-openai/) — quyết định doanh nghiệp framework.
- [Finout — Bedrock vs Vertex vs Azure FinOps](https://www.finout.io/blog/bedrock-vs.-vertex-vs.-azure-cognitive-a-finops-comparison-for-ai-spend) - cơ chế phân bổ song song.
