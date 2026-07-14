# Định tuyến Model như một Primitive giảm chi phí

> Một nhà môi giới động đánh giá mọi yêu cầu (loại nhiệm vụ, độ dài token, embedding sự giống nhau, độ tin cậy) và gửi các truy vấn đơn giản đến một model giá rẻ, leo thang những truy vấn phức tạp lên một model biên giới. Còn được gọi là model xếp tầng. Production nghiên cứu điển hình cho thấy giảm 20-60% chi phí ở chất lượng ISO trong US/UK/EU triển khai; cải thiện hiệu quả định tuyến 30% trên SaaS volume cao biến thành tiết kiệm hàng năm sáu con số. Bối cảnh năm 2026 là giá LLM inference giảm ~10 lần mỗi năm - GPT-4-class token đã tăng từ $20/M to ~$ 0.40/M từ cuối năm 2022 đến năm 2026. Hầu hết sự sụt giảm là phục vụ tốt hơn stacks (Giai đoạn 17 · 04-09), không phải phần cứng. Định tuyến là cách bạn chuyển đổi mức giảm giá đó thành lợi nhuận mà không có hồi quy sản phẩm. Chế độ thất bại là trôi model rẻ: lộ trình đẩy 40% sang model yếu hơn, chất lượng giảm 3-5% đối với các nhiệm vụ suy luận, không ai nhận thấy trong một quý. Cổng các tuyến đường theo các chỉ số chất lượng trực tuyến, không chỉ các bộ đánh giá ngoại tuyến.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, toy cascading router simulator)
**Kiến thức tiên quyết:** Giai đoạn 17 · 01 (Nền tảng LLM được quản lý), Giai đoạn 17 · 19 (AI Gateways)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Giải thích model xếp tầng: giá rẻ trước với kiểm tra độ tin cậy, leo thang khi độ tin cậy thấp.
- Liệt kê bốn tín hiệu định tuyến (phân loại nhiệm vụ, độ dài prompt embedding sự tương đồng với bộ cứng đã biết, sự tự tin từ lần vượt qua đầu tiên).
- Tính toán chi phí kết hợp dự kiến khi phân chia định tuyến mục tiêu và dung sai loss chất lượng.
- Đặt tên cho chỉ số giám sát trôi dạt (cổng chất lượng trực tuyến) bắt được model rẻ.

## Vấn đề

Chi phí dịch vụ của bạn $80k/month trên GPT-5. Phân tích của bạn cho thấy 70% truy vấn rất đơn giản: "mấy giờ ở Paris?" "Diễn đạt lại câu này." Một Haiku-class model xử lý những điều đó một cách hoàn hảo với 3% chi phí. 30% nhu cầu GPT-5Lý do của - mã hóa, toán học, lập kế hoạch nhiều bước.

Nếu bạn định tuyến 70% thành rẻ và 30% là đắt, hóa đơn của bạn sẽ giảm ~65% với cùng chất lượng sản phẩm. Đây là định tuyến. Bí quyết là xây dựng nhà môi giới mà không làm giảm chất lượng.

## Khái niệm

### Bốn tín hiệu định tuyến

1. **Phân loại nhiệm vụ**: simple/complex/codegen/math/chat. Có thể là một bộ phân loại dựa trên quy tắc, một LLM (Haiku-class tại $0.25/M), hoặc embedding tương tự như các nhóm được dán nhãn. Đầu ra: tuyến đường = rẻ / cân bằng / biên giới.

2. **Prompt dài**: prompts >4K tokens thường cần biên giới để gắn kết. Prompts <500 tokens thường không.

3. **Embedding Tương tự với tập cứng đã biết**: Nếu truy vấn gần (cosin > 0,88) với vùng lưu trữ cứng đã biết, hãy chuyển trực tiếp lên biên giới.

4. **Tự tin ngay từ lần vượt qua đầu tiên**: gửi đến giá rẻ; Nếu các probs nhật ký của model cho thấy độ tin cậy thấp HOẶC nó từ chối HOẶC xuất ngôn ngữ phòng ngừa rủi ro, hãy thử lại trên biên giới. Thêm độ trễ P95 trên ~10% lưu lượng truy cập nhưng tiết kiệm 50%+ trên 90% còn lại.

### Ba mẫu

**Pre-route** (bộ phân loại trước): Độ trễ ~5-10ms được thêm vào; nhanh nhất về tổng thể.

**Cascade** (giá rẻ trước, leo thang khi độ tin cậy thấp): ~1,2x độ trễ trung bình (chạy rẻ cộng với xác minh), ~2x khi leo thang. Sàn chất lượng tốt nhất.

**Lộ trình tổng hợp** (chạy song song với giá rẻ và biên giới cho một mẫu, chọn model thưởng): chất lượng cao nhất, chi phí cao nhất; Chỉ sử dụng cho các A/B. quan trọng

### Triển khai

AI gateways (Giai đoạn 17 · 19) lộ tuyến đường. LiteLLM có `router` config với dự phòng và định tuyến chi phí. Portkey có bảo vệ + định tuyến. Kong AI Gateway có định tuyến dựa trên plugin. Thị trường model của OpenRouter tiết lộ một API đề xuất.

Mã nguồn mở: RouteLLM (LMSYS), Not Diamond (thương mại) Prompt Mule.

### Đường cong giá năm 2026

| Model class | Cuối năm 2022 | 2026 | Thay đổi |
|-------------|-----------|------|--------|
| Chất lượng GPT-4 cấp | ~$20/M | ~$0.40/M | Rẻ hơn 50 lần |
| Biên giới (GPT-5, Claude 4) | — | ~$3-10/M | Bậc mới |

Hầu hết các cải tiến là phục vụ hiệu quả - bài học cốt lõi trong Giai đoạn 17 · 04-09 biến thành sự sụt giảm chi phí từ phía nhà cung cấp. Định tuyến cho phép bạn nắm bắt những lợi ích đó ở lớp ứng dụng thay vì đợi tất cả người dùng của bạn di chuyển sang cấp giá rẻ.

### Trôi dạt là rủi ro thực sự

Tuyến đường của bạn gửi 40% đến model giá rẻ. Trong sáu tháng, việc phân phối nhiệm vụ thay đổi (người dùng trở nên phức tạp hơn, đặt câu hỏi dài hơn). Bộ định tuyến không nhận thấy vì bộ phân loại của nó đã được huấn luyện trên dữ liệu Q1. Chất lượng giảm xuống một cách âm thầm. Không ai phàn nàn đủ lớn. Bạn phát hiện ra ở một đối thủ cạnh tranh benchmark bạn đã thua.

Cổng các tuyến đường theo các chỉ số chất lượng trực tuyến:

- Người dùng giơ ngón tay cái / ngón tay cái xuống cho mỗi tuyến đường.
- Đánh giá LLM tự động trên một mẫu được giữ lại (5%) trên mỗi tuyến đường.
- Tỷ lệ leo thang: nếu dòng thác đang tăng >30%, model giá rẻ đang bị định tuyến quá mức.
- Tỷ lệ từ chối trên mỗi tuyến đường.

### Những con số bạn nên nhớ

- Tiết kiệm định tuyến năm 2026 ở chất lượng ISO: 20-60% nghiên cứu điển hình.
- LLM giảm giá 2022-2026: ~10 lần mỗi năm.
- Cấp GPT-4 2022 so với 2026: ~$20/M → ~$ 0.40/M.
- Tác động đến độ trễ theo tầng: trung bình ~1,2x, tăng ~2x (~10% lưu lượng truy cập).

## Ứng dụng

`code/main.py` mô phỏng tuyến trước, xếp tầng và tập hợp trên một khối lượng công việc hỗn hợp. Báo cáo chi phí hỗn hợp, loss chất lượng và tỷ lệ leo thang.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-router-plan.md`. Với khối lượng công việc và ngân sách chất lượng, chọn một mẫu định tuyến và tín hiệu.

## Bài tập

1. Chạy `code/main.py`. Cascade đánh bại trước đường bay ở tầng accuracy nào?
2. Cơ sở người dùng của bạn là 30% doanh nghiệp (truy vấn phức tạp), 70% bậc miễn phí (đơn giản). Thiết kế phân chia định tuyến. Số liệu trực tuyến nào kiểm soát nó?
3. Một tuyến đường giảm chất lượng 2% nhưng tiết kiệm 40%. Đó có phải là một ship? Phụ thuộc vào sản phẩm - tranh luận cả hai.
4. Thực hiện kiểm tra độ tin cậy bằng logprobs từ OpenAI / Anthropic APIs. Ngưỡng bạn bắt đầu là gì?
5. Trong sáu tháng, tỷ lệ leo thang tăng từ 8% lên 22%. Chẩn đoán ba nguyên nhân và cách khắc phục cho từng nguyên nhân.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Định tuyến Model | "Nhà môi giới chi phí" | Lựa chọn động model cho mỗi yêu cầu |
| Model thác | "leo thang giá rẻ đầu tiên" | Chạy rẻ, rơi xuống biên giới vì sự tự tin thấp |
| Trước đường bay | "Phân loại trước" | Bộ phân loại trước; Không chạy lại |
| Lộ trình hòa tấu | "chọn song song" | Chạy nhiều lựa chọn model phần thưởng tốt nhất |
| Tỷ lệ leo thang | "% được định tuyến lên" | Phần yêu cầu xếp tầng đã leo thang |
| Tuyến đường LLM | "Bộ định tuyến LMSYS" | Thư viện bộ định tuyến OSS |
| Không phải kim cương | "Bộ định tuyến thương mại" | Sản phẩm định tuyến model SaaS |
| Trôi dạt | "rẻ" | Sự thay đổi phân phối mà bộ định tuyến không nhận thấy |
| Cổng chất lượng trực tuyến | "Kiểm tra trực tiếp" | Đánh giá LLM tự động sampling lưu lượng truy cập trực tiếp |

## Đọc thêm

- [AbhyashSuchi — Model Routing LLM 2026 Best Practices](https://abhyashsuchi.in/model-routing-llm-2026-best-practices/)
- [Lukas Brunner — Rise of Inference Optimization 2026](https://dev.to/lukas_brunner/the-rise-of-inference-optimization-the-real-llm-infra-trend-shaping-2026-4e4o)
- [RouteLLM paper / code](https://github.com/lm-sys/RouteLLM)
- [Not Diamond — model routing](https://www.notdiamond.ai/)
- [OpenRouter](https://openrouter.ai/) - nhiều model gateway với primitives định tuyến.
