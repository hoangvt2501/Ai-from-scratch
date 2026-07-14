# Xã hội tâm trí và tranh luận đa Agent

> Tiền đề năm 1986 của Minsky - trí thông minh là một xã hội của các chuyên gia - được khám phá lại mỗi thập kỷ. Vào năm 2023, Du và cộng sự đã biến nó thành một thuật toán cụ thể: nhiều trường hợp LLM đề xuất câu trả lời, đọc câu trả lời của nhau, phê bình và cập nhật. Qua N vòng, họ hội tụ trên một sự đồng thuận đánh bại CoT zero-shot và phản ánh về sáu nhiệm vụ lý luận và thực tế. Hai phát hiện quan trọng: cả **nhiều agents** và **nhiều vòng** đều đóng góp độc lập. Xã hội đánh bại một cuộc độc thoại một agent; Cuộc trao đổi nhiều vòng đánh bại one-shot bỏ phiếu.

**Loại:** Tìm hiểu + Xây dựng
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 16 · 04 (Primitive Model)
**Thời lượng:** ~60 phút

## Vấn đề

Tính nhất quán - lấy mẫu một model nhiều lần và lấy câu trả lời đa số - là cải tiến lý luận rẻ nhất mà bạn có thể sử dụng. Nó hoạt động, nhưng nó bão hòa nhanh chóng. Bạn có thể nhân đôi mẫu của mình và không thấy một bước nhảy có ý nghĩa khác.

Tranh luận phá vỡ sự bão hòa. Thay vì N mẫu độc lập từ một model, N agents đọc lý do của nhau và sửa đổi. Mối tương quan giữa các mẫu giảm (chúng không còn là i.i.d.) và điểm hội tụ thường chính xác khi bỏ phiếu i.i.d. là sai.

## Khái niệm

### Thuật toán Du et al. 2023

Từ arXiv:2305.14325 (ICML 2024):

1. Mỗi N agents tạo ra một câu trả lời ban đầu cho câu hỏi.
2. Đối với vòng r = 2..R: mỗi agent được hiển thị câu trả lời vòng r-1 của agents kia và được hỏi "xem xét những điều này, đưa ra câu trả lời cập nhật của bạn."
3. Sau các vòng R, đa số bỏ phiếu cho câu trả lời cuối cùng.

Bài kiểm tra trên MMLU, GSM8K, tiểu sử, TOÁN HỌC và benchmarks thực tế. Tranh luận liên tục đánh bại CoT và Tự phản ánh.

### Hai núm độc lập

Cắt bỏ từ cùng một bài báo:

- **Agent đếm một mình** (1 vòng, đa số phiếu bầu của N) đánh bại một agent trong hầu hết các nhiệm vụ, nhưng ổn định.
- **Chỉ đếm vòng** (1 agent nhìn thấy lý luận prior của chính nó) hầu như không giúp ích gì - điểm yếu đã biết của phản chiếu.
- **Cả hai cùng nhau **tạo ra những bước nhảy vọt lớn. Sự trao đổi nhiều vòng giữa nhiều agents thúc đẩy lợi nhuận.

### Tại sao nó hoạt động

Hai cơ chế:

1. **Tiếp xúc với sự bất đồng.** Khi một agent nhìn thấy chuỗi lý luận của một agent khác với một kết luận khác, nó phải biện minh hoặc cập nhật. Dù bằng cách nào, ngữ cảnh của vòng r + 1 phong phú hơn vòng r.
2. **Giảm lỗi tương quan.** Về tính nhất quán, tất cả các mẫu đều đến từ cùng một model, vì vậy các lỗi tương quan với nhau - bạn tính trung bình thành một câu trả lời sai một cách tự tin. Các models khác nhau hoặc các hạt khác nhau không tương quan. Các quan điểm tranh luận khác nhau không tương quan hơn nữa.

### Tranh luận không đồng nhất

A-HMAD và các phần theo dõi liên quan sử dụng *models cơ sở khác nhau* cho các agents khác nhau. Tranh luận Llama + Claude + GPT làm giảm sự sụp đổ của độc canh (Bài 26) vì các lỗi tương quan của một gia đình model không được chia sẻ bởi những gia đình khác.

Nhược điểm: một model yếu tham gia vào một cuộc tranh luận có thể kéo sự đồng thuận đến câu trả lời sai của nó (xem "Chúng ta có nên điên rồ không?", arXiv:2311.17371).

### NLSOM — phần mở rộng 129 agent

Zhuge et al. ("Mindstorms in Natural Language-Based Societies of Mind," arXiv: 2305.17066) đã mở rộng ý tưởng này lên các xã hội 129 thành viên. Kết quả: chuyên môn hóa và tự tổ chức xuất hiện theo quy mô và hệ thống vượt trội hơn agent đơn trong các nhiệm vụ như trả lời câu hỏi trực quan.

### Chế độ thất bại

- **Dòng thác Sycophance.** Tất cả agents tuân theo bất kỳ agent nào nghe có vẻ tự tin nhất. Cuộc tranh luận sụp đổ thành tiếng nói lớn nhất. Prompting cho các vai trò đối nghịch ("một agent phải tranh luận về quan điểm đối lập") sẽ giúp ích.
- **Chủ đề trôi dạt.** Các cuộc tranh luận trong nhiều vòng trôi dạt khỏi câu hỏi ban đầu. Giảm thiểu: tiêm lại câu hỏi mỗi vòng.
- **Tính toán nổ tung.** N agents × R vòng = N· R LLM gọi, mỗi cuộc gọi có một bối cảnh phát triển. Một cuộc tranh luận 5-agent, 5 vòng là 25 cuộc gọi trong bối cảnh đang phát triển. Chi phí cho mỗi câu hỏi có thể vượt quá 10× một cuộc gọi CoT.

## Tự xây dựng

`code/main.py` tổ chức một cuộc tranh luận 3 agent × 3 vòng về một câu hỏi toán học, trong đó mỗi agent bắt đầu với một câu trả lời khác nhau (có thể sai). Agents được viết kịch bản - mỗi "cập nhật" bằng cách tính trung bình câu trả lời của hàng xóm có trọng số bằng sự tự tin theo kịch bản. Sự hội tụ có thể nhìn thấy trong nhật ký từng vòng.

Bản demo cho thấy hai hiệu ứng chính:

- Một vòng trao đổi duy nhất agents tiến gần hơn đến câu trả lời đúng.
- Các vòng phụ qua vòng 2 cho thấy lợi nhuận giảm dần (phù hợp với cao nguyên của Du và cộng sự).

Chạy:

```
python3 code/main.py
```

## Ứng dụng

`outputs/skill-debate-configurator.md` cấu hình một cuộc tranh luận cho một nhiệm vụ mới: số agents, số vòng, tính không đồng nhất (cùng model và hỗn hợp), phân công vai trò (đối xứng so với một đối thủ). Nó cũng ước tính chi phí token trước khi bạn chạy.

## Sản phẩm bàn giao

Nếu bạn ship tranh luận:

- **Vòng giới hạn ở mức 3.** Du và cộng sự cho thấy 3 vòng thu được hầu hết lợi nhuận. Nhiều hơn là chi phí, không phải chất lượng.
- **Giới hạn agents ở mức 5.** Ngoài 5, bối cảnh cồng kềnh và chi phí chiếm ưu thế.
- **Không đồng nhất theo mặc định.** Ít nhất hai models cơ sở khác nhau trong nhóm.
- **Vị trí đối nghịch.** Một agent được nhắc nhở để không đồng ý. Phá vỡ sự đồng tính.
- **Ghi nhật ký mỗi vòng.** Các hệ thống tranh luận ẩn các vòng trung gian không thể được gỡ lỗi hoặc kiểm tra.

## Bài tập

1. Chạy `code/main.py`, sau đó đặt số vòng thành 5 và xem lợi nhuận giảm dần. Hội tụ bổ sung dừng lại ở vòng nào?
2. Thêm một agent thứ tư với vai trò đối nghịch: luôn không đồng ý với đa số hiện tại. Điều này có phá vỡ hoặc cải thiện sự hội tụ không?
3. Vẽ (in) điểm đồng ý mỗi vòng (phần agents trên câu trả lời đa số). Khi nào nó đạt 1.0 và điều đó có tương đương với "đúng" không?
4. Đọc Du và cộng sự. Phần 4 cắt bỏ. Sao chép kết quả "chỉ agents" so với "chỉ vòng" so với "cả hai" bằng cách sử dụng mã này.
5. Đọc "Chúng ta có nên điên rồ không?" (arXiv:2311.17371) và liệt kê hai biến thể tranh luận ngoài vòng tròn - ví dụ: do thẩm phán dẫn dắt, chuỗi tranh luận, đối thủ.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Xã hội tâm trí | "Ý tưởng của Minsky" | Trí thông minh với tư cách là chuyên gia tương tác; Khung hình năm 1986 hiện được thực hiện thông qua LLM tranh luận. |
| Tranh luận nhiều agent | "Agents tranh luận" | N agents đề xuất, phê bình lẫn nhau, sửa đổi qua các vòng R, bỏ phiếu đa số. |
| Đồng thuận | "Họ đồng ý" | Không phải sự thật nhận thức - chỉ là câu trả lời phân số trên đa số. Có thể tự tin sai. |
| Vòng | "Các bước trao đổi" | Một vòng = mỗi agent đọc các vòng khác và cập nhật một lần. |
| Tranh luận không đồng nhất | "Kết hợp model gia đình" | Sử dụng các models cơ sở khác nhau để phân tích lỗi. |
| Dòng thác Sycophancy | "Mọi người đều đồng ý với người ồn ào" | Tranh luận thất bại khi agents tuân theo agent tự tin nhất bất kể tính đúng đắn. |
| NLSOM | "Xã hội 129-agent" | Xã hội ngôn ngữ tự nhiên của tâm trí; Phiên bản thu nhỏ của Zhuge và cộng sự. |
| Lỗi tương quan | "Cùng một model, cùng một lỗi" | Tại sao tính nhất quán của bản thân bão hòa; tranh luận trên các quan điểm khác nhau không tương quan. |

## Đọc thêm

- [Du et al. — Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://arxiv.org/abs/2305.14325) — tài liệu tham khảo, ICML 2024
- [Zhuge et al. — Mindstorms in Natural Language-Based Societies of Mind](https://arxiv.org/abs/2305.17066) - 129-agent NLSOM
- [Should we be going MAD? A Look at Multi-Agent Debate Strategies for LLMs](https://arxiv.org/abs/2311.17371) - benchmarks biến thể tranh luận
- [Debate project page](https://composable-models.github.io/llm_debate/) — Mã, bản demo và chi tiết cắt bỏ của Du và cộng sự
