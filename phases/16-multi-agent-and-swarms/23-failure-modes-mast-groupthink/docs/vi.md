# Chế độ thất bại - MAST, Tư duy nhóm, Độc canh cấy, Lỗi xếp tầng

> Phân loại tham chiếu cho năm 2026 là **MAST** (Cemri và cộng sự, NeurIPS 2025, arXiv:2503.13657), bắt nguồn từ 1642 traces thực thi trên 7 MAS mã nguồn mở hiện đại cho thấy tỷ lệ thất bại **41–86,7%**. Ba loại gốc: **Vấn đề đặc tả **(41,77%) — vai trò mơ hồ, định nghĩa nhiệm vụ không rõ ràng; **Lỗi phối hợp** (36,94%) — sự cố giao tiếp, không đồng bộ trạng thái; **Lỗ hổng xác minh** (21,30%) — thiếu xác thực, không có kiểm tra chất lượng. Họ **Tư duy nhóm** (arXiv: 2508.05687) bổ sung: sự sụp đổ độc canh (cùng cơ sở model → các thất bại tương quan), sự phù hợp bias (agents củng cố lỗi của nhau), lý thuyết suy sụt giảm, động lực hỗn hợp, thất bại về độ tin cậy theo tầng. Ví dụ xếp tầng: các cơn bão thử lại trong đó thanh toán không thành công triggers thử lại đơn hàng, điều này trigger thử lại hàng tồn kho, làm quá tải dịch vụ hàng tồn kho (tải gấp 10 lần trong vài giây - cần bộ ngắt mạch). Ngộ độc trí nhớ: ảo giác của một agent đi vào ký ức chung, xuôi dòng agents coi nó là sự thật; accuracy phân rã dần dần, khiến chẩn đoán nguyên nhân gốc rễ trở nên đau đớn. **STRATUS** (NeurIPS 2025) báo cáo cải thiện giảm thiểu thành công gấp 1,5 lần thông qua agents phát hiện / chẩn đoán / xác nhận chuyên biệt. Bài học này coi các chế độ hỏng hóc là mục tiêu kỹ thuật class đầu tiên.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 16 · 13 (Bộ nhớ chung), Giai đoạn 16 · 14 (Đồng thuận và BFT), Giai đoạn 16 · 15 (Cấu trúc liên kết bỏ phiếu và tranh luận)
**Thời lượng:** ~75 phút

## Vấn đề

Hệ thống đa agent thất bại 41-86,7% thời gian trên các tác vụ thực tế (Cemri et al. 2025 đã đo lường điều này trên 7 MAS mã nguồn mở). Điều đó không thể gỡ lỗi bằng cách "chỉ cần thêm nhiều agents". Những thất bại có nguyên nhân cấu trúc. Phân loại MAST cung cấp cho bạn các danh mục. Bài học này ánh xạ từng danh mục với một mô hình phát hiện, chẩn đoán và giảm thiểu cụ thể để các con số không còn trông tùy tiện.

Thực tiễn production năm 2026 là coi các chế độ lỗi là đầu vào thiết kế. Kiến trúc của bạn không "đủ tốt" cho đến khi bạn có thể trỏ đến từng danh mục MAST và đặt tên cho biện pháp giảm thiểu mà bạn đã triển khai.

## Khái niệm

### Danh mục MAST

**Các vấn đề về thông số kỹ thuật (41,77% lỗi).** Nhiệm vụ của agent không được xác định đủ chặt chẽ. Ví dụ:

- Vai trò mơ hồ: hai agents đều nghĩ rằng họ là người phản biện.
- Nhiệm vụ không được chỉ định: "tóm tắt điều này" khi người dùng muốn một góc cụ thể.
- Tiêu chí thành công ngầm: agent không thể biết liệu nó có thành công hay không.

Giảm thiểu:
- Viết hợp đồng vai trò rõ ràng. Mỗi prompt của agent nêu rõ những gì nó làm *và những gì nó không làm*.
- Kiểm tra chấp nhận cho mỗi nhiệm vụ. Trước khi bắt đầu agent, hãy xác định "xong trông giống như X".
- Kiểm tra thông số kỹ thuật trước chuyến bay: một agent riêng rẽ xem xét định nghĩa nhiệm vụ trước khi gửi đi.

**Thất bại phối hợp (36,94%).** Sự cố giao tiếp hoặc trạng thái.

Ví dụ:
- Hai agents cập nhật trạng thái chia sẻ mà không cần đồng bộ hóa.
- Tin nhắn bị mất giữa agents (lỗi hàng đợi, timeout).
- Trôi dạt trạng thái: agent A nghĩ rằng nhiệm vụ đã hoàn thành; agent B vẫn đang thực hiện.

Giảm thiểu:
- Trạng thái được chia sẻ theo phiên bản với tính đồng thời lạc quan.
- Xác nhận rõ ràng cho các thư quan trọng (thử lại cho đến khi được xác nhận).
- checkpoints đồng bộ trạng thái định kỳ; phát hiện trôi sớm.

**Khoảng trống xác minh (21,30%).** Không kiểm tra độc lập về đầu ra.

Ví dụ:
- Một agent tuyên bố thành công; không ai xác minh.
- Chuỗi agents mỗi người tin tưởng vào đầu ra của prior.
- Kiểm tra phạm vi bị thiếu trên hành vi sáng tạo mới nổi.

Giảm thiểu:
- Người xác minh độc lập agent (Bài 13). Truy cập nguồn độc lập, chỉ đọc.
- Hợp đồng bàn giao rõ ràng: "Đầu ra của A phải vượt qua trình kiểm tra C trước khi B bắt đầu."
- Ghi nhật ký kết quả để phân tích sau khi thực hiện.

### Gia đình tư duy nhóm (arXiv: 2508.05687)

Năm lỗi liên quan khi agents đồng nhất hoặc bắt chước lẫn nhau:

**Sự sụp đổ của độc canh.** Cùng một dữ liệu cơ sở model hoặc training → các lỗi tương quan. Khi ba agents chia sẻ một LLM, họ chia sẻ ảo giác của nó.

**Sự phù hợp bias.** Agents điều chỉnh theo hướng đồng nghiệp lớn nhất hoặc tự tin nhất, ngay cả khi sai.

**Thiếu ToM.** Agents không model được niềm tin của nhau; sự phối hợp sụp đổ (Bài 18).

**Động lực hỗn hợp.** Agents với các động cơ liên kết một phần trôi dạt về phía trung gian thỏa hiệp, điều này không làm hài lòng ai.

**Lỗi độ tin cậy theo tầng.** Mẫu lỗi của một thành phần triggers các mẫu lỗi trong các thành phần phụ thuộc.

### Ví dụ xếp tầng - cơn bão thử lại

Một mô hình sự cố cổ điển năm 2026:

```
payment service fails 10% of requests
   ↓
order agent retries payment (exponential backoff but naive)
   ↓
each retry is a new order-inventory check
   ↓
inventory service sees 2x normal load
   ↓
inventory service starts timing out
   ↓
every order retries inventory check
   ↓
inventory service sees 10x normal load
   ↓
cluster goes down
```

Cách khắc phục là cổ điển: **cầu dao**. Khi tỷ lệ lỗi xuôi dòng vượt quá ngưỡng, đoản mạch với kết quả được lưu trong bộ nhớ cache hoặc mặc định. Cộng với ngân sách thử lại giới hạn cho mỗi yêu cầu.

Bộ ngắt mạch là một trong số ít các biện pháp giảm thiểu lỗi đa agent mà bạn mượn trực tiếp từ các hệ thống phân tán mà không cần sửa đổi.

### Ngộ độc trí nhớ (xem lại)

Từ Bài 13: ảo giác của một agent trở thành sự thật ký ức được chia sẻ; hạ lưu agents lý do về sự thật bị đầu độc. Theo thuật ngữ MAST, đây là khoảng trống xác minh ở lớp bộ nhớ chia sẻ.

Phân rã accuracy dần dần là triệu chứng. Bạn không gặp sự cố; bạn bị trôi chậm khó gây ra gốc rễ.

Giảm thiểu: nhật ký chỉ bổ sung, xuất xứ, trình xác minh không thể ghi. Đã được đề cập trong Bài 13.

### STRATUS — agents chuyên dụng để phát hiện lỗi

STRATUS (NeurIPS 2025) báo cáo cải thiện giảm thiểu thành công gấp 1,5 lần khi bạn triển khai:

- **Phát hiện agent.** Theo dõi các mẫu triệu chứng (bất đồng cao, thử lại tăng đột biến accuracy trôi dạt).
- **Chẩn đoán agent.** Với các triệu chứng, suy ra nguyên nhân gốc rễ có thể là do phân loại MAST.
- **Xác nhận agent.** Sau khi áp dụng biện pháp giảm thiểu, hãy kiểm tra xem các triệu chứng đã rõ chưa.

Đây là ứng phó sự cố kiểu SRE, được áp dụng cho các hệ thống agent. Ba vai trò đều có thể được LLM agents với prompts chuyên biệt.

### Kiểm tra chế độ lỗi

Thực tiễn tốt nhất năm 2026 là kiểm tra chế độ thất bại hàng năm (hoặc theo bản phát hành chuyên ngành):

1. **Trace mẫu.** Thu thập ~1000 traces khớp lệnh thực.
2. **Phân loại.** Đối với mỗi thất bại của trace, hãy ánh xạ đến các danh mục MAST + Tư duy nhóm.
3. **Tính toán tỷ lệ lỗi theo danh mục.** Danh mục nào thống trị hệ thống của bạn?
4. **Giảm thiểu xếp hạng.** Bản sửa lỗi nào sẽ loại bỏ nhiều lỗi nhất?
5. **Chọn 2-3 biện pháp giảm thiểu.** Thực hiện; kiểm toán lại quý sau.

Kỷ luật quan trọng hơn những lựa chọn cụ thể. Nếu không có kiểm toán, các lỗi sẽ hòa vào nhiễu và không bao giờ được giải quyết một cách có hệ thống.

### Khi hệ thống bị lỗi âm thầm

Loại lỗi nguy hiểm nhất là lỗi chính xác im lặng. Một hệ thống bị lỗi lớn (sự cố, ngoại lệ, cảnh báo) có thể được giám sát. Một hệ thống tạo ra đầu ra hợp lý nhưng sai không thể được phát hiện bằng nhật ký ngoại lệ. Đây là lý do tại sao lỗ hổng xác minh là danh mục đắt nhất cho mỗi lần thất bại mặc dù chúng chỉ là 21,30% theo số lượng.

Đầu tư vào:
- Đánh giá của con người dựa trên mẫu.
- Kiểm tra hồi quy dataset vàng.
- Kiểm tra chéo agent chéo trên các đầu ra quan trọng.

### Thất bại vs thất bại chậm

Một số thất bại là ngay lập tức; một số chậm. Lỗi ngay lập tức (timeout, không khớp schema, lỗi xác thực) rất rẻ để phát hiện. Thất bại chậm (nhiễm độc trí nhớ, trôi dạt độc cấu, mơ hồ vai trò) rất tốn kém để phát hiện và ngăn chặn.

Động thái kỹ thuật năm 2026: thiết bị chạy chậm để bạn có thể phát hiện trôi trước khi nó trở thành lỗi có thể nhìn thấy. Tỷ lệ thỏa thuận, tỷ lệ thử lại, phân phối độ dài đầu ra và khoảng cách chỉnh sửa giữa các phiên bản agent liên tiếp đều là những proxy hữu ích.

## Tự xây dựng

`code/main.py` thực hiện:

- `FailureTaxonomy` - phân loại các sự cố mô phỏng thành các danh mục MAST + Tư duy nhóm.
- `CircuitBreaker` - hoa văn cổ điển; mở ra khi tỷ lệ lỗi vượt quá ngưỡng.
- `RetryStormSimulator` - cho thấy sự thất bại theo tầng; Bật / tắt bộ ngắt mạch.
- `DetectionAgent` - trình khớp triệu chứng kiểu STRATUS theo kịch bản.

Chạy:

```
python3 code/main.py
```

Sản lượng dự kiến:
- Thử lại Storm mà không có bộ ngắt mạch: Lỗi hàng tồn kho bùng nổ (mô phỏng).
- với bộ ngắt mạch: nắp ở ngưỡng; phản hồi ở chế độ suy giảm được phục vụ.
- phát hiện agent gắn cờ mẫu và đặt tên cho danh mục MAST.

## Ứng dụng

`outputs/skill-mast-auditor.md` chạy kiểm tra chế độ lỗi kiểu MAST trên hệ thống đa agent. Traces → phân loại → xếp hạng giảm thiểu.

## Sản phẩm bàn giao

Kỷ luật chế độ thất bại trong production:

- **Kiểm toán MAST mỗi quý.** Không phải hàng năm. Danh mục thay đổi khi hệ thống của bạn phát triển.
- **Cầu dao ở khắp mọi nơi.** Mỗi cuộc gọi đi đến bất kỳ dịch vụ phụ thuộc nào. Ngưỡng mở mặc định ở tỷ lệ lỗi 5-10%.
- **Golden datasets.** Nhỏ, chất lượng cao, được kiểm tra thủ công. Kiểm tra hồi quy chống lại chúng hàng tuần.
- **Bộ ba STRATUS.** Phát hiện + Chẩn đoán + Xác nhận agents giám sát production. Chỉ bắt đầu với agent phát hiện; Thêm chẩn đoán khi các triệu chứng ồn ào.
- **Ngân sách thất bại.** SLO rõ ràng cho tỷ lệ thất bại theo danh mục. Vượt quá ngân sách triggers một cuộc trò chuyện shipping dừng lại.

## Bài tập

1. Chạy `code/main.py`. Xác nhận bộ ngắt mạch giới hạn cơn bão thử lại. Thay đổi ngưỡng thất bại và quan sát sự đánh đổi.
2. Triển khai **proxy lỗi chậm**: tỷ lệ thỏa thuận trên 3 agents song song. Khi nó giảm mạnh, trigger cảnh báo. Mô phỏng trôi dạt độc canh bằng cách dần dần tương quan agent đầu ra.
3. Đọc Cemri et al. (arXiv: 2503.13657). Chọn một trong 7 hệ thống MAS của họ và lập bản đồ 3 danh mục lỗi hàng đầu của nó. Làm thế nào để những điều này so sánh với những gì MAST dự đoán?
4. Đọc bài báo Tư duy nhóm (arXiv: 2508.05687). Xác định mô hình nào trong số năm mô hình khó phát hiện nhất trong production. Đề xuất một chỉ số proxy.
5. Thiết kế bộ ba phát hiện-chẩn đoán-xác nhận kiểu STRATUS cho một hệ thống đa agent cụ thể mà bạn biết. Phát hiện theo dõi những triệu chứng nào? Chẩn đoán khuyến nghị những biện pháp giảm thiểu nào? Xác thực xác nhận chúng hoạt động như thế nào?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| MAST | "Phân loại năm 2026" | Cemri 2025; 3 loại gốc + 14 loại lỗi phụ. |
| Vấn đề đặc điểm kỹ thuật | "Vai trò mơ hồ" | Nhiệm vụ hoặc vai trò không được xác định; agents không biết phải làm gì. |
| Thất bại phối hợp | "Trạng thái trôi dạt" | Sự cố giao tiếp hoặc đồng bộ hóa giữa agents. |
| Khoảng cách xác minh | "Không ai kiểm tra" | Đầu ra được chấp nhận mà không cần xác thực độc lập. |
| Gia đình tư duy nhóm | "Thất bại về tính đồng nhất" | Độc canh, phù hợp, thiếu ToM, động cơ hỗn hợp, xếp tầng. |
| Sự sụp đổ của độc canh | "Cùng một model, cùng ảo giác" | Lỗi tương quan từ dữ liệu model hoặc dữ liệu training cơ sở được chia sẻ. |
| Thử lại cơn bão | "Khuếch đại lỗi xếp tầng" | Một thất bại triggers thử lại để khuếch đại tải xuôi dòng. |
| Bộ ngắt mạch | "Thất bại nhanh về tỷ lệ lỗi" | Mở khi tỷ lệ lỗi vượt quá ngưỡng; đoản mạch với mặc định. |
| TẦNG | "Bộ ba ứng phó sự cố" | Phát hiện + chẩn đoán + xác nhận agents. Giảm thiểu thành công gấp 1,5 lần. |
| Ngộ độc trí nhớ | "Ảo giác lan truyền" | Sự thật về ký ức được chia sẻ bị ô nhiễm; hạ lưu agents lý do về chất độc. |

## Đọc thêm

- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) - Phân loại MAT, NeurIPS 2025
- [Groupthink failures in multi-agent LLMs](https://arxiv.org/abs/2508.05687) - độc canh cấy, sự phù hợp và phân loại năm họ
- [STRATUS — specialized agents for MAS incident response](https://neurips.cc/) - Mục nhập thủ tục NeurIPS 2025 (phát hiện + chẩn đoán + xác nhận)
- [Release It! — stability patterns (Nygard)](https://pragprog.com/titles/mnee2/release-it-second-edition/) - tham chiếu bộ ngắt mạch chuẩn
- [Anthropic — Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) - production ghi chú về chế độ lỗi
