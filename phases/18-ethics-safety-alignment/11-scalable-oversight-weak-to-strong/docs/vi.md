# Giám sát có thể mở rộng và khái quát hóa từ yếu đến mạnh

> Burns et al. (OpenAI Superalignment, "Khái quát hóa yếu đến mạnh", 2023) đã đề xuất một proxy cho vấn đề siêu liên kết: fine-tune một model mạnh bằng cách sử dụng các nhãn được tạo ra bởi một model yếu hơn. Nếu model mạnh khái quát hóa chính xác từ sự giám sát yếu kém không hoàn hảo, các phương pháp alignment quy mô con người hiện tại có thể mở rộng đến các hệ thống siêu phàm. Giám sát có thể mở rộng và W2SG bổ sung cho nhau. Giám sát có thể mở rộng (tranh luận, mô hình phần thưởng đệ quy, phân tách nhiệm vụ) làm tăng khả năng hiệu quả của người giám sát để có thể theo kịp model dưới sự giám sát. W2SG đảm bảo model mạnh mẽ khái quát hóa chính xác khỏi bất kỳ sự giám sát không hoàn hảo nào mà người giám sát cung cấp. Tranh luận giúp W2SG (arXiv:2501.13124, tháng 1 năm 2025) kết hợp chúng.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, W2SG gap simulator)
**Kiến thức tiên quyết:** Giai đoạn 18 · 01 (theo hướng dẫn), Giai đoạn 18 · 10 (Kiểm soát AI), Giai đoạn 09 (RL móng)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Xác định giám sát có thể mở rộng và khái quát hóa từ yếu đến mạnh và giải thích cách chúng bổ sung cho nhau.
- Mô tả Burns et al. Thiết lập thử nghiệm năm 2023: fine-tune GPT-4 sử dụng nhãn từ GPT-2.
- Giải thích chỉ số thu hồi khoảng cách hiệu suất (PGR) và những gì nó đo lường.
- Nêu ba cơ chế giám sát có thể mở rộng chính (tranh luận, mô hình phần thưởng đệ quy, phân tách nhiệm vụ) và một điểm mạnh của mỗi cơ chế.

## Vấn đề

Mọi kỹ thuật alignment cho đến nay trong Giai đoạn 18 đều giả định rằng người giám sát có thể đánh giá hành vi của model. Khi model là siêu nhân, người giám sát là mắt xích yếu. Câu hỏi siêu liên kết: một người giám sát yếu hơn có thể tạo ra một model mạnh hơn, liên kết hơn một cách đáng tin cậy không?

Burns và cộng sự giảm điều này thành một thiết lập thực nghiệm hoạt động: giám sát mạnh với yếu, đo lường bao nhiêu khả năng của model mạnh tồn tại sau sự giám sát yếu. Đây không phải là một giải pháp cho sự siêu liên kết - nó là một proxy cho phép chúng ta đo lường sự tiến bộ.

## Khái niệm

### W2SG: thiết lập Burns et al.

- model yếu: GPT-2 class.
- model mạnh mẽ: GPT-4 class.
- Mục tiêu: trần GPT-4 mạnh mẽ trong nhiệm vụ.

Thủ tục:
1. Nhận dự đoán zero-shot yếu của model về một nhiệm vụ.
2. Fine-tune model mạnh trên dữ liệu được gắn nhãn yếu.
3. Đo accuracy của fine-tuned model mạnh.

Đường cơ sở: accuracy của chính model yếu, trần giám sát của model mạnh trên nhãn vàng.

Chỉ số khoảng cách: Khoảng cách hiệu suất được phục hồi (PGR) = (fine-tuned - yếu) / (trần - yếu). PGR 1.0 có nghĩa là giám sát yếu sẽ thu hẹp hoàn toàn khoảng cách; PGR bằng 0 có nghĩa là giám sát yếu không giúp ích được gì.

### Burns et al. phát hiện thực nghiệm

Trong các nhiệm vụ NLP, giải đố cờ vua và mô hình hóa phần thưởng, PGR luôn tích cực: mạnh model khái quát hóa vượt ra ngoài những sai lầm của người giám sát yếu. PGR theo nhiệm vụ cụ thể thay đổi từ ~20% đến ~80%. Người model mạnh "hiểu" nhiệm vụ dự định từ các nhãn mác không hoàn hảo, điền vào priors được huấn luyện trước của nó.

Hạn chế Burns và cộng sự lưu ý:
- Khoảng cách yếu đến mạnh ở đây là khả năng chứ không phải alignment. Các model yếu được căn chỉnh; Một model siêu phàm thực sự có thể được sắp xếp bởi con người nhưng với khoảng cách năng lực lớn hơn nhiều.
- Sự khái quát hóa của model mạnh có thể là về priors nhiệm vụ hơn là khôi phục ý định sự thật cơ bản.
- Khơi gợi kiến thức tiềm ẩn của model mạnh là vấn đề thực sự; PGR đo lường một hoạt động cụ thể.

### Giám sát có thể mở rộng: ba cơ chế

- **Tranh luận (Irving et al. 2018).** Hai trường hợp U tranh luận về các mặt đối lập; Thẩm phán yếu đuối quyết định. Giả định: tìm ra một câu trả lời đúng thuyết phục dễ dàng hơn là tìm một câu trả lời sai thuyết phục. Công trình gần đây 2024-2025 (Khan và cộng sự, Michael và cộng sự) cho thấy tranh luận đôi khi hữu ích và đôi khi không, và liệu nó có giúp ích hay không phụ thuộc vào cấu trúc nhiệm vụ.
- **Mô hình phần thưởng đệ quy (Leike et al. 2018).** U giúp con người huấn luyện model phần thưởng cho U+1. Khả năng hiệu quả của giám thị tăng lên cùng với model.
- **Phân tách nhiệm vụ (Christiano, Shlegeris, Amodei 2018).** Phân tách một nhiệm vụ khó thành các nhiệm vụ con mà con người có thể kiểm tra, đệ quy. Giả định khả năng phân hủy.

Mỗi cơ chế giả định một cái gì đó về cấu trúc của nhiệm vụ hoặc alignment của các thành phần trung gian.

### Tại sao giám sát có thể mở rộng và W2SG bổ sung cho nhau

Giám sát có thể mở rộng làm tăng chất lượng tín hiệu hiệu quả của người giám sát.
W2SG thu hẹp khoảng cách với bất kỳ tín hiệu không hoàn hảo nào mà giám sát viên có thể cung cấp.

Lang et al. - Tranh luận giúp khái quát hóa yếu đến mạnh (arXiv: 2501.13124) kết hợp chúng: một giao thức tranh luận cung cấp các nhãn yếu tốt hơn và model mạnh được huấn luyện trên các nhãn đó. Lợi nhuận PGR được báo cáo trên các nhiệm vụ NLP.

### Bộ phim tổ chức

Nhóm Superalignment của OpenAI giải thể vào tháng 5 năm 2024 sau khi Jan Leike rời đi Anthropic. Chương trình nghị sự (giám sát có thể mở rộng, W2SG, nghiên cứu alignment tự động) tiếp tục tại Anthropic và tại các phòng thí nghiệm học thuật - MATS (Bài 28), Redwood (Bài 10), Apollo (Bài 8), METR (Bài 28). Cơ cấu tổ chức thay đổi; Các câu hỏi nghiên cứu thì không.

### Điều này phù hợp với Giai đoạn 18 ở đâu

Bài học 6-10 mô tả mối đe dọa và mô hình phòng thủ theo giả định U không đáng tin cậy. Bài 11 là mô hình tấn công: làm cho người giám sát đủ mạnh để xác minh alignment của U. Bài 12-16 sau đó chuyển sang công cụ thực tế của đánh giá đối thủ.

## Ứng dụng

`code/main.py` mô phỏng fine-tune W2SG trên một tác vụ tổng hợp. Trình ghi nhãn yếu có 70% accuracy với lỗi cấu trúc; Strong model có mức trần 95% trên nhãn vàng. Bạn fine-tune model mạnh trên nhãn yếu, đo lường PGR và so sánh với mạnh trên vàng và chỉ yếu.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-w2sg-pgr.md`. Với mô tả thiết lập giám sát, nó xác định người giám sát yếu, model mạnh, chất lượng giám sát và tính toán (hoặc yêu cầu) PGR. Nó đánh dấu liệu tuyên bố là "yếu có thể giám sát mạnh" hay "yếu + cơ chế giám sát có thể giám sát mạnh".

## Bài tập

1. Chạy `code/main.py`. Báo cáo PGR cho weak_accuracy = 0,60, 0,70, 0,80. Giải thích hình dạng của đường cong PGR.

2. Sửa đổi trình gắn nhãn yếu để có lỗi cấu trúc (ví dụ: luôn sai trên một class đầu vào cụ thể). PGR tăng, giảm hay giữ nguyên? Giải thích.

3. Đọc Burns et al. 2023 Phần 4.3 (NLP nhiệm vụ). Tái tạo trực giác "loss phụ trợ tự tin": khi model mạnh tự tin hơn nhãn yếu, ai thắng?

4. Thiết kế một giao thức giám sát có thể mở rộng kết hợp tranh luận và phân tách nhiệm vụ cho một nhiệm vụ kỹ thuật phần mềm. Đặt tên cho một chế độ lỗi của mỗi thành phần và giải thích cách tổ hợp giải quyết hoặc không giải quyết từng thành phần.

5. Nói rõ điều gì sẽ làm sai lệch tuyên bố "khái quát hóa yếu đến mạnh là một con đường khả thi để siêu liên kết". Hãy cụ thể về chữ ký thực nghiệm mà bạn cần xem.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| Giám sát có thể mở rộng | "Làm cho giám thị mạnh mẽ hơn" | Các cơ chế làm tăng khả năng đánh giá một model có năng lực hơn của giám thị |
| W2SG | "Yếu giám sát mạnh" | Fine-tuning model mạnh mẽ trên các nhãn yếu và đo lường khả năng phục hồi |
| PGR | "Khoảng cách hiệu suất đã được phục hồi" | (fine-tuned - yếu) / (trần - yếu); 1.0 = đóng hoàn toàn, 0 = không trợ giúp |
| Tranh luận | "hai trường hợp U tranh cãi" | Cơ chế giám sát có thể mở rộng trong đó một trọng tài yếu chọn giữa hai hậu vệ U |
| RRM | "Mô hình phần thưởng đệ quy" | U giúp huấn luyện model phần thưởng cho U+1; khả năng giám sát theo dõi U |
| Phân tách nhiệm vụ | "nhiệm vụ phụ kiểm tra con người" | Chia một nhiệm vụ khó thành các nhiệm vụ con mà con người có thể xác minh, đệ quy |
| Siêu căn chỉnh | "Sắp xếp AI siêu phàm" | Chương trình nghiên cứu liên quan đến việc sắp xếp models con người không thể đánh giá trực tiếp |

## Đọc thêm

- [Burns et al. — Weak-to-Strong Generalization (OpenAI 2023)](https://openai.com/index/weak-to-strong-generalization/) — bài báo W2SG
- [Irving, Christiano, Amodei — AI safety via debate (arXiv:1805.00899)](https://arxiv.org/abs/1805.00899) — cơ chế tranh luận
- [Leike et al. — Scalable agent alignment via reward modeling (arXiv:1811.07871)](https://arxiv.org/abs/1811.07871) — mô hình phần thưởng đệ quy
- [Khan et al. — Debating with More Persuasive LLMs Leads to More Truthful Answers (arXiv:2402.06782)](https://arxiv.org/abs/2402.06782) - Nghiên cứu thực nghiệm năm 2024 về tranh luận với những người tranh luận mạnh mẽ hơn
- [Lang et al. — Debate Helps Weak-to-Strong Generalization (arXiv:2501.13124)](https://arxiv.org/abs/2501.13124) - 2025 kết hợp tranh luận + W2SG
