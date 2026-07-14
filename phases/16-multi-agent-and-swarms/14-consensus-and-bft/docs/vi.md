# Sự đồng thuận và khả năng chịu lỗi của Byzantine đối với Agents

> Hệ thống phân tán cổ điển BFT đáp ứng LLMs ngẫu nhiên. Trong giai đoạn 2025-2026, ba hướng nghiên cứu đã xuất hiện: **CP-WBFT** (arXiv:2511.10400) cân nhắc mỗi phiếu bầu bằng một cuộc thăm dò tin cậy; **DecentLLMs** (arXiv:2507.14928) không có người dẫn đầu với các đề xuất worker song song và tổng hợp trung vị hình học; **WBFT** (arXiv:2505.05103) kết hợp bỏ phiếu có trọng số với Phân cụm cấu trúc phân cấp để phân chia các nút Core và Edge. Kết quả thực nghiệm trung thực từ "AI Agents có thể đồng ý không?" (arXiv:2603.01213) là ngay cả thỏa thuận vô hướng ngày nay cũng rất mong manh - một agent lừa đảo duy nhất có thể thỏa hiệp một hỗn hợp Agents. BFT là cần thiết nhưng không đủ. Bài học này xây dựng một giao thức BFT tối thiểu, đưa vào ba cuộc tấn công cụ thể của agent (nói dối byzantine, sự phù hợp sycophantic, độc canh lỗi tương quan) và đo lường cách mỗi biến thể đồng thuận đối phó.

**Loại:** Tìm hiểu + Xây dựng
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 16 · 07 (Xã hội Tâm trí và Tranh luận), Giai đoạn 16 · 13 (Bộ nhớ chia sẻ)
**Thời lượng:** ~75 phút

## Vấn đề

Bạn có N LLM agents mỗi người tạo ra một câu trả lời. Họ không đồng ý. Đa số phiếu chọn sai vì hai agents tương quan (cùng model cơ sở, cùng dữ liệu training, cùng chế độ lỗi). Một agent thứ ba tình cờ sai theo một cách mới lạ - vì vậy đa số là đa số giả.

Bây giờ thêm một agent lừa dối: nó cố tình nói dối. Hoặc một agent giả dối: nó đồng ý với bất kỳ ai nói cuối cùng. Trong BFT cổ điển, giả định là các nút Byzantine là một phần nhỏ `f < n/3` và hoạt động tùy ý. Thực tế năm 2026 là các nút LLM là ngẫu nhiên ngay cả khi trung thực, tương quan giữa các models và bị ảnh hưởng bởi đầu ra của nhau. Bạn không thể đối xử với họ như những cử tri Bernoulli độc lập.

BFT cổ điển (PBFT, 1999) không sai - nó không đầy đủ. Nó xử lý việc lật bit tùy ý. Nó không xử lý "ba agents trung thực chia sẻ ảo giác vì họ chia sẻ dữ liệu training". Bài học này được xây dựng từ nền tảng của PBFT và các lớp trên ba bản chuyển thể 2025-2026.

## Khái niệm

### BFT cổ điển mang lại cho bạn những gì

Khả năng chịu lỗi Byzantine thực tế (Castro & Liskov, OSDI 1999) chịu được `f < n/3` các nút Byzantine. Giao thức có ba giai đoạn (chuẩn bị trước, chuẩn bị, commit) và hai giai đoạn primitives (thông điệp đã ký, chứng chỉ số đại biểu). Thỏa thuận về một giá trị duy nhất giữa `n >= 3f + 1` nút trung thực hoặc độc hại.

Các đảm bảo mạnh mẽ nhưng giả định:

1. **Đứt gãy độc lập.** Byzantine không phối hợp.
2. **Các nút trung thực thực sự trung thực.** Tính đúng đắn của đầu ra trung thực không phải là vấn đề; Nghị định thư chỉ điều chỉnh sự bất đồng.
3. **Câu hỏi có một câu trả lời cơ bản là sự thật.** Sự đồng thuận về một sự kiện sai lầm vẫn là sự đồng thuận.

LLM agents vi phạm cả ba. Hai agents chạy cùng một cơ sở model chia sẻ lỗi. Một LLM "trung thực" vẫn bị ảo giác. Và đối với những câu hỏi mơ hồ, "sự thật" là những gì agents quyết định - không có lời tiên tri bên ngoài.

### Ba cuộc tấn công dành riêng cho LLM

**Lời nói dối của Byzantine.** Một agent đưa ra một câu trả lời cố ý sai. BFT cổ điển xử lý điều này nếu `f < n/3`.

**Sự tuân thủ sycophantic.** Một agent đọc câu trả lời của người khác trước khi bỏ phiếu và liên kết với bất kỳ ai nói cuối cùng. Không độc hại, nhưng tương quan với giọng nói lớn nhất. BFT cổ điển không ngăn chặn điều này vì agent vượt qua mọi kiểm tra chữ ký.

**Độc canh lỗi tương quan.** Ba agents chia sẻ một model cơ sở. Họ ảo giác cùng một câu trả lời sai. Đa số là sai. BFT cổ điển không giúp ích gì vì cả ba đều "thành thật" đồng ý.

### Các phản ứng 2025-2026

**CP-WBFT** (arXiv:2511.10400) — BFT có trọng số thăm dò độ tin cậy. Mỗi cử tri đính kèm một cuộc thăm dò độ tin cậy vào câu trả lời của mình (xác suất tự báo cáo hoặc dự đoán của model hiệu chuẩn riêng biệt). Tự tin cân trọng lượng phiếu bầu. Báo cáo cải thiện BFT +85,71% trên biểu đồ hoàn chỉnh. Giảm thiểu cho: sự tuân thủ sycophantic (tuân thủ agents có xu hướng thấp tự tin vào vị trí tình nguyện của họ).

**DecentLLMs** (arXiv:2507.14928) — Không có người lãnh đạo. Worker agents đề xuất song song, người đánh giá agents đề xuất điểm số, câu trả lời cuối cùng là trung vị hình học của các vị trí được ghi điểm. Mạnh mẽ khi `f < n/2`. Giảm thiểu cho: Lời nói dối của Byzantine và các lỗi tương quan (trung vị hình học mạnh mẽ với các giá trị ngoại lệ và kéo về phía cụm dày đặc, không phải trung bình thiên vị model).

**WBFT** (arXiv:2505.05103) — BFT có trọng số với phân cụm cấu trúc phân cấp. Trọng số phiếu bầu được chỉ định theo chất lượng phản hồi cộng với điểm tin cậy học được từ lịch sử. Cụm agents thành Lõi và Cạnh; Core agents phải đạt được sự đồng thuận trước, Edge agents theo sau. Giảm thiểu cho: khả năng mở rộng (Sự đồng thuận cốt lõi nhỏ và nhanh) và một phần đối với độc canh (Cốt lõi có thể được chọn cho sự đa dạng).

### Thực nghiệm: "AI Agents có thể đồng ý không?" (arXiv: 2603.01213)

Bài báo đo lường thỏa thuận vô hướng (LLM agents đồng ý về một giá trị số duy nhất) trên nhiều models biên giới. Phát hiện này không thoải mái:

- Ngay cả khi không có đối thủ, LLM agents không đồng ý về các câu hỏi vô hướng với tỷ lệ trên 30% trên nhiều benchmarks.
- Một agent duy nhất áp dụng tính cách lừa đảo có thể kéo sự đồng thuận của Hỗn hợp Agents 40+ điểm phần trăm so với đường cơ sở trung thực.
- Tỷ lệ bất đồng tương quan với sự đa dạng model - các nhóm không đồng nhất bất đồng nhiều hơn so với các nhóm đồng nhất (tốt: lỗi không tương quan) nhưng cũng trôi chậm hơn (xấu: thời gian đồng ý lâu hơn).

Bài học rút ra: BFT cung cấp cho bạn máy móc để căn chỉnh đầu ra, nhưng nó không cho bạn biết liệu đầu ra được căn chỉnh có đúng hay không. Kết hợp với xác minh (Giai đoạn 16 · 08 chuyên môn hóa vai trò), đa dạng (Giai đoạn 16 · 15 biến thể tranh luận) và đánh giá agents (Giai đoạn 16 · 24 benchmarks).

### Giao thức cốt lõi, bị tước bỏ

Một vòng BFT tối thiểu cho LLM agents:

```
1. task arrives; each agent i produces answer a_i
2. each agent attaches confidence probe c_i in [0, 1]
3. aggregator collects (a_i, c_i) from all n agents
4. aggregator groups by semantic cluster (equivalent answers)
5. aggregator computes weight for each cluster C:
     w(C) = sum_{i in C} c_i
6. winner = cluster with max weight, if max > threshold * sum(c_i)
   else: retry or escalate
7. minority clusters logged with provenance for post-hoc audit
```

Bước phân cụm ngữ nghĩa là bước xoắn cụ thể của LLM. Hai câu trả lời "nghiên cứu báo cáo 4,2%" và "cải thiện 4,2%" là cùng một cụm. Một kiểm tra bình đẳng chuỗi ngây thơ sẽ bỏ lỡ điều này. Trong production, hãy sử dụng một embedding model rẻ tiền hoặc chuẩn hóa rõ ràng.

### Điều chỉnh ngưỡng

Người `threshold` parameter quyết định khi nào nên chấp nhận và khi nào nên thử lại. Quá thấp: bạn chấp nhận đa số yếu. Quá cao: bạn không bao giờ chấp nhận bất cứ điều gì. Phạm vi thực nghiệm: 0,5-0,67 đối với `n=5-7` agents, cao hơn đối với `n` nhỏ hơn. Dưới một ngưỡng, hãy leo thang lên một con người hoặc một nhóm agent khác.

### Khi sự đồng thuận không giúp ích được gì

- **Câu hỏi mơ hồ.** Nếu câu hỏi không có ground truth thì sự đồng thuận là một ý kiến. Gọi nó như vậy.
- **Câu hỏi ghép.** "Viết mã và giải thích nó" - hai câu trả lời. Bỏ phiếu cho mỗi người một cách độc lập.
- **Nhiều vòng đối nghịch.** Nếu agents có thể quan sát prior vòng và bắt chước (tranh luận Du 2023), họ bắt đầu đồng ý với nhau bất kể sự thật. Ràng buộc các vòng (thường là 2-3).

## Tự xây dựng

`code/main.py` thực hiện:

- `AgentVoter` — một policy có kịch bản với (câu trả lời, sự tự tin).
- `MajorityVote` — số nhiều cổ điển.
- `CPWBFT` - bỏ phiếu theo trọng số tin cậy với phân cụm ngữ nghĩa.
- `DecentLLMs` — tổng hợp trung bình hình học trên các đề xuất được tính điểm.
- `Scenario` — chạy mỗi trình tổng hợp theo ba kiểu tấn công.

Các kiểu tấn công được thực hiện:

1. `byzantine`: một agent nói dối với sự tự tin cao.
2. `sycophancy`: một agent sao chép câu trả lời đầu tiên mà nó nhìn thấy, với sự tự tin tương ứng.
3. `monoculture`: Ba agents chia sẻ câu trả lời sai (lỗi tương quan) với độ tin cậy vừa phải.

Chạy:

```
python3 code/main.py
```

Đầu ra dự kiến: bảng (tấn công, tổng hợp) -> câu trả lời cuối cùng, với câu trả lời đúng được đánh dấu. Đa số thất bại trong trường hợp độc canh. Trọng số niềm tin của CPWBFT giảm thiểu sự đồng cảm. Điểm trung bình hình học của DecentLLM hướng tới cụm trung thực khi độc canh ít hơn một nửa dân số.

## Ứng dụng

`outputs/skill-consensus-designer.md` thiết kế một giao thức đồng thuận cho một tập hợp nhiều agent: phương pháp phân cụm, trọng số, ngưỡng và policy leo thang cho các vòng ngưỡng phụ.

## Sản phẩm bàn giao

Trước khi shipping bất kỳ cơ chế đồng thuận nào:

- **Thử nghiệm tấn công với ít nhất ba mẫu** ở trên. Giao thức của bạn sẽ thất bại một cách có thể dự đoán được, không phải âm thầm.
- **Ghi nhật ký mọi cụm thiểu số** với nguồn gốc. Các cụm thiểu số là hệ thống cảnh báo sớm của bạn cho các lỗi tương quan.
- **Thực thi các vòng có giới hạn.** Không có "tiếp tục tranh luận cho đến khi đồng ý" - điều đó thưởng cho sự đồng ý.
- **Thỏa thuận tách biệt với tính đúng đắn.** Đầu ra đồng thuận được chuyển đến người xác minh; Trình xác minh độc lập với tập hợp.
- **Theo dõi tỷ lệ thỏa thuận.** Tăng mạnh có nghĩa là sự phù hợp bias; một cú ngã mạnh có nghĩa là model trôi dạt.

## Bài tập

1. Chạy `code/main.py`. Xác nhận đa số thất bại trong cuộc tấn công độc canh nhưng CPWBFT giảm thiểu một phần nó khi độ tin cậy độc canh dưới 0,7.
2. Thêm kiểu tấn công thứ tư: **im lặng kiêng khem** - một agent từ chối trả lời ("Tôi không biết"). Mỗi công ty tổng hợp nên đối xử với việc bỏ phiếu trắng như thế nào? Thực hiện lựa chọn của bạn.
3. Hoán đổi cụm ngữ nghĩa từ chuẩn hóa chuỗi sang tương tự embedding (sử dụng bất kỳ embedding model mã nguồn mở nào). Điều gì xảy ra với cuộc tấn công sycophacy?
4. Đọc CP-WBFT (arXiv:2511.10400). Thực hiện bước hiệu chuẩn đầu dò độ tin cậy (một model hiệu chuẩn riêng biệt kiểm tra độ tin cậy tự báo cáo của từng agent). Đo lường mức tăng accuracy trong kịch bản độc canh.
5. Đọc "AI Agents có thể đồng ý không?" (arXiv: 2603.01213). Tái tạo một thí nghiệm thỏa thuận vô hướng đơn giản: ba agents, một câu hỏi vô hướng, prompt tính cách lừa đảo. CPWBFT hoặc DecentLLM có bắt được nó không?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| BFT | "Khả năng chịu lỗi Byzantine" | Nghị định thư Castro-Liskov 1999 cho sự đồng thuận với `f < n/3` lỗi tùy ý. |
| Tiếng Byzantine | "Bất kỳ hành vi xấu nào" | Một nút có thể nói dối, thả tin nhắn, thất bại một cách âm thầm - bất cứ điều gì ngoại trừ sự cố an toàn. |
| Đầu dò độ tin cậy | "Anh chắc chắn như thế nào?" | Xác suất tự báo cáo hoặc do người hiệu chuẩn dự đoán đính kèm với một phiếu bầu. |
| Phân cụm ngữ nghĩa | "Cùng một câu trả lời, từ khác nhau" | Nhóm các câu trả lời tương đương trước khi đếm phiếu. |
| Trung vị hình học | "Trung tâm mạnh mẽ" | Điểm giảm thiểu tổng khoảng cách đến các điểm mẫu. Mạnh mẽ đến các ngoại lệ, không giống như giá trị trung bình. |
| Độc canh | "Cùng một model, cùng một thất bại" | Lỗi tương quan khi agents chia sẻ dữ liệu training hoặc model cơ sở. |
| Sự phù hợp sycophantic | "Đồng ý với giọng nói lớn" | Phiếu bầu của một agent thiên vị đối với bất kỳ ai nói first/loudest. |
| Core/Edge | "BFT phân cấp" | Phân tách WBFT: đồng thuận Core nhỏ trước, các nút Edge theo sau. Độ trễ giới hạn. |

## Đọc thêm

- [Castro & Liskov — Practical Byzantine Fault Tolerance (OSDI 1999)](https://pmg.csail.mit.edu/papers/osdi99.pdf) — nền tảng
- [CP-WBFT — Confidence-Probe Weighted BFT](https://arxiv.org/abs/2511.10400) — trọng số phiếu bầu theo độ tin cậy
- [DecentLLMs — leaderless multi-agent consensus](https://arxiv.org/abs/2507.14928) — tổng hợp hình học-trung bình
- [WBFT — Weighted BFT with Hierarchical Structure Clustering](https://arxiv.org/abs/2505.05103) — Core/Edge phân chia cho độ trễ có giới hạn
- [Can AI Agents Agree?](https://arxiv.org/abs/2603.01213) - sự mong manh của thỏa thuận vô hướng và tấn công lừa đảo
