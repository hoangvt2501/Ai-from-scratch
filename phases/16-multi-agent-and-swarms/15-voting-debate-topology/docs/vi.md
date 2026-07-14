# Bỏ phiếu, tự nhất quán và tranh luận cấu trúc liên kết

> Tổng hợp rẻ nhất: mẫu N agents độc lập, đa số-bỏ phiếu. Wang et al. Tính nhất quán năm 2022 đã làm điều này với một model lần lấy mẫu N. Multi-agent mở rộng nó với agents **không đồng nhất** để thoát khỏi độc canh - models khác nhau, prompts khác nhau, nhiệt độ khác nhau, bối cảnh khác nhau. Ngoài phiếu bầu đa số, cấu trúc liên kết tranh luận còn quan trọng: MultiAgentBench (arXiv: 2503.01935, ACL 2025) đã đánh giá sự phối hợp của ngôi sao / chuỗi / cây / đồ thị và tìm thấy **biểu đồ tốt nhất cho nghiên cứu**, với "thuế phối hợp" qua ~4 agents. AgentVerse (ICLR 2024) ghi lại hai mô hình nổi lên - hành vi tình nguyện và hành vi tuân thủ - và sự phù hợp vừa là feature (tìm kiếm sự đồng thuận) vừa là rủi ro (tư duy nhóm, Bài 24). Bài học này lập bản đồ không gian cấu trúc liên kết, xây dựng từng biến thể và đo lường thuế phối hợp.

**Loại:** Tìm hiểu + Xây dựng
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 16 · 07 (Xã hội Tâm trí và Tranh luận), Giai đoạn 16 · 14 (Đồng thuận và BFT)
**Thời lượng:** ~75 phút

## Vấn đề

Tranh luận có thể cải thiện accuracy (Du và cộng sự, arXiv: 2305.14325). Nó cũng có thể làm suy giảm nó. Liệu tranh luận có giúp ích hay không phụ thuộc vào bốn lựa chọn cấu trúc:

1. Ai nói chuyện với ai (cấu trúc liên kết).
2. Có bao nhiêu vòng (Du 2023: cả vòng và agents đều quan trọng độc lập).
3. Liệu agents có không đồng nhất hay không (cơ sở khác nhau models phá vỡ độc canh).
4. Liệu có tiếng nói đối nghịch hay không (người điều khiển thép so với người điều khiển rơm).

Các nhóm "chạy 5 agents và bỏ phiếu" vào một nhiệm vụ thường thụt lùi so với một agent duy nhất. Những thất bại không phải là ngẫu nhiên. Họ theo dõi cấu trúc liên kết và tính không đồng nhất. Bài học này là bản đồ cấu trúc liên kết.

## Khái niệm

### Tính nhất quán, đường cơ sở một model

Wang et al. 2022 ("Tính nhất quán của bản thân cải thiện chuỗi suy luận") đã lấy mẫu model N lần tương tự ở temperature > 0 và đa số bỏ phiếu cho các câu trả lời theo con đường suy luận. Kết quả trên GSM8K: tăng đáng kể với N = 40 mẫu trên một giải mã tham lam duy nhất. Tính nhất quán là tiền thân của một agent cho việc bỏ phiếu nhiều agent.

Giới hạn: tính tự nhất quán sử dụng một model cơ sở. Sai số được tương quan bởi cấu trúc. Nếu model có bias hệ thống, tất cả N mẫu đều có chung nó.

### Bỏ phiếu nhiều agent, phần mở rộng không đồng nhất

Thay thế N samples bằng N * khác nhau * agents. Các models cơ sở khác nhau (Claude, GPT, Llama), prompts khác nhau, quyền truy cập công cụ khác nhau. Lợi ích: các lỗi không tương quan. Chi phí: khác nhau agents chi phí số tiền khác nhau; phối hợp chúng làm tăng thêm chi phí.

Tên kinh điển năm 2026 cho cuộc tranh luận không đồng nhất là **A-HMAD** - Cuộc tranh luận đa Agent không đồng nhất đối nghịch. Không được chấp nhận rộng rãi, nhưng các bài báo sử dụng thuật ngữ này cho "cuộc tranh luận models khác nhau, giúp giảm các lỗi tương quan từ sự sụp đổ của độc canh".

### Bốn cấu trúc liên kết

```
star                chain               tree                graph

    ┌─A─┐           A─B─C─D         ┌──A──┐              A───B
    │   │                           │     │              │ × │
    B   C                           B     C              D───C
    │   │                          / \   / \
    D   E                         D   E F   G           (fully connected)
```

Ngôi sao: một trung tâm, tất cả các trung tâm khác chỉ nói chuyện với trung tâm. Tương đương với giám sát worker không có kênh sau.
Chuỗi: tuyến tính, mỗi agent nhìn thấy đầu ra prior của một người. Giống như Pipeline.
Cây: phân cấp, được sử dụng bởi hệ thống agent phân cấp (Bài 06).
Biểu đồ: any-to-any. Bao gồm các nhóm được kết nối đầy đủ và các DAG tùy ý.

### Thuế điều phối (MultiAgentBench)

MultiAgentBench (MARBLE, ACL 2025, arXiv:2503.01935) đã đo điểm chuẩn sao, chuỗi, cây, đồ thị trên một bộ tác vụ bao gồm nghiên cứu, mã hóa và lập kế hoạch. Kết quả đo lường chính:

- Cấu trúc liên kết **Đồ thị** chiến thắng trong các nhiệm vụ nghiên cứu. Thông tin chảy từ bất kỳ đến bất kỳ; agents có thể phê bình lẫn nhau.
- **Ngôi sao** giành chiến thắng trong các nhiệm vụ thực tế trả lời nhanh. Bộ lọc và hợp nhất Hub.
- **Chuỗi** chiến thắng khi pipelines từng bước (tinh chỉnh theo giai đoạn).
- **Thuế phối hợp** xuất hiện qua ~4 agents trong cấu trúc liên kết đồ thị. Đồng hồ treo tường và chi phí token tăng nhanh hơn chất lượng.

Trần 4 agent là thực nghiệm, không phải cơ bản. Nó phản ánh năng lực ngữ cảnh LLM năm 2026: ngữ cảnh của mỗi agent chứa đầy đầu ra của các đồng nghiệp và giá trị cận biên của việc thêm agent N + 1 giảm khi mọi người có thể nhìn thấy tất cả mọi người.

### Chiến lược tranh luận đa Agent ("Chúng ta có nên điên rồ không?")

arXiv:2311.17371 là cuộc khảo sát năm 2023 về các chiến lược MAD. Phát hiện chính được sao chép bởi những người khác: Các biến thể MAD *tương tự về mặt cấu trúc* với tính nhất quán (sampling độc lập + tổng hợp) thường hoạt động kém hơn tính nhất quán khi sử dụng cùng một ngân sách. MAD giúp ích nhiều nhất khi agents thực sự không đồng nhất và cuộc tranh luận có cấu trúc đối nghịch (một agent tranh luận phản đối).

### Các mẫu mới nổi của AgentVerse

AgentVerse (ICLR 2024, https://proceedings.iclr.cc/paper_files/paper/2024/file/578e65cdee35d00c708d4c64bce32971-Paper-Conference.pdf) ghi lại hai hành vi xuất hiện từ cuộc tranh luận đa agent ngay cả khi không có thiết kế rõ ràng:

- **Tình nguyện viên.** Một agent đề nghị giúp đỡ ("Tôi có thể thực hiện bước tiếp theo") mà không được nhắc nhở. Hữu ích: nó phân bổ công việc cho agent có khả năng nhất cho một nhiệm vụ phụ.
- **Sự tuân thủ.** Một agent điều chỉnh lập trường của mình để phù hợp với một nhà phê bình, ngay cả khi nhà phê bình sai. Đây là cuộc tranh luận tương đương với sự sycophancy (Bài 14).

Sự phù hợp là lý do tại sao tranh luận cho đến khi thỏa thuận thưởng cho những kẻ bắt nạt. Các vòng có giới hạn với một trọng tài giảm nhẹ riêng biệt.

### Tính không đồng nhất: núm thực tế di chuyển accuracy

Một mô hình 2024-2026 trong tài liệu thực tế: hoán đổi một trong các agents N của bạn cho một model cơ sở khác sẽ mang lại mức tăng accuracy lớn hơn so với tăng N lên 1. Trực giác là độc canh - mỗi nguồn lỗi độc lập mới có giá trị hơn một mẫu tương quan bổ sung.

Trong giới hạn, tính không đồng nhất đánh bại số lượng. Ba models khác nhau đánh bại năm bản sao của một model trên hầu hết các nhiệm vụ có ground truth sạch.

### Phương pháp của bồi thẩm đoàn

framework Sibyl (được trích dẫn trong tài liệu Minsky-LLM) chính thức hóa một "bồi thẩm đoàn" - một tập hợp nhỏ các agents chuyên ngành tinh chỉnh các câu trả lời bằng cách bỏ phiếu ở mỗi giai đoạn. Không giống như phiếu đa số đơn giản, bồi thẩm đoàn có vai trò: một agent kiểm tra chéo, một cung cấp bối cảnh, một người chấm điểm hợp lý. Phương pháp của bồi thẩm đoàn là điểm trung gian giữa bỏ phiếu đơn giản (rẻ, dễ bị độc canh) và MAD đầy đủ (đắt tiền, dễ tuân thủ).

### Khi bỏ phiếu với tranh luận chiếm ưu thế

- Câu hỏi có ground truth (thực tế, toán học, hành vi mã). Hội tụ phiếu bầu có ý nghĩa.
- Agents có thể truy cập các nguồn hoặc công cụ khác nhau (có tính không đồng nhất).
- Các vòng được giới hạn (2-3 điển hình) và có một thẩm phán hoặc người xác minh riêng biệt.
- Ngân sách cho phép 3-5 agents. Ngoài 5-7 trên cấu trúc liên kết đồ thị, thuế phối hợp chiếm ưu thế.

### Khi bỏ phiếu với tranh luận đau đớn

- Câu hỏi được định hình theo ý kiến. Agents hội tụ đến bất kỳ câu trả lời nào có vẻ tự tin nhất, không đúng nhất.
- Tất cả agents đều có chung một model cơ bản. Độc canh làm cho sự đồng thuận trở nên vô nghĩa.
- Các vòng không giới hạn. Sự phù hợp luôn chiến thắng.
- Nhiệm vụ rất đơn giản. Một agent duy nhất có tính nhất quán ở N = 5 rẻ hơn và chính xác.

## Tự xây dựng

`code/main.py` thực hiện:

- `run_star(agents, hub, question)` - các cuộc thăm dò trung tâm mỗi worker, tổng hợp.
- `run_chain(agents, question)` — tinh chỉnh tuần tự.
- `run_tree(root, children, question)` — phân cấp với tổng hợp độ sâu-2.
- `run_graph(agents, question, rounds)` - tranh luận tất cả, các vòng có giới hạn.
- Một mặt số không đồng nhất theo kịch bản: mỗi agent có một `error_bias` cho biết sự sai lệch có hệ thống của nó.
- Một harness đo lường chạy mỗi cấu trúc liên kết ở N = 3, 5, 7 và báo cáo (accuracy, total_tokens, wallclock_simulated).

Chạy:

```
python3 code/main.py
```

Đầu ra dự kiến: bảng cấu trúc liên kết × N → (accuracy, tokens, độ trễ). Đồ thị giành chiến thắng ở N = 3-5 trong các nhiệm vụ kiểu nghiên cứu; ngôi sao chiến thắng trong các nhiệm vụ thực tế nhanh; biểu đồ tại N = 7 cho thấy thuế điều phối (độ trễ tăng nhanh hơn accuracy).

## Ứng dụng

`outputs/skill-topology-picker.md` là một skill đọc mô tả nhiệm vụ và đề xuất cấu trúc liên kết (ngôi sao / chuỗi / cây / đồ thị), N (số agents), cấu hình không đồng nhất (models cơ sở để sử dụng) và giới hạn tròn.

## Sản phẩm bàn giao

Đối với bất kỳ nhóm nào:

- Bắt đầu với **tự nhất quán ở N = 5 **bằng cách sử dụng một model cơ sở mạnh. Đó là đường cơ sở rẻ.
- Nâng cấp lên **bỏ phiếu không đồng nhất ở N=3** nếu accuracy quan trọng. Đo đồng bằng.
- Chỉ nâng cấp lên **cấu trúc liên kết tranh luận** nếu nhiệm vụ có cấu trúc (nghiên cứu, nhiều bước) và các vòng giới hạn là khả thi.
- Luôn ghi nhật ký cụm thiểu số. Khi một thiểu số kiên trì đúng, bạn có một tín hiệu đa dạng.
- Benchmark đồng hồ treo tường và tokens cùng với accuracy. "accuracy tốt hơn với chi phí gấp 10 lần" là một quyết định kinh doanh.

## Bài tập

1. Chạy `code/main.py`. Vẽ đường cong thuế phối hợp cho cấu trúc liên kết đồ thị: accuracy vs N, tokens vs N. Đường cong biến đổi ở N nào?
2. Triển khai A-HMAD: ba agents với các thành kiến khác nhau có chủ ý. Đường cơ sở hoàn toàn giống nhau bias như thế nào so với A-HMAD về cuộc tấn công độc canh từ Bài 14?
3. Thêm vai trò "thẩm phán" vào cấu trúc liên kết đồ thị không bỏ phiếu, chỉ ghi điểm đồng thuận cuối cùng. Điều này có thay đổi hành vi tuân thủ mới nổi không?
4. Đọc bài báo AgentVerse (ICLR 2024). Xác định hành vi mới nổi nào mà việc triển khai của bạn thể hiện mạnh mẽ nhất. Bạn có thể gợi ra hành vi ngược lại bằng cách thay đổi prompt không?
5. Đọc MultiAgentBench (arXiv: 2503.01935) Phần 4 (thí nghiệm cấu trúc liên kết). Tái tạo kết quả "graph-wins-research" trên một nhiệm vụ từ bài báo bằng cách sử dụng harness của bạn.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Tự nhất quán | "Mẫu N lần, bỏ phiếu" | Vương 2022. Đơn model, N temperature>0 mẫu, đa số bỏ phiếu về con đường suy luận. |
| Tính không đồng nhất | "Các models khác nhau" | Quần thể của các models cơ sở hoặc prompt họ khác nhau. Phá vỡ độc canh. |
| ĐIÊN | "Tranh luận nhiều agent" | Thuật ngữ chung cho agents trao đổi phê bình qua các vòng. Xem Du 2023. |
| A-HMAD | "MAD không đồng nhất đối nghịch" | Biến thể MAD nhấn mạnh cấu trúc models + đối nghịch khác nhau. |
| Cấu trúc liên kết | "Ai nói chuyện với ai" | Ngôi sao, chuỗi, cây, đồ thị. Xác định luồng thông tin. |
| Thuế phối hợp | "Lợi nhuận giảm dần" | Trên ~4 agents trên biểu đồ, chi phí tăng nhanh hơn chất lượng. |
| Hành vi tình nguyện | "Trợ giúp không được nhắc nhở" | Mô hình mới nổi của AgentVerse: một agent đề nghị thực hiện một bước. |
| Hành vi tuân thủ | "Thỏa thuận dưới áp lực" | Mô hình nổi lên của AgentVerse: một agent phù hợp với một nhà phê bình. |
| Ban giám khảo | "Bảng điều khiển chuyên dụng nhỏ" | Hòa tấu theo phong cách Sibyl với các vai trò (giám khảo, bối cảnh, người ghi điểm). |

## Đọc thêm

- [Wang et al. — Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) - đường cơ sở một model
- [Du et al. — Improving Factuality and Reasoning via Multiagent Debate](https://arxiv.org/abs/2305.14325) - cả agents VÀ các vòng đều quan trọng độc lập
- [MultiAgentBench / MARBLE](https://arxiv.org/abs/2503.01935) — cấu trúc liên kết benchmark hiển thị biểu đồ tốt nhất cho nghiên cứu, chuỗi cho pipelines
- [Should we be going MAD?](https://arxiv.org/abs/2311.17371) - Khảo sát chiến lược MAD; nhận thấy MAD thường thua sự tự nhất quán với ngân sách ngang nhau
- [AgentVerse (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/578e65cdee35d00c708d4c64bce32971-Paper-Conference.pdf) - tình nguyện viên và các mô hình mới nổi
- [MARBLE repo](https://github.com/ulab-uiuc/MARBLE) — tham khảo benchmark triển khai
