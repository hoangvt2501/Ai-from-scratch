# Bối cảnh Agent mã hóa tự động (2026)

> SWE-bench Verified đã tăng từ 4% lên 80,9% trong vòng chưa đầy ba năm. Tương tự Claude Sonnet 4.5 đạt 43,2% trên SWE-agent v1 và 59,8% trên Cline tự trị - giàn giáo xung quanh model giờ cũng quan trọng như chính model. OpenHands (trước đây là OpenDevin) là nền tảng được MIT cấp phép hoạt động tích cực nhất và vòng lặp CodeAct của nó thực hiện các hành động Python trực tiếp trong một sandbox thay vì các lệnh gọi công cụ JSON. Các con số tiêu đề che giấu một vấn đề về phương pháp luận: 161 trong số 500 nhiệm vụ SWE-bench Verified chỉ yêu cầu thay đổi 1–2 dòng và SWE-bench Pro (10+ nhiệm vụ dòng) ở mức 23–59% cho cùng một models biên giới.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, CodeAct vs JSON tool-call comparison)
**Kiến thức tiên quyết:** Giai đoạn 14 · 07 (Sử dụng công cụ), Giai đoạn 15 · 01 (Đường chân trời dài agents)
**Thời lượng:** ~45 phút

## Vấn đề

"agent mã hóa nào là tốt nhất" là câu hỏi sai. Câu hỏi đúng là: trên một phân phối nhiệm vụ phù hợp với công việc của tôi, với giàn giáo mà tôi sẽ chạy trong production, tôi nhận được độ tin cậy từ đầu đến cuối nào?

Từ năm 2022 đến năm 2026, lĩnh vực này đã học được rằng giàn giáo - lớp truy xuất, công cụ lập kế hoạch, sandbox, vòng lặp chỉnh sửa-xác minh, định dạng phản hồi - đang chịu tải. Claude Sonnet 4.5 trên SWE-agent v1 đạt 43.2% trên SWE-bench Verified; model tương tự bên trong giàn giáo tự trị của Cline đạt 59.8%. 16.6 điểm khác biệt tuyệt đối, cùng trọng số. model cơ sở là một thành phần; vòng lặp là sản phẩm.

Vấn đề đồng hành là độ bão hòa benchmark che giấu hồi quy. SWE-bench Verified gần bão hòa và phần đuôi nhiệm vụ dễ dàng (161 trong số 500 nhiệm vụ yêu cầu ≤2 dòng) kéo điểm số cao nhất. Chất lượng trong thế giới thực được đo lường tốt hơn trên các bản phân phối như SWE-bench Pro (10+ thay đổi dòng), trong đó các nhà lãnh đạo vẫn ở mức 23–59%.

## Khái niệm

### SWE-bench, một đoạn

SWE-bench (Jimenez và cộng sự) xử lý các vấn đề thực sự GitHub với các bản vá sự thật cơ bản và yêu cầu một agent tạo ra một bản vá giúp bộ thử nghiệm vượt qua. SWE-bench Verified (OpenAI, 2024) là một tập hợp con 500 nhiệm vụ do con người quản lý với các nhiệm vụ mơ hồ và bị hỏng đã bị loại bỏ. SWE-bench Pro là sự kế thừa khó hơn - các nhiệm vụ đòi hỏi 10+ dòng thay đổi, trong đó biên giới hiện tại agents ở mức 23–59%.

### Đường cong 2022 → 2026 thực sự cho thấy điều gì

- **2022**: nghiên cứu models ở mức ~4% trên SWE-bench thô.
- **2024**: GPT-4 + giàn giáo kiểu Devin ở mức ~14%; SWE-agent ở mức ~12%.
- **2025**: Claude 3.5/3.7 Sonnet bên trong Aider và SWE-agent đẩy vào phạm vi 40–55%.
- **2026**: Claude Sonnet 4.5 và các đối thủ cạnh tranh biên giới ở mức 70–80% + trên SWE-bench Verified. Bảng xếp hạng của Epoch AI theo dõi trực tiếp điều này.

Độ dốc đến từ ba nguồn kép: models cơ sở tốt hơn, giàn giáo tốt hơn (CodeAct, phản xạ, vòng xác minh) và benchmarks tốt hơn (Đã xác minh loại bỏ nhiễu).

### CodeAct so với các lệnh gọi công cụ JSON

OpenHands (All-Hands-AI, arXiv:2407.16741, trước đây là OpenDevin) đã đặt cược kiến trúc cụ thể: thay vì model phát ra các cuộc gọi công cụ JSON mà máy chủ giải mã và thực thi, model phát ra mã Python và hạt nhân kiểu Jupyter chạy nó trong một sandbox. agent có thể lặp lại các tệp, công cụ chuỗi và bắt các ngoại lệ của riêng nó trong một hành động.

Sự đánh đổi:

- **JSON gọi công cụ**: mọi hành động là một lượt; dễ dàng kiểm tra; tính thành phần hạn chế; an toàn theo mặc định vì mỗi lệnh gọi đều đi qua một trình xác thực rõ ràng.
- **CodeAct**: một hành động có thể là toàn bộ chương trình; thành phần; yêu cầu sandbox cứng (OpenHands sử dụng cách ly Docker); Chế độ thất bại bao gồm bất cứ thứ gì mà sandbox runtime cho phép.

Cả hai kiến trúc đều production. CodeAct chiếm ưu thế trong các nền tảng mở (OpenHands, smolagents). Các lệnh gọi công cụ JSON vẫn chiếm ưu thế trong các dịch vụ được quản lý (Anthropic Agents được quản lý, OpenAI Trợ lý) nơi nhà cung cấp kiểm soát người thực thi.

### Giàn giáo trong bối cảnh năm 2026

| Giàn giáo | Giấy phép | Thực hiện model | Tài sản đáng chú ý |
|---|---|---|---|
| Bàn tay mở (OpenDevin) | Tiểu bang MIT | CodeAct trong Docker | Nền tảng mở tích cực nhất; Luồng sự kiện có thể phát lại |
| SWE-agent | Tiểu bang MIT | Giao diện máy tính Agent (ACI) | Giàn giáo SWE-bench đầu cuối đầu tiên |
| Người trợ giúp | Apache-2 (tiếng Anh) | edit-via-diff trong repo địa phương | Giàn giáo tối thiểu, ổn định hồi quy mạnh |
| Cline | Apache-2 (tiếng Anh) | VS Code agent với công cụ policy | Giàn giáo mở đạt điểm cao nhất trên Sonnet 4.5 |
| Devin (Nhận thức) | Độc quyền | Máy ảo + công cụ lập kế hoạch được quản lý | Danh mục sản phẩm "kỹ sư phần mềm AI" đầu tiên |
| Mã Claude | Độc quyền | Chế độ quyền + quy trình | Bài 10 trình bày chi tiết vòng lặp agent |

### Tại sao giàn giáo chiếm ưu thế

Chạy mã hóa là một quỹ đạo dài (Bài 1). Độ tin cậy kết hợp qua các bước. Ba nơi mà giàn giáo mua điểm:

1. **Truy xuất**: tìm đúng tệp để đọc là nút thắt thầm lặng. ACI của SWE-agent, chỉ mục tệp của OpenHands và bản đồ repo của Aider đều tấn công điều này.
2. **Vòng lặp xác minh**: chạy kiểm tra, đọc stack traces và thử lại là 10+ điểm delta trên SWE-bench.
3. **Ngăn chặn lỗi**: một sandbox quay trở lại khi có lỗi ngăn ngừa thiệt hại kép. Cùng một model có và không có vòng xác minh trông giống như hai sản phẩm khác nhau.

### Độ bão hòa Benchmark và phân bố thực

Các tác giả và Epoch AI OpenHands đều đánh dấu rằng SWE-bench Verified có một cái đuôi dễ dàng: 161 trong số 500 nhiệm vụ chỉ cần 1-2 dòng thay đổi. Điểm số cao một phần được thúc đẩy bởi cái đuôi này. SWE-bench Pro giới hạn ở 10+ thay đổi dòng và trả về điểm số trong phạm vi 23–59% ngay cả đối với các hệ thống biên giới. Phân phối production của bạn gần như chắc chắn gần với Pro hơn là Đã xác minh.

Hàm ý khi chọn một agent: chạy một tập hợp con giống như Pro của tồn đọng lỗi của riêng bạn. Điểm số quan trọng là điểm số trên các nhiệm vụ đại diện cho những gì bạn ship.

## Ứng dụng

`code/main.py` so sánh hai giàn giáo agent đồ chơi trên một phân phối nhiệm vụ nhỏ cố định:

1. Một giàn giáo **JSON tool-call** thực hiện một hành động mỗi lượt.
2. Một giàn giáo **CodeAct** có thể phát ra một đoạn Python nhỏ cho mỗi hành động.

Cả hai đều sử dụng sơ khai "model" (quy tắc xác định) vì vậy việc so sánh cách ly giàn giáo khỏi chất lượng model. Kết quả cho thấy giàn giáo CodeAct giải quyết nhiều nhiệm vụ hơn trong ít lượt hơn với chi phí là bán kính vụ nổ trên mỗi hành động lớn hơn.

## Sản phẩm bàn giao

`outputs/skill-scaffold-audit.md` giúp bạn kiểm tra giàn giáo agent mã hóa được đề xuất trước khi áp dụng: chất lượng truy xuất, sự hiện diện của trình xác minh, cách ly sandbox và benchmark phù hợp với phân phối.

## Bài tập

1. Chạy `code/main.py`. Mỗi giàn giáo thực hiện bao nhiêu lượt trong cùng một bộ nhiệm vụ? Bán kính vụ nổ mỗi hành động của mỗi loại là bao nhiêu?

2. Đọc bài báo OpenHands (arXiv: 2407.16741). Bài báo lập luận rằng CodeAct đánh bại các cuộc gọi công cụ JSON đối với các nhiệm vụ phức tạp. Xác định một chế độ thất bại mà bài báo thừa nhận và viết một câu về thời điểm chế độ đó sẽ chiếm ưu thế trong production.

3. Chọn một tác vụ từ tồn đọng lỗi của bạn sẽ yêu cầu 10+ dòng thay đổi trên hai tệp. Ước tính xác suất thành công từ đầu đến cuối cho một model biên giới theo (a) JSON lệnh gọi công cụ và (b) CodeAct. Biện minh cho khoảng cách.

4. SWE-bench Verified có 161 tác vụ một tệp, 1–2 dòng. Xây dựng một điểm số loại trừ chúng. Bảng xếp hạng xáo trộn như thế nào?

5. Đọc "Giới thiệu SWE-bench Verified" (OpenAI). Giải thích phương pháp cụ thể được sử dụng để loại bỏ các nhiệm vụ mơ hồ và nêu tên một danh mục mà quản lý sẽ bỏ sót.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|---|---|---|
| SWE-băng ghế | "Mã hóa benchmark" | Các vấn đề GitHub thực sự với các bản vá và bộ thử nghiệm |
| SWE-bench đã được xác minh | "Tập con đã được làm sạch" | 500 nhiệm vụ do con người quản lý, hiện tại dễ dàng hơn |
| SWE-bench Pro | "Tập con khó hơn" | 10+ thay đổi dòng; Biên giới ở mức 23–59% |
| Đạo luật | "Mã dưới dạng hành động" | Agent phát ra Python; Hạt nhân kiểu Jupyter thực thi trong sandbox |
| JSON gọi công cụ | "Function calling" | Mỗi hành động là một JSON payload có cấu trúc được xác thực trước khi thực hiện |
| Giàn giáo | "Agent framework" | Truy xuất + lập kế hoạch + trình thực thi + trình xác minh vòng quanh model cơ sở |
| ACI (Giao diện máy tính Agent) | "Định dạng của SWE-agent" | Bộ lệnh được thiết kế cho LLM công thái học, không phải vỏ người |
| Vòng lặp xác minh | "Kiểm tra và thử lại" | Chạy kiểm tra, đọc đầu ra, sửa bản vá; Tăng độ tin cậy không model lớn nhất |

## Đọc thêm

- [Jimenez et al. — SWE-bench](https://www.swebench.com/) - benchmark và phương pháp luận ban đầu.
- [OpenAI — Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) - tập hợp con được quản lý được xây dựng như thế nào.
- [Wang et al. — OpenHands: An Open Platform for AI Software Developers](https://arxiv.org/abs/2407.16741) — Kiến trúc CodeAct và thiết kế luồng sự kiện.
- [Epoch AI — SWE-bench leaderboard](https://epoch.ai/benchmarks) - điểm số được theo dõi trực tiếp.
- [Anthropic — Measuring agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — khung hình mã hóa agent độ tin cậy trong đường chân trời dài.
