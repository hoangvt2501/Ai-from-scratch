# Không đồng bộ và Hogwild! Inference

> Giải mã suy đoán (Giai đoạn 10 · 15) song song hóa tokens trong một chuỗi. Nhiều agent frameworks song song trên toàn bộ chuỗi nhưng buộc phải phối hợp rõ ràng (bỏ phiếu, tách nhiệm vụ phụ). Hogwild! Inference (Rodionov et al., arXiv:2504.06261) làm một việc khác: chạy song song N thực thể của cùng một LLM với bộ nhớ đệm khóa-giá trị SHARED Mỗi worker thấy mọi worker khác được tạo ra tokens ngay lập tức. Các models lý luận hiện đại - QwQ, DeepSeek-R1 - có thể tự phối hợp thông qua bộ nhớ đệm được chia sẻ đó mà không cần bất kỳ fine-tuning nào. Cách tiếp cận này mang tính thử nghiệm nhưng nó mở ra một trục hoàn toàn mới của tính song song inference nằm trực giao với giải mã thông số kỹ thuật. Bài học này thực hiện trình mô phỏng Hogwild! hai worker trong stdlib Python và giải thích lý do tại sao sự hợp tác bộ nhớ đệm chia sẻ xuất hiện từ khả năng suy luận của model hiện có.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 10 · 12 (tối ưu hóa inference), Giai đoạn 10 · 15 (giải mã đầu cơ)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Mô tả ba cấu trúc liên kết LLM song song phổ biến (bỏ phiếu, nhiệm vụ phụ, Hogwild!) và đặt tên các vấn đề mà mỗi mục tiêu nhắm đến.
- Nêu cốt lõi Hogwild! Thiết lập: Nhiều workers, một KV cache chung, phối hợp khẩn cấp thông qua tự prompting.
- Tính toán tốc độ thời gian tường của Hogwild! như một hàm của `N` đếm worker, `p` song song cấp nhiệm vụ và `c` phối hợp trên cao.
- Triển khai Hogwild hai worker! mô phỏng vấn đề đồ chơi và quan sát bộ phận nhiệm vụ mới nổi.

## Vấn đề

Các LLMs hiện đại giải quyết các vấn đề khó bằng cách tạo ra các chuỗi lý luận dài - 5000 tokens logic từng bước là phổ biến, hàng chục nghìn tokens xảy ra với các bài toán sâu. Ở 35 tokens/sec giải mã trên model 70B, 50k tokens là 24 phút. Tương tác model thì không.

Giải mã suy đoán (Giai đoạn 10 · 15) giúp bạn tăng tốc gấp 3-5 lần bằng cách song song hóa trong một chuỗi. Ngoài ra, sự phụ thuộc tuần tự của giải mã tự hồi quy là trần cứng. Mỗi token mới phụ thuộc vào mọi prior token.

Câu hỏi hiển nhiên: chúng ta có thể song song giữa các trình tự không? Chạy nhiều bản sao của cùng một model về cùng một vấn đề, để họ hợp tác, yêu cầu họ phân chia công việc?

Prior việc: nhóm bỏ phiếu (chạy N models, chọn câu trả lời đa số), cây tư tưởng (branch các con đường suy luận và kết hợp lại) và đa agent frameworks (chỉ định mỗi agent một nhiệm vụ phụ, sử dụng điều phối viên). Tất cả những điều này đều giúp ích trong các lĩnh vực nhiệm vụ cụ thể. Tất cả chúng cũng giới thiệu bộ máy điều phối rõ ràng - quy tắc bỏ phiếu, logic branch và cắt tỉa, agent các giao thức nhắn tin agent.

Hogwild! Inference có một cách tiếp cận khác. N workers chia sẻ một KV cache duy nhất. Mỗi worker nhìn thấy mọi worker khác được tạo ra tokens ngay lập tức, như thể chúng là ngữ cảnh riêng của nó. Người workers - không có bất kỳ training hay fine-tuning nào - tìm ra cách phân chia công việc. models suy luận hiện đại (QwQ, DeepSeek-R1, chế độ suy luận gia đình Claude) có thể đọc bộ nhớ đệm được chia sẻ và nói những điều như "Tôi thấy worker 2 đã xử lý trường hợp cơ sở, vì vậy tôi sẽ làm việc trên bước quy nạp."

Việc tăng tốc phụ thuộc vào khối lượng công việc và thử nghiệm kể từ tháng 4 năm 2026. Nhưng ý tưởng này rất đáng biết vì nó mở ra một trục mới của inference song song.

## Khái niệm

### Thiết lập

Khởi tạo N worker processes, tất cả đều chạy cùng một LLM. Thay vì bộ nhớ đệm mỗi worker KV, hãy duy trì MỘT bộ nhớ đệm được chia sẻ. Khi worker `i` tạo token `t_j`, token được ghi vào bộ nhớ đệm được chia sẻ ở vị trí tiếp theo. Khi worker `k` thực hiện bước tiếp theo, nó sẽ đọc trạng thái hiện tại của bộ nhớ đệm (bao gồm mọi thứ mà tất cả N workers đã tạo ra cho đến nay).

Tại thời điểm bước, workers chạy đua để viết tokens. Không có chỉ số vị trí trên mỗi worker - bộ nhớ đệm là một chuỗi tăng trưởng duy nhất. Thứ tự được xác định bởi thời gian đến ghi.

### Tại sao sự phối hợp xuất hiện

Các workers chia sẻ một prompt. Thường là một cái gì đó như "Bạn là một trong N thực thể làm việc cùng nhau về vấn đề này. Mỗi thực thể đọc bộ nhớ được chia sẻ và có thể xem những gì các thực thể khác đã ghi. Tránh công việc dư thừa." prompt cộng với bộ nhớ đệm được chia sẻ là đủ. Lý luận models đọc bộ nhớ đệm, chú ý phần nào của vấn đề đã được thử và (thường nhưng không phải lúc nào) xoay trục đến các phần chưa được khám phá.

Hogwild! giấy (Rodionov và cộng sự, 2025) báo cáo các quan sát như:

- Workers xây dựng kế hoạch và truyền đạt chúng cho các workers khác thông qua bộ nhớ đệm.
- Workers nhận thấy lỗi trong lý luận của workers khác và gọi họ ra.
- Workers thích ứng khi một kế hoạch thất bại và đề xuất các giải pháp thay thế.
- Khi được nhắc kiểm tra dự phòng, workers phát hiện nó và xoay trục.

Không có điều nào trong số này đòi hỏi fine-tuning. Hành vi nổi lên đến từ khả năng suy luận mà model đã có.

### Việc đặt tên

Tên của bài báo dựa trên Hogwild! SGD (Recht et al., 2011), một optimizer cập nhật không đồng bộ. Phép so sánh: workers không đồng bộ của SGD tất cả đều ghi vào một parameter vector dùng chung; Hogwild! Inference workers tất cả đều ghi vào một KV cache chung. Cả hai đều dựa vào sự hội tụ thực nghiệm hơn là đảm bảo đồng bộ hóa.

### RoPE làm cho điều này trở nên dễ dàng

Vị trí quay Embeddings (RoPE, Su et al. 2021) mã hóa thông tin vị trí thông qua xoay trong vectors Q và K. Bởi vì các vị trí là vòng quay và không phải là độ lệch tích hợp, vị trí của token có thể thay đổi mà không cần tính toán lại mục KV cache. Khi worker `i` ghi vào bộ nhớ đệm được chia sẻ ở vị trí `p`, những workers khác đọc vị trí đó có thể sử dụng trực tiếp mục nhập được lưu trong bộ nhớ cache - không cần xoay lại.

Trong model vị trí đã học hoặc vị trí tuyệt đối, Hogwild! sẽ cần vô hiệu hóa bộ nhớ đệm trên mỗi lần ghi đồng thời. RoPE cho phép bộ nhớ đệm ổn định.

### Toán học thời gian tường

Hãy để `T_serial` là thời gian để một worker giải quyết vấn đề một mình. Hãy để `p` là phân số song song hóa ở cấp độ nhiệm vụ. Hãy để `c` là chi phí phối hợp mỗi bước (đọc bộ nhớ đệm mở rộng, quyết định viết gì).

Thời gian worker một lần: `T_serial`.
N-worker Hogwild! Thời gian, nếu phối hợp miễn phí: `T_serial * ((1 - p) + p / N)`. Amdahl cổ điển.
Với sự phối hợp trên cao: `T_serial * ((1 - p) + p / N) + c * steps_per_worker`.

Để một worker hoạt động hiệu quả, `c` phải nhỏ so với thời gian giải mã mỗi bước. Về lý luận models tạo ra tokens 5k+, workers có thể đủ khả năng phối hợp hàng trăm tokens và vẫn vượt lên trước. Trong các nhiệm vụ trò chuyện ngắn, sự phối hợp chiếm ưu thế và Hogwild! tệ hơn nối tiếp.

### Ví dụ cụ thể

Bài toán suy luận: 10k tokens chain-of-thought. Giả sử bài toán có `p = 0.7` nội dung song song (các chiến lược chứng minh khác nhau, phân tích trường hợp khác nhau) và `c = 200` tokens chi phí phối hợp trên mỗi worker. Với `N = 4` workers:

- Thời gian nối tiếp: 10000 bước giải mã.
- Hogwild! Thời gian: 10000 * (0.3 + 0.7 / 4) + 200 * 4 = 10000 * 0.475 + 800 = 5550 bước giải mã.
- Tăng tốc: 10000 / 5550 = 1.8x.

Điều đó là khiêm tốn. Nhưng đối với các bài toán suy luận dài hơn (50k tokens), chi phí phối hợp sẽ khấu hao và tăng tốc đẩy 2,5-3 lần. Hogwild! tương đương inference với tính song song cấp thread trong một ngôn ngữ cho phép bạn viết mã đa luồng một cách tự nhiên.

### Khi nào nên tiếp cận Hogwild!

- Các vấn đề suy luận dài (hàng ngàn tokens) trong đó nhiệm vụ có thể được song song giữa các mục tiêu phụ độc lập.
- Lý luận models đã được rèn luyện để suy nghĩ từng bước. Những models không lý luận không tự phối hợp tốt.
- Triển khai một nút có đủ VRAM để chứa bộ nhớ đệm dùng chung cộng với N worker processes. Bộ nhớ đệm được chia sẻ, nhưng mỗi worker có bộ nhớ kích hoạt riêng.

### Khi nào không nên

- Trò chuyện tương tác ngắn. Chi phí phối hợp chiếm ưu thế.
- Các tác vụ không song song (bằng chứng tuyến tính đơn, biên dịch đơn). N=1 là giá trị tối đa.
- models không lý luận. Không có sự phối hợp nào xuất hiện.
- Triển khai nhiều nút. Bộ nhớ đệm được chia sẻ cần đồng bộ hóa worker chéo rất nhanh. Nội bộ nút vẫn ổn; nút chéo là một thảm họa về độ trễ.

### Trạng thái thử nghiệm

Tính đến tháng 4 năm 2026, Hogwild! là một phương pháp nghiên cứu với việc triển khai PyTorch mã nguồn mở. Production áp dụng đã không xảy ra. Ba yếu tố cản trở:

1. Quản lý KV cache được chia sẻ trên các processes đồng thời là kỹ thuật không tầm thường.
2. Sự phối hợp khẩn cấp phụ thuộc vào nhiệm vụ; benchmarks vẫn đang được xây dựng.
3. Tốc độ tăng tốc khiêm tốn so với những gì giải mã đầu cơ đã mang lại và cả hai có thể được kết hợp nhưng kỹ thuật kết hợp là một lớp khác.

Đáng biết. Đáng để thử nghiệm. Chưa đáng để đặt cược một sản phẩm.

```figure
continuous-batching
```

## Tự xây dựng

`code/main.py` triển khai một món đồ chơi Hogwild! Mô phỏng:

- Hai worker processes, mỗi loại là một "LLM" xác định tạo ra một trong một số loại token (token công việc, quan sát-token, tọa độ-token) với xác suất đã biết.
- Bộ nhớ đệm dùng chung (chỉ là danh sách tokens) mà cả hai workers đọc và ghi.
- Một logic phối hợp đơn giản: khi một worker thấy rằng người kia đã tạo ra đủ công việc tokens trong một thể loại, nó sẽ chọn một thể loại khác.

Trình mô phỏng chạy cho ngân sách bước cố định và báo cáo:

- Tổng tokens công việc được sản xuất.
- Tổng thời gian tường (số worker bước).
- Tăng tốc hiệu quả trong một worker.
- Một trace trong số đó worker đã viết token nào.

### Bước 1: bộ nhớ đệm được chia sẻ

Một danh sách mà cả hai workers thêm vào. Khóa đơn giản (Python `threading.Lock`) trong một triển khai thực tế; chúng tôi mô phỏng với một bộ đếm.

### Bước 2: vòng lặp worker

Mỗi worker, trên mỗi bước:

- Đọc bộ nhớ cache được chia sẻ hiện tại.
- Quyết định loại token nào sẽ viết dựa trên những gì đã có.
- Viết một token.

### Bước 3: phỏng đoán phối hợp

Nếu thể loại X đã có K tokens trong bộ nhớ đệm và thể loại dự định của worker là X, worker chuyển sang thể loại Y. Đây là một đồ chơi thay thế cho hành vi model lý luận của "lưu ý điều này đã được đề cập, thay vào đó hãy làm điều gì đó khác."

### Bước 4: tăng tốc đo được

Chạy trình mô phỏng với N=1 worker và với N=2 workers, cùng tổng ngân sách bước. Đếm tokens công việc được tạo ra. N=2 sẽ tạo ra nhiều tokens công việc hơn khoảng 1,5-1,8 lần do phân chia nhiệm vụ theo hướng phối hợp.

### Bước 5: nhấn mạnh sự phối hợp

Giảm độ nhạy của phỏng đoán phối hợp. Chạy lại. Quan sát rằng nếu không có sự phối hợp tốt, N = 2 dư thừa tạo ra cùng một tokens và tốc độ giảm xuống dưới 1. Điều này phù hợp với quan sát của bài báo: thủ thuật chỉ hoạt động nếu workers có khả năng suy luận để tự phối hợp.

## Ứng dụng

Hogwild! tích hợp trong production tính đến tháng 4 năm 2026 là cấp nghiên cứu. Việc triển khai tham chiếu từ Yandex/HSE/IST dựa trên PyTorch và nhắm mục tiêu thiết lập đa process một nút trên DeepSeek-R1 và QwQ models.

Lộ trình áp dụng thực dụng:

1. Lập hồ sơ khối lượng công việc suy luận-nhiệm vụ của bạn. Đo lường tỷ lệ tokens mang tính khám phá (nhiều chiến lược, phân tích trường hợp, tìm kiếm) so với tuyến tính.
2. Nếu khám phá chiếm ưu thế, hãy chạy Hogwild! thử nghiệm hai worker. Đo lường sự cải thiện thời gian trên tường.
3. Nếu mức cải thiện dưới 1,3 lần, bạn đang ở trong chế độ phối hợp chi phối. Hoàn nguyên về một worker.
4. Nếu mức cải thiện trên 1,5 lần, hãy đẩy đến N = 4 và đo lại. Lợi nhuận giảm dần thường đạt khoảng N = 4-8.

Kết hợp với giải mã suy đoán: mỗi Hogwild! worker có thể sử dụng giải mã thông số kỹ thuật một cách độc lập. Hai lần tăng tốc nhân lên (đại khái), mang lại giải mã thông số kỹ thuật 3x và Hogwild! thành 5,4x hiệu quả so với giải mã một worker ngây thơ.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-parallel-inference-router.md`. Với hồ sơ khối lượng công việc lý luận (ngân sách token, hồ sơ song song nhiệm vụ, họ model, mục tiêu triển khai), nó định tuyến giữa bỏ phiếu, cây tư tưởng, đa agent, Hogwild! và các chiến lược giải mã đầu cơ.

## Bài tập

1. Chạy `code/main.py` với cài đặt mặc định. Xác nhận N = 2 Hogwild! configuration tạo ra nhiều tokens làm việc hơn so với đường cơ sở N = 1 trong cùng một thời gian tường.

2. Giảm cường độ của phỏng đoán phối hợp (đặt `coordination_weight=0.1`). Chạy lại. Cho thấy rằng tốc độ sụp đổ. Giải thích lý do: workers nỗ lực trùng lặp khi chúng không thể phối hợp.

3. Tính toán Hogwild dự kiến! Tăng tốc cho nhiệm vụ suy luận 50k token với `p=0.8, c=500` và N = 4 workers. Làm tương tự cho nhiệm vụ trò chuyện 1k token với `p=0.3, c=200` và N = 4. Tại sao một là chiến thắng và cái kia là loss?

4. Đọc Hogwild! Phần 4 của bài báo (đánh giá sơ bộ). Xác định hai chế độ thất bại mà các tác giả báo cáo. Mô tả cách một prompt phối hợp tốt hơn có thể giảm thiểu từng chế độ.

5. Kết hợp Hogwild! Với giải mã suy đoán trong đồ chơi: mỗi worker sử dụng giải mã thông số kỹ thuật 2 token bên trong. Báo cáo tốc độ nhân lên. Vấn đề kế toán nào phát sinh khi hai workers đều muốn mở rộng cùng một tiền tố bộ nhớ đệm được chia sẻ?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Hogwild! | "workers song song, bộ nhớ đệm dùng chung" | N trường hợp của cùng một LLM chạy đồng thời với một KV cache dùng chung; phối hợp khẩn cấp thông qua tự prompting |
| Chia sẻ KV cache | "Phương tiện phối hợp" | Một bộ đệm KV đang phát triển duy nhất mà tất cả workers đọc và ghi; cho phép hiển thị token tức thì trên workers |
| Phối hợp khẩn cấp | "Không cần training" | LLMs có khả năng suy luận có thể đọc bộ nhớ cache được chia sẻ và phân chia công việc mà không cần bất kỳ giao thức fine-tuning hoặc rõ ràng nào |
| Phối hợp trên cao (c) | "Tokens dành thời gian định hướng" | Chi phí mỗi worker để đọc bộ nhớ đệm mở rộng và quyết định phải làm gì; phải ở mức nhỏ so với tổng thời gian giải mã |
| Phân số song song (p) | "Những gì có thể chạy song song" | Tính song song ở cấp độ nhiệm vụ: phần của tổng công việc không tuần tự về bản chất |
| RoPE cho phép Hogwild! | "Các vị trí quay là thay đổi" | Bởi vì các vị trí là luân phiên, nên việc ghi vào bộ nhớ đệm dùng chung không yêu cầu tính toán lại prior tokens |
| Nhóm bỏ phiếu | "Chạy N, chọn đa số" | Cấu trúc liên kết inference song song đơn giản nhất; hữu ích cho việc phân loại, ít hơn cho suy luận dạng dài |
| Cây tư tưởng | "Branch và cắt tỉa" | Chiến lược suy luận khám phá nhiều branches và cắt tỉa; logic phối hợp rõ ràng |
| Đa agent framework | "Giao nhiệm vụ phụ" | Mỗi agent có một vai trò; một điều phối viên dàn nhạc; giao thức nặng nề trên |

## Đọc thêm

- [Rodionov et al. — Hogwild! Inference: Parallel LLM Generation via Concurrent Attention (arXiv:2504.06261)](https://arxiv.org/abs/2504.06261) - Hogwild! giấy, đánh giá sơ bộ về QwQ và DeepSeek-R1
- [Recht, Re, Wright, Niu — Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent (arXiv:1106.5730, NeurIPS 2011)](https://arxiv.org/abs/1106.5730) — Hogwild gốc!, nguồn gốc đặt tên
- [Su et al. — RoFormer: Enhanced Transformer with Rotary Position Embedding (arXiv:2104.09864)](https://arxiv.org/abs/2104.09864) — RoPE, thuộc tính làm cho bộ nhớ đệm chia sẻ trở nên dễ xử lý inference
- [Yao et al. — Tree of Thoughts: Deliberate Problem Solving with Large Language Models (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) - chiến lược suy luận cây tư tưởng Hogwild! nằm trực giao với
- [Leviathan et al. — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192) - giải mã suy đoán, song song trong trình tự Hogwild! sáng tác với
- [Hogwild! reference PyTorch implementation](https://github.com/eqimp/hogwild_llm) - nguồn tin cậy duy nhất cho các thí nghiệm của bài báo
