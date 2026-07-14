# Agent Kinh tế, Token Ưu đãi, Danh tiếng

> agents tự trị dài hạn (đường cong làm việc từ 1 giờ đến 8 giờ của METR) cần cơ quan kinh tế. **5-layer stack** mới nổi là: **DePIN** (điện toán vật lý) → **Danh tính** (W3C DID + vốn danh tiếng) → **Nhận thức** (RAG + MCP) → **Thanh toán** (trừu tượng hóa tài khoản) → **Quản trị** (Agentic DAO). Các mạng khuyến khích Production agent bao gồm **Bittensor** (mạng con TAO thưởng cho models cụ thể nhiệm vụ), **Fetch.ai / ASI Alliance** (ASI-1 Mini LLM + FET token) và **Gonka** (PoW dựa trên transformer phân bổ lại điện toán cho các tác vụ AI năng suất). Công việc học tập: LaMAS phi tập trung của AAMAS 2025 sử dụng **Phân bổ tín dụng giá trị Shapley** để thưởng công bằng cho những agents đóng góp; Nghiên cứu của Google "Thiết kế cơ chế cho models ngôn ngữ lớn" đề xuất **token đấu giá** với thanh toán giá thứ hai theo tổng hợp đơn điệu. Bài học này xây dựng một thị trường agent tối thiểu, áp dụng phân bổ tín dụng giá trị Shapley cho nhiều agent pipeline và chạy đấu giá token giá thứ hai để bộ máy lý thuyết trò chơi hạ cánh một cách cụ thể.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 16 · 16 (Đàm phán và Thương lượng), Giai đoạn 16 · 09 (Mạng Swarm song song)
**Thời lượng:** ~75 phút

## Vấn đề

Hệ thống đa agent trở nên phức tạp khi agents cùng tạo ra giá trị nhưng cần được khen thưởng riêng lẻ. Các cơ chế cổ điển - chia đều, người đóng góp cuối cùng lấy tất cả - là không công bằng hoặc có thể chơi được. Phần thưởng dựa trên liên minh thông qua các giá trị Shapley là công bằng theo xây dựng nhưng tốn kém để tính toán. Tài liệu 2025-2026 thúc đẩy các ước tính hữu ích: Shapley sampling, đấu giá tổng hợp đơn điệu và danh tiếng trên chuỗi tích lũy từ các đóng góp đã được xác nhận.

Ngoài phân bổ tín dụng, lĩnh vực này đã chuyển sang agents kinh tế thực tế: Bittensor TAO thưởng điện toán khai thác cho models cụ thể của mạng con fine-tune Fetch.ai/ASI thưởng cho việc sử dụng ASI-1 Mini LLM với FET tokens, Gonka phân bổ lại transformer bằng chứng công việc cho các nhiệm vụ AI năng suất. Agents giao dịch tự chủ tồn tại ngày nay; Câu hỏi đặt ra là làm thế nào để điều chỉnh các ưu đãi.

Bài học này coi các nền kinh tế agent như một nhóm vấn đề cụ thể - phân bổ tín dụng, thiết kế cơ chế và danh tiếng - và xây dựng mỗi nền kinh tế với phép toán tối thiểu để các ý tưởng gắn bó.

## Khái niệm

### Máy stack 5 lớp agent tiết kiệm

1. **DePIN (điện toán vật lý).** Cơ sở hạ tầng phi tập trung cho thuê GPU, lưu trữ, băng thông. Mạng con Bittensor, Mạng kết xuất, Akash. Không agent cụ thể; agents sử dụng nó.
2. **Danh tính.** Mã định danh phi tập trung (DID) W3C cung cấp cho mỗi agent một ID bền bỉ độc lập với bất kỳ nền tảng nào. Danh tiếng tích lũy cho DID. Giao thức mạng Agent (ANP) sử dụng DID làm lớp khám phá.
3. **Nhận thức.** Vòng lặp suy luận của agent: LLM + RAG + MCP. Đây là những gì các giai đoạn khác xây dựng.
4. **Thanh toán.** Trừu tượng hóa tài khoản (ERC-4337) cho phép agents thanh toán gas từ số dư của chính họ mà không cần nắm giữ ETH. Agents có thể thanh toán cho các dịch vụ, thanh toán lẫn nhau hoặc điện toán.
5. **Quản trị.** Agentic DAO: cấu trúc quản trị nơi con người *và* agents bỏ phiếu về các thay đổi giao thức, với quyền biểu quyết gắn liền với danh tiếng.

Không phải hệ thống production nào cũng sử dụng cả năm. Bittensor sử dụng 1, 2, một phần 3, một phần 4, không sử dụng 5. OpenAI agents không sử dụng ngoại trừ 3. stack là một bản đồ tham chiếu, không phải là một yêu cầu.

### Bittensor, Fetch.ai, Gonka — cái gì chạy

**Bittensor (TAO).** Mạng con là các nhiệm vụ chuyên biệt (mô hình hóa ngôn ngữ, tạo hình ảnh, dự báo). Các thợ đào gửi model đầu ra. Trình xác thực xếp hạng chúng; Chấm điểm theo trọng số cổ phần phân phối phần thưởng TAO. Mỗi mạng con có đánh giá riêng. Bài học kinh tế: trả tiền cho chất lượng đầu ra theo nhiệm vụ cụ thể, không phải sử dụng tính toán.

**Fetch.ai / ASI Alliance.** ASI-1 Mini LLM chạy trên mạng của Fetch.ai; người dùng trả tokens FET cho inference. Câu chuyện agents như ngang hàng mạnh mẽ hơn ở đây: một agent trên Fetch có thể gọi một người khác cho một nhiệm vụ và thanh toán bằng FET.

**Gonka.** Transformer bằng chứng công việc: "công việc" là chuyển tiếp của một transformer. Các thợ đào kiếm tiền bằng cách chạy các tác vụ inference có kết quả chính xác đã biết (từ dữ liệu training). PoW sản xuất tài nguyên thay vì PoW dựa trên băm.

Cả ba đều là lớp production tính đến tháng 4 năm 2026. Phân phối thanh toán khác nhau. Bittensor thưởng chất lượng so với trình xác thực mạng con; Tiện ích tìm nạp phần thưởng được đo lường bởi người dùng trả tiền; Gonka thưởng cho công việc inference có thể kiểm chứng.

### Phân bổ tín dụng giá trị Shapley

Ba agents cộng tác trong một nhiệm vụ. Điểm đầu ra là 0,8. Ai đã đóng góp gì?

Giá trị Shapley: phân bổ tín dụng duy nhất thỏa mãn bốn tiên đề (hiệu quả, đối xứng, tuyến tính, rỗng). Đối với agent `i`:

```
shapley(i) = (1/N!) * sum over all orderings O of (v(S_i_O ∪ {i}) - v(S_i_O))
```

trong đó `S_i_O` là tập hợp các agents trước `i` trong thứ tự `O`. Trong thực tế: liệt kê tất cả các hoán vị, ghi lại đóng góp cận biên của từng agent trong mỗi hoán vị, trung bình.

Đối với N = 3 agents, có 6 hoán vị. Đối với N = 10, 3,6M - vì vậy trong thực tế, bạn lấy mẫu thứ tự thay vì liệt kê.

### Đấu giá giá thứ hai để tổng hợp

Nghiên cứu của Google ("Thiết kế cơ chế cho models ngôn ngữ lớn") đề xuất đấu giá token giá thứ hai để tổng hợp LLM đầu ra. Thiết lập: N agents mỗi đề xuất hoàn thành; mỗi cái đều có một giá trị riêng để được chọn. Người đấu giá chọn đề xuất có giá trị cao nhất và trả giá trị *cao thứ hai*. Dưới tổng hợp đơn điệu (giá trị phụ thuộc vào đề xuất nào được chọn, không phải bao nhiêu đề xuất được đấu thầu), điều này là trung thực - agents giá thầu giá trị thực của chúng.

Tại sao điều này lại quan trọng đối với LLM hệ thống: bạn có thể thuê ngoài các tác vụ hoàn thành cho nhiều agents với mức giá khác nhau; Cuộc đấu giá chọn những gì tốt nhất + trả công bằng và agents không có động cơ để báo cáo sai.

### Vốn danh tiếng

Điểm danh tiếng bị ràng buộc bởi DID được tích lũy từ các đóng góp đã được xác nhận. Quy tắc cập nhật đơn giản:

```
rep(i, t+1) = alpha * rep(i, t) + (1 - alpha) * contribution_quality(i, t)
```

Với hệ số phân rã `alpha` gần bằng 1. Danh tiếng:

- Rẻ để đọc cho các quyết định định tuyến ("gửi nhiệm vụ khó khăn cho agents đại diện cao").
- Tốn kém để rèn (tích lũy theo thời gian, ràng buộc với DID).
- Có thể bị cắt giảm: các đóng góp không xác minh được sẽ trừ.

### AAMAS 2025 LaMAS phi tập trung

Đề xuất LaMAS (AAMAS 2025) kết hợp: nhận dạng DID, phân bổ tín dụng giá trị Shapley và cơ chế đấu giá đơn giản. Tuyên bố chính: phân cấp bước phân bổ tín dụng làm cho hệ thống có thể kiểm toán được và miễn nhiễm với thao tác một điểm.

### Nơi kinh tế sụp đổ

- **Thao túng oracle giá.** Nếu chức năng tín dụng có thể được chơi trò chơi, agents sẽ chơi nó. Mọi cơ chế đều cần một bài kiểm tra đối nghịch.
- **Sybil tấn công.** Một nhà điều hành quay N agents giả để thổi phồng đóng góp của chính họ. DID chậm nhưng không dừng được điều này; Chi phí rèn giũa danh tiếng là giảm thiểu.
- **Chi phí xác minh.** Phân bổ tín dụng chỉ công bằng khi người xác minh. Nếu xác minh rẻ (LLM nhỏ), nó có thể được đánh bạc; Nếu đắt tiền (bảng điều khiển con người), hệ thống không mở rộng quy mô.
- **Vượt quá quy định.** Agent nền kinh tế giao thoa với quy định tài chính. Bittensor, Fetch và Gonka đều hoạt động trong các vùng xám pháp lý ở một số khu vực pháp lý kể từ năm 2026.

### Khi agent nền kinh tế có ý nghĩa

- **Mạng mở với các toán tử không đồng nhất.** Không có nhóm nào kiểm soát tất cả agents.
- **Kết quả có thể xác minh.** Nếu không xác minh, phân bổ tín dụng chỉ là phỏng đoán.
- **Quy trình làm việc dài hạn.** One-shot nhiệm vụ không được hưởng lợi từ việc tích lũy danh tiếng.
- **Thanh toán được mã hóa là khả thi về mặt pháp lý** trong khu vực tài phán của bạn.

Trong các hệ thống công ty khép kín, kinh tế học nhường chỗ cho việc phân bổ đơn giản hơn (các nhà quản lý phân công công việc, các chỉ số là nội bộ). Các tài liệu kinh tế học chủ yếu áp dụng cho các mạng mở.

## Tự xây dựng

`code/main.py` thực hiện:

- `shapley(value_fn, agents)` — tính toán Shapley chính xác bằng cách liệt kê cho N nhỏ.
- `second_price_auction(bids)` - cơ chế trung thực; Người chiến thắng trả cao thứ hai.
- `Reputation` - Danh tiếng bị ràng buộc bởi DID với sự phân rã và chặt chém theo cấp số nhân.
- Bản demo 1: ba agents cộng tác, tín dụng thuộc tính Shapley chính xác.
- Demo 2: năm agents giá thầu cho một vị trí nhiệm vụ; Đấu giá giá thứ hai chọn người chiến thắng + thanh toán.
- Demo 3: 100 vòng giao nhiệm vụ cho agents với đại diện không đồng nhất; Định tuyến trọng số đại diện đánh ngẫu nhiên.

Chạy:

```
python3 code/main.py
```

Đầu ra dự kiến: Giá trị Shapley cho mỗi agent; kết quả đấu giá thể hiện cân đối giá thầu trung thực; Định tuyến trọng số đại diện cho thấy chất lượng tăng 10-20% so với ngẫu nhiên sau khi khởi động.

## Ứng dụng

`outputs/skill-economy-designer.md` thiết kế một nền kinh tế agent tối thiểu: lựa chọn lớp nhận dạng, cơ chế phân bổ tín dụng, cơ chế thanh toán, quy tắc danh tiếng.

## Sản phẩm bàn giao

Điều hành một nền kinh tế agent vào năm 2026:

- **Bắt đầu với danh tiếng, không phải tokens.** Danh tiếng rẻ để thực hiện và chỉ có giá trị; tokens làm tăng thêm sự phức tạp về pháp lý và kinh tế.
- **Xác minh trước khi nhận thưởng.** Không bao giờ phân phối tín dụng mà không có bước xác minh độc lập. Chất lượng tự báo cáo tích lũy các trò chơi sybil.
- **Mẫu Shapley, không phải Shapley-chính xác.** Mẫu 100-1000 đơn đặt hàng; Liệt kê chính xác không chia tỷ lệ.
- **Hệ số phân rã nắp và danh tiếng sàn.** Phân rã không giới hạn xóa sạch những người đóng góp hợp pháp; Phân rã quá chậm thưởng cho agents đại diện cao cũ.
- **Kiểm tra cơ chế đối nghịch.** Chạy các kịch bản đội đỏ trước khi mở mạng. Mỗi cơ chế đều có một lý thuyết trò chơi; Bạn muốn tìm ra những lỗ hổng chứ không phải những kẻ tấn công.

## Bài tập

1. Chạy `code/main.py`. Xác nhận giá trị Shapley tổng trên tổng giá trị (tiên đề hiệu quả). Thay đổi chức năng giá trị; Phân bổ Shapley có thay đổi theo hướng dự kiến không?
2. Thực hiện Shapley * sampling * (Monte Carlo trên K đặt hàng). K ảnh hưởng đến accuracy xấp xỉ như thế nào? So sánh với chính xác cho N = 4.
3. Thực hiện bước thành lập liên minh trước cuộc đấu giá: agents có thể merge vào các nhóm và đặt giá thầu như một đơn vị. Liên minh nào hình thành? Kết quả Pareto-có tốt hơn đấu thầu riêng lẻ không?
4. Đọc bài đăng về thiết kế cơ chế của Google Research. Xác định một giả định, nếu bị vi phạm, sẽ phá vỡ tính trung thực. Chế độ lỗi đó trông như thế nào trong cài đặt LLM?
5. Đọc bài báo LaMAS phi tập trung AAMAS 2025. Thực hiện bước Shapley của họ trên 10 agents trên một nhiệm vụ tổng hợp. Tính toán chính xác mất bao lâu? sampling đến gần như thế nào với 100 lần bốc thăm?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Mã PIN | "Cơ sở hạ tầng vật lý phi tập trung" | Token khuyến khích compute/storage/bandwidth. Bittensor, Akash, Render. |
| DID | "Mã định danh phi tập trung" | Thông số kỹ thuật W3C cho ID di động. Danh tiếng Agent ràng buộc với DID, không phải với một nền tảng. |
| ERC-4337 · | "Trừu tượng hóa tài khoản" | Tài khoản hợp đồng có thể tài trợ gas, cho phép thanh toán agent. |
| Giá trị Shapley | "Phân bổ tín dụng công bằng" | Phân bổ độc đáo đáp ứng hiệu quả, đối xứng, tuyến tính, rỗng. |
| Đấu giá giá thứ hai | "Đấu giá Vickrey" | Cơ chế trung thực: người chiến thắng trả giá thầu cao thứ hai. Tương thích với tổng hợp đơn sắc. |
| Vốn danh tiếng | "Điểm chất lượng tích lũy" | Điểm ràng buộc DID từ các đóng góp đã được xác nhận; phân rã theo thời gian. |
| Agentic DAO | "Agents + con người cai trị" | DAO với agent cử tri là class đầu tiên, quyền biểu quyết gắn liền với danh tiếng. |
| Tín chỉ TAO / FET / GPU | "Token giáo phái" | Bittensor TAO, Fetch.ai FET, các tokens DePIN khác nhau. |

## Đọc thêm

- [The Agent Economy](https://arxiv.org/abs/2602.14219) - Khảo sát năm 2026 về stack kinh tế agent 5 lớp
- [Google Research — Mechanism design for large language models](https://research.google/blog/mechanism-design-for-large-language-models/) — token đấu giá với tổng hợp đơn điệu
- [AAMAS 2025 — decentralized LaMAS](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p2896.pdf) — Phân bổ tín dụng giá trị Shapley
- [Bittensor TAO documentation](https://docs.bittensor.com/) — cấu trúc mạng con và phân phối phần thưởng
- [Fetch.ai / ASI Alliance](https://fetch.ai/) - ASI-1 Mini LLM và FET token
- [W3C Decentralized Identifiers (DIDs) spec](https://www.w3.org/TR/did-core/) — Nền tảng nhận dạng
