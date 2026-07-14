# Bộ nhớ được chia sẻ và các mẫu bảng đen

> Hai cách tiếp cận cùng tồn tại trong hệ thống đa agent vào năm 2026: **nhóm tin nhắn** (mọi người đều nhìn thấy tin nhắn của mọi người, như trong AutoGen GroupChat hoặc MetaGPT) và **bảng đen có đăng ký** (agents đăng ký các sự kiện có liên quan, như trong MCP nhận biết ngữ cảnh hoặc Ma trận framework). Cả hai đều là phần trạng thái duy nhất của hệ thống đa agent - có nghĩa là cả hai đều là nơi các lỗi thú vị tồn tại. Chế độ lỗi tham chiếu là **nhiễm độc bộ nhớ**: một agent ảo giác một "sự thật", người khác agents coi nó là đã được xác minh và accuracy phân rã dần dần theo cách khó gỡ lỗi hơn nhiều so với sự cố ngay lập tức. Bài học này xây dựng cả hai cấu trúc từ stdlib, tiêm một cuộc tấn công đầu độc và chỉ ra ba biện pháp giảm thiểu thực sự hiệu quả trong production.

**Loại:** Tìm hiểu + Xây dựng
**Ngôn ngữ:** Python (stdlib, `threading`)
**Kiến thức tiên quyết:** Giai đoạn 16 · 04 (Primitive Model), Giai đoạn 16 · 09 (Mạng Swarm song song)
**Thời lượng:** ~75 phút

## Vấn đề

Hệ thống đa agent cần một nơi để agents chia sẻ sự thật. Một tùy chọn theo nghĩa đen là "chuyển mọi thứ trong tin nhắn" - nhưng điều đó sẽ phát minh lại trạng thái được chia sẻ bằng cách sao chép thêm. Một cách khác là "cung cấp cho mọi người một nhật ký toàn cầu" - nhưng nhật ký toàn cầu phát triển không giới hạn và dễ dàng bị đầu độc. Thứ ba là "dự án lượt xem trên mỗi agent" - có thể mở rộng nhưng nặng schema.

Khi một trong những agents ảo giác và viết ảo giác về trạng thái chia sẻ, mọi agent hạ lưu đọc trạng thái đó đều chấp nhận ảo giác là sự thật. Vào thời điểm con người nhận ra, chuỗi lý luận đã sâu năm bước và nguyên nhân gốc rễ là thông điệp thứ ba từng được viết. Gỡ lỗi phân rã nhiều agent accuracy khó hơn gỡ lỗi sự cố.

Đây là đầu độc trí nhớ. Đây là họ thất bại được ghi nhận nhiều thứ hai trong phân loại MAST (Cemri et al., arXiv: 2503.13657) và nó có cấu trúc: bất kỳ thiết kế bộ nhớ chia sẻ nào không có nguồn gốc và trình xác minh không thể ghi cuối cùng sẽ hiển thị nó.

## Khái niệm

### Hai cấu trúc liên kết chính

**Nhóm tin nhắn đầy đủ.** Mỗi agent đọc mọi tin nhắn. AutoGen GroupChat và MetaGPT sử dụng điều này. Đơn giản, minh bạch, có thể kiểm tra được, nhưng không vượt quá ~10 agents vì ngữ cảnh của mỗi agent lấp đầy công việc của agents khác.

```
agent-A ──write──▶ ┌────────────────┐ ◀──read── agent-D
                   │ message pool   │
agent-B ──write──▶ │                │ ◀──read── agent-E
                   │ (global log)   │
agent-C ──write──▶ └────────────────┘ ◀──read── agent-F
```

**Bảng đen có đăng ký.** Agents tuyên bố quan tâm đến các chủ đề; Chất nền chỉ định tuyến các thông điệp có liên quan. CA-MCP (arXiv:2601.11595) và framework phi tập trung Ma trận (arXiv:2511.21686) sử dụng điều này. Mở rộng quy mô hơn nữa, nhưng yêu cầu thiết kế schema trước để làm cho đăng ký có ý nghĩa.

```
                   ┌─ topic: prices ──┐
agent-A ──pub────▶ │                  │ ──▶ agent-D (subscribed)
                   ├─ topic: orders ──┤
agent-B ──pub────▶ │                  │ ──▶ agent-E (subscribed)
                   ├─ topic: alerts ──┤
agent-C ──pub────▶ │                  │ ──▶ agent-F (subscribed)
                   └──────────────────┘
```

### Khi mỗi người chiến thắng

- **Toàn bộ nhóm **chiến thắng khi agents ít (< 10), không đồng nhất và cuộc trò chuyện ngắn ngủi. Lý luận về việc ai đã nói điều gì là tầm thường khi mọi người đều nhìn thấy mọi thứ.
- **Bảng đen** chiến thắng khi agents có nhiều, đồng nhất về vai trò nhưng nhiều trong trường hợp (swarms) và cuộc trò chuyện kéo dài. Định tuyến giúp tiết kiệm token chi phí và ô nhiễm bối cảnh.

Production hệ thống thường trộn lẫn: một hồ bơi nhỏ đầy ở trên cùng (lớp lập kế hoạch), bảng đen bên dưới (worker lớp).

### Ngộ độc trí nhớ, trong một tình huống

Ba agents làm việc trong một nhiệm vụ nghiên cứu. Agent A là một agent truy xuất. Agent B là một người tóm tắt. Agent C là một nhà phân tích.

1. A lấy một trang và viết một thông điệp đến trạng thái chia sẻ: "Nghiên cứu báo cáo sự cải thiện accuracy 42%."
2. Trang được tìm nạp thực sự cho biết "cải thiện 4,2%". Một ảo giác một số thập phân.
3. B, đọc trạng thái chia sẻ, viết: "42% mức tăng accuracy lớn được báo cáo (nguồn: A)."
4. C, đọc trạng thái được chia sẻ, viết: "Đề nghị áp dụng - mức tăng 42% là biến đổi."
5. Báo cáo cuối cùng trích dẫn con số 42% chưa bao giờ tồn tại.

Không có agent nào bị sập. Không có thử nghiệm nào thất bại. Hệ thống "hoạt động". Ảo giác đã vượt qua từ bối cảnh của một agent vào mọi lý luận của agent hạ lưu thông qua trạng thái chung.

### Tại sao đây là cấu trúc

Không có trạng thái chung, ảo giác của agent A vẫn ở trong bối cảnh của A. Hạ lưu agents sẽ tìm nạp lại hoặc rút lại và có thể bắt lỗi. Với trạng thái chia sẻ ngây thơ, bối cảnh của A trở thành bối cảnh của mọi người, và ảo giác được rửa sạch thành sự thật.

Vấn đề không phải là trạng thái chia sẻ - nó là trạng thái chia sẻ **không có nguồn gốc và không có người xác minh độc lập**. Ba biện pháp giảm thiểu giải quyết vấn đề này:

1. **Xuất xứ thuộc tính trên mỗi lần ghi.** Mọi mục trong trạng thái được chia sẻ ghi lại ai đã viết nó, khi nào, dưới prompt nào và (nếu có) nguồn mà agent trích dẫn. Hạ lưu agents đọc với sự hoài nghi về nguồn gốc.
2. **Phiên bản ghi; coi chúng là chỉ thêm vào.** Chỉnh sửa là một mục mới thay thế mục cũ, không phải là bản cập nhật tại chỗ. Dấu vết kiểm toán được bảo tồn.
3. **Giữ ít nhất một agent không thể ghi vào trạng thái được chia sẻ.** Trình xác minh chỉ đọc agent lấy mẫu các mục nhập, tìm nạp lại nguồn và gắn cờ sự không nhất quán. Bởi vì nó không thể ghi vào hồ bơi, nó không thể bị đầu độc bởi hồ bơi.

### Tiền lệ bảng đen (Hayes-Roth, 1985)

Mô hình bảng đen có trước LLM agents bốn thập kỷ. Hayes-Roth (1985, "Kiến trúc bảng đen để kiểm soát") mô tả các Nguồn tri thức chuyên nghiệp quan sát bảng đen toàn cầu, đóng góp các giải pháp từng phần và trigger các nguồn khác. Bảng đen 2026 (CA-MCP, Ma trận) là cùng một mẫu với LLM agents là Nguồn tri thức và các đốm màu JSON là giải pháp một phần. Các tài liệu cũ đã ghi lại các giải pháp để viết tranh chấp, kiểm soát cơ hội và tính nhất quán mà các hệ thống hiện đại khám phá lại.

### Chiếu so với chế độ xem đầy đủ

Một bảng đen thuần túy cung cấp cho mọi người đăng ký cùng một hình chiếu (phạm vi chủ đề). Một thiết kế mạnh mẽ hơn là **phép chiếu mỗi agent**: mỗi agent có một chế độ xem được tùy chỉnh cho vai trò của nó. Các công cụ giảm trạng thái của LangGraph là triển khai chuẩn năm 2026 - hàm rút gọn gấp trạng thái toàn cầu thành một lát cắt theo vai trò cụ thể.

Dự báo trên mỗi agent mở rộng hơn nữa nhưng cần schema. Nếu không có nó, bạn xây dựng lại phép chiếu đặc biệt trong mọi prompt của agent.

### Các mẫu tranh chấp ghi

Viết nhiều agents đồng thời là một vấn đề đồng thời, không chỉ là một vấn đề LLM. Ba mẫu hoạt động:

- **Biên kịch tuần tự (nhà sản xuất duy nhất).** Tất cả các bài viết đều thông qua một điều phối viên agent tuần tự. Đơn giản, nhưng là một nút thắt cổ chai.
- **Đồng thời lạc quan với phiên bản.** Mỗi mục có một phiên bản; Người viết không phù hợp với phiên bản và thử lại. Kỹ thuật cơ sở dữ liệu cổ điển.
- **Phân vùng chủ đề.** Các chủ đề khác nhau agents sở hữu các chủ đề khác nhau. Không có tranh cãi chéo chủ đề. Yêu cầu ranh giới phân vùng được thiết kế.

Hầu hết năm 2026 frameworks mặc định là người viết tuần tự vì các cuộc gọi LLM đủ chậm nên hiếm khi xảy ra tranh chấp và nút thắt cổ chai không gây tổn thương.

### Trình xác minh không thể ghi

Giảm thiểu khả năng chịu tải nhất là trình xác minh chỉ đọc. Quy tắc thực hiện:

- Người xác minh chia sẻ trạng thái với nhóm (đọc bảng đen hoặc nhóm).
- Trình xác minh không có mã ghi cho trạng thái được chia sẻ — chỉ có một kênh xác minh riêng biệt.
- Verifier tìm nạp độc lập các nguồn được trích dẫn trong các bài viết. Cờ bất đồng.
- Đầu ra của chính Verifier được chuyển đến một con người hoặc một quyết định riêng biệt agent, không bao giờ được đưa trở lại nhóm.

Nếu không có sự tách biệt này, đầu ra của người xác minh sẽ trở thành các mục mới trong nhóm, có nghĩa là một hồ bơi bị nhiễm độc sẽ đầu độc người xác minh, đầu độc các xác minh của nó.

## Tự xây dựng

`code/main.py` thực hiện cả hai cấu trúc liên kết trong stdlib Python cộng với một cuộc tấn công ngộ độc đồ chơi và ba biện pháp giảm thiểu.

- `MessagePool` — Nhật ký chỉ nối thread an toàn với tính năng đọc đầy đủ.
- `Blackboard` — pub/sub theo chủ đề với đăng ký theo agent.
- `ProvenanceEntry` - mọi bản ghi ghi (người viết, dấu thời gian, prompt_hash, source_uri).
- `PoisoningScenario` - thực hiện một nhiệm vụ nghiên cứu kéo dài ba agent trong đó agent A ảo giác một số thập phân. In báo cáo cuối cùng.
- `Verifier` — một agent chỉ đọc tìm nạp lại các nguồn và gắn cờ những mâu thuẫn. Chạy cùng một kịch bản với trình xác minh hiện diện.

Chạy:

```
python3 code/main.py
```

Sản lượng dự kiến:
- Chạy 1 (không có trình xác minh): 42% bị ảo giác lan truyền đến báo cáo cuối cùng.
- Chạy 2 (với trình xác minh): trình xác minh gắn cờ sự không nhất quán, nhóm được gắn nhãn "bị gắn cờ", báo cáo cuối cùng bao gồm việc rút lại.

## Ứng dụng

`outputs/skill-memory-auditor.md` là một skill kiểm tra thiết kế bộ nhớ chia sẻ của bất kỳ hệ thống đa agent nào để biết nguồn gốc, phiên bản và tách xác minh. Chạy nó trên các kiến trúc đa agent mới trước khi production.

## Sản phẩm bàn giao

Đối với bất kỳ thiết kế bộ nhớ chia sẻ nào:

- Ghi lại nguồn gốc trên mỗi lần viết: `(writer, timestamp, prompt_hash, tool_calls_cited, source_uri)`.
- Đặt nhật ký chỉ nối thêm. Sửa chữa là các mục mới tham chiếu đến mục được thay thế.
- Triển khai ít nhất một trình xác minh chỉ đọc agent có quyền truy cập nguồn độc lập.
- Định tuyến đầu ra của trình xác minh đến một kênh riêng biệt, không quay lại nhóm được chia sẻ.
- Ghi lại tỷ lệ ghi là supersession - tỷ lệ tăng là bằng chứng ban đầu của các mẫu ảo giác.

## Bài tập

1. Chạy `code/main.py`. Xác nhận chạy 1 lan truyền ảo giác và chạy 2 bắt được nó.
2. Thêm ảo giác thứ hai: agent B phát minh ra kích thước dataset. Người xác minh sẽ nắm bắt cả hai mà không cần điều chỉnh bằng tay.
3. Chuyển toàn bộ nhóm sang bảng đen với các phân vùng chủ đề (`prices`, `summaries`, `analyses`). Phân vùng chủ đề khiến tình huống ngộ độc nào khó thực hiện hơn và tình huống nào không giúp ích được gì?
4. Đọc Hayes-Roth (1985, "Kiến trúc bảng đen để kiểm soát"). Xác định hai mô hình kiểm soát từ bài báo không được thảo luận trong bài học này mà các hệ thống năm 2026 sẽ được hưởng lợi.
5. Đọc CA-MCP (arXiv: 2601.11595). Ánh xạ Kho ngữ cảnh được chia sẻ của nó với class MessagePool hoặc Blackboard trong `code/main.py`. CA-MCP thêm primitives nào lên trên?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Nhóm tin nhắn | "Lịch sử trò chuyện được chia sẻ" | Nhật ký chỉ nối mà mọi agent đều đọc. Hoàn toàn minh bạch, tỷ lệ kém. |
| Bảng đen | "Không gian làm việc chung" | Khóa chủ đề pub/sub. Agents đăng ký các chủ đề có liên quan. Mở rộng quy mô xa hơn. |
| Xuất xứ | "Ai đã viết gì" | Siêu dữ liệu trên mỗi lần ghi: người viết, dấu thời gian, prompt, nguồn. |
| Ngộ độc trí nhớ | "Ảo giác lan rộng" | Lỗi của một agent đi vào trạng thái chia sẻ, xuôi dòng agents chấp nhận nó như sự thật. |
| Chỉ thêm | "Không cập nhật tại chỗ" | Sửa chữa là các mục mới thay thế. Duy trì dấu vết kiểm tra. |
| Trình xác minh không ghi | "Kiểm toán viên độc lập" | agent chỉ đọc tìm nạp lại nguồn và gắn cờ sự không nhất quán. |
| Chiếu | "Chế độ xem có phạm vi" | Chế độ xem trên mỗi agent được tính toán từ trạng thái toàn cầu. Bộ giảm tốc LangGraph là trường hợp chuẩn. |
| Nguồn kiến thức | "Chuyên gia agent" | Thuật ngữ năm 1985 của Hayes-Roth dành cho một người tham gia bảng đen. |

## Đọc thêm

- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) - phân loại MAT; Ngộ độc trí nhớ là một phân họ thất bại phối hợp
- [CA-MCP — Context-Aware Multi-Server MCP](https://arxiv.org/abs/2601.11595) — Kho ngữ cảnh được chia sẻ cho MCP servers phối hợp
- [Matrix — decentralized multi-agent framework](https://arxiv.org/abs/2511.21686) — bảng đen dựa trên hàng đợi tin nhắn không có bộ điều phối trung tâm
- [LangGraph state and reducers](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — mẫu chiếu trên mỗi agent trong production
- [Anthropic — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — ghi chú xuất xứ và xác minh từ việc triển khai production
