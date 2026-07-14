# Di sản của FIPA-ACL và Đạo luật Lời nói

> Trước MCP, trước A2A, đã có FIPA-ACL. Năm 2000, IEEE Foundation for Intelligent Physical Agents phê chuẩn một ngôn ngữ giao tiếp agent với hai mươi biểu diễn, hai ngôn ngữ nội dung và một tập hợp các giao thức tương tác - contract net, subscribe/notify, request-when. Nó mờ nhạt khỏi ngành công nghiệp vì chi phí bản thể quá nặng nề đối với web, nhưng sự hồi sinh LLM của các hệ thống đa agent đang lặng lẽ thực hiện lại các ý tưởng tương tự mà không có ngữ nghĩa chính thức: JSON hợp đồng thay thế cho biểu diễn, ngôn ngữ tự nhiên thay thế cho bản thể. Bài học này đọc FIPA-ACL một cách nghiêm túc để bạn có thể thấy quyết định giao thức năm 2026 nào là tái tạo, đâu là mới lạ và đâu là làn sóng hiện tại sẽ khám phá lại các vấn đề mà những năm 2000 đã giải quyết.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 16 · 01 (Tại sao nên sử dụng Multi-Agent)
**Thời lượng:** ~60 phút

## Vấn đề

Bối cảnh giao thức agent năm 2026 rất bận rộn: MCP cho các công cụ, A2A cho agents, ACP cho kiểm toán doanh nghiệp, ANP cho niềm tin phi tập trung, NLIP cho nội dung ngôn ngữ tự nhiên, cộng với CA-MCP và hai chục đề xuất nghiên cứu. Mỗi thông số kỹ thuật tự công bố là nền tảng.

Đọc thành thật là hầu hết trong số họ đang khám phá lại một cây quyết định hai mươi năm tuổi rất cụ thể. Lý thuyết lời nói-hành động từ Austin (1962) và Searle (1969) cho chúng ta "lời nói là hành động". KQML (1993) đã biến nó thành một giao thức dây. FIPA-ACL (phê chuẩn năm 2000) đã tạo ra tiêu chuẩn tham chiếu: hai mươi biểu diễn, ngôn ngữ nội dung SL0/SL1, giao thức tương tác cho mạng hợp đồng và thông báo đăng ký. JADE và JACK là các nền tảng tham chiếu Java. Nỗ lực này đã mờ nhạt vào khoảng năm 2010 vì chi phí bản thể quá nặng nề và web đang chiến thắng.

Khi bạn nhìn vào `tools/call` của MCP, vòng đời nhiệm vụ của A2A hoặc kho ngữ cảnh được chia sẻ của CA-MCP, bạn đang xem xét lại các quyết định FIPA nhẹ nhàng hơn, JSON bản địa. Biết di sản cho bạn biết hai điều: "đổi mới" mới nào thực sự là phát minh lại và chế độ lỗi cũ nào mà các thông số kỹ thuật mới sẽ khám phá lại.

## Khái niệm

### Hành động lời nói, trong một đoạn văn

Austin nhận thấy rằng một số câu không mô tả thế giới - chúng thay đổi nó. "Tôi hứa." "Tôi yêu cầu." "Tôi tuyên bố." Ông gọi đây là những lời nói biểu diễn. Searle chính thức hóa năm loại: quyết đoán, chỉ thị, thỏa thuận, biểu cảm, tuyên bố. KQML (Finin et al., 1993) đã thực hiện hoạt động này cho phần mềm agents: một thông điệp là một biểu diễn (hành động) cộng với nội dung (hành động là gì). FIPA-ACL đã làm sạch các lỗ hổng của KQML và tiêu chuẩn hóa khoảng hai mươi hiệu suất.

### Hai mươi biểu diễn FIPA (danh sách một phần)

| Biểu diễn | Ý định |
|---|---|
| `inform` | "Tôi nói với bạn rằng P là sự thật" |
| `request` | "Tôi yêu cầu bạn làm X" |
| `query-if` | "P có đúng không?" |
| `query-ref` | "Giá trị của X là bao nhiêu?" |
| `propose` | "Tôi đề nghị chúng ta làm X" |
| `accept-proposal` | "Tôi chấp nhận lời đề nghị" |
| `reject-proposal` | "Tôi từ chối đề xuất" |
| `agree` | "Tôi đồng ý làm X" |
| `refuse` | "Tôi từ chối làm X" |
| `confirm` | "Tôi xác nhận P là đúng" |
| `disconfirm` | "Tôi phủ nhận P" |
| `not-understood` | "Tin nhắn của bạn không phân tích cú pháp" |
| `cfp` | "Kêu gọi đề xuất về X" |
| `subscribe` | "Thông báo cho tôi khi X thay đổi" |
| `cancel` | "Hủy X đang diễn ra" |
| `failure` | "Tôi đã thử X và thất bại" |

Danh sách đầy đủ có trong `fipa00037.pdf` (Cấu trúc thông báo FIPA ACL). Vấn đề không phải là ghi nhớ nó - vấn đề là mọi thứ trong số này tương ứng với một primitive một giao thức LLM cuối cùng sẽ thêm lại.

### Thông báo FIPA-ACL chuẩn

```
(inform
  :sender       agent1@platform
  :receiver     agent2@platform
  :content      "((price IBM 83))"
  :language     SL0
  :ontology     finance
  :protocol     fipa-request
  :conversation-id   conv-42
  :reply-with   msg-17
)
```

Bảy trường mang phong bì giao thức; Một trường (`content`) mang payload. rest của các trường chính xác là những gì bạn phát minh lại mỗi khi bạn thử lại, phân luồng và bản thể vào một giao thức JSON.

### Hai nền tảng kế thừa

**JADE** (Java Agent DEvelopment framework, 1999–2020) là runtime tuân thủ FIPA được sử dụng nhiều nhất. Agents mở rộng class cơ sở, trao đổi thông điệp ACL, chạy bên trong containers và phối hợp bằng cách sử dụng "hành vi". Thư viện giao thức tương tác shipped với contract-net, subscribe-notify, request-when và propose-accept.

**JACK** (Phần mềm định hướng Agent, thương mại) nhấn mạnh lý luận BDI (Niềm tin-Mong muốn-Ý định) trên các thông điệp FIPA. Trang trọng hơn, ít được chấp nhận hơn.

Cả hai đều suy giảm khi web stack sử dụng nhiều agent. MCP và A2A là "containers" runtime của năm 2026.

### Tại sao FIPA mờ nhạt

- **Chi phí ontology.** FIPA yêu cầu một ontology được chia sẻ để phân tích cú pháp `content`. Đồng ý về ontology là một process tiêu chuẩn kéo dài nhiều năm. Web chỉ sử dụng HTTP + JSON.
- **Ngữ nghĩa chính thức không ai sử dụng.** SL (Ngôn ngữ ngữ nghĩa) đưa ra các điều kiện chân lý nghiêm ngặt, nhưng hầu hết các hệ thống production sử dụng nội dung dạng tự do và bỏ qua chủ nghĩa hình thức.
- **Khóa công cụ.** JADE chỉ dành cho Java; JACK là thương mại. Các nhóm đa ngôn ngữ đã định tuyến xung quanh cả hai.
- **Internet đã giành chiến thắng stack.** REST, sau đó là JSON-RPC, sau đó gRPC thay thế transport của ACL.

### Sự hồi sinh LLM là FIPA-lite

So sánh `request` FIPA với MCP `tools/call`:

```
(request                                {
  :sender  agent1                         "jsonrpc": "2.0",
  :receiver tool-server                   "method":  "tools/call",
  :content "(lookup stock IBM)"           "params":  {"name":"lookup_stock",
  :ontology finance                                   "arguments":{"symbol":"IBM"}},
  :conversation-id c42                    "id": 42
)                                        }
```

Cùng một phong bì, cú pháp khác nhau. Cả hai đều mang theo: who, whom, intent, payload, correlation id. Cả hai đều không phải là một cuộc cách mạng so với cuộc cách mạng kia - chúng là sự đánh đổi khác nhau trên cùng một thiết kế.

Cuộc khảo sát năm 2025 của Liu và cộng sự ("Khảo sát về các giao thức tương tác Agent: MCP, ACP, A2A, ANP", arXiv: 2505.02279) làm cho dòng dõi này trở nên rõ ràng: MCP tương ứng với các hành vi lời nói sử dụng công cụ, A2A với các hành vi phát biểu ngang hàng agent, ACP cho các hành vi phát biểu theo dõi kiểm toán, ANP với các phần mở rộng nhận dạng phi tập trung. Các thông số kỹ thuật mới là hậu duệ của ACL với cú pháp JSON và ngữ nghĩa lỏng lẻo hơn.

### Sự đánh đổi, được nêu rõ ràng

**Những gì FIPA đã cung cấp cho bạn và thông số kỹ thuật hiện đại giảm:**

- Ngữ nghĩa chính thức - bạn có thể chứng minh `inform` ngụ ý người gửi tin vào nội dung.
- Một danh mục kinh điển của các trình diễn - bạn không cần phải tranh luận lại "chúng ta có nên có một `cancel`?".
- Nhiều thập kỷ của các mô hình giao thức tương tác - hợp đồng mạng, đăng ký-thông báo, đề xuất-chấp nhận - với các thuộc tính đúng đắn đã biết.

**Những gì thông số kỹ thuật hiện đại cung cấp cho bạn và FIPA không có:**

- JSON gốc payloads tương thích với mọi công cụ hiện đại.
- Nội dung ngôn ngữ tự nhiên mà LLMs có thể diễn giải mà không cần bản thể được mã hóa bằng tay.
- stack transport web (HTTP, SSE, WebSocket).
- Khám phá khả năng thông qua các tài liệu tự mô tả (MCP `listTools`, Thẻ A2A Agent).

Ngữ nghĩa ý định lỏng lẻo hơn để triển khai dễ dàng hơn. Đó là giao dịch chính xác.

### Các giao thức tương tác đáng để chuyển

FIPA shipped ~15 giao thức tương tác. Ba điều đáng được đưa vào LLM hệ thống đa agent:

1. **Giao thức mạng hợp đồng (CNP).** Người quản lý phát hành `cfp` (kêu gọi đề xuất); nhà thầu trả lời bằng `propose`; quản lý accepts/rejects. Đây là mô hình thị trường nhiệm vụ kinh điển (Giai đoạn 16 · 16 Đàm phán).
2. **Subscribe/Notify.** Người đăng ký gửi `subscribe`; Nhà xuất bản gửi `inform` bất cứ khi nào chủ đề thay đổi. Đây là mọi xe buýt sự kiện vào năm 2026.
3. **Yêu cầu-Khi nào.** "Làm X khi điều kiện Y được giữ nguyên." Hành động trì hoãn với các điều kiện tiên quyết. Tương tự 2026 là các nhiệm vụ bị trì hoãn trong các công cụ quy trình làm việc bền bỉ (Giai đoạn 16 · 22 Production Mở rộng quy mô).

Mỗi ánh xạ rõ ràng vào hàng đợi tin nhắn hiện đại, HTTP + thăm dò ý kiến hoặc SSE streaming.

### Điều gì bị hỏng khi bạn bỏ bản thể

Nếu không có bản thể học được chia sẻ, agents suy ra ý nghĩa từ nội dung ngôn ngữ tự nhiên. Chế độ lỗi năm 2026 được ghi lại là **trôi ngữ nghĩa**: hai agents sử dụng cùng một từ (`"customer"`) cho các khái niệm khác nhau một cách tinh tế, agent của người nhận hành động theo cách giải thích sai, không có trình xác thực schema nào bắt được nó. Yêu cầu bản thể của FIPA sẽ từ chối thông điệp tại thời điểm phân tích cú pháp.

Giảm thiểu mà không cần bản thể học đầy đủ:

- JSON Schema trên `content` - loại bỏ các lỗi cấu trúc ở dây.
- Nhập artifacts (A2A) — từ chối phương thức sai.
- Biểu diễn rõ ràng trong phong bì - làm cho ý định rõ ràng ngay cả khi nội dung là ngôn ngữ tự nhiên.

### Thông số kỹ thuật năm 2026, được ánh xạ đến di sản lời nói-hành động

| Thông số kỹ thuật hiện đại | Tương tự FIPA | Những gì nó lưu giữ | Những gì nó rơi |
|---|---|---|---|
| MCP `tools/call` | `request` | Ý định rõ ràng, ID tương quan | ngữ nghĩa chính thức, bản thể học |
| MCP `resources/read` | `query-ref` | Ý định rõ ràng, ID tương quan | ngữ nghĩa chính thức |
| A2A Vòng đời tác vụ | hợp đồng-net + yêu cầu-khi | Vòng đời không đồng bộ, chuyển đổi trạng thái | Đảm bảo tính đầy đủ về hình thức |
| A2A streaming sự kiện | subscribe/notify | Đẩy không đồng bộ | đăng ký nhập vị ngữ |
| Ngữ cảnh chia sẻ CA-MCP | bảng đen (Hayes-Roth 1985) | Bộ nhớ chia sẻ nhiều người ghi | tính nhất quán logic model |
| NLIP | nội dung ngôn ngữ tự nhiên | LLM bản địa | schema |

Đọc từ trên xuống dưới, mô hình là: giữ primitive cấu trúc, bỏ chủ nghĩa hình thức, để LLMs giấy lên trên sự mơ hồ.

## Tự xây dựng

`code/main.py` triển khai trình dịch FIPA-ACL thuần túy. Nó mã hóa và giải mã phong bì ACL chuẩn và hiển thị cách mọi MCP / A2A hình dạng tin nhắn giảm xuống cùng bảy trường. Bản demo:

- Mã hóa năm thông báo kiểu MCP và kiểu A2A dưới dạng FIPA-ACL.
- Giải mã FIPA-ACL trở lại tương đương hiện đại.
- Điều hành một cuộc đàm phán Contract Net đồ chơi giữa một người quản lý và ba nhà thầu bằng cách sử dụng `cfp`, `propose`, `accept-proposal`, `reject-proposal`.

Chạy:

```
python3 code/main.py
```

Đầu ra là một trace song song hiển thị từng thông điệp hiện đại ở cả dạng JSON 2026 và biểu mẫu FIPA-ACL, sau đó là một chuyến đi khứ hồi của giá thầu ròng hợp đồng. Giao thức tương tự primitives tồn tại trong chuyến đi khứ hồi; chỉ có cú pháp khác nhau.

## Ứng dụng

`outputs/skill-fipa-mapper.md` là một skill đọc bất kỳ thông số kỹ thuật giao thức agent nào và tạo ra ánh xạ FIPA-ACL. Sử dụng nó trước khi áp dụng một giao thức mới để trả lời: "Điều này có thực sự mới không, hay nó `inform` với cú pháp JSON?"

## Sản phẩm bàn giao

Không mang FIPA-ACL trở lại. Mang lại danh sách kiểm tra của nó:

- Mục đích primitive (biểu diễn) của mỗi tin nhắn là gì?
- Có mã tương quan cho yêu cầu-phản hồi và hủy bỏ không?
- Có ngôn ngữ nội dung rõ ràng (JSON-RPC, văn bản thuần túy, artifact gõ có cấu trúc) không?
- Các giao thức tương tác là class đầu tiên hay bạn đang triển khai lại contract-net từ đầu?
- Điều gì xảy ra khi hai agents không đồng ý về ý nghĩa nội dung (trôi dạt ngữ nghĩa)?

Ghi lại năm câu hỏi này cho bất kỳ giao thức mới nào trước khi bạn ship nó vào production.

## Bài tập

1. Chạy `code/main.py`. Quan sát mã hóa khứ hồi. Xác định hiệu suất FIPA nào tương ứng với việc tạo tác vụ `tools/call`, `resources/read` và A2A.
2. Mở rộng bản demo ròng hợp đồng với hiệu suất `cancel` cho phép người quản lý rút nhiệm vụ giữa chừng đấu thầu. Trường hợp thất bại nào `cancel` giải quyết mà chỉ thử lại thì không?
3. Đọc Cấu trúc thông báo FIPA ACL (http://www.fipa.org/specs/fipa00037/) phần 4.1–4.3. Chọn một biểu diễn không được đề cập trong bài học này và mô tả tương tự JSON-RPC hiện đại của nó.
4. Đọc Liu và cộng sự, arXiv: 2505.02279. Đối với mỗi MCP, A2A, ACP, ANP, hãy liệt kê các gia đình biểu diễn FIPA mà họ giữ và bỏ đi.
5. Thiết kế một Schema JSON tối thiểu cho lĩnh vực `content` của một `request` biểu diễn trong hệ thống của riêng bạn. Điều đó schema mang lại cho bạn điều gì mà ngôn ngữ tự nhiên thuần túy không có, và nó có giá bao nhiêu?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Hành động lời nói | "Một lời nói làm điều gì đó" | Austin/Searle: lời nói là hành động. Cha mẹ lý thuyết của ACL. |
| FIPA | "Thứ XML cũ đó" | Quỹ IEEE về Agents vật lý thông minh. ACL được tiêu chuẩn hóa vào năm 2000. |
| ACL | "Ngôn ngữ giao tiếp Agent" | Định dạng phong bì của FIPA: biểu diễn + nội dung + siêu dữ liệu. |
| Biểu diễn | "Động từ" | Mục đích class của tin nhắn: `inform`, `request`, `propose`, `cfp`, v.v. |
| KQML | "Tiền thân của FIPA" | Ngôn ngữ truy vấn và thao tác kiến thức (1993). Đơn giản hơn, hẹp hơn. |
| Bản thể học | "Từ vựng chung" | Một định nghĩa chính thức về các khái niệm mà ngôn ngữ nội dung nói đến. |
| SL0 / SL1 | "Ngôn ngữ nội dung FIPA" | Ngôn ngữ ngữ nghĩa cấp 0 và 1 — họ ngôn ngữ nội dung chính thức. |
| Ròng hợp đồng | "Thị trường nhiệm vụ" | Quản lý phát hành cfp; nhà thầu đề xuất; Người quản lý chấp nhận. Giao thức tương tác chuẩn. |
| Giao thức tương tác | "Mẫu thông điệp" | Một chuỗi các biểu diễn với tính đúng đắn đã biết: request-when, subscribe-notify, v.v. |

## Đọc thêm

- [Liu et al. — A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP](https://arxiv.org/html/2505.02279v1) — cuộc khảo sát kinh điển năm 2025 kết nối các thông số kỹ thuật hiện đại với di sản FIPA
- [FIPA ACL Message Structure Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) — định dạng phong bì năm 2000 đã được phê chuẩn
- [FIPA Communicative Act Library Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) — danh mục biểu diễn đầy đủ
- [MCP specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — công cụ sử dụng hiện đại tương đương với `request`/`query-ref`
- [A2A specification](https://a2a-protocol.org/latest/specification/) — tương đương với agent-ngang hàng hiện đại của contract-net và subscribe-notify
