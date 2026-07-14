# Bộ nhớ lai: Vector + Đồ thị + KV (Mem0)

> Mem0 (Chhikara et al., 2025) coi bộ nhớ là ba kho lưu trữ song song - vector cho sự tương đồng về ngữ nghĩa, KV để tra cứu dữ kiện nhanh, đồ thị cho suy luận mối quan hệ thực thể. Một lớp tính điểm hợp nhất ba khi truy xuất. Đây là tiêu chuẩn production 2026 cho bộ nhớ ngoài.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 14 · 07 (MemGPT), Giai đoạn 14 · 08 (Khối Letta)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Giải thích lý do tại sao một kho lưu trữ duy nhất (chỉ vector, chỉ đồ thị, chỉ KV) không đủ cho bộ nhớ agent.
- Kể tên ba cửa hàng song song của Mem0 và những gì mỗi cửa hàng tối ưu hóa.
- Mô tả tính điểm hợp nhất của Mem0 - mức độ liên quan, tầm quan trọng, thời gian gần đây - và lý do tại sao nó là một tổng có trọng số, không phải là một hệ thống phân cấp.
- Triển khai bộ nhớ ba kho đồ chơi trong stdlib với một `add()` ghi cho cả ba và một `search()` hợp nhất kết quả.

## Vấn đề

Một cửa hàng bị sai đối với một trong ba classes truy vấn:

- **Tương đồng ngữ nghĩa **- "chúng ta đã thảo luận gì về agent trôi dạt vào tuần trước?" Vector thắng; KV và đồ thị trượt.
- **Tra cứu sự thật** — "số điện thoại của người dùng là gì?" KV thắng; vector là lãng phí, đồ thị là quá mức cần thiết.
- **Lý luận mối quan hệ** — "khách hàng nào chia sẻ cùng một thực thể thanh toán?" Đồ thị thắng; vector và KV không thể trả lời.

Production agents phát hành cả ba trong một session. Một bộ nhớ lưu trữ duy nhất luôn sai đối với hai trong số họ. Đóng góp của Mem0 là nối dây cả ba phía sau một bề mặt `add`/`search` duy nhất với một chức năng tính điểm hợp nhất chúng.

## Khái niệm

### Ba cửa hàng song song

Mem0 (arXiv:2504.19413, tháng 4 năm 2025) vào ngày `add(text, user_id, metadata)`:

1. Trích xuất các dữ kiện ứng cử viên từ văn bản (một bước theo hướng LLM).
2. Ghi từng dữ kiện vào kho lưu trữ vector (embedding) để tìm kiếm ngữ nghĩa.
3. Ghi từng dữ kiện vào kho KV được khóa trên (user_id, fact_type, thực thể) để tra cứu O (1).
4. Ghi từng dữ kiện vào kho đồ thị (Mem0g) dưới dạng các cạnh được gõ cho các truy vấn mối quan hệ.

Trên `search(query, user_id)`:

1. Cửa hàng Vector trả về top-k bằng embedding cosine.
2. KV store trả về các lần truy cập trực tiếp được khóa trên truy vấn (user_id, loại, thực thể).
3. Kho đồ thị trả về biểu đồ con có thể truy cập được từ các thực thể truy vấn.
4. Một lớp tính điểm hợp nhất cả ba.

### Chấm điểm hợp nhất

```
score = w_relevance * relevance(q, record)
      + w_importance * importance(record)
      + w_recency * recency(record)
```

- **Mức độ liên quan** — vector cosin, KV khớp chính xác, trọng lượng đường dẫn đồ thị.
- **Tầm quan trọng** — được gắn thẻ tại thời điểm viết hoặc đã học (một số thông tin quan trọng hơn: tên, ID, policies).
- **Gần đây** — phân rã theo cấp số nhân theo thời gian kể từ lần ghi hoặc đọc cuối cùng.

Trọng lượng được điều chỉnh cho mỗi sản phẩm. `w_recency` agents trò chuyện cao hơn; `w_importance` cao hơn cho agents tuân thủ; `w_relevance` cao hơn cho agents truy xuất.

### Mem0g và lý luận thời gian

Mem0g thêm một máy dò xung đột. Khi một sự kiện mới mâu thuẫn với một cạnh hiện có, cạnh hiện có được đánh dấu là không hợp lệ nhưng không bị xóa. Các truy vấn thời gian ("thành phố của người dùng vào tháng Ba là gì?") đi qua biểu đồ con hợp lệ tại thời điểm.

Đây là hành vi cấp tuân thủ mà mẫu vô hiệu hóa của Letta khái quát hóa.

### Benchmark số

Báo cáo của Mem0 (2025):

- **LoCoMo** (bộ nhớ hội thoại dạng dài): 91.6
- **LongMemEval** (bộ nhớ theo từng đợt đường chân trời dài): 93,4
- **BEAM 1M** (benchmark bộ nhớ 1M-token): 64.1

Đường cơ sở so sánh (128k LLM ngữ cảnh đầy đủ, cửa hàng vector phẳng, KV phẳng) đều mất 10+ điểm. Chỉ riêng Benchmarks không biện minh cho sự lựa chọn - hình dạng hoạt động thì có - nhưng các con số cho thấy thiết kế nhiệt hạch không phải là một lỗi làm tròn.

### Phân loại phạm vi

Mem0 chia bộ nhớ theo phạm vi:

- **Bộ nhớ người dùng** — tồn tại trên sessions, được khóa trên `user_id`.
- **Session bộ nhớ **- tồn tại trong một thread.
- **Agent bộ nhớ** — trạng thái phiên bản trên mỗi agent.

Mỗi lần ghi chọn một phạm vi. Truy xuất có thể truy vấn trên các phạm vi với trọng số trên mỗi phạm vi. Trộn lẫn các phạm vi mà không suy nghĩ là cách bạn nhận được sự cố "trợ lý đã nói với Alice về dự án của Bob".

### Mô hình này sai ở đâu

- **Embedding trôi dạt.** Vector kết quả trông đúng trên hàng trăm truy vấn đầu tiên sẽ giảm xuống cấp khi kho dữ liệu phát triển. Thêm định kỳ embedding lại các bản ghi được sử dụng nhiều nhất.
- **KV schema creep.** `(user_id, type, entity)` trông đơn giản cho đến khi mọi đội thêm `type` của riêng họ. Kiểm tra loại được đặt hàng quý.
- **Bùng nổ đồ thị.** Một trình trích xuất nhiễu thêm 50 cạnh cho mỗi tin nhắn. Biểu đồ giới hạn ghi trên mỗi `add` cuộc gọi; Bỏ các cạnh có độ tin cậy thấp.

## Tự xây dựng

`code/main.py` triển khai mô hình ba cửa hàng trong stdlib:

- `VectorStore` - sự giống nhau ngây thơ token chồng chéo như một người thay thế embedding.
- `KVStore` - Dict được gõ vào `(user_id, fact_type, entity)`.
- `GraphStore` — các cạnh được gõ (chủ ngữ, quan hệ, đối tượng, hợp lệ).
- `Mem0` — mặt tiền cấp cao nhất với `add()`, `search()`, chấm điểm tổng hợp và truy xuất nhận biết phạm vi.
- Một trace hoạt động trên một cuộc trò chuyện nhiều người dùng, nhiều session.

Chạy nó:

```
python3 code/main.py
```

Đầu ra hiển thị ba đường dẫn recall riêng biệt cộng với top-k hợp nhất. Lật trọng số tính điểm ở đầu `main()` và xem thứ hạng thay đổi.

## Ứng dụng

- **Mem0 (Apache 2.0)** — sẵn sàng production. Tự lưu trữ với Postgres + Qdrant + Neo4j hoặc sử dụng cloud được quản lý.
- **Letta** — core/recall/archival ba tầng; Mang theo vector của riêng bạn và biểu đồ phụ trợ.
- **Zep** — giải pháp thay thế thương mại với KG thời gian và trích xuất thực tế.
- **Bản dựng tùy chỉnh** — khi bạn cần kiểm soát chính xác bộ trích xuất (tuân thủ) hoặc trọng số kết hợp (giọng nói agents khi thời gian gần đây chiếm ưu thế).

## Sản phẩm bàn giao

`outputs/skill-hybrid-memory.md` tạo ra một giàn giáo bộ nhớ ba cửa hàng với một bộ chấm điểm nhiệt hạch, phân loại phạm vi và vô hiệu hóa thời gian được kết nối.

## Bài tập

1. Thay thế đồ chơi vector sự giống nhau bằng một embedding model thật (câu-transformers, Ollama, OpenAI embeddings). Đo lường recall@10 trên một cuộc trò chuyện dài tổng hợp. Bảng xếp hạng có trôi dạt hơn 1000 bài viết không?
2. Thêm truy vấn thời gian: `search(query, as_of=timestamp)`. Chỉ trả lại các bản ghi hợp lệ tại hoặc trước thời điểm đó. Cửa hàng nào cần công việc nhất?
3. Triển khai trình phát hiện xung đột: nếu một sự kiện đến mâu thuẫn với cạnh đồ thị, hãy vô hiệu hóa cạnh cũ và ghi nhật ký cả hai. Thử nghiệm về "người dùng sống ở Berlin" - > "người dùng sống ở Lisbon".
4. Chuyển trình ghi điểm hợp nhất để bao gồm thứ nguyên `user_feedback` (giơ ngón tay cái lên trên các bản ghi được truy xuất). Làm thế nào để bạn ngăn chặn trò chơi (agent chỉ trả về các bản ghi mà nó đã thích)?
5. Đọc tài liệu Mem0 (`docs.mem0.ai`). Chuyển đồ chơi sang `mem0` cuộc gọi của khách hàng. So sánh chất lượng truy xuất trên cùng 20 truy vấn thử nghiệm.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Bộ nhớ kết hợp | "Vector cộng với đồ thị cộng với KV" | Ba cửa hàng được viết song song, hợp nhất khi truy xuất |
| Trích xuất sự thật | "Nhập bộ nhớ" | LLM bước chia văn bản thành các bộ (thực thể, quan hệ, thực tế) |
| Chấm điểm hợp nhất | "Xếp hạng mức độ liên quan" | Tổng trọng số về mức độ liên quan, tầm quan trọng, thời gian gần đây |
| Phạm vi | "Không gian tên bộ nhớ" | user / session / agent — xác định ai xem gì |
| Nhớ0g | "Biểu đồ bộ nhớ" | Các cạnh được nhập với giá trị thời gian cho các truy vấn mối quan hệ |
| Vô hiệu hóa tạm thời | "Xóa mềm" | Đánh dấu các cạnh mâu thuẫn không hợp lệ; Không bao giờ xóa |
| Embedding trôi dạt | "Thối rữa truy xuất" | Vector chất lượng giảm sút khi kho dữ liệu phát triển; nhúng lại định kỳ |

## Đọc thêm

- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) — giấy gốc
- [Mem0 docs](https://docs.mem0.ai/platform/overview) - production API, SDKs, quản lý cloud
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — tiền thân ngữ cảnh ảo
- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) - thiết kế anh em ba tầng
