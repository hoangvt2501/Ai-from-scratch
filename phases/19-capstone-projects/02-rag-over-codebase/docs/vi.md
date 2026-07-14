# Capstone 02 - RAG qua cơ sở mã (Tìm kiếm ngữ nghĩa chéo Repo)

> Mọi tổ chức kỹ thuật nghiêm túc vào năm 2026 đều chạy tìm kiếm mã nội bộ hiểu ý nghĩa, không chỉ chuỗi. Sourcegraph Amp, câu trả lời cơ sở mã của Cursor, biểu đồ doanh nghiệp của Augment, repomap của Aider, MCP nội bộ của Pinterest - cùng một hình dạng. Nhập nhiều repos, phân tích cú pháp với người trông cây, nhúng các đoạn chức năng và cấp class, tìm kiếm lai, xếp hạng lại, trả lời bằng trích dẫn. Capstone này yêu cầu bạn xây dựng một mã xử lý 2 triệu dòng mã trên 10 repos và tồn tại sau khi lập chỉ mục lại gia tăng trên mỗi lần đẩy git.

**Loại:** Đá đỉnh
**Ngôn ngữ:** Python (ingestion), TypeScript (API + UI)
**Kiến thức tiên quyết:** Giai đoạn 5 (NLP nền móng), Giai đoạn 7 (transformers), Giai đoạn 11 (LLM kỹ thuật), Giai đoạn 13 (công cụ), Giai đoạn 17 (cơ sở hạ tầng)
**Các giai đoạn thực hiện: **P5 · P7 · P11 · P13 · Trang 17
**Thời lượng:** 30 giờ

## Vấn đề

Đến năm 2026, mọi mã hóa biên giới đều agent ships lớp truy xuất cơ sở mã vì chỉ windows ngữ cảnh không giải quyết được các câu hỏi repo chéo. Bối cảnh 1 triệu token của Claude giúp; Nó không loại bỏ nhu cầu truy xuất xếp hạng. Tìm kiếm cosine ngây thơ trên các khối thô đầu độc kết quả trên mã được tạo, trên bản sao monorepo và trên đuôi dài của các ký hiệu hiếm khi được nhập khẩu. Câu trả lời production là tìm kiếm kết hợp (dày đặc + BM25) trên các khối nhận biết AST với trình xếp hạng lại, được hỗ trợ bởi biểu đồ tham chiếu biểu tượng.

Bạn học điều này bằng cách lập chỉ mục một nhóm thực - không phải một hướng dẫn nào repo - và đo lường MRR@10, độ trung thực của trích dẫn và độ mới gia tăng. Các chế độ lỗi là cơ sở hạ tầng: monorepo 100k tệp, đẩy chỉnh sửa một nửa tệp, truy vấn cần vượt qua bốn repos để trả lời chính xác.

## Khái niệm

Một pipeline nhập nhận biết AST phân tích cú pháp từng tệp bằng trình giữ cây, trích xuất các nút chức năng và class cũng như các đoạn ở ranh giới nút thay vì token windows cố định. Mỗi đoạn có ba biểu diễn: một embedding dày đặc (Voyage-code-3 hoặc nomic-embed-code), các thuật ngữ BM25 thưa thớt và một bản tóm tắt ngôn ngữ tự nhiên ngắn. Bản tóm tắt bổ sung phương thức có thể truy xuất thứ ba - người dùng hỏi "X được ủy quyền như thế nào" và bản tóm tắt đề cập đến "authz", ngay cả khi mã chỉ có `check_permission`.

Truy xuất là kết hợp. Một truy vấn kích hoạt cả tìm kiếm dày đặc và BM25, merges top-k và giao liên minh cho một công cụ xếp hạng lại chéo encoder (Cohere xếp hạng lại-3 hoặc bge-reranker-v2-gemma-2b). Danh sách được xếp hạng lại sẽ được chuyển đến một bộ tổng hợp ngữ cảnh dài (Claude Sonnet 4.7 với bộ nhớ đệm prompt hoặc Llama 3.3 70B tự lưu trữ) với các hướng dẫn để trích dẫn mọi yêu cầu theo tệp và phạm vi dòng. Câu trả lời không có trích dẫn sẽ bị từ chối bởi bộ lọc sau.

Sự mới mẻ gia tăng là vấn đề cơ sở hạ tầng. Git đẩy triggers một diff: tệp nào đã thay đổi, ký hiệu nào đã thay đổi. Chỉ các đoạn bị ảnh hưởng mới được nhúng lại. Các cạnh ký hiệu tệp chéo bị ảnh hưởng (imports, lệnh gọi phương thức) được tính toán lại. Chỉ số vẫn nhất quán mà không cần xử lý lại 2 triệu dòng mỗi commit.

## Kiến trúc

```
git push --> webhook --> ingest worker (LlamaIndex Workflow)
                           |
                           v
             tree-sitter parse + AST chunk
                           |
            +--------------+----------------+
            v              v                v
          dense        BM25 index       summary (LLM)
        (Voyage / bge)  (Tantivy)        (Haiku 4.5)
            |              |                |
            +------> Qdrant / pgvector <----+
                            |
                            v
                      symbol graph (Neo4j / kuzu)
                            |
  query --> LangGraph agent (retrieve -> rerank -> synth)
                            |
                            v
                 Claude Sonnet 4.7 1M context
                            |
                            v
                 answer + file:line citations
```

## Stack

- Phân tích cú pháp: tree-sitter với 17 ngữ pháp ngôn ngữ (Python, TS, Rust, Go, Java, C++, v.v.)
- embeddings mật độ: Voyage-code-3 (được lưu trữ) hoặc nomic-embed-code-v1.5 (tự lưu trữ), dự phòng bge-code-v1
- Chỉ số thưa thớt: Tantivy (Rust) với BM25F, trọng số trường trên tên biểu tượng so với thân
- Vector DB: Qdrant 1.12 với tìm kiếm kết hợp hoặc pgvector + pgvectorscale dành cho các nhóm dưới 50 triệu vectors
- Tóm tắt khối model: Claude Haiku 4.5 hoặc Gemini 2.5 Flash, prompt-cache
- Xếp hạng lại: Cohere rerank-3 hoặc bge-reranker-v2-gemma-2b tự lưu trữ
- Orchestration: Quy trình làm việc LlamaIndex để nhập, LangGraph để agent truy vấn
- Bộ tổng hợp: Claude Sonnet 4.7 (ngữ cảnh 1M) với bộ nhớ đệm prompt
- Biểu đồ biểu tượng: Neo4j (được quản lý) hoặc kuzu (nhúng) cho các cạnh import và gọi
- Observability: Langfuse spans mỗi bước truy xuất + tổng hợp

## Tự xây dựng

1. **Bộ đi bộ nhập.** Lặp lại lịch sử git trên mỗi lần hook đẩy. Thu thập các tệp đã thay đổi. Đối với mỗi tệp, phân tích cú pháp với tree-sitter, hàm trích xuất và các nút class với span nguồn đầy đủ của chúng. Phát ra các bản ghi khối `{repo, path, start_line, end_line, symbol, body}`.

2. **Chunk summarizer.** Batch các đoạn vào các cuộc gọi Haiku 4.5 với bộ nhớ đệm prompt trên phần mở đầu của hệ thống. Prompt: "Tóm tắt chức năng này trong một câu, đặt tên cho hợp đồng công cộng và tác dụng phụ của nó." Tóm tắt cửa hàng cùng với khối.

3. **Embedding pool.** Hai hàng đợi song song: dày đặc (Voyage-code-3 batch 128) và summary (cùng model, nhưng trên chuỗi tóm tắt). Viết vectors cho Qdrant bằng payload `{repo, path, start_line, end_line, symbol, kind}`.

4. **Chỉ số BM25.** Chỉ số Tantivy trọng số trường: trọng lượng tên ký hiệu 4, trọng lượng cơ thể ký hiệu 1, trọng lượng tóm tắt 2. Cho phép truy vấn "tìm hàm có tên X" cùng với "tìm hàm thực hiện X".

5. **Biểu đồ ký hiệu.** Đối với mỗi chunk, ghi các cạnh: imports (tệp này sử dụng ký hiệu Y từ repo Z), gọi (hàm này gọi phương thức M trên class C), kế thừa. Cửa hàng ở kuzu. Được sử dụng tại thời điểm truy vấn để mở rộng truy xuất qua ranh giới repo.

6. **Truy vấn agent.** LangGraph với ba nút. `retrieve` bắn dày đặc + BM25 song song, loại bỏ trùng lặp bằng (repo, đường dẫn, ký hiệu). `rerank` chạy cross-encoder trên top 50 và giữ top-10. `synth` gọi Claude Sonnet 4.7 với các đoạn được xếp hạng lại trong ngữ cảnh, lưu trữ system prompt, yêu cầu trích dẫn file:line.

7. **Thực thi trích dẫn.** Phân tích cú pháp đầu ra model; Bất kỳ thông báo xác nhận quyền sở hữu nào không có neo `(repo/path:start-end)` sẽ bị gắn cờ để yêu cầu lại hoặc bị loại bỏ. Trả về câu trả lời chỉ được trích dẫn cho người dùng.

8. **Chỉ mục lại gia tăng.** Trên mỗi webhook, hãy tính toán chênh lệch mức biểu tượng. Chỉ nhúng lại các đoạn có văn bản thay đổi. Tính toán lại các cạnh biểu tượng cho các đoạn có imports thay đổi. Đo lường: đẩy 50 tệp được lập chỉ mục lại trong vòng chưa đầy 60 giây cho nhóm 2M-LOC.

9. **Eval.** Gắn nhãn 100 câu hỏi chéo repo với câu trả lời bằng vàng. Đo lường MRR@10, nDCG@10, độ trung thực của trích dẫn (phần nhỏ của các tuyên bố có neo có thể xác minh) và độ trễ p50/p99.

## Ứng dụng

```
$ code-rag ask "how is S3 multipart abort wired into our retry budget?"
[retrieve]  12 chunks dense + 7 chunks bm25, 16 unique after dedup
[rerank]    top-5 kept (cohere rerank-3)
[synth]     claude-sonnet-4.7, cache hit rate 68%, 2.1s
answer:
  Multipart aborts are triggered by `AbortMultipartOnFail` in
  services/uploader/retry.go:122-148, which decrements the per-bucket
  retry budget defined in config/budgets.yaml:34-51 ...
  citations: [services/uploader/retry.go:122-148, config/budgets.yaml:34-51,
              libs/s3client/multipart.ts:44-61]
```

## Sản phẩm bàn giao

skill `outputs/skill-codebase-rag.md` có thể phân phối. Với một kho dữ liệu repos, nó sẽ đưa ra pipeline nhập, chỉ mục kết hợp và agent truy vấn, đồng thời trả về câu trả lời được trích dẫn cho bất kỳ câu hỏi nào repo chéo. Bảng đánh giá:

| Trọng lượng | Tiêu chí | Cách đo lường |
| :-: | --- | --- |
| 25 | Chất lượng truy xuất | MRR@10 và nDCG@10 trong một bộ 100 câu hỏi |
| 20 | Trích dẫn trung thành | Phần yêu cầu câu trả lời có thể xác minh file:line anchors |
| 20 | Độ trễ và quy mô | Độ trễ truy vấn p95 ở 10k QPS trên kích thước kho dữ liệu được lập chỉ mục |
| 20 | Độ chính xác của lập chỉ mục gia tăng | Thời gian từ git đẩy đến khi có thể tìm kiếm trên commit 50 tệp |
| 15 | UX và định dạng câu trả lời | Khả năng nhấp trích dẫn, xem trước đoạn trích, khả năng chi trả theo dõi |
| **100** |||

## Bài tập

1. Hoán đổi Voyage-code-3 cho nomic-embed-code tự lưu trữ. Đo đồng bằng MRR@10. Báo cáo liệu khoảng cách có thu hẹp khi bật xếp hạng lại hay không.

2. Chèn 20% mã được tạo (nguyên mẫu do LLM sản xuất) vào kho dữ liệu và đánh giá lại. Quan sát ngộ độc khi thu hồi. Thêm cờ "đã tạo" vào payload và giảm trọng lượng các lần truy cập đó.

3. Benchmark tìm kiếm kết hợp Qdrant so với pgvector + pgvectorscale ở kích thước kho dữ liệu của bạn. Báo cáo p99 ở batch cỡ 1.

4. Thêm kiểm tra trôi dựa trên sampling: hàng tuần, chạy lại đánh giá 100 câu hỏi. Cảnh báo khi MRR@10 giảm > 5%.

5. Mở rộng đến độ phân giải ký hiệu đa ngôn ngữ: một hàm Python gọi dịch vụ Go qua gRPC. Sử dụng biểu đồ biểu tượng để liên kết chúng.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| Phân đoạn nhận biết AST | "Phân tách cấp chức năng" | Cắt mã tại ranh giới nút giữ cây thay vì token windows cố định |
| Tìm kiếm kết hợp | "Dày đặc + thưa thớt" | Chạy BM25 và tìm kiếm vector song song, merge top-k, xếp hạng lại |
| Cross-encoder xếp hạng lại | "Cấp bậc giai đoạn hai" | Model điểm cho từng cặp (truy vấn, ứng viên) với nhau, chính xác hơn cosin |
| Prompt bộ nhớ đệm | "system prompt được lưu trong bộ nhớ đệm" | 2026 Claude / OpenAI feature giảm giá tokens tiền tố lặp lại lên đến 90% |
| Biểu đồ biểu tượng | "Đồ thị mã" | Các cạnh cho imports, cuộc gọi, kế thừa trên các tệp và repos |
| Trích dẫn trung thành | "Tỷ lệ trả lời có cơ sở" | Phần xác nhận quyền sở hữu mà người dùng có thể xác minh bằng cách nhấp vào điểm neo và đọc span được tham chiếu |
| Lập chỉ mục lại gia tăng | "Thời gian nhấn để tìm kiếm" | Đồng hồ treo tường từ git đẩy đến các ký hiệu đã thay đổi có thể truy vấn được |

## Đọc thêm

- [Sourcegraph Amp](https://ampcode.com) — production trí thông minh mã repo chéo
- [Sourcegraph Cody RAG architecture](https://sourcegraph.com/blog/how-cody-understands-your-codebase) — tài liệu tham khảo chuyên sâu cho capstone này
- [Aider repo-map](https://aider.chat/docs/repomap.html) - người trông cây được xếp hạng repo view
- [Augment Code enterprise graph](https://www.augmentcode.com) — biểu đồ biểu tượng thương mại RAG
- [Qdrant hybrid search docs](https://qdrant.tech/documentation/concepts/hybrid-queries/) — triển khai tham chiếu
- [Voyage AI code embeddings](https://docs.voyageai.com/docs/embeddings) — Chi tiết về Voyage-code-3
- [Cohere rerank-3](https://docs.cohere.com/reference/rerank) — tham chiếu chéo encoder
- [Pinterest MCP internal search](https://medium.com/pinterest-engineering) — tham chiếu nền tảng nội bộ
