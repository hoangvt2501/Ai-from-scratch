# So sánh các chiến lược phân đoạn

> Chunking quyết định những gì tha mồi của bạn có thể xuất hiện. Sai ranh giới và không có embedding model, không có trình xếp hạng lại, không LLM nào có thể sửa chữa thiệt hại ở hạ lưu.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 11 bài 04 (embeddings), 06 (RAG), 07 (RAG nâng cao); Nền tảng Giai đoạn 19 Theo dõi B (bài 20-29)
**Thời lượng:** ~90 phút

## Mục tiêu học tập
- Thực hiện năm chiến lược phân đoạn từ đầu: cửa sổ cố định, câu, phân tách đệ quy, phân cụm ngữ nghĩa và tiêu đề đánh dấu cấu trúc.
- Đo lường recall@k trên kho dữ liệu cố định với spans trả lời được dán nhãn vàng và giải thích lý do tại sao một chiến lược chiến thắng trên văn xuôi và một chiến lược khác thắng trên các tài liệu kỹ thuật.
- Đọc phân phối độ dài khối và nhận ra các chế độ thất bại mà mỗi chiến lược đưa vào: câu mồ côi, cắt ký hiệu giữa, đoạn chỉ tiêu đề, trôi ngữ nghĩa.
- Chọn mặc định cho một kho dữ liệu mới mà không cần chạy benchmark bằng cách kiểm tra ba thuộc tính: loại tài liệu, độ dài đoạn văn trung bình và liệu định dạng có cấu trúc rõ ràng hay không.

## Vấn đề

Mỗi RAG pipeline bắt đầu bằng cách cắt các tài liệu nguồn thành các mảnh đủ nhỏ để một embedding model phù hợp với chúng và đủ lớn để mỗi phần mang một ý tưởng khép kín. Việc lựa chọn nơi cắt không phải là một hyperparameter. Đó là giới hạn trên về những gì chó săn có thể trả lại.

Một truy vấn hỏi "ngưỡng hủy bỏ ngân sách trông như thế nào" chỉ có thể thành công nếu có thể đạt được phần giữ ngưỡng hủy bỏ. Nếu bộ chia cửa sổ cố định cắt giá trị ngưỡng khỏi ngữ cảnh xung quanh, embedding chuyển sang một cụm khác, điểm BM25 giảm, người xếp hạng lại nhìn thấy nhiễu và câu trả lời mà LLM tạo ra là sai. Bài báo năm 2024 "LongRAG: Tăng cường thế hệ tăng cường truy xuất với LLMs ngữ cảnh dài" đã đo lường sự thay đổi tuyệt đối 35% trong recall truy xuất hoàn toàn từ lựa chọn phân đoạn. Công việc tiếp theo vào năm 2025 về tiêu đề chunk theo ngữ cảnh đã thu hẹp khoảng cách nhưng không thu hẹp được khoảng cách.

Bài học này xây dựng năm chiến lược song song, chạy chúng với một kho dữ liệu cố định với spans trả lời được dán nhãn vàng và cho phép bạn tự đọc các con số recall.

## Khái niệm

```mermaid
flowchart LR
  Doc[Source Document] --> S1[Fixed Window]
  Doc --> S2[Sentence]
  Doc --> S3[Recursive Split]
  Doc --> S4[Semantic Cluster]
  Doc --> S5[Structural Markdown]
  S1 --> Chunks1[Chunks]
  S2 --> Chunks2[Chunks]
  S3 --> Chunks3[Chunks]
  S4 --> Chunks4[Chunks]
  S5 --> Chunks5[Chunks]
  Chunks1 --> Index[Embedding Index]
  Chunks2 --> Index
  Chunks3 --> Index
  Chunks4 --> Index
  Chunks5 --> Index
  Index --> Eval[Recall@k vs Gold Spans]
```

### Cửa sổ cố định

Đường cơ sở vũ phu. Cắt mọi N ký tự. Tùy chọn chồng lên nhau để một câu cắt ở vị trí N xuất hiện toàn bộ bên trong đoạn bắt đầu từ vị trí N - chồng chéo. Nhanh, xác định, khủng khiếp ở ranh giới. Sử dụng nó như một điều khiển, không phải mặc định.

### Câu

Phân chia ranh giới câu bằng một regex hoặc một máy trạng thái đơn giản. Đóng gói một hoặc nhiều câu thành một phần với ngân sách nhân vật mục tiêu. Ngừng cắt giữa từ. Vẫn cắt giữa đoạn và giữa phần. Mặc định trong nhiều RAG pipelines đầu và là một lựa chọn hợp lý cho văn xuôi không có cấu trúc nào khác.

### Phân tách đệ quy

Chiến lược phân cấp được phổ biến bởi các thư viện thời đại 2023. Cố gắng tách trên dấu phân cách mạnh nhất trước (dòng mới kép, đoạn văn), quay trở lại dấu phân cách tiếp theo (dòng mới đơn), sau đó đến câu, sau đó đến các ký tự. Đệ quy kết thúc khi chunk phù hợp với ngân sách. Mạnh về các tài liệu có cấu trúc không nhất quán vì nó thích ứng với từng khu vực.

### Phân cụm ngữ nghĩa

Nhúng từng câu. Cụm các câu liền kề có chung một chủ đề centroid. Cắt bất cứ khi nào độ tương tự chạy với tâm giảm xuống dưới ngưỡng. Ranh giới phản ánh ý nghĩa, không phải nhân vật. Xây dựng chậm hơn và phụ thuộc vào embedding model, nhưng có khả năng chống lại các tài liệu chuyển đổi chủ đề bên trong đoạn văn.

### Tiêu đề đánh dấu cấu trúc

Đối với các tài liệu có cấu trúc rõ ràng (đánh dấu, reStructuredText, các phần được đánh số kiểu RFC), hãy cắt ở ranh giới tiêu đề. Mỗi phần trở thành tiêu đề cộng với mọi thứ bên dưới nó xuống tiêu đề tiếp theo ở cùng cấp độ hoặc cao hơn. Các đoạn nhỏ nhất cho mỗi chủ đề, nhưng chỉ có sẵn khi kho dữ liệu được hình thành tốt.

### Cách recall@k đo lường lựa chọn ranh giới

Một truy vấn được gắn nhãn vàng mang các ký tự bù chính xác của câu trả lời span bên trong tài liệu nguồn. Sau khi cắt nhỏ, bạn hỏi: có bất kỳ khối top-k nào mà săn trả lại trùng với span vàng không? Nếu có, recall@k cho truy vấn đó là 1. Nếu không, nó là 0. Trung bình trên bộ truy vấn. Chạy cùng một đánh giá cho từng chiến lược và chênh lệch cho bạn biết ranh giới nào policy tồn tại trong kho dữ liệu bạn có.

## Tự xây dựng

`code/main.py` thực hiện:

- `fixed_window(text, size, overlap)` - đường cơ sở.
- `sentence_chunks(text, target)` - trình đóng gói câu đơn giản.
- `recursive_split(text, separators, target)` - đệ quy phân cấp.
- `semantic_chunks(text, similarity_threshold)` - cụm dựa trên trung tâm trên một embedding mô phỏng xác định.
- `structural_markdown(text)` - bộ chia nhận biết tiêu đề.
- `mock_embed(text, dim)` - một embedding dựa trên hàm băm để vòng lặp chạy ngoại tuyến.
- `DenseIndex` - hình dạng tương tự được sử dụng trong bài học truy xuất kết hợp của Giai đoạn 19 Track B.
- `eval_recall(strategy, corpus, queries, k)` - vòng lặp so sánh.
- Một `main()` chạy mọi chiến lược trên kho dữ liệu cố định và in một bảng recall@k.

Chạy nó:

```bash
python3 code/main.py
```

Đầu ra là một bảng nhỏ với một hàng trên mỗi chiến lược và một cột trên mỗi k. Câu thua trên vật cố định có cấu trúc. Đánh dấu cấu trúc giành chiến thắng trong trận đấu giảm giá. Đệ quy giữ riêng trên vật cố định hỗn hợp vì đệ quy thích ứng. Phân cụm ngữ nghĩa chiến thắng trên cố định văn xuôi khi không có tín hiệu cấu trúc hữu ích.

## Chế độ thất bại bảng sẽ không ẩn

**Câu mồ côi.** Đóng gói câu tạo ra các đoạn bỏ lỡ câu chủ đề. Sau đó, embedding trỏ vào cụm sai.

**Cắt ký hiệu giữa.** Cửa sổ cố định bên trong mã hoặc YAML sẽ chia đôi mã định danh. Hai nửa nhúng vào nhiễu.

**Các đoạn chỉ dành cho tiêu đề.** Đánh dấu cấu trúc phát ra một đoạn không chứa gì ngoài `## Title`. Lọc chúng ra hoặc đính kèm đoạn đầu tiên của đoạn tiếp theo.

**Trôi dạt ngữ nghĩa.** Phân cụm ngữ nghĩa bị cắt giảm khi kho dữ liệu đồng nhất về chủ đề. Một đoạn 5000 ký tự gói gọn nhiều câu trả lời cụ thể vào một embedding khuếch tán. Kết hợp ngữ nghĩa với giới hạn ký tự cứng.

**Cũ embeddings.** Phân cụm ngữ nghĩa sử dụng một embedding model. Nếu bạn thay đổi model, bạn cũng thay đổi các khối. Ghim đoạn model riêng biệt với model truy xuất hoặc xây dựng lại chỉ mục với nhau.

## Chọn mặc định mà không cần chạy benchmark

Ba thuộc tính quyết định chunker mặc định cho một kho dữ liệu mới.

| Bất động sản | Giá trị | Mặc định |
|----------|-------|---------|
| Loại tài liệu | Văn xuôi không có cấu trúc | Phân tách đệ quy, mục tiêu 800 |
| Loại tài liệu | Tài liệu Markdown / RFC / API | Giảm giá cấu trúc |
| Loại tài liệu | Mã | Nhận biết AST (ngoài phạm vi; xem Giai đoạn 19 bài 02) |
| Độ dài đoạn văn | Chủ đề dài, đơn lẻ | Câu, mục tiêu 500 |
| Độ dài đoạn văn | Chủ đề ngắn, hỗn hợp | Ngữ nghĩa, ngưỡng 0,6 |

Khi nghi ngờ, hãy chọn phân tách đệ quy. Đây là đường cơ sở chiến lược đơn mạnh nhất.

## Ứng dụng

Production mẫu:

- Chạy đánh giá trước khi bạn ship một pipeline mới; Đừng tin tưởng vào chiến lược mà thư viện của bạn mặc định.
- Chạy lại đánh giá bất cứ khi nào bạn thay đổi embedding model hoặc hỗn hợp kho dữ liệu; Người chiến thắng phụ thuộc vào kho dữ liệu.
- Duy trì tên chiến lược trong siêu dữ liệu của mỗi phần để bạn có thể phân bổ hồi quy sau này.

## Sản phẩm bàn giao

Hệ thống RAG đầu cuối của Track F trong bài 69 sử dụng chunker được chọn ở đây làm giai đoạn đầu tiên. harness đánh giá trong bài 68 đọc recall@k từ cùng một hình dạng mà `eval_recall` trả về trong bài học này. Chọn chiến lược giành chiến thắng trên kho dữ liệu của bạn và cung cấp nó về phía trước.

## Bài tập

1. Thêm chiến lược thứ sáu: token cửa sổ sử dụng `tiktoken` thay vì số ký tự. So sánh với cửa sổ cố định trên cùng một vật cố định.
2. Chèn 30 phần trăm khối mã vào cố định văn xuôi. Chạy lại bảng. Giải thích lý do tại sao mọi chiến lược ngoại trừ giảm giá cấu trúc đều mất recall.
3. Thay thế embedding xác định bằng  từ nhà cung cấp thực của dự án của bạn. Đo lường phân cụm ngữ nghĩa recall delta. Báo cáo mức chênh lệch giữa các chiến lược mở rộng hay thu hẹp.
4. Thêm một trường `summary` cho mỗi khối: mô tả trung tâm một câu. Chạy lại đánh giá với bản tóm tắt được thêm vào phần thân khối. Đo độ nâng recall.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| Recall@k | "Chúng ta đã nhận được phần phù hợp chưa?" | Phần truy vấn trong đó bất kỳ đoạn top-k nào chồng lên câu trả lời vàng span |
| Chồng chéo khối | "Cửa sổ trượt" | Bao gồm lại N ký tự cuối cùng của đoạn trước trong đoạn tiếp theo |
| Bộ chia cấu trúc | "Các phần nhận biết tiêu đề" | Cắt ở ranh giới H1/H2/H3; Văn bản tiêu đề là một phần của đoạn |
| Chunk ngữ nghĩa | "Các đoạn nhận biết chủ đề" | Nhúng câu, phân cụm theo sự tương đồng trung tâm, cắt khi trôi dạt |
| Trôi dạt tâm | "Thay đổi chủ đề" | Sự tương đồng cosin giữa giá trị trung bình đang chạy và câu tiếp theo giảm qua ngưỡng |

## Đọc thêm

- [LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs (arXiv 2406.15319)](https://arxiv.org/abs/2406.15319)
- [Anthropic, Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [LlamaIndex, Chunking strategies for production RAG](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/)
- Giai đoạn 11 bài 06 - RAG nguyên tắc cơ bản
- Giai đoạn 11 bài 07 - RAG nâng cao
- Giai đoạn 19 bài 65 - truy xuất kết hợp xếp hạng các khối được tạo ra ở đây
- Giai đoạn 19 bài 68 - harness đánh giá chấm điểm lựa chọn chiến lược trong production
