# RAG đa phương thức và truy xuất đa phương thức

> Tài liệu gốc tầm nhìn RAG là một lát cắt. Production RAG đa phương thức mở rộng hơn - truy xuất văn bản, hình ảnh, âm thanh và video cho các quy trình làm việc như lập kế hoạch chuyến đi ("tìm cho tôi một bữa nửa buổi thuần chay yên tĩnh với ánh sáng tự nhiên"), phân loại y tế ("chấn thương nào phù hợp với bức ảnh này + những ghi chú này"), thương mại điện tử ("trang phục tương tự như ảnh tự sướng này, theo kích thước của tôi") và dịch vụ hiện trường ("chẩn đoán âm thanh động cơ này cộng với ảnh của bộ phận"). Ba cuộc khảo sát năm 2025 - Abootorabi và cộng sự, Mei và cộng sự, Zhao và cộng sự - đã hệ thống hóa các vấn đề phụ: truy xuất đa phương thức, hợp nhất truy xuất, grounding thế hệ, đánh giá đa phương thức. Bài học này đọc các cuộc khảo sát và thiết kế một production pipeline.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib, cross-modal retriever with fusion + grounded generator)
**Kiến thức tiên quyết:** Giai đoạn 12 · 23 (ColPali), Giai đoạn 11 (RAG thông tin cơ bản)
**Thời lượng:** ~180 phút

## Mục tiêu học tập

- Thiết kế truy xuất đa phương thức: văn bản → hình ảnh, hình ảnh → văn bản, âm thanh → video, v.v.
- So sánh ba chiến lược hợp nhất: hợp nhất điểm, hợp nhất dựa trên attention MoE hợp nhất.
- Giải thích thế hệ grounding: "trích dẫn nguồn của bạn" trông như thế nào khi nguồn là sự kết hợp của các phương thức.
- Kể tên ba cuộc khảo sát RAG đa phương thức kinh điển vào năm 2025 và phân loại vấn đề phụ của chúng.

## Vấn đề

Phương thức đơn RAG là một mẫu đã được giải quyết: nhúng truy vấn, nhúng các khối, truy xuất, nhồi vào LLM. RAG đa phương thức yêu cầu:

1. Nhiều đầu truy xuất (mỗi phương thức cần embeddings trong một không gian tương thích).
2. Kết hợp kết quả truy xuất giữa các phương thức.
3. Thế hệ grounding trích dẫn các nguồn trên các phương thức.
4. Các chỉ số đánh giá bao gồm tín hiệu phương thức chéo.

Các cuộc khảo sát năm 2025 đều đi đến cùng một phân loại.

## Khái niệm

### Truy xuất đa phương thức

Truy xuất các tài liệu của phương thức B được đưa ra một truy vấn về phương thức A. Ba mẫu:

1. Không gian embedding chung. CLIP và CLAP tạo ra embeddings văn bản + hình ảnh / văn bản + âm thanh trong một không gian chung. Sự tương đồng cosin giữa các phương thức hoạt động trực tiếp. Giới hạn cho các cặp được huấn luyện CLIP.

2. Mỗi phương thức encoder + dịch. encoder văn bản + encoder hình ảnh + một mô-đun dịch nhỏ ánh xạ giữa các khoảng trắng. Sen2Sen của Gupta và cộng sự và các thiết kế năm 2024 khác. Linh hoạt nhưng tăng thêm độ phức tạp.

3. VLM như encoder. Sử dụng trạng thái ẩn của VLM làm biểu diễn truy xuất. Bất kỳ phương thức nào mà VLM hỗ trợ đều hoạt động. Chất lượng cao hơn, đắt hơn.

Lựa chọn: CLIP / SigLIP 2 cho văn bản + hình ảnh; CLAP cho văn bản + âm thanh; VLM-hidden-states cho đa phương thức ở chất lượng biên giới.

### Chiến lược hợp nhất

Bạn đã lấy được 10 kết quả: 5 hình ảnh, 3 đoạn văn bản, 2 clip âm thanh. Bạn merge như thế nào?

Hợp nhất điểm số (rẻ nhất). Mỗi phương thức có người truy xuất riêng, mỗi phương thức trả về điểm số. Chuẩn hóa điểm số trong phương thức sau đó tổng. Đơn giản, thường hoạt động.

Hợp nhất dựa trên Attention. Nối tất cả các mục đã truy xuất, để một mạng nhỏ attention trọng lượng chúng. Cần training.

MoE hợp nhất. Cổng các tuyến mạng lưới cho các chuyên gia theo phương thức cụ thể. Các loại truy vấn khác nhau định tuyến khác nhau - một câu hỏi trực quan có trọng số hình ảnh cao hơn.

Production mặc định: Hợp nhất điểm với một chút bias về phương thức chủ đạo của truy vấn. Nâng cấp lên MoE nếu A/B hiển thị chiến thắng rõ ràng trên miền của bạn.

### Thế hệ grounding

Người LLM nên trích dẫn mục nào được truy xuất đã thúc đẩy mỗi tuyên bố. Đối với đa phương thức:

- Nguồn văn bản: trích dẫn tiêu chuẩn `[1]`.
- Nguồn ảnh: `[img 3]` với chú thích ngắn.
- Âm thanh: `[audio 2 at 0:34]`.

Huấn luyện trình tạo với dữ liệu nhận biết grounding: mỗi xác nhận quyền sở hữu trong mục tiêu training được gắn thẻ bằng chỉ mục nguồn. Ở inference, model tự nhiên phát ra các trích dẫn.

### Các cuộc khảo sát năm 2025

Abootorabi et al. (arXiv:2502.08826, "Ask in Any Modality"): phân loại cho RAG đa phương thức. Bao gồm truy xuất, hợp nhất, thế hệ. Phạm vi phủ sóng rộng nhất.

Mei et al. (arXiv: 2504.08748, "Khảo sát RAG đa phương thức"): tập trung vào các chế độ benchmarks và lỗi nhiệm vụ phụ. Hữu ích cho thiết kế đánh giá.

Zhao và cộng sự (arXiv: 2503.18016): khảo sát tập trung vào tầm nhìn. Mạnh mẽ về công việc gia đình ColPali.

Đọc cả ba cung cấp cho bạn tình trạng nghệ thuật vào mùa xuân năm 2025. Hầu hết các vấn đề phụ vẫn còn mở.

### MuRAG — bài báo nền tảng

MuRAG (Chen và cộng sự, 2022) là RAG đa phương thức đầu tiên. Truy xuất hình ảnh + văn bản từ KB đa phương thức, tạo câu trả lời. Cho thấy tính khả thi trước làn sóng VLM. Các hệ thống hiện đại (REACT, VisRAG, M3DocRAG) được xây dựng trên đó.

### Một ví dụ production về kế hoạch chuyến đi

Truy vấn: "tìm cho tôi một bữa nửa buổi thuần chay yên tĩnh với ánh sáng tự nhiên."

Pipeline:

1. Phân tách truy vấn. "yên tĩnh" → audio/review từ khóa; "Bữa nửa buổi thuần chay" → mục thực đơn; "ánh sáng tự nhiên" → feature hình ảnh.
2. Truy xuất theo phương thức:
   - Truy xuất văn bản trên các bài đánh giá: "bữa nửa buổi thuần chay, bầu không khí yên tĩnh."
   - Lấy hình ảnh trên ảnh nhà hàng: "ánh sáng tự nhiên, thoáng mát."
   - Truy xuất âm thanh trên các clip âm thanh xung quanh: "decibel thấp, không có nhạc".
3. Điểm cầu chì. Mỗi nhà hàng có một điểm tổng hợp.
4. Top-k nhà hàng → VLM trình tạo với tất cả bằng chứng → trả lời bằng trích dẫn.

Điều này vượt xa RAG văn bản. Mỗi phương thức thêm tín hiệu mà chỉ riêng văn bản đã bỏ lỡ.

### Agentic RAG đa phương thức

Multi-hop: nếu lần truy xuất đầu tiên không trả về câu trả lời có độ tin cậy cao, LLM sẽ định dạng lại và truy xuất lại. Agentic RAG mẫu từ Giai đoạn 14 áp dụng tại đây. Ví dụ:

- Truy xuất top 10 ban đầu → LLM yêu cầu "quá ồn, lọc <40 dB" → truy xuất lại.
- Truy xuất hình ảnh → LLM thấy một người có menu → truy xuất văn bản menu → câu trả lời.

Tăng độ phức tạp nhưng xử lý các truy vấn mà truy xuất một lần không thể.

### Đánh giá

Đánh giá đa phương thức vẫn còn non nớt. Các proxy phổ biến:

- Recall@k cho mỗi phương thức.
- Hợp nhất top-k accuracy.
- Sự hài lòng từ đầu đến cuối do con người đánh giá.
- Nhiệm vụ cụ thể (đăng ký đã hoàn thành, mua hàng đã thực hiện).

Không có tiêu chuẩn benchmark spans tất cả các phương thức. Hầu hết các bài báo đánh giá về các nhiệm vụ cụ thể theo miền.

## Ứng dụng

`code/main.py`:

- Ba săn giả (văn bản, hình ảnh, âm thanh) hoạt động trên một kho nhà hàng chung.
- Hợp nhất điểm số kết hợp điểm phương thức với trọng số có thể định cấu hình.
- Một sơ khai trình tạo phát ra câu trả lời cuối cùng với các trích dẫn.
- Một vòng lặp agentic đơn giản để định hình lại truy vấn nếu độ tin cậy thấp.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-multimodal-rag-designer.md`. Với thông số kỹ thuật sản phẩm với luồng truy vấn đa phương thức, thiết kế retriever, fusion, generator và đánh giá.

## Bài tập

1. Đề xuất RAG đa phương thức phân loại y tế: truy vấn = ảnh chấn thương + triệu chứng văn bản. Phương thức nào truy xuất từ KB nào?

2. Hợp nhất điểm số là một tổng có trọng số đơn giản. Nó có chế độ thất bại nào mà MoE nhiệt hạch tránh được?

3. Đọc phân loại của Abootorabi và cộng sự (Phần 3). Ba bài toán phụ chuẩn là gì và làm thế nào để chúng ánh xạ đến sản phẩm bạn đã chọn?

4. Thiết kế thông số kỹ thuật cho RAG đa phương thức lập kế hoạch chuyến đi. Những chỉ số nào bao gồm recall hình ảnh, recall âm thanh và độ chính xác tổng hợp?

5. Agentic RAG nhiều bước có thuế độ trễ cho mỗi chuyến khứ hồi. Độ khó truy vấn nào mà accuracy đạt được biện minh cho độ trễ?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| Truy xuất đa phương thức | "Truy vấn một phương thức, truy xuất một phương thức khác" | Truy vấn văn bản truy xuất hình ảnh; truy vấn hình ảnh truy xuất văn bản; yêu cầu không gian chung hoặc phiên dịch |
| Hợp nhất điểm số | "Kết hợp điểm số" | Tổng trọng số của điểm truy xuất theo phương thức; Hợp nhất đơn giản nhất |
| MoE hợp nhất | "Chuyên gia định tuyến phương thức" | Mạng lưới chọn điểm số của phương thức nào để tin cậy cho mỗi truy vấn |
| Thế hệ nối đất | "Trích dẫn nguồn của bạn" | Mỗi tuyên bố trong câu trả lời được gắn thẻ chỉ mục nguồn |
| MuRAG | "RAG đa phương thức đầu tiên" | Bài báo năm 2022 thiết lập mô hình RAG đa phương thức |
| Agentic nhiều bước nhảy | "Công thức lại và thử lại" | LLM truy vấn lại những người truy vấn khi độ tin cậy của lần vượt qua đầu tiên thấp |

## Đọc thêm

- [Abootorabi et al. — Ask in Any Modality (arXiv:2502.08826)](https://arxiv.org/abs/2502.08826)
- [Mei et al. — A Survey of Multimodal RAG (arXiv:2504.08748)](https://arxiv.org/abs/2504.08748)
- [Zhao et al. — Vision RAG Survey (arXiv:2503.18016)](https://arxiv.org/abs/2503.18016)
- [Chen et al. — MuRAG (arXiv:2210.02928)](https://arxiv.org/abs/2210.02928)
- [Liu et al. — REACT (arXiv:2301.10382)](https://arxiv.org/abs/2301.10382)
