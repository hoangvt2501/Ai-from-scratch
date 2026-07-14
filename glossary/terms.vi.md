<!-- Bản dịch tiếng Việt; giữ nguyên thuật ngữ kỹ thuật, code, công thức, lệnh và URL. -->

# Thuật ngữ kỹ thuật AI

## A

### Agent
- **Mọi người nói gì:** "Một AI tự trị tự suy nghĩ và hành động"
- **Ý nghĩa thực sự của nó: **Một vòng lặp while trong đó một LLM quyết định công cụ nào sẽ gọi tiếp theo, thực thi nó, xem kết quả và lặp lại
- **Tại sao nó được gọi như vậy:** Mượn từ triết học - "agent" là bất cứ thứ gì có thể hành động trong thế giới. Trong AI, nó chỉ có nghĩa là "LLM + công cụ + vòng lặp"

### Attention
- **Những gì mọi người nói: **"Làm thế nào AI tập trung vào các phần quan trọng"
- **Nó thực sự có nghĩa là gì: **Một cơ chế trong đó mỗi token tính toán tổng trọng số của tất cả các giá trị tokens khác, với trọng số được xác định bởi mức độ liên quan của chúng (thông qua tích chấm của truy vấn và vectors khóa)
- **Tại sao nó được gọi như vậy:** Bài báo năm 2017 "Attention là tất cả những gì bạn cần" đặt tên nó bằng cách tương tự với attention chọn lọc của con người

### Alignment
- **Những gì mọi người nói: **"Làm cho AI an toàn"
- **Ý nghĩa thực sự: **Thách thức kỹ thuật của việc làm cho hành vi của hệ thống AI phù hợp với ý định, giá trị và sở thích của con người, bao gồm cả các trường hợp biên mà nhà thiết kế không lường trước được

### Tự hồi quy
- **Những gì mọi người nói: **"AI tạo ra từng từ một"
- **Ý nghĩa thực sự: **Một model dự đoán token tiếp theo có điều kiện trên tất cả các tokens trước đó, sau đó đưa dự đoán đó trở lại làm đầu vào cho bước tiếp theo. GPT, LLaMA và Claude đều tự hồi quy.

### Chức năng kích hoạt
- **Những gì mọi người nói: **"Thứ phi tuyến giữa các lớp"
- **Nó thực sự có nghĩa là gì: **Một hàm được áp dụng sau mỗi lớp tuyến tính giới thiệu tính phi tuyến. Nếu không có nó, xếp chồng bất kỳ số lượng lớp tuyến tính nào sẽ sụp đổ thành một phép biến đổi tuyến tính duy nhất. ReLU, GELU và SiLU là phổ biến nhất. Sự lựa chọn ảnh hưởng trực tiếp đến việc gradients lưu lượng trong quá trình training.

### Adam (Optimizer)
- **Những gì mọi người nói: **"Mặc định optimizer"
- **Ý nghĩa thực sự: **Ước tính khoảnh khắc thích ứng. Kết hợp động lượng (khoảnh khắc đầu tiên) với tốc độ học thích ứng trên mỗi parameter (khoảnh khắc thứ hai). Có bias điều chỉnh cho các bước đầu. Hoạt động tốt trên hầu hết các tác vụ mà không cần điều chỉnh nhiều.

### AdamW
- **Những gì mọi người nói: **"Adam nhưng tốt hơn"
- **Nó thực sự có nghĩa là gì: **Adam với sự phân rã trọng lượng tách rời. Trong Adam tiêu chuẩn, chính quy hóa L2 được chia tỷ lệ theo learning rate thích ứng trên mỗi parameter, đây không phải là điều bạn muốn. AdamW áp dụng phân rã trọng lượng trực tiếp lên trọng lượng, không phụ thuộc vào số liệu thống kê gradient. optimizer mặc định cho training transformers.

### Autograd
- **Mọi người nói gì:** "Tự động gradients"
- **Ý nghĩa thực sự:** Một hệ thống ghi lại các hoạt động trên tensors và tự động tính toán gradients thông qua phân biệt chế độ đảo ngược. Autograd của PyTorch xây dựng biểu đồ tính toán nhanh chóng (đồ thị động), trong khi JAX sử dụng chuyển đổi hàm (grad). Đây là điều làm cho backpropagation thực tế - bạn viết forward pass và framework tính toán tất cả các đạo hàm.

## B

### Kích thước Batch
- **Những gì mọi người nói: **"Có bao nhiêu ví dụ cùng một lúc"
- **Ý nghĩa thực sự:** Số lượng training ví dụ được xử lý trong một forward/backward vượt qua trước khi cập nhật trọng số. batches lớn hơn cho ước tính gradient ổn định hơn nhưng sử dụng nhiều bộ nhớ hơn. Giá trị điển hình: 32-512 cho training, lớn hơn cho inference. Kích thước Batch tương tác với learning rate -- gấp đôi batch, gấp đôi LR (quy tắc tỷ lệ tuyến tính).

### Backpropagation
- **Mọi người nói gì:** "Mạng nơ-ron học như thế nào"
- **Nó thực sự có nghĩa là gì: **Một thuật toán tính toán mức độ mỗi trọng số đóng góp vào lỗi bằng cách áp dụng quy tắc chuỗi ngược qua mạng, sau đó điều chỉnh trọng số theo tỷ lệ
- **Tại sao nó được gọi như vậy:** Lỗi lan truyền ngược từ đầu ra này sang đầu vào khác, từng lớp

## C

### Context Window
- **Những gì mọi người nói: **"Người AI có thể nhớ được bao nhiêu"
- **Nó thực sự có nghĩa là gì: **Số lượng tokens tối đa (đầu vào + đầu ra) phù hợp với một cuộc gọi API duy nhất. Không phải bộ nhớ - đó là một bộ đệm có kích thước cố định đặt lại mọi cuộc gọi

### Chuỗi tư tưởng (CoT)
- **Những gì mọi người nói:** "Làm cho AI suy nghĩ từng bước"
- **Nó thực sự có nghĩa là gì: **Một kỹ thuật prompting trong đó bạn yêu cầu model thể hiện các bước suy luận của nó, giúp cải thiện accuracy đối với các vấn đề nhiều bước vì mỗi bước điều kiện cho thế hệ token tiếp theo

### CNN (Mạng nơ-ron tích chập)
- **Những gì mọi người nói: **"Hình ảnh AI"
- **Ý nghĩa thực sự:** Một mạng nơ-ron sử dụng các hoạt động tích chập (bộ lọc trượt qua đầu vào) để phát hiện các mẫu cục bộ. Tích chập xếp chồng phát hiện các features ngày càng phức tạp: cạnh, kết cấu, đối tượng.

### CUDA
- **Những gì mọi người nói: **"Lập trình GPU"
- **Nó thực sự có nghĩa là gì: **Nền tảng điện toán song song của NVIDIA. Cho phép bạn chạy các hoạt động ma trận trên hàng nghìn lõi GPU cùng một lúc. PyTorch và TensorFlow sử dụng CUDA dưới mui xe.

### Chunking
- **Những gì mọi người nói:** "Chia tài liệu thành nhiều mảnh"
- **Ý nghĩa thực sự:** Chia văn bản thành các phân đoạn trước khi embedding để truy xuất. Kích thước khối xác định mức độ chi tiết của kết quả tìm kiếm. Quá nhỏ: mất ngữ cảnh. Quá lớn: làm loãng mức độ liên quan. Các chiến lược phổ biến: kích thước cố định với sự chồng chéo, dựa trên câu hoặc tách ngữ nghĩa. Kích thước khối điển hình: 256-512 tokens với 10-20% chồng chéo.

### Học tương phản
- **Những gì mọi người nói:** "Học bằng cách so sánh"
- **Nó thực sự có nghĩa là gì: **Training bằng cách kéo các cặp tương tự lại gần hơn và đẩy các cặp khác nhau ra xa nhau trong embedding không gian. CLIP sử dụng điều này: ghép cặp hình ảnh-văn bản so với cặp không khớp.

### Sự tương đồng cosin
- **Những gì mọi người nói: **"Hai vectors giống nhau như thế nào"
- **Ý nghĩa thực sự: **Cosin của góc giữa hai vectors: chấm (a, b) / (||một|| * ||b||). Phạm vi từ -1 (ngược lại) đến 1 (cùng hướng). Bỏ qua độ lớn, chỉ quan tâm đến phương hướng. Chỉ số tương tự tiêu chuẩn cho tìm kiếm embeddings và ngữ nghĩa.

### Entropy chéo
- **Những gì mọi người nói:** "Phân loại loss"
- **Nó thực sự có nghĩa là gì: **Đo lường sự khác biệt giữa hai phân phối xác suất. Để phân loại: -sum(y_true * log(y_pred)). Đối với models ngôn ngữ: log probability phủ định của token tiếp theo chính xác. Thấp hơn là tốt hơn. Perplexity chỉ là exp (cross-entropy).

## D

### Tăng cường dữ liệu
- **Những gì mọi người nói:** "Tạo ra nhiều dữ liệu training hơn"
- **Ý nghĩa thực sự: **Tạo các bản sao sửa đổi của dữ liệu hiện có (xoay hình ảnh, thêm nhiễu, diễn giải văn bản) để tăng training đặt sự đa dạng mà không cần thu thập dữ liệu mới. Giảm overfitting.

### Decoder
- **Những gì mọi người nói: **"Phần đầu ra"
- **Nó thực sự có nghĩa là gì: **Trong transformers, một decoder sử dụng self-attention nhân quả (được che giấu) để mỗi vị trí chỉ có thể tham gia vào các vị trí trước đó. GPT chỉ dành cho decoder. BERT chỉ dành cho encoder. T5 là encoder-decoder.

### Khuếch tán Model
- **Những gì mọi người nói:** "AI tạo ra hình ảnh từ nhiễu"
- **Ý nghĩa thực sự: **Một model được huấn luyện để đảo ngược process nhiễu dần dần - nó học cách dự đoán và loại bỏ nhiễu, và tại thời điểm tạo ra bắt đầu từ nhiễu thuần túy và khử nhiễu lặp đi lặp lại

### DPO (Tối ưu hóa tùy chọn trực tiếp)
- **Những gì mọi người nói: **"Một RLHF đơn giản hơn"
- **Ý nghĩa thực sự của nó: **Một phương pháp training bỏ qua phần thưởng hoàn toàn model - nó trực tiếp tối ưu hóa ngôn ngữ model thích phản hồi tốt hơn trong các cặp sở thích của con người

### Dropout
- **Những gì mọi người nói:** "Ngẫu nhiên tắt các tế bào thần kinh"
- **Ý nghĩa thực sự: **Trong training, hãy đặt ngẫu nhiên một phần kích hoạt thành không. Buộc mạng không phụ thuộc vào bất kỳ tế bào thần kinh đơn lẻ nào. Đã tắt trong khi inference. Chính quy hóa đơn giản nhưng hiệu quả.

## E

### Giá trị riêng
- **Những gì mọi người nói: **"Một số thứ toán học cho PCA"
- **Nó thực sự có nghĩa là gì: **Đối với ma trận A, một lambda có giá trị riêng thỏa mãn Av = lambda * v cho một số vector v. Nó cho bạn biết ma trận chia tỷ lệ vectors bao nhiêu theo hướng đó. Giá trị riêng lớn = hướng variance cao trong dữ liệu của bạn.

### Embedding
- **Những gì mọi người nói: **"Một số AI phép thuật biến từ thành số"
- **Ý nghĩa thực sự của nó: **Một ánh xạ đã học từ các mục rời rạc (từ, hình ảnh, người dùng) đến vectors dày đặc trong không gian liên tục, nơi các mục tương tự kết thúc gần nhau
- **Tại sao nó được gọi như vậy: **Các mục được "nhúng" trong một không gian hình học nơi khoảng cách có ý nghĩa

### Encoder
- **Những gì mọi người nói:** "Phần đầu vào"
- **Nó thực sự có nghĩa là gì: **Trong transformers, một encoder sử dụng self-attention hai chiều để mỗi vị trí có thể tham gia vào tất cả các vị trí. BERT chỉ dành cho encoder. Tốt cho việc hiểu các nhiệm vụ (phân loại, NER) nhưng không phải thế hệ.

### Epoch
- **Những gì mọi người nói:** "Một lần đi qua dữ liệu"
- **Nó thực sự có nghĩa là gì: **Chính xác như vậy. Một lần hoàn chỉnh qua mọi ví dụ trong bộ training. Nhiều epochs = xem dữ liệu nhiều lần. Nhiều epochs hơn có thể cải thiện việc học nhưng rủi ro overfitting.

## F

### Feature
- **Những gì mọi người nói:** "Một cột trong dữ liệu của bạn"
- **Ý nghĩa thực sự: **Một thuộc tính có thể đo lường riêng lẻ của dữ liệu. Trong ML cổ điển, bạn thiết kế features bằng tay. Trong deep learning, mạng tự động học features từ dữ liệu thô.

### Few-Shot
- **Những gì mọi người nói:** "Hãy cho AI một số ví dụ trước"
- **Ý nghĩa thực sự:** Bao gồm một số lượng nhỏ các ví dụ đầu vào-đầu ra trong prompt trước khi yêu cầu model thực hiện một tác vụ. Thông thường là 3-5 ví dụ. Các model khớp mẫu trên các ví dụ này để hiểu định dạng và hành vi mong muốn. Trái ngược với zero-shot (không có ví dụ) và fine-tuning (hàng ngàn ví dụ được nướng thành trọng lượng).

### Fine-tuning
- **Những gì mọi người nói:** "Training AI trên dữ liệu của bạn"
- **Ý nghĩa thực sự: **Bắt đầu với tạ của model được huấn luyện trước và tiếp tục training trên một dataset nhỏ hơn, cụ thể theo nhiệm vụ. Chỉ cập nhật trọng lượng hiện có, không bổ sung kiến thức mới từ đầu

### Function Calling
- **Những gì mọi người nói: **"AI có thể sử dụng công cụ"
- **Nó thực sự có nghĩa là gì: **Một cách có cấu trúc để LLMs yêu cầu thực thi các chức năng bên ngoài. Bạn định nghĩa các công cụ với JSON Schema mô tả, model xuất ra một đối tượng JSON có cấu trúc chỉ định hàm nào sẽ gọi với đối số nào, mã của bạn thực thi nó và kết quả quay trở lại model. Không giống như agents - function calling là cơ chế, agents là vòng lặp.

## G

### Guardrails
- **Mọi người nói gì:** "Bộ lọc an toàn cho AI"
- **Ý nghĩa thực sự:** Input/output các lớp xác thực xung quanh một LLM phát hiện và chặn nội dung độc hại, prompt nỗ lực tiêm, rò rỉ PII hoặc phản hồi lạc đề. Điển hình là pipeline: bộ lọc đầu vào -> LLM -> bộ lọc đầu ra. Có thể dựa trên quy tắc (biểu thức chính quy, danh sách từ khóa) hoặc dựa trên model (bộ phân loại chấm điểm an toàn).

### GPT
- **Những gì mọi người nói:** "ChatGPT" hoặc "The AI"
- **Ý nghĩa thực sự:** Generative Pre-trained Transformer — một kiến trúc cụ thể dự đoán token tiếp theo bằng cách sử dụng transformer chỉ dành cho decoder được huấn luyện trên kho dữ liệu văn bản lớn
- **Tại sao nó được gọi như vậy:** Generative (tạo văn bản), Pre-trained (được huấn luyện một lần trên dữ liệu lớn, sau đó được điều chỉnh), Transformer (kiến trúc)

### GAN (Mạng đối kháng tổng quát)
- **Mọi người nói gì:** "Hai AI chiến đấu với nhau"
- **Nó thực sự có nghĩa là gì: **Một mạng máy phát cố gắng tạo ra dữ liệu thực tế trong khi một mạng phân biệt đối xử cố gắng phân biệt thật và giả. Họ huấn luyện cùng nhau: máy phát điện trở nên tốt hơn trong việc đánh lừa người phân biệt đối xử và người phân biệt trở nên tốt hơn trong việc phát hiện hàng giả.

### Gradient
- **Những gì mọi người nói:** "Con dốc"
- **Ý nghĩa thực sự của nó: **Một vector của các đạo hàm từng phần chỉ theo hướng tăng mạnh nhất. Trong ML, bạn đi ngược lại với gradient (gradient descent) để giảm thiểu loss.

### Gradient Descent
- **Những gì mọi người nói: **"Làm thế nào AI cải thiện"
- **Nó thực sự có nghĩa là gì: **Một thuật toán tối ưu hóa điều chỉnh parameters theo hướng làm giảm chức năng loss dốc nhất, giống như đi bộ xuống dốc trong phong cảnh high-dimensional

## H

### Hyperparameter
- **Những gì mọi người nói: **"Cài đặt bạn điều chỉnh"
- **Ý nghĩa thực sự: **Các giá trị được đặt trước training kiểm soát chính training process: learning rate, kích thước batch, số lớp dropout tỷ lệ. Không giống như các model parameters (trọng số), chúng không được học từ dữ liệu.

### Ảo giác
- **Những gì mọi người nói:** "Người AI đang nói dối" hoặc "bịa đặt mọi thứ"
- **Ý nghĩa thực sự của nó: **model tạo ra văn bản nghe có vẻ hợp lý không dựa trên dữ liệu training hoặc ngữ cảnh nhất định của nó - nó hoàn thành mẫu, không truy xuất dữ kiện

## I

### Inference
- **Những gì mọi người nói: **"Chạy AI"
- **Ý nghĩa thực sự: **Sử dụng một model được huấn luyện để đưa ra dự đoán về dữ liệu mới. Không có cập nhật cân nặng nào xảy ra. Đây là những gì bạn làm trong production: gửi đầu vào, nhận đầu ra.

### Bias quy nạp
- **Những gì mọi người nói: **Chưa bao giờ nghe nói về nó
- **Ý nghĩa thực sự: **Các giả định được xây dựng trong kiến trúc của model. CNN giả định các mô hình cục bộ quan trọng (tích chập). RNN giả định các vấn đề về thứ tự (xử lý tuần tự). Transformers cho rằng mọi thứ có thể liên quan đến mọi thứ (attention). bias phù hợp giúp model học nhanh hơn từ ít dữ liệu hơn.

### JAX
- **Những gì mọi người nói:** "ML framework của Google"
- **Ý nghĩa thực sự của nó: **Một thư viện tương thích với NumPy bổ sung sự khác biệt tự động (grad), biên dịch JIT (jit), vectơ hóa tự động (vmap) và song song đa thiết bị (pmap). Không giống như phong cách hướng đối tượng của PyTorch, JAX hoàn toàn là chức năng - không có trạng thái ẩn, không có đột biến tại chỗ. Được Google DeepMind sử dụng cho AlphaFold, Gemini và nghiên cứu quy mô lớn.

## K

### KV Cache
- **Những gì mọi người nói:** "Làm cho inference nhanh hơn"
- **Nó thực sự có nghĩa là gì: **Trong quá trình tạo tự hồi quy, hãy lưu trữ các ma trận khóa và giá trị từ tokens trước để bạn không tính toán lại chúng ở mỗi bước. Đánh đổi bộ nhớ lấy tốc độ. Cần thiết để LLM inference nhanh.

## L

### Không gian tiềm ẩn
- **Những gì mọi người nói: **"Đại diện ẩn"
- **Ý nghĩa thực sự của nó: **Một không gian biểu diễn nén, học được trong đó các đầu vào tương tự ánh xạ đến các điểm lân cận. Bộ mã hóa tự động, VAE và models khuếch tán đều hoạt động trong không gian tiềm ẩn. Nó có chiều thấp hơn đầu vào nhưng nắm bắt được cấu trúc quan trọng.

### Learning Rate
- **Những gì mọi người nói: **"AI học nhanh như thế nào"
- **Ý nghĩa thực sự: **Một vô hướng kiểm soát kích thước bước trong quá trình gradient descent. Quá cao: vượt quá mức tối thiểu và phân kỳ. Quá thấp: hội tụ quá chậm hoặc bị kẹt. hyperparameter quan trọng nhất duy nhất.

### LLM (Model ngôn ngữ lớn)
- **Những gì mọi người nói: **"AI" hoặc "bộ não"
- **Ý nghĩa thực sự: **Một mạng nơ-ron dựa trên transformer được huấn luyện để dự đoán token tiếp theo trong một chuỗi, với hàng tỷ parameters, được huấn luyện trên dữ liệu văn bản quy mô internet

### LoRA (Chuyển thể cấp thấp)
- **Những gì mọi người nói: **"fine-tuning hiệu quả"
- **Ý nghĩa thực sự: **Thay vì cập nhật tất cả các trọng số, hãy chèn các ma trận cấp thấp nhỏ cùng với trọng lượng ban đầu. Chỉ những ma trận nhỏ này được huấn luyện, giảm bộ nhớ 10-100 lần

### Chức năng Loss
- **Những gì mọi người nói: **"AI sai làm sao"
- **Ý nghĩa thực sự của nó: **Một hàm đo lường khoảng cách giữa đầu ra dự đoán và đầu ra thực tế. Training giảm thiểu chức năng này. MSE để hồi quy, entropy chéo để phân loại, loss tương phản cho embeddings. Việc lựa chọn hàm loss xác định ý nghĩa của "tốt" đối với model.

## M

### Mixed Precision
- **Những gì mọi người nói: **"Training mẹo cho tốc độ"
- **Ý nghĩa thực sự:** Sử dụng float16 cho forward pass và hầu hết các hoạt động (nhanh hơn, ít bộ nhớ hơn) nhưng giữ float32 để tích lũy gradient và cập nhật trọng số (chính xác hơn). Tăng tốc gấp 2 lần với accuracy loss không đáng kể.

### MoE (Sự kết hợp của các chuyên gia)
- **Những gì mọi người nói: **"Chỉ một phần của model chạy"
- **Ý nghĩa thực sự: **Một model có nhiều mạng con "chuyên gia", trong đó một cơ chế định tuyến gửi mỗi đầu vào chỉ cho một vài chuyên gia. Toàn bộ model là rất lớn nhưng mỗi forward pass đều rẻ vì hầu hết các chuyên gia đều bị bỏ qua. Mixtral và GPT-4 sử dụng điều này.

### MCP (Giao thức ngữ cảnh Model)
- **Những gì mọi người nói: **"Một cách để AI sử dụng các công cụ"
- **Ý nghĩa thực sự:** Một giao thức mở (JSON-RPC qua stdio/HTTP) chuẩn hóa cách AI ứng dụng kết nối với các nguồn dữ liệu và công cụ bên ngoài, với schemas được nhập cho các công cụ, tài nguyên và prompts

## N

### NaN (không phải số)
- **Những gì mọi người nói: **"Training bị sập"
- **Ý nghĩa thực sự: **Giá trị dấu phẩy động cho biết kết quả chưa xác định (0/0, inf-inf). Trong training, NaN loss thường có nghĩa là: learning rate quá cao, bùng nổ gradients, log bằng không, hoặc chia cho không. Luôn luôn là điều đầu tiên cần kiểm tra khi training thất bại.

### Chuẩn hóa
- **Mọi người nói gì:** "Mở rộng quy mô dữ liệu"
- **Ý nghĩa thực sự: **Điều chỉnh các giá trị theo phạm vi tiêu chuẩn. Batch chuẩn hóa bình thường hóa trên một batch. Chuẩn hóa lớp được chuẩn hóa trên features. Cả hai đều ổn định training và cho phép tốc độ học tập cao hơn.

## O

### Overfitting
- **Những gì mọi người nói:** "Người model ghi nhớ dữ liệu"
- **Ý nghĩa thực sự: **model hoạt động tốt trên dữ liệu training nhưng kém trên dữ liệu không nhìn thấy. Nó học được nhiễu, không phải tín hiệu. Khắc phục với: nhiều dữ liệu hơn, chính quy hóa (dropout, giảm trọng lượng), dừng sớm, tăng cường dữ liệu, model đơn giản hơn.

### Optimizer
- **Những gì mọi người nói:** "Thứ cập nhật trọng lượng"
- **Ý nghĩa thực sự: **Một thuật toán sử dụng gradients để cập nhật model parameters. SGD là đơn giản nhất. Adam là phổ biến nhất. Mỗi optimizer có các tính chất khác nhau: tốc độ hội tụ, sử dụng bộ nhớ, độ nhạy với hyperparameters.

## P

### Parameter
- **Những gì mọi người nói: **"Kích thước Model"
- **Ý nghĩa thực sự của nó: **Một giá trị có thể học được trong model, thường là trọng lượng hoặc bias. "7B parameters" có nghĩa là 7 tỷ số có thể học được. Mỗi float32 parameter mất 4 byte, vì vậy 7B parameters = 28GB bộ nhớ chỉ dành cho trọng số.

### Perplexity
- **Những gì mọi người nói: **"Người model bối rối làm sao"
- **Nó thực sự có nghĩa là gì: **Hàm mũ của entropy chéo trung bình loss. Thấp hơn là tốt hơn. perplexity 10 có nghĩa là model không chắc chắn như thể nó được chọn đồng nhất trong số 10 tokens ở mỗi bước.

### Precision & Recall
- **Những gì mọi người nói:** "Accuracy số liệu"
- **Ý nghĩa thực sự:** Precision = số mục bạn đã gắn cờ, có bao nhiêu mục đúng. Recall = trong số tất cả các mục đúng, bạn đã tìm thấy bao nhiêu. Họ đánh đổi: bắt mọi email spam (recall cao) có nghĩa là nhiều cảnh báo giả hơn (precision thấp). F1 score là giá trị trung bình hài hòa của chúng. Sử dụng precision khi dương tính giả tốn kém recall khi âm tính giả tốn kém.

### Kỹ thuật Prompt
- **Những gì mọi người nói: **"Nói chuyện với AI đúng cách"
- **Ý nghĩa thực sự:** Thiết kế văn bản đầu vào để tạo ra đầu ra mong muốn một cách đáng tin cậy -- bao gồm prompts hệ thống, ví dụ few-shot, hướng dẫn định dạng và chain-of-thought triggers

### Prompt Tiêm
- **Những gì mọi người nói:** "Hack AI bằng lời"
- **Ý nghĩa thực sự:** Một cuộc tấn công trong đó văn bản độc hại trong đầu vào ghi đè lên system prompt hoặc hướng dẫn. Tiêm trực tiếp: người dùng gõ "Bỏ qua các hướng dẫn trước đó". Chèn gián tiếp: một tài liệu được truy xuất chứa các hướng dẫn ẩn. Tương đương với LLM tiêm SQL. Không có giải pháp hoàn chỉnh nào tồn tại - phòng thủ là các lớp xác thực đầu vào, lọc đầu ra và tách đặc quyền.

## Q

### QLoRA
- **Những gì mọi người nói: **"LoRA nhưng rẻ hơn"
- **Ý nghĩa thực sự: **LoRA lượng tử hóa. Giữ trọng lượng model đế đóng băng ở precision 4 bit (định dạng NF4) trong khi training LoRA bộ điều hợp ở 16 bit. Giảm bộ nhớ thêm 3-4 lần so với LoRA tiêu chuẩn. Một model 7B cần 14GB với LoRA phù hợp với 4-6GB với QLoRA. Chất lượng nằm trong khoảng 1% fine-tuning đầy đủ trên hầu hết các benchmarks.

## R

### RAG (Thế hệ tăng cường truy xuất)
- **Những gì mọi người nói:** "AI có thể tìm kiếm"
- **Ý nghĩa thực sự: **Một mẫu mà bạn truy xuất các tài liệu liên quan từ cơ sở tri thức (sử dụng sự tương đồng embedding), nhồi chúng vào prompt và để LLM trả lời dựa trên ngữ cảnh đó
- **Tại sao nó được gọi như vậy:** Truy xuất (tìm tài liệu) + Tăng cường (thêm vào prompt) + Thế hệ (LLM viết câu trả lời)

### RLHF (Học tăng cường từ phản hồi của con người)
- **Những gì mọi người nói:** "Làm thế nào họ làm cho AI hữu ích"
- **Ý nghĩa thực sự của nó:** Một training pipeline: (1) thu thập sở thích của con người trên model đầu ra, (2) huấn luyện model phần thưởng dựa trên các sở thích đó, (3) sử dụng PPO để tối ưu hóa LLM để tạo ra kết quả phần thưởng cao hơn

### Quantization
- **Những gì mọi người nói:** "Làm cho model nhỏ hơn"
- **Ý nghĩa thực sự: **Giảm precision trọng số model từ float32 (4 byte) xuống int8 (1 byte) hoặc int4 (0,5 byte). Giao dịch một lượng nhỏ accuracy để có bộ nhớ ít hơn 4-8 lần và inference nhanh hơn. GPTQ, AWQ và GGUF là những định dạng phổ biến.

### ReLU
- **Mọi người nói gì:** "Chức năng kích hoạt"
- **Ý nghĩa thực sự: **Đơn vị tuyến tính chỉnh lưu: f (x) = max (0, x). Kích hoạt phi tuyến tính đơn giản nhất. Tính toán nhanh, không bão hòa đối với các giá trị dương. Được sử dụng ở khắp mọi nơi vì nó hoạt động và rẻ. Các biến thể: LeakyReLU, GELU, SiLU.

### ĐỎ
- **Mọi người nói gì:** "Chỉ số tóm tắt"
- **Nó thực sự có nghĩa là gì: **Nghiên cứu định hướng Recall để đánh giá gisting. Các biện pháp trùng lặp giữa văn bản được tạo và văn bản tham chiếu. ROUGE-1 đếm các kết quả trùng khớp unigram, ROUGE-2 đếm các kết quả trùng khớp bigram, ROUGE-L tìm dãy con chung dài nhất. Rẻ tiền để tính toán nhưng chỉ đo lường sự giống nhau bề mặt - hai câu có cùng nghĩa nhưng các từ khác nhau đạt điểm kém.

## S

### Tìm kiếm ngữ nghĩa
- **Những gì mọi người nói:** "Tìm kiếm thông minh hiểu ý nghĩa"
- **Ý nghĩa thực sự:** Tìm tài liệu theo ý nghĩa thay vì đối sánh từ khóa. Nhúng truy vấn và tất cả tài liệu vào cùng một không gian vector, sau đó trả về tài liệu có embeddings gần nhất với embedding truy vấn. "Thanh toán không thành công" tìm thấy "giao dịch bị từ chối" mặc dù chúng không chia sẻ từ nào. Được hỗ trợ bởi cơ sở dữ liệu embedding models + vector.

### Streaming
- **Những gì mọi người nói:** "Nhìn thấy phản hồi xuất hiện từng từ"
- **Nó thực sự có nghĩa là gì: **LLM gửi tokens khi chúng được tạo thay vì chờ đợi phản hồi đầy đủ. Sử dụng các giao thức Server-Sent Events (SSE) hoặc WebSocket. Giảm độ trễ cảm nhận từ giây xuống mili giây trong token đầu tiên. Cần thiết cho production giao diện trò chuyện. Mỗi đoạn chứa một delta (một phần token hoặc từ).

### Self-Attention
- **Những gì mọi người nói:** "Làm thế nào model quyết định những gì cần tập trung vào"
- **Ý nghĩa thực sự: **Mỗi token tính toán truy vấn, khóa và giá trị vectors. Attention trọng số giữa hai tokens = tích điểm của truy vấn và khóa của họ, được chia tỷ lệ và tối đa mềm. Đầu ra = tổng giá trị có trọng số vectors. Cho phép mọi token nhìn thấy mọi token khác.

### SFT (Fine-Tuning được giám sát)
- **Những gì mọi người nói:** "Dạy model làm theo hướng dẫn"
- **Nó thực sự có nghĩa là gì: **Fine-tuning một model được huấn luyện trước về các cặp (hướng dẫn, phản hồi). model học cách tạo ra phản hồi được đưa ra hướng dẫn. Đây là điều biến một model cơ sở thành một model trò chuyện.

### Softmax
- **Những gì mọi người nói:** "Biến các con số thành xác suất"
- **Ý nghĩa thực sự: **softmax (x_i) = exp (x_i) / sum (exp (x_j)). Chuyển đổi một vector số thực tùy ý thành phân phối xác suất (tất cả dương, tổng bằng 1). Được sử dụng trong các đầu phân loại, trọng lượng attention và bất cứ nơi nào bạn cần xác suất.

### Swarm
- **Những gì mọi người nói: **"Một đám AI agents làm việc cùng nhau như những con ong"
- **Ý nghĩa thực sự của nó: **Trạng thái chia sẻ nhiều agents và phối hợp thông qua truyền tin nhắn, với hành vi nổi lên phát sinh từ các quy tắc cá nhân đơn giản hơn là kiểm soát trung tâm

## T

### System Prompt
- **Những gì mọi người nói:** "Hướng dẫn của AI"
- **Nó thực sự có nghĩa là gì: **Một thông điệp đặc biệt khi bắt đầu cuộc trò chuyện thiết lập hành vi, tính cách và ràng buộc của model. Được xử lý trước tin nhắn của người dùng. Người dùng không hiển thị trong hầu hết các giao diện người dùng. Xác định những gì model nên và không nên làm, giọng điệu, tùy chọn định dạng và tiêu điểm miền. Khác với prompts người dùng - prompts hệ thống được thiết lập bởi nhà phát triển.

### Tensor
- **Những gì mọi người nói:** "Một mảng đa chiều"
- **Ý nghĩa thực sự: **Cấu trúc dữ liệu cơ bản trong deep learning frameworks. tensor 0D là vô hướng, 1D là vector, 2D là ma trận, 3D + là tensor. Trong PyTorch và JAX, tensors theo dõi lịch sử tính toán của chúng để phân biệt tự động và có thể tồn tại trên CPU hoặc GPU. Tất cả các đầu vào, đầu ra, trọng số và gradients mạng nơ-ron đều tensors.

### Token
- **Những gì mọi người nói: **"Một từ"
- **Ý nghĩa thực sự của nó: **Một đơn vị từ phụ (thường là 3-4 ký tự trong tiếng Anh) được tạo ra bởi một tokenizer như BPE. "Không thể tin được" có thể là 3 tokens: "un" + "believ" + "able"

### Temperature
- **Mọi người nói gì:** "Cài đặt sáng tạo"
- **Nó thực sự có nghĩa là gì: **Một vô hướng chia logits trước softmax. Temperature=1 là mặc định. Cao hơn = phân phối phẳng hơn = nhiều đầu ra ngẫu nhiên hơn. Thấp hơn = phân bố sắc nét hơn = xác định hơn. Temperature=0 là argmax (luôn chọn token có khả năng cao nhất).

### Chuyển tiếp học tập
- **Những gì mọi người nói:** "Sử dụng model được huấn luyện trước"
- **Nó thực sự có nghĩa là gì: **Lấy một model được huấn luyện về một nhiệm vụ và điều chỉnh nó cho một nhiệm vụ khác. Các lớp đầu học các features chung (cạnh, mẫu cú pháp) chuyển giao. Chỉ các lớp sau mới cần training theo nhiệm vụ cụ thể. Đây là lý do tại sao bạn có thể fine-tune BERT cho bất kỳ nhiệm vụ NLP nào.

### Transformer
- **Mọi người nói gì:** "Kiến trúc đằng sau AI hiện đại"
- **Ý nghĩa thực sự của nó: **Một kiến trúc mạng nơ-ron processes trình tự bằng cách sử dụng self-attention (cho phép mọi vị trí tham gia vào mọi vị trí khác) thay vì lặp lại, cho phép song song hóa lớn
- **Tại sao nó được gọi như vậy: **Nó chuyển đổi các biểu diễn đầu vào thành biểu diễn đầu ra thông qua các lớp attention

## U

### Underfitting
- **Những gì mọi người nói: **"Người model không học"
- **Ý nghĩa thực sự: **model quá đơn giản để nắm bắt các mẫu trong dữ liệu. Training loss vẫn ở mức cao. Sửa chữa với: nhiều parameters hơn, nhiều lớp hơn, training dài hơn, chính quy hóa thấp hơn, features tốt hơn.

## V

### VAE (Bộ mã hóa tự động biến thể)
- **Những gì mọi người nói: **"Một model sinh sản"
- **Nó thực sự có nghĩa là gì: **Một bộ mã hóa tự động học một không gian tiềm ẩn mượt mà bằng cách buộc đầu ra encoder tuân theo phân phối Gaussian. Bạn có thể lấy mẫu từ bản phân phối này và giải mã để tạo dữ liệu mới. Thủ thuật tái tham số hóa làm cho nó có thể huấn luyện được thông qua backpropagation.

### Cơ sở dữ liệu Vector
- **Những gì mọi người nói:** "Một cơ sở dữ liệu đặc biệt dành cho AI"
- **Ý nghĩa thực sự:** Cơ sở dữ liệu được tối ưu hóa để lưu trữ vectors (mảng nổi dày đặc) và thực hiện tìm kiếm lân cận gần nhất nhanh chóng. Hoạt động cốt lõi trong các hệ thống tìm kiếm, RAG và đề xuất tương tự.

## W

### Trọng lượng
- **Những gì mọi người nói: **"Những gì model học được"
- **Nó thực sự có nghĩa là gì: **Một số duy nhất trong ma trận parameter của model. Một lớp tuyến tính có kích thước đầu vào 768 và kích thước đầu ra 3072 có 768 * 3072 = 2.359.296 trọng số. Training điều chỉnh từng trọng lượng để giảm thiểu chức năng loss.

### Phân rã trọng lượng
- **Những gì mọi người nói: **"Chính quy hóa"
- **Nó thực sự có nghĩa là gì: **Thêm một hình phạt tỷ lệ thuận với độ lớn của trọng số vào hàm loss. Tương đương với chính quy hóa L2. Ngăn trọng lượng phát triển quá lớn. Giá trị điển hình: 0,01-0,1.

## Z

### Zero-Shot
- **Những gì mọi người nói: **"Không cần training"
- **Ý nghĩa thực sự của nó: **Sử dụng model cho một nhiệm vụ mà nó không được huấn luyện rõ ràng, không có ví dụ cụ thể về nhiệm vụ nào trong prompt. model khái quát hóa từ trước khi training. Hoạt động bởi vì các models lớn đã thấy đủ đa dạng để xử lý các định dạng tác vụ mới.
