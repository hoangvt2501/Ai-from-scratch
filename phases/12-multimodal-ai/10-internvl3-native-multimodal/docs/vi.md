# InternVL3: Pretraining đa phương thức gốc

> Mỗi VLM mở trước InternVL3 đều tuân theo cùng một công thức ba bước: lấy một văn bản LLM được huấn luyện trên hàng nghìn tỷ tokens văn bản, bắt vít vào encoder tầm nhìn, sau đó fine-tune các đường nối. Điều này hoạt động nhưng có alignment nợ - văn bản LLM đã dành toàn bộ ngân sách pretraining cho văn bản thuần túy và không hiểu tokens trực quan. Khi bạn thêm tầm nhìn sau đó, LLM phải học lại cách liên hệ đầu vào trực quan với lý luận văn bản của nó mà không quên văn bản. InternVL3 (Zhu và cộng sự, tháng 4 năm 2025) bác bỏ cách tiếp cận post-hoc: chạy một pretraining, văn bản và đa phương thức xen kẽ từ bước một. Kết quả khớp với Gemini 2.5 Pro trên MMMU-Pro ở 78B tham số mở. Bài học này đọc trường hợp của pretraining bản địa và những gì thay đổi khi bạn thực hiện nó.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, training-corpus mixer)
**Kiến thức tiên quyết:** Giai đoạn 12 · 05, Giai đoạn 12 · 07 (công thức nấu ăn)
**Thời lượng:** ~120 phút

## Mục tiêu học tập

- Giải thích lý do tại sao các VLM training sau hoc tích lũy alignment nợ, trích dẫn ba triệu chứng có thể đo lường được (quên thảm khốc, trôi câu trả lời, không nhất quán văn bản trực quan).
- Mô tả hỗn hợp kho dữ liệu pretraining gốc của InternVL3 và lý do tại sao tỷ lệ văn bản: xen kẽ: chú thích lại quan trọng.
- So sánh V2PE (mã hóa vị trí trực quan thay đổi) với M-RoPE của Qwen2-VL.
- Đặt tên cho tối ưu hóa triển khai Visual Resolution Router (ViR) và Decoupled Vision-Language (DvD).

## Vấn đề

VLM training sau là mặc định. LLaVA, BLIP-2, Qwen-VL, Idefics — tất cả đều lấy một pretrained LLM sẵn (Llama, Vicuna, Qwen, Mistral) và thêm tầm nhìn. Các giai đoạn training thường giống như:

1. LLM đông lạnh + encoder tầm nhìn đóng băng + máy chiếu có thể huấn luyện, được huấn luyện về các cặp phụ đề để căn chỉnh embeddings.
2. Hủy đóng băng LLM, huấn luyện về dữ liệu hướng dẫn (LLaVA-Instruct, ShareGPT4V).
3. fine-tune tùy chọn dành riêng cho nhiệm vụ.

Ba triệu chứng của alignment nợ xuất hiện:

- Quên thảm khốc. Các VLM sau hoc quên skills chỉ văn bản. Điểm GSM8K giảm 5-10 điểm. Điểm Hellaswag giảm. Văn bản thuần túy agents thoái lui.
- Trả lời trôi dạt. Các cụm từ nhỏ của cùng một câu hỏi trực quan nhận được câu trả lời khác nhau. Tầm nhìn encoder kết nối với LLM với các ràng buộc yếu hơn so với tokens của chính LLM.
- Không nhất quán văn bản trực quan. Người VLM có thể mô tả một hình ảnh một cách chính xác và sau đó trả lời một câu hỏi mâu thuẫn với mô tả của chính nó. Các tokens trực quan không tham gia vào việc kiểm tra tính nhất quán nội bộ của LLM giống như cách văn bản làm.

Những triệu chứng này đã được ghi nhận đầy đủ. MM1.5 Phần 4 định lượng chúng. Việc cắt bỏ của LLaVA-OneVision gợi ý về chúng. pretraining bản địa là câu trả lời.

## Khái niệm

### pretraining đa phương thức gốc

InternVL3 huấn luyện từ đầu trên một kho dữ liệu đa phương thức gốc từ bước một. Sự kết hợp là:

- 40% dữ liệu chỉ có văn bản (FineWeb, Proof-Pile-2, v.v.)
- 35% dữ liệu văn bản hình ảnh xen kẽ (OBELICS, kiểu MMC4)
- 20% dữ liệu chú thích hình ảnh được ghép nối
- 5% dữ liệu văn bản video

tokens thị giác, tokens văn bản và tương tác đa phương thức đều tham gia vào cùng một loss ngay từ bước gradient đầu tiên. Không có alignment pretraining, không có máy chiếu đóng băng, không có thảm khốc quên phục hồi.

Training là một giai đoạn duy nhất cho model cơ sở. Điều chỉnh hướng dẫn theo sau, nhưng cơ sở model đã hiểu tokens thị giác với tư cách là công dân class đầu.

### V2PE (mã hóa vị trí trực quan thay đổi)

Qwen2-VL sử dụng M-RoPE với phân bổ trục cố định. InternVL3 giới thiệu V2PE: mã hóa vị trí khác nhau tùy theo loại phương thức (văn bản, hình ảnh, video) với tỷ lệ có thể học được. Trong thực tế:

- Văn bản tokens có được vị trí 1D (chỉ mục văn bản).
- Các bản vá hình ảnh có vị trí 2D (hàng, col).
- Khung hình video có vị trí 3D (thời gian, hàng, col).

Cả ba chia sẻ cùng một cơ sở tần số RoPE, nhưng phân bổ ẩn-mờ trên mỗi băng tần là một parameter đã học chứ không phải là một phân chia cố định. Tự do đánh đổi độ phân giải tần số thời gian và không gian trong quá trình pretraining.

Tuyên bố cắt bỏ của V2PE: 1-2 điểm trên video benchmarks trên M-RoPE ở cùng một tính toán. Không phải là một cuộc cách mạng, nhưng sạch hơn.

### Bộ định tuyến độ phân giải hình ảnh (ViR)

Tối ưu hóa triển khai. Không phải tất cả hình ảnh đều cần mã hóa độ phân giải đầy đủ. Một bức ảnh có một đối tượng ở độ chi tiết thấp sẽ lãng phí tokens khi được mã hóa ở 1280px gốc. ViR là một bộ phân loại nhỏ dự đoán độ phân giải tối thiểu cần thiết để trả lời câu hỏi, trước khi mã hóa.

Định tuyến có ba bậc: độ phân giải thấp (256 tokens), trung bình (576), cao (2048+). Đối với 60% truy vấn trong lưu lượng truy cập production, thấp hoặc trung bình là đủ. Hiệu ứng ròng: Thông lượng gấp 2-3 lần với chất lượng tương đương.

### Triển khai Ngôn ngữ Tầm nhìn Tách rời (DvD)

Khi bạn phân phát một VLM lớn, encoder tầm nhìn chạy một lần cho mỗi hình ảnh nhưng LLM chạy tự hồi quy cho mỗi token đầu ra. Hai thành phần có các nút thắt cổ chai khác nhau (tầm nhìn = GPU băng thông bộ nhớ cho conv + attention; LLM = KV cache). DvD chia chúng thành các GPUs riêng biệt với streaming ở giữa.

Đối với encoder model 8B + 400M, DvD tăng gần gấp đôi thông lượng trên mỗi nút so với cùng vị trí.

### Chất lượng một giai đoạn so với nhiều giai đoạn

Tuyên bố benchmark chính của InternVL3: ở 78 tỷ tham số, khớp với MMMU-Pro của Gemini 2.5 Pro. Ở 38B, trận đấu GPT-4o. Ở 8B, dẫn đầu bảng xếp hạng 8B mở. Tất cả trên một stage pretrain + công thức điều chỉnh hướng dẫn.

Giả thuyết nợ alignment có thể đo lường được: InternVL3-8B mất ít điểm benchmark văn bản (MMLU, GSM8K) hơn Qwen2.5-VL-7B trên một đơn vị tăng benchmark thị lực. model là một người tổng quát hơn vì training là một mảnh, không phải hai.

### InternVL3.5 và InternVL-U

InternVL3.5 (Tháng Tám 2025) mở rộng công thức. Cùng một cách tiếp cận pretrain gốc, nhiều dữ liệu hơn, nhiều tham số hơn. Cải tiến MMMU là tăng dần.

InternVL-U (2026) bổ sung thế hệ thống nhất — đầu ra hình ảnh qua đầu MMDiT trên cùng một đường trục. Chữ "U" là viết tắt của "Hiểu biết + thế hệ", theo đuổi models thống nhất theo phong cách Truyền máu (Bài 12.13). Cùng một xương sống pretrain bản địa hỗ trợ cả người đứng đầu sự hiểu biết và thế hệ.

### Đánh đổi pretraining bản địa

pretraining bản địa không miễn phí:

- Điện toán. Training một VLM mới từ đầu có chi phí tương đương với training LLM văn bản - hàng triệu GPU giờ. Thích ứng sau hoc tái sử dụng trọng lượng LLM hiện có, tiết kiệm hầu hết chi phí.
- Dữ liệu. Kho dữ liệu văn bản hình ảnh xen kẽ trên quy mô lớn là rất hiếm. OBELICS là 141 triệu tài liệu; MMC4 là 571M. Nhắn tin một mình ships ở 15T tokens. Sự khan hiếm dữ liệu pretraining đa phương thức là một hạn chế khó khăn.
- Tái sử dụng LLM cơ sở. Native pretraining từ bỏ tùy chọn thả vào một LLM mới sau đó. Post-hoc cho phép bạn hoán đổi Llama-3.1 lấy Llama-4 bằng cách chỉ huấn luyện lại bộ chuyển đổi.

Đặt cược mà InternVL3 thực hiện: khoản nợ alignment còn tệ hơn loss tái sử dụng. Người benchmarks ủng hộ tuyên bố. Chi phí sản xuất ngăn cản các phòng thí nghiệm trong tương lai sao chép với giá rẻ. Các VLMs sau sẽ tiếp tục tồn tại vì chúng vẫn rẻ hơn đối với hầu hết các dự án.

## Ứng dụng

`code/main.py` là một bộ trộn training-corpus và trình mô phỏng bộ định tuyến ViR. Nó:

- Lấy hỗn hợp kho dữ liệu mục tiêu (%text, %interleaved, %caption, %video) và tính toán các bước dự kiến cho mỗi phương thức.
- Mô phỏng định tuyến ViR trên một batch truy vấn (phân phối: 50% chi tiết thấp, 30% trung bình, 20% chi tiết cao) và báo cáo số lượng token trung bình.
- Báo cáo ước tính thông lượng DvD đưa ra encoder so với LLM FLOPs.
- In song song giữa pretraining sau khi so với gốc trong các thông số, tính toán, dữ liệu và các triệu chứng nợ alignment dự kiến.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-native-vs-posthoc-auditor.md`. Với một kế hoạch VLM training được đề xuất, nó kiểm tra xem nên đi theo bản địa hay sau đó, đánh dấu rủi ro alignment nợ và đề xuất một hỗn hợp kho dữ liệu. Sử dụng nó khi bạn đang định cỡ một dự án VLM mở mới và cần chọn chiến lược training.

## Bài tập

1. Ước tính delta tính toán giữa InternVL3-8B (pretrain gốc) và LLaVA-OneVision-7B (sau khi sử dụng). Tỷ lệ xấp xỉ GPU giờ? Điều gì giải thích khoảng cách?

2. InternVL3 báo cáo 40% văn bản / 35% xen kẽ / 20% chú thích / 5% video. Nếu nhiệm vụ mục tiêu của bạn nặng về video, hãy đề xuất một tỷ lệ mới và tranh luận tại sao model cơ sở vẫn cần dữ liệu văn bản và phụ đề đáng kể.

3. Đọc MM1.5 Phần 4 về quên. Kể tên chính xác benchmark nơi post-hoc training cho thấy hồi quy lớn nhất. Chi phí hồi quy là bao nhiêu?

4. ViR định tuyến 60% lưu lượng truy cập sang mã hóa độ phân giải thấp. Nó định tuyến sai loại truy vấn nào (gửi đến độ phân giải thấp khi cần độ phân giải cao)? Đề xuất ba chế độ lỗi bộ định tuyến.

5. DvD tách tầm nhìn và LLM thành các GPUs riêng biệt. DvD làm tổn hại đến thông lượng thay vì giúp đỡ theo mô hình lưu lượng nào?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| pretraining đa phương thức gốc | "Từ đầu cùng nhau" | Văn bản + hình ảnh + video tokens tham gia vào loss từ bước 1, không bắt vít sau này |
| Alignment nợ | "Hình phạt sau khi thực hiện" | Hồi quy có thể đo lường được trong skills văn bản và tính nhất quán của câu trả lời đến từ việc gắn tầm nhìn vào LLM |
| V2PE | "Mã hóa pos trực quan có thể thay đổi" | Phân bổ mã hóa vị trí có thể học được theo phương thức; Người kế nhiệm M-RoPE của InternVL3 |
| ViR | "Bộ định tuyến độ phân giải" | Bộ phân loại nhỏ chọn độ phân giải tối thiểu cần thiết cho mỗi truy vấn trước khi mã hóa, tiết kiệm inference tokens |
| Đĩa DVD | "Triển khai tách rời" | Tầm nhìn encoder trên một GPU, LLM trên một  khác, với sự chuyển giao dòng chảy; tăng gấp đôi thông lượng cho VLMs lớn |
| Thực tậpVL-U | "Thống nhất hiểu biết + thế hệ" | Tiếp theo năm 2026 bổ sung đầu tạo hình ảnh vào xương sống pretrain bản địa |
| Kho dữ liệu xen kẽ | "OBELICS / MMC4" | Tài liệu có văn bản và hình ảnh theo thứ tự đọc tự nhiên; nguyên liệu cho pretraining bản địa |

## Đọc thêm

- [Chen et al. — InternVL 1 (arXiv:2312.14238)](https://arxiv.org/abs/2312.14238)
- [Zhu et al. — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)
- [InternVL3.5 (arXiv:2508.18265)](https://arxiv.org/abs/2508.18265)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Zhang et al. — MM1.5 (arXiv:2409.20566)](https://arxiv.org/abs/2409.20566)
