# vLLM Phục vụ nội bộ: PagedAttention, Continuous Batching, Chunked Prefill

> Sự thống trị của vLLM vào năm 2026 dựa trên ba vụ vỡ nợ kép, không phải một thủ thuật nào. PagedAttention luôn bật. Hàng loạt liên tục đưa các yêu cầu mới vào batch đang hoạt động giữa các lần lặp giải mã. Các lát lấp đầy trước dài prompts giải mã tokens không bao giờ bị đói. Bật cả ba và Llama 3.3 70B FP8 trên một H100 SXM5 đẩy 2,200-2,400 tok/s ở 128 đồng thời - cao hơn khoảng 25% so với mặc định của vLLM và 3-4 lần một vòng lặp PyTorch ngây thơ. Bài học này đọc bộ lập lịch và hạt nhân attention ở cấp độ bạn có thể lập sơ đồ và kết thúc bằng một batcher liên tục đồ chơi trong `code/main.py` lên lịch điền trước và giải mã theo cách vLLM làm.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, toy continuous batching scheduler)
**Kiến thức tiên quyết:** Giai đoạn 17 · 01 (Phục vụ Model), Giai đoạn 11 (Kỹ thuật LLM)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Giải thích PagedAttention như một công cụ phân bổ KV cache: khối, bảng khối và lý do tại sao phân mảnh vẫn dưới 4% khi tải production.
- Sơ đồ lô liên tục ở cấp độ lặp: cách các trình tự đã hoàn thành rời khỏi batch và các chuỗi mới nối với nhau mà không bị cạn kiệt.
- Mô tả điền trước theo từng đoạn trong một câu và đặt tên cho chỉ số độ trễ mà nó bảo vệ (gợi ý: đó là đuôi TTFT, không phải thông lượng trung bình).
- Đặt tên cho 2026 vLLM v0.18.0 gotcha cắn các nhóm cho phép mọi tối ưu hóa cùng một lúc.

## Vấn đề

Một vòng lặp phục vụ PyTorch ngây thơ chạy một yêu cầu tại một thời điểm: mã hóa, điền trước, giải mã cho đến khi EOS, trả về. Tại một người dùng, điều này hoạt động. Ở một trăm, đó là một hàng đợi của những người kiên nhẫn. Bản sửa lỗi rõ ràng - hàng loạt tĩnh - đệm mọi yêu cầu vào prompt dài nhất trong cửa sổ, đệm mọi giải mã đến đầu ra dự kiến dài nhất và đình trệ toàn bộ batch trên trình tự chậm nhất. Bạn trả tiền cho khoảng đệm mà bạn không bao giờ sử dụng và các yêu cầu nhanh sẽ chờ đợi những yêu cầu chậm.

vLLM giải quyết ba vấn đề cùng một lúc. PagedAttention ngăn chặn sự phân mảnh KV cache ăn 60-80% bộ nhớ GPU theo cách phân bổ liền kề cổ điển. Batching liên tục cho phép các yêu cầu tham gia và để lại batch giữa mỗi lần lặp lại giải mã, vì vậy batch luôn có đầy đủ công việc thực tế. Điền trước theo khối chia 32k token prompt thành ~512 lát token xen kẽ với giải mã, vì vậy prompt dài không đóng băng mọi token giải mã trên GPU.

Mặc định production năm 2026 là cả ba. Bạn cần hiểu mỗi người làm gì vì các chế độ lỗi đều nằm trên bộ lập lịch chứ không phải model.

## Khái niệm

### PagedAttention như một hệ thống bộ nhớ ảo

Một KV cache là `num_layers × 2 × num_heads × head_dim × seq_len × bytes_per_element` cho mỗi chuỗi. Đối với Llama 3.3 70B ở 8192 tokens, đó là khoảng 1.25 GB cho mỗi trình tự trong BF16. Nếu bạn đặt trước 8192 vị trí cho mọi yêu cầu nhưng yêu cầu trung bình chỉ sử dụng 1500 tokens, bạn sẽ lãng phí khoảng 82% HBM mà bạn đã đặt trước. Lô cổ điển trả tiền lãng phí này.

PagedAttention vay mượn ý tưởng từ bộ nhớ ảo của hệ điều hành. KV cache không liền kề theo dãy. Nó được phân bổ trong các khối có kích thước cố định (mặc định là 16 tokens). Mỗi chuỗi có một bảng khối ánh xạ các vị trí token logic của nó với ID khối vật lý. Khi một chuỗi phát triển vượt qua các khối được phân bổ, một khối nữa sẽ được thêm vào. Khi nó kết thúc, các khối của nó trở lại hồ bơi.

Phân mảnh giảm từ 60-80% (cổ điển) xuống dưới 4% (PagedAttention). Bạn không bật PagedAttention bằng cờ — đây là trình phân bổ vLLM duy nhất ships. Núm là `--gpu-memory-utilization` (mặc định 0.9), cho vLLM biết cần dự trữ bao nhiêu HBM cho các khối KV sau khi tải trọng lượng và kích hoạt.

### Phân lô liên tục ở cấp độ lặp lại

"Dynamic batching" cũ đợi một cửa sổ (giả sử 10 ms) để lấp đầy một batch, sau đó chạy prefill + decode + decode + decode cho đến khi mọi trình tự kết thúc. Các trình tự nhanh rời đi sớm và không hoạt động trong khi GPU hoàn thành những trình tự chậm.

Hàng loạt liên tục hoạt động giữa mỗi bước giải mã. Gọi tập hợp các trình tự đang chạy là danh sách `RUNNING`. Ở mỗi lần lặp:

1. Bất kỳ chuỗi nào trong `RUNNING` vừa chạm vào EOS hoặc max_tokens đều bị xóa.
2. Trình lập lịch xem xét hàng đợi chờ. Nếu có các khối KV miễn phí, nó sẽ chấp nhận các trình tự mới (điền trước hoặc tiếp tục).
3. forward pass chạy trên bất cứ thứ gì hiện đang có trong `RUNNING`, phát ra một token mới cho mỗi chuỗi.

Kích thước batch không bao giờ được đệm vào một số cố định. Các chuỗi ở các vị trí khác nhau trong đầu ra của chúng chia sẻ một hợp nhất về phía trước. Vào năm 2026 vLLM, điều này được gọi là `V1 scheduler`. Bất biến chính: bộ lập lịch chạy một lần cho mỗi lần lặp giải mã, không phải một lần cho mỗi yêu cầu.

### Đổ đầy trước theo khối bảo vệ đuôi TTFT

Điền trước bị ràng buộc bằng điện toán. 32k-token prompt trên Llama 3.3 70B mất ~800 ms nạp trước tinh khiết trên một H100. Trong khi điền trước chạy, hãy giải mã tokens cho mọi trình tự khác trong batch chờ đợi. Trong vòng lặp phân phối, độ trễ token đầu tiên (TTFT) của một prompt dài trở thành độ trễ giữa các token (ITL) cho hàng chục người dùng khác.

Chèn trước theo khối chia phần điền trước thành các đoạn có kích thước cố định (mặc định là 512 tokens) và lên lịch cho từng đoạn dưới dạng một đơn vị. Giữa các khối, bộ lập lịch có thể nâng cao trình tự giải mã lên một token. Bạn đánh đổi một cú đánh độ trễ lấp đầy trước tuyệt đối nhỏ (vài ms mỗi khối) để lấy jitter thời gian giải mã thấp hơn nhiều. P99 ITL dưới tải hỗn hợp giảm từ ~50 ms xuống ~15 ms trong benchmarks đã xuất bản.

### Ba giá trị mặc định tương tác

Cả ba features giả định nhau. PagedAttention cung cấp cho bộ lập lịch một tài nguyên KV chi tiết để giao dịch. Việc phân lô liên tục cần nguồn lực chi tiết đó, vì vậy việc thừa nhận một trình tự mới không buộc phải cải tổ toàn cầu. Điền trước theo khối là một quyết định mà người lập lịch đưa ra trong cùng một danh sách `RUNNING` - đó là một policy lập lịch nữa, không phải là một hệ thống riêng biệt.

Bạn không cần phải biết mọi lá cờ. Bạn cần biết những gì bộ lập lịch tối ưu hóa: goodput dưới ngân sách khối KV, tùy thuộc vào việc cắt phần điền trước.

### Gotcha 2026 v0.18.0

Trong vLLM v0.18.0, bạn không thể kết hợp `--enable-chunked-prefill` với giải mã suy đoán model nháp (`--speculative-model`). Ngoại lệ được ghi nhận là giải mã suy đoán N-gram GPU trong bộ lập lịch V1. Các nhóm bật mọi lá cờ mà không đọc ghi chú phát hành sẽ gặp lỗi thời gian chạy khi khởi động, không phải hồi quy mềm. Nếu lợi nhuận đầu cơ của bạn đáng để cho phép điền trước theo từng đoạn, hãy xem lại lựa chọn - câu trả lời đúng vào năm 2026 thường là EAGLE-3 không có điền trước theo đoạn, không phải model nháp cộng với điền trước theo đoạn không biên dịch.

### Những con số bạn nên nhớ

- Llama 3.3 70B FP8, H100 SXM5, 128 đồng thời, cả ba bật: 2,200-2,400 tok/s.
- Cùng model, vLLM mặc định (không có phần nạp trước): ~1.800 tok/s.
- Cùng một model, ngây thơ PyTorch vòng lặp chuyển tiếp: ~600 tok/s.
- Chất thải phân mảnh KV trong PagedAttention khi tải production: <4%.
- P99 ITL dưới tải hỗn hợp: ~15 ms với prefill theo khối, ~50 ms không có.

### Trình lập lịch trông như thế nào

```
while True:
    finished = [s for s in RUNNING if s.is_done()]
    for s in finished: release_blocks(s); RUNNING.remove(s)

    while WAITING and have_free_blocks_for(WAITING[0]):
        s = WAITING.pop(0)
        allocate_initial_blocks(s)
        RUNNING.append(s)

    # schedule prefill chunks + decode in one batch
    batch = []
    for s in RUNNING:
        if s.in_prefill:
            batch.append(next_prefill_chunk(s))   # e.g. 512 tokens
        else:
            batch.append(decode_one_token(s))     # 1 token

    run_forward(batch)                            # one fused GPU call
```

`code/main.py` chính xác là vòng lặp này trong stdlib Python với số lượng token giả và độ trễ chuyển tiếp giả. Chạy nó cho thấy cách điền trước theo từng đoạn giữ cho các trình tự giải mã tồn tại trong quá trình điền trước dài.

```figure
tensor-parallel
```

## Ứng dụng

`code/main.py` mô phỏng bộ lập lịch kiểu vLLM với features có thể chuyển đổi. Chạy nó để xem:

- Chế độ `NAIVE`: một yêu cầu tại một thời điểm, không có hàng loạt.
- Chế độ `STATIC`: đệm và chờ, lô cổ điển.
- Chế độ `CONTINUOUS`: nhập và phát hành ở cấp độ lặp lại.
- Chế độ `CONTINUOUS + CHUNKED`: điền trước các lát cắt xen kẽ với giải mã.

Đầu ra hiển thị tổng thông lượng (tokens mỗi giây ảo), trung bình TTFT và P99 ITL. Hàng `CONTINUOUS + CHUNKED` sẽ chiếm ưu thế trên lưu lượng truy cập hỗn hợp.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-vllm-scheduler-reader.md`. Với config phục vụ (kích thước batch, sử dụng bộ nhớ KV, kích thước điền trước theo khối, config suy đoán), nó tạo ra chẩn đoán lập lịch trình nêu tên mặc định nào trong ba mặc định đang tắc nghẽn và những gì cần điều chỉnh.

## Bài tập

1. Chạy `code/main.py`. So sánh `STATIC` với `CONTINUOUS` trên khối lượng công việc với các yêu cầu ngắn và dài hỗn hợp. Khoảng cách thông lượng đến từ đâu - hiệu quả điền trước, hiệu quả giải mã hoặc độ trễ đuôi?
2. Sửa đổi bộ lập lịch đồ chơi để thêm `--max-num-batched-tokens`. Giá trị phù hợp cho H100 chạy Llama 3.3 70B FP8 là bao nhiêu? (Gợi ý: nó là một hàm của kích thước khối KV và số lượng khối tự do, không phải HBM thô.)
3. Đọc lại ghi chú phát hành vLLM v0.18.0. Những tổ hợp cờ nào loại trừ lẫn nhau? Liệt kê chúng.
4. Tính toán lãng phí phân mảnh KV cache cho trace 1.000 yêu cầu với trung bình 1.500 tokens đầu ra, tiêu chuẩn 600 tokens, theo (a) phân bổ liên tục cho mỗi yêu cầu ở mức tối đa 8192, (b) PagedAttention với khối 16 token.
5. Giải thích trong một đoạn tại sao phần điền trước giúp P99 ITL nhưng không giúp thông lượng riêng lẻ. Chiến thắng thông lượng đến từ đâu trong thực tế?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| PagedChú ý | "Thủ thuật KV" | Bộ phân bổ khối kích thước cố định cho KV cache; phân mảnh <4% |
| Bảng khối | "Bảng trang" | Ánh xạ mỗi trình tự từ vị trí token logic đến khối KV vật lý |
| Lô liên tục | "Batching động, nhưng đúng" | Admit/release quyết định được thực hiện mỗi lần giải mã |
| Điền trước theo khối | "Tách điền trước" | Chia phần điền trước dài thành 512 lát token xen kẽ với giải mã |
| TTFT | "lần token đầu tiên" | Điền trước + hàng đợi + mạng; bị chi phối bởi prefill ở prompts dài |
| ITL | "Độ trễ giữa các token" | Thời gian giữa tokens giải mã liên tiếp; bị chi phối bởi batch kích thước |
| Goodput | "thông lượng đáp ứng SLO" | Tokens/sec nơi mọi yêu cầu vẫn đạt được mục tiêu TTFT và ITL |
| Bộ lập lịch V1 | "Bộ lập lịch mới" | bộ lập lịch năm 2026 của vLLM; Giải mã thông số kỹ thuật N-gram là đường dẫn tương thích với khối điền trước |
| `--gpu-memory-utilization` | "Núm bộ nhớ" | Phần HBM dành riêng cho khối KV sau trọng số và kích hoạt |

## Đọc thêm

- [vLLM documentation — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/) — nguồn chính thức về khả năng tương thích điền trước và giải mã suy đoán.
- [vLLM Release Notes (NVIDIA)](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/index.html) — Nhịp phát hành năm 2026 và hành vi dành riêng cho phiên bản.
- [vLLM Blog — PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) - bài viết ban đầu vẫn xác định cách suy nghĩ về công cụ phân bổ.
- [PagedAttention paper (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180) - phân tích phân mảnh và thiết kế bộ lập lịch.
- [Aleksa Gordic — Inside vLLM](https://www.aleksagordic.com/blog/vllm) - hướng dẫn chi tiết về bộ lập lịch V1 với biểu đồ ngọn lửa.
