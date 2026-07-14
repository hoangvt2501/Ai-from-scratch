# Attention thưa thớt bản địa (DeepSeek NSA)

> Ở tokens 64k, attention chiếm 70-80% độ trễ giải mã. Mỗi phòng thí nghiệm model mở đều có kế hoạch khắc phục. NSA của DeepSeek (bài báo tốt nhất ACL 2025) là một trong những bị mắc kẹt: ba attention branches song song - tokens hạt thô nén, tokens chi tiết được giữ lại có chọn lọc và windows trượt cho ngữ cảnh cục bộ - được kết hợp thông qua một cổng đã học. Nó được căn chỉnh phần cứng (thân thiện với hạt nhân), có thể huấn luyện nguyên bản (hoạt động ở training trước, không được bắt vít ở inference) và trên giải mã 64k, nó chạy nhanh hơn FlashAttention trong khi phù hợp hoặc đánh bại chất lượng attention đầy đủ. Bài học này xây dựng ba branches từ đầu đến cuối và cho thấy lý do tại sao sự thưa thớt là có thể phân biệt từ đầu đến cuối.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 7 · 12 (KV cache, flash-attention), Giai đoạn 7 · 15 (attention biến thể), Giai đoạn 10 · 16 (attention vi sai)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Nêu ba attention branches của NSA và những gì mỗi người nắm bắt.
- Giải thích tại sao NSA là "có thể huấn luyện nguyên bản" trong khi prior phương pháp thưa thớt attention chỉ inference.
- Tính toán mức tiết kiệm điện toán attention của NSA so với attention đầy đủ ở ngữ cảnh 64k như một hàm của kích thước khối nén và top-k lựa chọn.
- Thực hiện kết hợp ba branch trong Python stdlib trên một trình tự tổng hợp ngắn và xác minh các trọng lượng cổng hoạt động.

## Vấn đề

attention đầy đủ ở độ dài trình tự N tốn `O(N^2)` thời gian và `O(N)` KV cache cho mỗi lớp. Ở tokens 64k, số băng thông tính toán và bộ nhớ là thảm họa. Ước tính lý thuyết đo lường từ bài báo của NSA: attention chiếm 70-80% tổng độ trễ giải mã ở mức 64k. Mọi thứ xuôi dòng - TTFT, tokens/sec, chi phí trên một triệu tokens - bị chi phối bởi chi phí attention.

attention thưa thớt là câu trả lời rõ ràng. Prior nỗ lực rơi vào hai nhóm. Sự thưa thớt của mẫu cố định (cửa sổ trượt, sải bước, khối cục bộ) vứt bỏ thông tin và thất bại trong các tác vụ recall tầm xa. Độ thưa thớt Inference thời gian (cắt tỉa KV cache, H2O, StreamingLLM) được áp dụng cho một model được huấn luyện trước trên attention dày đặc và chỉ phục hồi một phần nhỏ của khả năng tăng tốc vì model chưa bao giờ được yêu cầu định tuyến thông tin thông qua mẫu thưa thớt.

Native Sparse Attention (Yuan et al., DeepSeek + PKU + UW, bài báo tốt nhất ACL 2025, arXiv:2502.11089) thực hiện cả hai: một mô hình thưa thớt mà model học được trong quá trình training trước, được thực hiện như một thuật toán căn chỉnh hạt nhân thực sự tiết kiệm điện toán ở mức inference. Hai năm kể từ bây giờ, NSA hoặc hậu duệ trực tiếp là attention mặc định trên mọi model ngữ cảnh dài biên giới.

## Khái niệm

### Ba branches song song

Đối với mỗi truy vấn, NSA chạy attention ba lần, chống lại ba quan điểm khác nhau của KV cache:

1. **branch nén.** Tokens được nhóm thành các khối có kích thước `l` (thường là 32 hoặc 64). Mỗi khối được nén thành một token tóm tắt duy nhất thông qua một MLP nhỏ đã học. Truy vấn tham gia vào các tokens nén này, có được một cái nhìn chi tiết về toàn bộ trình tự.

2. **Đã chọn branch.** Sử dụng điểm số attention từ branch nén, các khối top-k có liên quan nhất đến truy vấn hiện tại được xác định. Các tokens chi tiết (không nén) từ các khối đó được đọc và truy vấn tham gia tất cả chúng. Hãy coi branch attention nén là tín hiệu định tuyến cho lựa chọn.

3. **branch cửa sổ trượt.** Truy vấn tham gia vào `W` tokens gần đây nhất (thường là 512) cho ngữ cảnh cục bộ. branch này nắm bắt các mẫu tầm ngắn nặng về cấu trúc (cú pháp, đồng tham chiếu cục bộ) mà hai mẫu còn lại có thể bỏ sót.

Ba đầu ra branch được kết hợp thông qua một cổng mỗi vị trí đã học:

```
out = g_cmp * out_cmp + g_sel * out_sel + g_win * out_win
```

`g_cmp, g_sel, g_win` là trọng số cổng từ một MLP nhỏ trên truy vấn. Chúng không cần phải tính tổng thành 1 - chúng có thể trọng số branches một cách độc lập.

### Tại sao điều này là "có thể huấn luyện được"

Bước lựa chọn (top-k khối) là rời rạc. Các hoạt động rời rạc phá vỡ gradient luồng. Prior công việc attention thưa thớt hoặc bỏ qua backprop thông qua lựa chọn (giới hạn training) hoặc sử dụng thư giãn liên tục không mang lại độ thưa thớt thực sự ở inference.

NSA né tránh điều này: branch attention nén LÀ một attention hạt thô có thể phân biệt được trên toàn bộ chuỗi. Hoạt động top-k chỉ sử dụng lại các điểm attention hàng đầu từ branch nén để chọn khối hạt mịn nào để tải. Gradients chảy qua các điểm branch nén (ảnh hưởng đến cả đầu ra nén VÀ logic lựa chọn), và đóng góp của các khối được chọn vào đầu ra cuối cùng cũng có thể phân biệt được. Hoạt động `top_k` không thể vi phân là không hoạt động trên biểu đồ tính toán chuyển tiếp - nó chỉ kiểm soát khối nào được tải từ bộ nhớ.

Đây là lý do tại sao NSA có thể được sử dụng trước khi training từ đầu đến cuối. model học cách định tuyến thông tin qua ba branches cùng nhau, tạo ra một mô hình thưa thớt mà inference thực sự mang lại tốc độ như đã hứa.

### Hạt nhân được căn chỉnh phần cứng

Hạt nhân của NSA được thiết kế cho hệ thống phân cấp bộ nhớ GPU hiện đại. Hạt nhân tải các truy vấn theo nhóm GQA (vòng lặp bên ngoài), tìm nạp các khối KV thưa thớt tương ứng trên mỗi nhóm (vòng lặp bên trong) và chạy attention trên SRAM. Bởi vì mỗi nhóm truy vấn nhìn thấy các khối được chọn giống nhau (lựa chọn là mỗi nhóm truy vấn, không phải trên mỗi đầu truy vấn), tải KV được khấu hao trên toàn nhóm. Cường độ số học vẫn cao.

Bài báo báo cáo các hạt nhân Triton chạy nhanh hơn 9 lần so với FlashAttention trên các giải mã 64k, với tỷ lệ tăng tốc tăng theo độ dài trình tự. Các hạt nhân tiến và lùi đều được cung cấp.

### Ngân sách điện toán

Giả sử `N` là độ dài trình tự, `l` kích thước khối nén, `k` số lượng lựa chọn top-k `w` cửa sổ trượt, `b` kích thước khối đã chọn (thường bằng `l`).

- branch nén: `O(N/l)` khóa cho mỗi truy vấn, vì vậy `O(N * N / l)` tổng số.
- branch đã chọn: `O(k * b)` khóa cho mỗi truy vấn, vì vậy `O(N * k * b)`.
- branch trượt: `O(w)` khóa cho mỗi truy vấn, vì vậy `O(N * w)`.

Tổng: `O(N * (N/l + k*b + w))`.

Với `N = 64k, l = 64, k = 16, b = 64, w = 512`: chi phí cho mỗi truy vấn là `1000 + 1024 + 512 = 2536 keys`. Toàn bộ attention là `64000 keys`. Giảm điện toán gấp 25 lần.

Với `N = 128k, l = 64, k = 16, b = 64, w = 512`: chi phí cho mỗi truy vấn là `2000 + 1024 + 512 = 3536 keys`. attention đầy đủ là `128000 keys`. Giảm 36 lần. Lợi ích tăng lên theo độ dài trình tự, đó là toàn bộ vấn đề.

### So sánh nó như thế nào

| Phương pháp | Có thể phân biệt | Tăng tốc inference thực sự | recall tầm xa |
|--------|---------------|----------------------|-------------------|
| Chỉ cửa sổ trượt | Có | Có | không thành công |
| Sải bước / khối thưa thớt | Có | Có | Một phần |
| Cắt tỉa KV (H2O, StreamingLLM) | N/A (inference lần) | Có | Một phần |
| MoBA (Moonshot) | Một phần | Có | tốt |
| NSA | Có (gốc) | Có (9x ở 64K) | trận đấu đầy đủ attention |

MoBA (Moonshot, arXiv:2502.13189) đã được xuất bản đồng thời và thực hiện cách tiếp cận tương tự ba là tốt hơn một, áp dụng nguyên tắc MoE cho các khối attention. NSA và MoBA là hai kiến trúc cần biết cho bối cảnh dài hạn trước training năm 2026.

```figure
sliding-window-attention
```

## Tự xây dựng

`code/main.py` thực hiện ba branches trên một chuỗi tổng hợp ngắn và hiển thị:

- MLP nén (một đường cơ sở trung bình đơn giản được sử dụng để sư phạm rõ ràng; NSA thực sự sử dụng MLP đã học).
- Lựa chọn khối top-k được thúc đẩy bởi điểm số branch nén.
- Cửa sổ trượt attention ở `w` tokens cuối cùng.
- Sự kết hợp có cổng.
- Bản in tính toán so với attention đầy đủ.

### Bước 1: nén tokens thành khối

```python
def compress(K, l):
    n = len(K)
    n_blocks = (n + l - 1) // l
    out = []
    for b in range(n_blocks):
        start, end = b * l, min((b + 1) * l, n)
        block = K[start:end]
        summary = [sum(row[d] for row in block) / len(block) for d in range(len(K[0]))]
        out.append(summary)
    return out
```

### Bước 2: branch attention nén

Chạy softmax attention truy vấn đối với các khóa nén. Điểm branch nén gấp đôi như tín hiệu cho lựa chọn top-k.

### Bước 3: top-k lựa chọn khối

Chọn các chỉ số của `k` khối nén có điểm cao nhất. Tải tokens không nén ban đầu từ các khối đó và chạy attention trên chúng.

### Bước 4: attention cửa sổ trượt

Lấy `w` tokens cuối cùng và chạy attention tiêu chuẩn chống lại họ.

### Bước 5: cổng + kết hợp

Một MLP nhỏ trên truy vấn tạo ra ba trọng số cổng. Đầu ra cuối cùng là tổng có trọng số của ba đầu ra branch.

### Bước 6: tính toán đếm

In số khóa tham gia mỗi truy vấn cho mỗi branch và tổng số. So sánh với `N` (attention đầy đủ). Trên một tổng hợp 1024 token với `l = 32, k = 4, w = 128`, NSA thấy `32 + 128 + 128 = 288` khóa cho mỗi truy vấn so với 1024 cho toàn bộ attention - ít hơn 3,5 lần.

## Ứng dụng

NSA đang shipping trong bối cảnh dài hạn của DeepSeek trước khi training pipeline. Tình trạng hội nhập trong inference stacks công cộng tính đến tháng 4 năm 2026:

- **DeepSeek nội bộ**: trọng số gốc, được xuất bản sử dụng NSA hoặc DSA kế nhiệm của nó (Deepseek Sparse Attention).
- **vLLM**: hỗ trợ thử nghiệm của NSA trong quá trình phát triển cho trọng số DeepSeek-V3.x.
- **SGLang**: NSA benchmarks xuất bản; production đường dẫn theo sau vLLM.
- **llama.cpp / CPU**: không được hỗ trợ; chi phí của quá trình phân hủy hạt nhân là không đáng giá ở CPU thông lượng.

Khi nào cần tiếp cận NSA:

- Chạy trước khi training hoặc tiếp tục chạy training nhắm mục tiêu vào bối cảnh hơn 64 nghìn với ngân sách điện toán nghiêm túc.
- Inference về checkpoints ngữ cảnh dài của DeepSeek. Trọng số là bản địa của NSA.

Khi nào không:

- Phục vụ một model được huấn luyện trước attention dày đặc hiện có. Bạn không thể trang bị thêm NSA mà không tiếp tục training.
- Bối cảnh dưới 16k. Chi phí ba branch chiếm ưu thế trong việc tiết kiệm.
- Trò chuyện tương tác Batch-1. Lợi ích giải mã nhạy cảm với độ trễ, nhưng chỉ ở ngữ cảnh dài.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-nsa-integrator.md`. Với một đặc tả trước khi chạy training ngữ cảnh dài, nó tạo ra một kế hoạch tích hợp NSA: kích thước khối nén, top-k, cửa sổ trượt, chiều rộng cổng MLP, lựa chọn hạt nhân và các đánh giá ngữ cảnh dài cụ thể sẽ biện minh cho sự thay đổi kiến trúc.

## Bài tập

1. Chạy `code/main.py` trên bản tổng hợp 1024 token. Quét `(l, k, w)` qua ba cài đặt trước và số lượng điện toán in. Xác định cài đặt trước đạt được số lượng khóa thấp nhất cho mỗi truy vấn trong khi vẫn giữ recall 95% so với attention đầy đủ trong thử nghiệm trong đống cỏ khô.

2. Thay thế máy nén nhóm trung bình bằng một MLP nhỏ đã học (2 lớp, ẩn 32). Huấn luyện nó trên một tác vụ tổng hợp trong đó tín hiệu là giá trị trung bình của một khối. Đo khoảng cách perplexity so với đường cơ sở nhóm trung bình trên dữ liệu được giữ lại.

3. Triển khai cổng MLP. Nó lấy truy vấn làm đầu vào và xuất ra ba vô hướng. Cho thấy rằng cổng hoạt động hợp lý: trọng số gần như đồng đều trên các truy vấn ngẫu nhiên, trọng lượng nặng trên branch đã chọn khi truy vấn chạm vào một khối phía sau.

4. Tính toán ngân sách bộ nhớ KV cache cho một model 70B hỗ trợ NSA ở ngữ cảnh 128k. Đầu KV là 8, đầu mờ 128, BF16. So sánh với full attention và MLA (Giai đoạn 10 · 14 hiển thị số của MLA). Xác định độ dài trình tự trong đó branch KV cache chi tiết của NSA bằng full attention.

5. Đọc Phần 4 của bài báo của NSA (arXiv:2502.11089) và giải thích trong ba câu lý do tại sao điểm attention của branch nén được sử dụng lại để lựa chọn top-k thay vì tính điểm định tuyến riêng biệt. Gắn câu trả lời với luồng gradient.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| branch nén | "Tầm nhìn thô" | Attention qua các khóa trung bình khối cung cấp ngữ cảnh toàn cầu trong các khóa O (N/l) cho mỗi truy vấn |
| Các branch chọn lọc | "Top-k khối" | attention chi tiết trên các khối `k` có điểm branch nén cao nhất |
| Cửa sổ trượt | "Bối cảnh địa phương" | Attention trong `W` tokens qua đối với các mô hình ngắn hạn |
| Khả năng huấn luyện bản địa | "Huấn luyện trước với sự thưa thớt" | Mô hình thưa thớt được học trong quá trình training trước, không được bắt vít khi inference |
| Kích thước khối nén l | "Quy mô nhóm cho chế độ xem thô" | Có bao nhiêu tokens đưa merged vào một bản tóm tắt; 32-64 điển hình |
| Top-k | "Các khối cần giữ" | Số lượng khối nén có tokens không nén được đọc; 16 điển hình |
| Cửa sổ trượt W | "Bán kính attention cục bộ" | Điển hình là 512; ngắn hơn làm tổn thương tính mạch lạc cục bộ, lãng phí lâu hơn tính toán |
| Cổng Branch | "Làm thế nào để kết hợp ba" | Đầu ra MLP cho mỗi vị trí có trọng số đóng góp của ba branches |
| Phần cứng alignment | "Sự thưa thớt thân thiện với nhân" | Mẫu thưa thớt được chọn để hạt nhân GPU thực tế đạt được tốc độ lý thuyết |
| DSA | "Người kế nhiệm NSA" | Deepseek Sparse Attention, kiến trúc theo NSA trong dòng dõi của DeepSeek |

## Đọc thêm

- [Yuan et al. — Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (arXiv:2502.11089, ACL 2025 Best Paper)](https://arxiv.org/abs/2502.11089) — tờ giấy
- [DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) – gia đình kiến trúc mà NSA nhắm mục tiêu
- [Moonshot AI — MoBA: Mixture of Block Attention for Long-Context LLMs (arXiv:2502.13189)](https://arxiv.org/abs/2502.13189) - làm việc đồng thời, attention kiểu MoE qua các khối
- [Beltagy et al. — Longformer: The Long-Document Transformer (arXiv:2004.05150)](https://arxiv.org/abs/2004.05150) — nguồn gốc cửa sổ trượt
- [Xiao et al. — StreamingLLM: Efficient Streaming Language Models with Attention Sinks (arXiv:2309.17453)](https://arxiv.org/abs/2309.17453) - NSA cơ sở thưa thớt inference thời gian cải thiện
- [Dao et al. — FlashAttention-2 (arXiv:2307.08691)](https://arxiv.org/abs/2307.08691) - các hạt nhân NSA cơ bản đầy đủ attention đánh bại ở mức 64k
