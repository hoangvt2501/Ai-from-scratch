# Tại sao Transformers - Các vấn đề với RNN

> RNN process tokens từng cái một. Transformers process tất cả tokens cùng một lúc. Đặt cược kiến trúc duy nhất đó đã thay đổi mọi đường cong mở rộng quy mô trong deep learning sau năm 2017.

**Loại:** Học
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 3 (Deep Learning Core), Giai đoạn 5 · 09 (Trình tự đến trình tự), Giai đoạn 5 · 10 (Cơ chế Attention)
**Thời lượng:** ~45 phút

## Vấn đề

Trước năm 2017, mọi chuỗi hiện đại model trên hành tinh - ngôn ngữ, dịch thuật, lời nói - đều là một mạng nơ-ron lặp đi lặp lại. LSTM và GRU đã giành được benchmarks dịch thuật tương đương ImageNet trong nửa thập kỷ. Chúng là công cụ duy nhất mà bất kỳ ai có.

Họ có ba điểm yếu chết người. Tính toán tuần tự có nghĩa là bạn không thể song song dọc theo trục thời gian: token `t+1` cần trạng thái ẩn từ token `t`. Một chuỗi 1.024 token có nghĩa là 1.024 bước nối tiếp trên một GPU có thể thực hiện 1.000.000 thao tác dấu phẩy động mỗi chu kỳ. Training thời gian đồng hồ treo tường được chia tỷ lệ tuyến tính với độ dài trình tự trên phần cứng được thiết kế song song.

Biến mất gradients có nghĩa là thông tin ở phía sau 50 tokens đã được nén qua 50 phi tuyến tính. Các đơn vị tái phát có cổng (LSTM, GRU) làm mềm sự nghiền nát nhưng không bao giờ loại bỏ nó. Sự phụ thuộc tầm xa - "cuốn sách tôi đọc mùa hè năm ngoái trên máy bay đến Kyoto là..." - thường thất bại.

Các trạng thái ẩn có chiều rộng cố định có nghĩa là encoder ép toàn bộ chuỗi nguồn vào một vector duy nhất trước khi decoder nhìn thấy bất cứ thứ gì. Không quan trọng nguồn là 5 tokens hay 500; nút cổ chai có hình dạng giống nhau.

Bài báo năm 2017 "Attention là tất cả những gì bạn cần" đã đề xuất một điều triệt để: loại bỏ hoàn toàn sự tái phát. Hãy để mọi vị trí tham gia vào mọi vị trí khác song song. Huấn luyện một phép nhân ma trận lớn thay vì 1.024 phép nhân tuần tự.

Kết quả thống trị mọi phương thức vào năm 2026. Ngôn ngữ (GPT-5, Claude 4, Llama 4), thị giác (ViT, DINOv2, SAM 3), âm thanh (Whisper), sinh học (AlphaFold 3), robot (RT-2). Cùng một khối, đầu vào khác nhau.

## Khái niệm

![RNN sequential compute vs Transformer parallel attention](../assets/rnn-vs-transformer.svg)

**Tái phát như một nút thắt cổ chai.** RNN tính toán `h_t = f(h_{t-1}, x_t)`. Mỗi bước phụ thuộc vào trước đó. Bạn không thể tính toán `h_5` trước `h_4`. Trên GPUs hiện đại với 10.000+ lõi song song, điều này lãng phí 99% silicon trong một chuỗi dài.

**Attention dưới dạng chương trình phát sóng.** Self-attention tính toán đồng thời `output_i = sum_j(a_ij * v_j)` cho mọi cặp `(i, j)`. Toàn bộ ma trận attention N×N điền vào một matmul hàng loạt. Không có bước nào phụ thuộc vào bước khác. GPUs thích nó.

**Tăng tốc không phải là hằng số.** Đó là sự khác biệt giữa độ sâu nối tiếp và độ sâu nối tiếp `O(1)` `O(N)`. Trong thực tế, transformers huấn luyện nhanh hơn 5–10× mỗi epoch trên phần cứng phù hợp ở N = 512 và khoảng cách mở rộng theo độ dài trình tự cho đến khi bạn chạm vào bức tường bộ nhớ `O(N²)` của attention (mà Flash sau đó Attention sửa chữa - xem Bài 12).

**Chi phí transformers bao nhiêu.** Bộ nhớ Attention có thể thay đổi theo `O(N²)`. Đối với bối cảnh 2K, tốt. Đối với ngữ cảnh 128K, bạn cần windows trượt, ngoại suy RoPE, lát gạch Flash Attention hoặc các biến thể attention tuyến tính. Sự tái phát `O(N)` cả về thời gian và trí nhớ; transformers đánh đổi thời gian để lấy trí nhớ và sau đó giành lại thời gian thông qua tính song song.

**Sự thay đổi bias quy nạp.** RNN giả định tính địa phương và gần đây. Transformers không giả định gì - mọi cặp đều là ứng cử viên cho attention. Đó là lý do tại sao transformers cần nhiều dữ liệu hơn để huấn luyện tốt nhưng mở rộng quy mô hơn nữa khi họ có nó. Chinchilla (2022) đã chính thức hóa điều này: nếu có đủ tokens, một transformer luôn đánh bại RNN có số lượng parameter bằng nhau.

## Tự xây dựng

Không có mạng nơ-ron ở đây - chúng tôi mô phỏng nút cổ chai cốt lõi bằng số để bạn cảm nhận được khoảng trống trên máy tính xách tay của mình.

### Bước 1: đo độ sâu nối tiếp

Xem `code/main.py`. Chúng ta xây dựng hai chức năng. Một mã hóa một chuỗi dưới dạng một chuỗi các phép cộng (nối tiếp, giống như RNN). Người ta mã hóa nó dưới dạng rút gọn song song (phát sóng, như attention). Cùng một toán học, biểu đồ phụ thuộc khác nhau.

```python
def rnn_style(xs):
    h = 0.0
    for x in xs:
        h = 0.9 * h + x   # can't parallelize: h depends on previous h
    return h

def attention_style(xs):
    return sum(xs) / len(xs)  # every x is independent
```

Chúng ta tính thời gian cả hai trên các chuỗi lên đến 100.000 phần tử. Phiên bản RNN là O (N) và một CPU pipeline duy nhất. Ngay cả trong Python thuần túy, việc rút gọn kiểu attention đánh bại nó ở độ dài ≥ 1.000 vì `sum()` của Python được thực hiện trong C và lặp lại mà không cần thông dịch chi phí cho mỗi bước.

### Bước 2: đếm các phép toán lý thuyết

Cả hai thuật toán đều làm N cộng. Sự khác biệt là *độ sâu phụ thuộc*: có bao nhiêu hoạt động phải xảy ra tuần tự trước khi hoạt động tiếp theo có thể bắt đầu. Độ sâu RNN = N. Độ sâu Attention = log(N) với việc giảm cây hoặc 1 với quét song song. Chiều sâu, không phải số lượng op, quyết định thời gian GPU.

### Bước 3: chia tỷ lệ thực nghiệm trên các chuỗi dài

Chúng ta in một bảng thời gian làm cho khoảng cách O (N) hiển thị. Trên máy tính xách tay Mac 2026, chuỗi dưới 1.000 phần tử là quá nhanh để đo lường. Trình tự 100.000 cho thấy một bản quét tuyến tính rõ ràng. Mở rộng quy mô đó lên 16.384 token transformer với LSTM 12 lớp tương đương và bạn sẽ thấy lý do tại sao đồng hồ treo tường training lại là một công cụ cản trở vào năm 2016.

## Ứng dụng

Khi nào vẫn nên chọn RNN vào năm 2026:

| Tình huống | Chọn |
|-----------|------|
| Streaming inference, từng token một, bộ nhớ không đổi | RNN hoặc model không gian trạng thái (Mamba, RWKV) |
| Chuỗi rất dài (>1 triệu tokens) trong đó bộ nhớ attention bùng nổ | attention tuyến tính, Mamba 2, Linh cẩu |
| Thiết bị biên không có bộ tăng tốc matmul | RNN có thể tách biệt theo chiều sâu vẫn giành chiến thắng trên FLOPs/watt |
| Bất kỳ thứ gì khác (training, inference hàng loạt, ngữ cảnh lên đến 128K) | Transformer |

Các models không gian trạng thái (SSM) như Mamba về cơ bản là các RNN với tham số hóa có cấu trúc mang lại cho chúng những gì tốt nhất của cả hai: bộ nhớ quét `O(N)`, training song song thông qua quét chọn lọc. Chúng khôi phục 90% chất lượng transformer với khả năng mở rộng ngữ cảnh dài tốt hơn. Vào năm 2026, hầu hết các phòng thí nghiệm biên giới huấn luyện SSM+transformer models lai (ví dụ: Jamba, Samba) - sự tái phát không chết, nó là một thành phần.

## Sản phẩm bàn giao

Xem `outputs/skill-architecture-picker.md`. skill chọn một kiến trúc cho một vấn đề trình tự mới với các ràng buộc về độ dài, thông lượng và ngân sách training. Nó phải luôn từ chối đề xuất RNN thuần túy cho training chạy trên 1B tokens mà không nêu rõ sự đánh đổi.

## Bài tập

1. **Dễ dàng.** Lấy `rnn_style` từ `code/main.py` và thay thế trạng thái ẩn vô hướng bằng độ dài 64 vector trạng thái ẩn. Đo lường lại. Chi phí nối tiếp tăng bao nhiêu với kích thước trạng thái ẩn?
2. **Trung bình.** Thực hiện tổng tiền tố song song (quét Hillis-Steele) trong Python thuần túy. Xác minh rằng nó tạo ra cùng một đầu ra số như quét nối tiếp trên độ dài 1024. Đếm độ sâu.
3. **Cứng.** Chuyển giảm kiểu attention thành PyTorch trên GPU. Thời gian cả hai khi bạn quét độ dài trình tự từ 64 đến 65.536. Vẽ và giải thích hình dạng đường cong.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Tái phát | "RNN là tuần tự" | Tính toán trong đó bước `t` phụ thuộc vào bước `t-1`, buộc thực thi nối tiếp dọc theo trục thời gian. |
| Độ sâu nối tiếp | "Biểu đồ sâu như thế nào" | Chuỗi hoạt động phụ thuộc dài nhất; đồng hồ treo tường giới hạn ngay cả trên phần cứng vô hạn. |
| Attention | "Hãy để tokens nhìn nhau" | Tổng có trọng số `sum_j a_ij v_j` trong đó `a_ij` đến từ điểm tương tự giữa vị trí i và j. |
| Context window | "Người model nhìn thấy bao nhiêu" | Số vị trí mà một lớp attention có thể lấy làm đầu vào; tỷ lệ chi phí bộ nhớ bậc hai ở đây. |
| bias quy nạp | "Các giả định được đưa vào kiến trúc" | Prior về dữ liệu trông như thế nào; CNN giả định sự bất biến của bản dịch, RNN giả định tính gần đây. |
| model không gian trạng thái | "RNN với đại số đằng sau nó" | Lặp lại được tham số hóa cho training song song thông qua ma trận không gian trạng thái có cấu trúc. |
| Nút thắt cổ chai bậc hai | "Tại sao bối cảnh lại tốn kém như vậy" | Bộ nhớ Attention = `O(N²)` độ dài trình tự; Flash Attention ẩn các hằng số, không phải tỷ lệ. |

## Đọc thêm

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) - tờ báo đã giết chết sự tái diễn trong NLP chính thống.
- [Bahdanau, Cho, Bengio (2014). Neural MT by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) - nơi attention được sinh ra, bắt vít vào một RNN.
- [Hochreiter, Schmidhuber (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) - bài báo LSTM gốc, để ghi lại.
- [Gu, Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) - câu trả lời lặp đi lặp lại hiện đại cho transformers.
