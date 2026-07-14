# Mô hình tự hồi quy trực quan (VAR): Dự đoán quy mô tiếp theo

> Khuếch tán models lấy mẫu lặp đi lặp lại theo thời gian (các bước khử nhiễu). Các mẫu VAR lặp đi lặp lại theo tỷ lệ - nó dự đoán token 1x1, sau đó là 2x2, sau đó là 4x4, cho đến độ phân giải cuối cùng, mỗi thang đo điều hòa trên tỷ lệ trước đó. Bài báo năm 2024 cho thấy VAR phù hợp GPT-style luật tỷ lệ để tạo hình ảnh và đánh bại DiT ở cùng ngân sách tính toán. Bài học này xây dựng cơ chế cốt lõi.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (with PyTorch)
**Kiến thức tiên quyết:** Giai đoạn 7 Bài 03 (Multi-Head Attention), Giai đoạn 8 Bài 06 (DDPM)
**Thời lượng:** ~90 phút

## Vấn đề

Thế hệ tự hồi quy thống trị mô hình ngôn ngữ vì nó mở rộng quy mô có thể dự đoán được: tính toán nhiều hơn, parameters hơn, perplexity thấp hơn, đầu ra tốt hơn. Tạo hình ảnh có hai lần thử nghiệm AR chính trước năm 2024: PixelRNN/PixelCNN (pixel-by-pixel) và DALL-E 1 / Parti / MuseGAN (token-by-token trên mã VQ-VAE).

Cả hai đều gặp phải vấn đề về thứ tự thế hệ. Pixel và tokens được sắp xếp theo lưới 2D, nhưng model AR phải truy cập chúng theo thứ tự raster 1D. Một pixel góc sớm không biết hình ảnh cuối cùng sẽ trở thành gì. Chất lượng tạo giảm tỷ lệ thấp hơn so với văn bản GPT-on và không bao giờ đạt được chất lượng model khuếch tán ở điện toán phù hợp.

VAR khắc phục sự cố thứ tự tạo bằng cách thay đổi những gì đang được tạo. Thay vì dự đoán từng hình ảnh tokens một trong không gian, VAR dự đoán toàn bộ hình ảnh ở độ phân giải tăng dần. Bước 1: dự đoán token 1x1 (hình ảnh tổng thể "tóm tắt"). Bước 2: dự đoán lưới tokens 2x2 (features thô hơn). Bước 3: dự đoán lưới 4x4. Bước K: dự đoán lưới (H/8)x(W/8) cuối cùng.

Mỗi thang đo tham gia vào tất cả các thang đo trước đó (nhân quả theo "thứ tự thang đo") và song song trong thang đo riêng của nó. Vấn đề trật tự biến mất: toàn bộ hình ảnh ở tỷ lệ k được tạo ra trong một lần transformer.

## Khái niệm

### Máy Tokenizer đa thang đo VQ-VAE

VAR cần **tokenizer rời rạc đa tỷ lệ**. Đối với hình ảnh x, nó tạo ra một chuỗi các lưới token có độ phân giải cao hơn dần dần:

```
x -> encoder -> latent f
f -> tokenize at 1x1: token grid z_1 of shape (1, 1)
f -> tokenize at 2x2: token grid z_2 of shape (2, 2)
...
f -> tokenize at (H/p)x(W/p): token grid z_K of shape (H/p, W/p)
```

Mỗi z_k sử dụng cùng một cuốn sách mã (kích thước điển hình 4096-16384). tokenization ở mỗi thang đo không độc lập - nó được huấn luyện để tổng các số dư ở mỗi thang đo tái tạo f:

```
f ≈ upsample(embed(z_1), target_size) + ... + upsample(embed(z_K), target_size)
```

Đây là một biến thể **VQ **còn lại. Thang đo k nắm bắt những gì tỷ lệ 1..k-1 bị bỏ lỡ. Decoder lấy tổng của tất cả các tỷ lệ embeddings và tạo ra hình ảnh.

tokenizer VQ đa thang đo được huấn luyện một lần (như VQGAN) và sau đó được đông lạnh. Tất cả các công việc tạo ra được thực hiện bởi model tự hồi quy ở trên cùng.

### Dự đoán quy mô tiếp theo

model tổng quát là một transformer nhìn thấy tokens từ tất cả các thang đo trước đó và dự đoán tokens ở thang đo tiếp theo.

Cấu trúc trình tự đầu vào:
```
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

Vị trí embeddings mã hóa cả chỉ số tỷ lệ và vị trí không gian trong thang đo. Attention là nhân quả theo thứ tự tỷ lệ: token ở tỷ lệ k, vị trí (i, j) có thể tham gia vào tất cả các tokens ở thang đo 1..k và tokens ở tỷ lệ k xuất hiện sớm hơn theo bất kỳ thứ tự nội thang nào được sử dụng (VAR sử dụng attention vị trí cố định không có nhân quả nội thang đo - tất cả các vị trí trong thang đo được dự đoán song song).

Training loss: Ở mỗi thang điểm K, dự đoán tokens z_k cho tất cả các tokens thang prior. loss entropy chéo trên các mã VQ rời rạc. Cấu trúc tương tự như GPT ngoại trừ "chuỗi" bây giờ có cấu trúc tỷ lệ.

### Thế hệ

Tại inference:
```
generate z_1 = sample from p(z_1)                    # 1 token
generate z_2 = sample from p(z_2 | z_1)              # 4 tokens in parallel
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16 tokens in parallel
...
decode: f = sum of embed-and-upsample scales 1..K
image = VAE_decoder(f)
```

Đối với K = 10 thang đo, việc tạo là 10 transformer lần chuyển tiếp. Mỗi lần vượt qua tạo ra toàn bộ thang đo song song - không có tự hồi quy trên mỗi token trong một thang đo. Đối với hình ảnh 256x256, đây là khoảng 10 đường chuyền so với 28-50 của DiT.

### Tại sao Next-Scale chiến thắng Next-Token

Ba chiến thắng về cấu trúc:
1. **Từ thô đến mịn phù hợp với số liệu thống kê hình ảnh tự nhiên.** Nhận thức thị giác của con người và datasets hình ảnh đều thể hiện các quy tắc phụ thuộc vào tỷ lệ: cấu trúc tần số thấp ổn định và có thể dự đoán được; Chi tiết tần số cao có điều kiện đối với nội dung tần số thấp. Dự đoán quy mô tiếp theo khai thác điều này.
2. **Tạo song song trong quy mô.** Không giống như GPT-style token AR, VAR tạo ra tất cả tokens ở một tỷ lệ trong một bước. Độ dài phát hiệu quả là quy mô log thay vì tuyến tính.
3. **Không có thứ tự phát bias.** Tokens ở quy mô k xem tất cả thang đo k-1; Không có bias "trái" hoặc "trên" buộc tokens sớm phải commit trước khi ngữ cảnh muộn có sẵn.

### Luật tỷ lệ

Tian và cộng sự đã chứng minh rằng VAR tuân theo đường cong tỷ lệ định luật lũy thừa cho FID trên ImageNet - giống như GPT làm với perplexity. Tăng gấp đôi parameters hoặc tính toán đáng tin cậy sẽ giảm một nửa lỗi. Đây là model tạo hình ảnh đầu tiên thể hiện loại hành vi mở rộng này rõ ràng như ngôn ngữ models. Kết quả là các dự đoán quy mô VAR trở nên có thể dự đoán được từ tính toán, không phải phỏng đoán thực nghiệm trên mỗi kiến trúc.

### Mối quan hệ với sự khuếch tán

VAR và khuếch tán chia sẻ cùng một câu chuyện nén dữ liệu: cả hai đều chia bài toán tạo thành một chuỗi các bài toán con dễ dàng hơn.

- Khuếch tán: dần dần thêm nhiễu, học cách hoàn tác một bước.
- VAR: dần dần thêm độ phân giải, học cách dự đoán thang đo tiếp theo.

Chúng là những trục khác nhau thông qua vấn đề. Cả hai đều mang lại các phân phối có điều kiện có thể xử lý được. Về mặt kinh nghiệm, VAR nhanh hơn ở inference (ít đường chuyền hơn, tất cả đều song song trong một thang đo) và phù hợp hoặc đánh bại DiT trên ImageNet có điều kiện class. VAR có điều kiện văn bản (VARclip, HART) là một hướng nghiên cứu tích cực.

## Tự xây dựng

Trong `code/main.py` bạn sẽ:
1. Xây dựng một tokenizer VQ đa tỷ lệ nhỏ trên dữ liệu "hình ảnh" tổng hợp (vòng Gaussian 2D).
2. Huấn luyện một transformer kiểu VAR **để dự đoán tỷ lệ tiếp theo tokens.
3. Sample bằng cách gọi transformer 4 lần (4 thang đo) và giải mã.
4. Xác minh rằng training được sắp xếp theo tỷ lệ làm cho việc tạo song song trong một thang đo.

Đây là một triển khai đồ chơi. Vấn đề là xem mặt nạ attention có cấu trúc tỷ lệ và tạo song song trong quy mô thực sự hoạt động.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-var-tokenizer-designer.md` - một skill để thiết kế một tokenizer đa tỷ lệ: số lượng thang đo, tỷ lệ tỷ lệ, kích thước sách mã, chia sẻ còn lại decoder kiến trúc.

## Bài tập

1. **Cắt bỏ số lượng thang đo.** Huấn luyện VAR với 4, 6, 8, 10 thang đo. Đo lường chất lượng tái tạo so với số lần chuyền tự hồi quy. Nhiều thang đo hơn = dư mịn hơn = chất lượng tốt hơn nhưng nhiều đường chuyền hơn.

2. **Kích thước sách mã.** Tàu tokenizers với kích thước sách mã 512, 4096, 16384. Sách mã lớn hơn cho việc tái tạo tốt hơn nhưng dự đoán khó hơn. Tìm đầu gối.

3. **Kiểm tra song song trong thang đo.** Đối với VAR được huấn luyện, hãy đo mẫu attention một cách rõ ràng. Trong thang đo k, các model có tham gia vào các vị trí tỷ lệ chéo nhưng không tham gia vào thang đo nội bộ không? Xác minh việc triển khai mặt nạ.

4. **Tỷ lệ VAR so với DiT.** Đối với cùng một tác vụ có điều kiện class ImageNet, hãy huấn luyện VAR và DiT ở ngân sách tham số phù hợp (ví dụ: 33 triệu, 130 triệu, 458 triệu). Biểu đồ FID so với điện toán. VAR nên vượt qua DiT ở mỗi kích thước - tái tạo kết quả của bài báo ở quy mô nhỏ.

5. **Text conditioning.** Mở rộng VAR để lấy embedding văn bản (CLIP gộp) làm đầu vào điều kiện bổ sung qua adaLN. Đây là công thức HART. FID cải thiện bao nhiêu trên sampling căn chỉnh văn bản?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|----------------------|
| VAR | "Tự động hồi quy trực quan" | Tạo hình ảnh bằng cách dự đoán quy mô tiếp theo trên một kim tự tháp lưới VQ token |
| Dự đoán quy mô tiếp theo | "Dự đoán thô hơn, sau đó mịn hơn" | model dự đoán tokens ở thang độ phân giải tăng dần, điều hòa trên tất cả các thang đo trước đó |
| tokenizer VQ đa tỷ lệ | "VQ còn lại" | VQ-VAE tạo ra lưới K token có độ phân giải tăng dần, với decoder tổng hợp tất cả các thang đo |
| Thang đo k | "Kim tự tháp cấp k" | Một trong các mức độ phân giải K, từ 1x1 ở k = 1 đến (H/p) x (W/p) ở k = K |
| Song song trong quy mô | "Một chuyển tiếp cho mỗi thang đo" | Tất cả tokens ở tỷ lệ k được dự đoán trong một transformer vượt qua, không tự hồi quy |
| Thang đo nhân quả | "attention theo thứ tự theo tỷ lệ" | Token ở thang đo k có thể tham gia vào tất cả các thang đo 1..k nhưng không tham gia vào thang đo k + 1..K |
| VQ dư | "Phụ gia tokenization" | tokens của mỗi thang đo mã hóa phần còn lại do các thang đo thấp hơn để lại; decoder tổng tất cả quy mô embeddings |
| Luật tỷ lệ VAR | "Tỷ lệ GPT hình ảnh" | FID tuân theo một định luật lũy thừa có thể dự đoán được trong tính toán, giống như ngôn ngữ models' perplexity |
| HART | "VAR kết hợp + văn bản" | Biến thể VAR có điều kiện văn bản kết hợp giải mã lặp lại kiểu MaskGIT với cấu trúc thang đo của VAR |
| Vị trí tỷ lệ embedding | "(tỉ lệ, hàng, col) ba" | Mã hóa vị trí mang cả chỉ số tỷ lệ và tọa độ không gian trong thang đo |

## Đọc thêm

- [Tian et al., 2024 — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905) — bài báo VAR, tài liệu tham khảo chính thức
- [Peebles and Xie, 2022 — "Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748) — DiT, đường cơ sở so sánh khuếch tán
- [Esser et al., 2021 — "Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2012.09841) — VQGAN, tokenizer đa quy mô của VAR dòng tokenizer mở rộng
- [van den Oord et al., 2017 — "Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937) — VQ-VAE, nền tảng của tokenization hình ảnh rời rạc
- [Tang et al., 2024 — "HART: Efficient Visual Generation with Hybrid Autoregressive Transformer"](https://arxiv.org/abs/2410.10812) — VAR có điều kiện văn bản
