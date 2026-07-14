# Bản vá Encoder tầm nhìn

> Một model thị giác đọc pixel cần có tokenizer cho pixel. Bản vá embedding là tokenizer đó. Cắt hình ảnh thành một lưới hình vuông, làm phẳng từng hình vuông, chiếu nó qua một layer tuyến tính, sau đó thêm tín hiệu vị trí 2D để transformer biết vị trí của mỗi hình vuông trong hình ảnh gốc.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 19 bài 30-37 (Nền tảng theo dõi B)
**Thời lượng:** ~90 phút

## Mục tiêu học tập

- Mã hóa hình ảnh thành một chuỗi embeddings bản vá có độ dài cố định.
- Triển khai phép chiếu bản vá dựa trên `Conv2d` phù hợp với phép toán của mở ra sau đó tuyến tính.
- Xây dựng vị trí hình sin 2D xác định embedding token thứ tự mã hóa vị trí không gian.
- Xác minh số lượng bản vá, hình dạng embedding và `Conv2d` / mở tương đương trên thiết bị cố định tổng hợp.

## Vấn đề

Một transformer ăn một chuỗi vectors. Hình ảnh là lưới 3 kênh. Đọc mọi pixel dưới dạng token sẽ làm bùng nổ độ dài chuỗi: hình ảnh RGB 224x224 là 150.528 tokens, điều mà transformer 12 lớp không thể mua được trong attention. Đọc hình ảnh như một vector phẳng khổng lồ vứt bỏ địa phương, mà lớp attention không thể phục hồi. Công việc của giao diện người dùng encoder là nén lưới pixel thành vài trăm tokens mà mỗi phần tóm tắt một vùng vuông.

Bản vá embedding giải quyết vấn đề này bằng một phép chiếu tuyến tính. Hình ảnh 224x224 được cắt thành các bản vá 16x16 tạo ra lưới 14x14 gồm 196 bản vá. Mỗi bản vá được làm phẳng từ `(3, 16, 16) = 768` giá trị pixel thành một vector, sau đó một lớp tuyến tính ánh xạ nó với kích thước ẩn của model. transformer thấy 196 tokens kích thước `hidden` (thường là 768) cộng với một token CLS. Đó là một trình tự mà rest của mạng có thể nhai.

## Khái niệm

```mermaid
flowchart LR
  Image[224x224x3 image] --> Cut[cut into 16x16 patches]
  Cut --> Grid[14x14 grid of patches]
  Grid --> Flatten[flatten each patch]
  Flatten --> Proj[linear projection]
  Proj --> Tokens[196 tokens of dim hidden]
  Tokens --> Pos[add 2D sinusoidal position]
  Pos --> Out[final token sequence]
```

### Tại sao lại là bản vá chứ không phải pixel

Attention là bậc hai về độ dài dãy. Một chuỗi 196 token có giá `196 * 196 = 38,416` attention điểm cho mỗi đầu mỗi lớp; Một chuỗi 150.528 token có giá `150,528 * 150,528 = 22.6 billion`. Các bản vá giúp giảm 590.000 lần điện toán attention và một vùng 16x16 duy nhất mang đủ tín hiệu cho các tác vụ tầm nhìn cấp cao. Chi phí là một loss chi tiết không gian chi tiết bên trong một bản vá, đó là lý do tại sao các stacks đa phương thức xuôi dòng thường chạy branch độ phân giải cao thứ hai khi bản địa hóa tốt quan trọng.

### Tại sao phép chiếu tuyến tính là đủ

Mỗi bản vá được coi là một vector độc lập. Phép chiếu tìm hiểu một cơ sở: máy dò cạnh, bộ lọc màu, kết cấu đơn giản. Một lớp tuyến tính đơn nhỏ (`768 * 768 = 589,824` parameters đối với ViT-Base) và huấn luyện nhanh. Các thân tích chập sâu hơn tồn tại (ViT "lai"), nhưng phép chiếu tuyến tính phẳng là tiêu chuẩn và hầu hết các encoders ship trọng lượng mở hiện đại nhất có hình dạng chính xác này.

### Thủ thuật `Conv2d`

Một `Conv2d(in_channels=3, out_channels=hidden, kernel_size=patch_size, stride=patch_size)` không có khoảng đệm cho kết quả số giống như mở ra sau đó tuyến tính, bởi vì mỗi vị trí đầu ra chấm các pixel vá dựa trên một bộ lọc. Sự tích chập là phép chiếu bản vá và hầu hết các cơ sở mã production ship nó theo cách đó vì nó nhanh hơn trên GPU và sử dụng ít định hình lại hơn.

### Vị trí embeddings

Tokens không mang theo thứ tự nào ra khỏi hình chiếu. embedding hình sin 2D cung cấp cho mỗi token một tín hiệu cố định mã hóa vị trí `(row, col)` của nó. Một nửa kích thước embedding mã hóa vị trí hàng với sin/cos ở nhiều tần số; nửa còn lại mã hóa vị trí cột. Mã hóa mang tính xác định để bạn có thể hoán đổi độ phân giải mà không cần huấn luyện lại và nội suy rõ ràng vào các lưới mà model chưa từng thấy tại training thời điểm.

| Thành phần | Hình dạng | Parameters |
|-----------|-------|------------|
| Chiếu bản vá (`Conv2d`) | `(hidden, 3, patch, patch)` | `3 * P * P * hidden + hidden` |
| Vị trí embedding (cố định) | `(num_patches, hidden)` | 0 (tính toán, không học) |
| CLS token (đã học) | `(1, hidden)` | `hidden` |

Đối với ViT-Base/16 ở độ phân giải 224: 590.592 parameters trong phép chiếu, 768 trong token CLS và không cho vị trí hình sin. Bài tiếp theo (59) stacks một transformer 12 lớp trên đầu giao diện người dùng này.

### Tương đương như một kiểm tra sự tỉnh táo

Bước vá lỗi có hai cách chính tả: phép chiếu `Conv2d` và mở ra rõ ràng sau đó tuyến tính. Chúng phải tạo ra cùng một đầu ra cho cùng một trọng lượng. Nếu không, phép toán mở ra là sai, và rest của encoder được xây dựng trên cát. Các bài kiểm tra trong bài học này thực hiện sự tương đương đó.

## Tự xây dựng

`code/main.py` thực hiện:

- `PatchEmbed`, một `Conv2d` bọc `nn.Module` để chiếu bản vá.
- `sinusoidal_2d(grid_h, grid_w, dim)`, một hàm không trạng thái xây dựng bảng vị trí 2D.
- `VisionFrontEnd`, bao gồm embedding bản vá, thêm trước CLS và thêm vị trí vào một forward pass.
- Một trình trợ giúp `synthesize_image(seed)` xây dựng một thiết bị cố định 224x224x3 xác định từ `numpy.random`.
- Một bản demo chạy một hình ảnh cố định qua giao diện người dùng và in hình dạng đầu ra, tiêu chuẩn token CLS và một hàng của embedding vị trí.

Chạy nó:

```bash
python3 code/main.py
```

Đầu ra: thiết bị cố định 224x224 được mã hóa thành một chuỗi hình dạng `(1, 197, 768)`. token đầu tiên là CLS; 196 tiếp theo là bản vá tokens. Vị trí embedding các định mức đồng nhất trong một hàng, đó là chữ ký hình sin.

## Ứng dụng

Giao diện người dùng bản vá tương tự hiển thị trong mọi model ngôn ngữ thị giác hiện đại: CLIP ViT-L/14, SigLIP, DINOv2, họ Qwen-VL và stack InternVL đều bắt đầu từ phép chiếu bản vá `Conv2d` cộng với tín hiệu vị trí. Sự khác biệt giữa các gia đình sống ở hạ nguồn (CLS so với không có CLS, tokens thanh ghi, kích thước bản vá khác nhau 14 so với 16, độ phân giải động thông qua các vị trí nội suy). Giao diện người dùng trong bài học này là chất nền mà mọi models đứng trên đó.

## Kiểm tra

`code/test_main.py` bao gồm:

- Số bản vá khớp với `(image_size / patch_size) **2`
- hình dạng đầu ra khớp với `(batch, num_patches + 1, hidden)`
- Phép chiếu `Conv2d` tương đương với việc mở ra thủ công sau đó tuyến tính trên một thiết bị cố định nhỏ
- Bảng vị trí hình sin là xác định giữa các cuộc gọi
- CLS token phát sóng qua batch mờ mà không bị rò rỉ

Chạy chúng:

```bash
python3 -m unittest code/test_main.py
```

## Bài tập

1. Thay thế vị trí hình sin bằng một `nn.Parameter` đã học và so sánh epoch loss đầu tiên trong một nhiệm vụ phân loại tổng hợp nhỏ. Các vị trí đã học giành chiến thắng ở độ phân giải cố định; hình sin chiến thắng khi bạn thay đổi độ phân giải sau training.

2. Hoán đổi `Conv2d` cho một `nn.Unfold` rõ ràng cộng với `nn.Linear` và xác nhận các kết quả đầu ra phù hợp với dung sai float. Cùng một toán học, hai cách để đánh vần nó.

3. Thêm hỗ trợ cho kích thước bản vá không vuông (ví dụ: 32x16 cho đầu vào có diện rộng) và xác minh bảng vị trí xử lý lưới không vuông.

4. Cấu hình bước vá ở batch kích thước 1, 8, 64. Phép chiếu bản vá hiếm khi là nút thắt cổ chai; các lớp attention ở hạ lưu chiếm ưu thế.

5. Huấn luyện phần đầu trước như một máy vắt feature đông lạnh trên dataset hình dạng tổng hợp 4 class (hình tròn, hình vuông, hình tam giác, ngôi sao). Đầu ra token CLS phải tách biệt tuyến tính.

## Thuật ngữ chính

| Thuật ngữ | Nó có nghĩa là gì |
|------|---------------|
| Bản vá | Vùng phụ hình vuông của hình ảnh, thường là 14x14 hoặc 16x16 |
| Bản vá embedding | Phép chiếu tuyến tính của một mảng phẳng đến độ mờ ẩn |
| Độ dài trình tự | Số tokens sau khi tokenization bản vá, thường cộng với CLS |
| Vị trí hình sin | Tín hiệu sin/cos cố định mã hóa tọa độ lưới 2D |
| CLS token | Đã học vector thêm vào đầu trình tự làm đầu gộp |

## Đọc thêm

- Một hình ảnh có giá trị 16x16 từ (ViT, 2021) cho khung nhúng bản vá gốc.
- Attention là tất cả những gì bạn cần (2017) cho công thức vị trí hình sin được điều chỉnh ở đây thành 2D.
- Giấy DINOv2 để đăng ký tokens, một phần mở rộng bạn có thể thêm dưới dạng bài tập 6.
