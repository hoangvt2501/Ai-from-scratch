# Inpainting, Outpainting & Chỉnh sửa hình ảnh

> Chuyển văn bản thành hình ảnh tạo ra những điều mới. Inpainting sửa chữa những cái cũ. Trong production, 70% công việc hình ảnh có thể lập hóa đơn là chỉnh sửa - hoán đổi nền, xóa logo, mở rộng canvas, tạo lại bàn tay. Inpainting là nơi khuếch tán kiếm được sự giữ lại của nó.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 8 · 07 (Khuếch tán tiềm ẩn), Giai đoạn 8 · 08 (ControlNet & LoRA)
**Thời lượng:** ~75 phút

## Vấn đề

Một khách hàng gửi một bức ảnh sản phẩm hoàn hảo với một dấu hiệu gây mất tập trung ở nền. Bạn muốn xóa dấu hiệu và để mọi thứ khác giống hệt nhau. Bạn không thể chạy văn bản thành hình ảnh từ đầu - kết quả sẽ có màu sắc khác, ánh sáng khác, góc sản phẩm khác nhau. Bạn muốn tái tạo *chỉ* vùng được che giấu và bạn muốn tái tạo tôn trọng bối cảnh xung quanh.

Đó là inpainting. Các biến thể:

- **Inpainting.** Tái tạo bên trong mặt nạ, giữ các pixel bên ngoài.
- **Outpainting.** Tái tạo bên ngoài mặt nạ (hoặc bên ngoài canvas), giữ bên trong.
- **Chỉnh sửa hình ảnh.** Tái tạo toàn bộ hình ảnh nhưng giữ nguyên ngữ nghĩa hoặc cấu trúc trung thực với bản gốc (SDEdit, InstructPix2Pix).

Mỗi lần khuếch tán pipeline vào năm 2026 ships chế độ inpainting. Flux.1-Fill, Sơn khuếch tán ổn định, SDXL-Inpaint, DALL-E 3 Chỉnh sửa. Chúng hoạt động trên cùng một nguyên tắc.

## Khái niệm

![Inpainting: mask-aware denoising with context-preserving reinjection](../assets/inpainting.svg)

### Cách tiếp cận ngây thơ (và tại sao nó sai)

Chạy chuyển văn bản thành hình ảnh tiêu chuẩn với mặt nạ. Ở mỗi bước sampling, thay thế vùng không được che giấu của tiềm ẩn nhiễu bằng hình ảnh sạch khuếch tán về phía trước. Nó hoạt động... tồi tệ. Ranh giới artifacts chảy máu vì model không có thông tin về những gì có trong khu vực đeo mặt nạ.

### model sơn thích hợp

Huấn luyện U-Net đã sửa đổi có 9 kênh đầu vào thay vì 4:

```
input = concat([ noisy_latent (4ch), encoded_image (4ch), mask (1ch) ], dim=channel)
```

Các kênh bổ sung là bản sao của hình ảnh nguồn được mã hóa VAE cộng với mặt nạ một kênh. Tại thời điểm training, bạn che ngẫu nhiên các vùng của hình ảnh và huấn luyện model chỉ khử nhiễu vùng được che trong khi vùng không được che được đưa ra dưới dạng tín hiệu điều hòa sạch. Ở inference, model có thể "nhìn thấy" những gì xung quanh khu vực đeo mặt nạ và tạo ra các kết quả hoàn thiện mạch lạc.

SD-Inpaint, SDXL-Inpaint, Flux-Fill đều sử dụng đầu vào 9 kênh (hoặc tương tự) này. Bộ khuếch tán `StableDiffusionInpaintPipeline`, `FluxFillPipeline`.

### SDEdit (Meng và cộng sự, 2022) — chỉnh sửa miễn phí

Thêm nhiễu vào hình ảnh nguồn lên đến một số `t` trung gian, sau đó chạy chuỗi ngược từ `t` xuống 0 với một prompt mới. Không huấn luyện lại. Sự lựa chọn bắt đầu `t` đánh đổi sự trung thành để lấy tự do sáng tạo:

- `t/T = 0.3` → gần giống với nguồn gốc, những thay đổi nhỏ về phong cách
- `t/T = 0.6` → chỉnh sửa vừa phải, bảo toàn cấu trúc thô
- `t/T = 0.9` → được tạo ra từ việc bảo quản nguồn gần nhiễu, tối thiểu

### InstructPix2Pix (Brooks và cộng sự, 2023)

Fine-tune một model khuếch tán trên `(input_image, instruction, output_image)` ba. Tại inference, điều kiện trên cả hình ảnh đầu vào và hướng dẫn văn bản ("làm cho nó hoàng hôn", "thêm một con rồng"). Hai thang đo CFG: tỷ lệ hình ảnh và tỷ lệ văn bản.

### Sơn lại (Lugmayr và cộng sự, 2022)

Giữ một model khuếch tán vô điều kiện tiêu chuẩn. Ở mỗi bước đảo ngược, lấy mẫu lại - thỉnh thoảng quay trở lại trạng thái ồn ào hơn và tái tạo. Tránh ranh giới artifacts. Được sử dụng khi bạn không có model inpainting được huấn luyện.

## Tự xây dựng

`code/main.py` thực hiện sơ đồ vẽ 1-D đồ chơi trên dữ liệu 5 chiều. Chúng ta huấn luyện DDPM trên dữ liệu hỗn hợp 5-D trong đó mỗi mẫu là 5 phao từ một trong hai cụm. Ở inference, chúng ta "che giấu" 2 trong số 5 chiều, đưa vào phiên bản nhiễu về phía trước của ba chiều không được che giấu ở mỗi bước và chỉ tạo lại các chiều được che nạ.

### Bước 1: Dữ liệu DDPM 5-D

```python
def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster
```

### Bước 2: huấn luyện khử nhiễu trên tất cả 5 độ mờ

DDPM tiêu chuẩn. Đầu ra Net dự đoán nhiễu 5-D cho đầu vào nhiễu 5-D.

### Bước 3: tại inference, đảo ngược nhận biết mặt nạ

```python
def inpaint_step(x_t, mask, clean_image, alpha_bars, t, rng):
    # replace unmasked dims with a freshly noised version of the clean source
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_image[i] + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ...then run the normal reverse step on x_t
```

Đây là cách tiếp cận ngây thơ và nó hoạt động trên dữ liệu 1-D đồ chơi. Vẽ hình ảnh thực sử dụng đầu vào 9 kênh vì tính mạch lạc kết cấu quan trọng hơn.

### Bước 4: sơn ngoài

Outpainting là inpainting với mặt nạ đảo ngược: che canvas mới (trước đây không tồn tại), lấp đầy rest bằng bản gốc. Mục tiêu training giống hệt nhau.

## Cạm bẫy

- **Đường nối.** Cách tiếp cận ngây thơ để lại ranh giới có thể nhìn thấy vì thông tin gradient không chảy qua mặt nạ. Khắc phục: làm giãn mặt nạ thêm 8-16 pixel hoặc sử dụng model sơn thích hợp.
- **Rò rỉ mặt nạ.** Nếu vùng không được che mặt nạ của hình ảnh điều hòa có chất lượng thấp hoặc nhiễu, nó sẽ gây ô nhiễm thế hệ bên trong mặt nạ. Giảm nhiễu hoặc làm mờ một chút.
- **CFG tương tác với kích thước mặt nạ.** CFG cao trên mặt nạ nhỏ = miếng dán bão hòa. Giảm CFG cho các chỉnh sửa nhỏ.
- **SDEsửa đổi vách đá trung thực.** Đi từ `t/T = 0.5` này sang `t/T = 0.6` có thể đánh mất danh tính của đối tượng. Quét và checkpoint.
- **Prompt không khớp.** prompt nên mô tả hình ảnh *toàn bộ*, không chỉ nội dung mới. "Một con mèo ngồi trên ghế" không phải "một con mèo".

## Ứng dụng

| Nhiệm vụ | Pipeline |
|------|----------|
| Loại bỏ đồ vật, mặt nạ nhỏ | SD-Inpaint hoặc Flux-Fill, tiêu chuẩn prompt |
| Thay thế bầu trời | SD-Inpaint + "bầu trời xanh lúc hoàng hôn" |
| Mở rộng canvas | Chế độ sơn ngoài SDXL (lông vũ 8px) hoặc Flux-Fill với mặt nạ sơn ngoài |
| Tái tạo bàn tay/khuôn mặt | SD-Inpaint với prompt mô tả lại đối tượng + ControlNet-Openpose |
| Thay đổi kiểu của một vùng | SDSửa đổi tại `t/T=0.5` trên vùng mặt nạ |
| "Làm cho nó hoàng hôn" | InstructPix2Pix hoặc Flux-Kontext |
| Thay thế nền | Mặt nạ SAM → SD-Inpaint |
| Độ trung thực cực cao | Flux-Fill hoặc GPT-Image (lưu trữ) cho các trường hợp khó nhất |

SAM (Meta's Segment Anything, 2023) + diffusion inpaint là pipeline xóa nền năm 2026. SAM 2 (2024) hoạt động trên video.

## Sản phẩm bàn giao

Tiết kiệm `outputs/skill-editing-pipeline.md`. Skill lấy hình ảnh gốc + chỉnh sửa mô tả + mặt nạ tùy chọn (hoặc SAM prompt) và đầu ra: phương pháp tạo mặt nạ, model cơ sở, tỷ lệ CFG (hình ảnh + văn bản), SDEdit-t hoặc chế độ sơn và danh sách kiểm tra QA.

## Bài tập

1. **Dễ dàng.** Trong `code/main.py`, hãy thay đổi phần kích thước được che từ 0,2 đến 0,8. Chất lượng sơn (phần còn lại trong mờ mặt nạ) tương đương với thế hệ vô điều kiện ở phần nào?
2. **Trung bình.** Thực hiện RePaint: ở mỗi bước đảo ngược thứ 10, hãy lùi lại 5 bước (thêm nhiễu) và khử nhiễu lại. Đo lường xem nó có làm giảm ranh giới còn lại ở mép mặt nạ hay không.
3. **Khó.** Sử dụng bộ khuếch tán Hugging Face để so sánh: SD 1.5 Inpaint + ControlNet-Openpose vs Flux.1-Fill trên 20 tác vụ tái tạo khuôn mặt. Tuân thủ tư thế điểm số và bảo tồn danh tính riêng biệt.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Sơn | "Lấp đầy lỗ hổng" | Tái tạo bên trong mặt nạ; giữ các pixel bên ngoài. |
| Sơn ngoài | "Mở rộng canvas" | Tái tạo bên ngoài canvas; giữ bên trong. |
| U-Net 9 kênh | "Sơn đúng cách model" | U-Net với 'ồn ào \ | mã nguồn \ | mặt nạ' làm đầu vào. |
| SDSửa đổi | "Img2img với độ ồn" | Nhiễu theo thời gian `t`, khử nhiễu với prompt mới. |
| Hướng dẫnPix2Pix | "Chỉnh sửa chỉ có văn bản" | Fine-tuned khuếch tán trên (hình ảnh, hướng dẫn, đầu ra) gấp ba lần. |
| Sơn lại | "Không huấn luyện lại" | Nhiễu lại định kỳ trong quá trình đảo ngược để giảm đường nối. |
| SAM | "Phân đoạn bất cứ điều gì" | Trình tạo mặt nạ bằng cách nhấp chuột hoặc hộp; cặp với inpaint. |
| Thông lượng-Kontext | "Chỉnh sửa theo ngữ cảnh" | Biến thể thông lượng chấp nhận hình ảnh tham chiếu + hướng dẫn chỉnh sửa. |

## Lưu ý Production: pipelines chỉnh sửa nhạy cảm với độ trễ

Người dùng chỉnh sửa hình ảnh mong đợi các chuyến đi khứ hồi dưới 5 giây. SDXL-Inpaint 30 bước ở 1024² là 3-4 giây trên L4, cộng với việc tạo mặt nạ SAM (~200 ms) và VAE encode/decode (kết hợp ~500 ms). Trong khung production, đây là giới hạn TTFT chứ không phải giới hạn thông lượng - batch 1, tính đồng thời thấp, giảm thiểu mọi giai đoạn:

- **SAM-H là cái chậm.** SAM-H ở 1024² là ~200 ms; SAM-ViT-B là ~40 ms với loss chất lượng nhỏ. SAM 2 (video) thêm chi phí tạm thời; Không sử dụng nó để chỉnh sửa một hình ảnh.
- **Bỏ qua mã hóa khi có thể.** `pipe.image_processor.preprocess(img)` mã hóa thành tiềm ẩn. Nếu bạn có tiềm ẩn từ thế hệ trước (điển hình trong giao diện người dùng chỉnh sửa lặp lại), hãy chuyển chúng trực tiếp qua `latents=...` để bỏ qua một mã hóa VAE.
- **Sự giãn nở mặt nạ cũng quan trọng đối với thông lượng.** Một mặt nạ nhỏ có nghĩa là hầu hết forward pass U-Net bị lãng phí (dù sao các pixel không được che mặt nạ cũng bị kẹp). `diffusers`' `StableDiffusionInpaintPipeline` chạy U-Net đầy đủ bất kể; Chỉ có các biến thể inpaint thích hợp 9 kênh mới khai thác điện toán được che giấu.
- **Flux-Kontext là câu trả lời năm 2025.** Đơn forward pass trên `(source_image, instruction)` - không có mặt nạ riêng biệt, không quét nhiễu SDEsửa đổi. Trên H100, nó ships chỉnh sửa trong ~1,5 giây. Bài học kiến trúc: thu gọn các giai đoạn.

## Đọc thêm

- [Lugmayr et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865) - sơn không training.
- [Meng et al. (2022). SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073) — SDSửa đổi.
- [Brooks, Holynski, Efros (2023). InstructPix2Pix](https://arxiv.org/abs/2211.09800) - chỉnh sửa hướng dẫn văn bản.
- [Kirillov et al. (2023). Segment Anything](https://arxiv.org/abs/2304.02643) - SAM, nguồn mặt nạ.
- [Ravi et al. (2024). SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) — video SAM.
- [Hertz et al. (2022). Prompt-to-Prompt Image Editing with Cross-Attention Control](https://arxiv.org/abs/2208.01626) — chỉnh sửa cấp attention.
- [Black Forest Labs (2024). Flux.1-Fill and Flux.1-Kontext](https://blackforestlabs.ai/flux-1-tools/) - 2024 dụng cụ.
