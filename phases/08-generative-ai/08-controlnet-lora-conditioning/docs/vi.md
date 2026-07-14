# ControlNet, LoRA và điều hòa

> Chỉ riêng văn bản là một tín hiệu điều khiển vụng về. ControlNet cho phép bạn sao chép model khuếch tán pretrained và điều khiển nó bằng bản đồ độ sâu, khung hình tạo dáng, vẽ nguệch ngoạc hoặc hình ảnh cạnh. LoRA cho phép bạn fine-tune parameter model 2 tỷ training 10 triệu parameters. Họ đã cùng nhau biến Stable Diffusion từ một món đồ chơi thành hình ảnh năm 2026 pipeline ships đó ở mọi cơ quan.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 8 · 07 (Khuếch tán tiềm ẩn), Giai đoạn 10 (LLMs từ đầu - cho LoRA nền)
**Thời lượng:** ~75 phút

## Vấn đề

Một prompt như "một người phụ nữ mặc váy đỏ dắt chó đi dạo trên một con phố đông đúc" không cung cấp cho model thông tin về đang ở đâu, người phụ nữ đang ở tư thế gì hoặc * góc nhìn * của đường phố. Văn bản ghim khoảng 10% những gì bạn cần để chỉ định hình ảnh. Sự rest là trực quan và không thể được mô tả hiệu quả bằng lời.

Training một model có điều kiện mới từ đầu cho mọi tín hiệu (tư thế, độ sâu, độ shải, phân đoạn) là điều cấm đoán. Bạn muốn giữ cho đường trục SDXL 2.6 tham số đóng băng, gắn một mạng bên nhỏ đọc điều hòa và để nó thúc đẩy features trung gian của đường trục Đó là ControlNet.

Bạn cũng muốn dạy model khái niệm mới (khuôn mặt, sản phẩm, phong cách của bạn) mà không cần huấn luyện lại toàn bộ model. Bạn muốn một delta nhỏ hơn 100 lần. Đó là LoRA - bộ điều hợp cấp thấp cắm vào trọng lượng attention hiện có.

ControlNet + LoRA + văn bản = bộ công cụ của học viên năm 2026. Hầu hết production hình ảnh pipelines lớp 2-5 LoRA, 1-3 ControlNets và Bộ chuyển đổi IP trên đế SDXL / SD3 / Flux.

## Khái niệm

![ControlNet clones the encoder; LoRA adds low-rank deltas](../assets/controlnet-lora.svg)

### ControlNet (Zhang và cộng sự, 2023)

Lấy một pretrained SD. *Clone* nửa encoder của U-Net. Đóng băng bản gốc. Huấn luyện bản sao để chấp nhận đầu vào điều hòa bổ sung (cạnh, độ sâu, tư thế). Kết nối bản sao trở lại nửa decoder của bản gốc bằng các kết nối bỏ qua * không-tích chập * (1×1 convs được khởi tạo về không - bắt đầu như một no-op, tìm hiểu một delta).

```
SD U-Net decoder:   ... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

Zero-conv init có nghĩa là ControlNet bắt đầu dưới dạng danh tính - không gây hại ngay cả trước khi training. Huấn luyện trên 1M (prompt, điều kiện, hình ảnh) tăng gấp ba lần với loss khuếch tán tiêu chuẩn.

ControlNets cho mỗi phương thức ship dưới dạng models bên nhỏ (~360M đối với SDXL, ~70M đối với SD 1.5). Bạn có thể soạn chúng tại inference:

```
features += weight_a * control_a(depth) + weight_b * control_b(pose)
```

### LoRA (Hu và cộng sự, 2021)

Đối với bất kỳ lớp tuyến tính nào `W ∈ R^{d×d}` trong model, hãy đóng băng `W` và thêm một delta cấp thấp:

```
W' = W + ΔW,  ΔW = B @ A,  A ∈ R^{r×d},  B ∈ R^{d×r}
```

với `r << d`. Xếp hạng 4-16 là tiêu chuẩn cho attention, xếp hạng 64-128 cho các giai điệu tinh tế nặng. Số lượng parameters mới: `2 · d · r` thay vì `d²`. Đối với attention SDXL có `d=640`, `r=16`: 20 nghìn tham số cho mỗi bộ chuyển đổi thay vì 410 nghìn — giảm 20 lần. Trên toàn bộ model: một LoRA thường là 20-200MB so với 5GB cơ bản.

Tại inference bạn có thể mở rộng quy mô LoRA: `W' = W + α · B @ A`. `α = 0.5-1.5` là bình thường. Nhiều LoRA stack cộng thêm (với cảnh báo thông thường là chúng tương tác theo cách phi tuyến tính).

### Bộ chuyển đổi IP (Ye và cộng sự, 2023)

Một bộ chuyển đổi nhỏ chấp nhận một *hình ảnh* làm điều kiện (cùng với văn bản). Sử dụng encoder hình ảnh CLIP để tạo tokens hình ảnh, đưa chúng vào cross-attention cùng với tokens văn bản. ~20MB cho mỗi model cơ sở. Cho phép bạn thực hiện "tạo hình ảnh theo kiểu tham chiếu này" mà không cần LoRA.

## Ma trận khả năng kết hợp

| Công cụ | Những gì nó kiểm soát | Kích thước | Trường hợp sử dụng |
|------|------------------|------|-------------|
| Mạng kiểm soát | Cấu trúc không gian (tư thế, độ sâu, cạnh) | 70-360 MB | Bố cục, bố cục chính xác |
| LoRA | Phong cách, chủ đề, khái niệm | 20-200MB | Cá nhân hóa, phong cách |
| Bộ chuyển đổi IP | Phong cách hoặc chủ đề từ hình ảnh tham chiếu | 20 MB | Không có văn bản nào có thể mô tả giao diện |
| Đảo ngược văn bản | Khái niệm đơn lẻ như một token mới | 10KB | Di sản, chủ yếu được thay thế bằng LoRA |
| Gian hàng giấc mơ | fine-tune đầy đủ về một chủ đề | 2-5GB | Danh tính mạnh mẽ, điện toán cao |
| Bộ chuyển đổi T2I | Lighter ControlNet thay thế | 70 MB | Thiết bị biên, inference ngân sách |

ControlNet ≈ không gian. LoRA ≈ ngữ nghĩa. Sử dụng cả hai.

## Tự xây dựng

`code/main.py` mô phỏng hai cơ chế trên 1-D:

1. **LoRA.** Một lớp tuyến tính pretrained `W`. Đóng băng nó. Huấn luyện một `B @ A` cấp thấp sao cho `W + BA` khớp với lớp tuyến tính mục tiêu. Cho thấy rằng `r = 1` là đủ để học một hiệu chỉnh hạng 1 một cách hoàn hảo.

2. **ControlNet-lite.** Một dự đoán "cơ sở đóng băng" và một "mạng bên" đọc một tín hiệu bổ sung. Đầu ra của mạng phụ được kiểm soát bởi một vô hướng có thể học được được khởi tạo thành không (phiên bản zero-conv của chúng tôi). Tập luyện và xem cổng tăng lên.

### Bước 1: LoRA toán

```python
def lora(W, A, B, x, alpha=1.0):
    # W is frozen; A, B are the trainable low-rank factors.
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### Bước 2: mạng bên zero-init

```python
side_out = control_net(x, condition)
gated = gate * side_out  # gate initialized to 0
h = base(x) + gated
```

Ở bước 0, đầu ra giống hệt với cơ sở. Các bản cập nhật training ban đầu `gate` chậm - không có sự trôi dạt thảm khốc.

## Cạm bẫy

- **LoRA mở rộng quy mô quá mức.** `α = 2` hoặc `α = 3` là một thủ thuật phổ biến "làm cho nó mạnh hơn" tạo ra đầu ra cách điệu / hỏng quá mức. Giữ `α ≤ 1.5`.
- **Xung đột trọng lượng ControlNet.** Sử dụng Pose ControlNet ở trọng lượng 1.0 và Depth ControlNet ở trọng lượng 1.0 thường vượt quá giới hạn. Tổng trọng số ≈ 1.0 là mặc định an toàn.
- **LoRA trên sai đế.** SDXL LoRA âm thầm không hoạt động trên SD 1.5 vì kích thước attention không khớp. Bộ khuếch tán sẽ cảnh báo trong 0.30+.
- **Trôi ngược văn bản.** Tokens được huấn luyện trên một checkpoint trôi dạt tồi tệ trên một  khác. LoRA di động hơn.
- **LoRA Hợp nhất và lưu trữ trọng lượng.** Bạn có thể nướng một LoRA vào đế model quả cân để inference nhanh hơn (không runtime bổ sung), nhưng bạn sẽ mất khả năng chia tỷ lệ `α` khi runtime. Giữ cả hai phiên bản.

## Ứng dụng

| Mục tiêu | pipeline năm 2026 |
|------|---------------|
| Tái tạo phong cách nghệ thuật của thương hiệu | LoRA được huấn luyện trên ~30 hình ảnh được tuyển chọn ở hạng 32 |
| Đặt khuôn mặt của tôi vào một hình ảnh được tạo ra | DreamBooth hoặc LoRA + IP-Adapter-FaceID |
| Tư thế cụ thể + prompt | ControlNet-Openpose + SDXL + văn bản |
| Bố cục nhận biết độ sâu | Điều khiểnĐộ sâu mạng + SD3 |
| Tài liệu tham khảo + prompt | Bộ chuyển đổi IP + văn bản |
| Bố cục chính xác | ControlNet-Scribble hoặc ControlNet-Canny |
| Thay thế nền | ControlNet-Seg + Inpainting (Bài 09) |
| Kiểu 1 bước nhanh | LCM-LoRA trên SDXL-Turbo |

## Sản phẩm bàn giao

Tiết kiệm `outputs/skill-sd-toolkit-composer.md`. Skill nhận một tác vụ (nội dung đầu vào: prompt, hình ảnh tham chiếu tùy chọn, tư thế tùy chọn, độ sâu tùy chọn, viết nguệch ngoạc tùy chọn) và xuất stack công cụ, trọng lượng và giao thức hạt giống có thể tái tạo.

## Bài tập

1. **Dễ dàng.** Trong `code/main.py`, thay đổi xếp hạng LoRA `r` từ 1 đến 4. LoRA khớp chính xác với đồng bằng mục tiêu hạng 2 ở cấp độ nào?
2. **Trung bình.** Huấn luyện hai LoRA riêng biệt trên hai phép biến đổi mục tiêu. Tải chúng lại với nhau và thể hiện sự tương tác bổ sung của chúng. Khi nào tương tác phá vỡ tính tuyến tính?
3. **Cứng.** Sử dụng bộ khuếch tán để stack: SDXL-base + Canny-ControlNet (trọng lượng 0.8) + kiểu LoRA (α 0.8) + Bộ chuyển đổi IP (trọng lượng 0.6). Đo lường sự cân bằng tuân thủ FID và prompt vì trọng số stack khác nhau.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Mạng kiểm soát | "Kiểm soát không gian" | encoder nhân bản + bỏ qua zero-conv; đọc một hình ảnh điều hòa. |
| Không tích chập | "Bắt đầu như bản sắc" | 1×1 conv được khởi tạo về không; ControlNet bắt đầu như không hoạt động. |
| LoRA | "Bộ chuyển đổi cấp thấp" | `W + B @ A`, `r << d`; Ít hơn 100 lần so với một fine-tune đầy đủ. |
| Hạng R | "Núm vặn" | nén LoRA; 4-16 điển hình, 64+ để cá nhân hóa nặng. |
| α | "Sức mạnh LoRA" | Runtime quy mô của đồng bằng LoRA. |
| Bộ chuyển đổi IP | "Hình ảnh tham khảo" | Bộ chuyển đổi điều hòa hình ảnh nhỏ thông qua tokens hình ảnh CLIP. |
| Gian hàng giấc mơ | "fine-tune chủ đề đầy đủ" | Huấn luyện toàn bộ model trên ~30 hình ảnh của một đối tượng. |
| Đảo ngược văn bản | "token mới" | Chỉ embedding học một từ mới; di sản, hầu hết được thay thế. |

## Lưu ý Production: hoán đổi LoRA, làn ControlNet, giao bóng nhiều tenant

Một SaaS chuyển văn bản thành hình ảnh thực sự phục vụ hàng trăm LoRA và hàng chục ControlNet trên cùng một checkpoint cơ sở. Vấn đề phục vụ trông rất giống LLM nhiều tenant (tài liệu production bao gồm trường hợp LLM theo lô liên tục và LoRAX / S-LoRA):

- **LoRA hoán đổi nóng, không merge.** Hợp nhất `W' = W + α·B·A` vào cơ sở cho inference mỗi bước nhanh hơn ~3-5% nhưng đóng băng `α` và đế. Giữ cho LoRA nóng trong VRAM như delta rank-r; Bộ khuếch tán hiển thị `pipe.load_lora_weights()` + `pipe.set_adapters([...], adapter_weights=[...])` để kích hoạt theo yêu cầu. Chi phí hoán đổi là trọng số `2 · d · r · num_layers` - tỷ lệ MB, dưới giây.
- **ControlNet là làn attention thứ hai.** encoder nhân bản chạy song song với cơ sở. Hai ControlNets với trọng lượng 1.0 mỗi lần = hai đường chuyền về phía trước bổ sung cho mỗi bước, không phải một đường chuyền merged. Khoảng trống kích thước Batch giảm bậc hai. Ngân sách cho ~1,5× chi phí bước cho mỗi ControlNet đang hoạt động.
- **LoRA lượng tử hóa nữa.** Nếu bạn lượng tử hóa cơ sở (xem Bài 07, Thông lượng trên 8GB), delta LoRA cũng lượng tử hóa rõ ràng thành 8-bit hoặc 4-bit. Tải kiểu QLoRA cho phép bạn stack 5-10 LoRA trên đế Flux 4 bit mà không làm nổ bộ nhớ.

Cụ thể về thông lượng: Máy tính xách tay Flux-on-8GB của Niels lượng tử cơ sở thành 4-bit; Xếp chồng một LoRA kiểu (`pipe.load_lora_weights("user/style-lora")`) trên cơ sở lượng tử hóa đó ở `weight_name="pytorch_lora_weights.safetensors"` vẫn hoạt động. Đây là công thức mà hầu hết các cơ quan SaaS ship vào năm 2026.

## Đọc thêm

- [Zhang, Rao, Agrawala (2023). Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) - ControlNet.
- [Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — LoRA (ban đầu dành cho LLMs; cổng để khuếch tán).
- [Ye et al. (2023). IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721) - Bộ chuyển đổi IP.
- [Mou et al. (2023). T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability](https://arxiv.org/abs/2302.08453) — giải pháp thay thế nhẹ hơn cho ControlNet.
- [Ruiz et al. (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) - DreamBooth.
- [HuggingFace Diffusers — ControlNet / LoRA / IP-Adapter docs](https://huggingface.co/docs/diffusers/training/controlnet) — tài liệu tham khảo pipelines.
