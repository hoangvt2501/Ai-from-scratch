# Music Generation - MusicGen, Stable Audio, Suno và The Licensing Earthquake

> Thế hệ âm nhạc năm 2026: Suno v5 và Udio v4 thống trị thương mại; MusicGen, Mở âm thanh ổn định và mã nguồn mở dẫn đầu ACE-Step. Vấn đề kỹ thuật hầu hết đã được giải quyết. Vấn đề pháp lý (dàn xếp 500 triệu đô la của Warner Music, dàn xếp UMG) đã định hình lại lĩnh vực này vào năm 2025-2026.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 6 · 02 (Quang phổ), Giai đoạn 4 · 10 (Khuếch tán Models)
**Thời lượng:** ~75 phút

## Vấn đề

Văn bản → một clip nhạc dài từ 30 giây đến 4 phút, với lời bài hát, giọng hát và cấu trúc. Ba vấn đề phụ:

1. **Tạo nhạc cụ.** Văn bản như "trống hip-hop lo-fi với phím ấm" → âm thanh. MusicGen, Âm thanh ổn định, AudioLDM.
2. **Tạo bài hát (có giọng hát + lời bài hát).** "Bài hát đồng quê về những đêm mưa ở Texas" → bài hát đầy đủ. Suno, Udio, YuE, ACE-Bước.
3. **Có điều kiện / có thể kiểm soát.** Mở rộng clip hiện có, tạo lại cầu nối, hoán đổi thể loại, tách gốc hoặc inpaint. Lớp sơn + tách thân của Udio là feature năm 2026 phù hợp.

## Khái niệm

![Music generation: token-LM vs diffusion, the 2026 model map](../assets/music-generation.svg)

### Token LM qua tokens codec thần kinh

**MusicGen** của Meta (2023, MIT) và nhiều dẫn xuất: điều kiện trên text/melody embeddings, tự hồi quy dự đoán EnCodec tokens (32 kHz, 4 sách mã), giải mã bằng EnCodec. 300 triệu - 3,3 tỷ tham số. Đường cơ sở vững chắc; vật lộn qua 30 giây.

**ACE-Step** (mã nguồn mở, 4B XL phát hành tháng 4 năm 2026) mở rộng điều này cho thế hệ có điều kiện lời bài hát đầy đủ. Điều gần gũi nhất của cộng đồng mở với Suno.

### Khuếch tán qua mels hoặc tiềm ẩn

**Âm thanh ổn định (2023)** và **Mở âm thanh ổn định (2024)**: khuếch tán tiềm ẩn trên âm thanh nén. Vượt trội về các vòng lặp, thiết kế âm thanh, kết cấu môi trường xung quanh. Không giỏi trong các bài hát đầy đủ có cấu trúc.

**AudioLDM / AudioLDM2**: chuyển văn bản thành âm thanh thông qua khuếch tán tiềm ẩn kiểu T2I, khái quát hóa thành âm nhạc, hiệu ứng âm thanh, giọng nói.

### Lai (production) — Suno, Udio, Lyria

Trọng lượng kín. Có khả năng là bộ mã hóa AR LM + bộ mã hóa dựa trên khuếch tán với các đầu giọng nói / trống / giai điệu chuyên dụng. Suno v5 (2026) là sản phẩm dẫn đầu về chất lượng ELO 1293. Udio v4 bổ sung inpainting + tách thân (bass, trống, giọng hát tải xuống riêng biệt).

### Đánh giá

- **FAD (Khoảng cách âm thanh Fréchet).** Khoảng cách cấp Embedding giữa phân phối âm thanh được tạo và âm thanh thực bằng cách sử dụng VGGish hoặc PANN features. Thấp hơn là tốt hơn. MusicGen nhỏ: 4.5 FAD trên MusicCaps; SOTA ~3.0.
- **Âm nhạc (chủ quan).** Sở thích của con người. Suno v5 ELO 1293 dẫn đầu.
- **alignment văn bản-âm thanh.** Điểm CLAP giữa prompt và đầu ra.
- **Âm nhạc artifacts.** Chuyển tiếp lệch nhịp, trôi dạt giọng hát, loss cấu trúc sau 30 giây.

## Bản đồ model 2026

| Model | Tham số | Chiều dài | Giọng hát | Giấy phép |
|-------|--------|--------|--------|---------|
| MusicGen-lớn | 3,3 tỷ | 30 giây | Không | Tiểu bang MIT |
| Mở âm thanh ổn định | 1,2 tỷ | 47 giây | Không | Tính ổn định phi thương mại |
| ACE-Step XL (Tháng Tư 2026) | 4 tỷ | &gt; 2 phút | Có | Apache-2.0 |
| YuE | 7 tỷ | &gt; 2 phút | Vâng, đa ngôn ngữ | Apache-2.0 |
| Suno v5 (đã đóng cửa) | ? | 4 phút | Vâng, ELO 1293 | thương mại |
| Udio v4 (đã đóng) | ? | 4 phút | Có + thân cây | thương mại |
| Google Lyria 3 (đã đóng) | ? | Thời gian thực | Có | thương mại |
| Nhạc MiniMax 2.5 | ? | 4 phút | Có | API thương mại |

## Bối cảnh pháp lý (2025-2026)

- **Dàn xếp Warner Music vs Suno.** 500 triệu đô la. WMG hiện có quyền giám sát AI giống nhau, bản quyền âm nhạc và các bản nhạc do người dùng tạo trên Suno. Thỏa thuận UMG tương tự trên Udio.
- **Đạo luật AI của Liên minh Châu Âu** + **California SB 942**: Nhạc do AI tạo phải được tiết lộ.
- **Riffusion / MusicGen **thuộc MIT không có hành lý tuân thủ nhưng cũng không có giọng hát thương mại.

Các mẫu an toàn để ship:

1. Chỉ tạo nhạc cụ (MusicGen, Mở âm thanh ổn định MIT/CC0 đầu ra).
2. Sử dụng APIs thương mại (Suno, Udio, ElevenLabs Music) với giấy phép mỗi thế hệ.
3. Huấn luyện trên danh mục sở hữu hoặc được cấp phép (hầu hết các doanh nghiệp kết thúc ở đây).
4. Gắn thẻ các thế hệ bằng hình mờ + siêu dữ liệu.

## Tự xây dựng

### Bước 1: tạo bằng MusicGen

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

Ba kích thước: `small` (300M, nhanh), `medium` (1.5B), `large` (3.3B). Nhỏ là đủ để "ý tưởng hạ cánh".

### Bước 2: điều hòa giai điệu

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

Giai điệu MusicGen lấy sắc độ và giữ nguyên giai điệu trong khi hoán đổi âm sắc. Hữu ích cho "cho tôi giai điệu này như một tứ tấu dây."

### Bước 3: Đánh giá FAD

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

Tính toán khoảng cách VGGish-embedding. Hữu ích cho các bài kiểm tra hồi quy cấp thể loại; không phải là một sự thay thế cho người nghe.

### Bước 4: thêm vào quy trình làm việc LLM nhạc

Kết hợp với các ý tưởng từ Bài học 7-8:

```python
prompt = "Write a 30-second jazz loop. Describe the drums, bass, and piano voicing."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## Ứng dụng

| Mục tiêu | Stack |
|------|-------|
| Thiết kế âm thanh nhạc cụ | Mở âm thanh ổn định |
| Trò chơi / nhạc thích ứng | Google Lyria RealTime (đã đóng) |
| Bài hát đầy đủ có giọng hát (quảng cáo) | Suno v5 hoặc Udio v4 với giấy phép rõ ràng |
| Bài hát đầy đủ với giọng hát (mở) | ACE-Step XL hoặc YuE |
| Leng keng quảng cáo ngắn | Giai điệu MusicGen có điều kiện trên một tham chiếu ngân nga |
| Nền video âm nhạc | MusicGen + Khuếch tán video ổn định |

## Những cạm bẫy vẫn ship vào năm 2026

- **Rửa bản quyền prompts.** "Bài hát theo phong cách của Taylor Swift" - thương mại Suno/Udio lọc những thứ này ngay bây giờ, mở models không. Thêm danh sách bộ lọc của riêng bạn.
- **Lặp lại / trôi qua 30 giây.** Vòng lặp models AR. Crossfade nhiều thế hệ hoặc sử dụng ACE-Step để gắn kết cấu trúc.
- **Nhịp độ trôi dạt.** Models đi lang thang khỏi BPM. Sử dụng thẻ BPM trong prompt và sau bộ lọc với `beat_track` của librosa.
- **Độ rõ ràng của giọng hát.** Suno rất xuất sắc; models cởi mở thường nhão về lời nói. Nếu lời bài hát quan trọng, hãy sử dụng API hoặc fine-tune thương mại.
- **Đầu ra đơn âm.** Mở models tạo đơn âm hoặc âm thanh nổi giả. Nâng cấp với tái tạo âm thanh nổi thích hợp (ezst, khuếch tán âm thanh nổi của Cartesia).

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-music-designer.md`. Chọn model, chiến lược cấp phép, kế hoạch độ dài/cấu trúc và siêu dữ liệu tiết lộ để triển khai thế hệ âm nhạc.

## Bài tập

1. **Dễ dàng.** Chạy `code/main.py`. Nó tạo ra tiến trình hợp âm "tổng quát" + mẫu trống dưới dạng biểu tượng ASCII - một phim hoạt hình thế hệ âm nhạc. Phát lại thông qua bất kỳ trình kết xuất MIDI nào nếu bạn muốn.
2. **Trung bình.** Cài đặt `audiocraft`, tạo clip dài 10 giây trên 4 thể loại prompts với MusicGen-small, đo FAD so với một bộ thể loại tham chiếu.
3. **Khó.** Sử dụng ACE-Step (hoặc MusicGen-melody), tạo ra ba biến thể của cùng một giai điệu với prompts âm sắc khác nhau. Tính toán sự tương đồng của CLAP với prompt để xác minh alignment.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| THỜI TRANG | FID âm thanh | Khoảng cách Fréchet giữa các phân phối embedding của thực và được tạo ra. |
| Sắc độ | Giai điệu dưới dạng cao độ | vector 12 độ mờ trên mỗi khung hình; đầu vào điều hòa giai điệu. |
| Thân cây | Rãnh nhạc cụ | Tách âm trầm / trống / giọng hát / giai điệu dưới dạng WAV. |
| Sơn | Tái tạo một phần | Che cửa sổ thời gian; model tái tạo chỉ như vậy. |
| Vỗ tay | CLIP văn bản-âm thanh | embedding âm thanh-văn bản tương phản; Đánh giá văn bản-âm thanh alignment. |
| Bộ mã hóa | Bộ giải mã âm nhạc | Codec thần kinh của Meta được sử dụng bởi MusicGen; 32 kHz, 4 sách mã. |

## Đọc thêm

- [Copet et al. (2023). MusicGen](https://arxiv.org/abs/2306.05284) - benchmark tự hồi quy mở.
- [Evans et al. (2024). Stable Audio Open](https://arxiv.org/abs/2407.14358) — mặc định thiết kế âm thanh.
- [ACE-Step](https://github.com/ace-step/ACE-Step) - mở trình tạo toàn bộ bài hát 4B, Tháng Tư 2026.
- [Suno v5 platform docs](https://suno.com) - công ty dẫn đầu về chất lượng thương mại.
- [AudioLDM2](https://arxiv.org/abs/2308.05734) - khuếch tán tiềm ẩn cho âm nhạc + hiệu ứng âm thanh.
- [WMG-Suno settlement coverage](https://www.musicbusinessworldwide.com/suno-warner-music-settlement/) - Tiền lệ tháng 11 năm 2025.
