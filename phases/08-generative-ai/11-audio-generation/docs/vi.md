# Tạo âm thanh

> Âm thanh là tín hiệu 1-D ở tần số 16-48 kHz. Một clip dài năm giây là 80-240k mẫu. Không transformer trực tiếp tham gia vào trình tự đó. Giải pháp cho mọi model âm thanh production vào năm 2026 đều giống nhau: codec thần kinh (Encodec, SoundStream, DAC) nén âm thanh thành tokens rời rạc ở tần số 50-75 Hz và model transformer hoặc khuếch tán tạo ra tokens.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 6 · 02 (Âm thanh Features), Giai đoạn 6 · 04 (ASR), Giai đoạn 8 · 06 (DDPM)
**Thời lượng:** ~45 phút

## Vấn đề

Ba tác vụ tạo âm thanh:

1. **Chuyển văn bản thành giọng nói.** Cung cấp văn bản, tạo giọng nói. Lời nói rõ ràng là dải hẹp và có cấu trúc ngữ âm mạnh mẽ - được giải quyết tốt bằng cách transformer trên tokens. VALL-E (Microsoft), NaturalSpeech 3, ElevenLabs, OpenAI TTS.
2. **Tạo nhạc.** Được đưa ra một prompt (văn bản, giai điệu, tiến trình hợp âm, thể loại), sản xuất âm nhạc. Phân phối rộng hơn nhiều. MusicGen (Meta), Âm thanh ổn định 2.5, Suno v4, Udio, Riffusion.
3. **Hiệu ứng âm thanh / thiết kế âm thanh.** Cho một prompt, tạo ra âm thanh xung quanh hoặc Foley. AudioGen, AudioLDM 2, Mở âm thanh ổn định.

Cả ba đều chạy trên cùng một chất nền: codec âm thanh thần kinh + token-AR hoặc bộ tạo khuếch tán.

## Khái niệm

![Audio generation: codec tokens + transformer or diffusion](../assets/audio-generation.svg)

### Codec âm thanh thần kinh

Encodec (Meta, 2022), SoundStream (Google, 2021), Codec âm thanh mô tả (DAC, 2023). Một encoder tích chập nén dạng sóng thành một vector mỗi bước thời gian; vector quantization dư (RVQ) chuyển đổi mỗi vector thành một dòng các chỉ số sách mã K. Decoder đảo ngược nó. Âm thanh 24 kHz ở tốc độ 2 kbps sử dụng 8 sách mã RVQ ở 75 Hz = 600 tokens/sec.

```
waveform (16000 samples/sec)
    └─ encoder conv ─┐
                     ├─ RVQ layer 1 → indices at 75 Hz
                     ├─ RVQ layer 2 → indices at 75 Hz
                     ├─ ...
                     └─ RVQ layer 8
```

### Hai mô hình tổng quát ở trên cùng

**Token-tự hồi quy.** Làm phẳng RVQ tokens thành một trình tự, chạy transformer chỉ decoder. MusicGen sử dụng "song song trì hoãn" để phát ra các luồng sách mã K song song với độ lệch trên mỗi luồng. VALL-E tạo tokens giọng nói từ prompt văn bản + mẫu giọng nói 3 giây.

**Khuếch tán tiềm ẩn.** Đóng gói codec tokens dưới dạng tiềm ẩn liên tục hoặc model chúng bằng sự khuếch tán phân loại. Âm thanh ổn định 2.5 sử dụng kết hợp luồng trên các tiềm ẩn âm thanh liên tục. AudioLDM 2 sử dụng khuếch tán văn bản thành âm thanh.

Xu hướng 2024-2026: kết hợp luồng đang chiến thắng cho âm nhạc (inference nhanh hơn, mẫu sạch hơn) trong khi token-AR vẫn thống trị giọng nói vì nó tự nhiên là nhân quả và phát trực tuyến tốt.

## Production cảnh quan

| Hệ thống | Nhiệm vụ | Xương sống | Độ trễ |
|--------|------|----------|---------|
| ElevenLabs V3 | TTS | Token-AR + bộ mã hóa thần kinh | ~300ms token đầu tiên |
| OpenAI GPT-4o âm thanh | Giọng nói song công | AR đa phương thức đầu cuối | ~200 mili giây |
| Lời nói tự nhiên 3 | TTS | Khớp luồng tiềm ẩn | Không streaming |
| Âm thanh ổn định 2.5 | Âm nhạc / SFX | DiT + khớp luồng trên các tiềm ẩn âm thanh | ~10 giây cho clip 1 phút |
| Suno v4 | Bài hát đầy đủ | Không được tiết lộ; Nghi ngờ token-AR | ~30 giây mỗi bài hát |
| Udio phiên bản 1.5 | Bài hát đầy đủ | Không được tiết lộ | ~30 giây mỗi bài hát |
| MusicGen 3.3B | Âm nhạc | Token-AR trên Encodec 32kHz | Thời gian thực |
| Thủ công âm thanh 2 | Âm nhạc + SFX | Kết hợp luồng | ~5 giây cho clip 5 giây |
| Riffusion v2 | Âm nhạc | Khuếch tán quang phổ | ~10 giây |

## Tự xây dựng

`code/main.py` mô phỏng ý tưởng cốt lõi: huấn luyện một token transformer tiếp theo nhỏ trên các chuỗi "token âm thanh" tổng hợp được tạo ra từ hai "kiểu" riêng biệt (xen kẽ tokens thấp và cao cho kiểu A, đoạn đường dốc đơn điệu cho kiểu B). Điều kiện về kiểu dáng và mẫu.

### Bước 1: tokens âm thanh tổng hợp

```python
def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # "speech-like": alternating
        return [i % vocab_size for i in range(length)]
    # "music-like": ramp
    return [(i * 3) % vocab_size for i in range(length)]
```

### Bước 2: huấn luyện một token dự đoán nhỏ

Một công cụ dự đoán kiểu bigram có điều kiện theo phong cách. Vấn đề là mẫu: codec tokens → entropy chéo training → sampling tự hồi quy.

### Bước 3: lấy mẫu có điều kiện

Với token kiểu và token bắt đầu, hãy lấy mẫu token tiếp theo từ phân phối dự đoán. Tiếp tục trong 20-40 tokens.

## Cạm bẫy

- **Chất lượng codec giới hạn chất lượng đầu ra.** Nếu codec không thể thể hiện âm thanh một cách trung thực, thì không có chất lượng trình tạo nào giúp ích được gì. DAC là mở tốt nhất hiện tại.
- **Tích lũy lỗi RVQ.** Mỗi lớp RVQ models phần còn lại của lớp trước đó. Lỗi trên lớp 1 lan truyền. Sampling với temperature 0 trên các lớp cao hơn sẽ hữu ích.
- **Cấu trúc âm nhạc.** 30 giây tokens là 20k + tokens ở 75 Hz. Khó cho transformers. MusicGen sử dụng cửa sổ trượt + prompt tiếp tục; Âm thanh ổn định sử dụng các clip ngắn hơn + làm mờ chéo.
- **Artifacts ở ranh giới.** Làm mờ chéo giữa các clip được tạo cần thêm chồng chéo cẩn thận.
- **Khẩu vị dữ liệu sạch.** Trình tạo nhạc cần hàng chục nghìn giờ âm nhạc được cấp phép. Vụ kiện Suno / Udio RIAA (2024) đã đưa điều này lên bề mặt.
- **Đạo đức nhân bản giọng nói.** Một mẫu 3 giây cộng với một prompt văn bản là đủ để VALL-E / XTTS / ElevenLabs sao chép giọng nói. Mọi production model đều cần phát hiện lạm dụng + danh sách chọn không tham gia.

## Ứng dụng

| Nhiệm vụ | stack năm 2026 |
|------|------------|
| TTS thương mại | ElevenLabs, OpenAI TTS hoặc Azure Neural |
| Sao chép giọng nói (đã xác minh sự đồng ý) | XTTS v2 (mở) hoặc ElevenLabs Pro |
| Nhạc nền, nhanh | Âm thanh ổn định 2.5 API, Suno hoặc Udio |
| Âm nhạc có lời bài hát | Suno v4 hoặc Udio v1.5 |
| Hiệu ứng âm thanh / Foley | AudioCraft 2, ElevenLabs SFX hoặc Mở âm thanh ổn định |
| agent thoại thời gian thực | GPT-4o thời gian thực hoặc Gemini Trực tiếp |
| Nghiên cứu âm nhạc trọng lượng mở | MusicGen 3.3B, Mở âm thanh ổn định 1.0, AudioLDM 2 |
| Lồng tiếng / dịch thuật | HeyGen, lồng tiếng ElevenLabs |

## Sản phẩm bàn giao

Tiết kiệm `outputs/skill-audio-brief.md`. Skill lấy một bản tóm tắt âm thanh (nhiệm vụ, thời lượng, kiểu, giọng nói, giấy phép) và đầu ra: lưu trữ model +, định dạng prompt (thẻ thể loại, mô tả kiểu, điểm đánh dấu cấu trúc), codec + trình tạo + chuỗi vocoder, giao thức hạt giống và kế hoạch đánh giá (điểm MOS / CLAP / CER cho TTS / người dùng A/B).

## Bài tập

1. **Dễ dàng.** Chạy `code/main.py` và thiết lập phong cách rõ ràng. Xác minh các trình tự được tạo khớp với mẫu của kiểu.
2. **Trung bình.** Thêm giải mã song song trễ: mô phỏng 2 luồng tokens phải lệch 1 bước. Huấn luyện một người dự đoán chung.
3. **Hard.** Sử dụng HuggingFace transformers để chạy MusicGen-small cục bộ. Tạo clip dài 10 giây với ba prompts khác nhau; A/B để tuân thủ phong cách.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Bộ giải mã | "Nén thần kinh" | Encoder / decoder cho âm thanh; đầu ra điển hình là 50-75 Hz tokens. |
| RVQ | "VQ còn lại" | Xếp tầng của bộ định lượng K mỗi models phần còn lại của trước đó. |
| Token | "Một ký hiệu codec" | Lập chỉ mục rời rạc vào một cuốn sách mã; 1024 hoặc 2048 điển hình. |
| Song song trì hoãn | "Sách mã bù đắp" | Phát ra các luồng K token với độ lệch so le để giảm độ dài trình tự. |
| Kết hợp luồng | "Chiến thắng năm 2024 cho âm thanh" | Thay thế đường dẫn thẳng hơn cho sự khuếch tán; sampling nhanh hơn. |
| prompt giọng nói | "Mẫu 3 giây" | Loa embedding hoặc tiền tố token điều khiển giọng nói nhân bản. |
| Quang phổ Mel | "Hình ảnh" | Quang phổ tri giác log-magnitude; được sử dụng bởi nhiều hệ thống TTS. |
| Bộ mã hóa | "Mel để vẫy tay" | Thành phần thần kinh chuyển đổi quang phổ mel trở lại âm thanh. |

## Production lưu ý: âm thanh là một vấn đề streaming

Âm thanh là phương thức đầu ra duy nhất mà người dùng mong đợi sẽ đến *khi nó được tạo ra*, không phải tất cả cùng một lúc. Nói một cách production, điều này có nghĩa là TPOT quan trọng (Time Per Output Token) vì tốc độ nghe của người dùng là thông lượng mục tiêu - không phải tốc độ đọc của họ. Đối với âm thanh 16kHz được mã hóa ở ~75 tokens/second (Encodec), server phải tạo ≥75 tokens/sec cho mỗi người dùng để giữ cho phát lại mượt mà.

Hai hậu quả kiến trúc:

- **models âm thanh phù hợp với luồng không thể phát trực tuyến một cách tầm thường.** Stable Audio 2.5 và AudioCraft 2 hiển thị độ dài clip cố định trong một lần chuyển tiếp. Để phát trực tuyến, bạn chia nhỏ clip và chồng lên ranh giới — hãy nghĩ đến khuếch tán cửa sổ trượt — thêm 100-300ms độ trễ so với codec AR model.

Nếu sản phẩm là "trò chuyện thoại trực tiếp" hoặc "tiếp tục nhạc trong thời gian thực", hãy chọn đường dẫn AR codec. Nếu đó là "hiển thị clip dài 30 giây khi gửi", thì tính năng so khớp luồng sẽ chiến thắng về chất lượng và tổng độ trễ.

## Đọc thêm

- [Défossez et al. (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — tiêu chuẩn codec.
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — codec âm thanh thần kinh đầu tiên được sử dụng rộng rãi.
- [Kumar et al. (2023). High-Fidelity Audio Compression with Improved RVQGAN (DAC)](https://arxiv.org/abs/2306.06546) - DAC.
- [Wang et al. (2023). Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)](https://arxiv.org/abs/2301.02111) — VALL-E.
- [Copet et al. (2023). Simple and Controllable Music Generation (MusicGen)](https://arxiv.org/abs/2306.05284) - MusicGen.
- [Liu et al. (2023). AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734) - AudioLDM 2.
- [Stability AI (2024). Stable Audio 2.5](https://stability.ai/news/introducing-stable-audio-2-5) - Chuyển văn bản thành nhạc năm 2025 với kết hợp luồng.
