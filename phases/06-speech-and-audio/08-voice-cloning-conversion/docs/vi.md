# Sao chép giọng nói và chuyển đổi giọng nói

> Sao chép giọng nói đọc văn bản của bạn bằng giọng nói của người khác. Chuyển đổi giọng nói viết lại giọng nói của bạn thành giọng nói của người khác trong khi vẫn giữ nguyên những gì bạn nói. Cả hai đều dựa trên cùng một sự phân chiếm: tách biệt danh tính người nói khỏi nội dung.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 6 · 06 (Nhận dạng loa), Giai đoạn 6 · 07 (TTS)
**Thời lượng:** ~75 phút

## Vấn đề

Vào năm 2026, một đoạn âm thanh dài 5 giây là đủ để tạo ra một bản sao chất lượng cao của giọng nói của bất kỳ ai với GPU tiêu dùng. ElevenLabs, F5-TTS, OpenVoice v2, VoiceBox đều ship zero-shot hoặc few-shot sao chép. Công nghệ này là một điều may mắn (TTS trợ năng, lồng tiếng, giọng nói hỗ trợ) và một vũ khí (cuộc gọi lừa đảo, deepfake chính trị, trộm cắp IP).

Hai nhiệm vụ liên quan chặt chẽ:

- **Sao chép giọng nói (mặt TTS):** văn bản + giọng nói tham chiếu 5 giây → âm thanh trong giọng nói đó.
- **Chuyển đổi giọng nói (phía giọng nói):** âm thanh nguồn (người A nói X) + giọng nói tham chiếu của người B → âm thanh của B nói X.

Cả hai đều yếu tố dạng sóng vào (nội dung, người nói, văn xuôi) và kết hợp lại nội dung từ nguồn này với người nói từ nguồn khác.

Ràng buộc chính mà bạn hiện ship phải tuân theo vào năm 2026: **Cổng hình mờ và cổng chấp thuận được yêu cầu về mặt pháp lý trong EU (Đạo luật AI, có hiệu lực vào tháng 8 năm 2026) và ở California (AB 2905, có hiệu lực vào năm 2025)**. pipeline của bạn phải phát ra hình mờ không nghe được và từ chối các bản sao không có sự đồng thuận.

## Khái niệm

![Voice cloning vs conversion: factorize, swap speaker, recombine](../assets/voice-cloning.svg)

**Zero-shot nhân bản.** Chuyển một clip dài 5 giây cho một model đã được huấn luyện trên hàng nghìn loa. Người nói encoder ánh xạ clip đến embedding loa; Các điều kiện TTS decoder trên embedding đó cộng với văn bản.

Được sử dụng bởi: F5-TTS (2024), YourTTS (2022), XTTS v2 (2024), OpenVoice v2 (2024).

**Few-shot fine-tuning.** Ghi lại 5-30 phút của giọng nói mục tiêu. LoRA-fine-tune một model cơ sở trong một giờ. Chất lượng nhảy vọt từ "ổn" đến "không thể phân biệt". Coqui và ElevenLabs đều ủng hộ mô hình này; cộng đồng sử dụng nó với F5-TTS.

**Chuyển đổi giọng nói (VC).** Hai họ:

- **Nhận dạng-tổng hợp.** Chạy model giống ASR để trích xuất biểu diễn nội dung (ví dụ: posteriors âm vị mềm, PPG), sau đó tổng hợp lại với embedding của người nói đích. Mạnh mẽ với ngôn ngữ và trọng âm. Được sử dụng bởi KNN-VC (2023), Diff-HierVC (2023).
- **Gỡ rối.** Huấn luyện một bộ mã hóa tự động tách nội dung, người nói và ngữ điệu trong không gian tiềm ẩn ở nút cổ chai. Hoán đổi embedding loa ở inference. Chất lượng thấp hơn nhưng nhanh hơn. Được sử dụng bởi AutoVC (2019), các biến thể VITS-VC.

**Sao chép dựa trên codec thần kinh (2024+).** VALL-E, VALL-E 2, NaturalSpeech 3, VoiceBox — coi âm thanh là tokens rời rạc từ SoundStream / EnCodec, huấn luyện một model tự hồi quy hoặc khớp luồng lớn qua codec tokens. Chất lượng tương đương với ElevenLabs trong prompts ngắn.

### Đạo đức, không phải là một chốt

**Watermarking.** PerTh (Perth) và SilentCipher (2024) nhúng ID ~16-32 bit một cách không thể nhận thấy trong âm thanh. Tồn tại sau khi mã hóa lại, streaming và các chỉnh sửa phổ biến. Mã nguồn mở sẵn sàng Production.

**Cổng đồng ý.** Phải ghép nối mọi đầu ra sao chép với bản ghi đồng ý có thể xác minh. "Tôi, Rohit, vào ngày 22 tháng 4 năm 2026, cho phép tiếng nói này cho mục đích X." Lưu trữ trong nhật ký rõ ràng giả mạo.

**Phát hiện.** AASIST, RawNet2 và Wav2Vec2-AASIST ship làm máy dò. Thử thách ASVspoof 2025 đã công bố EER là 0,8–2,3% đối với các máy dò hiện đại so với đầu ra ElevenLabs, VALL-E 2 và Bark.

### Số (2026)

| Model | Zero-shot? | SECS (sim mục tiêu) | WER (thông tin) | Tham số |
|-------|-----------|--------------------|--------------|--------|
| F5-TTS | Có | 0.72 | 2.1% | 335 triệu |
| XTTS phiên bản 2 | Có | 0.65 | 3.5% | 470 triệu |
| OpenVoice phiên bản 2 | Có | 0.70 | 2.8% | 220 triệu |
| VALL-E 2 | Có | 0.77 | 2.4% | 370 triệu |
| Hộp thoại | Có | 0.78 | 2.1% | 330 triệu |

SECS > 0,70 thường không thể phân biệt được với mục tiêu đối với hầu hết người nghe.

## Tự xây dựng

### Bước 1: phân hủy với nhận dạng-tổng hợp (demo chỉ mã trong main.py)

```python
def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    speaker_emb = target_embedder.encode(ref_audio)
    mel = tts_model(text, speaker=speaker_emb)
    return vocoder(mel)
```

Khái niệm đơn giản; khối lượng thực hiện đang ở `tts_model` và diễn giả encoder.

### Bước 2: zero-shot nhân bản với F5-TTS

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

Bản ghi tham khảo phải khớp chính xác với âm thanh; không phù hợp phá vỡ alignment.

### Bước 3: chuyển đổi giọng nói với KNN-VC

```python
import torch
from knnvc import KNNVC  # 2023 model, https://github.com/bshall/knn-vc
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VC chạy WavLM để trích xuất embeddings trên mỗi khung hình cho nhóm nguồn và đích, sau đó thay thế từng khung nguồn bằng hàng xóm gần nhất trong nhóm. Phi tham số, hoạt động với một phút phát biểu mục tiêu.

### Bước 4: nhúng hình mờ

```python
from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)   # returns payload bytes
```

~32 bit payload, có thể phát hiện sau khi mã hóa lại MP3 và nhiễu nhẹ.

### Bước 5: cổng chấp thuận

```python
def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "Signed consent required"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## Ứng dụng

stack năm 2026:

| Tình huống | Chọn |
|-----------|------|
| Bản sao zero-shot 5 giây, mã nguồn mở | F5-TTS hoặc OpenVoice v2 |
| Nhân bản production thương mại | ElevenLabs Sao chép giọng nói tức thì v2.5 |
| Chuyển đổi giọng nói (viết lại) | KNN-VC hoặc Diff-HierVC |
| Nhiều loa fine-tune | Bộ chuyển đổi loa StyleTTS 2 + |
| Nhân bản đa ngôn ngữ | XTTS v2 hoặc VALL-E X |
| Phát hiện deepfake | Wav2Vec2-AASIST |

## Cạm bẫy

- **Bản ghi tham chiếu bị lệch.** F5-TTS và tương tự yêu cầu văn bản tham chiếu khớp chính xác với âm thanh tham chiếu, bao gồm dấu câu.
- **Tham chiếu vang dội.** Echo giết chết bản sao. Ghi âm khô ráo, đóng mic.
- **Sự không phù hợp về cảm xúc.** Training tham chiếu "vui vẻ" tạo ra các bản sao vui vẻ của mọi thứ. Khớp cảm xúc tham chiếu với mục tiêu sử dụng.
- **Rò rỉ ngôn ngữ.** Nhân bản một người nói tiếng Anh sau đó yêu cầu model nói tiếng Pháp thường mang trọng âm; sử dụng models đa ngôn ngữ (XTTS, VALL-E X).
- **Không có hình mờ.** Không thể vận chuyển hợp pháp ở EU từ tháng 8 năm 2026.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-voice-cloner.md`. Thiết kế pipeline nhân bản hoặc chuyển đổi với cổng đồng ý + hình mờ + mục tiêu chất lượng.

## Bài tập

1. **Dễ dàng.** Chạy `code/main.py`. Thể hiện hoán đổi loa-embedding bằng cách tính toán cosin giữa hai "loa" hoán đổi trước và sau.
2. **Trung bình.** Sử dụng OpenVoice v2 để sao chép giọng nói của chính bạn. Đo SECS giữa tham chiếu và sao chép. Đo lường CER thông qua Whisper.
3. **Khó.** Áp dụng hình mờ SilentCipher cho 20 bản sao, chạy chúng qua mã hóa + giải mã MP3 128 kbps, phát hiện payload. Báo cáo accuracy bit.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Zero-shot bản sao | 5 giây là đủ | Pretrained model + loa embedding; không training. |
| PPG | Posteriorgram ngữ âm | Mỗi khung hình ASR posteriors được sử dụng làm đại diện nội dung bất khả tri về ngôn ngữ. |
| KNN-VC | Chuyển đổi hàng xóm gần nhất | Thay thế từng khung nguồn bằng khung nhóm mục tiêu gần nhất. |
| Bộ giải mã thần kinh TTS | Phong cách VALL-E | AR model hơn EnCodec/SoundStream tokens. |
| Hình mờ | Chữ ký không nghe được | Các bit được nhúng trong âm thanh, tồn tại sau khi mã hóa lại. |
| GIÂY | Sao chép độ trung thực | Cosin giữa embeddings loa mục tiêu và loa nhân bản. |
| NGƯỜI YÊU | Máy dò deepfake | model chống giả mạo; phát hiện giọng nói tổng hợp. |

## Đọc thêm

- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — sao chép zero-shot SOTA mã nguồn mở.
- [Baevski et al. / Microsoft (2023). VALL-E](https://arxiv.org/abs/2301.02111) và [VALL-E 2 (2024)](https://arxiv.org/abs/2406.05370) — TTS codec thần kinh.
- [Qian et al. (2019). AutoVC](https://arxiv.org/abs/1905.05879) - chuyển đổi giọng nói dựa trên gỡ rối.
- [Baas, Waubert de Puiseau, Kamper (2023). KNN-VC](https://arxiv.org/abs/2305.18975) — VC dựa trên truy xuất.
- [SilentCipher (2024) — Audio Watermarking](https://github.com/sony/silentcipher) — Hình mờ âm thanh 32 bit sẵn sàng cho production.
- [ASVspoof 2025 results](https://www.asvspoof.org/) - Cuộc đua vũ trang giữa máy dò và bộ tổng hợp, cập nhật năm 2026.
