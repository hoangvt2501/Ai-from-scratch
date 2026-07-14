# Capstone 12 — Video Understanding Pipeline (Cảnh, QA, Tìm kiếm)

> Mười hai phòng thí nghiệm sản xuất Marengo + Pegasus. VideoDB shipped API CRUD cho video. Molmo 2 của AI2 đã xuất bản VLM checkpoints mở. Gemini ngữ cảnh dài xử lý hàng giờ video nguyên bản. TimeLens-100K xác định grounding thời gian trên quy mô lớn. pipeline năm 2026 đã được giải quyết: phân đoạn cảnh, chú thích mỗi cảnh + embedding, alignment bản ghi, chỉ mục nhiều vector và truy vấn trả lời bằng dấu thời gian (bắt đầu, kết thúc) cộng với bản xem trước khung hình. Viên đá đỉnh đang nuốt 100 giờ, đánh vào benchmarks công cộng và đo lường ảo giác về các câu hỏi đếm và hành động.

**Loại:** Đá đỉnh
**Ngôn ngữ:** Python (pipeline), TypeScript (UI)
**Kiến thức tiên quyết:** Giai đoạn 4 (CV), Giai đoạn 6 (phát biểu), Giai đoạn 7 (transformers), Giai đoạn 11 (LLM kỹ thuật), Giai đoạn 12 (đa phương thức), Giai đoạn 17 (cơ sở hạ tầng)
**Các giai đoạn thực hiện:** P4 · P6 · P7 · P11 · P12 · Trang 17
**Thời lượng:** 30 giờ

## Vấn đề

QA video dạng dài là vấn đề đa phương thức ngốn băng thông nhất ở quy mô năm 2026. Gemini 2.5 Pro có thể đọc video dài 2 giờ nguyên bản, nhưng việc nhập 100 giờ video vào kho dữ liệu có thể truy vấn vẫn yêu cầu chỉ mục cấp cảnh. Hình dạng production kết hợp phân đoạn cảnh (TransNetV2 hoặc PySceneDetect), phụ đề cho mỗi cảnh với VLM (Gemini 2.5, Qwen3-VL-Max hoặc Molmo 2), alignment bản chép lời (Whisper-v3-turbo với dấu thời gian từ) và chỉ mục nhiều vector lưu trữ phụ đề, embedding khung hình và bản chép lời cạnh nhau. Truy vấn pipeline câu trả lời với dấu thời gian (bắt đầu, kết thúc) cộng với bản xem trước khung.

Benchmarks là công khai (ActivityNet-QA, NeXT-GQA) cộng với bộ tùy chỉnh 100 truy vấn của riêng bạn. Ảo giác khi đếm và các câu hỏi kiểu hành động là thất bại được biết đến class; capstone đo lường nó một cách rõ ràng.

## Khái niệm

Ba pipelines chạy song song khi nhập. **Phân đoạn cảnh** cắt video thành các cảnh. **VLM phụ đề** tạo chú thích cho mỗi cảnh và embedding khung hình từ khung hình chính. **ASR alignment** tạo dấu thời gian cấp từ. Ba luồng được nối với nhau bởi (scene_id, phạm vi thời gian). Mỗi cảnh có ba loại vector trong chỉ mục nhiều vector (Qdrant): embedding chú thích, embedding khung hình chính, embedding bản ghi.

Tại thời điểm truy vấn, câu hỏi ngôn ngữ tự nhiên sẽ chống lại cả ba vectors; kết quả merge với RRF; bộ chuyển đổi grounding thời gian (kiểu TimeLens) tinh chỉnh cửa sổ (bắt đầu, kết thúc) trong cảnh trên cùng. Bộ tổng hợp VLM (Gemini 2.5 Pro hoặc Qwen3-VL-Max) lấy truy vấn + cảnh hàng đầu + khung hình được cắt và câu trả lời với dấu thời gian được trích dẫn và bản xem trước khung hình.

Đo lường ảo giác rất quan trọng. Câu hỏi đếm ("có bao nhiêu người vào phòng?") và kiểu hành động ("đầu bếp có rót trước khi khuấy không?") nổi tiếng là không đáng tin cậy. Báo cáo accuracy riêng biệt với các câu hỏi mô tả.

## Kiến trúc

```
video file / URL
      |
      v
PySceneDetect / TransNetV2  (scene segmentation)
      |
      +--- per-scene keyframe --- VLM caption + frame embedding
      |                            (Gemini 2.5 Pro / Qwen3-VL-Max / Molmo 2)
      |
      +--- audio channel --- Whisper-v3-turbo ASR + word timestamps
      |
      v
multi-vector Qdrant: {caption_emb, keyframe_emb, transcript_emb}
      |
query:
  dense queries against all three -> RRF merge -> top-k scenes
      |
      v
TimeLens / VideoITG temporal grounding (refine start/end within scene)
      |
      v
VLM synth: query + top scenes + frame previews
      |
      v
answer + (start, end) timestamps + frame thumbs + citations
```

## Stack

- Phân đoạn cảnh: TransNetV2 (hiện đại 2024-26) hoặc PySceneDetect
- ASR: Whisper-v3-turbo thông qua tiếng thì thầm nhanh hơn với dấu thời gian từ
- VLM phụ đề + người trả lời: Gemini 2.5 Pro hoặc Qwen3-VL-Max hoặc Molmo 2
- grounding thời gian: Bộ chuyển đổi được huấn luyện TimeLens-100K hoặc VideoITG
- Mục lục: Qdrant với hỗ trợ nhiều vector (chú thích / khung / bản ghi)
- Giao diện người dùng: Next.js 15 với trình phát video HTML5 và hình thu nhỏ cảnh
- Đánh giá: ActivityNet-QA, NeXT-GQA, bộ gắn nhãn tay 100 câu hỏi tùy chỉnh
- Ảo giác benchmark: đếm và tập hợp con kiểu hành động với nhãn cầm tay

## Tự xây dựng

1. **Nhập walker.** Chấp nhận URL YouTube hoặc MP4 cục bộ. Giảm tỷ lệ xuống 720p nếu cần. Kiên trì `{video_id, file_path}`.

2. **Phân đoạn cảnh.** Chạy TransNetV2 hoặc PySceneDetect để tạo `[{scene_id, start_ms, end_ms, keyframe_path}]`. Mục tiêu 100 giờ: ~6k-8k cảnh.

3. **ASR vượt qua.** Chạy Whisper-v3-turbo trên âm thanh; xuất dấu thời gian cấp từ; chia thành các lát bản chép lời cho mỗi cảnh.

4. **VLM chú thích.** Mỗi cảnh, hãy gọi Gemini 2.5 Pro (hoặc Qwen3-VL-Max) với khung hình chính và mẫu phụ đề ngắn. Sản xuất chú thích + embedding khung.

5. **Chỉ mục đa vector.** Bộ sưu tập Qdrant với ba vectors được đặt tên. Payload: `{video_id, scene_id, start_ms, end_ms, keyframe_url}`.

6. **Truy vấn.** Câu hỏi ngôn ngữ tự nhiên kích hoạt ba truy vấn dày đặc; merge với sự hợp nhất cấp bậc đối ứng; top-k = 5 cảnh.

7. **grounding thời gian.** Chạy bộ chuyển đổi kiểu TimeLens trên cảnh trên cùng để tinh chỉnh cửa sổ (bắt đầu, kết thúc) trong cảnh.

8. **VLM synth.** Gọi Gemini 2.5 Pro với truy vấn + 3 clip cảnh hàng đầu (dưới dạng hình ảnh hoặc clip ngắn) + bản ghi. Yêu cầu trích dẫn `(video_id, start_ms, end_ms)`.

9. **Eval.** Chạy ActivityNet-QA và NeXT-GQA. Xây dựng một tập hợp tùy chỉnh 100 truy vấn. Báo cáo tổng thể accuracy + phân tích trên mỗi class (đếm, hành động, mô tả).

## Ứng dụng

```
$ video-qa ask --url=https://youtube.com/watch?v=X "how many cars pass the intersection in the first minute?"
[scene]    23 scenes detected
[asr]      transcript complete, 4m12s
[index]    69 vectors written (23 scenes x 3)
[query]    top scene: scene 3 [01:32-01:54], confidence 0.84
[ground]   refined window: [00:12-00:58]
[synth]    gemini 2.5 pro, 1.4s
answer:    5 cars pass the intersection between 00:12 and 00:58.
citations: [scene 3: 00:12-00:58]
          [frame preview at 00:14, 00:27, 00:44, 00:51, 00:57]
```

## Sản phẩm bàn giao

`outputs/skill-video-qa.md` là sản phẩm được giao. Với URL YouTube hoặc video đã tải lên, pipeline lập chỉ mục các cảnh và trả lời các câu hỏi bằng các trích dẫn có dấu thời gian.

| Trọng lượng | Tiêu chí | Cách đo lường |
| :-: | --- | --- |
| 25 | IoU grounding thời gian | Giao lộ trên liên minh trên bộ grounding bị giữ |
| 20 | QA accuracy | NeXT-GQA và 100 truy vấn tùy chỉnh |
| 20 | Thông lượng nhập | Số giờ video trên mỗi đô la chi tiêu |
| 20 | Giao diện người dùng và UX trích dẫn | Liên kết dấu thời gian, dải hình thu nhỏ, nhảy đến khung hình |
| 15 | Tỷ lệ ảo giác | Đếm và loại hành động accuracy riêng biệt |
| **100** |||

## Bài tập

1. Hoán đổi Gemini 2.5 Pro lấy Qwen3-VL-Max trên thẻ phụ đề. Báo cáo delta chất lượng phụ đề trên mẫu 50 cảnh do con người xếp hạng.

2. Giảm embedding khung hình trên mỗi cảnh xuống một vector gộp thay vì nhiều vector. Đo hồi quy truy xuất.

3. Xây dựng chế độ "đếm nghiêm ngặt": bộ tổng hợp trích xuất từng trường hợp được đếm bằng dấu thời gian và người dùng nhấp để xác minh. Đo lường xem xác minh người dùng có làm giảm ảo giác hay không.

4. Benchmark phí nhập: số giờ video trên mỗi đô la trên ba lựa chọn VLM. Chọn điểm ngọt ngào.

5. Thêm bản ghi được phân loại bởi người nói: chạy tính năng quay số của người nói pyannote trên âm thanh và nhúng bản ghi cho mỗi người nói. Thể hiện các câu hỏi "Alice đã nói gì về X?".

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| Phân đoạn cảnh | "Phát hiện bắn" | Cắt video thành các cảnh ở ranh giới cảnh quay |
| Chỉ số đa vector | "Chú thích + khung + bản ghi" | Bộ sưu tập Qdrant với vectors được đặt tên cho mỗi biểu diễn |
| grounding thời gian | "Chính xác thì nó xảy ra khi nào" | Tinh chỉnh cửa sổ (bắt đầu, kết thúc) cho câu trả lời truy vấn |
| Khung embedding | "Biểu diễn trực quan" | Một vector embedding của khung hình chính; được sử dụng để tương tự cảnh và hình ảnh |
| Nhiệt hạch RRF | "Hợp nhất thứ hạng đối ứng" | Merge chiến lược trên nhiều danh sách được xếp hạng; Một thủ thuật truy xuất lai cổ điển |
| Đếm ảo giác | "Đếm sai" | Chế độ thất bại đã biết của VLMs đối với câu hỏi "có bao nhiêu X" |
| Mạng hoạt động-QA | "Video-QA benchmark" | Video dài QA accuracy benchmark |

## Đọc thêm

- [AI2 Molmo 2](https://allenai.org/blog/molmo2) — mở VLM checkpoints
- [TimeLens (CVPR 2026)](https://github.com/TencentARC/TimeLens) — grounding thời gian trên quy mô lớn
- [Gemini Video long-context](https://deepmind.google/technologies/gemini) — tham chiếu được lưu trữ
- [VideoDB](https://videodb.io) — Tham chiếu API CRUD cho video
- [Twelve Labs Marengo + Pegasus](https://www.twelvelabs.io) — tham khảo thương mại
- [TransNetV2](https://github.com/soCzech/TransNetV2) - phân đoạn cảnh model
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) - Giải pháp thay thế mở cổ điển
- [ActivityNet-QA](https://arxiv.org/abs/1906.02467) — tham khảo đánh giá benchmark
