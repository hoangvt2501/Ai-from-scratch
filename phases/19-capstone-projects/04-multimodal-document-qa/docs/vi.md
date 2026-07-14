# Capstone 04 - QA tài liệu đa phương thức (PDF, Bảng, Biểu đồ)

> Biên giới tài liệu-QA năm 2026 đã chuyển từ OCR sau đó là văn bản và hướng tới tương tác muộn ưu tiên tầm nhìn. ColPali, ColQwen2.5 và ColQwen3-omni coi mỗi trang PDF là một hình ảnh, nhúng nó với nhiều vector tương tác muộn và để truy vấn tham gia trực tiếp vào các bản vá. Trên 10-K tài chính, các bài báo khoa học và ghi chú viết tay, mô hình này đánh bại OCR đầu tiên với một tỷ lệ lớn. Xây dựng pipeline từ đầu đến cuối trên 10 nghìn trang và xuất bản cạnh nhau với OCR sau đó là văn bản.

**Loại:** Đá đỉnh
**Ngôn ngữ:** Python (pipeline), TypeScript (viewer UI)
**Kiến thức tiên quyết:** Giai đoạn 4 (thị giác máy tính), Giai đoạn 5 (NLP), Giai đoạn 7 (transformers), Giai đoạn 11 (kỹ thuật LLM), Giai đoạn 12 (đa phương thức), Giai đoạn 17 (cơ sở hạ tầng)
**Các giai đoạn thực hiện:** P4 · P5 · P7 · P11 · P12 · Trang 17
**Thời lượng:** 30 giờ

## Vấn đề

Các doanh nghiệp ngồi trên các tệp PDF OCR pipelines lộn xộn: quét 10-K với các bảng xoay, các bài báo khoa học dày đặc với các phương trình, biểu đồ chỉ có ý nghĩa dưới dạng hình ảnh, chú thích viết tay. Coi đây là văn bản đầu tiên có nghĩa là mất một nửa tín hiệu. Câu trả lời năm 2026 là truy xuất nhiều vector tương tác muộn trên hình ảnh trang thô. ColPali (Illuin Tech) đã giới thiệu nó; ColQwen2.5-v0.2 và ColQwen3-omni đã đẩy accuracy. Trên ViDoRe v3, điểm truy xuất ưu tiên tầm nhìn trên OCR sau đó là văn bản bằng lề có ý nghĩa - và khoảng cách mở rộng trên biểu đồ, bảng và chữ viết tay.

Sự đánh đổi là lưu trữ và độ trễ. Một embedding ColQwen là ~2048 bản vá vectors trên mỗi trang, không phải một vector 1024 mờ duy nhất. Bóng bay lưu trữ thô. DocPruner (2026) mang lại 50% cắt tỉa mà không có accuracy loss đo lường được. Bạn sẽ lập chỉ mục 10 nghìn trang, đo lường nDCG@5 ViDoRe v3, cung cấp câu trả lời dưới 2 giây và so sánh trực tiếp với đường cơ sở OCR sau đó là văn bản.

## Khái niệm

Tương tác muộn có nghĩa là mọi truy vấn token điểm so với mọi bản vá token và điểm tối đa cho mỗi truy vấn token được tổng hợp. Bạn có được kết hợp chi tiết mà không cần một vector gộp nào. Chỉ mục đa vector (Vespa, Qdrant multi-vector hoặc AstraDB) lưu trữ embeddings mỗi bản vá và chạy MaxSim tại thời điểm truy xuất.

Người trả lời là một model ngôn ngữ thị giác lấy truy vấn cộng với các trang được truy xuất top-k dưới dạng hình ảnh và viết câu trả lời với các vùng bằng chứng (hộp giới hạn hoặc tham chiếu trang). Qwen3-VL-30B, Gemini 2.5 Pro và InternVL3 là những lựa chọn biên giới năm 2026. Đối với phương trình và ký hiệu khoa học, dự phòng OCR (Nougat, dấu chấm. ocr) được ghép vào dưới dạng kênh văn bản tùy chọn.

Đánh giá là một ma trận hai chiều. Một trục: kiểu nội dung (đoạn văn bản thuần túy, bảng dày đặc, biểu đồ bar/line, ghi chú viết tay, phương trình). Trục khác: phương pháp truy xuất (tương tác muộn trước tầm nhìn so với OCR sau đó là văn bản so với kết hợp). Mỗi ô được nDCG@5 và trả lời accuracy. Báo cáo là sản phẩm được giao.

## Kiến trúc

```
PDFs -> page renderer (PyMuPDF, 180 DPI)
           |
           v
  ColQwen2.5-v0.2 embed (multi-vector per page, ~2048 patches)
           |
           +------> DocPruner 50% compression
           |
           v
   multi-vector index (Vespa or Qdrant multi-vector)
           |
query ----+----> retrieve top-k pages (MaxSim)
           |
           v
  VLM answerer: Qwen3-VL-30B | Gemini 2.5 Pro | InternVL3
    inputs: query + top-k page images + optional OCR text
           |
           v
  answer with cited page numbers + evidence regions
           |
           v
  Streamlit / Next.js viewer: highlighted boxes on source page
```

## Stack

- Kết xuất trang: PyMuPDF (fitz) ở 180 DPI, chuẩn hóa theo chiều dọc
- model tương tác muộn: ColQwen2.5-v0.2 hoặc ColQwen3-omni (nhóm vidore trên Hugging Face)
- Chỉ số: Vespa với trường đa vector, hoặc Qdrant đa vector hoặc AstraDB với MaxSim
- Cắt tỉa: DocPruner 2026 policy (giữ các bản vá có độ variance cao, nén 50% ở < 0.5% accuracy loss)
- OCR dự phòng (phương trình / bảng dày đặc): dấu chấm. ocr hoặc kẹo hạnh nhân
- VLM người trả lời: Qwen3-VL-30B tự lưu trữ hoặc lưu trữ Gemini 2.5 Pro; InternVL3 làm dự phòng
- Đánh giá: ViDoRe v3 benchmark, M3DocVQA cho lý luận nhiều trang
- Giao diện người xem: Next.js 15 với lớp phủ canvas cho các vùng bằng chứng

## Tự xây dựng

1. **Nhập.** Đi bộ một kho dữ liệu gồm 10 nghìn trang PDF trên 10-K, bài báo khoa học và tài liệu được quét. Hiển thị mỗi trang thành PNG 1536x2048. Kiên trì `{doc_id, page_num, image_path}`.

2. **Nhúng.** Chạy ColQwen2.5-v0.2 trên mỗi hình ảnh trang. Hình dạng đầu ra ~2048 embeddings bản vá mờ 128. Áp dụng DocPruner để giữ nửa tín hiệu cao nhất. Ghi vào trường đa vector của Vespa hoặc đa vector Qdrant.

3. **Truy vấn.** Đối với mỗi truy vấn đến, hãy nhúng với tháp truy vấn (embeddings cấp token). Chạy MaxSim dựa trên chỉ mục: đối với mỗi truy vấn token, lấy max dot-product trên bản vá trang embeddings, sum. Trả về top-k trang.

4. **Tổng hợp.** Gọi Qwen3-VL-30B với truy vấn và hình ảnh 5 trang trên cùng. Prompt: "Trả lời chỉ bằng các trang được cung cấp. Trích dẫn từng tuyên bố theo (doc_id, trang) và đặt tên khu vực (hình, bảng, đoạn)."

5. **Vùng bằng chứng.** Sau khi process câu trả lời để trích xuất các khu vực được trích dẫn. Nếu VLM phát ra các hộp giới hạn (Qwen3-VL thì có), hãy hiển thị chúng dưới dạng lớp phủ trong trình xem.

6. **OCR dự phòng.** Đối với các trang được xác định là dày đặc phương trình (heuristic trên hình ảnh variance), hãy chạy Nougat hoặc dấu chấm. ocr và chuyển văn bản OCR dưới dạng một kênh bổ sung cùng với hình ảnh.

7. **Eval.** Chạy ViDoRe v3 (truy xuất nDCG@5) và M3DocVQA (accuracy QA nhiều trang). Đồng thời chạy pipeline OCR sau đó văn bản trên cùng một kho dữ liệu với cùng một bộ tổng hợp. Tạo ma trận tiếp cận × kiểu nội dung.

8. **UI.** Nguyên mẫu Streamlit trước; Next.js trình xem 15 production với lớp phủ khu vực bằng chứng từng trang.

## Ứng dụng

```
$ doc-qa ask "what was the 2024 operating margin change for segment EMEA?"
[retrieve]   top-5 pages in 320ms (ColQwen2.5, MaxSim, Vespa)
[synth]      qwen3-vl-30b, 1.4s, cited (form-10k-2024, p. 88) + (..., p. 92)
answer:
  EMEA operating margin moved from 18.2% to 16.8%, a 140bp decline.
  cited: 10-K-2024.pdf p.88 (Table 4, Segment Operating Margin)
         10-K-2024.pdf p.92 (MD&A, Operating Performance)
[viewer]     open with highlighted bounding boxes overlaid on p.88 Table 4
```

## Sản phẩm bàn giao

`outputs/skill-doc-qa.md` mô tả sản phẩm: một hệ thống QA tài liệu đa phương thức ưu tiên tầm nhìn được điều chỉnh theo một kho dữ liệu cụ thể và được đánh giá dựa trên đường cơ sở OCR sau đó là văn bản trên ViDoRe v3.

| Trọng lượng | Tiêu chí | Cách đo lường |
| :-: | --- | --- |
| 25 | ViDoRe v3 / M3DocVQA accuracy | Số Benchmark so với đường cơ sở OCR văn bản và bảng xếp hạng đã xuất bản |
| 20 | Bằng chứng-khu vực grounding | Phần nhỏ của các khu vực được trích dẫn thực sự chứa câu trả lời span |
| 20 | Kỹ thuật lưu trữ và độ trễ | Tỷ lệ nén DocPruner, chỉ số p95, câu trả lời p95 |
| 20 | Lý luận nhiều trang | Accuracy trên một bộ nhiều trang gồm 100 câu hỏi được dán nhãn thủ công |
| 15 | UX kiểm tra nguồn | Độ rõ nét của người xem, độ trung thực của lớp phủ, các công cụ so sánh song song |
| **100** |||

## Bài tập

1. Đo lường ColQwen2.5-v0.2 so với ColQwen3-omni trên cùng một kho dữ liệu. Một trang nào đúng và trang kia bỏ lỡ? Thêm thẻ "content class" vào chỉ mục để định tuyến theo loại.

2. Cắt tỉa mạnh embeddings (75%, 90%). Tìm vách đá nén: điểm mà ViDoRe nDCG@5 giảm xuống dưới đường cơ sở OCR.

3. Xây dựng kết hợp: chạy song song OCR văn bản và ColQwen, hợp nhất với RRF, xếp hạng lại bằng cross-encoder. Con lai có đánh bại một mình không? Nó giúp ích nhiều nhất ở đâu?

4. Hoán đổi Qwen3-VL-30B lấy VLM nhỏ hơn (Qwen2.5-VL-7B). Đo đường cong accuracy trên đô la.

5. Thêm hỗ trợ ghi chú viết tay. Hiển thị kho chữ viết tay, được nhúng với ColQwen, đo lường truy xuất. So sánh với OCR pipeline chữ viết tay.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| Tương tác muộn | "Truy xuất theo phong cách ColPali" | Truy vấn tokens điểm so với các bản vá trang một cách độc lập; Tổng hợp MaxSim |
| Đa vector | "Mỗi bản vá embedding" | Mỗi tài liệu có nhiều vectors, không có một vector gộp chung |
| Tối đa Sim | "Chấm điểm tương tác muộn" | Đối với mỗi truy vấn token, hãy lấy sự tương đồng tối đa so với vectors tài liệu; Tổng |
| Trình cắt tỉa DocPruner | "Nén bản vá" | Cắt tỉa năm 2026 giữ 50% các mảng có accuracy loss không đáng kể |
| ViDoRe phiên bản 3 | "Truy xuất tài liệu benchmark" | Tiêu chuẩn năm 2026 để đo lường truy xuất tài liệu trực quan |
| Vùng bằng chứng | "Hộp giới hạn được trích dẫn" | Một hộp bbox trên trang nguồn bản địa hóa câu trả lời span |
| OCR dự phòng | "Kênh phương trình" | Văn bản pipeline sử dụng cùng với tầm nhìn cho các trang nặng phương trình hoặc bảng |

## Đọc thêm

- [ColPali (Illuin Tech) repository](https://github.com/illuin-tech/colpali) — truy xuất tài liệu tương tác muộn tham khảo
- [ColPali paper (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449) — bài báo phương pháp cơ bản
- [ColQwen family on Hugging Face](https://huggingface.co/vidore) — checkpoints sẵn sàng production
- [M3DocRAG (Adobe)](https://arxiv.org/abs/2411.04952) — đa phương thức đa trang RAG cơ sở
- [Vespa multi-vector tutorial](https://docs.vespa.ai/en/colpali.html) — stack phục vụ tham khảo
- [Qdrant multi-vector support](https://qdrant.tech/documentation/concepts/vectors/#multivectors) — chỉ mục thay thế
- [AstraDB multi-vector](https://docs.datastax.com/en/astra-db-serverless/databases/vector-search.html) — chỉ mục được quản lý thay thế
- [Nougat OCR](https://github.com/facebookresearch/nougat) — dự phòng OCR có khả năng phương trình
