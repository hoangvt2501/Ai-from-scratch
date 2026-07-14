# Chế độ lỗi: Tại sao Agents Break

> MASFT (Berkeley, 2025) lập danh mục 14 chế độ lỗi nhiều agent trong 3 danh mục. Phân loại của Microsoft ghi lại cách các lỗi AI hiện có khuếch đại trong cài đặt agentic. Dữ liệu lĩnh vực công nghiệp hội tụ trên năm chế độ định kỳ: hành động ảo giác, phạm vi leo tầng, lỗi xếp tầng, loss ngữ cảnh, lạm dụng công cụ.

**Loại:** Tìm hiểu + Xây dựng
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 14 · 05 (Tự tinh chỉnh và PHÊ BÌNH), Giai đoạn 14 · 24 (Observability)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Kể tên ba loại lỗi của MASFT và ít nhất bốn chế độ cụ thể trong mỗi loại.
- Giải thích lý do tại sao agentic thất bại khuếch đại các chế độ lỗi AI hiện có (bias, ảo giác).
- Mô tả năm chế độ định kỳ trong ngành và các biện pháp giảm thiểu của chúng.
- Triển khai trình phát hiện stdlib gắn thẻ agent traces với nhãn chế độ lỗi.

## Vấn đề

Các nhóm ship agents điều đó làm việc trên 90% traces. 10% thất bại không phải là nhiễu ngẫu nhiên - chúng rơi vào một số lượng nhỏ các loại định kỳ. Khi bạn có thể đặt tên cho chúng, bạn có thể theo dõi và sửa chúng.

## Khái niệm

### MASFT (Berkeley, arXiv: 2503.13657)

Phân loại lỗi hệ thống đa Agent. 14 chế độ lỗi được chia thành 3 loại. Kappa 0.88 của Cohen - các danh mục có thể phân biệt được một cách đáng tin cậy.

Tuyên bố trung tâm: lỗi là lỗi thiết kế cơ bản trong các hệ thống đa agent, không LLM hạn chế cần sửa chữa bằng models cơ sở tốt hơn.

### Phân loại chế độ lỗi của Microsoft trong hệ thống Agentic AI

- Các lỗi AI hiện có (bias, ảo giác, rò rỉ dữ liệu) khuếch đại trong các cài đặt agentic.
- Những thất bại mới xuất hiện từ quyền tự chủ: hành động ngoài ý muốn trên quy mô lớn, lạm dụng công cụ, trôi dạt nhiệm vụ.
- Sách trắng là registry rủi ro cho các sản phẩm agentic.

### Mô tả đặc điểm lỗi trong Agentic AI (arXiv:2603.06847)

- Thất bại phát sinh từ orchestration, sự tiến hóa của trạng thái bên trong và tương tác với môi trường.
- Không chỉ là "mã xấu" hay "đầu ra model xấu".

### LLM Agent Khảo sát ảo giác (arXiv: 2509.18970)

Hai biểu hiện chính:

1. **Độ lệch theo hướng dẫn **- agent không tuân theo system prompt.
2. **Lạm dụng ngữ cảnh tầm xa** — agent quên hoặc áp dụng sai ngữ cảnh từ các lượt trước.

Lỗi chủ ý phụ: Bỏ sót (bỏ lỡ bước), Dự phòng (bước lặp lại), Rối loạn (các bước không theo thứ tự).

### Năm chế độ định kỳ trong ngành

Các phân tích thực địa của Arize, Galileo, NimbleBrain 2024-2026 hội tụ vào:

1. **Hành động ảo giác.** Agent gọi một công cụ không tồn tại hoặc bịa đặt các lập luận.
2. **Phạm vi leo thang.** Agent mở rộng nhiệm vụ ngoài yêu cầu của người dùng (tạo thêm PR, gửi thêm email).
3. **Lỗi xếp tầng.** Một cuộc gọi sai triggers hiệu ứng xuôi dòng. Ảo giác SKU ảo triggers bốn cuộc gọi API - một sự cố đa hệ thống.
4. **Bối cảnh loss.** Các nhiệm vụ có đường chân trời dài quên đi các ràng buộc về lượt sớm.
5. **Lạm dụng công cụ.** Gọi đúng công cụ với các đối số sai hoặc công cụ sai hoàn toàn.

Xếp tầng là kẻ giết người. Agents không thể phân biệt "Tôi đã thất bại" với "nhiệm vụ là không thể" và thường ảo giác thông báo thành công trên 400 lỗi để đóng vòng lặp.

### Giảm thiểu: cổng ở mọi bước

Cổng xác minh tự động ở mọi bước của chuỗi suy luận, kiểm tra grounding thực tế so với trạng thái môi trường. Cụ thể:

- Bộ phân loại an toàn mỗi bước (Bài 21).
- Xác thực đối số gọi công cụ (Bài 06).
- Kiểm tra chéo nội dung được truy xuất so với các sự kiện đã biết (Bài 05, CRITIC).
- Phát hiện ảo giác thành công bằng cách thăm dò lại trạng thái (tệp có thực sự được tạo không?).

### Giám sát lỗi xảy ra ở đâu

- **Gắn thẻ chỉ gặp sự cố.** Hầu hết các lỗi agent đều tạo ra kết quả có vẻ hợp lệ. Cần kiểm tra cấp độ nội dung.
- **Không có đường cơ sở.** Phát hiện trôi dạt cần một điều tốt cuối cùng; Nếu không có nó, bạn không thể nói "điều này đang trở nên tồi tệ hơn".
- **Cảnh báo quá mức.** Mỗi lỗi tạo ra một trang. Cụm và giới hạn tốc độ.

## Tự xây dựng

`code/main.py` triển khai trình gắn thẻ chế độ lỗi stdlib:

- Một trace dataset tổng hợp bao gồm năm chế độ.
- Các chức năng của máy dò trên mỗi chế độ (các mẫu chữ ký trên các lệnh gọi công cụ, đầu ra, hành động lặp lại).
- Trình gắn thẻ gắn nhãn từng trace và báo cáo phân phối chế độ.

Chạy nó:

```
python3 code/main.py
```

Đầu ra: nhãn trên mỗi trace + phân phối tổng hợp, tái tạo rẻ tiền của những gì Phoenix's trace các bề mặt phân cụm.

## Ứng dụng

- **Phượng hoàng** để phân cụm trôi production (Bài 24).
- **Langfuse** để phát lại session +chú thích.
- **Tùy chỉnh** đối với chữ ký dành riêng cho miền mà nền tảng observability của bạn không thể phát hiện.

## Sản phẩm bàn giao

`outputs/skill-failure-detector.md` tạo ra các trình phát hiện chế độ lỗi phù hợp với miền của bạn, được kết nối với cửa hàng trace.

## Bài tập

1. Thêm một trình phát hiện cho "ảo giác thành công": agent trả về thành công nhưng trạng thái mục tiêu không thay đổi.
2. Gắn thẻ 100 traces thực từ sản phẩm bạn đã xây dựng. Chế độ nào chiếm ưu thế? Chi phí sửa chữa nó là bao nhiêu?
3. Triển khai chỉ số "bán kính xếp tầng": với sự cố ở bước N, nó ảnh hưởng đến bao nhiêu bước xuôi dòng?
4. Đọc 14 chế độ lỗi của MASFT. Chọn ba áp dụng cho sản phẩm của bạn. Máy dò ghi.
5. Nối một máy dò vào một công việc CI: xây dựng không thành công nếu > = 5% traces tag một chế độ.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| KHỐI LƯỢNG | "Phân loại thất bại đa agent" | Phân loại 14 chế độ Berkeley |
| Lỗi xếp tầng | "Thất bại gợn sóng" | Một sai lầm ban đầu lan truyền qua N bước |
| Bối cảnh loss | "Quên đi ràng buộc" | Ngã rẽ dài làm giảm sự thật về ngã rẽ sớm |
| Lạm dụng công cụ | "Công cụ sai / đối số sai" | Cuộc gọi hợp lệ, gọi sai |
| Ảo giác thành công | "Hoàn thành giả" | Agent tuyên bố thành công trên 400; trạng thái không thay đổi |
| Phạm vi leo | "Tiếp cận quá mức" | Agent làm được nhiều hơn những gì được yêu cầu |
| Độ lệch theo hướng dẫn | "Bất tuân" | Bỏ qua system prompt hoặc ràng buộc người dùng |
| Lỗi ý định phụ | "Lập kế hoạch lỗi" | Thiếu sót, dư thừa, mất trật tự trong thực hiện kế hoạch |

## Đọc thêm

- [Cemri et al., MASFT (arXiv:2503.13657)](https://arxiv.org/abs/2503.13657) - 14 chế độ lỗi, 3 danh mục
- [Microsoft, Taxonomy of Failure Mode in Agentic AI Systems](https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/microsoft/final/en-us/microsoft-brand/documents/Taxonomy-of-Failure-Mode-in-Agentic-AI-Systems-Whitepaper.pdf) - Đăng ký rủi ro
- [Arize Phoenix](https://docs.arize.com/phoenix) - phân cụm trôi dạt trong thực tế
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — khi các mẫu đơn giản hơn tránh hoàn toàn các chế độ
