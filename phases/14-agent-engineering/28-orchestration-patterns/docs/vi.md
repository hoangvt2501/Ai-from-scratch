# Orchestration mẫu: Giám sát, Swarm, Phân cấp

> Bốn mô hình orchestration lặp lại trong năm 2026 frameworks: giám sát worker, swarm / ngang hàng, phân cấp, tranh luận. Hướng dẫn của Anthropic: "Đó là về việc xây dựng hệ thống phù hợp với nhu cầu của bạn." Bắt đầu đơn giản; Chỉ thêm cấu trúc liên kết khi một agent cộng với năm mẫu quy trình làm việc là không đủ.

**Loại:** Tìm hiểu + Xây dựng
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 14 · 12 (Mẫu quy trình làm việc), Giai đoạn 14 · 25 (Tranh luận nhiều Agent)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Đặt tên cho bốn mẫu orchestration định kỳ và thời điểm mỗi mẫu phù hợp.
- Mô tả đề xuất LangChain năm 2026: giám sát dựa trên lệnh gọi công cụ so với thư viện giám sát.
- Giải thích quy tắc "xây dựng hệ thống phù hợp" của Anthropic và cách nó kiểm soát lựa chọn cấu trúc liên kết.
- Triển khai cả bốn trong stdlib dựa trên một LLM có kịch bản chung.

## Vấn đề

Các nhóm tiếp cận "đa agent" trước khi họ cần. Bốn mô hình lặp đi lặp lại trên frameworks; Khi bạn có thể đặt tên cho chúng, bạn có thể chọn đúng - hoặc bỏ qua hoàn toàn cấu trúc liên kết.

## Khái niệm

### Giám sát-worker

- Một LLM định tuyến trung tâm gửi đến các agents chuyên gia.
- Quyết định: quay trở lại bản thân, giao cho chuyên gia, chấm dứt.
- Các chuyên gia không nói chuyện với nhau; Tất cả định tuyến đều thông qua người giám sát.

Frameworks: LangGraph `create_supervisor`, Anthropic điều phối-workers, Process phân cấp CrewAI.

**Đề xuất LangChain năm 2026: **thực hiện giám sát thông qua các cuộc gọi công cụ trực tiếp thay vì `create_supervisor`. Cung cấp khả năng kiểm soát kỹ thuật ngữ cảnh tốt hơn - bạn quyết định chính xác những gì mỗi chuyên gia nhìn thấy.

### Swarm / ngang hàng

- Agents bàn giao trực tiếp thông qua bề mặt dụng cụ dùng chung.
- Không có bộ định tuyến trung tâm.
- Độ trễ thấp hơn người giám sát (ít bước nhảy hơn).
- Khó lý luận hơn (không có điểm kiểm soát duy nhất).

Frameworks: Cấu trúc liên kết LangGraph swarm, OpenAI Agents SDK bàn giao (khi tất cả agents có thể chuyển giao cho tất cả những người khác).

### Phân cấp

- Giám sát viên quản lý phụ giám sát quản lý workers.
- Được triển khai dưới dạng biểu đồ con lồng nhau trong LangGraph; phi hành đoàn lồng nhau trong CrewAI.
- Mở rộng quy mô cho dân số agent lớn với chi phí hoạt động phức tạp.

Khi bạn cần: khi ngân sách ngữ cảnh của một người giám sát không thể chứa mô tả của tất cả các chuyên gia.

### Tranh luận

- Người đề xuất song song + phê bình chéo lặp đi lặp lại (Bài 25).
- Không thực sự orchestration - xác minh nhiều hơn - nhưng hiển thị như một lựa chọn cấu trúc liên kết trong frameworks.

### Phi hành đoàn CrewAI so với Flow

CrewAI chính thức hóa hai chế độ triển khai:

- **Flow** để tự động hóa theo hướng sự kiện xác định (điểm bắt đầu được đề xuất cho production).
- **Phi hành đoàn** để cộng tác dựa trên vai trò tự chủ.

Điều này trực giao với bốn mẫu trên nhưng ánh xạ đến cấu trúc liên kết: Dòng chảy thường là giám sát hoặc phân cấp; Phi hành đoàn thường là người giám sát với một bộ định tuyến LLM.

### Hướng dẫn của Anthropic

"Thành công trong không gian LLM không phải là xây dựng hệ thống phức tạp nhất. Đó là về việc xây dựng hệ thống phù hợp với nhu cầu của bạn."

Thứ tự quyết định:

1. Các agent đơn + mẫu quy trình làm việc (Bài 12) — bắt đầu tại đây.
2. Giám sát-worker - khi bạn có 2-4 bác sĩ chuyên khoa.
3. Swarm - khi độ trễ quan trọng hơn sự rõ ràng của lý luận.
4. Phân cấp — chỉ khi ngân sách ngữ cảnh giám sát không thành công.
5. Tranh luận - khi accuracy quan trọng hơn chi phí.

### Mô hình này sai ở đâu

- **Tư duy ưu tiên cấu trúc liên kết.** "Chúng ta cần đa agent" trước khi xác định vấn đề mà đa agent giải quyết.
- **Chuyển giao nảy trong swarm.** A -> B -> A -> B. Sử dụng bộ đếm nhảy.
- **Hệ thống phân cấp giả.** Ba lớp vì "doanh nghiệp"; hai đội thực tế. Sụp đổ.

## Tự xây dựng

`code/main.py` triển khai cả bốn mẫu trong stdlib đối với một LLM có kịch bản:

- `Supervisor` - bộ định tuyến trung tâm.
- `Swarm` — ngang hàng với bàn giao trực tiếp.
- `Hierarchical` - giám sát viên của người giám sát.
- `Debate` — người đề xuất song song + phê bình.

Mỗi mẫu xử lý cùng một nhiệm vụ ba ý định (hoàn tiền / lỗi / bán hàng). Trace hình dạng khác nhau.

Chạy nó:

```
python3 code/main.py
```

Đầu ra: mỗi mẫu trace + số op. Người giám sát sạch sẽ nhất; swarm là ngắn nhất; phân cấp là sâu nhất; tranh luận là tốn kém nhất.

## Ứng dụng

- **LangGraph** dành cho người giám sát và phân cấp (biểu đồ con lồng nhau).
- **OpenAI Agents SDK** đối với bàn giao dưới dạng công cụ (hình người giám sát).
- **CrewAI Flow** cho production xác định.
- **Tùy chỉnh** để tranh luận hoặc khi bạn muốn kiểm soát chính xác.

## Sản phẩm bàn giao

`outputs/skill-orchestration-picker.md` chọn một cấu trúc liên kết và thực hiện nó.

## Bài tập

1. Chuyển đổi worker giám sát thành swarm bằng cách xóa bộ định tuyến. Điều gì phá vỡ? Điều gì cải thiện?
2. Thêm bộ đếm bước nhảy vào swarm: từ chối sau 3 lần bàn giao. Nó có bắt được A->B->A nảy không?
3. Xây dựng hệ thống phân cấp hai cấp cho miền 12 chuyên gia. Ngân sách ngữ cảnh thất bại ở đâu nếu không lồng nhau?
4. Lập hồ sơ bốn mẫu trên khối lượng công việc hình production. Cái nào thắng trên chỉ số nào (độ trễ, chi phí, accuracy, khả năng gỡ lỗi)?
5. Đọc bài đăng "Xây dựng Agents hiệu quả" của Anthropic. Ánh xạ từng luồng production của bạn đến một trong bốn luồng. Bất kỳ điều nào không lập bản đồ rõ ràng?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Giám sát-worker | "Bộ định tuyến + chuyên gia" | LLM trung ương gửi bác sĩ chuyên khoa; họ không nói chuyện với nhau |
| Swarm | "Ngang hàng" | Bàn giao trực tiếp thông qua các công cụ được chia sẻ; Không có bộ định tuyến trung tâm |
| Phân cấp | "Người giám sát của người giám sát" | Biểu đồ con lồng nhau cho quần thể lớn |
| Tranh luận | "Người đề xuất + phê bình" | Người đề xuất song song, phê bình chéo (Bài 25) |
| Giám sát dựa trên lệnh gọi công cụ | "Người giám sát không có thư viện" | Triển khai giám sát dưới dạng công cụ trực tiếp để kiểm soát ngữ cảnh |
| Phi hành đoàn | "Nhóm tự trị" | Chế độ cộng tác dựa trên vai trò của CrewAI |
| Dòng chảy | "Quy trình làm việc xác định" | Chế độ production theo hướng sự kiện của CrewAI |

## Đọc thêm

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) - năm mẫu + agent so với quy trình làm việc
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — giám sát, swarm, phân cấp
- [CrewAI docs](https://docs.crewai.com/en/introduction) - Phi hành đoàn vs Dòng chảy
- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) - mô hình tranh luận
