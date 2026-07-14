# Tranh luận và cộng tác đa Agent

> Du et al. (ICML 2024, "Society of Minds") chạy N model trường hợp đề xuất câu trả lời một cách độc lập, sau đó lặp đi lặp lại phê bình lẫn nhau qua các vòng R để hội tụ. Cải thiện tính thực tế, tuân thủ quy tắc, lập luận. Cấu trúc liên kết thưa thớt đánh bại lưới đầy đủ về chi phí token.

**Loại:** Tìm hiểu + Xây dựng
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 14 · 12 (Mẫu quy trình làm việc), Giai đoạn 14 · 05 (Tự tinh chỉnh và PHÊ BÌNH)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Giải thích giao thức tranh luận: N người đề xuất, vòng R, hội tụ về một câu trả lời chung.
- Mô tả lý do tại sao tranh luận cải thiện tính thực tế, tuân thủ quy tắc và lý luận.
- Giải thích cấu trúc liên kết thưa thớt: không phải mọi người tranh luận đều cần nhìn thấy nhau.
- Thực hiện một cuộc tranh luận stdlib về một LLM có kịch bản với các biến thể toàn lưới và thưa thớt; Đo lường chi phí token so với accuracy.

## Vấn đề

Tự tinh chỉnh (Bài 05) là một trong những model phê bình bản thân - rủi ro tư duy nhóm. CRITIC (Bài 05) đặt nền tảng phê bình trong các công cụ bên ngoài - không phải lúc nào cũng có sẵn. Tranh luận giới thiệu một phương thức thứ ba: nhiều trường hợp, phê bình chéo, hội tụ bởi bất đồng.

## Khái niệm

### Hiệp hội Tâm trí (Du và cộng sự, ICML 2024)

- N model trường hợp độc lập đề xuất câu trả lời cho cùng một câu hỏi.
- Qua các vòng R, mỗi model đọc đề xuất của người khác và phê bình chúng.
- Models cập nhật câu trả lời của họ dựa trên những lời phê bình.
- Sau các vòng R, trả về câu trả lời hội tụ.

Các thí nghiệm ban đầu sử dụng N = 3, R = 2 do chi phí. Accuracy cải thiện với nhiều agents hơn và nhiều vòng hơn về các vấn đề khó (MMLU, GSM8K, Hiệu lực nước cờ vua, tạo tiểu sử).

Sự kết hợp giữa model đánh bại các cuộc tranh luận model đơn: ChatGPT + Bard cùng nhau > một mình.

### Cấu trúc liên kết thưa thớt

"Cải thiện cuộc tranh luận đa Agent với cấu trúc liên kết giao tiếp thưa thớt" (arXiv: 2406.11776, 2024-2025) cho thấy cuộc tranh luận toàn lưới không phải lúc nào cũng tối ưu. Cấu trúc liên kết thưa thớt (ngôi sao, vòng, trung tâm và nan hoa) có thể phù hợp với accuracy với chi phí token thấp hơn. Mỗi người tranh luận chỉ nhìn thấy một tập hợp con của các đồng nghiệp.

Ý nghĩa:

- Lưới đầy đủ N = 5, R = 3 = 5 × 3 = 15 đề xuất, mỗi đề xuất đọc 4 đồng nghiệp = 60 phản biện.
- Ngôi sao N=5, R=3 (một trung tâm + 4 nan hoa) = 15 đề xuất, nan hoa chỉ đọc trung tâm = 12 phản biện.

### Khi tranh luận giúp ích

- **Tính thực tế.** N đề xuất độc lập, kiểm tra chéo làm giảm ảo giác.
- **Tuân theo quy tắc.** Tính hợp lệ của nước đi cờ vua - một model bỏ lỡ một quy tắc, những người khác bắt được nó.
- **Lý luận mở.** Nhiều khung thu hẹp câu trả lời đúng.

### Khi tranh luận đau đớn

- **UX nhạy cảm với độ trễ.** Các vòng nối tiếp N × R là độ trễ mà bạn có thể không có.
- **Thang đo nhạy cảm với chi phí.** N × R tokens cho mỗi câu hỏi.
- **Tra cứu thực tế đơn giản.** Một tra cứu rẻ hơn năm cuộc tranh luận.

### Phiên bản thực tế năm 2026

- **Anthropic orchestrator-workers** (Bài 12) — một biến thể của cuộc tranh luận với một bước tổng hợp.
- **Giám sát LangGraph** (Bài 13) — bộ định tuyến trung tâm + agents chuyên gia có thể thực hiện tranh luận như một nút.
- **OpenAI Agents SDK** (Bài 16) — agents bàn giao qua lại để phê bình lặp đi lặp lại.
- **Đánh giá nhiều agent** — tranh luận cặp + optimizer đánh giá cho tín hiệu đánh giá.

### Mô hình này sai ở đâu

- **Sự sụp đổ hội tụ.** Tất cả agents hội tụ về câu trả lời sai đầu tiên. Giảm thiểu với các vòng bất đồng bắt buộc.
- **Lỗi trung tâm.** Trong cấu trúc liên kết sao, một trung tâm xấu sẽ làm hỏng tất cả mọi người. Xoay hoặc sử dụng nhiều trung tâm.
- **Prompt đồng nhất.** Tất cả các agents đều sử dụng cùng một prompt; họ đưa ra những câu trả lời giống nhau. Sử dụng prompts and/or models đa dạng.

## Tự xây dựng

`code/main.py` triển khai tranh luận stdlib:

- `Debater` class (LLM theo kịch bản với sự trôi dạt ý kiến của mỗi người tranh luận).
- `FullMeshDebate` và `SparseDebate` người chạy.
- Ba câu hỏi: một thực tế, một dựa trên quy tắc, một lý luận.
- Số liệu: câu trả lời hội tụ, vòng đến hội tụ, tổng số hoạt động phê bình.

Chạy nó:

```
python3 code/main.py
```

Đầu ra: accuracy và chi phí cho mỗi giao thức; thưa thớt phù hợp với toàn bộ 2/3 câu hỏi với chi phí thấp hơn.

## Ứng dụng

- **Anthropic người điều phối workers **cho các cuộc tranh luận 2-3 worker đơn giản.
- **LangGraph** cho cuộc tranh luận nhiều vòng có trạng thái với điểm kiểm tra.
- **Tùy chỉnh** để nghiên cứu hoặc đảm bảo tính đúng đắn chuyên biệt.

## Sản phẩm bàn giao

`outputs/skill-debate.md` giàn giáo tạo ra một cuộc tranh luận nhiều agent với cấu trúc liên kết có thể cấu hình, N, R và quy tắc hội tụ.

## Bài tập

1. Thực hiện quy tắc "bất đồng bắt buộc": trong vòng 1, mỗi người tranh luận phải đưa ra một đề xuất riêng biệt. Đo ảnh hưởng đến tốc độ hội tụ.
2. Thêm một tổng hợp có trọng số độ tin cậy: người tranh luận trở lại (câu trả lời, sự tự tin); trọng lượng tổng hợp bằng sự tự tin. Nó có giúp ích không?
3. Hoán đổi một "agent" cho một LLM có kịch bản khác với các ý kiến khác nhau. Tính không đồng nhất có cải thiện accuracy không?
4. Đo lường chi phí token cho lưới đầy đủ so với thưa thớt trên 3 câu hỏi của bạn. Chi phí cốt truyện so với accuracy.
5. Đọc bài báo của Society of Minds. Chuyển đồ chơi của bạn sang N = 5, R = 3. Điều gì phá vỡ? Điều gì trở nên tốt hơn?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Tranh luận | "Phê bình đa agent" | N người đề xuất, R vòng phê bình chéo, hội tụ |
| Lưới đầy đủ | "Mọi người đều đọc tất cả mọi người" | Mọi người tranh luận đều đọc mọi đồng nghiệp mỗi vòng |
| Cấu trúc liên kết thưa thớt | "Chế độ xem ngang hàng hạn chế" | Những người tranh luận chỉ đọc một tập hợp con của các đồng nghiệp |
| Trung tâm và nan hoa | "Cấu trúc liên kết sao" | Một người tranh luận trung tâm, nan hoa N-1 chỉ đọc trung tâm |
| Hội tụ | "Thỏa thuận" | Những người tranh luận hội tụ về một câu trả lời chung |
| Xã hội tâm trí | "Du et al. bài tranh luận" | Phương pháp tranh luận đa agent ICML 2024 |

## Đọc thêm

- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) — cuộc tranh luận đa agent kinh điển
- [Sparse Communication Topology (arXiv:2406.11776)](https://arxiv.org/abs/2406.11776) — kết quả cấu trúc liên kết thưa thớt
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — orchestrator-workers như một biến thể tranh luận
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — đối tác tự phê bình một model
