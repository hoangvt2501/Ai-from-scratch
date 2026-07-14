# Làm theo hướng dẫn như tín hiệu Alignment

> Mỗi lời phê bình sau này về RLHF lập luận chống lại pipeline này. Trước khi bạn nghiên cứu cách áp lực tối ưu hóa làm bóp méo proxy, bạn phải xem proxy. InstructGPT (Ouyang và cộng sự, 2022) đã xác định kiến trúc tham chiếu: fine-tuning giám sát về các cặp lệnh-phản hồi, phần thưởng model huấn luyện về xếp hạng ưu tiên theo cặp và PPO chống lại phần thưởng model với hình phạt KL đối với SFT policy. InstructGPT 1.3B được ưa chuộng hơn GPT-3. 175B Kết quả duy nhất đó là lý do mọi phòng thí nghiệm biên giới vào năm 2026 vẫn ships một phòng thí nghiệm hậu training pipeline hình RLHF.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, toy three-stage pipeline)
**Kiến thức tiên quyết:** Giai đoạn 10 · 06 (SFT), Giai đoạn 10 · 07 (RLHF), Giai đoạn 10 · 08 (DPO)
**Thời lượng:** ~45 phút

## Mục tiêu học tập

- Đặt tên cho ba giai đoạn của pipeline InstructGPT và loss được sử dụng trong mỗi giai đoạn.
- Giải thích lý do tại sao model điều chỉnh lệnh 1.3B đánh bại GPT-3 175B thô về đánh giá sở thích của con người.
- Nêu rõ hình phạt KL trong giai đoạn 3 đang bảo vệ chống lại điều gì và tại sao việc loại bỏ nó lại sụp đổ thành hành vi tìm kiếm chế độ.
- Mô tả thuế alignment và giảm thiểu PPO-ptx mà Ouyang và cộng sự đã sử dụng để chống lại nó.

## Vấn đề

Ngôn ngữ được huấn luyện trước models văn bản hoàn chỉnh. Họ không trả lời các câu hỏi. Hỏi GPT-3 "viết một hàm Python đảo ngược danh sách" và bạn thường nhận được một prompt khác, bởi vì hầu hết phân phối training là văn bản web tiếp tục với nhiều văn bản web hơn. model đang làm công việc của mình - công việc là sai.

proxy mọi phòng thí nghiệm nghiêm túc được sử dụng để khắc phục điều này là sở thích của con người. Hai lần hoàn thành sẽ được chuyển đến một người đánh giá; người đánh giá chọn cái tốt hơn; một phần thưởng model học được người đánh giá. Sau đó, vòng lặp RL chuyển policy theo hướng đầu ra phần thưởng model điểm cao. Đó là toàn bộ luận án InstructGPT trong ba câu. rest của bài báo là kỹ thuật.

## Khái niệm

### Giai đoạn 1: fine-tuning giám sát (SFT)

Thu thập các cặp phản hồi prompt trong đó phản hồi là những gì một người có ý định tốt sẽ viết. Ouyang và cộng sự đã sử dụng 13k prompts từ các máy dán nhãn và OpenAI API. Fine-tune model cơ sở trên dữ liệu này với loss entropy chéo tiêu chuẩn.

Những gì SFT mang lại cho bạn: model bây giờ trả lời các câu hỏi thay vì tiếp tục chúng. Những gì nó không cung cấp cho bạn: bất kỳ tín hiệu nào về câu trả lời mà người đánh giá thích khi bội số là hợp lý.

### Giai đoạn 2: phần thưởng model (RM)

Đối với mỗi prompt, lấy mẫu K hoàn thành từ model SFT. Một người dán nhãn xếp hạng họ. Huấn luyện một model phần thưởng chấm điểm bất kỳ cặp phản hồi prompt nào sao cho đối với các cặp mà `y_w` được ưu tiên hơn `y_l`:

```
L_RM = -log sigmoid(r(x, y_w) - r(x, y_l))
```

Đây là loss ưu tiên theo cặp Bradley-Terry. RM thường được khởi tạo từ model SFT với đầu LM được thay thế bằng đầu vô hướng.

Phần thưởng models nhỏ: 6B là đủ cho 175B InstructGPT. Chúng cũng rất mong manh - phần 5 của bài báo chủ yếu nói về các hành vi hack phần thưởng xuất hiện ở quy mô nhỏ.

### Giai đoạn 3: PPO với hình phạt KL

Xác định mục tiêu:

```
J(pi) = E_{x~D, y~pi(.|x)} [ r(x, y) ] - beta * KL(pi(.|x) || pi_SFT(.|x))
```

Tối đa hóa với PPO. Thuật ngữ KL giúp `pi` không trôi dạt xa khỏi policy SFT. Nếu không có nó, optimizer tìm thấy các ví dụ đối nghịch - các chuỗi đạt điểm cao dưới RM vì RM chưa bao giờ nhìn thấy chúng, không phải vì con người thực sự thích chúng.

Hệ số KL `beta` là RLHF hyperparameter quan trọng nhất. Quá thấp: hack phần thưởng. Quá cao: không có cải thiện so với SFT.

### Thuế alignment

Sau RLHF, model được con người ưa thích nhưng thoái lui về benchmarks tiêu chuẩn (SQuAD, HellaSwag, DROP). Ouyang và cộng sự gọi đây là thuế alignment và sửa nó bằng PPO-ptx: trộn pre-training gradients vào mục tiêu RL để model không quên cách thực hiện các nhiệm vụ xuôi dòng mà nó chưa bao giờ được thưởng.

```
J_ptx(pi) = J(pi) + gamma * E_{x~D_pretrain} [ log pi(x) ]
```

PPO-ptx trở thành tiêu chuẩn. Anthropic, DeepMind và Meta đều sử dụng một số biến thể.

### Kết quả

InstructGPT 1.3B (SFT + RM + PPO-ptx) được các nhà dán nhãn ưa thích hơn cơ sở 175B GPT-3 khoảng 70% thời gian. Khoảng cách mở rộng về prompts kiểm tra ẩn từ lưu lượng truy cập production. Hai điều cần đọc từ con số này:

1. Alignment là một trục khác với khả năng. 175B model có nhiều khả năng hơn; 1,3 tỷ model có nhiều alignment hơn; những người dán nhãn thích cái được căn chỉnh.
2. Sàn khả năng được thiết lập bởi model cơ sở. Bạn không thể RLHF một model cơ sở để biết những sự thật mà nó chưa bao giờ thấy.

### Tại sao đây là điểm tham chiếu cho Giai đoạn 18

Mọi lời phê bình trong các bài học sau - hack phần thưởng (Bài 2), DPO (Bài 3), sycophancy (Bài 4), CAI (Bài 5), sleeper agents (Bài 7), alignment giả mạo (Bài 9) - đều phản đối một số phần của pipeline này. Phần thưởng cho các cuộc tấn công hack giai đoạn 2. DPO sụp đổ giai đoạn 2 và 3. CAI thay thế trình dán nhãn của con người. Sycophancy cho thấy trình dán nhãn là một tín hiệu thiên vị. Alignment giả mạo cho thấy policy có thể đi vòng quanh giai đoạn 3 hoàn toàn. Bạn không thể theo dõi bất kỳ lời phê bình nào trong số này mà không có pipeline trong đầu trước.

## Ứng dụng

`code/main.py` mô phỏng ba giai đoạn trên dữ liệu sở thích đồ chơi. Cơ sở "policy" là một đồng xu thiên vị hơn các hành động {A, B, C}. SFT giai đoạn 1 bắt chước các hành động của trình gắn nhãn trên 200 prompts. Giai đoạn 2 phù hợp với phần thưởng Bradley-Terry model từ 500 bảng xếp hạng theo cặp. Giai đoạn 3 chạy một bản cập nhật PPO đơn giản với hình phạt KL đối với SFT policy. Bạn có thể xem phần thưởng tăng lên, sự phân kỳ KL tăng lên và sự trôi dạt của policy - và bạn có thể tắt thuật ngữ KL để xem hack phần thưởng xuất hiện trong 50 bước cập nhật.

Những gì cần xem:

- Quỹ đạo phần thưởng với `beta = 0.1` vs `beta = 0.0`.
- KL(pi || pi_SFT) qua training bước.
- Phân phối hành động cuối cùng so với sở thích của người dán nhãn.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-instructgpt-explainer.md`. Với một mô tả RLHF pipeline hoặc một bản tóm tắt bài báo, nó xác định giai đoạn nào trong ba giai đoạn đang được sửa đổi, loss nào đang được sử dụng ở mỗi giai đoạn và liệu có hình phạt KL hoặc bộ chính quy tương đương hay không.

## Bài tập

1. Chạy `code/main.py`. Đặt `beta = 0.0` và báo cáo phân phối hành động sau 200 bước PPO. Giải thích hành vi tìm kiếm chế độ trong một đoạn văn.

2. Sửa đổi model phần thưởng để có +0,5 bias cho hành động B (lỗi phần thưởng mô phỏng). Chạy PPO với `beta = 0.1`. Hình phạt KL có ngăn cản policy khai thác bias không? Sự bóc lột trở nên rõ ràng ở `beta` nào?

3. Đọc Ouyang et al. (arXiv: 2203.02155) Hình 1. Tái tạo đường cong sở thích của người dán nhãn bằng cách chạy PPO trong 1, 5, 20, 100 bước và đo tùy chọn so với model SFT.

4. Phần 4.3 của bài báo báo cáo 1.3B InstructGPT đánh bại 175B GPT-3 khoảng 70% thời gian. Tại sao tỷ lệ này lại cao hơn trên production prompts ẩn so với prompts của chính người dán nhãn?

5. Thay thế PPO loss bằng DPO (Giai đoạn 10 · 08) trên cùng một dữ liệu ưu tiên. So sánh policy drift cuối cùng (KL với SFT) và phần thưởng cuối cùng. Phương pháp nào trôi dạt xa hơn với phần thưởng phù hợp?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| Mạng SFT | "Điều chỉnh hướng dẫn" | Giai đoạn 1: fine-tune entropy chéo trên các cặp phản ứng prompt |
| Phần thưởng model | "RM" | Hồi quy vô hướng trên (prompt, phản hồi) được huấn luyện với Bradley-Terry trên nhãn theo cặp |
| Bradley-Terry | "loss ưu tiên theo cặp" | -log sigmoid (r_w - r_l); Giảm xếp hạng theo cặp thành phân loại nhị phân |
| Hình phạt KL | "Người chính quy" | 'beta * KL(pi \ | \ | pi_SFT)' — giữ RL policy gần mỏ neo SFT |
| PPO-ptx | "PPO với pretraining mix" | Thêm một phần likelihood nhật ký trước khi training vào mục tiêu PPO để bù đắp thuế alignment |
| Thuế Alignment | "Hồi quy RLHF" | Sự sụt giảm sau RLHF so với benchmarks tiêu chuẩn mà RLHF không nhắm mục tiêu |
| Tùy chọn người dán nhãn | "ground truth" | Mẫu xếp hạng con người; RM là một proxy thống kê cho điều này, không phải cho "giá trị con người" |

## Đọc thêm

- [Ouyang et al. — Training language models to follow instructions with human feedback (arXiv:2203.02155)](https://arxiv.org/abs/2203.02155) — bài báo InstructGPT, nền tảng cho mọi RLHF pipeline sau đó
- [Stiennon et al. — Learning to summarize from human feedback (arXiv:2009.01325)](https://arxiv.org/abs/2009.01325) — người tiền nhiệm RLHF để tóm tắt
- [Christiano et al. — Deep reinforcement learning from human preferences (arXiv:1706.03741)](https://arxiv.org/abs/1706.03741) — công thức RL dựa trên sở thích ban đầu
- [Bai et al. — Training a Helpful and Harmless Assistant with RLHF (arXiv:2204.05862)](https://arxiv.org/abs/2204.05862) — Phần mở rộng HH của Anthropic của pipeline InstructGPT
