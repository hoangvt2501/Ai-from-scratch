# Từ CLIP đến BLIP-2 — Q-Former như Cầu nối Phương thức

> CLIP căn chỉnh hình ảnh và văn bản nhưng không thể tạo chú thích, trả lời câu hỏi hoặc tổ chức cuộc trò chuyện. BLIP-2 (Salesforce, 2023) đã giải quyết vấn đề đó bằng một cầu nối nhỏ có thể huấn luyện: 32 truy vấn có thể học được vectors tham dự qua features của ViT bị đóng băng qua cross-attention, sau đó cắm trực tiếp vào luồng đầu vào của LLM bị đóng băng. Cầu nối 188M parameters kết nối LLM 11B với ViT-g/14. Mọi VLM dựa trên bộ điều hợp cho đến năm 2026 - MiniGPT-4, InstructBLIP, anh em họ của LLaVA - đều là hậu duệ. Bài học này đọc kiến trúc của Q-Former, giải thích training hai giai đoạn của nó và xây dựng một phiên bản đồ chơi cung cấp tokens trực quan vào một decoder văn bản đóng băng.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib, cross-attention + learnable-query demo)
**Kiến thức tiên quyết:** Giai đoạn 12 · 02 (CLIP), Giai đoạn 7 (Transformers)
**Thời lượng:** ~180 phút

## Mục tiêu học tập

- Giải thích lý do tại sao nút thắt cổ chai có thể huấn luyện giữa encoder thị lực đóng băng và LLM đóng băng đánh bại việc tinh chỉnh từ đầu đến cuối về chi phí và độ ổn định.
- Triển khai một khối cross-attention trong đó một tập hợp cố định các truy vấn có thể học được tham gia vào features hình ảnh bên ngoài.
- Đi qua pretraining hai giai đoạn của BLIP-2: đại diện (ITC + ITM + ITG) sau đó tạo ra (LM loss với decoder đông lạnh).
- So sánh Q-Former với máy chiếu MLP đơn giản hơn được sử dụng trong LLaVA và tranh luận khi mỗi lựa chọn thắng.

## Vấn đề

Bạn có một ViT đóng băng tạo ra 256 bản vá tokens mờ 1408 cho mỗi hình ảnh. Bạn có một LLM 7B bị đóng băng mong đợi token embeddings của 4096 mờ. Cầu nối rõ ràng - một lớp tuyến tính từ 1408 đến 4096 - hoạt động, nhưng việc đưa tất cả 256 bản vá tokens vào ngữ cảnh của LLM tốn thêm 256 tokens cho mỗi hình ảnh. Hơn một batch gồm 32 hình ảnh là 8192 tokens chỉ được tiêu thụ bởi phương thức trực quan.

Câu hỏi BLIP-2: bạn có thể nén hình ảnh 256 token thành ít tokens hơn nhiều (ví dụ 32) trong khi vẫn giữ đủ thông tin cho LLM chú thích, trả lời câu hỏi và lý do về hình ảnh không? Và bạn có thể huấn luyện cây cầu này mà không chạm vào xương sống đóng băng, giữ chi phí training chỉ ở mức parameters của cây cầu không?

Câu trả lời: một Q-Former. 32 "truy vấn" có thể học được vectors tham gia chéo vào tokens bản vá của ViT, tạo ra một bản tóm tắt trực quan dài 32 token mà LLM sử dụng. Tổng cộng 188 triệu parameters. Được huấn luyện với các mục tiêu tương phản, phù hợp và tổng quát trước khi chạm vào LLM.

## Khái niệm

### Truy vấn có thể học được

Thủ thuật cốt lõi của Q-Former: thay vì để văn bản của LLM tokens tham gia vào các bản vá hình ảnh, hãy giới thiệu một tập hợp mới gồm 32 vectors `Q` truy vấn có thể học được và để * họ * tham gia vào các bản vá hình ảnh. Các truy vấn là parameters của model - chúng được học trong quá trình training và 32 truy vấn giống nhau được sử dụng cho mọi hình ảnh.

Sau cross-attention, mỗi truy vấn chứa một bản tóm tắt nén của hình ảnh - "mô tả đối tượng chính", "mô tả nền", "đếm các đối tượng", v.v. Các truy vấn không chuyên về nhãn ngữ nghĩa theo nghĩa đen; chúng học bất kỳ mã hóa nào làm giảm tổn thất xuôi dòng.

### Kiến trúc

Q-Former là một transformer nhỏ (12 lớp, ~100M tham số) với hai đường dẫn:

1. Đường dẫn truy vấn: 32 truy vấn vectors luồng qua self-attention (giữa chúng), sau đó cross-attention qua tokens bản vá của ViT bị đóng băng, sau đó là FFN.
2. Đường dẫn văn bản: một encoder văn bản giống BERT chia sẻ trọng số self-attention và FFN với đường dẫn truy vấn. Cross-attention bị tắt cho đường dẫn văn bản.

Tại thời điểm training, cả hai đường dẫn đều chạy. Các truy vấn và văn bản tương tác thông qua self-attention được chia sẻ, có nghĩa là các truy vấn có thể điều kiện dựa trên văn bản cho các tác vụ cần nó (ITM, ITG). Tại inference thời điểm chuyển giao VLM, chỉ có các truy vấn đi qua, mang lại 32 tokens trực quan.

### Hai giai đoạn training

BLIP-2 huấn luyện trước trong hai giai đoạn:

Giai đoạn 1: học đại diện (không LLM). Ba tổn thất:
- ITC (tương phản văn bản hình ảnh): Tương phản kiểu CLIP giữa tokens truy vấn gộp và token CLS văn bản.
- ITM (đối sánh hình ảnh-văn bản): bộ phân loại nhị phân - cặp hình ảnh-văn bản này có khớp không? Khai thác cứng-âm.
- ITG (tạo văn bản dựa trên hình ảnh): LM nhân quả trên văn bản, có điều kiện trên các truy vấn. Buộc các truy vấn mã hóa nội dung có thể tạo văn bản.

Chỉ có tàu Q-Former tàu. ViT bị đóng băng. Không có LLM nào liên quan.

Giai đoạn 2: học tập tổng quát. Đính kèm một LLM đóng băng (OPT-2.7B hoặc Flan-T5-XL, v.v.). Chiếu 32 đầu ra truy vấn vào embedding mờ của LLM thông qua một lớp tuyến tính nhỏ. Thêm chúng vào prompt văn bản. Chỉ huấn luyện phép chiếu tuyến tính và Q-Former trên LM loss trên chuỗi prompt + hình ảnh + chú thích được nối nhau.

Sau stage 2, phép chiếu Q-Former + là bộ chuyển đổi trực quan đầy đủ. Tại inference: hình ảnh → ViT → Q-Former → proj tuyến tính → được thêm vào trước văn bản → đóng băng LLM phát ra đầu ra.

### Kinh tế Parameter

BLIP-2 với ViT-g/14 (1.1B, đông lạnh) + OPT-6.7B (6.7B, đông lạnh) + Q-Former (188 triệu, huấn luyện) = tổng cộng 8B, 188 triệu được huấn luyện. Chỉ riêng Q-Former là ~2.4% toàn bộ parameters của stack. Chi phí Training phản ánh điều này: ngày trên một số ít A100 so với tuần cho đầu đến cuối.

Chất lượng: BLIP-2 phù hợp hoặc đánh bại Flamingo-80B trên zero-shot VQA trong khi nhỏ hơn 50 lần. Cây cầu hoạt động.

### InstructBLIP và Q-Former nhận biết hướng dẫn

InstructBLIP (2023) mở rộng Q-Former với một đầu vào bổ sung: chính văn bản hướng dẫn. Tại thời điểm cross-attention, các truy vấn hiện có quyền truy cập vào cả bản vá hình ảnh và hướng dẫn. Các truy vấn có thể chuyên biệt hóa theo lệnh ("đếm xe", "mô tả tâm trạng") thay vì học một bản tóm tắt cố định duy nhất. Benchmark đạt được lợi ích từ các nhiệm vụ bị trì hoãn.

### MiniGPT-4 và cách tiếp cận chỉ dành cho máy chiếu

MiniGPT-4 giữ lại Q-Former nhưng chỉ huấn luyện phép chiếu tuyến tính đầu ra trong khi đóng băng mọi thứ khác. Rẻ, nhưng chi phí là chất lượng — các truy vấn là của BLIP-2, không phải của bạn. Tốt cho việc lặp lại nhanh chóng, không phải kiến trúc tốt nhất.

### Tại sao LLaVA trở nên đơn giản hơn

LLaVA (2023, Bài học 12.05) đã thay thế Q-Former bằng MLP 2 lớp đơn giản chiếu mọi bản vá ViT token vào không gian LLM - 576 tokens cho mỗi hình ảnh cho lưới 24x24, tất cả đều được đưa vào LLM. Nén kém hơn nhưng cho phép LLM tham gia qua các bản vá thô. Vào thời điểm đó, điều này còn gây tranh cãi; vào cuối năm 2023, nó chiếm ưu thế vì dữ liệu hướng dẫn trực quan (LLaVA-Instruct-150k) đã chứng minh rằng MLP có thể được huấn luyện để bảo toàn đủ tín hiệu. Sự đánh đổi: ngữ cảnh của LLaVA lấp đầy nhanh hơn, nhưng nó mở rộng tự nhiên thành nhiều hình ảnh và video.

Đến năm 2026, sự phân chia lĩnh vực: Q-Former tồn tại khi token ngân sách quan trọng (video dài, nhiều hình ảnh); Máy chiếu MLP thống trị khi chất lượng thô trên mỗi token là ưu tiên.

### cross-attention cổng: Chim hồng hạc, tổ tiên

Flamingo (Bài 12.04) có trước BLIP-2 và sử dụng cùng một ý tưởng cross-attention nhưng ở mọi lớp LLM bị đóng băng, không phải là một cầu nối duy nhất. BLIP-2 cho thấy bạn chỉ có thể nén vào lớp đầu vào và vẫn hoạt động. Gemini và Idefics kết hợp cả hai: tokens đầu vào xen kẽ cộng với cross-attention có cổng tùy chọn cho few-shot trong ngữ cảnh.

### Hậu duệ năm 2026

- Q-Trước: BLIP-2, InstructBLIP, MiniGPT-4 và hầu hết các models ngôn ngữ video vì lý do ngân sách token.
- Bộ lấy mẫu lại bộ nhận thức: Biến thể của Flamingo (Bài 12.04); Họ Idefics, Đại bàng, OmniMAE.
- Máy chiếu MLP: LLaVA, LLaVA-NeXT, LLaVA-OneVision, Cambrian-1.
- Hồ bơi Attention: VILA, PaliGemma.

Cả bốn đều hợp lệ. Câu hỏi quyết định là liệu bạn có bị hạn chế về ngân sách token hay chất lượng trên mỗi token hay không.

## Ứng dụng

`code/main.py` xây dựng một cross-attention kiểu Q-Former stdlib:

1. Mô phỏng bản vá hình ảnh 256 tokens (mờ 128).
2. Khởi tạo 32 truy vấn có thể học được (mờ 128).
3. Chạy cross-attention chấm-sản phẩm tỷ lệ (Q từ truy vấn K/V từ các bản vá).
4. Chiếu đến LLM-mờ (512) thông qua một lớp tuyến tính.
5. Xuất ra 32 tokens hình ảnh sẵn sàng cho LLM.

Tất cả toán học trong Python thuần túy (các vòng lặp lồng nhau trên vectors). Đồ chơi nhưng hình dạng chính xác. Ma trận trọng lượng attention được in để bạn có thể xem mỗi truy vấn được lấy từ bản vá nào.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-modality-bridge-picker.md`. Đưa ra VLM configuration mục tiêu (số lượng encoder token tầm nhìn, ngân sách ngữ cảnh LLM, ràng buộc triển khai, mục tiêu chất lượng), nó đề xuất Q-Former vs MLP vs Perceiver resampler với một lý do ngắn gọn và ước tính parameter đếm cho mỗi cầu nối.

## Bài tập

1. Triển khai khối cross-attention trong PyTorch. Xác minh rằng với 32 truy vấn và 256 keys/values, ma trận trọng số attention là 32 x 256 và mỗi hàng tổng là 1 sau softmax.

2. Trong BLIP-2 giai đoạn 1, Q-Former chạy ba tổn thất đồng thời: ITC, ITM, ITG. Viết chữ ký chuyển tiếp cho từng mã giả. Cái nào yêu cầu văn bản encoder đường dẫn hoạt động?

3. So sánh số lượng parameter: Q-Former (12 lớp, 768 lớp) so với máy chiếu MLP 2 lớp (1408 → 4096, hai lớp). Chi phí Q-Former 188M ở quy mô training LLM nào?

4. Đọc Phần 3.2 của bài báo BLIP-2 (arXiv:2301.12597) về cách khởi tạo Q-Former Giải thích lý do tại sao khởi tạo từ cơ sở BERT (không ngẫu nhiên) tăng tốc độ hội tụ.

5. Đối với video dài 10 phút ở tốc độ 1 FPS được lấy mẫu đến 60 khung hình, hãy tính toán chi phí token trên mỗi khung hình ở mức (Q-Former → 32 tokens/frame) so với (máy chiếu MLP → 576 tokens/frame). Cái nào phù hợp với 128k-token LLM context window?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Q-Cựu | "Truy vấn transformer" | transformer nhỏ với 32 vectors truy vấn có thể học được tham gia chéo các features ViT bị đóng băng |
| Truy vấn có thể học được | "prompt mềm cho thị lực" | Một tập hợp các parameters cố định đóng vai trò là phía truy vấn của cross-attention; đã học trên mỗi model, được chia sẻ trên tất cả các đầu vào |
| Cross-attention | "Q từ đây, K/V từ đó" | Attention nơi truy vấn, khóa và giá trị đến từ các nguồn khác nhau; cách các truy vấn lấy từ các bản vá ViT |
| ITC | "Hình ảnh-văn bản tương phản" | loss kiểu CLIP được áp dụng cho các truy vấn gộp Q-Former so với CLS văn bản |
| ITM | "Khớp hình ảnh-văn bản" | Bộ phân loại nhị phân trên các cặp cứng-âm-được khai thác; buộc các truy vấn phân biệt các không khớp chi tiết |
| ITG | "Tạo văn bản dựa trên hình ảnh" | LM nhân quả loss trong đó văn bản được tạo có điều kiện dựa trên truy vấn; buộc truy vấn mã hóa nội dung có thể giải mã văn bản |
| Hai giai đoạn pretraining | "Đại diện sau đó tạo ra" | Giai đoạn 1 huấn luyện Q-Former một mình (ITC/ITM/ITG); Giai đoạn 2 gắn LLM đông lạnh và chỉ huấn luyện phép chiếu + Q-Former |
| Xương sống đông lạnh | "Đừng tinh chỉnh" | Tầm nhìn encoder và trọng lượng LLM được cố định; chỉ có tàu cầu |
| Đầu chiếu | "Tuyến tính đến LLM mờ" | Lớp tuyến tính cuối cùng ánh xạ đầu ra Q-Former với kích thước embedding của LLM |
| Bộ lấy mẫu lại bộ nhận thức | "Phiên bản của Flamingo" | cross-attention truy vấn có thể học tương tự, được Flamingo sử dụng ở mọi lớp thay vì như một cầu nối duy nhất |

## Đọc thêm

- [Li et al. — BLIP-2 (arXiv:2301.12597)](https://arxiv.org/abs/2301.12597) - bài báo cốt lõi.
- [Li et al. — BLIP (arXiv:2201.12086)](https://arxiv.org/abs/2201.12086) - người tiền nhiệm với bộ ba ITC/ITM/ITG.
- [Li et al. — ALBEF (arXiv:2107.07651)](https://arxiv.org/abs/2107.07651) - "căn chỉnh trước khi hợp nhất" - tổ tiên khái niệm của giai đoạn 1 training.
- [Dai et al. — InstructBLIP (arXiv:2305.06500)](https://arxiv.org/abs/2305.06500) — Q-Former nhận biết hướng dẫn.
- [Zhu et al. — MiniGPT-4 (arXiv:2304.10592)](https://arxiv.org/abs/2304.10592) - cách tiếp cận chỉ dành cho máy chiếu.
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — kiến trúc chung cho cross-attention truy vấn có thể học được.
