# Tiêu chí công bằng - Nhóm, Cá nhân, Phản thực tế

> Ba gia đình cấu trúc văn học công bằng. Công bằng nhóm: bình đẳng nhân khẩu học, tỷ lệ cân bằng, sử dụng có điều kiện accuracy bình đẳng - tỷ lệ trung bình giữa các nhóm được bảo vệ. Công bằng cá nhân (Dwork et al. 2012): các cá nhân tương tự nhận được các quyết định tương tự; Điều kiện Lipschitz trên bản đồ quyết định. Công bằng phản thực tế (Kusner et al. 2017): một quyết định là công bằng đối với một cá nhân nếu nó không thay đổi khi các thuộc tính nhạy cảm bị thay đổi phản thực tế. Kết quả lý thuyết năm 2024 (NeurIPS 2024): có sự đánh đổi vốn có giữa CF-vs-accuracy; một phương pháp bất khả tri model chuyển đổi một dự đoán tối ưu nhưng không công bằng thành một CF có accuracy loss giới hạn. Backtrack counterfactuals (arXiv:2401.13935, tháng 1 năm 2024): mô hình mới tránh yêu cầu can thiệp vào các thuộc tính được bảo vệ hợp pháp. Hòa giải triết học (ICLR Blogposts 2024): với biểu đồ nhân quả, việc thỏa mãn các biện pháp công bằng nhóm nhất định đòi hỏi sự công bằng phản thực tế.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, three-criteria comparison)
**Kiến thức tiên quyết:** Giai đoạn 18 · 20 (bias), Giai đoạn 02 (ML cổ điển)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Nêu ba tiêu chí công bằng nhóm (bình đẳng nhân khẩu học, tỷ lệ cân bằng, sử dụng có điều kiện accuracy bình đẳng) và một kết quả bất khả thi.
- Mô tả sự công bằng cá nhân thông qua công thức Lipschitz năm 2012 của Dwork et al.
- Mô tả tính công bằng phản thực tế và sự phụ thuộc vào đồ thị nhân quả của nó.
- Giải thích phản thực tế và lý do tại sao chúng tránh được vấn đề can thiệp vào thuộc tính được bảo vệ.

## Vấn đề

Bài 20 là về đo lường bias. Bài 21 là về việc xác định tiêu chuẩn công bằng mà phép đo nên phục vụ. Ba gia đình đưa ra các tiêu chuẩn khác nhau về mặt cấu trúc - một model có thể công bằng nhóm và cá nhân không công bằng, công bằng phản thực tế và không công bằng nhóm. Chọn một tiêu chuẩn là một quyết định policy; không có tiêu chuẩn nào là tối ưu trên toàn cầu.

## Khái niệm

### Công bằng nhóm

- **Tương đương nhân khẩu học.** P (Y = 1 | A = a) = P (Y = 1 | A=a') cho tất cả các nhóm. Tỷ lệ chấp nhận bình đẳng.
- **Tỷ lệ cân bằng.** P(Y=1 | Y * = y, A = a) = P (Y = 1 | Y*=y, A=a'). TPR và FPR bằng nhau giữa các nhóm.
- **Sử dụng có điều kiện accuracy bình đẳng.** P(Y*=y | Y = y, A = a) = P (Y * = y | Y = y, A = a'). Giá trị dự đoán bằng nhau giữa các nhóm.

Không thể (Chouldechova, Kleinberg-Mullainathan-Raghavan 2017): ba điều này không thể được thỏa mãn đồng thời với tỷ lệ cơ bản không bằng nhau.

### Công bằng cá nhân

Dwork và cộng sự 2012. Một bản đồ quyết định f là công bằng riêng lẻ đối với một chỉ số tương tự cụ thể của nhiệm vụ d nếu |f(x) - f(x')| <= L * d(x, x') đối với một số hằng số Lipschitz L. Các cá nhân tương tự nhận được các quyết định tương tự.

Yêu cầu định nghĩa d. Policy câu hỏi, không phải thống kê.

### Công bằng phản thực tế

Kusner và cộng sự 2017. Một quyết định là công bằng phản thực tế đối với cá nhân i nếu, theo model nhân quả của dân số, quyết định không thay đổi khi các thuộc tính nhạy cảm của i bị thay đổi phản thực tế.

Yêu cầu DAG nhân quả. DAG là một lựa chọn mô hình. Sự công bằng phản thực tế chỉ được biện minh như DAG.

### Sự đánh đổi giữa CF-vs-accuracy

Lý thuyết NeurIPS 2024: có một sự đánh đổi cố hữu giữa tính công bằng phản thực tế và accuracy dự đoán. Một phương pháp bất khả tri model có thể chuyển đổi một yếu tố dự đoán tối ưu nhưng không công bằng thành một phương pháp CF, với chi phí accuracy giới hạn. Chi phí accuracy phụ thuộc vào độ lớn của hệ số thuộc tính nhạy cảm trong yếu tố dự đoán không công bằng tối ưu.

### Ngược lại phản thực tế

arXiv:2401.13935 (tháng 1 năm 2024). Các phản thực tế truyền thống yêu cầu can thiệp vào thuộc tính nhạy cảm - "liệu quyết định có thay đổi nếu người này là một giới tính khác." Về mặt pháp lý, điều này có vấn đề: các thuộc tính được bảo vệ không thể bị can thiệp vào luật phân loại.

Ngược lại phản thực tế lật ngược hướng: thay vì can thiệp vào thuộc tính, hãy hỏi sự kết hợp nào giữa features thực tế của cá nhân sẽ tạo ra kết quả phản thực tế. Điều này tránh được sự phản đối pháp lý.

### Hòa giải triết học

Bài đăng trên blog của ICLR 2024. Với một biểu đồ nhân quả trong tay, việc thỏa mãn một số biện pháp công bằng nhóm nhất định đòi hỏi sự công bằng phản thực tế. Ba họ không trực giao; chúng là những khía cạnh khác nhau của cùng một cấu trúc nhân quả cơ bản.

Điều này không giải quyết được các định lý bất khả thi (tỷ lệ cơ sở không bằng nhau vẫn ngăn cản sự công bằng của nhóm đồng thời). Nhưng nó cho thấy sự đối lập rõ ràng giữa "nhóm" và "cá nhân / phản thực tế" một phần là một artifact của việc không rõ ràng về model nhân quả.

### Điều này phù hợp với Giai đoạn 18 ở đâu

Bài 20 là bias đo lường. Bài 21 là định nghĩa công bằng. Bài 22 là quyền riêng tư (quyền riêng tư khác biệt). Bài 23 là watermarking. Đây là những bài học liền kề với phân bổ bổ sung cho Bài học 7-11 liền kề với sự lừa dối.

## Ứng dụng

`code/main.py` xây dựng một dataset phân loại nhị phân đồ chơi với thuộc tính nhạy cảm và tỷ lệ cơ sở không bằng nhau. Tính toán tính ngang bằng nhân khẩu học, tỷ lệ cân bằng và sử dụng có điều kiện accuracy bình đẳng trên một bộ phân loại đơn giản. Quan sát ba số liệu không đồng ý. Áp dụng trọng số lại cho sự ngang bằng nhân khẩu học và quan sát chi phí của nó đối với hai phương thức còn lại.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-fairness-criterion.md`. Đưa ra yêu cầu công bằng hoặc policy, xác định tiêu chí nào đang được yêu cầu, liệu model có thể đáp ứng các tiêu chí còn lại theo tỷ lệ cơ bản không bằng nhau được tuyên bố hay không và DAG nhân quả mà yêu cầu phụ thuộc vào.

## Bài tập

1. Chạy `code/main.py`. Báo cáo ba chỉ số nhóm trên dữ liệu mặc định. Áp dụng trọng số lại và báo cáo lại theo mục tiêu nhân khẩu học-chẵn lẻ.

2. Thực hiện Dwork et al. 2012 chỉ số công bằng cá nhân bằng cách sử dụng L2 trên features không nhạy cảm. Báo cáo có bao nhiêu cặp vi phạm Lipschitz với hằng số L = 1.

3. Đọc Kusner et al. 2017. Xây dựng một DAG nhân quả hai feature đơn giản để chấm điểm sơ yếu lý lịch và xác định điều kiện công bằng phản thực tế mà nó ngụ ý.

4. Báo cáo phản thực tế năm 2024 tránh can thiệp vào các thuộc tính được bảo vệ. Mô tả một tình huống mà điều này quan trọng đối với việc tuân thủ pháp luật.

5. Hòa giải ICLR 2024 lập luận rằng sự công bằng nhóm và phản thực tế là những khía cạnh của cùng một cấu trúc. Chọn hai trong số ba tiêu chí trong `code/main.py` và nêu giả định nhân quả sẽ làm cho chúng tương đương.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| Sự tương đương về nhân khẩu học | "Tỷ lệ bằng nhau" | P (Y = 1 | A = a) bằng nhau giữa các nhóm |
| Tỷ lệ cân bằng | "TPR/FPR bình đẳng" | Tỷ lệ dương tính thật và dương tính giả bằng nhau giữa các nhóm |
| Sử dụng có điều kiện accuracy | "PPV/NPV bình đẳng" | Giá trị dự đoán bằng nhau giữa các nhóm |
| Công bằng cá nhân | "Tình trạng Lipschitz" | Các cá nhân tương tự cũng nhận được quyết định tương tự |
| Công bằng phản thực tế | "Bất biến thay đổi nhân quả" | Quyết định không thay đổi theo thay đổi thuộc tính phản thực tế |
| Ngược lại phản thực tế | "Giải thích qua thực tế" | Lý luận phản thực tế lùi lại từ kết quả, không chuyển tiếp từ thuộc tính |
| Định lý bất khả thi | "Ba cuộc xung đột" | Chouldechova / KMR 2017: tiêu chí nhóm loại trừ lẫn nhau theo tỷ lệ cơ sở không bình đẳng |

## Đọc thêm

- [Dwork et al. — Fairness through Awareness (arXiv:1104.3913)](https://arxiv.org/abs/1104.3913) - công bằng cá nhân
- [Kusner, Loftus, Russell, Silva — Counterfactual Fairness (arXiv:1703.06856)](https://arxiv.org/abs/1703.06856) - công bằng phản thực tế
- [Chouldechova — Fair prediction with disparate impact (arXiv:1703.00056)](https://arxiv.org/abs/1703.00056) - không thể
- [Backtracking Counterfactuals (arXiv:2401.13935)](https://arxiv.org/abs/2401.13935) - mô hình mới cho các can thiệp thuộc tính được bảo vệ
