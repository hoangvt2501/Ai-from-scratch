<!-- Bản dịch tiếng Việt; giữ nguyên thuật ngữ kỹ thuật, code, công thức, lệnh và URL. -->

# Đóng góp

Bài học, bản dịch, bản sửa lỗi, đầu ra - tất cả đều được hoan nghênh. Một đóng góp cho mỗi lần kéo
yêu cầu giữ đánh giá nhanh chóng và cho phép số lượng người đóng góp và công việc tín dụng
một cách chính xác.

## Quan trọng: README và ROADMAP cung cấp cho trang web

`site/build.js` phân tích cú pháp `README.md`, `ROADMAP.md` và `glossary/terms.md` thành
tạo `site/data.js`. Hai mẫu phải nguyên vẹn trong bất kỳ pull request nào
chạm vào các tập tin đó:

- Tiêu đề giai đoạn ở dạng bài học `### Phase N: Name \`X hoặc
`<details><summary><b>Phase N — Name</b> ... <code>X lessons</code> ... <em>Description</em></summary>` hình thức.
- Bảng bài học có hình dạng cột `| # | Lesson | Type | Lang |` (hoặc
`| # | Project | Combines | Lang |` cho bảng capstone). Cột `Lang`
Chấp nhận văn bản thuần túy (`Python, TypeScript`) hoặc cờ biểu tượng cảm xúc cũ
(`🐍 🟦 🦀 🟣 ⚛️`); Cả hai đều tương đương với trình phân tích cú pháp.
- Glyph trạng thái ROADMAP (`✅`, `🚧`, `⬚`) trên tiêu đề pha và hàng bài học.
Đừng thay thế chúng bằng văn bản - trình phân tích cú pháp sẽ loại bỏ các ký tự chính xác.

Chạy `node site/build.js` sau khi chỉnh sửa các tệp đó; `git diff site/data.js`
sẽ chỉ hiển thị thay đổi dấu thời gian nếu sửa đổi của bạn an toàn về cấu trúc.

## Cách đóng góp

### 1. Thêm một bài học mới

Mỗi bài học sống trong `phases/XX-phase-name/NN-lesson-name/` với cấu trúc này:

```
NN-lesson-name/
├── code/           At least one runnable implementation
├── notebook/       Jupyter notebook for experimentation (optional)
├── docs/
│   └── en.md       Lesson documentation (required)
└── outputs/        Prompts, skills, or agents this lesson produces (if applicable)
```

**Định dạng tài liệu bài học** (`en.md`):

```markdown
# Lesson Title

> One-line motto — the core idea in one sentence.

## The Problem

Why does this matter? What can't you do without this?

## The Concept

Explain with diagrams, visuals, and intuition. Code comes later.

## Build It

Step-by-step implementation from scratch.

## Use It

Now use a real framework or library to do the same thing.

## Ship It

The prompt, skill, agent, or tool this lesson produces.

## Exercises

1. Exercise one
2. Exercise two
3. Challenge exercise
```

### 2. Thêm bản dịch

Tạo tệp mới trong thư mục `docs/` của bài học bất kỳ:

```
docs/
├── en.md    (English — always required)
├── zh.md    (Chinese)
├── ja.md    (Japanese)
├── es.md    (Spanish)
├── hi.md    (Hindi)
└── ...
```

Giữ nguyên cấu trúc như phiên bản tiếng Anh. Dịch nội dung chứ không phải mã.

### 3. Thêm đầu ra

Nếu một bài học tạo ra một prompt, skill, agent hoặc MCP server có thể tái sử dụng:

1. Tạo nó trong thư mục `outputs/` của bài học
2. Thêm tham chiếu trong chỉ mục `outputs/` cấp cao nhất

**Prompt định dạng:**

```markdown
---
name: prompt-name
description: What this prompt does
phase: 14
lesson: 01
---

[System prompt or template here]
```

**Skill định dạng:**

```markdown
---
name: skill-name
description: What this skill teaches
version: 1.0.0
phase: 14
lesson: 01
tags: [agents, loops]
---

[Skill content here]
```

### 4. Sửa lỗi hoặc cải thiện các bài học hiện có

- Sửa mã không chạy
- Cải thiện giải thích
- Thêm sơ đồ tốt hơn
- Cập nhật thông tin lỗi thời

### 5. Thêm bài tập hoặc dự án

Nhiều bài tập và dự án luôn được hoan nghênh, đặc biệt là những bài tập kết nối nhiều giai đoạn.

## Hướng dẫn

- **Mã phải chạy.** Mọi tệp mã phải thực thi mà không có lỗi với các phần phụ thuộc được liệt kê.
- **Không có nhận xét trong mã.** Mã phải tự giải thích. Sử dụng tài liệu để giải thích.
- **Ngôn ngữ tốt nhất cho công việc.** Đừng ép buộc Python nơi TypeScript hoặc Rust là lựa chọn tốt hơn.
- **Xây dựng từ đầu trước.** Luôn thực hiện khái niệm từ các nguyên tắc đầu tiên trước khi hiển thị phiên bản framework.
- **Giữ nó thực tế.** Lý thuyết phục vụ thực tiễn, không phải ngược lại.
- **Không AI cẩu.** Viết như một con người. Hãy thẳng thắn. Cắt chất độn.

## Pull Request Process

1. Ngã ba repository
2. Tạo feature branch (`git checkout -b add-lesson-phase3-gradient-descent`)
3. Thực hiện thay đổi của bạn
4. Đảm bảo tất cả mã chạy
5. Gửi pull request có mô tả rõ ràng

## Quy tắc ứng xử

Xem [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Hãy tử tế, giúp đỡ, xây dựng.

## Phong cách

- Văn xuôi trực tiếp. Cắt chất độn. Phù hợp với giọng điệu của sách hướng dẫn, không phải bản sao tiếp thị.
- Không có biểu tượng cảm xúc trang trí trong tiêu đề. Cờ biểu tượng cảm xúc cột Lang là một
ngoại lệ và chỉ vì trình phân tích cú pháp ánh xạ chúng.
- Mã chạy nguyên trạng với các phần phụ thuộc được liệt kê trong bài học.
- Xây dựng từ đầu trước, framework thứ hai.
