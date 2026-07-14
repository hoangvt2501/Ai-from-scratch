<!-- Bản dịch tiếng Việt; giữ nguyên thuật ngữ kỹ thuật, code, công thức, lệnh và URL. -->

# Mẫu bài học

Sử dụng mẫu này khi tạo bài học mới. Sao chép cấu trúc thư mục và điền vào nội dung.

## Cấu trúc thư mục

```
NN-lesson-name/
├── code/
│   ├── main.py            (primary implementation)
│   ├── main.ts            (TypeScript version, if applicable)
│   ├── main.rs            (Rust version, if applicable)
│   └── main.jl            (Julia version, if applicable)
├── notebook/
│   └── lesson.ipynb       (Jupyter notebook for experimentation)
├── docs/
│   └── en.md              (lesson documentation)
└── outputs/
    ├── prompt-*.md         (prompts produced by this lesson)
    └── skill-*.md          (skills produced by this lesson)
```

## Định dạng tài liệu (docs/en.md)

```markdown
# [Lesson Title]

> [One-line motto — the core idea that sticks]

**Type:** Build | Learn
**Languages:** Python, TypeScript, Rust, Julia (list what's used)
**Prerequisites:** [List prior lessons needed]
**Time:** ~[estimated time] minutes

## The Problem

[2-3 paragraphs. What can't you do without this? Why should you care?
Make it concrete — show a scenario where not knowing this hurts.]

## The Concept

[Explain with diagrams and intuition. No code yet.
Use ASCII diagrams, tables, or link to visuals in the web app.
Build mental models before implementation.]

## Build It

[Step-by-step implementation from scratch.
Start with the simplest version, then add complexity.
Every code block should be runnable on its own.]

### Step 1: [Name]

[Explanation]

    [code block]

### Step 2: [Name]

[Explanation]

    [code block]

[...continue...]

## Use It

[Now show how frameworks/libraries do the same thing.
Compare your from-scratch version to the library version.
This proves the concept and introduces practical tools.]

## Ship It

[What reusable artifact does this lesson produce?
Could be a prompt, a skill, an agent, an MCP server, or a tool.
Include it here and save it in the outputs/ folder.]

## Exercises

1. [Easy — reinforce the core concept]
2. [Medium — apply it to a different problem]
3. [Hard — extend or combine with prior lessons]

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| [term] | [common misconception] | [actual definition] |

## Further Reading

- [Resource 1](url) — [why it's worth reading]
- [Resource 2](url) — [why it's worth reading]
```

## Hướng dẫn tệp mã

- Mã phải chạy mà không có lỗi
- Không có bình luận - mã phải tự giải thích
- Sử dụng ngôn ngữ phù hợp nhất với chủ đề
- Bao gồm `requirements.txt` hoặc tương đương nếu có phần phụ thuộc
- Bắt đầu đơn giản, xây dựng sự phức tạp
- Mỗi chức năng và class nên có mục đích rõ ràng

## Định dạng tệp đầu ra

### Prompts

```markdown
---
name: prompt-name
description: What this prompt does
phase: [phase number]
lesson: [lesson number]
---

[Prompt content]
```

### Skills

```markdown
---
name: skill-name
description: What this skill teaches
version: 1.0.0
phase: [phase number]
lesson: [lesson number]
tags: [relevant, tags]
---

[Skill content]
```
