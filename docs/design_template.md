# Design Document: Multi-Agent Research System

## Problem

Hệ thống cần tự động tiếp nhận các câu hỏi nghiên cứu kỹ thuật phức tạp (ví dụ: *"Research GraphRAG state-of-the-art and write a summary"*), thu thập các nguồn tài liệu xác thực từ web/offline corpus, phân tích đối chiếu các luận điểm mâu thuẫn/trade-offs, và tổng hợp thành báo cáo hoàn chỉnh có trích dẫn nguồn chuẩn xác (citations).

## Why multi-agent?

1. **Phân tách trách nhiệm (Separation of Concerns)**: Single-agent thường bị giới hạn bởi context window hoặc xu hướng hallucinate khi vừa phải tự tìm kiếm, vừa tự phân tích và vừa tự viết dài.
2. **Khả năng kiểm chứng độc lập (Independent Verification & Grounding)**: Chia nhỏ thành Researcher (thu thập sự thật) -> Analyst (phân tích mâu thuẫn) -> Writer (tổng hợp văn phong) giúp dữ liệu được kiểm định chéo và bám sát bằng chứng.
3. **Quản trị trạng thái và Guardrails**: Supervisor điều phối luồng thực thi, đảm bảo có điều kiện dừng (`max_iterations`), tránh lặp vòng hoặc suy diễn ngoài tài liệu.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| **Supervisor** | Điều phối luồng, kiểm tra guardrails và ra quyết định bước tiếp theo | `ResearchState` | Route kế tiếp (`researcher`, `analyst`, `writer`, `done`) | Vòng lặp vô hạn (khắc phục bằng `max_iterations`) |
| **Researcher** | Tìm kiếm web (Tavily) hoặc Offline Corpus v2, lọc nguồn và rút trích sự thật | `state.request.query` | `state.sources`, `state.research_notes` | Không tìm thấy nguồn phù hợp (khắc phục bằng multi-tier fallback) |
| **Analyst** | Đánh giá luận điểm, so sánh trade-offs và phát hiện mâu thuẫn | `state.research_notes`, `state.sources` | `state.analysis_notes` | Phân tích hời hợt / lặp lại ghi chú (khắc phục bằng prompt 4 phần chuyên sâu) |
| **Writer** | Tổng hợp báo cáo kỹ thuật hoàn chỉnh kèm danh mục trích dẫn | `state.research_notes`, `state.analysis_notes`, `state.sources` | `state.final_answer` | Bỏ quên trích dẫn nguồn (khắc phục bằng format citation bắt buộc) |

## Shared state (`ResearchState`)

- `request`: Chứa `ResearchQuery` (query, max_sources, audience).
- `iteration`: Bộ đếm vòng lặp để kích hoạt guardrail giới hạn.
- `route_history`: Lịch sử các bước chuyển agent để audit và vẽ timeline.
- `sources`: Danh sách `SourceDocument` (title, url/ID, snippet).
- `research_notes`: Ghi chú sự thật từ Researcher.
- `analysis_notes`: Báo cáo phân tích chuyên sâu từ Analyst.
- `final_answer`: Báo cáo tổng hợp cuối cùng từ Writer.
- `agent_results`: Kết quả chi tiết và metadata token/latency của từng Agent.
- `trace`: Nhật ký sự kiện phục vụ tracing (LangSmith / Langfuse / Local).
- `errors`: Danh sách các lỗi phát sinh trong quá trình chạy.

## Routing policy

```text
       [Entry]
          │
          ▼
   ┌──────────────┐
   │  Supervisor  │ ◄───────┐
   └──────┬───────┘         │
          │ (conditional)   │
   ┌──────┴──────┬──────────┴────────┐
   ▼             ▼                   ▼
Researcher    Analyst              Writer
   │             │                   │
   └─────────────┴───────────────────┘
          │ (final_answer present)
          ▼
        [END]
```

- Khởi tạo: Chưa có tài liệu ➔ `researcher`
- Đã có `research_notes` / `sources` ➔ `analyst`
- Đã có `analysis_notes` ➔ `writer`
- Đã có `final_answer` hoặc chạm `max_iterations` ➔ `done` (`END`)

## Guardrails

- **Max iterations**: Cấu hình `MAX_ITERATIONS=6` ngắt chu trình chống lặp vô hạn.
- **Timeout**: Cấu hình `TIMEOUT_SECONDS=60` chống nghẽn mạng khi gọi API.
- **Retry**: Tích hợp cơ chế *Exponential Backoff Retry* (3 lần) qua thư viện `tenacity`.
- **Fallback**: 
  - LLM: Tự động fallback giữa OpenRouter và OpenAI.
  - Search: Tự động chuyển tầng Tavily ➔ Offline Corpus v2 ➔ Fallback Doc.
- **Validation**: Toàn bộ dữ liệu vào/ra được validate chặt chẽ qua Pydantic v2 schemas.

## Benchmark plan

- **Câu hỏi benchmark**: *"Research GraphRAG state-of-the-art and write a summary"*
- **Tiêu chí đo lường**:
  - *Latency*: Thời gian thực thi (wall-clock seconds).
  - *Estimated Cost*: Chi phí token USD ($0.15/1M input, $0.60/1M output).
  - *Quality Score*: Thang điểm 0-10 đánh giá cấu trúc, chiều sâu và độ hoàn thiện.
  - *Citation Coverage*: Tỉ lệ các luận điểm có trích dẫn nguồn kiểm chứng được.
  - *Failure Rate*: Tỉ lệ thất bại (0% là tối ưu).
- **Kết quả kỳ vọng**: Multi-Agent cho chất lượng vượt trội (10/10 vs 6/10), 100% citation coverage, chấp nhận đánh đổi latency cao hơn (~38s vs ~16s) để đạt tính chính xác học thuật cao.
