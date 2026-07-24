# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 14h00–18h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.7, 1.2 và 1.8 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Hà Nội."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi? Ở mức nào phản hồi bắt đầu
kém mạch lạc?** (2–3 câu)
> Khi temperature tăng từ 0.0 đến 0.7, phản hồi trở nên phong phú, đa dạng từ vựng hơn nhưng vẫn giữ đúng logic và sự thật lịch sử. Khi tăng lên 1.2, câu văn bắt đầu mang tính sáng tạo cao và dùng từ ngữ lạ; đến mức 1.8, phản hồi bắt đầu xuất hiện lỗi lặp từ, mất mạch lạc và suy diễn sai sự thật (hallucination).

### Câu 1.2 — Chọn temperature cho sản phẩm

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý,
và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**
> Tôi sẽ đặt temperature = 0.0 cho trợ lý hợp đồng pháp lý để đảm bảo tính chính xác tuyệt đối, nhất quán và không tự ý sáng tạo các điều khoản gây rủi ro pháp lý. Ngược lại, tôi chọn temperature = 0.8 đến 1.0 cho trợ lý viết slogan để kích thích tính sáng tạo, tạo ra nhiều ý tưởng độc đáo và góc nhìn mới mẻ.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần,
mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ cho workload này
(dựa trên bảng giá trong template). Nêu một trường hợp model lớn xứng đáng
với chi phí và một trường hợp model nhỏ là lựa chọn đúng:**
> > Tổng lượng token đầu ra mỗi ngày: 20.000 user × 2 lần × 500 token = 20.000.000 token.
> - Chi phí gpt-4o ($0.010/1K token): 20.000 × 0.010 = $200 / ngày.
> - Chi phí gpt-4o-mini ($0.0006/1K token): 20.000 × 0.0006 = $12 / ngày (rẻ hơn ~16.6 lần).
> Model lớn (gpt-4o) xứng đáng chi phí cho các nhiệm vụ đòi hỏi tư duy logic phức tạp như lập trình hệ thống, phân tích văn bản pháp lý. Model nhỏ (gpt-4o-mini) là lựa chọn tối ưu cho các tác vụ đơn giản như phân loại phản hồi khách hàng, tóm tắt câu ngắn hoặc chatbot tư vấn quy trình cơ bản.
---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích máy học (machine learning) là gì?"** nhưng hai system prompt
khác nhau:
- "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ."
- "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp."

**Hai phản hồi khác nhau như thế nào (giọng văn, độ dài, mức kỹ thuật)?
Từ đó rút ra system prompt điều khiển được những khía cạnh nào của phản hồi?**
(3–4 câu)
> Phản hồi của nhà thơ có giọng văn giàu hình ảnh, dùng các ví dụ so sánh ẩn dụ (như việc một đứa trẻ tập đi) và hoàn toàn không dùng thuật ngữ chuyên ngành. Trong khi đó, kỹ sư senior trả lời ngắn gọn, tập trung vào định nghĩa kỹ thuật chính xác kèm đoạn code Python giả lập đơn giản. Điều này chứng minh System Prompt điều khiển được: giọng văn (persona), mức độ chuyên sâu kỹ thuật, cấu trúc/định dạng đầu ra và phạm vi từ vựng sử dụng.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~150 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Nếu dùng ước lượng thô để dự
toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán thiếu hay thừa —
và vì sao?**
> Thử nghiệm với đoạn văn tiếng Việt 150 từ: công thức thô `150 / 0.75` ước tính là 200 token, nhưng `count_tokens` (tiktoken) trả về khoảng 270–300 token (chênh lệch cao hơn ~35–50%). Nếu dùng ước lượng thô cho ứng dụng tiếng Việt, chúng ta sẽ dự toán **thiếu** ngân sách thực tế. Nguyên nhân là do tiếng Việt chứa nhiều từ ghép, từ có dấu thanh làm bộ tách từ Byte-Pair Encoding (BPE) của tiktoken phải phân rã thành nhiều sub-token hơn so với tiếng Anh.



## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Xét ba ứng dụng: (a) chatbot văn bản, (b) trợ lý giọng nói đọc to phản hồi,
(c) pipeline dịch tài liệu chạy ngầm ban đêm. Ứng dụng nào hưởng lợi nhiều
nhất từ streaming, ứng dụng nào không cần — và tại sao?** (1 đoạn văn)
> Chatbot văn bản hưởng lợi nhiều nhất từ streaming vì giúp giảm thời gian chờ phản hồi (perceived latency), mang lại cảm giác tương tác tự nhiên theo thời gian thực cho người dùng. Trợ lý giọng nói cũng tận dụng streaming để nhận trước từng câu văn bản và chuyển đổi thành giọng nói (TTS) cuốn chiếu. Ngược lại, pipeline dịch tài liệu chạy ngầm ban đêm hoàn toàn không cần streaming vì là tác vụ xử lý lô (batch processing) không có người dùng chờ trực tiếp, hệ thống chỉ cần kết quả hoàn chỉnh cuối cùng.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**Khi API quá tải và hàng nghìn client cùng retry, exponential backoff giúp
gì so với delay cố định? Tra cứu thêm: kỹ thuật "jitter" (thêm độ trễ ngẫu
nhiên) giải quyết vấn đề gì còn sót lại?**
> Exponential backoff giúp tăng khoảng thời gian chờ theo cấp số nhân ($2^{attempt}$) giữa các lần thử lại, giảm mật độ yêu cầu dồn dập và cho hệ thống phía server có đủ thời gian tự phục hồi khi quá tải. Tuy nhiên, nếu hàng nghìn client cùng bị lỗi tại một thời điểm, chúng vẫn sẽ gửi lại yêu cầu đồng loạt vào các mốc thời gian cố định. Kỹ thuật "Jitter" (thêm độ trễ ngẫu nhiên vào mỗi khoảng chờ) giúp phá vỡ sự đồng bộ này, phân tán đều các lượt retry để tránh sự cố "thảm họa dồn dập" (Thundering Herd Problem).

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Viết lại system prompt bạn dùng cho trợ lý của mình. Chỉ ra 2 chỗ trong
prompt mà nếu xóa đi, hành vi trợ lý sẽ thay đổi rõ rệt — và mô tả thay đổi
đó:**
> System prompt: `"Bạn là trợ giảng thân thiện của khóa AI, luôn trả lời bằng tiếng Việt ngắn gọn dưới 3 câu và giải thích có kèm ví dụ thực tế."`
> 1. Nếu xóa cụm `"luôn trả lời bằng tiếng Việt"`: Trợ lý sẽ tự động trả lời bằng tiếng Anh hoặc ngôn ngữ khác nếu người dùng nhập câu hỏi bằng tiếng nước ngoài.
> 2. Nếu xóa cụm `"ngắn gọn dưới 3 câu"`: Trợ lý sẽ phân tích và giải thích rất dài dòng, chi tiết, dẫn đến việc tiêu tốn token đầu ra không cần thiết. 

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn giữ history 4 lượt cuối. Hãy mô tả một tình huống hội thoại
cụ thể mà giới hạn này khiến trợ lý trả lời sai/mất ngữ cảnh, và đề xuất một
cách khắc phục (ví dụ: tóm tắt các lượt cũ, tăng giới hạn có chọn lọc...):**
> Tình huống: Người dùng liệt kê 5 yêu cầu phân tích dữ liệu ở lượt hỏi thứ 1. Sau khi trao đổi qua lại 4 lượt hỏi-đáp ngắn khác, đến lượt thứ 6 người dùng bảo "Hãy thực hiện yêu cầu thứ 3 trong danh sách tôi đã gửi", trợ lý sẽ không thể trả lời đúng do thông tin lượt 1 đã bị cắt khỏi history (chỉ giữ 4 lượt gần nhất tức 8 messages).
> Cách khắc phục: Triển khai cơ chế tóm tắt lịch sử (Conversation Summary Buffer) — khi lịch sử vượt quá 4 lượt, sử dụng một model nhỏ (như `gpt-4o-mini`) chạy ngầm để tóm tắt các thông tin quan trọng từ các lượt cũ thành 1 đoạn văn vắn tắt và gán đoạn tóm tắt đó vào System Prompt.
---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên GitHub cá nhân và nộp link repo vào vlearn (theo hướng dẫn README)
