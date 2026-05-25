# Sarcasm Detection in Online Comments using Machine Learning

Đồ án môn **CS221 - Xử lý ngôn ngữ tự nhiên** tại Trường Đại học Công nghệ Thông tin, ĐHQG-HCM. Đề tài tập trung xây dựng hệ thống phát hiện bình luận mỉa mai trong hội thoại trực tuyến, sử dụng dữ liệu Reddit và các mô hình học sâu dựa trên **DistilBERT**.

## 1. Thông tin đồ án

| Hạng mục | Nội dung |
|---|---|
| Tên đề tài | Sarcasm Detection in Online Comments using Machine Learning |
| Môn học | CS221 - Xử lý ngôn ngữ tự nhiên |
| Lớp | CS221.Q21 |
| Giảng viên hướng dẫn | TS. Nguyễn Trọng Chỉnh |
| Sinh viên thực hiện | Lê Xuân Song Lĩnh - 23520845<br>Trần Khoa Tuấn - 22521611<br>Nguyễn Chí Thanh - 23521448 |
| Bài toán | Phân loại nhị phân: bình luận mỉa mai / không mỉa mai |
| Ngôn ngữ dữ liệu | Tiếng Anh |
| Bộ dữ liệu | SARC 2.0 / Sarcasm on Reddit |

## 2. Mục tiêu đề tài

Mỉa mai là hiện tượng ngôn ngữ trong đó nghĩa bề mặt của câu nói có thể trái ngược với ý định thật sự của người viết. Trong các bài toán xử lý ngôn ngữ tự nhiên như phân tích cảm xúc, khai thác ý kiến hoặc giám sát nội dung, nếu không nhận diện được mỉa mai thì hệ thống rất dễ hiểu sai ý nghĩa của bình luận.

Đồ án này hướng đến các mục tiêu chính:

- Xây dựng mô hình tự động phát hiện mỉa mai trong bình luận trực tuyến.
- Khai thác **ngữ cảnh hội thoại**, cụ thể là cặp `parent_comment` và `comment`, thay vì chỉ xét một câu đơn lẻ.
- So sánh ba mức mô hình từ đơn giản đến nâng cao:
  - Level 1: chỉ dùng bình luận phản hồi.
  - Level 2: dùng cả bình luận phản hồi và ngữ cảnh cha.
  - Level 3: bổ sung thêm đặc trưng cảm xúc.
- Đánh giá mô hình bằng các chỉ số Accuracy, Precision, Recall, F1-score, ROC và AUC.
- Phân tích các trường hợp dự đoán sai để hiểu rõ hạn chế của mô hình.

## 3. Dữ liệu

Đồ án sử dụng bộ dữ liệu **SARC 2.0 / Sarcasm on Reddit**, được công bố trên Kaggle:

```text
https://www.kaggle.com/datasets/danofer/sarcasm
```

File dữ liệu chính được sử dụng trong đồ án là:

```text
train-balanced-sarcasm.csv
```

Trong notebook, dữ liệu được đọc từ đường dẫn Kaggle:

```python
/kaggle/input/cs213data/sarcasm.csv
```

Khi chạy ở máy cá nhân, cần tải dataset từ Kaggle và đổi lại biến đường dẫn tương ứng trong các notebook.

### 3.1 Kích thước và cấu trúc dữ liệu

Bộ dữ liệu gốc có khoảng **1.010.826 dòng** và **10 cột**:

| Cột | Ý nghĩa |
|---|---|
| `label` | Nhãn phân loại, 0 là không mỉa mai, 1 là mỉa mai |
| `comment` | Bình luận cần phân loại |
| `author` | Tên người dùng Reddit |
| `subreddit` | Chủ đề / cộng đồng Reddit |
| `score` | Điểm số của bình luận |
| `ups` | Số lượt upvote |
| `downs` | Số lượt downvote |
| `date` | Ngày bình luận |
| `created_utc` | Thời gian tạo theo chuẩn UTC |
| `parent_comment` | Bình luận cha, đóng vai trò ngữ cảnh |

Trong quá trình xây dựng mô hình, nhóm tập trung vào ba cột quan trọng nhất:

```text
label, comment, parent_comment
```

### 3.2 Ý nghĩa nhãn

| Nhãn | Ý nghĩa |
|---|---|
| `0` | Bình luận không mỉa mai |
| `1` | Bình luận có mỉa mai |

Trong SARC 2.0, nhãn mỉa mai được suy ra từ ký hiệu `/s`, một quy ước thường dùng trên Reddit để đánh dấu giọng điệu mỉa mai. Ký hiệu này đã được loại bỏ khỏi văn bản để tránh mô hình học theo dấu hiệu bề mặt.

## 4. Phương pháp thực hiện

Pipeline tổng thể của đồ án gồm các bước:

```text
Dữ liệu Reddit
      ↓
Khám phá và phân tích dữ liệu
      ↓
Tiền xử lý văn bản
      ↓
Tokenization bằng DistilBERT Tokenizer
      ↓
Huấn luyện 3 cấp độ mô hình
      ↓
Đánh giá trên tập validation / test
      ↓
So sánh kết quả và phân tích lỗi
```

### 4.1 Tiền xử lý dữ liệu

Các bước tiền xử lý chính:

- Điền giá trị rỗng cho `comment` và `parent_comment`.
- Chuẩn hóa khoảng trắng thừa.
- Loại bỏ hoặc làm sạch các ký tự không cần thiết.
- Tạo cột `comment_pp` và `parent_pp` sau khi tiền xử lý.
- Tính điểm cảm xúc bằng **TextBlob** cho Level 3.
- Chia dữ liệu thành tập train và validation/test theo chiến lược giữ cân bằng nhãn.

### 4.2 Tokenization

Đồ án sử dụng tokenizer của DistilBERT:

```python
DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
```

Các thiết lập chính:

| Tham số | Giá trị |
|---|---:|
| `MODEL_NAME` | `distilbert-base-uncased` |
| `MAX_LEN` | 128 |
| `BATCH_SIZE` khi train | 64 |
| `BATCH_SIZE` khi test | 128 |
| `NUM_EPOCHS` | 2 |
| `LEARNING_RATE` | 2e-5 |
| Optimizer | AdamW |
| Loss function | BCEWithLogitsLoss |

Lý do chọn `MAX_LEN = 128`: phần lớn bình luận trong dataset khá ngắn, khoảng 99.9% bình luận có độ dài không vượt quá 128 token. Thiết lập này giúp giảm chi phí tính toán nhưng vẫn bao phủ gần như toàn bộ dữ liệu.

## 5. Mô hình sử dụng

Đồ án triển khai ba cấp độ mô hình để đánh giá vai trò của nội dung bình luận, ngữ cảnh hội thoại và đặc trưng cảm xúc.

### 5.1 Level 1 - DistilBERT Baseline

Mô hình đầu tiên chỉ sử dụng nội dung của bình luận phản hồi (`comment`). Đây là mô hình cơ sở để đánh giá xem nếu không có ngữ cảnh, hệ thống có thể nhận diện mỉa mai tốt đến đâu.

```text
comment → DistilBERT → CLS embedding → Classifier → Sarcasm / Non-sarcasm
```

Ưu điểm:

- Đơn giản.
- Tốc độ suy diễn nhanh hơn.
- Ít dữ liệu đầu vào hơn.

Hạn chế:

- Không hiểu được các trường hợp mỉa mai phụ thuộc vào ngữ cảnh.
- Dễ nhầm giữa phát ngôn tiêu cực trực tiếp và mỉa mai.

### 5.2 Level 2 - Dual DistilBERT Encoder

Mô hình thứ hai sử dụng cả bình luận phản hồi và bình luận cha. Hai văn bản được đưa qua DistilBERT để lấy biểu diễn ngữ nghĩa, sau đó ghép lại trước khi phân loại.

```text
comment         → DistilBERT → CLS_comment  ┐
                                             ├→ Concatenate → Classifier → Output
parent_comment  → DistilBERT → CLS_parent   ┘
```

Ý tưởng chính của Level 2 là mỉa mai thường không thể nhận diện bằng một câu đơn lẻ. Nhiều bình luận chỉ trở nên mỉa mai khi đặt cạnh phát ngôn trước đó.

Đây là mô hình được chọn làm mô hình tốt nhất của đồ án vì đạt hiệu năng cao nhất và ổn định nhất.

### 5.3 Level 3 - Dual DistilBERT with Sentiment

Mô hình thứ ba mở rộng Level 2 bằng cách thêm đặc trưng cảm xúc từ TextBlob:

```text
comment, parent_comment → Dual DistilBERT embeddings
sentiment_comment, sentiment_parent, sentiment_diff → Sentiment features
Dual embeddings + sentiment features → Classifier → Output
```

Level 3 giúp kiểm tra liệu thông tin cảm xúc có cải thiện khả năng phát hiện mỉa mai hay không.

Kết quả thực nghiệm cho thấy đặc trưng cảm xúc có ích trong một số trường hợp riêng lẻ, nhưng không vượt trội so với Level 2. Nguyên nhân có thể là DistilBERT đã tự học được nhiều tín hiệu ngữ nghĩa và cảm xúc từ dữ liệu văn bản, trong khi TextBlob đôi khi gây nhiễu với ngôn ngữ mạng xã hội.

## 6. Kết quả thực nghiệm

Kết quả trên tập test/validation được tổng hợp như sau:

| Mô hình | Accuracy | F1-score lớp Sarcastic |
|---|---:|---:|
| Level 1 - Comment Only | 0.774591 | 0.773182 |
| Level 2 - Dual Encoder | **0.782387** | **0.781066** |
| Level 3 - Dual Encoder + Sentiment | 0.781022 | 0.780522 |

### 6.1 Nhận xét kết quả

- **Level 2 đạt kết quả tốt nhất**, cho thấy ngữ cảnh hội thoại là yếu tố quan trọng trong bài toán phát hiện mỉa mai.
- Việc dùng thêm `parent_comment` giúp mô hình hiểu được sự đối lập giữa câu trả lời và ngữ cảnh trước đó.
- Level 3 không cải thiện đáng kể so với Level 2, chứng tỏ đặc trưng cảm xúc thủ công từ TextBlob chưa đủ mạnh để tạo khác biệt lớn.
- Mô hình Level 2 được lựa chọn là mô hình tối ưu vì cân bằng tốt giữa hiệu năng, độ ổn định và độ phức tạp triển khai.

## 7. Cấu trúc thư mục

Cấu trúc file trong đồ án:

```text
SARCASM DETECTION/
│
├── eda-data.ipynb
├── 3TL_1.ipynb
├── 3TL_2.ipynb
├── val-test.ipynb
├── Báo cáo đồ án.pdf
└── Dữ liệu phân tích thủ công.docx
```

Ý nghĩa các file:

| File | Mô tả |
|---|---|
| `eda-data.ipynb` | Phân tích dữ liệu, thống kê dataset, biểu đồ phân bố nhãn, độ dài văn bản, word cloud và minh họa quy trình xử lý |
| `3TL_1.ipynb` | Notebook huấn luyện mô hình DistilBERT và các biến thể ban đầu |
| `3TL_2.ipynb` | Notebook huấn luyện các mô hình nâng cao, bao gồm Dual Encoder và Dual Encoder + Sentiment |
| `val-test.ipynb` | Đánh giá mô hình đã lưu, so sánh kết quả, chạy demo dự đoán và phân tích lỗi |
| `Báo cáo đồ án.pdf` | Báo cáo tổng hợp lý thuyết, phương pháp, cài đặt, thực nghiệm và kết luận |
| `Dữ liệu phân tích thủ công.docx` | Tập các mẫu dữ liệu được phân tích thủ công để hỗ trợ giải thích nhãn và phân tích lỗi |

Lưu ý: file ZIP hiện tại không kèm dataset `.csv` và model weight `.pt` do dung lượng lớn. Cần tải dataset/model riêng nếu muốn chạy lại toàn bộ pipeline.

## 8. Cài đặt môi trường

Khuyến nghị chạy trên Kaggle Notebook, Google Colab hoặc máy cá nhân có GPU NVIDIA.

### 8.1 Yêu cầu cơ bản

- Python 3.9+
- PyTorch
- Transformers
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- tqdm
- TextBlob
- WordCloud

### 8.2 Cài đặt thư viện

```bash
pip install torch torchvision torchaudio
pip install transformers scikit-learn pandas numpy matplotlib seaborn tqdm textblob wordcloud
```

Nếu dùng TextBlob lần đầu, có thể cần tải thêm tài nguyên:

```bash
python -m textblob.download_corpora
```

## 9. Hướng dẫn chạy project

### Bước 1: Tải dữ liệu

Tải dataset từ Kaggle:

```text
https://www.kaggle.com/datasets/danofer/sarcasm
```

Sau đó đặt file CSV vào thư mục dữ liệu của bạn. Nếu chạy trên Kaggle, có thể thêm dataset vào notebook và chỉnh đường dẫn như sau:

```python
DATA_PATH = "/kaggle/input/cs213data/sarcasm.csv"
```

Nếu chạy local:

```python
DATA_PATH = "./data/sarcasm.csv"
```

### Bước 2: Chạy EDA

Mở và chạy notebook:

```text
eda-data.ipynb
```

Notebook này dùng để:

- Kiểm tra kích thước dataset.
- Quan sát phân bố nhãn.
- Thống kê độ dài câu.
- Vẽ word cloud.
- Minh họa quá trình làm sạch, tokenization và sentiment extraction.

### Bước 3: Huấn luyện mô hình

Mở notebook huấn luyện:

```text
3TL_1.ipynb
3TL_2.ipynb
```

Trong các notebook, có thể chọn mô hình cần chạy bằng cách bỏ comment các dòng tương ứng:

```python
# Level 1: Comment Only
run_training(DistilBertBaseline, df, tokenizer, 'level1', 'level1_distilbert')

# Level 2: Comment + Parent
run_training(DualDistilBert, df, tokenizer, 'level2', 'level2_dual')

# Level 3: Comment + Parent + Sentiment
run_training(DualDistilBertWithSentiment, df, tokenizer, 'level3', 'level3_dual_sent')
```

Các model sau khi huấn luyện sẽ được lưu dưới dạng:

```text
level1_distilbert.pt
level2_dual.pt
level3_dual_sent.pt
```

Trên Kaggle, mặc định model được lưu tại:

```python
/kaggle/working
```

### Bước 4: Đánh giá mô hình

Mở notebook:

```text
val-test.ipynb
```

Notebook này dùng để:

- Load các model đã huấn luyện.
- Đánh giá trên tập test.
- In Accuracy và F1-score.
- Vẽ confusion matrix, ROC curve.
- So sánh leaderboard ba mô hình.
- Chạy thử demo dự đoán với một cặp `parent_comment` và `comment`.
- Phân tích lỗi giữa các mô hình.

Trong notebook, biến đường dẫn model mặc định là:

```python
MODEL_DIR = "/kaggle/input/modelnlp"
```

Nếu chạy local hoặc model nằm ở thư mục khác, cần đổi lại:

```python
MODEL_DIR = "./models"
```

## 10. Demo dự đoán

Notebook `val-test.ipynb` có phần demo dự đoán một câu phản hồi dựa trên ngữ cảnh cha.

Ví dụ:

```text
Context: Doesn't make it a bad game.
Reply: Obviously if someone dislikes it it means it's a bad game
```

Kết quả dự đoán mẫu:

```text
Level 1 (Reply Only)        → Sarcastic
Level 2 (Dual Encoder)      → Sarcastic
Level 3 (Dual + Sentiment)  → Sarcastic
```

Mỗi mô hình trả về xác suất mỉa mai và nhãn dự đoán cuối cùng.

## 11. Phân tích lỗi

Qua quá trình phân tích lỗi, các nhóm lỗi chính bao gồm:

### 11.1 Nhiễu cảm xúc từ ngữ cảnh

Một số câu không mỉa mai nhưng chứa từ ngữ tiêu cực mạnh có thể khiến mô hình dự đoán nhầm là mỉa mai.

### 11.2 Mỉa mai ngầm cần kiến thức nền

Nhiều bình luận trên Reddit cần hiểu văn hóa cộng đồng, kiến thức xã hội hoặc ngữ cảnh ngoài văn bản. Các mô hình DistilBERT có thể gặp khó khăn trong nhóm này.

### 11.3 Nhầm giữa phê bình trực tiếp và mỉa mai

Một câu tiêu cực không nhất thiết là mỉa mai. Mô hình đôi khi bị nhầm khi gặp các từ như `waste`, `bad`, `stupid`, `generic`, v.v.

### 11.4 Câu quá ngắn hoặc ám chỉ

Các câu trả lời rất ngắn như meme, joke hoặc câu nói phi lý thường khó phân loại vì thiếu tín hiệu ngữ nghĩa rõ ràng.

## 12. Kết luận

Đồ án cho thấy việc khai thác ngữ cảnh hội thoại có vai trò quan trọng trong bài toán phát hiện mỉa mai. So với mô hình chỉ xét bình luận đơn lẻ, mô hình **Dual DistilBERT Encoder** sử dụng cả `comment` và `parent_comment` đạt kết quả tốt hơn về Accuracy và F1-score.

Kết quả chính:

- Mô hình baseline chỉ dùng comment đạt F1-score khoảng **0.7732**.
- Mô hình Dual Encoder đạt F1-score cao nhất khoảng **0.7811**.
- Việc thêm sentiment bằng TextBlob không giúp cải thiện đáng kể, thậm chí có thể gây nhiễu trong một số trường hợp.

Do đó, mô hình được đề xuất cuối cùng là:

```text
Level 2 - Dual DistilBERT Encoder
```

## 13. Hạn chế

- Dataset Reddit chứa nhiều tiếng lóng, meme, viết tắt, lỗi chính tả và ngữ cảnh cộng đồng khó hiểu.
- Cách gán nhãn dựa trên ký hiệu `/s` có thể gây nhiễu vì không phải mọi bình luận mỉa mai đều được đánh dấu.
- TextBlob là công cụ sentiment dựa nhiều vào luật, chưa đủ tinh tế cho ngôn ngữ mạng xã hội.
- Kiến trúc Dual Encoder hiện tại chỉ ghép biểu diễn ở tầng cuối, chưa cho phép token của comment tương tác trực tiếp với token của parent comment.
- Mô hình chưa khai thác tri thức nền hoặc thông tin ngoài văn bản.

## 14. Hướng phát triển

Một số hướng cải tiến trong tương lai:

- Thay TextBlob bằng mô hình sentiment dựa trên Transformer, ví dụ RoBERTa fine-tuned cho sentiment analysis.
- Thử kiến trúc Cross-Attention để comment và parent comment tương tác sâu hơn ở cấp token.
- So sánh với các mô hình mạnh hơn như BERT, RoBERTa, DeBERTa hoặc Longformer.
- Thử thêm đặc trưng subreddit, score, thời gian hoặc metadata người dùng.
- Xây dựng giao diện demo web bằng Streamlit hoặc Gradio.
- Mở rộng sang phát hiện mỉa mai đa phương thức, kết hợp văn bản và hình ảnh.

## 15. Lỗi thường gặp và cách xử lý

### 15.1 Không tìm thấy dataset

Lỗi thường gặp:

```text
File not found at /kaggle/input/cs213data/sarcasm.csv
```

Cách xử lý:

- Kiểm tra lại dataset đã được add vào Kaggle Notebook chưa.
- Đổi biến `DATA_PATH` hoặc `DF_PATH` sang đúng đường dẫn file CSV.

### 15.2 Không tìm thấy model weight

Lỗi thường gặp:

```text
FileNotFoundError: level2_dual.pt
```

Cách xử lý:

- Kiểm tra lại đã chạy huấn luyện và lưu model chưa.
- Kiểm tra biến `MODEL_DIR` trong `val-test.ipynb`.
- Đảm bảo tên file model trùng với notebook đánh giá.

### 15.3 CUDA out of memory

Cách xử lý:

- Giảm `BATCH_SIZE` từ 64 xuống 32 hoặc 16.
- Giảm `MAX_LEN` nếu chỉ muốn thử nghiệm nhanh.
- Chạy từng model riêng thay vì chạy liên tục cả ba model.
- Xóa cache GPU sau mỗi lần chạy:

```python
import torch, gc

gc.collect()
torch.cuda.empty_cache()
```

### 15.4 Tính sentiment quá lâu

Tính TextBlob sentiment cho hơn 1 triệu dòng có thể mất nhiều thời gian. Nếu chỉ chạy Level 1 hoặc Level 2, có thể bỏ qua bước tạo `sentiment_comment` và `sentiment_parent`.

## 16. Tài liệu tham khảo

- Khodak et al., *A Large Self-Annotated Corpus for Sarcasm*, 2018.
- Kaggle, *Sarcasm on Reddit*: `https://www.kaggle.com/datasets/danofer/sarcasm`
- Hugging Face, *DistilBERT Documentation*: `https://huggingface.co/docs/transformers/model_doc/distilbert`
- PyTorch Documentation: `https://pytorch.org/docs/stable/index.html`
- Hugging Face Transformers: `https://github.com/huggingface/transformers`

---

## Ghi chú

README này được viết để mô tả đầy đủ project ở mức có thể đưa lên GitHub. Khi nộp hoặc public repository, nên bổ sung thêm các thư mục sau nếu có:

```text
data/       # không nên commit file dữ liệu lớn, chỉ để hướng dẫn tải
models/     # chứa model weight nếu được phép chia sẻ
figures/    # chứa confusion matrix, ROC curve, training loss
```

Nếu repository public, nên thêm `.gitignore` để tránh đẩy dataset và model weight dung lượng lớn lên GitHub.
