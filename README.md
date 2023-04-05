# Big Data and Application

Đồ án cuối kỳ môn học Dữ Liệu Lớn và Ứng Dụng (UEH)

# Tổng quan dự án
1. Thực hiện crawl dữ liệu từ các playlist trên Spotify thông qua API của Spotify.
2. EDA (Exploratory Data Analysis).
2. Tiền xử lý dữ liệu: Chỉnh dạng dữ liệu, thêm dữ liệu cần thiết. 
5. Xây dựng hệ thống đề xuất bài hát dựa trên phương pháp phân cụm K-Means.
6. Xây dựng hệ thống đề xuất bài hát dựa trên mô hình hồi quy (Linear Regression, Random Forest Regressor), mô hình Keras Neural Network.
7. Xây dựng hệ thống đề xuất bài hát dựa trên Cosine Similarity 
8. Machine Learning: Dự đoán xem user có thích bài hát đó hay không dựa trên các mô hình phân lớp (Decision Tree, SVM, KNN, AdaBoost, Random Forest, Gradient Boosting)

# Tổng quan dữ liệu 
Dữ liệu thu thập được gồm có: 9901 dòng và 23 cột
```
* id: ID của bài hát 
* title: Tên bài hát
* all_artists: Nghệ sĩ trình diễn bản nhạc
* album_name: Tên album chứa bản nhạc
* popularity: Loại biến rời rạc, giá trị từ 0 đến 100, mô tả mức độ phổ biến của bài hát, càng cao càng phổ biến
* release_date: Ngày phát hành
* explicit: Cho biết bài hát có ẩn chứa từ ngữ, nội dung nhảy cảm liên quan đến tình dục, chém giết, chính trị tôn giáo,… hay không. Giá trị “TRUE” là “Có” và giá trị “FALSE” là “Không”
* speechiness: Xác suất thể hiện có sự hiện diện của tiếng nói (khác với tiếng hát) trong bài hát hay không. Có giá trị trong khoảng 0.0 - 1.0, giá trị càng cao thì đây khả năng cao là bài diễn thuyết, sách nói,...
* danceability: Mô tả bài hát có phù hợp để nhảy hay không dựa trên sự kết hợp, độ mạnh yếu của nhịp độ, nhịp điệu bài hát, có giá trị từ 0 - 1.0
* energy: Mô tả độ dồn dập của bài hát, có giá trị từ 0.0 - 1.0, càng gần 1.0 thì bài hát càng tạo cảm giác dồn dập, dồn nén cao và ngược lại
* key : Cao độ trung bình của hát, giá trị là số nguyên, tính theo chuẩn Pitch Class Notation, nếu không có cao độ giá trị là -1
* loudness: Độ to trung bình của bài hát (tính theo đơn vị dB), giá trị thường rơi vào khoảng -60.0 - 0.0 dB
* mode: Loại biến rời rạc, mô tả điệu thức của bài hát, chỉ có hai giá trị là 0 (giọng Thứ) và 1 (giọng Trưởng)
* acousticness: Cường độ Acoustic của bài hát hay nói cách khác là xác suất bài hát này có tính chất acoustic, có giá trị từ 0.0 - 1.0
* instrumentalness: Độ đo thể hiện tính instrumental (không lời) của bài hát, có giá trị trong khoảng 0.0 - 1.0, giá trị càng cao thì càng ít giọng hát trong bài hát
* liveness: Độ đo thể hiện tính live (nhạc sống, có sự hiện diện của khán giả trong lúc thu âm) của bài hát, có giá trị trong khoảng 0.0 - 1.0
* valence: Biểu thị tính tích cực của bài hát, giá trị càng cao thì bài hát càng có tính chất tích cực (vui, phấn khởi), càng thấp thì bài hát càng buồn
* tempo: Mô tả tốc độ của bài hát, càng cao chứng tỏ bài đó có nhịp càng nhanh và ngược lại
* duration_ms: Thời lượng bài hát 
* time_signature: Số chỉ nhịp của bài
* artist_pop: Độ phổ biến của ca sĩ
* genres_list: List thể loại nhạc của bài hát
* genres: Thể loại của bài hát
```
| Tên cột         | Ý nghĩa       |
| -------------   | ------------- |
|id               |ID của bài hát |
|title            |Tên bài hát|
|all_artists      |Nghệ sĩ trình diễn bản nhạc|
|album_name       |Tên album chứa bản nhạc|
|popularity       |Loại biến rời rạc, giá trị từ 0 đến 100, mô tả mức độ phổ biến của bài hát, càng cao càng phổ biến|
|release_date     |Ngày phát hành|
|explicit         |Cho biết bài hát có ẩn chứa từ ngữ, nội dung nhảy cảm liên quan đến tình dục, chém giết, chính trị tôn giáo,… hay không. Giá trị “TRUE” là “Có” và giá trị “FALSE” là “Không”|
|speechiness      |Xác suất thể hiện có sự hiện diện của tiếng nói (khác với tiếng hát) trong bài hát hay không. Có giá trị trong khoảng 0.0 - 1.0, giá trị càng cao thì đây khả năng cao là bài diễn thuyết, sách nói,...|
|danceability     |Mô tả bài hát có phù hợp để nhảy hay không dựa trên sự kết hợp, độ mạnh yếu của nhịp độ, nhịp điệu bài hát, có giá trị từ 0 - 1.0|
|energy           |Mô tả độ dồn dập của bài hát, có giá trị từ 0.0 - 1.0, càng gần 1.0 thì bài hát càng tạo cảm giác dồn dập, dồn nén cao và ngược lại|
|key              |Cao độ trung bình của hát, giá trị là số nguyên, tính theo chuẩn Pitch Class Notation, nếu không có cao độ giá trị là -1|
|loudness         |Độ to trung bình của bài hát (tính theo đơn vị dB), giá trị thường rơi vào khoảng -60.0 - 0.0 dB|
|mode             |Loại biến rời rạc, mô tả điệu thức của bài hát, chỉ có hai giá trị là 0 (giọng Thứ) và 1 (giọng Trưởng)|
|acousticness     |Cường độ Acoustic của bài hát hay nói cách khác là xác suất bài hát này có tính chất acoustic, có giá trị từ 0.0 - 1.0|
|instrumentalness | Độ đo thể hiện tính instrumental (không lời) của bài hát, có giá trị trong khoảng 0.0 - 1.0, giá trị càng cao thì càng ít giọng hát trong bài hát|
|liveness         |Độ đo thể hiện tính live (nhạc sống, có sự hiện diện của khán giả trong lúc thu âm) của bài hát, có giá trị trong khoảng 0.0 - 1.0|
|valence          |Biểu thị tính tích cực của bài hát, giá trị càng cao thì bài hát càng có tính chất tích cực (vui, phấn khởi), càng thấp thì bài hát càng buồn|
|tempo            |Mô tả tốc độ của bài hát, càng cao chứng tỏ bài đó có nhịp càng nhanh và ngược lại|
|duration_ms      |Thời lượng bài hát|
|time_signature   | Số chỉ nhịp của bài|
|artist_pop       |Độ phổ biến của ca sĩ|
|genres_list      |List thể loại nhạc của bài hát|
|genres           |Thể loại của bài hát|
