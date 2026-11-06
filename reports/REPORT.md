# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 6/11/2026

**Runtime Colab:** CPU/GPU T4

**Python / PyTorch / Ultralytics:** Python

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> \\\\\\\*\\\\\\\*Cách dùng mẫu này:\\\\\\\*\\\\\\\* copy/điền trên máy, lưu thành `REPORT.md`, rồi \\\\\\\*\\\\\\\*upload\\\\\\\*\\\\\\\* vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1\\\\\\\_lab\\\\\\\_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp \\\\\\\*\\\\\\\*link repo\\\\\\\*\\\\\\\* trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

|Từ|Nghĩa ngắn|
|-|-|
|Prediction|Kết quả model vừa đoán ra; chưa phải nhãn đúng|
|Ground truth|Nhãn do người gán theo quy tắc|
|Taxonomy|Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80)|
|Checkpoint|File model `.pt` đã huấn luyện sẵn|
|Guideline|Quy tắc gán nhãn do người viết|
|Annotator|Người gán nhãn|
|Reviewer / QC|Người kiểm tra nhãn|
|Escalation|Ca khó → hỏi Lab Coach, không tự bịa|
|Model score|Số “tin” của model; không phải điểm chất lượng nhãn người|

## 1\. Phân loại ảnh

**Mở file:** `day1\\\\\\\_lab\\\\\\\_outputs/classification\\\\\\\_predictions.json` và `visuals/classification\\\\\\\_top5.png`.  
Chỉ dùng ảnh `sample\\\\\\\_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class\\\\\\\_id`=468 ; `class\\\\\\\_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy\\\\\\\_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
* Record này nói về cả bức ảnh vì mô hình thực hiện tác vụ phân loại nội dung của toàn bộ ảnh và trong Json cũng không có thông tin về tọa độ của bounding box hay vị trí cụ thể của một vật thể
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
* Class list không phải do model tự nghĩ ra khi dự đoán, mà đã được xác định theo taxonomy đi kèm checkpoint đã huấn luyện. Mô hình sẽ dự đoán ảnh thuộc lớp nào trong danh sách các lớp đã có sẵn.
4. Vì sao cần ghi cả `class\\\\\\\_id`, `class\\\\\\\_name` và `taxonomy\\\\\\\_name`, không chỉ ghi mỗi tên lớp?
* Cần ghi đầy đủ cả 3 thông tin để xác định chuẩn xác các nhãn và tránh nhầm lẫn khi sử dụng các mô hình và bộ dữ liệu khác nhau
--> ví dụ: nếu dùng một taxonomy khác mà cùng 1 id là 468 (record trên câu hỏi 1) thì sẽ mang một ý nghĩa khác.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
* Guidline cần quy định rõ cách chọn object hoặc nội dung của ảnh, ví dụ là ưu tiên những object lớn nổi bật hoặc có đặc điểm nào đó phù hợp với mục tiêu của bộ dữ liệu
--> ví dụ: trong ảnh traffic có nhiều loại xe, nếu guidline yêu cầu chọn vật thể lớn thì sẽ ưu tiên chọn xe bus hoặc nếu guidline yêu cầu phương tiện xuất hiện nhiều nhất thì cần đếm số lượng trước khi chọn nhãn
6. Vì sao không được lấy `score` của model làm ground truth?
* Score chỉ thể hiện mức độ tin cậy của mô hình với kết quả dự đoán chứ không chứng minh kết quả đó chính xác, ground truth nên được xác định dựa trên ảnh thực tế, taxonomy và guidline gán nhãn, sau đó phải được kiểm tra chất lượng để đảm bảo yêu cầu

## 2\. Phát hiện vật thể

**Mở file:** `day1\\\\\\\_lab\\\\\\\_outputs/detection\\\\\\\_predictions.json` và `visuals/detection\\\\\\\_predictions.png`.  
Chỉ dùng ảnh `sample\\\\\\\_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class\\\\\\\_name`=person ; `score`=0.912625 ; `bbox\\\\\\\_xyxy`=\[385.33, 69.24, 498.92, 348.92] ; `bbox\\\\\\\_width`=113.58 ; `bbox\\\\\\\_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
* Box là khung chữ nhật mà mô hình dự đoán để xác định người trong picture:

  * Góc trên bên trái nằm ở tọa độ (385.33, 69.24).
  * Góc dưới bên phải nằm ở tọa độ (498.92, 348.92).
  * Chiều rộng của box là 113.58 pixel.
  * Chiều cao của box là 279.68 pixel.
--> Tọa độ được tính từ góc trên bên trái của ảnh, trục X tăng từ trái sang phải và trục Y tăng từ trên xuống dưới.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
* Chọn ngưỡng: 0.35 và 0.60
* Có 11 record với sample là kitchen, từ đó có thể tính với mức ngưỡng là 0.35 -> 11 prediction(tất cả có score >= 0.35), và mức ngưỡng 0.60 -> 6 prediction(có 6 record có score >= 0.60)
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
* Khi hạ ngưỡng sẽ giữ lại nhiều prediction hơn kể cả những prediction có do tin cậy thấp. Do đó reviewer cần kiểm tra bounding box để phát hiện những trường hợp mô hình nhận diện sai vật thể
--> ví dụ: Trong picture kitchen, khi sử dung ngưỡng 0.35, mô hình giữ lại 11 prediction trong đó có 1 record với score là 0.381542, do điểm tin cậy thấp nên reviewer phải kiểm tra xem có đúng vật thể không hay là model nhận diện nhầm
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
* Khi gán nhẫn cho một vật thể, người gán nhãn cần vẽ bounding box sao cho khung hình chữ nhật ôm sát vật thể nhất có thể, không lấy quá nhiều phần nền xung quanh và dính vật thể khác
--> ví dụ: khi phát hiện một người (person) trong ảnh kitchen, annotator cần vẽ box sát bao quanh người đó, sao cho các cạnh của box nằm sát phạm vi vật thể theo quy định của guidline
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
* Trong ảnh kitchen, có 1 lò nướng(oven) có tọa độ \[0.11, 187.54, 195.96, 292.81]. Vì tọa độ x đầu tiên gần bằng 0 nên bounding box nằm sát mép trái của ảnh, có khả năng sẽ khiến lò nướng bị cắt mất 1 phần --> trường hợp này nên viết thành quy tắc trong guideline: Khi vật thể bị cắt ở mép ảnh nhưng vẫn nhận diện được, annotator chỉ vẽ bounding box bao quanh phần vật thể nằm trong ảnh, không vẽ ra ngoài phạm vi ảnh. Trong trường hợp vật thể bị cắt quá nhiều, không xác định được là vật gì thì hỏi Lab Coach

## 3\. Phân đoạn theo từng đối tượng

**Mở file:** `day1\\\\\\\_lab\\\\\\\_outputs/segmentation\\\\\\\_predictions.json` và `visuals/segmentation\\\\\\\_prediction.png`.  
Chỉ dùng ảnh `sample\\\\\\\_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance\\\\\\\_id`= kitchen-001 ; `class\\\\\\\_name`= person ; `score`= 0.899318 ; số điểm polygon=`polygon\\\\\\\_point\\\\\\\_count`=348 ; 3–5 điểm đầu của `polygon\\\\\\\_xy`=\[\[446.0, 70.0], \[445.0, 71.0], \[444.0, 71.0], \[443.0, 72.0], \[442.0, 72.0]]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
* Bounding box chỉ là một khung hình chữ nhật bao quanh vật thể nên không thể hiện rõ hình dạng thực tế. Trong khi đó, polygon sử dụng nhiều điểm tọa độ nối với nhau để tạo đường biên theo hình dạng vật thể
--> Ví dụ: trong ảnh kitchen, đối tượng person được tô màu xanh. Polygon có thể mô tả đường viền quanh đầu, vai, hai tay và chân của người đó, thay vì chỉ tạo một khung chữ nhật bao quanh
3. `instance\\\\\\\_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
* Instance\_id dùng để phân biệt từng đối tượng riêng biệt được phát hiện trong một ảnh, kể cả khi các đối tượng thuộc cùng một lớp.
--> Ví dụ:

  * class\_id = 0: Là mã lớp người (person) trong COCO-80.
  * instance\_id = kitchen-001: Dùng để xác định một người cụ thể được phát hiện trong ảnh kitchen.
  * Nếu có thêm người khác, người đó sẽ có một instance\_id khác
* instance\_id cũng không phải tracking\_id vì tracking ID được dùng để theo dõi cùng một đối tượng qua nhiều khung hình của video, còn instance ID trong bài này chỉ xác định đối tượng trong một ảnh
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
* Khi vẽ mask, annotator cần vẽ đường biên bám sát phần vật thể nhìn thấy được, không lấy thêm vùng nền hoặc những vật thể khác xung quanh. Quy tắc này giúp mask thể hiện đúng hình dạng vật thể và giữ tính thống nhất khi gán nhãn cho nhiều ảnh khác nhau
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
* Trong ảnh kitchen, các chiếc bát (bowl) ở phía dưới bên trái ảnh được đặt gần nhau, có chỗ bị che khuất bởi những vật dụng khác trên bàn. Vì vậy, việc xác định đường biên riêng của từng chiếc bát có thể gặp khó khăn do đó
--> Ví dụ: nếu một chiếc bát bị chiếc bát khác che mất một phần, em chỉ vẽ mask cho phần chiếc bát đang nhìn thấy và vẫn tách riêng hai đối tượng nếu có thể xác định được
* Nếu phần bị che quá lớn hoặc không xác định được ranh giới giữa các vật thể thì chọn escalation để tránh tự suy đoán làm sai nhãn

## 4\. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

|Tác vụ|Ground truth trông như thế nào? (đơn vị nhãn)|Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON|Annotator làm gì?|Reviewer xem gì?|
|-|-|-|-|-|
|Phân loại ảnh|Một label class cho toàn bộ ảnh|ảnh traffic có nhiều loại xe nhưng mô hình dự đoán cab với score 0.510915, khó xác định nhãn đại diện cho cả ảnh|quan sát ảnh và chọn một nhãn cho guidline|Kiểm tra nhãn có đúng nội dung ảnh và tuân theo guideline không|
|Phát hiện vật thể|Tên lớp và bounding box cho từng vật thể|Trong ảnh kitchen, mô hình dự đoán cup với score 0.381542. Điểm thấp nên cần kiểm tra xem có nhận diện nhầm vật thể hay không|Xác định từng vật thể, gán tên lớp và vẽ box bao quanh|Kiểm tra tên lớp, vị trí box, vật thể bị bỏ sót và box trùng nhau|
|Instance segmentation|Tên lớp và mask/polygon cho từng vật thể riêng biệt|Ảnh kitchen có nhiều bát nằm gần nhau ở phía dưới bên trái, một số phần bị che khuất nên khó xác định chính xác đường biên|Gán nhãn từng vật thể và vẽ mask theo phần nhìn thấy được|Kiểm tra đường biên mask, phần nền bị tô nhầm và các vật thể có được tách riêng không|

## 5\. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
* Chỉ sử dung hình ảnh và dữ lieu đúng phạm vi bài lab, không đưa thông tin cá nhân hoặc dữ lieu nhạy cảm và báo cáo, file json hay github
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
* Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi bài Lab, em sẽ dừng xử lý và báo ngay cho Lab Coach để được kiểm tra và hướng dẫn cách xử lý, không tự ý sử dụng hoặc chia sẻ dữ liệu đó.

## 6\. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

* \[ ] `REPORT.md` (file này đã điền)
* \[ ] `day1\\\\\\\_lab\\\\\\\_outputs/classification\\\\\\\_predictions.json`
* \[ ] `day1\\\\\\\_lab\\\\\\\_outputs/detection\\\\\\\_predictions.json`
* \[ ] `day1\\\\\\\_lab\\\\\\\_outputs/segmentation\\\\\\\_predictions.json`
* \[ ] `day1\\\\\\\_lab\\\\\\\_outputs/IMAGE\\\\\\\_ATTRIBUTION.md`
* \[ ] `day1\\\\\\\_lab\\\\\\\_outputs/visuals/classification\\\\\\\_top5.png`
* \[ ] `day1\\\\\\\_lab\\\\\\\_outputs/visuals/detection\\\\\\\_predictions.png`
* \[ ] `day1\\\\\\\_lab\\\\\\\_outputs/visuals/segmentation\\\\\\\_prediction.png`
* \[ ] Ô validation cuối notebook báo `PASS`
* \[ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
* \[ ] Đã nộp **link repository** trên LMS

