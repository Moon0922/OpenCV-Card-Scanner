# Card Scanner
### A card scanner built in Python using OpenCV
### Here are some examples of images before and after scan:
<img src="sample/2.jpg" height="450"><br>
<img src="result/2_card1.jpg" height="200">
<img src="result/2_card2.jpg" height="200">
<img src="result/2_card3.jpg" height="200">
<br>
<img src="sample/345.jpg" height="450">
<br>
<img src="result/345_card1.jpg" height="200">
<img src="result/345_card2.jpg" height="200">
<img src="result/345_card3.jpg" height="200">
<img src="result/345_card4.jpg" height="200">
<img src="result/345_card5.jpg" height="200">
<img src="result/345_card6.jpg" height="200">
<img src="result/345_card7.jpg" height="200">
<img src="result/345_card8.jpg" height="200">
<img src="result/345_card9.jpg" height="200">
<br>
<img src="sample/3523.jpg" height="450"><br>
<img src="result/3523_card1.jpg" height="200">
<img src="result/3523_card2.jpg" height="200">
<img src="result/3523_card3.jpg" height="200">

### Usage
```
python main.py (--images <IMG_DIR> | --image <IMG_PATH>) [-i]
```
* to scan a image:
```
python main.py --image sample/2.jpg
```
* Alternatively, to scan all images in a directory without any input:
```
python main.py --images sample
