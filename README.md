# A2N   
## Blog
[zhihu](https://zhuanlan.zhihu.com/p/393586665 "Brand·R")<br>
## architecture
paper：https://arxiv.org/abs/2104.09497<br>
![image](https://github.com/REN-HT/A2N/blob/main/images/A2N.jpg)<br>  
![image](https://github.com/REN-HT/A2N/blob/main/images/A2B.jpg)<br>
## implementation
* train.py for train，you can run the file alone<br>
* main.py for test，you can run the file alone<br>
* change the path when you try to load data<br>
* you need to change some parameters in AAN.py and DataSet.py for different scale<br>
## train
![image](https://github.com/REN-HT/A2N/blob/main/images/aan_L1_2x_400.jpg)<br>
validation set: select 5 images from div2k 100 validation set, then clipping them to 25 images<br>
## result
|        |    Set5(PSNR/SSIM)    |    Set14(PSNR/SSIM)    |
|--------|-----------------------|------------------------|
| 2x     |     38.088/0.9610     |      33.837/0.9197     |
| 4x     |     32.269/0.8959     |      28.750/0.7868     |
## display
### 2x
![image](https://github.com/REN-HT/A2N/blob/main/images/2x.png)<br>
### 4x
![image](https://github.com/REN-HT/A2N/blob/main/images/4x.png)<br>
