# VISION
## Description:

This work implements “Multiview Contrastive Learning for Completely Blind Video
Quality Assessment of User Generated Content” in keras/tensorflow. If you are using the codes, cite the following article:

>Shankhanil Mitra and Rajiv Soundararajan. 2022. Multiview Contrastive Learning for Completely Blind Video Quality Assessment of User Generated Content. In Proceedings of the 30th ACM International Conference on Multimedia (MM '22). Association for Computing Machinery, New York, NY, USA, 1914–1924. https://doi.org/10.1145/3503161.3548064Shankhanil Mitra and Rajiv Soundararajan. 2022. Multiview Contrastive Learning for Completely Blind Video Quality Assessment of User Generated Content. In Proceedings of the 30th ACM International Conference on Multimedia (MM '22). Association for Computing Machinery, New York, NY, USA, 1914–1924. https://doi.org/10.1145/3503161.3548064

 
![VISION](https://github.com/Shankhanil006/VISION/blob/main/cmc_final%20(1).png?raw=true)

## Frame and Frame difference based Feature Encoder
We use the function dualData_contrastive_FvsFDiff.py to learn quality aware feature using frame and frame difference.

## Frame difference and Optical Flow based Feature Encoder
We use the function dualData_contrastive_FDiffvsOFlow.py to learn quality aware feature using optical flow and frame difference.
## Reference Feature Generator
Use the functions reference_feat_spatial.py, reference_feat_temporal.py, reference_feat_flow.py to generate reference features from pristine frames, frame difference, and optical flow map.

## Test Frame and Frame difference based network stream
Use the funtion test_spatiotemporal.py to predict the quality of videos with frame and frame difference based encoders.

## Test Frame difference and Optical Flow based network stream
Use the funtion test_flowtemporal.py to predict the quality of videos with optical flow and frame difference based encoders.

## **Prediction**:
Overall VISION index of videos are predicted using Quality_Estimator.py. Use the pre-trained model on LIVE-FB Large-Scale Social Video Quality for evaluateing VISION for any video.

## ** Pre-trained Models**:
[Google Drive link](https://drive.google.com/drive/folders/1TgWrM74Yo2Fg5kDnKRQJdKvytmQwWYvG?usp=sharing)

