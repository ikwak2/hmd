
## 파일 설명

* **model0_kernel_reduction.ipynb**  - toy model의 kernel size를 (3, 3)에서 (1, 1)로 축소 (6가지 조합의 feature (mel, cqt, stft)로 실험 진행)
* **model1_BC_ResMax.ipynb**  -  BC-ResMax 모형 적용 (6가지 조합의 feature (mel, cqt, stft)로 실험 진행)  
* **model2_LCNN_FFM_feature_tun.ipynb** - 교수님 LCNN 모형에 FFM feature tunning 진행  
* **model3_LCNN_rr_qrs.ipynb** - LCNN 모형에 rr interval(jk)과 qrs interval(sy) 변수 추가  
* **Heart Rate Variability (HRV) analysis.ipynb** -QRS Code 확인
***
## 2022.08.09. 화
* LCNN_fix 파일 + qrs
* LCNN_fix 파일 + qrs seq : 너무 오랜 시간 소요됨
* LCNN_fix 파일 + wav2vac2 (lstm)
* LCNN_fix 파일 + wav2vac2 (1D CNN) : LSTM 보다는 성능이 좋았으나, 교수님께서 올려주신 LCNN보다 성능 큰 성능 향상은 없었음
```
threshold:  0.75
#Murmur scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.723,0.544,0.420,0.634,0.734,13963.775

#Outcome scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.662,0.665,0.629,0.634,0.695,11878.815
```


> **QRS interval**    
: wide한지 narrow한지 ECG 간격을 통해 부정맥 원인 부위를 짐작할 수 있다.  

![qrs](https://user-images.githubusercontent.com/54921677/180925312-3e1fa6ea-b30a-41e5-8f6a-c27fc3ff30d2.png)

[QRS interval](https://litfl.com/qrs-interval-ecg-library/)

1. Narrow   
![narrow](https://user-images.githubusercontent.com/54921677/180925865-62e10cc7-374e-4703-a565-909695710ca5.PNG)   

2. Broad   
![Broad](https://user-images.githubusercontent.com/54921677/180925869-b14626c3-67d6-4a52-9654-e79ae96b66bd.PNG)   
