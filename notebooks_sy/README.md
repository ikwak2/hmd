
## 파일 설명

* **model0_kernel_reduction.ipynb**  - toy model의 kernel size를 (3, 3)에서 (1, 1)로 축소 (6가지 조합의 feature (mel, cqt, stft)로 실험 진행)
* **model1_BC_ResMax.ipynb**  -  BC-ResMax 모형 적용 (6가지 조합의 feature (mel, cqt, stft)로 실험 진행)  
* **model2_LCNN_FFM_feature_tun.ipynb** - 교수님 LCNN 모형에 FFM feature tunning 진행  
* **model3_LCNN_rr_qrs.ipynb** - LCNN 모형에 rr interval(jk)과 qrs interval(sy) 변수 추가  
* **Heart Rate Variability (HRV) analysis.ipynb** -QRS Code 확인
* **Upgrading_models_LCNN_fix_RRinterval_w2v2.ipynb** - LCNN fix 모형 + RR interval + w2v2 적용 (RR interval 사용방법 포함) 
 
## Upgrading_models_LCNN_fix_RRinterval_w2v2_ext
```
threshold:  0.65
#Murmur scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.744,0.603,0.475,0.728,0.741,15377.795

#Outcome scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.690,0.697,0.630,0.649,0.789,10488.714
```


## Upgrading_models3_LCNN-cost_rr_qrs2_RR
```
threshold:  0.25
#Murmur scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.754,0.594,0.516,0.791,0.763,15283.424

#Outcome scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.684,0.670,0.532,0.560,0.712,12192.411
```
## Upgrading_models_LCNN_fix_RRinterval_w2v2
```
threshold:  0.5
#Murmur scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.762,0.613,0.543,0.822,0.779,17981.697

#Outcome scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.693,0.683,0.406,0.518,0.794,13683.005
```
## Upgrading_models3_LCNN-cost_rr_qrs2
```
threshold:  0.5
#Murmur scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.763,0.608,0.562,0.843,0.801,17140.771

#Outcome scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.740,0.724,0.649,0.660,0.532,16117.494
```



 
 
## LCNN fix
```
#Murmur scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.764,0.611,0.526,0.796,0.787,16753.292

#Outcome scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.713,0.707,0.557,0.602,0.801,11237.276
```
 
## LCNN fix + RR 
```
#Murmur scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.760,0.610,0.545,0.822,0.790,17566.761

#Outcome scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.716,0.712,0.589,0.607,0.734,11453.603
```
## W2V2
960h pre-train 모형이 조금 더 성능이 나아 보였음

## LCNN fix + RR + W2V2(ext =F)
(LCNN 모델에 적용, Model에서 w2v2 적용한 값 사용);Upgrading_models_LCNN_fix_RRinterval_w2v2
```
#Murmur scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.764,0.605,0.505,0.770,0.763,16138.476

#Outcome scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.669,0.676,0.559,0.581,0.719,11879.111
```
```
threshold:  0.5
#Murmur scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.760,0.610,0.555,0.832,0.795,17770.578

#Outcome scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.716,0.712,0.360,0.518,0.835,14704.473
```
## LCNN fix + RR + W2V2(ext = T) -> 학습중
Upgrading_models_LCNN_fix_RRinterval_w2v2_ext
```
```

## LCNN fix + RR + W2V2
(get_feature에서 w2v2 적용한 값 모델에 적용, 1D CNN); Upgrading_models_LCNN_fix_RRinterval_w2v2_copy1
```
#Murmur scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.762,0.612,0.530,0.806,0.760,16936.442

#Outcome scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.685,0.679,0.549,0.597,0.799,11350.013
```

## LCNN + qrs + w2v2 -> 학습중
현재 다른 모델보다 안정적으로 val_acc가 높음. mel, stft, cqt 모두 사용
```
```

## LCNN + qrs + RR -> 학습중


## 2022.08.09. 화
* LCNN_fix 파일 + qrs
* LCNN_fix 파일 + qrs seq : 너무 오랜 시간 소요됨
* LCNN_fix 파일 + wav2vec2 (lstm)
* LCNN_fix 파일 + wav2vec2 (1D CNN) : LSTM 보다는 성능이 좋았으나, 교수님께서 올려주신 LCNN보다 성능 큰 성능 향상은 없었음
```
threshold:  0.75
#Murmur scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.723,0.544,0.420,0.634,0.774,13963.775

#Outcome scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.662,0.665,0.629,0.634,0.695,11878.815
```

## QRS interval    
: wide한지 narrow한지 ECG 간격을 통해 부정맥 원인 부위를 짐작할 수 있다.  

![qrs](https://user-images.githubusercontent.com/54921677/180925312-3e1fa6ea-b30a-41e5-8f6a-c27fc3ff30d2.png)

[QRS interval](https://litfl.com/qrs-interval-ecg-library/)

1. Narrow   
![narrow](https://user-images.githubusercontent.com/54921677/180925865-62e10cc7-374e-4703-a565-909695710ca5.PNG)   

2. Broad   
![Broad](https://user-images.githubusercontent.com/54921677/180925869-b14626c3-67d6-4a52-9654-e79ae96b66bd.PNG)   
