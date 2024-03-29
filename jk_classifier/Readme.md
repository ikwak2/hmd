# Interval Sec 단위 실험
- 동일 모델 기준 실험결과 4sec 단위로 peak detection 하는 것이 가장 acc  가 뛰어났음. 
- 4sec 기준으로 피처 추출 예정

|  **sec** 	| **weighted accuracy** 	|  **cost** 	|  **roc**  	| **murmur acc** 	| **out acc** 	|  **th**  	|
|:--------:	|:---------------------:	|:---------:	|:---------:	|:--------------:	|:-----------:	|:--------:	|
|   1sec   	|         0.779         	|   13226   	|   0.759   	|      0.482     	|    0.959    	|   0.55   	|
|   2sec   	|         0.771         	|   13197   	|   0.763   	|      0.842     	|    0.949    	|   0.55   	|
|   3sec   	|         0.784         	|   11595   	|   0.766   	|      0.842     	|    0.796    	|    0.6   	|
| **4sec** 	|       **0.795**       	| **12807** 	| **0.772** 	|    **0.921**   	|  **0.663**  	| **0.45** 	|
|   5sec   	|         0.782         	|   14495   	|   0.767   	|      0.947     	|      1      	|   0.45   	|
|   6sec   	|         0.792         	|   13715   	|   0.771   	|      0.895     	|     0.98    	|    0.5   	|
|   7sec   	|         0.771         	|   11717   	|   0.757   	|      0.816     	|    0.929    	|    0.6   	|
|   8sec   	|         0.763         	|   13907   	|   0.767   	|      0.868     	|     0.98    	|    0.5   	|





# 최적 파라메터 적용 
 - 교수님께서 알려주신 최적 파라메터 적용.
 - 피처부분 수정하여 재추출 완료.
 - 제너레이터 적용 후 모델 학습 진행 됨.
 - Wav2vec2 에서 나온 차원이 기존 (374,32) 였는데..멜스펙토그램 뽑을 때 윈랭스 홉랭스 경우 참고하여.. 차원 순서 변경 하니... (32,374) 좀 더 학습이 잘되는 느낌적인 느낌(?)
# interval ( 피크 디텍션 ) & wav2vec2 Feature  ->  WAVENET  
 - 참고사이트 : https://velog.io/@changdaeoh/Convolutionforsequence
 - LSTM 을 사용하면 학습 시간이 많이 소요 되어서 Conv1D 로 적용하는 방법을 찾아보다가 블로그 발견하여 적용해봄.
 - dilated causal convolution 을 쌓아서 진행 . 몇개 블럭을 쌓을지는 여러가지 방법으로 시도 예정


####   WAVENET _Block 2개
```
threshold:  0.6 
#Murmur scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.771,0.617,0.568,0.848,0.814,17770.578

#Outcome scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.718,0.720,0.612,0.639,0.799,10562.444
```
#### WAVENET_Block 2개 & Wav2Vec2 피처 제외 
```
threshold:  0.55
#Murmur scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.765,0.620,0.582,0.864,0.811,18810.552

#Outcome scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.709,0.703,0.467,0.565,0.837,12676.239
```

#### WAVENET_Block 3개
```
#Murmur scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.765,0.607,0.533,0.806,0.792,16955.190

#Outcome scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.722,0.729,0.634,0.654,0.798,10356.678
```

#### WAVENET_Block 1개
```
threshold:  0.45
#Murmur scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.771,0.619,0.520,0.785,0.792,16353.014

#Outcome scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.688,0.701,0.374,0.529,0.846,14495.049
```

# wav2vec2 feature를 이미지 처리 -> mobilenetv2
 - 이미지 처리 하기 위해서 2차원 wav2vec2 feature에서 차원 축 추가
 - mobilenetv2에 맞게 이미지 리사이징 
 - 기존 모델에서 mobilenetv2 추가 후 학습 진행 
 - weighted accuracy 는 그렇게 높지 않지만, cost가 **9900** 으로 떨어짐. 
```
#Murmur scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.758,0.608,0.509,0.780,0.757,16548.052

#Outcome scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.734,0.744,0.644,0.670,0.837,9904.543
```

# Wav2Vec2 Feature 를 이미치 처리 -> ResNet50 V2 에 넣어봄.
- cost가 더 떨어질까봐 시도해봤는데 생각보다 많이 떨어지지는 않음.
```
threshold:  0.5
#Murmur scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.756,0.590,0.534,0.812,0.774,16940.728

#Outcome scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.687,0.664,0.525,0.565,0.748,11964.332
```

# Interval(1sec) & Wav2Vec2_Pretrained ->  LCNN
 - 생각보다 성능이 많이 향상되지는 않았음. Mel 을 제외하고 실험해볼예정

```
threshold:  0.45
#Murmur scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.767,0.623,0.506,0.764,0.782,16161.004

#Outcome scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.723,0.737,0.415,0.545,0.844,13697.681

#Murmur scores (per class)
Classes,Present,Unknown,Absent
AUROC,0.949,0.500,0.851
AUPRC,0.869,0.073,0.928
F-measure,0.679,0.000,0.840
Accuracy,0.947,0.000,0.791

#Outcome scores (per class)
Classes,Abnormal,Normal
AUROC,0.723,0.723
AUPRC,0.769,0.706
F-measure,0.690,0.139
Accuracy,0.990,0.075
```
# Interval(1sec) & Wav2Vec2_Pretrained & Except mel ->  LCNN
- 성능이 낮음.
- Trainable = False로 진행(메모리부족, 도커에서 True 해서 학습 진행중)
- Wav2Vec2  & Dense 추가하여서 실험해볼 예정
```
#Murmur scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.701,0.546,0.453,0.733,0.625,18189.426

#Outcome scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.670,0.676,0.476,0.555,0.799,12476.741
```
#Interval(1sec) $ Wav2Vec2_Trainable = True  -> LCNN
- true 로 했을 때 성능이 좀 더 뛰어나지기는 하지만 높은 수치는 아님
- 도커에서 Wav2vec2 &trainable True & Dense로 실험해볼 예정
```
threshold:  0.25
#Murmur scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.764,0.615,0.494,0.749,0.774,14731.785

#Outcome scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.743,0.751,0.339,0.513,0.840,15140.890

#Murmur scores (per class)
Classes,Present,Unknown,Absent
AUROC,0.942,0.500,0.849
AUPRC,0.853,0.073,0.918
F-measure,0.649,0.000,0.833
Accuracy,0.947,0.000,0.770
```

# Interval( 3sec/5sec/10sec 등) & Mel ->LCNN 시도예정
