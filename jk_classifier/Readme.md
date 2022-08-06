# 최적 파라메터 적용 
 - 교수님께서 알려주신 최적 파라메터 적용.
 - 피처부분 수정하여 재추출 완료.
 - 제너레이터 적용 후 모델 학습 진행 됨.
 - Wav2vec2 에서 나온 차원이 기존 (374,32) 였는데..멜스펙토그램 뽑을 때 윈랭스 홉랭스 참고하여 차원 바꾸니.. (32,374) 좀 더 학습이 잘되는 느낌적인 느낌(?)
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
#### WAVENET_Block 1개
```
#Murmur scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.771,0.619,0.520,0.785,0.792,16353.014

#Outcome scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.688,0.701,0.374,0.529,0.846,14495.049
```
