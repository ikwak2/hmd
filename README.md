# Heart murmur detection
Participating [heart murmur detection challenge](https://moody-challenge.physionet.org/)

## example [code package](https://github.com/physionetchallenges/python-classifier-2022)

## evaluation [code package](https://github.com/physionetchallenges/evaluation-2022)

## 진행 아이디어들 (뭘 해야할지 잘 모르겠는 학생들은 아래 아이디어 주제들 중 하나 잡고 진행해봐도 좋음)

- Toy 모형 외에 다른 모형들 시도해보기: ResMax, BC-ResMax, DDWS, 등등등 
  - 주: 데이터가 많지 않기때문에 이미지 대용량 데이터 모형들(ex. ResNet50, 100, EfficientNet 등등) 은 overfitting으로 잘 안될 가능성 높음. 그래도 mixup, ffm, specaug 등등과 함께 쓰면 트레이닝 가능할지도? 트레이닝 가능하다고 해도, 후반 레이어들은 필터를 (1,1) 쓰는 것을 권장함 (Receptive field 문제 때문에) 
- log-melspectrogram 외에 추가 피쳐 고려해보기: CQT, Raw-audio feature 등등(Done, Upgrading models2.ipynb)
- 지금 피쳐 파라메터 조절해보며 피쳐 파라메터 최적화 해보기
- murmur probability 2개로 나눠서 중간부분 unknown 으로 모형화 (ordinal regression도 공부해보고 이거저거 해봐야할듯?)
- Threshold 조절해서 스코어 최적화 (Done, Controlling thresholds.ipynb)
- pytorch 진행하는 학생들: 1. AAIST (raw audio 피쳐로 graph neural net 쓰는 모형) 시도해 봐도 좋을듯 2. AAIST 임베딩과 다른 모형 임베딩 합쳐서 2단 트레이닝으로 구성해서 앙상블모형 작성해봐도 좋을듯 (이건 tensorflow로 푸는 쪽도 고려해볼만함)

## Submissions (6/10)

- LCNN using melspectrogram with mixup, FFM RP augmentation
```
#Murmur scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.772,0.627,0.526,0.806,*0.771*,16532.976
#Outcome scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost
0.690,0.696,0.454,0.555,0.827,*12885.663*

# Leader board scores
Murmur, outcome
0.723, 10022
```

- Wav2vec2 - 태인학생 모형
```
# Leader board scores  ## 이유는 모르겠지만 두번 평가된것같음
Murmur, outcome
0.709, 9903
0.668, 10290.577
```

- LCNN using melspectrogram with mixup, FFM RP augmentation, RR, wav2vec2 - 정국학생 모형

```
Murmur, outcome
0.747, 10200
# Leader board scores  
Murmur, outcome
?, ?
```

- LCNN using melspectrogram with mixup, cutout augmentation - 최적화 버전한번 더 내 봄

```
Murmur, outcome
0.790, 11250
# Leader board scores 
Murmur, outcome
?, ?
```

- LCNN using melspectrogram with mixup, cutout augmentation, RR 평균 추가 

```
Murmur, outcome
0.810, 11636
# Leader board scores  
Murmur, outcome
?, ?
```

