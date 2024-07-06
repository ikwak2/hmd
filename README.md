# Participation of [Heart murmur detection challenge (2022 PhysioNet Challenge)](https://moody-challenge.physionet.org/)
This repository contains the code from our team's participation in the 2022 heart murmur detection competition, as well as the code related to the extended work published in our subsequent paper, "SpectroHeart: A Deep Neural Network Approach to Heart Murmur Detection Using Spectrogram and Peak Interval Features." 

## example (baseline code from the challenge) [code package](https://github.com/physionetchallenges/python-classifier-2022)

## evaluation code from the challenge [code package](https://github.com/physionetchallenges/evaluation-2022)

## Our Submission records from the challenge (10/10)

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

- Wav2vec2 - Taein's model
```
# Leader board scores  
Murmur, outcome
0.668, 10290.577
```

- 뭔지 모르겠는데 섞인 것 같은 스코어
```
0.709, 9903
```

- LCNN using melspectrogram with mixup, FFM RP augmentation, RR, wav2vec2 - Junguk's Model

```
Murmur, outcome
0.747, 10200
# Leader board scores  
Murmur, outcome
0.709, 9920.278
```

- LCNN using melspectrogram with mixup, cutout augmentation - refined one

```
Murmur, outcome
0.790, 11250
# Leader board scores 
Murmur, outcome
0.727, 11167.537
```

- LCNN using melspectrogram with mixup, cutout augmentation, PI feature added. (Our final submission)

```
Murmur, outcome
0.810, 11636
# Leader board scores  
Murmur, outcome
0.734, 9493.204 
```

- LCNN using melspectrogram with mixup, cutout augmentation, Another version with PI mean feature. 

```
Murmur, outcome
0.774, 11736
# Leader board scores  
Murmur, outcome
0.716, 10056.078
```

- LCNN using melspectrogram with mixup, cutout augmentation, Another version with PI mean feature. 

```
# Leader board scores  
Murmur, outcome
0.727, 10627.094
```

- LCNN using melspectrogram with mixup, cutout augmentation, Another version with PI mean feature. 

```
# Leader board scores  
Murmur, outcome
0.706, 12253.91
```

- LCNN using melspectrogram with mixup, FFM RP augmentation, PI mean, wav2vec2 - Junguk's model

```
# Leader board scores  
Murmur, outcome
0.723, 9715.077
```

## [Results for the 2022 challenge](https://moody-challenge.physionet.org/2022/results/) 

## [Source codes](https://physionet.org/static/published-projects/challenge-2022/1.0.0/sources/) 


