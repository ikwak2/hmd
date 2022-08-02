# Peak _ Detection 
 - 교수님께서 알려주신 사이트 https://github.com/kevinmgamboa/S1-S1-Phonocardiogram-Peak-Detection-Method-in-Python 에서 나온 방법을 참고하여 Detection 진행해봄
 - 쥬피터노트북 파일 Peak_Detection_연습 파일 확인. 실제로 잘 잡히는 듯함.
 - 하이퍼파라메터는 Min_distance 인것 같고 기본값은 min_distance =2000 인데 1000 값으로 바꾸니 실제로 좀 피크 디텍션 잘하는듯함.
 - 깃허브 클론 말고도 또 Peakutils 를 설치해야함.https://bitbucket.org/lucashnegri/peakutils/src/master/ 참고
# 모델 결과
 - Peak Detection 한 후 interval 계산. 각 파일마다 길이가 달라 길이 맞춰서  array 형태로 인풋 진행.
 - Interval은 LSTM 층 쌓아서 모델 학습
 - 기존보다 성능이 조금 나아지긴함. (Wetighted accuracy = 0.787)
 - 평균을 내서 스칼라 값으로 진행해보는것, Con1d-Lstm  등 다른 학습모델 트레이닝 시도 진행 예정
