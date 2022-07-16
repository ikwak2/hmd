## 파일 설명

- EDA.ipynb: 파일 탐색, 음성신호 시각화 및 피쳐 추출 함수 제작
- imputing_hw.ipynb: 키, 몸무게를 imputation 한다면 어떤 전략이 좋을지? 근데 중요한 변수가 아닌 것 같아서 안해도 상관없을듯.. (아직 적용 안함)
- model construction iy.ipynb: 초기 모형 제작
- model_iy2.ipynb, model_iy3.ipynb: 본 대회 모형 제작 관련
- Upgrading_models1.ipynb: toy 모형을 이용한 초기모형 실험
- Upgrading_models2_adding_features.ipynb: cqt, stft 피쳐 추가 실험
- Upgrading_models3_LCNN.ipynb: melspec 피쳐에 LCNN 모형 실험
- Upgrading_models4_ResMax.ipynb: melspec 피쳐에 ResMax 모형 실험
- Upgrading_models5_DDWS.ipynb: melspec 피쳐에 DDWS 모형 실험, DDWS, BC-ResNet 은 생각보다 잘 되지는 않았는데.. 실험 필요?
- Upgrading_models6_ordinal_regression.ipynb: melspec 으로 ordinal coding 실험.. 좋은 것 같기도 하고.. 비슷한 것 같기도 함. evaluation set의 unknown class 분포도 시각화해 본 결과 지금 학습시킨 모형들로는 unknwon 그룹에 대한 모형화가 잘 안되고 있는 것으로 파악됨. 차라리 unknown 그룹 없이 하는 모형이 결과가 더 좋았음 
- Upgrading_models3_LCNN-cost_sensitive learning.ipynb: LCNN 버전에 cost sensitive learning 방식으로 murmur그룹에 가중치 더 줘서 트레이닝.. 대회 메트릭 기준으로는 성능이 더 향상됨
- run4hp.py: 여러 파라메터 시뮬레이션 해보는 용도 코드. 문제점: cqt, stft 안써도 다 뽑아서 너무 느림.. 수정필요함, 기타 버그들 존재
- run4hp2.py: 자잘한 버그 수정, 피쳐 필요한 것만 뽑도록 수정함. 속도 개선


