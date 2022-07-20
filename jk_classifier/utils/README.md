## 편의를 위해 제작된 util 함수들

- get_feature.py : 피쳐 추출 함수들
   - feature_extract_melspec(fnm, samp_sec=20, sr = 4000, pre_emphasis = 0, hop_length=256, win_length = 512, n_mels = 100) : fnm 파일이름, samp_sec: 샘플링하는 초, sr: sampling rate, pre_emphasis: 0이면 적용 안하고 0.97 정도로 적용해보는 실험도 필요할 듯. 노이즈 제거한다는 효과가 있다고 알려져있는데 실험 필요. win_length: fft를 통해 주파수 도메인 피쳐 뽑는 윈도우 크기. hop_length = window 를 얼만큼 씩 전진시킬지? 보통 win_length 의 반을 많이 사용함. n_mels: mel bin 갯수, 프리퀀시 쪽에서 몇개의 피쳐 고려할 지 지정함.
   - feature_extract_stft(fnm, samp_sec=20, sr = 4000, pre_emphasis = 0, hop_length=256, win_length = 512) : mel 처리 전단계 피쳐. 따라서 프리퀀시 부분이 melspec 보다 크고, log 스케일로 프리퀀시가 증가하지 않음. 옵션 설명은 위의 melspec과 동일
   - feature_extract_cqt(fnm, samp_sec=20, sr = 4000, pre_emphasis = 0, filter_scale = 1, n_bins = 80, fmin = 10): cqt 피쳐 뽑는 함수, filter_scale 은 1을 그대로 쓰는것을 추천, n_bins 는 bin 크기위에 n_mel과 동일함. fmin은 minimum frequency 세팅.
   - get_murmur_loc(data): 주어진 환자 데이터에서 #Murmur locations: 뒷부분의 murmur location 가져오는 함수
   - get_features_3lb(data): age,sex,hw,preg,loc,mel1 피쳐 추출해서 학습용으로 정리하는 함수
   - get_feature_one(patient_data, verbose = 0) : 제출 후 run_challenge 모형 돌릴 때, 각 환자별로 피쳐추출 필요. 앞의 get_features_3lb 대응 버전의 한 환자 피쳐 추출 함수
   - get_features_3lb_all(data_folder, patient_files_trn, samp_sec=20, pre_emphasis = 0, hop_length=256, win_length = 512, n_mels = 100, filter_scale = 1, n_bins = 80, fmin = 10) : 음성 피쳐 옵션 고려하여 피쳐 정리하도록 함수 확장함. 함수 사용 예제: Upgrading_models2_adding_features.ipynb
   - get_features_3lb_all_one(data_folder, patient_files_trn, samp_sec=20, pre_emphasis = 0, hop_length=256, win_length = 512, n_mels = 100, filter_scale = 1, n_bins = 80, fmin = 10) : 위에 정의된 함수 대응버전. 각 환자별로 피쳐 추출 함수: Upgrading_models2_adding_features.ipynb   
- Generator0.py : generator class 들 포함함. Mixup, FMM 구현됨. 두가지 버전: Generator0, DataGenerator
   - 이유는 모르겠지만 Generator0 이 성능이 조금 더 좋게 나오는 것 같음
   - 속도는 DataGenerator가 더 빠름
   - DataGenerator 에는 SpecAug 도 구현되어 적용가능
   - 사용법은 model_iy3.ipynb, Upgrading_models1.ipynb 참고   
- models.py: 실험 모형들 정의해두기: get_toy(), 등등등 모형들은 다 toy모형들이라 실험 많이 더 해봐야함