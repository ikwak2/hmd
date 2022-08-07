# 파이토치용 wav2vec2
- https://huggingface.co/blog/fine-tune-wav2vec2-english 참고하여 모델을 적절히 고쳤음

## experiment_wav2vec2
- 가장 basic하게 모델 불러와서 끝에 layer를 단순히 linear 1층으로 고침, 전체 weight를 낮은 lr로 학습

## experiment_wav2vec2_dnn
- 끝 layer를 linear 2층으로 수정, 전체 weight를 낮은 lr로 학습

## experiment_wav2vec2_freeze
- 끝 layer를 linear 1층으로 수정 후, 마지막 linear 1층을 제외하고 weight를 freeze 시켰음

## experiment_wav2vec2_freeze_dnn
- 끝 layer를 linear 2층으로 수정 후, 그것을 제외하고 weight를 freeze

# 텐서플로용 wav2vec2
- https://www.tensorflow.org/hub/tutorials/wav2vec2_saved_model_finetuning 에 나와있긴한데 저는 파이토치만 실험했습니다,,!
