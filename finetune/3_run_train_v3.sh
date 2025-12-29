#!/bin/bash
set -e

# 필수 패키지 설치 확인
pip install -U FlagEmbedding

# 학습 실행
# V3: Solar Pro 2로 생성된 12,000개 합성 QA 데이터 사용
# 에폭을 5로 늘려 충분히 학습

torchrun --nproc_per_node 1 \
-m FlagEmbedding.finetune.embedder.encoder_only.m3 \
--output_dir ./finetuned_bge_m3_v3 \
--model_name_or_path BAAI/bge-m3 \
--train_data ./data/train_data_v3.jsonl \
--learning_rate 1e-5 \
--fp16 \
--num_train_epochs 5 \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 16 \
--dataloader_drop_last True \
--query_max_len 128 \
--passage_max_len 512 \
--train_group_size 8 \
--negatives_cross_device \
--logging_steps 10 \
--save_steps 1000 \
--normalize_embeddings True \
--temperature 0.02

echo ">>> 학습 완료! 모델이 ./finetuned_bge_m3_v3 에 저장되었습니다."
