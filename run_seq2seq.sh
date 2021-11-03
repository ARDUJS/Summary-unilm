#python3  run_seq2seq.py --data_dir data_news/ --src_file sputniknews_train.json --model_type unilm --model_name_or_path data/torch_unilm_model --output_dir output_dir_2/ --max_seq_length 512 --max_position_embeddings 512 --do_train --do_lower_case --train_batch_size 16 --learning_rate 1e-5 --num_train_epochs 10 --model_recover_path output_dir/model.2.bin 
mkdir -p ../result

# python3  run_seq2seq.py \
#     --data_dir ../data/ \
#     --src_file train_clean.json \
#     --model_type nezha \
#     --model_name_or_path ../../pretrain_model/nezha-large-www \
#     --output_dir ../output_dir_sum_100w_nezha/ \
#     --max_seq_length 512 \
#     --max_position_embeddings 512 \
#     --do_train \
#     --do_lower_case \
#     --train_batch_size 384 \
#     --gradient_accumulation_steps 32 \
#     --learning_rate 2e-5 \
#     --num_train_epochs 10 \

python3  run_seq2seq.py \
    --data_dir ../data/ \
    --src_file train_clean.json \
    --model_type unilm \
    --model_name_or_path ../../pretrain_model/torch_unilm_model \
    --output_dir ../output_dir_sum_100w_16/ \
    --max_seq_length 512 \
    --max_position_embeddings 512 \
    --do_train \
    --do_lower_case \
    --train_batch_size 176 \
    --learning_rate 5e-5 \
    --num_train_epochs 10 \

# --model_recover_path ../output_dir_sum_100w/model.16.bin

# --mask_whole_word 
# --model_recover_path ../output_dir_sum_100w/model.10.bin

# python3  run_seq2seq.py \
#     --data_dir ../data/ \
#     --src_file train_clean.json \
#     --model_type nezha \
#     --model_name_or_path ../../pretrain_model/nezha-base-www/ \
#     --output_dir ../output_dir_sum_nezha-base-wwm/ \
#     --max_seq_length 512 \
#     --max_position_embeddings 512 \
#     --do_train \
#     --do_lower_case \
#     --train_batch_size 384 \
#     --gradient_accumulation_steps 32 \
#     --learning_rate 2e-5 \
#     --num_train_epochs 10 \

# --mask_whole_word \
# --num_workers 30 \
# --mask_source_words \


# --model_recover_path ../output_dir_lbh_w_time_poster_v0/model.1.bin


# train_poster_format.txt

# ../../pretrain_model/nezha-large-www 
# ../../pretrain_model/nezha-cn-base