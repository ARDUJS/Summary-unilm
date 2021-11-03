python3 decode_seq2seq.py \
    --model_type unilm \
    --model_name_or_path ../../pretrain_model/torch_unilm_model \
    --model_recover_path ../output_dir_sum_100w_16/model.7.bin \
    --max_seq_length 512 \
    --input_file ../data/dev.json  \
    --output_file ../result/predict_clw_6.json \
    --do_lower_case \
    --batch_size 4 \
    --beam_size 8 \
    --max_tgt_length 32 \
    --min_len 10 \
    --length_penalty 1.2 \
    --forbid_duplicate_ngrams


# python3 decode_seq2seq.py \
#     --model_type nezha \
#     --model_name_or_path ../../pretrain_model/nezha-base-www/ \
#     --model_recover_path ../output_dir_sum_nezha-base-wwm/model.\*.bin \
#     --max_seq_length 512 \
#     --input_file ../data/one.json  \
#     --output_file ../result/predict_clw.json \
#     --do_lower_case \
#     --batch_size 1 \
#     --beam_size 1 \
#     --max_tgt_length 32 \
#     --min_len 10 \
#     --length_penalty 1.5 \
#     --forbid_duplicate_ngrams

# --input_file ../data_lbh/text.txt
# --input_file ../data_lbh/single_doc_100_time_loc_list.json
# ../output_dir_test/model.6.bin  loss = 1.3   rouge = 0.3
# rouge-1:0.4514 rouge-2:0.2860 rouge-l:0.4083
# rouge-1:0.1700 rouge-2:0.1012 rouge-l:0.1462  (../data_lbh/single_doc_100_time_loc_list.json) (../output_dir_lbh_w/model.9.bin)
# rouge-1:0.1943 rouge-2:0.1190 rouge-l:0.1653

# ../output_dir_lbh_w_time_poster_v0/model.1.bin

# ../../pretrain_model/nezha-base-www/
# ../output_dir_sum_nezha-base-wwm/