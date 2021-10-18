python3 decode_seq2seq.py \
    --model_type unilm \
    --model_name_or_path ../data/torch_unilm_model \
    --model_recover_path ../output_dir_lbh_w_time/model.1.bin \
    --max_seq_length 512 \
    --input_file ../data_lbh/text.txt  \
    --output_file ../predict_lbh.json \
    --do_lower_case \
    --batch_size 32 \
    --beam_size 1 \
    --max_tgt_length 128

# --input_file ../data_lbh/text.txt
# --input_file ../data_lbh/single_doc_100_time_loc_list.json
# ../output_dir_test/model.6.bin  loss = 1.3   rouge = 0.3
# rouge-1:0.4514 rouge-2:0.2860 rouge-l:0.4083
# rouge-1:0.1700 rouge-2:0.1012 rouge-l:0.1462  (../data_lbh/single_doc_100_time_loc_list.json) (../output_dir_lbh_w/model.9.bin)
# rouge-1:0.1943 rouge-2:0.1190 rouge-l:0.1653
