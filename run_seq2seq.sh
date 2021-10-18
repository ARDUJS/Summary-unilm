#python3  run_seq2seq.py --data_dir data_news/ --src_file sputniknews_train.json --model_type unilm --model_name_or_path data/torch_unilm_model --output_dir output_dir_2/ --max_seq_length 512 --max_position_embeddings 512 --do_train --do_lower_case --train_batch_size 16 --learning_rate 1e-5 --num_train_epochs 10 --model_recover_path output_dir/model.2.bin 


python3  run_seq2seq.py \
    --data_dir ../data_lbh/ \
    --src_file single_doc_100_time_loc_list_3k.json \
    --model_type unilm \
    --model_name_or_path ../data/torch_unilm_model \
    --output_dir ../output_dir_lbh_w_time/ \
    --max_seq_length 512 \
    --max_position_embeddings 512 \
    --do_train \
    --do_lower_case \
    --train_batch_size 16 \
    --learning_rate 1e-5 \
    --num_train_epochs 100 \
    --model_recover_path ../output_dir_lbh_w/model.9.bin
