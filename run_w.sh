python3 -u run_seq2seq_w.py --data_dir ../data_lbh/ --src_file train.txt --model_type albert_1 --model_name_or_path voidful/albert_chinese_base  --output_dir ../output_dir_albert_f/ --max_seq_length 512 --max_position_embeddings 512 --do_train --train_batch_size 16 --learning_rate 1e-5 --num_train_epochs 11

