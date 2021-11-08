### Reference
> https://github.com/YunwenTechnology/Unilm

### 环境
- python 3.6
- transformers       2.6.0
- torch              1.8.0
### 预训练模型
- 预训练模型 和 训练数据(需要后处理为对应格式)
```
链接：https://pan.baidu.com/s/1kfKd5f6dhp8RCtR5PCoLvg 
提取码：o29e
```
### Train
#### 数据格式
```
{"src_text": "日前,方舟子发文直指林志颖旗下爱碧丽推销假保健品,引起哗...", "tgt_text": "林志颖公司疑涉虚假..."}
{"src_text": "韩方应对路径可以概括为:企业道歉担责;政府公正不护短;民间祈福关...", "tgt_text": "从韩亚航空事故看其应对路径"}
{"src_text": "63岁退休教师谢淑华,拉着人力板车,历时1年,走了2万4千里路...", "tgt_text": "女子用板车拉九旬老母环游中国1年走2万4千里"}
```
#### 命令
> sh run_seq2seq.sh
```
mkdir -p ../result
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
    --num_train_epochs 10
```

### 验证
#### 数据格式
```
同上
```
#### 命令
> sh run_decode.sh
```
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
```

### 推理
#### 数据格式
```
2019年，埃及在非洲55国中脱颖而出，成为了非...
53岁的女医生易无庸马上对其进行心肺复苏，两台...
报道称，德尚最终被法国球迷成功“解救”。继推...
```
#### 命令
```
python3 decode_seq2seq.py \
    --model_type unilm \
    --model_name_or_path ../../pretrain_model/torch_unilm_model \
    --model_recover_path ../output_dir_sum_100w_16/model.7.bin \
    --max_seq_length 512 \
    --input_file ../data/test_sum_1.json  \
    --output_file ../result/predict_clw_6.json \
    --do_lower_case \
    --batch_size 4 \
    --beam_size 8 \
    --max_tgt_length 32 \
    --min_len 10 \
    --length_penalty 1.2 \
    --forbid_duplicate_ngrams \
    --do_raw
```

#### 实验结果
- 验证集：33.28 rouge-L

#### 优点
- 推理速度快
