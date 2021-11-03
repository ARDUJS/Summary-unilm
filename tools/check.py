from tools import srl_process, load_json_list, tqdm, save_json, load_json, save_json_list

def check_time(data, key='predict'):
    if isinstance(data, str):
        data = load_json(data)
    # data = load_json(path)
    pred_temporal_n = 0
    src_tmporal_n = 449
    res = []
    res_temporal = []
    for it in tqdm(data):
        srl = srl_process(it[key])
        it['srl'] = srl
        if len(srl["srl_list"]) > 0:
            pred_temporal_n += 1
            res_temporal.append(it)
        # if len(srl_process(it['src_text'])["srl_list"]) > 0:
        #     src_tmporal_n += 1
        
        res.append(it)
    print(f"total: {len(data)}, pred_temporal_n: {pred_temporal_n}, src_tmporal_n: {src_tmporal_n}")
    return res, res_temporal

# check_time("./result/___output_dir_lbh_w_time_poster_v0_model_1_bin_compare.json")   # total: 501, pred_temporal_n: 91, src_tmporal_n: 449; rouge-1:0.4719 rouge-2:0.2989 rouge-l:0.4238
# check_time("./result/___output_dir_lbh_w_model_9_bin_compare.json")   # rouge-1:0.4531 rouge-2:0.2872 rouge-l:0.4100
# data = load_json_list("../data_lbh/train_merge_final.txt")
# res, res_temporal = check_time(data, key="tgt_text")
# save_json("../data_lbh/train_merge_srl.json", res)
# save_json("../data_lbh/train_merge_res_temporal.json", res_temporal)
# check_time("./result/___output_dir_lbh_w_time_poster_v1_model_1_bin_compare.json")