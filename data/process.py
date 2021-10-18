from tools import srl_process, load_json_list, tqdm, save_json_list, multi_process, del_none

path = "../data_lbh/train.txt"
data = load_json_list(path)

def get_time_data_one(it):
    ret = srl_process(it["tgt_text"])
    if len(ret['srl_list']) > 0:
        return it
    return None

res = del_none(multi_process(get_time_data_one, data, mp=False))

print(f"tgt包含时间 共 {len(res)} 个")
# res = load_json_list(path)
save_json_list("../data_lbh/train_time.txt", res)
