from tqdm import tqdm
import multiprocessing as mp
from joblib import Parallel, delayed
try:
    from ltp import LTP
except :
    pass
import json
# ltp = LTP() # 默认加载 Small 模型

def init_srl():
    global ltp
    ltp = LTP()

def srl_process(text, is_print = False, required_temporal = True):
    '''
    输入: text
    输出: 包含时间的语义角色标注 
        {
            "text": text, 
            "srl_list":[
                [[type, start, end, text], [type, start, end, text]],
            ]
        }
    demo:
    srl_process("在美国主导下，美英澳三国于2021年9月15日宣布建立新的三边安全伙伴关系，美英将支持澳海军建立核潜艇部队，澳将与美英合作在澳建造核潜艇。澳方随即宣布，撕毁与法国海军集团签订的数百亿美元潜艇大单。")
    '''
    res = {"text": "", "srl_list":[]}
    sents = ltp.sent_split([text])
    segs, hidden = ltp.seg(sents)
    srls = ltp.srl(hidden, keep_empty=False)
    cur_text = ""
    cur_len = 0
    for seg, srl in zip(segs, srls):
        tmp = []
        for token_srl in srl:
            role = seg[token_srl[0]]
            # tmp.append(role)
            args = token_srl[1]
            args_type = [it[0] for it in args]
            if "ARGM-TMP" in args_type or not required_temporal:
                for arg in args:
                    start = cur_len+len("".join(seg[:arg[1]]))
                    end = cur_len+len("".join(seg[:arg[2]+1]))
                    _type = arg[0]
                    _entity = "".join(seg[arg[1]: arg[2]+1])
                    if _type == "ARGM-TMP":
                        _type = "temporal"
                    tmp.append([role, _type, start, end, _entity])
                    if is_print:
                        print(_type, _entity)
                if is_print: 
                    print("*"*60)
        cur_text += "".join(seg)
        cur_len += len("".join(seg))
        if len(tmp) > 0:
            res["srl_list"] += tmp
    res["text"] = cur_text
    return res

# res = srl_process("华盛顿—美国两艘军舰星期四(12月31日)航行经过台湾海峡", required_temporal=False)
# for it in res['srl_list']:
#     print(it)
#     print("*"*60)

def save_file(path, res):
    with open(path, "w", encoding="utf-8") as f:
        for it in res:
            f.write(it)
            f.write("\n")

def load_json_list(path):
    f = open(path, "r", encoding="utf-8")
    data = [eval(it) for it in f.readlines()]
    f.close()
    return data

def save_json_list(path, res):
    f = open(path, "w", encoding="utf-8")
    for it in res:
        f.write(json.dumps(it, ensure_ascii=False))
        f.write("\n")
    print(f"data save in {path}, total: {len(res)}")

def save_json(path, data):
    if isinstance(data, set):
        data = list(data)
    with open(path,'w',encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"data save in {path}, total: {len(data)}")

def load_json(path):
    with open(path,'r',encoding='utf-8') as f:
        data = json.load(f)
    return data

def multi_process(func, lst, num_cores=mp.cpu_count(), backend='multiprocessing', mp = True):
    if mp:
        workers = Parallel(n_jobs=num_cores, backend=backend)
        output = workers(delayed(func)(one) for one in tqdm(lst))
    else:
        output = [func(one) for one in tqdm(lst)]
    return output

def del_none(data):
    return [it for it in data if it != None and len(it) > 0]

def split_sent(sent_list):
    return ltp.sent_split(sent_list)