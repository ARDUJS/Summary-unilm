from rouge_metric import PyRouge
from tools import load_json

##
# 计算rouge
##
def avg_rouge(data, pre_key="predict" ,ref_key="tgt_text", mode="f"):
    if isinstance(data, str):
        data = load_json(data)
    rouge1r, rouge2r, rouge3r = 0, 0, 0
    rouge = PyRouge(rouge_n=(1, 2, 4), rouge_l=True, rouge_w=False, rouge_s=False, rouge_su=False, skip_gap=4)
    for it in data:
        pre = it[pre_key]
        pre = "".join(pre.replace("[UNK]", "").split(" ")).strip()
        pre = " ".join(pre)
        ref = " ".join(it[ref_key])
        score = rouge.evaluate([pre],[[ref]])
        rouge1r += score['rouge-1'][mode]
        rouge2r += score['rouge-2'][mode]
        rouge3r += score['rouge-l'][mode]
        # rouge1r += score['rouge-1']['f']
        # rouge2r += score['rouge-2']['f']
        # rouge3r += score['rouge-l']['f']
    return rouge1r/len(data),rouge2r/len(data),rouge3r/len(data)
    


# print(avg_rouge("./result/___output_dir_lbh_w_model_9_bin_compare.json"))