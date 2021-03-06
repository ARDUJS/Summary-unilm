from rouge_metric import PyRouge
import os
import json
from tools import save_json

def read_file(filename):
    with open(filename,"r") as f:
        cout = f.readlines()
    return cout

def avg_rouge(ref_file,pre_file, source_file, name=""):
    # ref_file = read_file(ref_dir)
    # pre_file = read_file(pre_dir)
    # print(len(ref_file))
    # print(len(pre_file))
    assert len(ref_file)==len(pre_file)
    _len = len(ref_file)
    score_list = []
    rouge1r=0
    rouge2r=0
    rouge3r=0
    str1 = "predict tgt_text    src_text\n"
    res = []
    for ref, pre, source in zip(ref_file, pre_file, source_file):
        # ref = json.loads(json.dumps(eval(ref)))
        # source = ref['src_text']
        pre = "".join(pre.replace("[UNK]", "").split(" ")).strip()
        pre = " ".join(pre)
        ref = " ".join(ref)
        str1 += ("".join(pre.split(" ")).strip() + "\t" + "".join(ref.split(" ")).strip() +"\t"+ source.strip() + "\n") 
        res.append({
            "predict": "".join(pre.split(" ")).strip(),
            "tgt_text": "".join(ref.split(" ")).strip(),
            "src_text": source.strip()
            })
        rouge = PyRouge(rouge_n=(1, 2, 4), rouge_l=True, rouge_w=False, rouge_s=False, rouge_su=False, skip_gap=4)
        # print(pre, ref)
        score = rouge.evaluate([pre],[[ref]])
        rouge1r += score['rouge-1']['f']
        rouge2r += score['rouge-2']['f']
        rouge3r += score['rouge-l']['f']
    name = name.replace(".", "_").replace("/", "_")+"_compare.json"
    save_json(f"../result/{name}", res)
    # print(str1)
    print("rouge-1:%.4f rouge-2:%.4f rouge-l:%.4f" %( rouge1r/_len,rouge2r/_len,rouge3r/_len))
    return rouge1r/len(ref_file), rouge2r/len(ref_file), rouge3r/len(ref_file)

#if __name__ == "__main__":
#    ref_dir = "./song_data/test_data.json"
    # ref_dir = "./data_news/sputniknews_test.json"
#    pre_dir = "./predict_song.json"
#    rouge1r,rouge2r,rougelr = avg_rouge(ref_dir,pre_dir)
#    print("rouge-1:%.4f rouge-2:%.4f rouge-l:%.4f" %( rouge1r,rouge2r,rougelr))
