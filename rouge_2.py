# from rouge_metric import PyRouge
import os
import json
from rouge import Rouge


rouge = Rouge()
def read_file(filename):
    with open(filename,"r") as f:
        cout = f.readlines()
    return cout

def avg_rouge(ref_dir,pre_dir):
    ref_file = read_file(ref_dir)
    pre_file = read_file(pre_dir)
    print(len(ref_file))
    print(len(pre_file))
    assert len(ref_file)==len(pre_file)
    score_list = []
    rouge1r=0
    rouge2r=0
    rouge3r=0
    str1 = "predict tgt_text    src_text\n"
    for ref,pre in zip(ref_file, pre_file):
        ref = json.loads(json.dumps(eval(ref)))
        source = ref['src_text']
        ref = " ".join(ref['tgt_text'])
        str1 += ("".join(pre.split(" ")).strip() + "\t" + "".join(ref.split(" ")).strip() +"\t"+ source.strip() + "\n") 
        # rouge = PyRouge(rouge_n=(1, 2, 4), rouge_l=True, rouge_w=False, rouge_s=False, rouge_su=False, skip_gap=4)
        rouge = rouge.get_scores([pre], [ref])
        score = rouge.evaluate([pre],[ref])
        rouge1r += score['rouge-1']['r']
        rouge2r += score['rouge-2']['r']
        rouge3r += score['rouge-l']['r']
    str1 = str(rouge1r/len(ref_file)) + "\n" + str1
    open(pre_dir+".compare", "w", encoding="utf-8").write(str1)
    return rouge1r/len(ref_file),rouge2r/len(ref_file),rouge3r/len(ref_file)

if __name__ == "__main__":
    ref_dir = "./song_data/test_data.json"
    pre_dir = "./predict_2.json"
    rouge1r,rouge2r,rougelr = avg_rouge(ref_dir,pre_dir)
    print("rouge-1:%.4f rouge-2:%.4f rouge-l:%.4f" %( rouge1r,rouge2r,rougelr))
