from tools import load_json, save_file, save_json
from tools.cal_rouge import avg_rouge
from tools.cut_rep import cut_rep


def cut_rep_process():
    paths = ["../result/___output_dir_sum_100w_model_10_bin_compare.json"]
    res = []
    data_list = [load_json(path) for path in paths]
    res_rouge = []
    for i in range(len(data_list[0])):
        f = 0
        ref_ans = ""
        for data in data_list:
            predict, flag = cut_rep(data[i]['predict'])
            if i == 43:
                print(data[i]['predict'], "=>", predict)
            if flag == True:
                print(data[i]['predict'], "=>", predict)
                print("*"*60)
            if len(predict) > 1:
                res.append(predict)
            else:
                res.append(data[i]['predict'])
        res_rouge.append({"predict": res[-1], "tgt_text": data_list[0][i]['tgt_text'], "src_text": data_list[0][i]['src_text']})
    save_file("./tmp.txt", res)
    save_json("./tmp.json", res_rouge)
    print(avg_rouge(data_list[0], mode="f"))
    print(avg_rouge(res_rouge, mode="f"))

# def ensemle(path_dir):

cut_rep_process()
    
    

