from tools import srl_process, load_json_list, tqdm, save_json, load_json, save_json_list, init_srl, save_file
from tools.cal_rouge import avg_rouge
from tools.check import check_time
import re

###
# 通过语义角色标注后处理
# input: it 
# 输入格式： {"summary": xxx, "doc": xx}
# demo: {"summary": "美国防部:川普将于2019年11月10日阅兵", "doc": "美国国防部表示,川普总统所要求的阅兵式至少推迟到明年才举办。五角大楼发言人陆军上校曼宁在周四表示:“我们最初定于2018年11月10日阅兵,但现在已经同意探索在2019年阅兵的机会。”最近有媒体报道称,阅兵的费用估算从大约1200万美元增加到9200万美元。美国国防部长吉姆·马蒂斯星期四在一架美国军用飞机上对记者说,他已经为这次检阅军队提供了初步指导,但他还没有看到任何成本估算。根据五角大楼3月发布的一份备忘录,接受检阅的军人将从白宫走到美国国会大厦。这个备忘录说,阅兵式的重点是美国独立战争以来美国军人做出的贡献。参加阅兵的有带轮子的军用车辆,但是没有坦克。参加阅兵的一些军队将穿着不同历史时期的军装,阅兵结束时飞过天空的飞机也将根据资源情况包括一些老式飞机。美国总统川普和夫人梅拉妮亚去年7月14日作为法国总统马克龙和夫人布丽吉特的嘉宾观看了法国国庆日活动。川普随后提出了阅兵的想法。美国很少举行阅兵式。1991年在华盛顿举行过,为的是庆祝在海湾战争中把萨达姆的伊拉克军队赶出科威特。更多评论"}
# 输出格式: 是否改成， 改写后
# 改写后 ： {"summary": xxx, "doc": xx, "summary_post": xx}
###
def srl_post_process(it, key="summary"):
    summary_central_word_dict = text2srl_format(it[key], required_temporal=False)
    doc_central_word_dict = text2srl_format(it['doc'], required_temporal=True)
    f = False
    for summary_central_word in summary_central_word_dict:
        if summary_central_word in doc_central_word_dict:
            doc_tmp = doc_central_word_dict[summary_central_word][0]['temporal']
            doc_tmp_bool = bool(re.search(r'\d', doc_tmp))
            if not doc_tmp_bool:
                continue
            if 'temporal' in summary_central_word_dict[summary_central_word][0]:
                it['summary_post'] = it[key].replace(summary_central_word_dict[summary_central_word][0]['temporal'], \
                doc_central_word_dict[summary_central_word][0]['temporal'])
            else:
                summary_srl = summary_central_word_dict[summary_central_word][0]
                A0 = summary_srl.get("A0", "")
                if len(A0) != 0:
                    _i = it[key].find(A0) + len(A0)
                    it['summary_post'] = it[key][:_i] + doc_tmp + it[key][_i:]
                else:
                    it['summary_post'] = doc_tmp + "," + it[key]
                # input()
            f = True
    if not f:
        it['summary_post'] = it['summary']
    return f, it

def text2srl_format(text, required_temporal=False):
    srl_list = srl_process(text, required_temporal=required_temporal)["srl_list"]
    pre_central_word = ""
    central_word_dict = {}
    srl_dict = {}
    for srl in srl_list:
        if len(pre_central_word) == 0 or pre_central_word == srl[0]:
            pre_central_word = srl[0]
            srl_dict[srl[1]] = srl[4]
        else:
            central_word_dict[pre_central_word] = central_word_dict.get(pre_central_word, [])
            if len(srl_dict.keys()) > 0:
                central_word_dict[pre_central_word].append(srl_dict)
                srl_dict = {}
                srl_dict[srl[1]] = srl[4]
            pre_central_word = srl[0]
        pass
    if len(srl_dict.keys()) > 0:
        central_word_dict[srl[0]] = central_word_dict.get(srl[0], [])
        central_word_dict[srl[0]].append(srl_dict)
    return central_word_dict

def remove_colon(it_list):
    res = []
    for it in it_list:
        colon_list = it["tgt_text"].split(":")
        if len(colon_list) > 0 and len(colon_list[0]) < 6:
            it["tgt_text"] = "".join(colon_list[1:])
            print("...........")
            print(it)
        res.append(it)
    return res

def post_process(data, src_key='src_text', tgt_key='tgt_text'):
    if isinstance(data, str):
        data = load_json(data)
    res = []
    res_post_process = []
    post_n = 0
    for it in tqdm(data):
        # tmp = {"summary": it[tgt_key], "doc": it[src_key]}
        it["summary"] = it[tgt_key]
        it["doc"] = it[src_key]
        if tgt_key != "summary":
            del it[tgt_key]
        if src_key != "doc":
            del it[src_key]
        f, it = srl_post_process(it)
        if f:
            post_n += 1
            res_post_process.append(it)
        res.append(it)
    print(f"total: {len(data)}, post_process: {post_n}")
    save_json("./post_process_tmp.json", res_post_process)
    return res

def main():
    data = load_json_list("../data_lbh/train.txt")
    data_format = []
    for it in data:
        data_format.append({"doc": it.get("description", "")+it['src_text'], "summary": it['tgt_text']})

    # data_format = remove_colon(data_format)

    print(f"total {len(data_format)} data")

    # it ={"summary": "美国防部:川普将于2019年11月10日阅兵", "doc": "美国国防部表示,川普总统所要求的阅兵式至少推迟到明年才举办。五角大楼发言人陆军上校曼宁在周四表示:“我们最初定于2018年11月10日阅兵,但现在已经同意探索在2019年阅兵的机会。”最近有媒体报道称,阅兵的费用估算从大约1200万美元增加到9200万美元。美国国防部长吉姆·马蒂斯星期四在一架美国军用飞机上对记者说,他已经为这次检阅军队提供了初步指导,但他还没有看到任何成本估算。根据五角大楼3月发布的一份备忘录,接受检阅的军人将从白宫走到美国国会大厦。这个备忘录说,阅兵式的重点是美国独立战争以来美国军人做出的贡献。参加阅兵的有带轮子的军用车辆,但是没有坦克。参加阅兵的一些军队将穿着不同历史时期的军装,阅兵结束时飞过天空的飞机也将根据资源情况包括一些老式飞机。美国总统川普和夫人梅拉妮亚去年7月14日作为法国总统马克龙和夫人布丽吉特的嘉宾观看了法国国庆日活动。川普随后提出了阅兵的想法。美国很少举行阅兵式。1991年在华盛顿举行过,为的是庆祝在海湾战争中把萨达姆的伊拉克军队赶出科威特。更多评论"}
    data_format_time = []
    for it in tqdm(data_format):
        try:
            _, it = srl_post_process(it)
            if _:
                data_format_time.append(it)
        except Exception as e:
            print(e)
            print(it) 

        # print(_, it['summary'])

    print(f"total contain temporal {len(data_format_time)} data")
    save_json("../data_lbh/train_poster.txt", data_format_time)

def to_model_format(path):
    data = load_json(path)
    res = []
    for it in data:
        res.append({"tgt_text": it["summary_post"], "src_text": it["doc"]})
    res = remove_colon(res)
    save_json_list("../data_lbh/train_poster_format.txt", res)
    
# to_model_format("../data_lbh/train_poster.txt")
# post_process()

def srl2text(it):
    pass

def src_text_add_temporal(data):
    if isinstance(data, str):
        data = load_json_list(data)
    for it in tqdm(data):
        print(it)
        if 'srl' not in it:
            it['srl'] = text2srl_format(it['src_text'])
            print(it['srl'])
        break


# src_text_add_temporal("../data_lbh/text.txt")

def json2json_list(path, tgt_path):
    data = load_json(path)
    save_json_list(tgt_path, data)

def json2file(path, tgt_path, key):
    data = load_json(path)
    res = [it[key] for it in data]
    save_file(tgt_path, res)

# json2json_list("../data_lbh/train_merge_res_temporal.json", "../data_lbh/train_merge_res_temporal_list.json")

# path = "./result/___output_dir_lbh_w_model_9_bin_compare.json"
# path = "./result/___output_dir_lbh_w_time_poster_v0_model_1_bin_compare.json"
init_srl()
path = "./tmp.json"
out_path = "./tmp_post.json"
print(
    avg_rouge(path, pre_key='predict', ref_key='tgt_text')
)
check_time(path, key="predict")

post_data = post_process(path, tgt_key='predict')
save_json(out_path, post_data)
print(avg_rouge(post_data, pre_key="summary_post", ref_key="tgt_text"))
check_time(post_data, key="summary_post")
json2file(out_path, "./tmp_post.text", "summary_post")