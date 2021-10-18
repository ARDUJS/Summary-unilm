from tools import load_json, save_json_list

path = "../data_lbh/single_doc_100_time_loc.json"
data = load_json(path)
path = "../data_lbh/single_doc_100_time_loc_list_3k.json"
save_json_list(path, data*30)