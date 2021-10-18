import json

f = open("data_lbh/text.txt","r",encoding="utf-8")
res = []
for i in f.readlines():
	res.append(eval(i))

count = 0
for i in res:
	count += min(512, len(i["src_text"]))

print(count, count//len(res), count/(1*60+29))
