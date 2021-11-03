from typing import List

def get_next(s:str)->List[int]:
    i = 0
    j = -1
    n = len(s)
    nxt = [-1]*(n+1)
    while i<n:
        if j==-1 or s[i]==s[j]:
            i += 1
            j += 1
            nxt[i] = j
        else:
            j = nxt[j]
    return nxt

def func(s:str)->(str,bool):
    t = s[::-1]
   # print(s,t)
    nxt = get_next(t)
    pos = 0
    for i in range(1,len(t)+1):
        if (i-nxt[i]) <= nxt[i]:
            pos = max(pos,nxt[i])
        #print(nxt[i],i-nxt[i],pos,t[i-1])
    return t[pos:][::-1],pos!=0

def cut_rep(s: str)->(str, bool):
    ans, flag = func(s)
    if flag == False:
        return ans, flag
    return cut_rep(ans)[0], True
    
import jieba

if __name__=='__main__':
    k = jieba.lcut('2015年末郑州新建机场、万滩机场、万滩机场')
    for x in [k]:
        ans, flag = cut_rep(x)
        print(x,ans,flag)

# 铁骑路面执勤路面救治急救车救治急救车救治急救车救治急救车救治急救车救治急救车救治急救车
# 救治车救治车救治车救治车救治车救治车救治车
#     path = "../../predict_clw.json"
#     total = 0
#     ok_n = 0
#     res = []
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             ans, flag = func(line)
#             res.append(ans)
#             if flag:
#                 ok_n += 1
#     print(f"total: {total}, ok_n: {ok_n}")

#     path = "../../predict_clw_cut_rep.txt"
#     with open(path, "w", encoding="utf-8") as f:
#         for it in res:
#             f.write(it)
#             f.write("\n")
#     print(path)


        
