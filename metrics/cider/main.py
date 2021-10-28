import json
from cider import Cider
import pandas as pd 
with open('/media/shenjuanjuan/新加卷1/comment evaluation/data process/dataset/sjj/mine/mine_stem_can.json', 'r') as f:
    can = json.load(f)
with open('/media/shenjuanjuan/新加卷1/comment evaluation/data process/dataset/sjj/mine/mine_stem_ref.json', 'r') as f:
    ref = json.load(f)

print(len(can))
def cider():
    path = '/media/shenjuanjuan/新加卷1/comment evaluation/data process/dataset/sjj/mine/mine.pkl'
    source = pd.read_pickle(path)
    print(source)
    print(source.columns.values)
    # md = source['methodname_st'].tolist()
    # api = source['api_st'].tolist()
    # iden = source['iden_st'].tolist()
    scorer = Cider()
    (score, scores) = scorer.compute_score(can, ref)
    print(score)
    print(scores)
    print(type(scores))
    print(len(scores))
    df = pd.DataFrame(scores)
    df.columns = ['cider']
    print(df)
    
    # source.drop(['cider2'], axis=1, inplace=True)
    source = pd.concat([source, df], axis=1)
    source['cider'] = source['cider'] * 10
    # source['bleu4_3'] = source['bleu4_3'] * 100
    print(source)
    print(source['cider'].mean(axis=0))
    ans = source[['human','cider']]
    print(ans.corr())
    source.to_pickle(path)
    # out = '/media/shenjuanjuan/新加卷2/comment evaluation/data process/dataset/test/signed/selected6319_identi_qubiaodian_cxhy_cider_test1.csv'
    # source.to_csv(out)

cider()