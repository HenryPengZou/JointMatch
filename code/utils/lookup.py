import pandas as pd

## Label mappings


isear_label_dict = {"joy":0,"fear":1,"anger":2,"sadness":3,"disgust":4,"shame":5,"guilt":6}
isear_emo_dict = {v: k for k, v in isear_label_dict.items()}


goemotions_label_dict= {"admiration":0,"amusement":1,"anger":2, "annoyance":3,"approval":4,"caring":5,"confusion":6,"curiosity":7,"desire":8,"disappointment":9,"disapproval":10,"disgust":11,"embarrassment":12,"excitement":13,"fear":14,"gratitude":15,"grief":16,"joy":17,"love":18,"nervousness":19,"optimism":20,"pride":21,"realization":22,"relief":23,"remorse":24,"sadness":25,"surprise":26,"neutral":27}
goemotions_emo_dict = {v: k for k, v in goemotions_label_dict.items()}

