from model.cand_gen import *
from model.dataset_gen import *
from model.model_hybrid import *
from evalute import evalute

train='train'
val='dev'
methodname='can_neg'
langs=['ar','bn','en','es','fa','fi','fr','hi','id','ja','ko','ru','sw','te','th','zh']
algos=['mdpr','bm25','qld']

dataset_gen(train,algos,langs)
cand_gen(val,algos,langs)
model_hybrid(train,val,langs,methodname)
evalute(val,methodname,langs)
