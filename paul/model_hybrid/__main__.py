from .candidate_gen import *
from .frel_gen import *
from .fit_predict import *
from ..evalute import evalute
from .models import MyRidgeClassifier_NonNeg,MyRidgeClassifier_NonCross_NonNeg
from ..hybrid import hybrid

langs=['ar','bn','es','fa','fi','fr','hi','id','ja','ko','ru','sw','te','th','zh']

algos=['mdpr_tydi','mdpr','qld']
candidate_gen('dev',algos,langs,'mdpr_tydi_qld_by_model')
frel_gen('train',algos,langs,'mdpr_tydi_qld_by_model')
fit_predict(MyRidgeClassifier_NonArg_NonNeg,'mdpr_tydi_qld_by_model','train','dev',langs,'mdpr_tydi_qld_by_model',100)
hybrid('dev',['bm25_0.13'],[1],['en'],'mdpr_tydi_qld_by_model',100)
evalute('dev','mdpr_tydi_qld_by_model',['ar','bn','en','es','fa','fi','fr','hi','id','ja','ko','ru','sw','te','th','zh'])
