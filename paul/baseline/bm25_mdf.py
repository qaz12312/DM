import os
import shutil
if not os.path.exists('topics'):
    os.makedirs('topics')

def eachf(dsp,beta):
    for lang in ['ar','bn','en','es','fa','fi','fr','hi','id','ko','ru','sw','te','th']:
        try:
            src=f'miracl/miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-{dsp}.tsv'
            dst=f'topics/{lang}_{dsp}.tsv'
            mp={}
            with open(src,'r',encoding='utf-8') as f:
                data=f.readlines()
                n=len(data)
                for li in data:
                    _,li=li.split('\t')
                    li=li.replace('?','')
                    li=li.split(' ')
                    ss=set()
                    for i in li:
                        ss.add(i)
                    for i in ss:
                        if i in mp:mp[i]+=1
                        else: mp[i]=1
                
                outs=[]
                for li in data:
                    idx,li=li.split('\t')
                    li=li.replace('?','')
                    li=li.split(' ')
                    out=[]
                    for i in li:
                        if mp[i]<=n*beta and i!='':
                            out.append(i)
                    outs.append(idx+'\t'+' '.join(out))

                with open(dst,'w',encoding='utf-8') as f:
                    for out in outs:
                        out:str
                        if out.endswith('\n'):
                            f.write(out)
                        else:f.write(out+'\n')
        except:
            pass

    for lang in ['ja','zh']:
        try:
            src=f'miracl/miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-{dsp}.tsv'
            dst=f'topics/{lang}_{dsp}.tsv'
            shutil.copyfile(src,dst)
        except:
            pass

    for lang in ['ar','bn','en','es','fa','fi','fr','hi','id','ja','ko','ru','sw','te','th','zh']:
        # BM25 runfiles
        topic=f'topics/{lang}_{dsp}.tsv'
        rufiles_dir=f'runfiles/bm25_{beta}_h1000'
        os.system(f'python -m pyserini.search.lucene \
            --language {lang} \
            --topics {topic} \
            --index miracl-v1.0-{lang} \
            --output {rufiles_dir}/{lang}_{dsp}.txt \
            --batch 128 --threads 16 --bm25 --hits 1000')

for dsp in ['train','dev','test-a']:
    for beta in [0.08,0.13]:
        eachf(dsp,beta)


