"""Controlled software diagnostic on CC-BY-4.0 garment meshes.

This is an instrumentation/unit test over engineered cues, not a validation of
the seven-criterion audit protocol or a computer-vision benchmark.
"""
from __future__ import annotations

import csv, hashlib, json, math, random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT.parent/'06_professional_garment_modeling'/'raw'/'zenodo_3d_garments_with_sewing_patterns'/'sample_subset_200'
OUT=ROOT/'experiments'/'exp04_asset_level_validation'
FIG=ROOT/'manuscript'/'figures'
KINDS=['component','silhouette','topology','material','motif','metadata']
SEVERITIES=[0.15,0.30,0.50,0.75]

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def load_obj(path):
    v=[]; f=[]
    for line in path.read_text(encoding='utf-8',errors='ignore').splitlines():
        if line.startswith('v '): v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith('f '):
            ids=[int(x.split('/')[0])-1 for x in line.split()[1:]]
            for i in range(1,len(ids)-1): f.append([ids[0],ids[i],ids[i+1]])
    return np.asarray(v,float),np.asarray(f,int)
def features(v,f):
    lo=v.min(0); hi=v.max(0); ext=np.maximum(hi-lo,1e-9)
    tri=v[f] if len(f) else np.empty((0,3,3)); area=float((np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)*.5).sum()) if len(f) else 0
    edges=set()
    for a,b,c in f:
        edges.update((tuple(sorted((int(a),int(b)))),tuple(sorted((int(b),int(c)))),tuple(sorted((int(c),int(a))))))
    return {'vertices':len(v),'faces':len(f),'extent_x':ext[0],'extent_y':ext[1],'extent_z':ext[2],'aspect_xz':ext[0]/ext[2],'surface_area':area,'edge_count':len(edges),'face_vertex_ratio':len(f)/max(1,len(v))}
def write_obj(path,v,f):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as o:
        for x,y,z in v: o.write(f'v {x:.7g} {y:.7g} {z:.7g}\n')
        for a,b,c in f: o.write(f'f {a+1} {b+1} {c+1}\n')
def rel(a,b,key): return abs(b[key]-a[key])/max(abs(a[key]),1e-9)
def intervene(v,f,kind,s):
    vv=v.copy(); ff=f.copy(); extra={'material_metallic':0.,'motif_offset':0.,'metadata_swap':0.}
    if kind=='component':
        cutoff=np.quantile(vv[:,2],1-s*.55); keep=vv[:,2]<=cutoff; ff=ff[np.all(keep[ff],axis=1)]
    elif kind=='silhouette': vv[:,0]=(vv[:,0]-vv[:,0].mean())*(1+s*.8)+vv[:,0].mean()
    elif kind=='topology': ff=ff[np.arange(len(ff))%max(2,int(round(1/max(s,1e-3))))!=0]
    elif kind=='material': extra['material_metallic']=s
    elif kind=='motif': extra['motif_offset']=s
    elif kind=='metadata': extra['metadata_swap']=s
    return vv,ff,extra
def cue_scores(base,cur,extra):
    return {
      'component':min(1.,rel(base,cur,'vertices')+rel(base,cur,'faces')),
      'silhouette':min(1.,rel(base,cur,'aspect_xz')),
      'topology':min(1.,rel(base,cur,'face_vertex_ratio')+rel(base,cur,'edge_count')),
      'material':extra['material_metallic'], 'motif':extra['motif_offset'], 'metadata':extra['metadata_swap']}
def baselines(c):
    geom=max(c[k] for k in ['component','silhouette','topology'])
    visual=max(c[k] for k in ['silhouette','material','motif'])
    meta=c['metadata']; equal=float(np.mean(list(c.values())))
    return {'geometry_only':geom,'visual_only':visual,'metadata_only':meta,'equal_weight':equal,'single_threshold':float(max(c.values())>=.25),'criterion_max_diagnostic':max(c.values())}
def best_threshold(y,s):
    best=(0.5,-1)
    for t in np.linspace(0.005,1,200):
        f=precision_recall_fscore_support(y,np.asarray(s)>=t,average='binary',zero_division=0)[2]
        if f>best[1]: best=(float(t),float(f))
    return best[0]
def cluster_bootstrap_ci(frame,score_col,t,seed=7,n=1000):
    """Resample independent parent assets, retaining all records per parent."""
    rng=np.random.default_rng(seed); vals=[]; parents=frame.asset_id.unique()
    for _ in range(n):
        sampled=rng.choice(parents,len(parents),replace=True)
        blocks=[frame[frame.asset_id.eq(parent)] for parent in sampled]
        z=pd.concat(blocks,ignore_index=True)
        vals.append(precision_recall_fscore_support(z.is_failure,z[score_col]>=t,average='binary',zero_division=0)[2])
    return np.quantile(vals,[.025,.975]).tolist()

def main():
    OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(exist_ok=True)
    paths=[]
    for cat in ['pants_straight_sides','skirt_2_panels']:
        paths += sorted((SOURCE/cat).rglob('*_sim.obj'))[:10]
    manifest=[]; rows=[]
    for i,path in enumerate(paths):
        aid=f'ZG{i+1:03d}'; v,f=load_obj(path); base=features(v,f)
        within_category=i % 10
        split='calibration' if within_category < 6 else 'test'
        manifest.append({'asset_id':aid,'category':path.parts[-3],'source_relpath':str(path.relative_to(SOURCE)).replace('\\','/'),'sha256':sha(path),'license':'CC-BY-4.0','source_record':'https://zenodo.org/records/5267549','split':split,'vertices':len(v),'faces':len(f),'units':'dataset_native','software':'Python controlled-intervention script','generation_command':'python scripts/run_asset_level_benchmark.py','claim_status':'technical'})
        # Negative controls and nuisance transformations.
        # Twenty-four balanced nuisance controls: camera/background/lighting and
        # image compression do not alter the mesh; coordinate quantization is a
        # storage proxy and should remain below the failure threshold.
        controls=[]
        for j in range(6):
            controls += [('camera_view',j/5,v,f,{}),('background',j/5,v,f,{}),('lighting',j/5,v,f,{}),('compression',j/5,np.round(v,5-j//2),f,{})]
        for name,s,vv,ff,ex in controls:
            cur=features(vv,ff); c=cue_scores(base,cur,{'material_metallic':0,'motif_offset':0,'metadata_swap':0}); bs=baselines(c)
            rows.append({'asset_id':aid,'split':split,'kind':name,'severity':s,'is_failure':0,**{f'cue_{k}':z for k,z in c.items()},**bs})
        for kind in KINDS:
            for s in SEVERITIES:
                vv,ff,ex=intervene(v,f,kind,s); cur=features(vv,ff); c=cue_scores(base,cur,ex); bs=baselines(c)
                rows.append({'asset_id':aid,'split':split,'kind':kind,'severity':s,'is_failure':1,**{f'cue_{k}':z for k,z in c.items()},**bs})
                if i==0:
                    stem=f'{aid}_{kind}_s{int(s*100):02d}'
                    if kind in {'component','silhouette','topology'}: write_obj(OUT/'representative_variants'/(stem+'.obj'),vv,ff)
                    (OUT/'representative_variants'/(stem+'.json')).write_text(json.dumps({'asset_id':stem,'parent_asset_id':aid,'intervention_type':kind,'intervention_severity':s,'target_criterion':kind,'source_record':'https://zenodo.org/records/5267549','license':'CC-BY-4.0','claim_status':'technical','generation_command':'python scripts/run_asset_level_benchmark.py'},indent=2),encoding='utf-8')
    pd.DataFrame(manifest).to_csv(OUT/'asset_manifest.csv',index=False)
    df=pd.DataFrame(rows); df.to_csv(OUT/'intervention_records.csv',index=False)
    cal=df[df.split=='calibration']; test=df[df.split=='test']; metrics=[]
    for method in ['geometry_only','visual_only','metadata_only','equal_weight','single_threshold','criterion_max_diagnostic']:
        t=best_threshold(cal.is_failure,cal[method]); y=test.is_failure.values; s=test[method].values; pred=s>=t
        pr,re,f1,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0); tn,fp,fn,tp=confusion_matrix(y,pred).ravel()
        ci=cluster_bootstrap_ci(test,method,t)
        metrics.append({'method':method,'threshold':t,'precision':pr,'recall':re,'f1':f1,'specificity':tn/(tn+fp),'roc_auc':roc_auc_score(y,s),'pr_auc':average_precision_score(y,s),'f1_ci_low':ci[0],'f1_ci_high':ci[1]})
    pd.DataFrame(metrics).to_csv(OUT/'baseline_metrics.csv',index=False)
    # Per-kind held-out performance and localization.
    t=[x['threshold'] for x in metrics if x['method']=='criterion_max_diagnostic'][0]; pk=[]
    for kind in KINDS:
        # Leave-one-intervention-out threshold calibration.
        cal_loo=cal[cal.kind!=kind]; t_loo=best_threshold(cal_loo.is_failure,cal_loo.criterion_max_diagnostic)
        z=test[(test.kind==kind)|test.kind.isin(['camera_view','background','lighting','compression'])]; pr,re,f1,_=precision_recall_fscore_support(z.is_failure,z.criterion_max_diagnostic>=t_loo,average='binary',zero_division=0); pk.append({'heldout_kind':kind,'threshold_from_other_kinds':t_loo,'precision':pr,'recall':re,'f1':f1})
    pd.DataFrame(pk).to_csv(OUT/'ood_by_intervention.csv',index=False)
    loc=[]; mat=[]
    for _,r in test[test.is_failure==1].iterrows():
        cues=np.array([r['cue_'+k] for k in KINDS]); pred=KINDS[int(np.argmax(cues))]; loc.append({'true':r.kind,'predicted':pred,'correct':int(pred==r.kind),'target_cue':r['cue_'+r.kind],'max_offtarget':max(r['cue_'+k] for k in KINDS if k!=r.kind)})
    ld=pd.DataFrame(loc); ld.to_csv(OUT/'criterion_localization.csv',index=False)
    for a in KINDS:
        mat.append([int(((ld.true==a)&(ld.predicted==b)).sum()) for b in KINDS])
    pd.DataFrame(mat,index=KINDS,columns=KINDS).to_csv(OUT/'localization_confusion.csv')
    mono=[]
    for kind in KINDS:
        z=df[df.kind==kind]; vals=z.groupby('severity')['cue_'+kind].mean(); rho=spearmanr(vals.index,vals.values).statistic; violations=int(np.sum(np.diff(vals.values)<-1e-10)); mono.append({'kind':kind,'spearman_severity_target':rho,'monotonicity_violations':violations,'mean_offtarget_leakage':ld[ld.true==kind].max_offtarget.mean()})
    pd.DataFrame(mono).to_csv(OUT/'monotonicity_isolation.csv',index=False)
    nuisance=df[df.is_failure==0]; failures=df[df.is_failure==1]; inv={'designed_null_control_false_alarm_rate':float((nuisance.criterion_max_diagnostic>=t).mean()),'designed_null_score_sd':float(nuisance.criterion_max_diagnostic.std()),'mean_designed_null_score':float(nuisance.criterion_max_diagnostic.mean()),'mean_engineered_failure_score':float(failures.criterion_max_diagnostic.mean())}
    (OUT/'summary.json').write_text(json.dumps({'assets':len(paths),'records':len(df),'calibration_assets':12,'test_assets':8,'category_balance':{'calibration':{'trousers':6,'skirts':6},'test':{'trousers':4,'skirts':4}},'bootstrap_unit':'parent asset','license':'CC-BY-4.0','source_record':'https://zenodo.org/records/5267549','criterion_max_diagnostic':next(x for x in metrics if x['method']=='criterion_max_diagnostic'),'localization_accuracy':float(ld.correct.mean()),'designed_null_controls':inv,'claim_boundary':'Software instrumentation diagnostic only. Material, motif, and metadata channels read engineered intervention variables; camera/background/lighting controls are not rerendered. This experiment does not validate Equation (1), the seven-criterion rubric, visual invariance, or cultural correctness.'},indent=2),encoding='utf-8')
    curve=[]
    for th in np.linspace(0,1,201):
        pr,re,f1,_=precision_recall_fscore_support(cal.is_failure,cal.criterion_max_diagnostic>=th,average='binary',zero_division=0); curve.append({'threshold':th,'precision':pr,'recall':re,'f1':f1})
    pd.DataFrame(curve).to_csv(OUT/'threshold_curve.csv',index=False)
    # Figures.
    md=pd.DataFrame(metrics); fig,axs=plt.subplots(1,3,figsize=(11,3.2)); axs[0].barh(md.method,md.f1,xerr=[md.f1-md.f1_ci_low,md.f1_ci_high-md.f1]); axs[0].set_xlim(0,1); axs[0].set_title('Held-out detection F1 (95% CI)'); mm=pd.DataFrame(mono); axs[1].barh(mm.kind,mm.spearman_severity_target); axs[1].set_xlim(0,1); axs[1].set_title('Severity monotonicity'); im=axs[2].imshow(np.asarray(mat),cmap='Blues'); axs[2].set_xticks(range(6),KINDS,rotation=45,ha='right'); axs[2].set_yticks(range(6),KINDS); axs[2].set_title('Criterion localization');
    for i in range(6):
        for j in range(6): axs[2].text(j,i,mat[i][j],ha='center',va='center',fontsize=7)
    fig.tight_layout(); fig.savefig(FIG/'asset_level_validation.pdf',bbox_inches='tight'); fig.savefig(FIG/'asset_level_validation.png',dpi=220,bbox_inches='tight'); plt.close(fig)

if __name__=='__main__': main()
