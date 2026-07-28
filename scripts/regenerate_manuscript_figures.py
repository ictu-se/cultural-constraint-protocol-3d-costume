"""Regenerate manuscript figures with submission-safe labels."""
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'experiments'/'exp02_diverse_authenticity_benchmark'
VAL=ROOT/'experiments'/'exp03_validation_extensions'
OUT=ROOT/'manuscript'/'figures'; OUT.mkdir(exist_ok=True)
D=pd.read_csv(DATA/'scenario_detail.csv')
D_assess=D[D.record_status.eq('assessable')].copy()
D_doc=D[D.profile_status.eq('documentary')].copy()
S=pd.read_csv(DATA/'scenario_summary.csv')
PM=pd.read_csv(DATA/'aggregate_profile_method.csv')
RM=pd.read_csv(DATA/'aggregate_regime_method.csv')

methods=['expert_reference_upper_bound','retrieval_reference_guided','panel_metadata_hybrid','metadata_guided_hybrid','pattern_driven_generation','texture_motif_transfer','image_to_3d_reconstruction','text_to_3d_generic']
mlabel={'expert_reference_upper_bound':'Sim. oracle','retrieval_reference_guided':'Retrieval ref.','panel_metadata_hybrid':'Panel+metadata','metadata_guided_hybrid':'Metadata hybrid','pattern_driven_generation':'Pattern-driven','texture_motif_transfer':'Motif transfer','image_to_3d_reconstruction':'Image-to-3D','text_to_3d_generic':'Text-to-3D'}
profiles=['ao_dai_modern_school','ao_dai_modern_wedding','ao_dai_early_20c','khan_dong_aodai_performance','ao_tu_than_festival','ao_ngu_than_ceremonial','court_inspired_formal','cham_formal_reference','hmong_festival_reference','dao_red_headwear_reference','generic_ethnic_minorities_prompt','mixed_period_stress_case']
plabel={'ao_dai_modern_school':'Ao dai school','ao_dai_modern_wedding':'Ao dai wedding','ao_dai_early_20c':'Ao dai early 20c','khan_dong_aodai_performance':'Khan dong+ao dai','ao_tu_than_festival':'Ao tu than','ao_ngu_than_ceremonial':'Ao ngu than','court_inspired_formal':'Placeholder B','cham_formal_reference':'Cham ref.','hmong_festival_reference':'Hmong ref.','dao_red_headwear_reference':'Placeholder A','generic_ethnic_minorities_prompt':'Generic ethnic prompt','mixed_period_stress_case':'Mixed-period stress'}
criteria=['garment_components','silhouette_and_proportion','sewing_pattern_plausibility','material_and_drape','motif_and_texture','regional_period_consistency','wearing_context']
clabel=[x.replace('_',' ') for x in criteria]

def save(fig,name):
    fig.tight_layout(); fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight'); fig.savefig(OUT/(name+'.png'),dpi=220,bbox_inches='tight'); plt.close(fig)
def heat(ax,mat,xt,yt,title,cbar='Score',fmt='.0f',vmin=None,vmax=None,cmap='viridis'):
    im=ax.imshow(mat,aspect='auto',cmap=cmap,vmin=vmin,vmax=vmax)
    ax.set_xticks(range(len(xt))); ax.set_xticklabels(xt,rotation=38,ha='right',fontsize=7)
    ax.set_yticks(range(len(yt))); ax.set_yticklabels(yt,fontsize=7); ax.set_title(title,fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]): ax.text(j,i,format(mat[i,j],fmt),ha='center',va='center',fontsize=5,color='white' if im.norm(mat[i,j])>.62 else 'black')
    plt.colorbar(im,ax=ax,label=cbar,fraction=.035,pad=.02)

piv=PM.pivot(index='profile',columns='method',values='mean_audit_score').reindex(index=profiles,columns=methods)
fig,ax=plt.subplots(figsize=(7.2,4.0)); heat(ax,piv.values,[mlabel[x] for x in methods],[plabel[x] for x in profiles],'Profile–archetype scores'); save(fig,'score_heatmap')

agg=PM.groupby('method').agg(score=('mean_audit_score','mean'),hard=('mean_hard_flags','mean')).reindex(methods)
fig,axs=plt.subplots(1,2,figsize=(7.2,2.9)); y=np.arange(8); axs[0].barh(y,agg.score,color=plt.cm.viridis(np.linspace(.15,.85,8))); axs[0].set_yticks(y); axs[0].set_yticklabels([mlabel[x] for x in methods],fontsize=7); axs[0].invert_yaxis(); axs[0].set_xlabel('Mean score'); axs[1].barh(y,agg.hard,color=plt.cm.magma(np.linspace(.15,.85,8))); axs[1].set_yticks(y); axs[1].set_yticklabels([]); axs[1].invert_yaxis(); axs[1].set_xlabel('Mean hard flags'); save(fig,'method_score_hardflags')

pa=D_assess.groupby('profile').agg(score=('audit_score','mean'),hard=('hard_flags','mean')).reindex(profiles)
fig,ax=plt.subplots(figsize=(7.2,2.8)); x=np.arange(12); ax.bar(x,pa.score,color='#2a9d8f'); ax2=ax.twinx(); ax2.plot(x,pa.hard,color='#e76f51',marker='o',lw=1.5); ax.set_xticks(x); ax.set_xticklabels([plabel[x] for x in profiles],rotation=45,ha='right',fontsize=7); ax.set_ylabel('Mean score'); ax2.set_ylabel('Mean hard flags'); ax.set_title('Profile difficulty'); save(fig,'profile_difficulty')

cm=np.array([[D_assess.loc[D_assess.method==m,'score_'+c].mean() for c in criteria] for m in methods])
fig,ax=plt.subplots(figsize=(7.2,3.5)); heat(ax,cm,[x.replace('_',' ') for x in criteria],[mlabel[x] for x in methods],'Criterion-level behavior','Ordinal score','.2f',0,3,'YlGnBu'); save(fig,'criterion_method_heatmap')

reg=['balanced','construction_strict','motif_strict','heritage_strict','visual_lenient']; rp=RM.pivot(index='regime',columns='method',values='mean_audit_score').reindex(index=reg,columns=methods)
fig,ax=plt.subplots(figsize=(7.2,2.7)); heat(ax,rp.values,[mlabel[x] for x in methods],[x.replace('_',' ') for x in reg],'Scenario-regime sensitivity'); save(fig,'regime_method_heatmap')

fig,axs=plt.subplots(1,2,figsize=(7.2,2.8)); rates=[]
for r in reg:
    z=S[S.regime==r].winner_margin; rates.append([(z<1).mean()*100,(z<2).mean()*100]); axs[1].boxplot([S[S.regime==r].winner_margin for r in reg],labels=[x.split('_')[0] for x in reg],showfliers=False)
rates=np.array(rates); x=np.arange(5); axs[0].bar(x-.18,rates[:,0],.36,label='Margin <1'); axs[0].bar(x+.18,rates[:,1],.36,label='Margin <2'); axs[0].set_xticks(x); axs[0].set_xticklabels([a.split('_')[0] for a in reg],rotation=35,ha='right',fontsize=7); axs[0].set_ylabel('Scenarios (%)'); axs[0].legend(fontsize=7); axs[1].tick_params(axis='x',labelrotation=35,labelsize=7); axs[1].set_ylabel('Winner margin'); save(fig,'winner_stability_margin')

fig,ax=plt.subplots(figsize=(7.2,3.0)); sample=[D_assess.loc[D_assess.method==m,'audit_score'].sample(5000,random_state=7).values for m in methods]; ax.boxplot(sample,labels=[mlabel[m] for m in methods],showfliers=False); ax.tick_params(axis='x',labelrotation=35,labelsize=7); ax.set_ylabel('Score'); ax.set_title('Record-level score distributions'); save(fig,'score_distribution')

fail=np.array([[(D_assess.loc[D_assess.method==m,'score_'+c] < D_assess.loc[D_assess.method==m,'hard_threshold']).mean() for c in criteria] for m in methods])
fig,ax=plt.subplots(figsize=(7.2,3.5)); heat(ax,fail,clabel,[mlabel[x] for x in methods],'Criterion hard-failure rate','Failure rate','.2f',0,1,'YlOrRd'); save(fig,'criterion_failure_rate')

low=D.noise_scale<=D.noise_scale.quantile(.25); high=D.noise_scale>=D.noise_scale.quantile(.75); lt=D.hard_threshold<=D.hard_threshold.quantile(.25); ht=D.hard_threshold>=D.hard_threshold.quantile(.75)
noise=[D_assess.loc[high&D_assess.method.eq(m),'audit_score'].mean()-D_assess.loc[low&D_assess.method.eq(m),'audit_score'].mean() for m in methods]; thresh=[D_assess.loc[ht&D_assess.method.eq(m),'hard_flags'].mean()-D_assess.loc[lt&D_assess.method.eq(m),'hard_flags'].mean() for m in methods]
fig,axs=plt.subplots(1,2,figsize=(7.2,3.0)); axs[0].barh(y,noise,color='#e76f51'); axs[1].barh(y,thresh,color='#2a9d8f');
for ax,title in zip(axs,['Noise sensitivity amplitude','Threshold sensitivity amplitude']): ax.set_yticks(y); ax.set_yticklabels([mlabel[x] for x in methods] if ax is axs[0] else [],fontsize=7); ax.invert_yaxis(); ax.set_title(title,fontsize=9)
save(fig,'perturbation_sensitivity')

var=PM.groupby('method').std_audit_score.agg(['mean','min','max']).reindex(methods); pairs=PM.sort_values('std_audit_score',ascending=False).head(10)
fig,axs=plt.subplots(1,2,figsize=(7.2,3.2)); axs[0].barh(y,var['mean'],xerr=[var['mean']-var['min'],var['max']-var['mean']],color='#457b9d'); axs[0].set_yticks(y); axs[0].set_yticklabels([mlabel[x] for x in methods],fontsize=7); axs[0].invert_yaxis(); axs[0].set_xlabel('Score SD'); labs=[plabel[p]+' / '+mlabel[m] for p,m in zip(pairs.profile,pairs.method)]; axs[1].barh(range(10),pairs.std_audit_score,color='#f4a261'); axs[1].set_yticks(range(10)); axs[1].set_yticklabels(labs,fontsize=5); axs[1].invert_yaxis(); axs[1].set_xlabel('Score SD'); save(fig,'profile_method_variability')

rw=pd.read_csv(DATA/'aggregate_regime_winners.csv'); non=rw[rw.method!='expert_reference_upper_bound']; p=non.pivot_table(index='regime',columns='method',values='winner_count',aggfunc='sum',fill_value=0).reindex(reg,fill_value=0)
fig,ax=plt.subplots(figsize=(7.2,2.8)); bottom=np.zeros(5)
for m in ['retrieval_reference_guided','metadata_guided_hybrid','panel_metadata_hybrid']:
    vals=p[m].values if m in p else np.zeros(5); ax.bar(range(5),vals,bottom=bottom,label=mlabel[m]); bottom+=vals
ax.set_xticks(range(5)); ax.set_xticklabels([x.split('_')[0] for x in reg],fontsize=7); ax.set_ylabel('Non-oracle wins'); ax.legend(fontsize=7); save(fig,'regime_winner_composition')

oracle=piv['expert_reference_upper_bound']; best=piv.drop(columns='expert_reference_upper_bound').max(axis=1); gap=(oracle-best).sort_values(ascending=False)
fig,ax=plt.subplots(figsize=(5.0,4.2)); yy=np.arange(12); ax.barh(yy,gap.values,color=['#d62828' if v>22 else '#f77f00' if v>20 else '#2a9d8f' for v in gap]); ax.set_yticks(yy); ax.set_yticklabels([plabel[x] for x in gap.index],fontsize=7); ax.invert_yaxis(); ax.set_xlabel('Gap to simulated-oracle score'); ax.set_title('Best non-oracle gap by profile'); save(fig,'profile_gap_to_expert')

# Rebuild technical validation with safe labels.
a=pd.read_csv(VAL/'ablation.csv'); ss=pd.read_csv(VAL/'seed_stability.csv')
fig,axes=plt.subplots(1,2,figsize=(8.2,3.5)); full=a[a.variant=='full']
for v in ['no_profile_difficulty','single_regime','no_failure_events','no_interactions']:
    z=a[a.variant==v].set_index('method').reindex(methods); axes[0].plot(range(8),z.score_change_from_full,marker='o',label=v.replace('_',' '))
axes[0].axhline(0,color='black',lw=.7); axes[0].set_xticks(range(8)); axes[0].set_xticklabels([mlabel[x] for x in methods],rotation=45,ha='right',fontsize=7); axes[0].set_ylabel('Points vs. full'); axes[0].set_title('Ablation: score change'); axes[0].legend(fontsize=6)
sm=ss.groupby('method').mean_score.agg(['mean','std']).reindex(methods); axes[1].bar(range(8),sm['mean'],yerr=sm['std'],color='#2a9d8f'); axes[1].set_xticks(range(8)); axes[1].set_xticklabels([mlabel[x] for x in methods],rotation=45,ha='right',fontsize=7); axes[1].set_title('Ten-seed stability'); axes[1].set_ylabel('Mean score ± SD')
save(fig,'technical_validation')
