"""Rebuild derived numeric indexes from preserved evidence; no training."""
from pathlib import Path
import csv,gzip,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def put(repo,name,s):
    p=repo/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s.strip()+'\n')
CLIP=CIL=ROOT
def write_csv(p, rows, fields=None):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def readcsv(p):
    if not p.exists() and p.with_name(p.name+'.gz').exists():p=p.with_name(p.name+'.gz')
    opener=gzip.open if p.suffix=='.gz' else open
    with opener(p,'rt',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def link(repo,p,label=None):
    if not p.exists() and p.with_name(p.name+'.gz').exists():p=p.with_name(p.name+'.gz')
    rel=str(p.relative_to(repo));return f'[{label or rel}](../{rel})'

CIL_ROOTS=['projects/efcil_direction_diagnostic', 'cds-main', 'cdsp_handoff_20260813', 'pcds_structure_sgds_official', 'revisitingcil-inspect.uZlsNf', 'structure_multicenter_separation', 'structure_e3_followup', 'structure_similarity_aware_followup', 'structure_similarity_control_experiments', 'structure_selective_similarity_sweep', 'structure_rp_protoclad_task5', 'structure_stored_apr_rp', 'structure_prototype_oracle_audit', 'fecam_boundary_followup', 'variation_consistency_pilot', 'error_trajectory_diagnostic', 'cross_task_ca_reweighting', 'b1_ranking_preserving_correction', 'type_a_generative_augmentation', 'generation_local_all_transition_audit', 'ia10_bob_vs_strong_full', 'research_report_for_professor', '0923_code_bundle', 'ptm_final_handoff_stage_bXo40S', 'ptm_guided_handoff_final_stage_VsYbp3', 'ptm_guided_handoff_stage_uVZlnb']
def clip_results():
    graph=[]
    for folder in sorted((CLIP/'CLIP/그래프').iterdir()):
        if not folder.is_dir() or folder.name=='휴지통':continue
        pp=[p for p in folder.rglob('*.csv') if p.name in {'adapter_accuracy_summary.csv','accuracy_summary.csv','margin_adapter_accuracy_summary.csv','uncertain_adapter_accuracy_summary.csv'}]
        if not pp:continue
        for p in pp:
            if not p.stat().st_size:continue
            for r in readcsv(p):
                if r.get('epoch')=='50' and r.get('shot')=='16':
                    graph.append(dict(group=folder.name,shot=16,epoch=50,split=r['split'],dataset=r['dataset'],mean_accuracy_percent=100*float(r['mean_accuracy']),recorded_n_support_seeds=r.get('n_seeds',r.get('n','')),source_file=str(p.relative_to(CLIP)),evidence_tier='graph_summary_only_unless_raw_linked'))
    write_csv(CLIP/'results/graph_epoch50_shot16.csv',graph)
    raw=[];seen=set();empty=0
    for p in sorted((CLIP/'CLIP').rglob('results.json')):
        if not p.stat().st_size:empty+=1;continue
        try:j=json.loads(p.read_text())
        except:continue
        if any(s in str(p).lower() for s in ['smoke']):continue
        args=j.get('args',{});key=hashlib.sha256(p.read_bytes()).hexdigest();duplicate=key in seen;seen.add(key)
        for shot in j.get('shots',[]):
            final=shot.get('final',{})
            for split,metrics in [('target',{args.get('target_name',j.get('target_name','target')):final.get('target',{})}),('preserve',final.get('preserve',{}))]:
                for dataset,m in metrics.items():
                    if 'accuracy' not in m:continue
                    raw.append(dict(package=str(p.relative_to(CLIP).parts[1:3]),run=str(p.parent.relative_to(CLIP)),sample_seed=args.get('sample_seed',''),shot=shot.get('shot'),epoch=final.get('epoch',''),split=split,dataset=dataset,accuracy_percent=100*m['accuracy'],delta_from_zero_shot_pp=100*m['delta_from_zero_shot'] if 'delta_from_zero_shot' in m else '',model=args.get('model',''),pretrained=args.get('pretrained',''),lr=args.get('lr',''),prompt_seed=args.get('prompt_seed',''),split_seed=args.get('seed',''),selection_method=args.get('selection_method',args.get('selection_mode','')),duplicate_file=duplicate,source_file=str(p.relative_to(CLIP))))
    write_csv(CLIP/'results/raw_final_results.csv',raw)
    lines=['# Recorded results and evidence boundaries','','All accuracy values below are percentages. These are 16-shot / epoch-50 graph summaries, not a certified matched leaderboard. Each row retains the recorded support-seed count. Exact split/resource/config matching is unavailable for several graph-only packages.','','| Graph group | DTD target | ImageNet preserve | CIFAR-100 preserve | Recorded support seeds |','|---|---:|---:|---:|---:|']
    for name in dict.fromkeys(r['group'] for r in graph):
        rr=[r for r in graph if r['group']==name];v={r['dataset']:r for r in rr}
        def val(k):return f"{v[k]['mean_accuracy_percent']:.3f}" if k in v else '—'
        lines.append(f"| {name} | {val('texture')} | {val('imagenet')} | {val('cifar100')} | {rr[0]['recorded_n_support_seeds']} |")
    lines+=['','Original base zero-shot: texture 42.482%, ImageNet 60.500%, CIFAR-100 64.930%. Do not infer a graph-specific delta without checking its recorded baseline and evaluation setup. All graph rows and source paths are in [graph_epoch50_shot16.csv](../results/graph_epoch50_shot16.csv).', '', f'{empty} archived `results.json` files are empty. This count includes copies; it is not a failed-run count. No numbers were recovered from those files. The wrappers and entropy package have empty local raw records while standalone graph tables survive; reference-distillation and GIFT graph summaries likewise do not reconstruct missing original args/checkpoints. Text-drift and base CoOp have nonempty raw records.', '', 'Graph groups beginning `4_` and `5_` retain their historical labels. The shown `5_` final rows record 49–50 support seeds, while many base and `4_` tables record 100. Equal epoch/shot/seed-count does not establish equal teacher/reference resources. Trash-folder graph versions are retained but excluded here.', '', 'The [raw index](../results/raw_final_results.csv) records every parseable non-smoke final result with its source. `duplicate_file=true` means a byte-identical `results.json` was already encountered; do not count it again as a replicate. Packages and standalone plots may still contain logically duplicated runs with different serialization. Selected best cases and post-hoc teacher choices are diagnostics, not unbiased population averages.', '', 'JS-quantile diagnostics have an implemented trainer and nonempty plotted summaries in `CLIP/코드/CoOp_js_tradeoff/plots/`, but no surviving `results.json`. Inspect the plot summary, fallback counts and teacher/resource boundaries before comparing it with the base experiments.']
    put(CLIP,'docs/RESULTS.md','\n'.join(lines))
    return graph,raw

def cil_results():
    rows=[]
    canonical=[x for x in CIL_ROOTS if not x.startswith(('0923','ptm_','research_report'))]
    for name in canonical:
        for p in sorted((CIL/name).rglob('summary.json')):
            if 'snapshot_objects' in p.parts:continue
            try:j=json.loads(p.read_text())
            except:continue
            if 'Inc-Acc' not in j:continue
            cfg=p.parent/'resolved_config.json';c=json.loads(cfg.read_text()) if cfg.exists() and cfg.stat().st_size else {}
            protocol=c.get('protocol_id',c.get('protocol',c.get('dataset','unknown')))
            n=j.get('expected_tasks',len(j.get('A_by_task',[])))
            if protocol=='imageneta':protocol=f'IA{n}'
            if protocol in {'car196','cars196'}:protocol=f'CAR{n}'
            smoke=any(x in str(p).lower() for x in ['smoke','preflight','scratch','feasibility'])
            task=p.parent/'task_summary.csv';t=readcsv(task) if task.exists() else []
            old=t[-1].get('old_accuracy','') if t else '';new=t[-1].get('new_accuracy','') if t else ''
            partition=c.get('task_partition',[])
            rows.append(dict(project=name,experiment_id=j.get('experiment_id',p.parent.parent.name),protocol=protocol,partition=json.dumps(partition),seed=j.get('seed',c.get('seed','')),completed_tasks=j.get('completed_tasks',''),expected_tasks=n,status=j.get('status','unknown'),evidence_role='smoke_not_performance' if smoke else 'recorded_run',Inc_Acc=j.get('Inc-Acc',''),Last_Acc=j.get('Last-Acc',''),Forgetting=j.get('Forgetting',''),Old=old,New=new,config_file=str(cfg.relative_to(CIL)) if cfg.exists() else '',source_file=str(p.relative_to(CIL)),config_sha256=hashlib.sha256(cfg.read_bytes()).hexdigest() if cfg.exists() else ''))
    write_csv(CIL/'results/cil_run_index.csv',rows)
    lines=['# Results: comparable blocks and diagnostic boundaries','','Accuracy and forgetting are %, deltas pp. Model seed is 1993 in these main blocks. No cross-protocol average or global ranking is calculated. The [run index](../results/cil_run_index.csv) retains completed/incomplete/smoke distinctions and source/config paths. Source run IDs are namespaced by project; `B1` in different studies does not necessarily mean the same method.','', '## Reduced selective study: matched 5-task blocks','','| Protocol | Run | Inc | Last | Forgetting ↓ | Old | New | ΔInc vs Structure |','|---|---|---:|---:|---:|---:|---:|---:|']
    source=CIL/'structure_selective_similarity_sweep/outputs/metrics_per_seed.csv'
    for r in readcsv(source):
        if r['protocol'] not in {'IA5','CAR5'} or r['run_id'] not in {'Structure','A_P005','B_P020_A100','C_P020_L030','C_P020_L100'} or r['status']!='complete':continue
        lines.append('| '+r['protocol']+' | '+r['run_id']+' | '+' | '.join(f"{float(r[k]):.3f}" for k in ['Inc-Acc','Last-Acc','Forgetting','Old','New','delta_Inc-Acc'])+' |')
    lines+=['',f'Source: {link(CIL,source)}. A/B improve some stability measures but lose IA5 new accuracy; Method C preserves IA5 new accuracy better but does not improve CAR5 incremental accuracy. No common incremental-accuracy winner is established. See the [final reduced report](../structure_selective_similarity_sweep/reports/reduced_stage1_final_20260923/report.md).', '', '## RP and stored birth prototypes','','| Protocol | Method | Inc | Last | F ↓ | Old | New |','|---|---|---:|---:|---:|---:|---:|']
    for project in ['structure_rp_protoclad_task5','structure_stored_apr_rp']:
        for r in rows:
            if r['project']==project and r['status']=='complete' and r['evidence_role']=='recorded_run':
                vals=[f"{float(r[k]):.3f}" if r[k]!='' else '—' for k in ['Inc_Acc','Last_Acc','Forgetting','Old','New']]
                lines.append('| '+str(r['protocol'])+' | '+r['experiment_id']+' | '+' | '.join(vals)+' |')
    lines+=['','Old RP CAR5 runs have no completed result. StoredAPR CAR5 is not a substitute for that missing ablation. IA5 λ=.30 auxiliary RP has a small accuracy improvement; stronger λ and StoredAPR degrade aspects of new learning. See [RP report](../structure_rp_protoclad_task5/rp_protoclad_report.md) and [StoredAPR final report](../structure_stored_apr_rp/reports/stored_apr_rp_final_20260923/report.md).', '', '## Later acquisition diagnostics and interventions','','| Matched contrast / stage | IA10 | CAR10 | Unit / interpretation |','|---|---:|---:|---|','| Cross-task CA 2× minus its control, final seen | −0.197 | +0.075 | pp; original CA parity failed |','| Conditional task shift C2−C0, final seen | −0.527 | −0.460 | pp; local ranking preserved |','| Generated G−strong S, separate T1 new | +1.143 | +1.074 | pp; not a full T1→T10 trajectory |','| G−S, T9 new all-seen | 0.000 | −0.528 | pp; intervention from common T8 |','| G−S, T10 T9 cohort all-seen | 0.000 | −0.792 | pp; same cohort, real-only T10 |','| G−S, T9 cohort local net | +1 | −5 | sample counts; N=147 / 758 |','| G−S, T10 same cohort local net | 0 | −8 | sample counts; N=147 / 758 |', '', 'Each contrast uses its own internal control. Sources: [cross-task raw IA](../cross_task_ca_reweighting/IA10/summary.json), [CAR](../cross_task_ca_reweighting/CAR10/summary.json), [C2 IA](../b1_ranking_preserving_correction/IA10/metrics.json), [CAR](../b1_ranking_preserving_correction/CAR10/metrics.json), [generation report](../type_a_generative_augmentation/final_report.md), [paired local/all audit](../generation_local_all_transition_audit/final_report.md). Tables retain the original completed-record figures; index regeneration does not infer their mechanisms.', '', 'The [full numerical evidence table](../research_report_for_professor/key_results.csv) records oracle moments, source matching, probes, separation and error measures with units and precise scope. Oracle/actual old banks and test-defined subgroups are diagnostic-only and cannot enter a legal EFCIL method leaderboard. Original checkpoint/main versus post-after_task trajectory evaluation is kept separate: IA10 982 versus 983 correct in relevant records. The numerical mechanism of the one-sample discrepancy is not fully established.', '', '## Latest incomplete branch','','`ia10_bob_vs_strong_full/run_manifest.json` records `interrupted_or_failed`. The process-attempt list has successful environment/CPU checks and failed BOB GPU smoke. The README predates these attempts. No completed S-full or BOB-full result is reported here. Compatibility repair records are retained; their presence is not proof that the GPU failure was resolved.']
    put(CIL,'docs/RESULTS.md','\n'.join(lines))
    return rows

if __name__=="__main__":
    print("Rebuilt CLIP index",len(clip_results()[1]))

# Include both historical families and future experiment records in one index.
records=json.loads((ROOT/'experiments/registry.json').read_text())
records += [json.loads(p.read_text()) for p in sorted((ROOT/'experiments').glob('*/experiment.json'))]
items=[];seen=set()
for r in records:
    if r['id'] in seen:raise SystemExit('duplicate experiment ID: '+r['id'])
    seen.add(r['id'])
    items.append(dict(id=r['id'],name=r.get('name',''),parent=r.get('parent',''),status=r.get('status','unknown'),code=r.get('code',r.get('project','')),result_path=r.get('result_path',r.get('results_root','')),analysis_document=r.get('analysis_document','')))
write_csv(ROOT/'results/experiment_status.csv',items)
lines=['# Experiment status index','','Historical and newly registered experiments. Status is explicitly recorded, never inferred from an output directory name.','','| ID | Name | Parent | Status | Analysis |','|---|---|---|---|---|']
for r in items:
    target=r['analysis_document'];ref=f"[analysis](../{target})" if target else 'see catalogue'
    lines.append('| '+r['id']+' | '+r['name']+' | '+r['parent']+' | '+r['status']+' | '+ref+' |')
put(ROOT,'docs/EXPERIMENT_INDEX.md','\n'.join(lines))
print('Indexed experiment records',len(items))
