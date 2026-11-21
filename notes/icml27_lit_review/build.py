import json,re,difflib
def norm(s): return re.sub(r'[^a-z0-9 ]','',(s or '').lower())
A=json.load(open('arxiv_recs.json')); D=json.load(open('doi_recs.json'))
NR=json.load(open('new_resolved.json')); EX=json.load(open('extra_resolved.json')); SW=json.load(open('sweep.json'))
DROP_AX={'2412.15115','2305.05176','2406.18665'}                 # Qwen report, FrugalGPT, RouteLLM: grant-only
DROP_DOI={'10.1126/science.208.4447.1019','10.1016/S0020-7373(84)80003-6','10.1371/journal.pone.0096580'}  # wrong DOIs in notes / off-topic
DROP_NEW={'Gifting in Multi-Agent Reinforcement Learning','Learning to Incentivize Information Acquisition',
          'Agentic Misalignment Deterrence: Incorporating Probability and Stakeholders to Decision-Making',
          'Solving Stackelberg Prediction Game with Least Squares Loss via Spherically Constrained Least Squares Reformulation'}
KEEP_SW='''2504.05259 2510.09462 2511.02823 2512.15688 2602.10153 2604.03968 2605.09684 2605.15377 2606.10456 2606.11998 2607.02514 2607.13087
2607.19321 2610.03818 2610.24052 2610.26935 2610.26664 2506.05296 2606.06529 2610.28787 2610.37055 2610.33728 2411.07087 2102.02304 2106.06060
2404.15059 2504.21579 2606.05867 2610.03253 2610.36989 2012.00892 2507.22049 2508.02076 2605.26448 2505.03961 2504.07831 2609.18204 2406.07875
2412.02091 2509.14485 2210.09646 2610.09002 2403.15128 2409.00993 2602.02625 1901.10059 2112.10859 2302.14604 2408.09686 2510.08607 2605.02063
2301.08278 2202.09859 1906.09874 2001.05458 2101.11967 2106.03927 2305.00768 2402.12416 2506.15887 2508.17696 2601.06382 2602.15407 2605.24516
2605.30003 2607.04710 2608.07280 2610.01667 2610.07881 1909.03510 2304.10351 2508.02421'''.split()
CORE=set('''1702.03037 1707.06600 2107.06857 2211.13746 2503.14576 2610.13710 1803.08884 2106.09012 2001.09318 2006.06051 2208.10469 1806.04067
2108.02755 2410.08345 2602.16662 2404.16698 2312.06942 2409.07985 2607.07368 2607.11751 2604.25085 1303.0356 2608.12323 2607.09766 2609.38262
2607.28317 2608.14668 2610.37055 2506.05296 2302.14604 1901.10059 2605.24516 2407.18074 2402.14090 2301.08278 2112.10859 2601.11369 2506.23276
2609.15293 2609.29547 2502.14143 2012.08630 2010.00575 2610.28787 2304.10351 2406.07875'''.split())
CORE_TITLES={'Monitored Markov Decision Processes','Oracles & Followers: Stackelberg Equilibria in Deep Multi-Agent Reinforcement Learning',
 'Adaptive Incentive Design','Mediated Multi-Agent Reinforcement Learning','When Security Games Go Green: Designing Defender Strategies to Prevent Poaching and Illegal Fishing',
 'Reinforcement Learning with State Observation Costs in Action-Contingent Noiselessly Observable Markov Decision Processes','Restitution Over Deterrence: Formal Enforcement for LLM Commons Governance',
 'Learning Optimal Commitment to Overcome Insecurity','Deep Reinforcement Learning for Green Security Games with Real-Time Information','Social Environment Design'}
RULES=[('AI control & monitoring',r'\b(control|monitor\w*|sabotage|scheming|collu\w+|subver\w+|untrusted|honeypot|red-team\w*|insider)'),
 ('Scalable oversight',r'(scalable oversight|debate|weak-to-strong|weak llms judging|judge)'),
 ('MARL social dilemmas',r'(reinforcement learning|multi-agent rl|marl|melting pot|social dilemma|socialjax|lab2d|ppo|opponent|reciprocity)'),
 ('Incentive & mechanism design',r'(mechanism|incentiv\w*|contract|tax\w*|economist|principal|legislat\w*|mediat\w*|stackelberg|policy teaching|environment design|planner|regulat\w*)'),
 ('Sanctions, norms & institutions',r'(norm\b|norms|sanction\w*|punish\w*|enforce\w*|reputation|institution\w*|govern\w*|restitution|deterr\w*|compliance|honesty)'),
 ('Auditing & security games',r'(audit\w*|inspection|security game|green security|commitment|costly state verification|randomi[sz]ed oversight)'),
 ('Costly observation / partial monitoring',r'(observation cost|state observation costs|active measur\w*|monitored markov|observation-constrained|act-then-measure)'),
 ('LLM agents: cooperation & games',r'(llm|language model|generative agent|gpt|homo silicus)'),
 ('Commons & resources',r'(commons|common-pool|common pool|fisher\w*|harvest|sustainab\w*|resource)'),
 ('MAS safety & contagion',r'(jailbreak|prompt infection|contagion|epidemic|virus|viruses|topolog\w*|spread|propagat\w*|steganograph\w*|manipulated knowledge)'),
 ('Eval methodology & validity',r'(evaluation awareness|being evaluated|sensitivity to spurious|pimmur|statistical precipice|validity|benchmark)')]
def tags_for(title,abstract=''):
    t=(title+' ').lower(); ta=(title+' '+(abstract or '')[:600]).lower(); out=[]
    for name,rx in RULES:
        if re.search(rx,t) or (name not in ('LLM agents: cooperation & games','Commons & resources') and re.search(rx,ta) and name in ('MARL social dilemmas','Costly observation / partial monitoring')): out.append(name)
    return out or ['Other / background']
items=[]
def creators(names):
    out=[]
    for n in names:
        n=n.strip()
        if not n: continue
        parts=n.split()
        out.append({'firstName':' '.join(parts[:-1]),'lastName':parts[-1],'creatorType':'author'} if len(parts)>1 else {'name':n,'creatorType':'author'})
    return out
def add_arxiv(aid,r,src,extra_tags=()):
    it={'itemType':'preprint','title':r['title'],'creators':creators(r['authors']),'date':r['date'],'abstractNote':r.get('abstract') or '',
        'url':f'https://arxiv.org/abs/{aid}','repository':'arXiv','archiveID':f'arXiv:{aid}','DOI':r.get('doi') or '',
        'extra':(f"Published: {r['journal']}\n" if r.get('journal') else '')+f'arXiv: {aid}'}
    tg=['ICML27']+[f'topic: {x}' for x in tags_for(r['title'],r.get('abstract'))]+[src]+list(extra_tags)
    if aid in CORE or r['title'] in CORE_TITLES: tg.append('★ core for gameplan')
    it['tags']=[{'tag':x} for x in tg]; items.append(it)
TYPEMAP={'journal-article':'journalArticle','book':'book','book-chapter':'bookSection','proceedings-article':'conferencePaper','posted-content':'preprint','article':'journalArticle'}
def add_pub(r,src,doi=None,oa_type=None):
    typ=TYPEMAP.get(r.get('type') or oa_type or 'journal-article','journalArticle')
    it={'itemType':typ,'title':r['title'],'creators':creators(r['authors']),'date':r.get('date') or '','abstractNote':re.sub('<[^>]+>','',r.get('abstract') or ''),
        'DOI':doi or r.get('doi') or '','url':f'https://doi.org/{doi or r.get("doi")}' if (doi or r.get('doi')) else ''}
    j=r.get('journal') or ''
    if typ=='journalArticle': it.update(publicationTitle=j,volume=r.get('volume') or '',issue=r.get('issue') or '',pages=r.get('pages') or '')
    elif typ=='conferencePaper': it['proceedingsTitle']=j
    elif typ=='bookSection': it['bookTitle']=j
    elif typ=='book': it['publisher']=r.get('publisher') or ''
    tg=['ICML27']+[f'topic: {x}' for x in tags_for(r['title'],r.get('abstract'))]+[src]
    if r['title'] in CORE_TITLES: tg.append('★ core for gameplan')
    it['tags']=[{'tag':x} for x in tg]; items.append(it)
# 1. project-noted arXiv papers
for aid,r in A.items():
    if aid in DROP_AX: continue
    add_arxiv(aid,r,'from: project notes')
# 2. project-noted DOIs
for d,r in D.items():
    if d in DROP_DOI: continue
    add_pub(r,'from: project notes',doi=d)
# 3. new MARL/oversight titles, extras
for t,v in list(NR.items())+list(EX.items()):
    if t in DROP_NEW or not v['rec'] or v['score']<0.9: continue
    r=v['rec']
    if r['source']=='arxiv': add_arxiv(r['id'],r,'added: MARL & oversight (28 Oct)')
    else: add_pub(r,'added: MARL & oversight (28 Oct)',oa_type=('journal-article' if r.get('type')=='article' else r.get('type')))
# 4. sweep
for aid in KEEP_SW:
    add_arxiv(aid,SW[aid],'added: arXiv sweep 2025-26')
# 5. manual: Bell & Madisetti (ICML 2026 page only)
items.append({'itemType':'conferencePaper','title':'Restitution Over Deterrence: Formal Enforcement for LLM Commons Governance','creators':[{'lastName':'Bell','firstName':'','creatorType':'author'},{'lastName':'Madisetti','firstName':'','creatorType':'author'}],
  'date':'2026','proceedingsTitle':'International Conference on Machine Learning (ICML 2026)','url':'https://icml.cc/virtual/2026/78631','extra':'Recorded from the ICML 2026 virtual page; full author names not checked.',
  'tags':[{'tag':x} for x in ['ICML27','topic: Sanctions, norms & institutions','topic: LLM agents: cooperation & games','topic: Commons & resources','from: project notes','★ core for gameplan']]})
# Konuganti from refs.bib (not found in indexes)
items.append({'itemType':'conferencePaper','title':'Agentic Misalignment Deterrence: Incorporating Probability and Stakeholders to Decision-Making','creators':[{'lastName':'Konuganti','firstName':'','creatorType':'author'}],
  'date':'2026','extra':'From paper/paper_v6 refs.bib (konuganti2026deterrence); not found in arXiv/OpenAlex on 28 Oct, check the venue.',
  'tags':[{'tag':x} for x in ['ICML27','topic: Sanctions, norms & institutions','topic: LLM agents: cooperation & games','from: project notes']]})
# dedupe by title (prefer published DOI record over preprint, keep union of tags)
best={}
for it in items:
    k=norm(it['title'])[:80]
    match=next((kk for kk in best if difflib.SequenceMatcher(None,kk,k).ratio()>0.92),None)
    if match is None: best[k]=it; continue
    old=best[match]; keep,other=(it,old) if (old['itemType']=='preprint' and it['itemType']!='preprint') else (old,it)
    tags={t['tag'] for t in keep['tags']}|{t['tag'] for t in other['tags']}
    keep['tags']=[{'tag':t} for t in sorted(tags)]
    if other.get('archiveID'): keep['extra']=(keep.get('extra','')+f"\nAlso: {other['archiveID']}").strip()
    best[match]=keep
final=list(best.values())
for i,it in enumerate(final): it['id']=f'icml27_{i}'
json.dump(final,open('zotero_items.json','w'))
from collections import Counter
c=Counter(t['tag'] for it in final for t in it['tags'])
print(len(items),'before dedupe ->',len(final))
for k,v in sorted(c.items(),key=lambda x:-x[1]): print(f'{v:4d}  {k}')
