import json,urllib.request,urllib.parse,xml.etree.ElementTree as ET,time,sys,difflib,re
ns={'a':'http://www.w3.org/2005/Atom','x':'http://arxiv.org/schemas/atom'}
def norm(s): return re.sub(r'[^a-z0-9 ]','',s.lower())
def sim(a,b): return difflib.SequenceMatcher(None,norm(a),norm(b)).ratio()
def arxiv(title):
    q='ti:"'+re.sub(r'[:\-–,?]',' ',title)+'"'
    url='http://export.arxiv.org/api/query?'+urllib.parse.urlencode({'search_query':q,'max_results':5})
    for _ in range(3):
        try: d=urllib.request.urlopen(url,timeout=60).read(); break
        except Exception as e: time.sleep(5)
    else: return None
    best=None
    for e in ET.fromstring(d).findall('a:entry',ns):
        t=' '.join(e.find('a:title',ns).text.split()); s=sim(t,title)
        if not best or s>best[0]:
            best=(s,dict(source='arxiv',id=e.find('a:id',ns).text.split('/abs/')[-1].split('v')[0],title=t,
              authors=[a.find('a:name',ns).text for a in e.findall('a:author',ns)],date=e.find('a:published',ns).text[:10],
              abstract=' '.join(e.find('a:summary',ns).text.split()),
              doi=(e.find('x:doi',ns).text if e.find('x:doi',ns) is not None else None),
              journal=(e.find('x:journal_ref',ns).text if e.find('x:journal_ref',ns) is not None else None)))
    return best
def openalex(title):
    url='https://api.openalex.org/works?'+urllib.parse.urlencode({'search':title,'per-page':5,'mailto':'research@example.org'})
    try: r=json.load(urllib.request.urlopen(url,timeout=60))
    except Exception: return None
    best=None
    for w in r.get('results',[]):
        t=w.get('title') or ''; s=sim(t,title)
        if not best or s>best[0]:
            inv=w.get('abstract_inverted_index') or {}
            ab=' '.join(k for k,_ in sorted(((k,p) for k,ps in inv.items() for p in ps),key=lambda x:x[1]))
            best=(s,dict(source='openalex',id=w['id'],title=t,authors=[a['author']['display_name'] for a in w.get('authorships',[])],
              date=w.get('publication_date') or str(w.get('publication_year')),abstract=ab,doi=(w.get('doi') or '').replace('https://doi.org/','') or None,
              journal=((w.get('primary_location') or {}).get('source') or {}).get('display_name'),type=w.get('type')))
    return best
if __name__=='__main__':
    items=json.load(open(sys.argv[1])); out={}
    for tag,title in items:
        b=arxiv(title); time.sleep(3)
        if not b or b[0]<0.9:
            o=openalex(title)
            if o and (not b or o[0]>b[0]): b=o
        out[title]=dict(tag=tag,score=round(b[0],3) if b else 0,rec=b[1] if b else None)
        print(f"{out[title]['score']:.2f} {tag:10s} {title[:70]} -> {(b[1]['title'][:60]+' ['+b[1]['source']+']') if b else 'NONE'}",flush=True)
    json.dump(out,open(sys.argv[2],'w'))
