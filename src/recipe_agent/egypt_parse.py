"""Parsers for fifi.cooking Egyptian recipes: Arabic quantities -> grams, time text -> minutes,
ingredient name -> price-map row. Pure Python, offline."""
import re, json, csv, sys
AR = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
def norm(s):
    s = s.translate(AR)
    s = re.sub('[\u064B-\u0652\u0640]', '', s)
    s = s.replace('پ','ب').replace('ڤ','ف').replace('گ','ك')  # Persian/Urdu letters in some recipes
    return s.replace('أ','ا').replace('إ','ا').replace('آ','ا').replace('ى','ي').replace('ة','ه')

# ---------- time ----------
def parse_minutes(t):
    """Arabic time text -> minutes (upper bound for ranges). 'بدون طهو' -> 0. Unparseable -> None."""
    if not t: return None
    t = norm(str(t)).strip()
    if re.search(r'بدون', t): return 0.0
    total = 0.0; found = False
    m = re.search(r'(\d+(?:\.\d+)?)(?:\s*(?:-|الي|الى|او)\s*(\d+(?:\.\d+)?))?\s*ساع', t)
    if m:
        h = max(float(m.group(1)), float(m.group(2) or 0)); found = True
        if re.search(r'ساع\w*\s*و\s*(نص|نصف)', t): h += 0.5
        total += h * 60
    elif re.search(r'ساعتين|ساعتان', t):
        total += 120; found = True
        if re.search(r'و\s*(نص|نصف)', t): total += 30
    elif re.search(r'ساعه', t):
        total += 60; found = True
        if re.search(r'ساعه\s*و\s*(نص|نصف)', t): total += 30
        elif re.search(r'ساعه\s*و\s*ربع', t): total += 15
    m2 = re.search(r'(\d+(?:\.\d+)?)(?:\s*(?:-|الى|الي|او)\s*(\d+(?:\.\d+)?))?\s*(?:دقيق|دقايق|دقائق|د\b)', t)
    if m2:
        total += max(float(m2.group(1)), float(m2.group(2) or 0)); found = True
    if re.search(r'بضع ساعات|ساعات', t) and not found: return None   # vague, not a number
    return total if found else None

def parse_servings(s):
    if not s: return None
    s = norm(str(s)); n = re.findall(r'\d+(?:\.\d+)?', s)
    if not n: return None
    v = [float(x) for x in n[:2]]
    return sum(v) / len(v)

# ---------- quantity ----------
WORDNUM = {'ربع':0.25,'نص':0.5,'نصف':0.5,'تلت':1/3,'ثلث':1/3,'تلات':3,'ثلاث':3,'ثلاثه':3,'اتنين':2,'اثنين':2,'اثنان':2,'واحد':1,'واحده':1}
DUAL = ['معلقتين','ملعقتين','كوبايتين','كوبين','معلقات','معالق','ملاعق','حبتين','بيضتين','فصين']
VOL = {'tbsp':['ملعقه كبيره','معلقه كبيره','ملاعق كبيره','معالق كبيره','معلقتين كبار','معلقات كبار','ملعقه اكل','معلقه اكل'],
       'tsp':['ملعقه صغيره','معلقه صغيره','ملاعق صغيره','معالق صغيره','معلقات صغيره','معلقتين صغيرتين','معلقتين صغار','ملعقه شاي','معلقه شاي'],
       'cup':['كوباي','كوب','اكواب','فنجان']}
PIECE = [('حبه','piece'),('حبات','piece'),('حبتين','piece'),('ثمره','piece'),('ثمار','piece'),('بيضه','piece'),('بيضات','piece'),('بيضتين','piece'),
         ('فصوص','clove'),('فص','clove'),('راس','head'),('رووس','head'),('علبه','can'),('علب','can'),('قطع','piece'),('قطعه','piece'),
         ('قرنين','piece'),('قرون','piece'),('قرن','piece'),('بصل','piece'),('ليمون','piece'),('جزر','piece'),('خس','piece'),('ارغف','piece'),('رغيف','piece'),('صدر','piece'),('عود','piece'),('مكعب','piece'),('باكه','piece'),('باكيت','piece'),('حبتان','piece'),('شريحه','piece'),('شرائح','piece'),('شرايح','piece'),('ورقه','piece'),('ورقتين','piece'),('دجاج','whole_chicken'),('فراخ','whole_chicken'),('فرخه','whole_chicken'),('رغيف','piece'),('مخ','piece')]
TASTE = ['حسب الرغبه','حسب الحاجه','حسب الذوق','للتحليه','للتزيين','للتقديم','قليل','للتحمير','رشه','رشات','لازم']

def first_number(t):
    t = norm(t).replace('حوالي','').replace('تقريبا','')
    if re.search(r'ثلاثه\s+ارباع|تلت\w*\s+ارباع', t): return 0.75, 'word'
    m = re.search(r'(\d+)\s+(\d+)/(\d+)', t)
    if m: return int(m.group(1)) + int(m.group(2))/int(m.group(3)), 'num'
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:-|–|—|الي|الى|او)\s*(\d+(?:\.\d+)?)', t)
    if m: a, b = float(m.group(1)), float(m.group(2)); return (a+b)/2, 'range'
    m = re.search(r'(\d+)/(\d+)', t)
    if m: return int(m.group(1))/int(m.group(2)), 'num'
    m = re.search(r'\d+(?:\.\d+)?', t)
    if m: 
        v = float(m.group(0))
        if re.search(r'\d\s*و\s*(نص|نصف)', t): v += .5
        if re.search(r'\d\s*و\s*ربع', t): v += .25
        return v, 'num'
    for w in DUAL[:3]+DUAL[5:6]:
        pass
    if re.search(r'معلقتين|ملعقتين|معلقتان|ملعقتان|كوبايتين|كوبين|كوبان|حبتين|بيضتين|فصين|ورقتين|(?:بصل|ليمون|جزر|قرن|فلفل|طماطم|خيار|بطاطس|باذنجان|بتنجان|كوسه)\w*(?:تين|تان|ين)(?:\s|$)', t): return 2, 'word'
    if re.search(r'معلقات|معالق|ملاعق', t): return 2, 'word'  # "معلقات" without number ~ 2 (flagged low confidence)
    for w, v in WORDNUM.items():
        if re.search(r'(^|\s)'+w+r'(\s|$)', t):
            base = v
            if re.search(w+r'\s*(كيلو|كوب|ملعقه|معلقه|راس)', t) or True:
                if re.search(r'\sو\s*(نص|نصف)', t): base += .5
                return base, 'word'
    return None, None

def parse_amount(text):
    """returns dict(qty, unit, to_taste, confidence)"""
    t = norm(str(text or '')).strip()
    out = dict(qty=None, unit=None, to_taste=False, conf='none')
    if not t: return out
    if any(norm(w) in t for w in TASTE) and not re.search(r'\d', t):
        out['to_taste'] = True; out['conf'] = 'to_taste'; return out
    qty, how = first_number(t)
    unit = None
    if re.search(r'كيلو|كجم|كغ', t): unit = 'kg'
    elif re.search(r'جرام|غرام|جم\b', t): unit = 'g'
    elif re.search(r'مليلتر|ملليتر|ميليلتر|\d\s*مل\b', t): unit = 'ml'
    elif re.search(r'(?<!ملي)(?<!ميلي)لتر', t): unit = 'l'  # مليلتر is ml, not litres
    elif re.search(r'رشه|رشات|رشة', t) and not re.search(r'رشفه', t): unit = 'pinch'; qty = qty or 1
    else:
        for u, keys in VOL.items():
            if any(k in t for k in map(norm, keys)): unit = u; break
        if unit is None and re.search(r'معلق|ملعق|معالق|ملاعق', t): unit = 'tbsp'; out['conf'] = 'low'  # size not given
        if unit is None:
            for k, u in PIECE:
                if norm(k) in t: unit = u; break
    if qty is None and unit in ('tbsp','tsp','cup','piece','can','clove','head','kg') and re.search(r'^(معلق|ملعق|كوب|كيلو|راس|علب|فرخ)', t): qty = 1  # bare unit word = 1
    if qty is None and unit == 'whole_chicken': qty = 1
    if qty is None and unit == 'piece' and re.search(r'ليمونه|بصله|جزره|فلفله|^قرن$', t): qty = 1
    out.update(qty=qty, unit=unit)
    if out['conf'] == 'none': out['conf'] = 'ok' if (qty and unit) else ('count_assumed' if qty else 'none')
    if how == 'range' and out['conf'] == 'ok': out['conf'] = 'range_midpoint'
    return out

# ---------- mapping ----------
def load_map(path):
    rows = list(csv.DictReader(open(path, encoding='utf-8-sig')))
    for r in rows: r['_kws'] = [norm(k.strip()) for k in r['keywords'].split('|') if k.strip()]
    return rows   # file order = priority (specific first)

def match(name, rows):
    n = norm(name)
    hit = None
    for r in rows:
        for k in r['_kws']:
            exact = k.startswith('=')
            k = k.lstrip('=')
            # keyword must start a word (optional leading ال); '=' also needs word end
            pat = r'(?<![\u0600-\u06FF])(?:ال)?' + re.escape(k) + (r'(?![\u0600-\u06FF])' if exact else '')
            if re.search(pat, n): hit = r; break
        if hit: break
    # "فراخ مفرومة" must not be priced as minced beef: a poultry / seafood word in the
    # name wins over the minced-meat keywords (the map is first-match-wins by file order).
    if hit and hit['key'] in ('minced_beef', 'beef_generic', 'buffalo_meat'):
        for pat, target in ((r'فراخ|فرخه|دجاج|كتكوت', 'chicken'),
                            (r'جمبري|روبيان|قريدس', 'shrimp'),
                            (r'بلطي', 'tilapia'), (r'تونه|تونا', 'tuna')):
            if re.search(pat, n):
                hit = next((r for r in rows if r['key'] == target), hit)
                break
    # "مرق لحم" / "مرقة دجاج" is broth, not meat: a broth word wins over the meat keywords.
    if hit and hit['key'] != 'broth' and re.search(r'(?<![\u0600-\u06FF])(?:ال)?(?:مرقه|مرق|شوربه|استوك)(?![\u0600-\u06FF])', n):
        hit = next((r for r in rows if r['key'] == 'broth'), hit)
    return hit

def f(x):
    try: return float(x)
    except: return None

def grams(a, r, ing_name=''):
    q, u = a['qty'], a['unit']
    if q is None: return None
    if u is None and r is not None and f(r['g_per_whole']):
        # a bare count of a whole bird ("دجاجة", "2 فراخ") is whole birds, not generic pieces
        nn = norm(ing_name)
        if re.search(r'فرخ|فراخ|دجاج', nn) and not re.search(
                r'صدر|صدور|وراك|افخاذ|ورك|اجنح|جناح|قوانص|كبده|شاورما|مفروم|كفته|قطع|شرائح|شرايح|بانيه', nn):
            u = 'whole_chicken'
    if u is None and (r is None or not f(r['g_per_piece'])): return None
    if u == 'kg': return q*1000
    if u == 'ml': return q*(f(r['density_g_per_ml']) or 1.0) if r else q
    if u == 'pinch': return 0.5*q
    if u is None and r is not None and f(r['g_per_piece']): u = 'piece'
    if u == 'g': return q
    if u == 'l': return q*1000*(f(r['density_g_per_ml']) or 1.0) if r else q*1000
    if r is None:
        return {'tbsp':15,'tsp':5,'cup':240}.get(u, None) and q*{'tbsp':15,'tsp':5,'cup':240}[u]
    dens = f(r['density_g_per_ml']) or 1.0
    if u == 'cup':  return q * (f(r['g_per_cup']) or 240*dens)
    if u == 'tbsp': return q * (f(r['g_per_tbsp']) or 15*dens)
    if u == 'tsp':  return q * (f(r['g_per_tsp']) or 5*dens)
    if u == 'clove': return q * (f(r['g_per_clove']) or 4)
    if u == 'head':  return q * (f(r['g_per_head']) or 40)
    if u in ('piece','can','whole_chicken'):
        key = {'piece':'g_per_piece','can':'g_per_can','whole_chicken':'g_per_whole'}[u]
        return q * f(r[key]) if f(r[key]) else None
    return None
