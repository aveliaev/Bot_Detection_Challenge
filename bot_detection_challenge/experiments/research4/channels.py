"""Cycle 4: disjoint information channels over the EXISTING cycle-1/2 features (no new features)."""
import sys; from pathlib import Path
R4 = Path(__file__).resolve().parent; ROOT = R4.parent
sys.path.insert(0, str(ROOT / 'cycle1')); sys.path.insert(0, str(ROOT / 'research2'))
from features import family_map
from cv import POOL_COLS

DROP = {'tech_platform_spellings', 'dow'}  # technical / shift-analysis only


def channels(cols):
    fm = family_map(cols)
    ch = {'mouse': [], 'behavior': [], 'item': [], 'context': []}
    for c in cols:
        f = fm[c]
        if c in DROP:
            continue
        if c in ('is_web', 'ptr_none_web'):
            ch['context'].append(c)            # platform / absence of a pointer = metadata, not mouse behaviour
        elif f == 'G_pointer':
            ch['mouse'].append(c)
        elif f == 'H_ua_platform':
            ch['context'].append(c)
        elif f in ('E_items', 'J_missing') or c.startswith('pop_') or c == 'ev_per_item':
            ch['item'].append(c)
        else:                                  # activity, cookie age, intervals, events, sequence, search, time, lastN
            ch['behavior'].append(c)
    ch['item'] += list(POOL_COLS)
    return ch
