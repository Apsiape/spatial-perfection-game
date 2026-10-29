"""Generate the complete rational game, using only the standard library."""
import argparse
import itertools
import json
from pathlib import Path

triples = list(itertools.product((1, -1), repeat=3))
rows = [s for s in triples if s[0]*s[1]*s[2] == 1]
cols = [[s for s in triples if s[0]*s[1]*s[2] == p] for p in (1, 1, -1)]
chosen = [(1, 1, 1), (1, -1, -1)]
aq = [('row', i) for i in range(3)] + [('bridge', x) for x in range(2)]
bq = [('col', j) for j in range(3)] + [('bridge', y) for y in range(2)]

def answers(q, alice):
    kind, i = q
    if kind == 'row':
        return rows + ['f']
    if kind == 'col':
        return cols[i] + ['f']
    return [s for s in (cols[0] if alice else rows) if s != chosen[i]] + [1, -1]

def accepted(q, r, a, b):
    kind, i = q
    other, j = r
    if kind == 'row' and other == 'col':
        return a == b == 'f' if a == 'f' or b == 'f' else a[j] == b[i]
    if kind == 'bridge' and other == 'col':
        return True if j else (b in ('f', chosen[i]) if isinstance(a, int) else a == b)
    if kind == 'row' and other == 'bridge':
        return True if i else (a in ('f', chosen[j]) if isinstance(b, int) else a == b)
    return True if not isinstance(a, int) or not isinstance(b, int) else a*b == (-1)**(i*j)

def question(h, q, alice):
    kind, i = q
    name = ('R' if kind == 'row' else 'C' if kind == 'col' else 'T' if alice else 'U')
    index = i + 1 if kind in ('row', 'col') else i
    return f'{h}:{name}{index}'

def generate():
    al = {question(h, q, True): answers(q, True) + (['bot'] if h == 2 else [])
          for h in (1, 2) for q in aq}
    bo = {question(h, q, False): answers(q, False) + (['bot'] if h == 2 else [])
          for h in (1, 2) for q in bq}
    bo['label'] = ['A', 'B']
    pairs = []
    for h in (1, 2):
        for q in aq:
            for r in bq:
                x, y = question(h, q, True), question(h, r, False)
                forbidden = []
                for ai, a in enumerate(al[x]):
                    for bi, b in enumerate(bo[y]):
                        win = (a == b == 'bot') if h == 2 and ('bot' in (a, b)) else accepted(q, r, a, b)
                        if not win:
                            forbidden.append([ai, bi])
                pairs.append(dict(alice=x, bob=y, probability=[1, 52], forbidden=forbidden))
        x = question(h, ('row', 0), True)
        forbidden = [[ai, bi] for ai, a in enumerate(al[x]) for bi, b in enumerate(bo['label'])
                     if ((a != 'f') if h == 1 else (a in ('f', 'bot'))) != (b == 'A')]
        pairs.append(dict(alice=x, bob='label', probability=[1, 52], forbidden=forbidden))
    assert len(al) == 10 and len(bo) == 11 and len(pairs) == 52
    assert max(map(len, al.values())) == max(map(len, bo.values())) == 6
    return dict(format='rational-nonlocal-game-1', scoring=[0, 1],
                answer_encoding='Integers are signs; length-three arrays are sign triples; indices are zero based.',
                alice=al, bob=bo, supported_pairs=pairs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('game.json'))
    args = parser.parse_args()
    args.output.write_text(json.dumps(generate(), indent=2) + '\n', encoding='utf-8', newline='\n')
    print('GAME: 10 Alice questions, 11 Bob questions, at most 6 answers, 52 uniform pairs')
