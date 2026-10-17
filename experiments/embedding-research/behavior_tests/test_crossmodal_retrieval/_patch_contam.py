# -*- coding: utf-8 -*-
"""给 probe_image_contamination.py 追加坏样本明细输出。

避开 heredoc 的反斜杠转义问题，直接以字符串构造插入。
"""
import io

P = "probe_image_contamination.py"
s = io.open(P, encoding="utf-8").read()

anchor = ('          f"{int(np.isnan(E_img_c[:len(I)]).any(axis=1).sum())}", '
          'flush=True)\n')

NL = chr(10)
Q = chr(34) * 3

extra = NL + '    # 稀疏大偏差的形态要归因到具体机制，不能只报有 7% 偏了。' + NL
extra += '    d = 1 - np.array([cosmat(T1[i:i + 1], E_true_c[i:i + 1])[0, 0]' + NL
extra += '                      for i in range(len(caps))])' + NL
extra += '    bad = np.where(d > 1e-3)[0]' + NL
extra += '    print(f"{0}偏差 > 0.001 的 {1} 条明细：".format(' + NL
extra += '        chr(10), len(bad)), flush=True)' + NL
extra += '    for i in bad:' + NL
extra += '        toks = caps[i].split()' + NL
extra += '        print(f"  [{i}] diff={d[i]:.5f} {len(toks)}词 "' + NL
extra += '              f"norm_cached={np.linalg.norm(E_true_c[i]):.4f} "' + NL
extra += '              f"norm_fresh={np.linalg.norm(T1[i]):.4f} "' + NL
extra += '              f"| {caps[i][:64]}", flush=True)' + NL
extra += '    if len(bad):' + NL
extra += '        lens = [len(caps[i].split()) for i in bad]' + NL
extra += '        alll = [len(c.split()) for c in caps]' + NL
extra += '        print(f"  坏样本词数 {sorted(lens)}", flush=True)' + NL
extra += '        print(f"  全部词数中位 {int(np.median(alll))}，"' + NL
extra += '              f"坏样本词数中位 {int(np.median(lens))}", flush=True)' + NL
extra += '        print(f"  坏样本索引 {bad.tolist()}", flush=True)' + NL
extra += '        print(f"  坏样本索引 mod 32 {[int(i) % 32 for i in bad]}",' + NL
extra += '              flush=True)' + NL
extra += '        r = [np.linalg.norm(E_true_c[i]) / max(np.linalg.norm(T1[i]), 1e-9)' + NL
extra += '             for i in bad]' + NL
extra += '        print(f"  坏样本 norm 比 cached/fresh 中位 "' + NL
extra += '              f"{float(np.median(r)):.4f}", flush=True)' + NL

assert anchor in s, "锚点未匹配"
s = s.replace(anchor, anchor + extra)
io.open(P, "w", encoding="utf-8").write(s)
print("已追加，行数", len(s.split(NL)))
import ast
ast.parse(s)
print("语法通过")
