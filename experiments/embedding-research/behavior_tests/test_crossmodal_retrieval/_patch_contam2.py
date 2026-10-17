# -*- coding: utf-8 -*-
"""修正版：不用 chr(10) 拼 f-string，改用普通 print 输出分隔细节。"""
import ast
import io

P = "probe_image_contamination.py"
s = io.open(P, encoding="utf-8").read()

# 回退上一次的坏补丁：截回锚点处
anchor = ('          f"{int(np.isnan(E_img_c[:len(I)]).any(axis=1).sum())}", '
          'flush=True)\n')
if anchor not in s:
    # 已被上一次坏补丁破坏，从备份思路改为按锚点截断到 main 结尾前
    raise SystemExit("锚点缺失，需先恢复原文件")

head = s.split(anchor)[0] + anchor

NL = chr(10)
extra = NL
extra += '    # 稀疏大偏差的形态要归因到具体机制，不能只报有 7% 偏了。' + NL
extra += '    d = 1 - np.array([cosmat(T1[i:i + 1], E_true_c[i:i + 1])[0, 0]' + NL
extra += '                      for i in range(len(caps))])' + NL
extra += '    bad = np.where(d > 1e-3)[0]' + NL
extra += '    print("", flush=True)' + NL
extra += '    print("偏差 > 0.001 的明细 共 %d 条" % len(bad), flush=True)' + NL
extra += '    for i in bad:' + NL
extra += '        toks = caps[i].split()' + NL
extra += '        print("  [%d] diff=%.5f %d词 norm_cached=%.4f norm_fresh=%.4f | %s"' + NL
extra += '              % (i, d[i], len(toks), np.linalg.norm(E_true_c[i]),' + NL
extra += '                 np.linalg.norm(T1[i]), caps[i][:64]), flush=True)' + NL
extra += '    if len(bad):' + NL
extra += '        lens = [len(caps[i].split()) for i in bad]' + NL
extra += '        alll = [len(c.split()) for c in caps]' + NL
extra += '        print("  坏样本词数 %s" % sorted(lens), flush=True)' + NL
extra += '        print("  全部词数中位 %d，坏样本词数中位 %d"' + NL
extra += '              % (int(np.median(alll)), int(np.median(lens))), flush=True)' + NL
extra += '        print("  坏样本索引 %s" % bad.tolist(), flush=True)' + NL
extra += '        print("  坏样本索引 mod 32 %s" % [int(i) % 32 for i in bad],' + NL
extra += '              flush=True)' + NL
extra += '        r = [np.linalg.norm(E_true_c[i]) / max(np.linalg.norm(T1[i]), 1e-9)' + NL
extra += '             for i in bad]' + NL
extra += '        print("  坏样本 norm 比 cached/fresh 中位 %.4f"' + NL
extra += '              % float(np.median(r)), flush=True)' + NL

tail = NL + NL + 'if __name__ == "__main__":' + NL + '    main()' + NL
out = head + extra + tail
ast.parse(out)
io.open(P, "w", encoding="utf-8").write(out)
print("已重写并通过语法检查")
