const fs = require('fs');
const path = require('path');

const BOX_CHARS = '┌└┐┘│├┤┬┴┼─═║╔╗╚╝╠╣╦╩╬▲►▼◄►▸▹▶▷▾◀◆■□▪▫○●⊕⊗∞≈≡≠≤≥±×÷←→↑↓↔↕↖↗↘↙↩↪↻↺↼⇄⇅';
const MD_DIR = 'docs/zh';
const REPORT_PATH = process.argv[2];
if (!REPORT_PATH) throw new Error('Usage: node scan_diagrams.js <report-path>');

function walkSync(dir, cb) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const e of entries) {
    if (e.name.startsWith('.')) continue;
    const p = path.join(dir, e.name);
    if (e.isDirectory()) walkSync(p, cb);
    else if (e.name.endsWith('.md')) cb(p);
  }
}

const fileBlocks = {};

walkSync(MD_DIR, (filePath) => {
  try {
    const content = fs.readFileSync(filePath, 'utf-8');
    const lines = content.split('\n');
    const blocks = [];

    // 1. Unicode art blocks
    let blockStart = null, blockLines = [];
    for (let i = 0; i < lines.length; i++) {
      const hasBoxChar = [...BOX_CHARS].some(c => lines[i].includes(c));
      if (hasBoxChar) {
        if (!blockStart) blockStart = i + 1;
        blockLines.push(lines[i]);
      } else if (blockStart !== null) {
        if (blockLines.length >= 2) {
          blocks.push({
            type: 'unicode_art',
            start: blockStart,
            end: i,
            sample: blockLines.slice(0, 4),
            totalLines: blockLines.length
          });
        }
        blockStart = null;
        blockLines = [];
      }
    }
    if (blockStart && blockLines.length >= 2) {
      blocks.push({
        type: 'unicode_art',
        start: blockStart,
        end: lines.length,
        sample: blockLines.slice(0, 4),
        totalLines: blockLines.length
      });
    }

    // 2. Mermaid code blocks
    let inMermaid = false, mmStart = null;
    for (let i = 0; i < lines.length; i++) {
      if (lines[i].includes('```mermaid')) {
        inMermaid = true; mmStart = i + 1;
      } else if (inMermaid && lines[i].trim().startsWith('```')) {
        blocks.push({
          type: 'mermaid',
          start: mmStart,
          end: i,
          sample: lines.slice(mmStart, Math.min(mmStart + 5, i)).map(l => l.slice(0, 120)),
          totalLines: i - mmStart
        });
        inMermaid = false; mmStart = null;
      }
    }

    if (blocks.length > 0) fileBlocks[filePath] = blocks;
  } catch(e) { /* skip unreadable */ }
});

function classifyMermaid(text) {
  if (/^flowchart|^graph\s+(TD|TB|BT|LR|RL|TOP|BOTTOM)/m.test(text)) return 'flowchart';
  if (/^sequenceDiagram/m.test(text)) return 'sequence';
  if (/^classDiagram/m.test(text)) return 'class';
  if (/^erDiagram/m.test(text)) return 'er';
  if (/^gantt/m.test(text)) return 'gantt';
  if (/^pie/m.test(text)) return 'pie';
  if (/^mindmap/m.test(text)) return 'mindmap';
  if (/^stateDiagram/m.test(text)) return 'state';
  if (/^journey/m.test(text)) return 'journey';
  return 'other_mermaid';
}

function classifyUnicode(lines) {
  const text = lines.join('\n');
  if (/[├└─│┌┐┘┴┬┼]/.test(text) && /[▼►]/.test(text)) return 'tree_flow';
  if (/[├└─│┌┐┘┴┬┼]/.test(text)) return 'tree';
  if (/[▼▲]/.test(text)) return 'flow_with_direction';
  if (/[►▸▶▷]/.test(text)) return 'arrow_flow';
  if (/[│─┌┐└┘]/.test(text)) return 'box_drawing';
  return 'unicode_other';
}

const TYPE_LABELS = {
  flowchart: 'Mermaid Flowchart',
  sequence: 'Mermaid Sequence Diagram',
  class: 'Mermaid Class Diagram',
  er: 'Mermaid ER Diagram',
  gantt: 'Mermaid Gantt',
  pie: 'Mermaid Pie Chart',
  mindmap: 'Mermaid Mindmap',
  state: 'Mermaid State Diagram',
  journey: 'Mermaid Journey',
  other_mermaid: 'Mermaid (其他)',
  tree_flow: 'Unicode 树形流向图',
  tree: 'Unicode 树形结构',
  flow_with_direction: 'Unicode 带方向流程',
  arrow_flow: 'Unicode 箭头流程',
  box_drawing: 'Unicode 框图 (│─┌┐└┘├┤┬┴┼)',
  unicode_other: 'Unicode 其他字符图'
};

const totalFiles = Object.keys(fileBlocks).length;
const totalBlocks = Object.values(fileBlocks).reduce((s, b) => s + b.length, 0);
const totalLinesCount = Object.values(fileBlocks).reduce(
  (s, blocks) => s + blocks.reduce((ss, b) => ss + b.totalLines, 0), 0
);

let report = `# ASCII/Unicode 流程图扫描报告\n\n`;
report += `> 扫描目录：docs/zh\n`;
report += `> 生成时间：${new Date().toLocaleString('zh-CN', {timeZone:'Asia/Shanghai'})}\n`;
report += `> 扫描范围：docs/zh/ 下所有 .md 文件（共 ${62} 个）\n\n`;
report += '---\n\n';

report += `## 统计摘要\n\n`;
report += `| 指标 | 数值 |\n|---|---|\n`;
report += `| 包含图表的文件数 | ${totalFiles} |\n`;
report += `| 图表块总数 | ${totalBlocks} |\n`;
report += `| 图表总行数 | ${totalLinesCount} |\n\n`;

// Type summary
const typeCount = {};
for (const blocks of Object.values(fileBlocks)) {
  for (const block of blocks) {
    let subType;
    if (block.type === 'mermaid') {
      subType = classifyMermaid(block.sample.join('\n'));
    } else {
      subType = classifyUnicode(block.sample);
    }
    typeCount[subType] = (typeCount[subType] || 0) + 1;
  }
}

report += `### 图表类型分布\n\n`;
report += `| 类型 | 块数 | 说明 |\n|---|---|---|\n`;
const TYPE_DESC = {
  flowchart: '流程图 / 方框图',
  sequence: '时序图',
  class: '类图',
  er: '实体关系图',
  gantt: '甘特图',
  pie: '饼图',
  mindmap: '思维导图',
  state: '状态图',
  journey: '旅程图',
  other_mermaid: '其他 Mermaid 类型',
  tree_flow: '树形流程（含方向箭头）',
  tree: '树形目录/架构',
  flow_with_direction: '带方向标记的流程',
  arrow_flow: '纯箭头连线图',
  box_drawing: '纯框线 ASCII 图'
};
for (const [t, c] of Object.entries(typeCount).sort((a, b) => b[1] - a[1])) {
  report += `| ${TYPE_LABELS[t]} | ${c} | ${TYPE_DESC[t] || ''} |\n`;
}
report += '\n---\n\n';

report += `## 详细清单\n\n`;
report += `按文件分组，每块列出行号范围、类型与内容预览。\n\n`;

const sortedFiles = Object.keys(fileBlocks).sort();
for (const filePath of sortedFiles) {
  const blocks = fileBlocks[filePath];
  const relPath = filePath.replace('docs/zh/', '');
  report += `### 📄 ${relPath}\n\n`;
  
  const fileTypeSet = new Set();
  for (const block of blocks) {
    let typeLabel, detail;
    if (block.type === 'mermaid') {
      const subType = classifyMermaid(block.sample.join('\n'));
      typeLabel = TYPE_LABELS[subType];
      fileTypeSet.add(typeLabel);
      detail = block.sample.map(l => '  `' + l.replace(/`/g, "'") + '`').join('\n');
    } else {
      const subType = classifyUnicode(block.sample);
      typeLabel = TYPE_LABELS[subType];
      fileTypeSet.add(typeLabel);
      detail = block.sample.map(l => '  `' + l.replace(/`/g, "'") + '`').join('\n');
    }
    report += `**行 ${block.start}–${block.end}**　` + `**${block.totalLines} 行**　` + `类型：${typeLabel}\n\n`;
    report += detail + '\n\n';
  }
  
  report += `> **本文件图表类型**：${[...fileTypeSet].join(' / ')}\n\n`;
  report += '---\n\n';
}

fs.writeFileSync(REPORT_PATH, report, 'utf-8');
console.log(`Done. Files=${totalFiles}, Blocks=${totalBlocks}, TotalLines=${totalLinesCount}`);
