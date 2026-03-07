#!/usr/bin/env python3
"""
Search Tree Visualizer for io-chess-engine.
Professional visualization with mate/TB score formatting.
"""

import subprocess
import json
import sys
import os
import re

ENGINE_PATH = "../engine/build/chess_engine"
MODEL_PATH = "/home/io/USI/io-bot/io-chess-engine/training/onnx/light_moe_64f4b"
OUTPUT_HTML = "search_tree.html"
DEFAULT_FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"

def generate_html(edges, start_fen):
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Search Tree Analysis</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/chess.js/0.10.3/chess.min.js"></script>
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #1e293b;
            --border: #334155;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --accent: #3b82f6;
            --positive: #22c55e;
            --negative: #ef4444;
            --mate: #f59e0b;
            --tablebase: #a855f7;
        }}
        
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Inter', -apple-system, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            overflow: hidden;
        }}
        
        #container {{ width: 100vw; height: 100vh; }}
        
        .node {{ cursor: pointer; }}
        
        .node-card {{
            fill: var(--bg-card);
            stroke: var(--border);
            stroke-width: 1.5px;
            rx: 8;
            transition: all 0.15s ease;
        }}
        
        .node-card:hover {{ stroke: var(--accent); stroke-width: 2px; }}
        .node-card.positive {{ stroke: var(--positive); }}
        .node-card.negative {{ stroke: var(--negative); }}
        .node-card.mate {{ stroke: var(--mate); stroke-width: 2px; }}
        .node-card.tablebase {{ stroke: var(--tablebase); stroke-width: 2px; }}
        .node-card.root {{ stroke: var(--accent); stroke-width: 2.5px; }}
        
        .link {{
            fill: none;
            stroke: var(--border);
            stroke-width: 1.5px;
            stroke-opacity: 0.5;
        }}
        
        .move-label {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            font-weight: 600;
            fill: var(--text-primary);
        }}
        
        .score-label {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            fill: var(--text-secondary);
        }}
        
        .score-label.positive {{ fill: var(--positive); }}
        .score-label.negative {{ fill: var(--negative); }}
        .score-label.mate {{ fill: var(--mate); font-weight: 600; }}
        .score-label.tablebase {{ fill: var(--tablebase); font-weight: 600; }}
        
        .more-circle {{
            fill: var(--bg-secondary);
            stroke: var(--accent);
            stroke-width: 1.5px;
            cursor: pointer;
        }}
        
        .more-circle:hover {{ fill: #2563eb; }}
        
        .more-text {{
            font-size: 10px;
            fill: var(--accent);
            font-weight: 600;
            pointer-events: none;
        }}
        
        .panel {{
            position: fixed;
            background: rgba(30, 41, 59, 0.95);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            backdrop-filter: blur(12px);
            z-index: 1000;
        }}
        
        #info-panel {{
            top: 16px;
            left: 16px;
            width: 220px;
        }}
        
        #pv-panel {{
            top: 16px;
            right: 16px;
            width: 320px;
        }}
        
        .panel h3 {{
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        #info-panel h3 {{ color: var(--accent); }}
        #pv-panel h3 {{ color: var(--mate); }}
        
        .panel p {{
            font-size: 11px;
            color: var(--text-secondary);
            line-height: 1.6;
        }}
        
        kbd {{
            background: var(--bg-primary);
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 1px 5px;
            font-size: 10px;
            font-family: 'JetBrains Mono', monospace;
        }}
        
        .legend {{
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid var(--border);
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
            font-size: 10px;
            color: var(--text-secondary);
        }}
        
        .legend-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        
        .pv-line {{
            margin: 6px 0;
            padding: 10px 12px;
            background: var(--bg-primary);
            border-radius: 8px;
            border-left: 3px solid var(--border);
        }}
        
        .pv-line.best {{ border-left-color: var(--positive); }}
        
        .pv-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }}
        
        .pv-rank {{
            font-size: 10px;
            color: var(--text-secondary);
            font-weight: 500;
        }}
        
        .pv-score {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            font-weight: 600;
        }}
        
        .pv-score.positive {{ color: var(--positive); }}
        .pv-score.negative {{ color: var(--negative); }}
        .pv-score.mate {{ color: var(--mate); }}
        .pv-score.tablebase {{ color: var(--tablebase); }}
        
        .pv-moves {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: var(--text-primary);
            word-break: break-all;
            line-height: 1.5;
        }}
    </style>
</head>
<body>
    <div id="info-panel" class="panel">
        <h3>🌳 Search Tree</h3>
        <p>
            <kbd>Click</kbd> expand/collapse<br>
            <kbd>+N</kbd> load more siblings<br>
            <kbd>Scroll</kbd> zoom <kbd>Drag</kbd> pan
        </p>
        <div class="legend">
            <div class="legend-item"><div class="legend-dot" style="background:var(--accent)"></div>Root</div>
            <div class="legend-item"><div class="legend-dot" style="background:var(--positive)"></div>Winning (+0.5+)</div>
            <div class="legend-item"><div class="legend-dot" style="background:var(--negative)"></div>Losing (-0.5-)</div>
            <div class="legend-item"><div class="legend-dot" style="background:var(--mate)"></div>Checkmate</div>
            <div class="legend-item"><div class="legend-dot" style="background:var(--tablebase)"></div>Tablebase</div>
        </div>
    </div>
    
    <div id="pv-panel" class="panel">
        <h3>📊 Principal Variations</h3>
        <div id="pv-lines"></div>
    </div>
    
    <div id="container"></div>
    
    <script>
        const rawEdges = {json.dumps(edges)};
        const START_FEN = "{start_fen}";
        
        // Score formatting constants
        const MATE_THRESHOLD = 25000;
        const TB_THRESHOLD = 800000;
        
        function formatScore(score) {{
            const abs = Math.abs(score);
            const sign = score >= 0 ? '+' : '-';
            
            if (abs >= TB_THRESHOLD) {{
                return {{ text: 'TB' + sign, type: 'tablebase' }};
            }}
            if (abs >= MATE_THRESHOLD) {{
                const movesToMate = Math.ceil((30000 - abs) / 2);
                return {{ text: 'M' + sign + Math.max(1, movesToMate), type: 'mate' }};
            }}
            
            const cp = (score / 100).toFixed(2);
            if (score > 50) return {{ text: cp, type: 'positive' }};
            if (score < -50) return {{ text: cp, type: 'negative' }};
            return {{ text: cp, type: 'neutral' }};
        }}
        
        function getCardClass(score, isRoot) {{
            if (isRoot) return 'node-card root';
            const abs = Math.abs(score);
            if (abs >= TB_THRESHOLD) return 'node-card tablebase';
            if (abs >= MATE_THRESHOLD) return 'node-card mate';
            if (score > 50) return 'node-card positive';
            if (score < -50) return 'node-card negative';
            return 'node-card';
        }}
        
        // Deduplicate edges
        const edgeMap = new Map();
        for (const e of rawEdges) {{
            const key = e.p + '_' + e.m;
            if (!edgeMap.has(key) || e.d > edgeMap.get(key).d) edgeMap.set(key, e);
        }}
        const edges = Array.from(edgeMap.values());
        
        // Build adjacency
        const adj = {{}};
        const allChildren = new Set();
        for (const e of edges) {{
            if (!adj[e.p]) adj[e.p] = [];
            adj[e.p].push({{ hash: e.c, move: e.m, score: e.s, depth: e.d }});
            allChildren.add(e.c);
        }}
        
        let rootHash = null;
        for (const e of edges) {{
            if (!allChildren.has(e.p)) {{ rootHash = e.p; break; }}
        }}
        if (!rootHash && edges.length) rootHash = edges[0].p;
        
        // Build tree
        const TOP_K = 4, MAX_DEPTH = 10;
        let nid = 0;
        
        function buildTree(hash, move, score, path, depth) {{
            if (depth > MAX_DEPTH) return null;
            const node = {{ id: 'n' + (++nid), hash, move: move || 'Root', score: score || 0, path, depth, children: [], _hidden: [] }};
            const ch = adj[hash] || [];
            if (ch.length) {{
                const sorted = [...ch].sort((a, b) => depth % 2 === 0 ? b.score - a.score : a.score - b.score);
                sorted.slice(0, TOP_K).forEach(c => {{
                    const child = buildTree(c.hash, c.move, c.score, [...path, c.move], depth + 1);
                    if (child) node.children.push(child);
                }});
                node._hidden = sorted.slice(TOP_K).map(c => ({{ hash: c.hash, move: c.move, score: c.score, path: [...path, c.move], depth: depth + 1 }}));
            }}
            return node;
        }}
        
        const treeData = buildTree(rootHash, 'Start', 0, [], 0);
        if (!treeData) throw new Error('No tree');
        
        // Extract PVs
        function extractPV(hash, maxD = 10) {{
            const pv = [];
            let cur = hash, d = 0;
            while (d < maxD) {{
                const ch = adj[cur];
                if (!ch || !ch.length) break;
                const sorted = [...ch].sort((a, b) => d % 2 === 0 ? b.score - a.score : a.score - b.score);
                pv.push({{ move: sorted[0].move, score: sorted[0].score }});
                cur = sorted[0].hash;
                d++;
            }}
            return pv;
        }}
        
        const rootCh = adj[rootHash] || [];
        const sortedRoot = [...rootCh].sort((a, b) => b.score - a.score);
        const pvs = sortedRoot.slice(0, 3).map((c, i) => {{
            const cont = extractPV(c.hash, 9);
            return {{ rank: i + 1, score: c.score, moves: [c.move, ...cont.map(x => x.move)] }};
        }});
        
        // Populate PV panel
        const pvContainer = document.getElementById('pv-lines');
        pvs.forEach((pv, i) => {{
            const fmt = formatScore(pv.score);
            const div = document.createElement('div');
            div.className = 'pv-line' + (i === 0 ? ' best' : '');
            div.innerHTML = `
                <div class="pv-header">
                    <span class="pv-rank">Line ${{pv.rank}}</span>
                    <span class="pv-score ${{fmt.type}}">${{fmt.text}}</span>
                </div>
                <div class="pv-moves">${{pv.moves.join(' ')}}</div>
            `;
            pvContainer.appendChild(div);
        }});
        
        // D3 Visualization
        const width = window.innerWidth, height = window.innerHeight;
        const nodeW = 90, nodeH = 115;
        
        const svg = d3.select('#container').append('svg').attr('width', width).attr('height', height);
        const g = svg.append('g');
        
        svg.call(d3.zoom().scaleExtent([0.1, 3]).on('zoom', e => g.attr('transform', e.transform)));
        svg.call(d3.zoom().transform, d3.zoomIdentity.translate(180, height / 2).scale(0.65));
        
        const tree = d3.tree().nodeSize([nodeH + 20, nodeW + 70]);
        let root = d3.hierarchy(treeData);
        root.x0 = 0; root.y0 = 0;
        
        function collapseAll(d) {{ if (d.children) {{ d._children = d.children; d._children.forEach(collapseAll); d.children = null; }} }}
        if (root.children) root.children.forEach(collapseAll);
        
        function renderBoard(path) {{
            const game = new Chess(START_FEN === 'startpos' ? undefined : START_FEN);
            path.forEach(m => {{ if (m && m !== 'Start') try {{ game.move({{ from: m.slice(0,2), to: m.slice(2,4), promotion: m[4] }}); }} catch(e) {{ try {{ game.move(m); }} catch(e2) {{}} }} }});
            const board = game.board(), size = 64, sq = 8;
            let s = '';
            for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) {{
                s += `<rect x="${{c*sq}}" y="${{r*sq}}" width="${{sq}}" height="${{sq}}" fill="${{(r+c)%2 ? '#b58863' : '#f0d9b5'}}"/>`;
                const p = board[r][c];
                if (p) {{
                    const sym = {{'k':'♚','q':'♛','r':'♜','b':'♝','n':'♞','p':'♟','K':'♔','Q':'♕','R':'♖','B':'♗','N':'♘','P':'♙'}}[p.color==='w'?p.type.toUpperCase():p.type];
                    s += `<text x="${{c*sq+sq/2}}" y="${{r*sq+sq/2+2}}" text-anchor="middle" dominant-baseline="middle" font-size="${{sq*0.9}}" fill="${{p.color==='w'?'#fff':'#000'}}" stroke="${{p.color==='w'?'#000':'#fff'}}" stroke-width="0.6">${{sym}}</text>`;
                }}
            }}
            return s;
        }}
        
        const diagonal = (s, d) => `M${{s.y}},${{s.x}} C${{(s.y+d.y)/2}},${{s.x}} ${{(s.y+d.y)/2}},${{d.x}} ${{d.y}},${{d.x}}`;
        
        function update(source) {{
            const t = tree(root), nodes = t.descendants(), links = t.links();
            nodes.forEach(d => d.y = d.depth * 140);
            
            const link = g.selectAll('.link').data(links, d => d.target.data.id);
            link.enter().insert('path', 'g').attr('class', 'link').attr('d', () => {{ const o = {{x: source.x0, y: source.y0}}; return diagonal(o, o); }})
                .merge(link).transition().duration(250).attr('d', d => diagonal(d.source, d.target));
            link.exit().transition().duration(250).attr('d', () => {{ const o = {{x: source.x, y: source.y}}; return diagonal(o, o); }}).remove();
            
            const node = g.selectAll('.node').data(nodes, d => d.data.id);
            const nodeEnter = node.enter().append('g').attr('class', 'node').attr('transform', `translate(${{source.y0}},${{source.x0}})`).on('click', (e, d) => {{ if (d.children) {{ d._children = d.children; d.children = null; }} else if (d._children) {{ d.children = d._children; d._children = null; }} update(d); }});
            
            nodeEnter.append('rect').attr('class', d => getCardClass(d.data.score, d.depth === 0)).attr('width', nodeW).attr('height', nodeH).attr('x', -nodeW/2).attr('y', -nodeH/2);
            nodeEnter.each(function(d) {{ d3.select(this).append('g').attr('transform', `translate(-32, ${{-nodeH/2 + 6}})`).html(renderBoard(d.data.path)); }});
            
            nodeEnter.append('text').attr('class', 'move-label').attr('y', nodeH/2 - 20).attr('text-anchor', 'middle').text(d => d.data.move);
            nodeEnter.append('text').attr('class', d => 'score-label ' + formatScore(d.data.score).type).attr('y', nodeH/2 - 8).attr('text-anchor', 'middle').text(d => d.depth > 0 ? formatScore(d.data.score).text : '');
            
            node.merge(nodeEnter).transition().duration(250).attr('transform', d => `translate(${{d.y}},${{d.x}})`);
            node.exit().transition().duration(250).attr('transform', `translate(${{source.y}},${{source.x}})`).remove();
            
            nodes.forEach(d => {{ d.x0 = d.x; d.y0 = d.y; }});
            
            // More buttons
            const moreData = nodes.filter(d => d.data._hidden && d.data._hidden.length && d.children);
            const more = g.selectAll('.more-node').data(moreData, d => d.data.id + '_m');
            const moreEnter = more.enter().append('g').attr('class', 'more-node').on('click', (e, d) => {{
                e.stopPropagation();
                d.data._hidden.forEach(h => {{ const n = buildTree(h.hash, h.move, h.score, h.path, h.depth); if (n) {{ const hNode = d3.hierarchy(n); collapseAll(hNode); hNode.depth = d.depth + 1; hNode.parent = d; if (!d.children) d.children = []; d.children.push(hNode); }} }});
                d.data._hidden = [];
                update(d);
            }});
            moreEnter.append('circle').attr('class', 'more-circle').attr('r', 18);
            moreEnter.append('text').attr('class', 'more-text').attr('text-anchor', 'middle').attr('dy', 3);
            
            more.merge(moreEnter).transition().duration(250).attr('transform', d => {{ const last = d.children[d.children.length - 1]; return `translate(${{last.y}},${{last.x + nodeH/2 + 24}})`; }});
            more.merge(moreEnter).select('text').text(d => `+${{d.data._hidden.length}}`);
            more.exit().remove();
        }}
        
        update(root);
    </script>
</body>
</html>'''
    return html


def run_engine(fen, depth=6):
    uci_commands = f"""uci
setoption name ExportTree value true
setoption name ExportTreeDepth value {depth}
setoption name Threads value 1
setoption name EvalDevice value ONNX-CPU
ucinewgame
position fen {fen}
go depth {depth}
quit
"""
    cmd = [ENGINE_PATH, "--model", MODEL_PATH]
    print(f"Running engine with FEN: {fen}")
    print(f"Search depth: {depth}")
    try:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, _ = p.communicate(input=uci_commands, timeout=60)
    except Exception as e:
        print(f"Error: {e}")
        return None
    m = re.search(r"info string json_tree (\[.*\])", out)
    if not m:
        print("Error: No JSON tree found")
        print(out[-1000:])
        return None
    try:
        return json.loads(m.group(1))
    except Exception as e:
        print(f"JSON parse error: {e}")
        return None


def main():
    fen = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FEN
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    edges = run_engine(fen, depth)
    if not edges:
        print("Failed to get tree data")
        sys.exit(1)
    output_path = os.path.join(os.path.dirname(__file__), OUTPUT_HTML)
    with open(output_path, 'w') as f:
        f.write(generate_html(edges, fen))
    print(f"Extracted {len(edges)} edges")
    
    # Save raw JSON for importing into v2
    json_path = "search_tree_data.json"
    with open(json_path, 'w') as f:
        json.dump(edges, f)
    print(f"Raw data saved to {json_path}")

    output_path = os.path.join(os.path.dirname(__file__), OUTPUT_HTML)
    with open(output_path, 'w') as f:
        f.write(generate_html(edges, fen))
    print(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    main()

