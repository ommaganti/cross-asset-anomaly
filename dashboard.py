from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd

from .pairs import STRUCTURAL_PAIRS, expected_sign
from .portfolio import risk_delta_at_date, CANONICAL_PORTFOLIOS


CLASS_COLORS = {
    "equity": "#e15759",
    "fx": "#4e79a7",
    "commodity": "#f28e2b",
    "rates": "#59a14f",
}


def _round_or_none(x, nd):
    if x is None or (isinstance(x, float) and (np.isnan(x) or not np.isfinite(x))):
        return None
    return round(float(x), nd)


def build_dashboard_data(
    pearson_pw_df: pd.DataFrame,        # columns: (a,b) tuples -> rolling corr series
    z_pearson_df: pd.DataFrame,         # same columns -> z-score series
    stress: pd.DataFrame,               # index=date, has total_breaks/pct/regime/stress_z
    pc1: pd.Series,
    rets_uni: pd.DataFrame,             # raw (not vol-adj) returns for the universe
    asset_list: list[str],
    class_map: dict[str, str],
    events: pd.DataFrame,               # index=date, col 'event'
    leadlag_lag_df: pd.DataFrame | None = None,  # columns (a,b) -> best-lag series (structural pairs)
    snapshot_freq: str = "W-FRI",
    primary_window: int = 60,
    risk_current_window: int = 60,
    risk_baseline_window: int = 1260,
) -> dict:
    asset_idx = {a: i for i, a in enumerate(asset_list)}

    # Fixed pair list = columns present in the rolling-corr frame, as [i,j] index pairs
    pair_cols = [c for c in pearson_pw_df.columns if c[0] in asset_idx and c[1] in asset_idx]
    pairs_index = [[asset_idx[a], asset_idx[b]] for (a, b) in pair_cols]

    structural_set = {tuple(sorted((p.a, p.b))) for p in STRUCTURAL_PAIRS}

    # --- Snapshots (weekly): corr + z per pair ---
    corr_dates = pearson_pw_df.dropna(how="all").index
    if len(corr_dates) == 0:
        snap_dates = pd.DatetimeIndex([])
    else:
        weekly = pd.Series(1, index=corr_dates).resample(snapshot_freq).last().dropna().index
        snap_dates = corr_dates[corr_dates.isin(weekly)]
        if len(snap_dates) == 0:
            snap_dates = corr_dates[::5]

    def _lag_at(d, c):
        if leadlag_lag_df is None or c not in leadlag_lag_df.columns or d not in leadlag_lag_df.index:
            return None
        v = leadlag_lag_df.at[d, c]
        if pd.isna(v) or int(v) == 0:
            return None
        return int(v)

    has_leadlag = leadlag_lag_df is not None and not leadlag_lag_df.empty

    snapshots = []
    for d in snap_dates:
        corr_row = pearson_pw_df.loc[d]
        z_row = z_pearson_df.loc[d] if d in z_pearson_df.index else pd.Series(dtype=float)
        corr_arr = [_round_or_none(corr_row.get(c), 2) for c in pair_cols]
        z_arr = [_round_or_none(z_row.get(c) if len(z_row) else None, 1) for c in pair_cols]
        lag_arr = [_lag_at(d, c) for c in pair_cols] if has_leadlag else None
        rk = risk_delta_at_date(rets_uni, d, CANONICAL_PORTFOLIOS["60_40"],
                                current_window=risk_current_window,
                                baseline_window=risk_baseline_window)
        snap = {
            "date": d.strftime("%Y-%m-%d"),
            "corr": corr_arr,
            "z": z_arr,
            "risk": None if rk is None else {
                "delta_total_pct": _round_or_none(rk["delta_total_pct"], 1),
                "delta_corr_only_pct": _round_or_none(rk["delta_corr_only_pct"], 1),
                "vol_current_ann": _round_or_none(rk["vol_current_ann"] * 100, 2),
                "vol_baseline_ann": _round_or_none(rk["vol_baseline_ann"] * 100, 2),
            },
        }
        if lag_arr is not None:
            snap["lag"] = lag_arr
        snapshots.append(snap)

    # --- Daily series for the stress timeline ---
    if not stress.empty:
        sd = stress.dropna(subset=["total_breaks"])
        daily_dates = sd.index
        pc1_aligned = pc1.reindex(daily_dates) if pc1 is not None else pd.Series(index=daily_dates, dtype=float)
        daily = {
            "dates": [d.strftime("%Y-%m-%d") for d in daily_dates],
            "stress": [int(x) for x in sd["total_breaks"].fillna(0).tolist()],
            "pct": [_round_or_none(x, 2) for x in sd.get("pct_universe_broken", pd.Series(index=daily_dates)).tolist()],
            "regime": [None if pd.isna(x) else str(x) for x in sd.get("stress_regime", pd.Series(index=daily_dates)).tolist()],
            "pc1": [_round_or_none(x, 3) for x in pc1_aligned.tolist()],
        }
    else:
        daily = {"dates": [], "stress": [], "pct": [], "regime": [], "pc1": []}

    # --- Events within the daily window ---
    ev = []
    if events is not None and not events.empty and daily["dates"]:
        lo, hi = pd.Timestamp(daily["dates"][0]), pd.Timestamp(daily["dates"][-1])
        for d, row in events.iterrows():
            if lo <= d <= hi:
                ev.append({"date": d.strftime("%Y-%m-%d"), "label": str(row["event"])})

    return {
        "assets": [{"ticker": a, "class": class_map.get(a, "other"),
                    "color": CLASS_COLORS.get(class_map.get(a, "other"), "#888")} for a in asset_list],
        "class_colors": CLASS_COLORS,
        "pairs_index": pairs_index,
        "pair_is_structural": [1 if tuple(sorted(c)) in structural_set else 0 for c in pair_cols],
        "pair_expected_sign": [expected_sign(*c) or "" for c in pair_cols],
        "snapshots": snapshots,
        "daily": daily,
        "events": ev,
        "primary_window": primary_window,
        "meta": {
            "n_snapshots": len(snapshots),
            "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        },
    }


def write_dashboard(data: dict, out_path: str) -> str:
    html = _HTML_TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    with open(out_path, "w") as f:
        f.write(html)
    return out_path


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Cross-Asset Correlation Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
<style>
  :root { --bg:#0f1117; --panel:#171a21; --line:#262b36; --txt:#d7dbe3; --dim:#8b93a3; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--txt);
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
  header { padding:10px 16px; border-bottom:1px solid var(--line); display:flex;
           align-items:center; gap:24px; flex-wrap:wrap; }
  header h1 { font-size:15px; margin:0; font-weight:600; letter-spacing:.3px; }
  .stat { display:flex; flex-direction:column; line-height:1.2; }
  .stat .k { font-size:10px; color:var(--dim); text-transform:uppercase; letter-spacing:.5px; }
  .stat .v { font-size:18px; font-weight:600; }
  .grid { display:grid; grid-template-columns: 1fr 1fr; grid-template-rows: 220px 1fr;
          gap:10px; padding:10px; height:calc(100vh - 56px); }
  .panel { background:var(--panel); border:1px solid var(--line); border-radius:8px;
           position:relative; overflow:hidden; }
  .panel h2 { font-size:11px; color:var(--dim); margin:0; padding:8px 12px;
              text-transform:uppercase; letter-spacing:.6px; border-bottom:1px solid var(--line); }
  #network { grid-row: 1 / 3; }
  .legend { position:absolute; bottom:8px; left:12px; font-size:10px; color:var(--dim); }
  .legend span { display:inline-flex; align-items:center; margin-right:10px; }
  .legend i { width:9px; height:9px; border-radius:50%; margin-right:4px; display:inline-block; }
  svg { display:block; width:100%; }
  .tooltip { position:fixed; pointer-events:none; background:#000d; border:1px solid var(--line);
             padding:6px 8px; border-radius:5px; font-size:11px; z-index:50; display:none; }
  .alert-row { display:flex; justify-content:space-between; font-size:11px; padding:2px 12px; }
  .alert-row .z { font-variant-numeric:tabular-nums; }
  .pill { font-size:9px; padding:1px 5px; border-radius:8px; margin-left:6px; }
  .scrub-label { font-size:11px; fill:var(--txt); }
  .big { font-size:34px; font-weight:700; }
  .sub { font-size:11px; color:var(--dim); }
  .risk-wrap { padding:12px; display:flex; gap:20px; align-items:flex-start; }
  .evsel { background:#0e1015; color:var(--txt); border:1px solid var(--line);
           border-radius:5px; font-size:12px; padding:3px 6px; margin-top:2px; max-width:240px; }
</style>
</head>
<body>
<header>
  <h1>CROSS-ASSET CORRELATION ANOMALY MONITOR</h1>
  <div class="stat"><span class="k">Date</span><span class="v" id="hdr-date">—</span></div>
  <div class="stat"><span class="k">Stress regime</span><span class="v" id="hdr-regime">—</span></div>
  <div class="stat"><span class="k">Concurrent breaks</span><span class="v" id="hdr-breaks">—</span></div>
  <div class="stat"><span class="k">60/40 corr-risk Δ</span><span class="v" id="hdr-risk">—</span></div>
  <div class="stat"><span class="k">Jump to event</span>
       <select id="event-jump" class="evsel"><option value="">— select —</option></select></div>
  <div class="stat" style="margin-left:auto"><span class="k">◀ drag the timeline / press space to play ▶</span>
       <span class="sub" id="hdr-meta"></span></div>
</header>
<div class="grid">
  <div class="panel" id="p-stress"><h2>Systemic stress index — drag to replay history</h2></div>
  <div class="panel" id="network"><h2>Correlation network — anomalous links glow red, nodes drift as relationships break</h2>
    <div class="legend" id="net-legend"></div>
  </div>
  <div class="panel" id="p-heat"><h2>Z-score heatmap (60d corr vs own history)</h2></div>
  <div class="panel" id="p-risk"><h2>Portfolio risk & top breaks at selected date</h2></div>
</div>
<div class="tooltip" id="tip"></div>
<script>
const DATA = __DATA__;
const tip = d3.select("#tip");
function showTip(html, e){ tip.html(html).style("display","block")
  .style("left",(e.clientX+12)+"px").style("top",(e.clientY+12)+"px"); }
function hideTip(){ tip.style("display","none"); }

const A = DATA.assets, N = A.length;
const PI = DATA.pairs_index, STRUCT = DATA.pair_is_structural, ESIGN = DATA.pair_expected_sign;
const SNAPS = DATA.snapshots, DAILY = DATA.daily;
let active = SNAPS.length - 1;   // start at most recent

// ---------- NETWORK ----------
const netEl = document.getElementById("network");
const nW = () => netEl.clientWidth, nH = () => netEl.clientHeight - 28;
const netSvg = d3.select("#network").append("svg").attr("height", "calc(100% - 28px)");
const gLink = netSvg.append("g").attr("id","glink"),
      gNode = netSvg.append("g").attr("id","gnode"),
      gArrow = netSvg.append("g").attr("id","garrow");
const nodes = A.map((a,i)=>({i, ticker:a.ticker, color:a.color, cls:a.class,
                             x: nW()/2 + 120*Math.cos(i/N*6.28), y: nH()/2 + 120*Math.sin(i/N*6.28)}));
let links = PI.map((p,k)=>({k, source:nodes[p[0]], target:nodes[p[1]], corr:0, z:0, lag:null,
                            struct:STRUCT[k], esign:ESIGN[k]}));
const CORR_EDGE_MIN = 0.30;

const sim = d3.forceSimulation(nodes)
  .force("charge", d3.forceManyBody().strength(-180))
  .force("center", d3.forceCenter(nW()/2, nH()/2))
  .force("collide", d3.forceCollide(22))
  .force("link", d3.forceLink(links).id(d=>d.i).distance(d=>120*(1-Math.abs(d.corr)))
                   .strength(d=> Math.abs(d.corr) >= CORR_EDGE_MIN ? Math.min(Math.abs(d.corr),0.9) : 0.002))
  .on("tick", ticked);

const linkSel = gLink.selectAll("line");
const nodeSel = gNode.selectAll("g");

function zColor(z){
  if (z===null) return "rgba(120,120,120,0.05)";
  const az = Math.abs(z);
  if (az < 2.0) return "rgba(130,138,150,0.10)";
  if (az < 2.5) return "rgba(240,170,70,"+Math.min(az/4,0.5)+")";
  return "rgba(214,40,40,"+Math.min(az/4,0.95)+")";          // anomalous = red glow
}
function nodeAnomaly(i, snap){
  let s=0; PI.forEach((p,k)=>{ if((p[0]===i||p[1]===i) && snap.z[k]!==null) s+=Math.abs(snap.z[k]); });
  return s;
}

function buildNetwork(){
  let lsel = gLink.selectAll("line").data(links);
  lsel.exit().remove();
  linkAll = lsel.enter().append("line").merge(lsel)
    .on("mousemove",(e,d)=>showTip(
      `<b>${A[d.source.i].ticker} ↔ ${A[d.target.i].ticker}</b><br>corr ${d.corr??'–'} · z ${d.z??'–'}`+
      (d.lag?`<br><span style="color:#7fd8ff">lead-lag: ${d.lag>0?A[d.source.i].ticker+' leads by '+d.lag:A[d.target.i].ticker+' leads by '+(-d.lag)} bars</span>`:'')+
      (d.struct?`<br><span style="color:#f0a">structural${d.esign?': expect '+d.esign:''}</span>`:''), e))
    .on("mouseleave",hideTip);

  let nsel = gNode.selectAll("g").data(nodes);
  const ng = nsel.enter().append("g").call(d3.drag()
      .on("start",(e,d)=>{ if(!e.active) sim.alphaTarget(.3).restart(); d.fx=d.x; d.fy=d.y; })
      .on("drag",(e,d)=>{ d.fx=e.x; d.fy=e.y; })
      .on("end",(e,d)=>{ if(!e.active) sim.alphaTarget(0); d.fx=null; d.fy=null; }));
  ng.append("circle").attr("stroke","#0b0d12").attr("stroke-width",1.5);
  ng.append("text").attr("text-anchor","middle").attr("dy",3)
    .attr("font-size",9).attr("fill","#fff").attr("pointer-events","none").text(d=>d.ticker.replace("=X","").replace("DX-Y.NYB","DXY"));
  window.linkAll = gLink.selectAll("line");
  window.nodeAll = gNode.selectAll("g");
}

function ticked(){
  window.linkAll && window.linkAll
    .attr("x1",d=>d.source.x).attr("y1",d=>d.source.y)
    .attr("x2",d=>d.target.x).attr("y2",d=>d.target.y);
  window.nodeAll && window.nodeAll.attr("transform",d=>`translate(${d.x},${d.y})`);
  window.arrowAll && window.arrowAll.attr("transform",d=>{
    const sx=d.source.x, sy=d.source.y, tx=d.target.x, ty=d.target.y;
    const mx=sx+(tx-sx)*0.55, my=sy+(ty-sy)*0.55;
    let ang=Math.atan2(ty-sy,tx-sx)*180/Math.PI;
    if(d.lag<0) ang+=180;            // arrowhead points toward the laggard
    return `translate(${mx},${my}) rotate(${ang})`;
  });
}

function applySnapshot(idx, reheat){
  const snap = SNAPS[idx];
  links.forEach((l,k)=>{ l.corr = snap.corr[k]===null?0:snap.corr[k]; l.z = snap.z[k];
                         l.lag = snap.lag ? snap.lag[k] : null; });
  const vis = d=> Math.abs(d.corr)>=CORR_EDGE_MIN || (d.lag!==null && d.lag!==undefined);
  window.linkAll && window.linkAll
    .attr("stroke", d=> vis(d) ? zColor(d.z) : "rgba(120,120,120,0.0)")
    .attr("stroke-width", d=> vis(d) ? 0.6+2.6*Math.abs(d.corr) : 0);
  // lead-lag direction arrows (any shifted structural relationship)
  const lagged = links.filter(d=> d.lag!==null && d.lag!==undefined);
  const asel = gArrow.selectAll("path").data(lagged, d=>d.k);
  asel.exit().remove();
  asel.enter().append("path").attr("d","M0,-3.2L6.5,0L0,3.2Z").attr("fill","#7fd8ff").attr("opacity",0.95);
  window.arrowAll = gArrow.selectAll("path");
  window.nodeAll && window.nodeAll.select("circle")
    .attr("r", d=> 6 + 1.4*Math.sqrt(nodeAnomaly(d.i, snap)))
    .attr("fill", d=> d.color);
  sim.force("link").strength(d=> Math.abs(d.corr) >= CORR_EDGE_MIN ? Math.min(Math.abs(d.corr),0.9) : 0.002);
  if (reheat){ sim.alpha(0.5).restart(); }
  // header + dependent panels
  document.getElementById("hdr-date").textContent = snap.date;
  const di = DAILY.dates.indexOf(snap.date);
  const reg = di>=0 ? DAILY.regime[di] : "—";
  document.getElementById("hdr-regime").textContent = reg || "—";
  document.getElementById("hdr-regime").style.color =
     reg==="extreme"?"#d62828":reg==="elevated"?"#f0a850":"#59a14f";
  document.getElementById("hdr-breaks").textContent = di>=0 ? DAILY.stress[di] : "—";
  const rk = snap.risk;
  const hr = document.getElementById("hdr-risk");
  if (rk){ hr.textContent = (rk.delta_corr_only_pct>=0?"+":"")+rk.delta_corr_only_pct+"%";
           hr.style.color = rk.delta_corr_only_pct>=10?"#d62828":rk.delta_corr_only_pct<=-10?"#59a14f":"#d7dbe3"; }
  else hr.textContent="—";
  drawHeat(snap); drawRisk(snap); moveCursor(snap.date);
}

// ---------- STRESS TIMELINE ----------
const stEl = document.getElementById("p-stress");
const stSvg = d3.select("#p-stress").append("svg").attr("height","calc(100% - 28px)");
let stX, cursorLine;
function drawStress(){
  stSvg.selectAll("*").remove();
  const w = stEl.clientWidth, h = stEl.clientHeight - 28, m={l:34,r:10,t:8,b:18};
  const dates = DAILY.dates.map(d=>new Date(d));
  stX = d3.scaleTime().domain(d3.extent(dates)).range([m.l, w-m.r]);
  const y = d3.scaleLinear().domain([0, d3.max(DAILY.stress)||1]).range([h-m.b, m.t]);
  // regime bands
  const regColor = {calm:"rgba(89,161,79,0.05)", elevated:"rgba(240,168,80,0.08)", extreme:"rgba(214,40,40,0.13)"};
  for(let i=0;i<dates.length;i++){
    const c = regColor[DAILY.regime[i]]; if(!c) continue;
    const x0 = stX(dates[i]), x1 = i+1<dates.length?stX(dates[i+1]):x0+1;
    stSvg.append("rect").attr("x",x0).attr("y",m.t).attr("width",Math.max(x1-x0,1)).attr("height",h-m.b-m.t).attr("fill",c);
  }
  const area = d3.area().x((d,i)=>stX(dates[i])).y0(y(0)).y1((d,i)=>y(DAILY.stress[i]));
  stSvg.append("path").datum(DAILY.stress).attr("fill","rgba(214,40,40,0.45)").attr("d",area);
  // PC1 factor-structure trace on a secondary axis (systemic co-movement)
  const pc1 = (DAILY.pc1||[]).map(v=>v===null?NaN:v);
  if (pc1.some(v=>!isNaN(v))){
    const pe = d3.extent(pc1.filter(v=>!isNaN(v)));
    const yP = d3.scaleLinear().domain([pe[0]*0.95, pe[1]*1.05]).range([h-m.b, m.t]);
    const line = d3.line().defined((d,i)=>!isNaN(pc1[i])).x((d,i)=>stX(dates[i])).y((d,i)=>yP(pc1[i]));
    stSvg.append("path").datum(pc1).attr("fill","none").attr("stroke","#4ea0e0")
         .attr("stroke-width",1.2).attr("opacity",0.85).attr("d",line);
    stSvg.append("g").attr("transform",`translate(${w-m.r},0)`).attr("color","#4ea0e0")
         .call(d3.axisRight(yP).ticks(3).tickFormat(d3.format(".0%")).tickSize(0)).attr("font-size",8);
    stSvg.append("text").attr("x",w-m.r-2).attr("y",m.t+2).attr("text-anchor","end")
         .attr("font-size",9).attr("fill","#4ea0e0").text("PC1 share");
  }
  stSvg.append("g").attr("transform",`translate(0,${h-m.b})`).attr("color","#5b6273")
       .call(d3.axisBottom(stX).ticks(8).tickSize(0)).attr("font-size",9);
  stSvg.append("g").attr("transform",`translate(${m.l},0)`).attr("color","#a8475a")
       .call(d3.axisLeft(y).ticks(3).tickSize(0)).attr("font-size",9);
  // events
  DATA.events.forEach(ev=>{ const x=stX(new Date(ev.date));
    stSvg.append("line").attr("x1",x).attr("x2",x).attr("y1",m.t).attr("y2",h-m.b)
      .attr("stroke","rgba(255,255,255,0.12)");
    stSvg.append("circle").attr("cx",x).attr("cy",m.t+3).attr("r",3).attr("fill","#cdd3df")
      .style("cursor","pointer").on("mousemove",(e)=>showTip("<b>"+ev.date+"</b><br>"+ev.label,e)).on("mouseleave",hideTip);
  });
  cursorLine = stSvg.append("line").attr("y1",m.t).attr("y2",h-m.b)
      .attr("stroke","#fff").attr("stroke-width",1.5).attr("opacity",0.9);
  stSvg.append("rect").attr("x",m.l).attr("y",m.t).attr("width",w-m.l-m.r).attr("height",h-m.b-m.t)
       .attr("fill","transparent").style("cursor","col-resize")
       .on("mousedown",function(e){ scrubbing=true; pickDate(e); })
       .on("mousemove",function(e){ if(scrubbing) pickDate(e); })
       .on("mouseup",()=>scrubbing=false).on("mouseleave",()=>scrubbing=false);
}
let scrubbing=false;
function pickDate(e){
  const mx = d3.pointer(e)[0]; const t = stX.invert(mx);
  // nearest snapshot
  let best=0, bd=Infinity;
  SNAPS.forEach((s,i)=>{ const dd=Math.abs(new Date(s.date)-t); if(dd<bd){bd=dd;best=i;} });
  if(best!==active){ active=best; applySnapshot(active, true); } else moveCursor(SNAPS[active].date);
}
function moveCursor(dateStr){ if(cursorLine) cursorLine.attr("x1",stX(new Date(dateStr))).attr("x2",stX(new Date(dateStr))); }

// ---------- HEATMAP ----------
const heatEl = document.getElementById("p-heat");
const heatSvg = d3.select("#p-heat").append("svg").attr("height","calc(100% - 28px)");
const zScale = d3.scaleDiverging(d3.interpolateRdBu).domain([4,0,-4]);
function drawHeat(snap){
  heatSvg.selectAll("*").remove();
  const w=heatEl.clientWidth, h=heatEl.clientHeight-28, m={l:42,t:6,r:6,b:36};
  const cw=(w-m.l-m.r)/N, ch=(h-m.t-m.b)/N;
  const Z = Array.from({length:N},()=>Array(N).fill(null));
  PI.forEach((p,k)=>{ Z[p[0]][p[1]]=snap.z[k]; Z[p[1]][p[0]]=snap.z[k]; });
  for(let r=0;r<N;r++) for(let c=0;c<N;c++){
    const z=Z[r][c];
    heatSvg.append("rect").attr("x",m.l+c*cw).attr("y",m.t+r*ch).attr("width",cw-0.5).attr("height",ch-0.5)
      .attr("fill", r===c?"#333":(z===null?"#14161c":zScale(Math.max(-4,Math.min(4,z)))))
      .on("mousemove",(e)=> z!==null && showTip(`<b>${A[r].ticker} ↔ ${A[c].ticker}</b><br>z ${z}`,e))
      .on("mouseleave",hideTip);
  }
  A.forEach((a,i)=>{
    heatSvg.append("text").attr("x",m.l-3).attr("y",m.t+i*ch+ch/2+3).attr("text-anchor","end")
      .attr("font-size",7).attr("fill",a.color).text(a.ticker.replace("=X","").replace("DX-Y.NYB","DXY"));
    heatSvg.append("text").attr("x",m.l+i*cw+cw/2).attr("y",h-m.b+12).attr("text-anchor","end")
      .attr("transform",`rotate(-90,${m.l+i*cw+cw/2},${h-m.b+12})`)
      .attr("font-size",7).attr("fill",a.color).text(a.ticker.replace("=X","").replace("DX-Y.NYB","DXY"));
  });
}

// ---------- RISK PANEL ----------
const riskEl = document.getElementById("p-risk");
const riskDiv = d3.select("#p-risk").append("div").attr("id","risk-body");
function drawRisk(snap){
  const rk = snap.risk;
  // top breaks
  const arr = PI.map((p,k)=>({a:A[p[0]].ticker,b:A[p[1]].ticker,z:snap.z[k],struct:STRUCT[k],
                              corr:snap.corr[k],esign:ESIGN[k]}))
                .filter(d=>d.z!==null).sort((x,y)=>Math.abs(y.z)-Math.abs(x.z)).slice(0,8);
  let html = '<div class="risk-wrap">';
  if(rk){
    const col = rk.delta_corr_only_pct>=10?"#d62828":rk.delta_corr_only_pct<=-10?"#59a14f":"#d7dbe3";
    html += `<div style="min-width:150px">
      <div class="sub">60/40 vol now vs baseline</div>
      <div class="big">${rk.vol_current_ann}%<span class="sub"> vs ${rk.vol_baseline_ann}%</span></div>
      <div class="sub" style="margin-top:6px">correlation-driven risk Δ</div>
      <div class="big" style="color:${col}">${rk.delta_corr_only_pct>=0?'+':''}${rk.delta_corr_only_pct}%</div>
      <div class="sub" style="margin-top:6px">total risk Δ ${rk.delta_total_pct>=0?'+':''}${rk.delta_total_pct}%</div>
    </div>`;
  } else html += '<div class="sub" style="min-width:150px">risk n/a (insufficient history)</div>';
  html += '<div style="flex:1"><div class="sub" style="padding:0 0 4px 0">Largest correlation anomalies</div>';
  arr.forEach(d=>{
    const col = Math.abs(d.z)>=2.5?"#ff6b6b":Math.abs(d.z)>=2?"#f0a850":"#9aa3b2";
    const viol = d.struct && d.esign && ((d.esign==="positive"&&d.corr<0)||(d.esign==="negative"&&d.corr>0));
    html += `<div class="alert-row"><span>${d.a} ↔ ${d.b}`+
      (d.struct?`<span class="pill" style="background:#3a2a4a;color:#d9a">struct</span>`:'')+
      (viol?`<span class="pill" style="background:#5a1a1a;color:#f88">SIGN FLIP</span>`:'')+
      `</span><span class="z" style="color:${col}">z ${d.z>0?'+':''}${d.z}</span></div>`;
  });
  html += '</div></div>';
  riskDiv.html(html);
}

// ---------- LEGEND + PLAY ----------
const leg = document.getElementById("net-legend");
leg.innerHTML = Object.entries(DATA.class_colors).map(([k,v])=>
  `<span><i style="background:${v}"></i>${k}</span>`).join("") +
  `<span><i style="background:#d62828"></i>anomalous link</span>` +
  `<span><span style="color:#7fd8ff;font-size:12px">▶</span>&nbsp;lead-lag (points to laggard)</span>`;
document.getElementById("hdr-meta").textContent =
  DATA.meta.n_snapshots + " weekly snapshots · generated " + DATA.meta.generated;

// Jump-to-event picker
function nearestSnap(dateStr){
  const t = new Date(dateStr); let best=0, bd=Infinity;
  SNAPS.forEach((s,i)=>{ const dd=Math.abs(new Date(s.date)-t); if(dd<bd){bd=dd;best=i;} });
  return best;
}
const evSel = document.getElementById("event-jump");
DATA.events.slice().reverse().forEach(ev=>{
  const o=document.createElement("option"); o.value=ev.date; o.textContent=ev.date+"  "+ev.label;
  evSel.appendChild(o);
});
evSel.addEventListener("change",()=>{ if(!evSel.value) return;
  active=nearestSnap(evSel.value); applySnapshot(active,true); });

let playing=false, timer=null;
function togglePlay(){ playing=!playing;
  if(playing){ timer=setInterval(()=>{ active=(active+1)%SNAPS.length; applySnapshot(active,false);
    if(active===0) moveCursor(SNAPS[0].date); }, 120); }
  else clearInterval(timer); }
window.addEventListener("keydown",e=>{ if(e.code==="Space"){ e.preventDefault(); togglePlay(); }});

function render(){ drawStress(); buildNetwork(); applySnapshot(active, true); }
window.addEventListener("resize", ()=>{ drawStress(); applySnapshot(active,false); });
render();
</script>
</body>
</html>
"""
