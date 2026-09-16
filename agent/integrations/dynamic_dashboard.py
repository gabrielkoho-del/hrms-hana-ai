"""
agent/integrations/dynamic_dashboard.py
Dynamic Dashboard FastAPI service with embedded single-query dashboard HTML.

Endpoints:
  - GET  /dashboard.html        — serves the single-query dashboard
  - GET  /api/health            — dataset counts / connection status
  - GET  /api/datasets          — available datasets with schema
  - POST /api/query             — execute a structured chart config
  - POST /api/ai-query          — execute a natural-language query
  - GET  /api/kpis              — KPI summary numbers
  - GET  /api/datasets/{name}/data — preview rows for a dataset

Data layer:
  Uses the DAB bridge directly so the frontend can render live charts
  without a Superset license or MCP server.
"""
import json
import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from agent.integrations.dab_client import DABTenantClientManager

logger = logging.getLogger("hr_agent")

# Embedded single-query dashboard HTML
_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>myHR Analytics</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap" rel="stylesheet">
<link rel="icon" href="data:,">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#08080d;--bg2:#0e0e16;--card:#13131e;--card-h:#1a1a2a;--bdr:#1f1f32;--txt:#e4e4ed;--muted:#5e5e78;--accent:#f59e0b;--accent2:#d97706;--glow:rgba(245,158,11,.15);--ok:#10b981;--err:#ef4444;--info:#06b6d4;--side-w:256px;--top-h:58px}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'DM Sans',sans-serif;background:var(--bg);color:var(--txt);overflow-x:hidden;min-height:100vh}
h1,h2,h3,h4,h5{font-family:'Space Grotesk',sans-serif}
::-webkit-scrollbar{width:5px;height:5px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:var(--bdr);border-radius:3px}

.side{position:fixed;left:0;top:0;bottom:0;width:var(--side-w);background:var(--bg2);border-right:1px solid var(--bdr);display:flex;flex-direction:column;z-index:100;transition:transform .3s cubic-bezier(.4,0,.2,1);overflow-y:auto}
.side.hide{transform:translateX(calc(-1 * var(--side-w)))}
.side-brand{padding:18px;border-bottom:1px solid var(--bdr);display:flex;align-items:center;gap:10px;flex-shrink:0}
.side-logo{width:34px;height:34px;background:linear-gradient(135deg,var(--accent),#f97316);border-radius:9px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:13px;color:#000;font-family:'Space Grotesk',sans-serif;flex-shrink:0}
.side-brand span{font-family:'Space Grotesk',sans-serif;font-weight:600;font-size:15px;white-space:nowrap}
.side-nav{padding:10px 8px;flex-shrink:0}
.side-nav a{display:flex;align-items:center;gap:9px;padding:9px 13px;border-radius:7px;color:var(--muted);text-decoration:none;font-size:13px;font-weight:500;transition:all .15s;cursor:pointer}
.side-nav a:hover{background:rgba(255,255,255,.04);color:var(--txt)}
.side-nav a.on{background:var(--glow);color:var(--accent)}
.side-nav a i{width:17px;text-align:center;font-size:13px}
.sec{padding:4px 8px}
.sec-t{font-size:9.5px;font-weight:600;text-transform:uppercase;letter-spacing:1.1px;color:var(--muted);padding:10px 13px 5px}
.ds-item{display:flex;align-items:center;gap:9px;padding:7px 13px;border-radius:7px;color:var(--muted);font-size:12.5px;cursor:pointer;transition:all .15s}
.ds-item:hover{background:rgba(255,255,255,.04);color:var(--txt)}
.ds-item i{width:15px;text-align:center;font-size:11px}
.ds-item .cnt{margin-left:auto;font-size:10.5px;background:rgba(255,255,255,.06);padding:2px 7px;border-radius:9px;flex-shrink:0}
.side-foot{margin-top:auto;padding:14px;border-top:1px solid var(--bdr);flex-shrink:0}

.wrap{margin-left:var(--side-w);min-height:100vh;transition:margin-left .3s cubic-bezier(.4,0,.2,1)}
.wrap.wide{margin-left:0}

.top{height:var(--top-h);border-bottom:1px solid var(--bdr);display:flex;align-items:center;padding:0 20px;gap:14px;background:rgba(8,8,13,.85);backdrop-filter:blur(12px);position:sticky;top:0;z-index:50;flex-wrap:nowrap;overflow-x:auto}
.top-tog{background:none;border:none;color:var(--muted);cursor:pointer;font-size:15px;padding:5px;border-radius:5px;transition:all .15s;flex-shrink:0}
.top-tog:hover{color:var(--txt);background:rgba(255,255,255,.06)}
.bc{font-size:12.5px;color:var(--muted);display:flex;align-items:center;gap:5px;white-space:nowrap;flex-shrink:0;overflow:hidden;text-overflow:ellipsis}
.bc .cur{color:var(--txt);font-weight:500}
.top-f{display:flex;align-items:center;gap:7px;margin-left:auto;flex-shrink:0;flex-wrap:nowrap}
.fsel{background:var(--card);border:1px solid var(--bdr);color:var(--txt);padding:5px 10px;border-radius:5px;font-size:12px;font-family:'DM Sans',sans-serif;cursor:pointer;outline:none;transition:border-color .15s}
.fsel:focus{border-color:var(--accent)}
.tbtn{background:var(--card);border:1px solid var(--bdr);color:var(--muted);padding:5px 11px;border-radius:5px;font-size:12px;cursor:pointer;display:flex;align-items:center;gap:5px;transition:all .15s;font-family:'DM Sans',sans-serif;white-space:nowrap;flex-shrink:0}
.tbtn:hover{border-color:var(--accent);color:var(--accent)}
.conn{font-size:10px;padding:3px 8px;border-radius:10px;display:flex;align-items:center;gap:4px;flex-shrink:0}
.conn.ok{background:rgba(16,185,129,.12);color:var(--ok)}
.conn.err{background:rgba(239,68,68,.12);color:var(--err)}
.conn.ld{background:rgba(245,158,11,.12);color:var(--accent)}
.conn-dot{width:5px;height:5px;border-radius:50%;background:currentColor}

.dash{padding:20px}
.query-banner{background:var(--card);border:1px solid var(--bdr);border-radius:11px;padding:14px 18px;margin-bottom:20px;display:flex;align-items:center;gap:12px;animation:cin .3s ease-out}
.query-banner i{color:var(--accent);font-size:16px;flex-shrink:0}
.query-banner .qb-text{flex:1;min-width:0}
.query-banner .qb-label{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;font-weight:600;margin-bottom:2px}
.query-banner .qb-value{font-size:14px;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.query-banner .qb-interpretation{font-size:11px;color:var(--muted);margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}

.kpi-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin-bottom:20px}
.kpi{background:var(--card);border:1px solid var(--bdr);border-radius:11px;padding:18px;position:relative;overflow:hidden;transition:all .2s}
.kpi:hover{border-color:rgba(255,255,255,.1);transform:translateY(-2px)}
.kpi-ic{width:38px;height:38px;border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:15px;margin-bottom:12px}
.kpi-lb{font-size:11.5px;color:var(--muted);font-weight:500;margin-bottom:5px}
.kpi-vl{font-family:'Space Grotesk',sans-serif;font-size:26px;font-weight:700;line-height:1}
.kpi-td{font-size:11.5px;margin-top:7px;display:flex;align-items:center;gap:4px}
.kpi-td.up{color:var(--ok)}.kpi-td.dn{color:var(--err)}

.cgrid{display:grid;grid-template-columns:repeat(12,1fr);gap:16px}
@media(max-width:768px){.cgrid{grid-template-columns:1fr}.sz-4,.sz-6,.sz-8,.sz-12{grid-column:span 1}}
.sz-4{grid-column:span 4}
.sz-6{grid-column:span 6}
.sz-8{grid-column:span 8}
.sz-12{grid-column:span 12}
.ccard{background:var(--card);border:1px solid var(--bdr);border-radius:12px;overflow:hidden;transition:all .25s;animation:cin .4s ease-out both}
@keyframes cin{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)}}
.ccard:hover{border-color:rgba(255,255,255,.08)}
.ccard-h{display:flex;align-items:center;justify-content:space-between;padding:14px 16px 0}
.ccard-t{font-family:'Space Grotesk',sans-serif;font-size:14px;font-weight:600;display:flex;align-items:center;gap:8px}
.ccard-t .dot{width:7px;height:7px;border-radius:50%;background:var(--accent);animation:pls 2s ease-in-out infinite}
@keyframes pls{0%,100%{opacity:.4}50%{opacity:1}}
.ccard-a{display:flex;gap:1px;opacity:0;transition:opacity .15s}
.ccard:hover .ccard-a{opacity:1}
.ccard-a button{background:none;border:none;color:var(--muted);padding:4px 6px;border-radius:4px;cursor:pointer;font-size:11.5px;transition:all .15s}
.ccard-a button:hover{background:rgba(255,255,255,.06);color:var(--txt)}
.ccard-b{padding:10px 16px 16px;position:relative;height:340px}
.ccard-b canvas{width:100%!important;height:100%!important}
.ccard-f{padding:0 16px 14px;display:flex;gap:14px;font-size:11px;color:var(--muted)}
.ccard-f span{display:flex;align-items:center;gap:3px}

.sk{background:linear-gradient(90deg,var(--card) 25%,var(--card-h) 50%,var(--card) 75%);background-size:200% 100%;animation:shm 1.5s infinite;border-radius:7px}
@keyframes shm{0%{background-position:200% 0}100%{background-position:-200% 0}}

.empty-st{text-align:center;padding:100px 40px;color:var(--muted)}
.empty-st i{font-size:52px;opacity:.15;margin-bottom:20px;display:block}
.empty-st h2{font-size:20px;color:var(--txt);margin-bottom:8px;font-weight:600}
.empty-st p{font-size:14px;margin-bottom:6px;max-width:420px;margin-left:auto;margin-right:auto;line-height:1.6}
.empty-st .sub{font-size:12.5px;opacity:.5;margin-bottom:24px}
.empty-st code{background:var(--card);border:1px solid var(--bdr);padding:4px 10px;border-radius:6px;font-size:12px;color:var(--accent);display:inline-block;margin-top:4px}

.err-banner{background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.2);border-radius:10px;padding:14px 18px;margin-bottom:20px;display:flex;align-items:center;gap:10px;font-size:13px;color:var(--err)}
.err-banner i{font-size:16px;flex-shrink:0}
.err-banner button{margin-left:auto;background:none;border:1px solid rgba(239,68,68,.3);color:var(--err);padding:4px 12px;border-radius:5px;font-size:11.5px;cursor:pointer;font-family:'DM Sans',sans-serif;white-space:nowrap}
.err-banner button:hover{background:rgba(239,68,68,.1)}

.ov{position:fixed;inset:0;z-index:200;background:rgba(0,0,0,.6);backdrop-filter:blur(4px);display:flex;align-items:center;justify-content:center;opacity:0;pointer-events:none;transition:opacity .2s}
.ov.on{opacity:1;pointer-events:all}
.mdl{background:var(--bg2);border:1px solid var(--bdr);border-radius:14px;width:92%;max-height:85vh;overflow:hidden;display:flex;flex-direction:column;transform:scale(.95);transition:transform .2s}
.ov.on .mdl{transform:scale(1)}
.mdl-m{max-width:760px}.mdl-xl{max-width:1340px}
.mdl-h{display:flex;align-items:center;justify-content:space-between;padding:18px 22px;border-bottom:1px solid var(--bdr)}
.mdl-h h3{font-size:15px;font-weight:600}
.mdl-x{background:none;border:none;color:var(--muted);cursor:pointer;font-size:17px;padding:3px 7px;border-radius:5px;transition:all .15s}
.mdl-x:hover{background:rgba(255,255,255,.06);color:var(--txt)}
.mdl-b{padding:22px;overflow-y:auto;flex:1}
.mdl-f{padding:14px 22px;border-top:1px solid var(--bdr);display:flex;justify-content:flex-end;gap:9px}

.dt-w{overflow-x:auto;max-height:380px;overflow-y:auto}
.dt{width:100%;border-collapse:collapse;font-size:12px}
.dt th{background:var(--card);color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.5px;padding:9px 13px;text-align:left;position:sticky;top:0;font-size:10.5px;border-bottom:1px solid var(--bdr)}
.dt td{padding:8px 13px;border-bottom:1px solid var(--bdr);color:var(--txt);max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.dt tr:hover td{background:rgba(255,255,255,.02)}

.tc{position:fixed;bottom:20px;right:20px;z-index:300;display:flex;flex-direction:column-reverse;gap:7px}
.toast{background:var(--card);border:1px solid var(--bdr);padding:11px 16px;border-radius:9px;font-size:12.5px;display:flex;align-items:center;gap:9px;animation:tin .3s ease-out;box-shadow:0 8px 28px rgba(0,0,0,.4);min-width:260px}
@keyframes tin{from{opacity:0;transform:translateX(36px)}to{opacity:1;transform:translateX(0)}}
.toast.ok{border-left:3px solid var(--ok)}.toast.er{border-left:3px solid var(--err)}.toast.inf{border-left:3px solid var(--info)}

.bg-g{position:fixed;pointer-events:none;z-index:0}
.bg-g1{top:-180px;right:-80px;width:460px;height:460px;background:radial-gradient(circle,rgba(245,158,11,.05),transparent 70%)}
.bg-g2{bottom:-180px;left:-80px;width:460px;height:460px;background:radial-gradient(circle,rgba(6,182,212,.035),transparent 70%)}
.ldot{width:6px;height:6px;border-radius:50%;background:var(--ok);display:inline-block;animation:lp 1.5s ease-in-out infinite}
@keyframes lp{0%,100%{box-shadow:0 0 0 0 rgba(16,185,129,.4)}50%{box-shadow:0 0 0 5px rgba(16,185,129,0)}}
.no-ch{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;color:var(--muted);font-size:13px;gap:6px}
.no-ch i{font-size:28px;opacity:.25}
</style>
</head>
<body>

<div class="bg-g bg-g1"></div>
<div class="bg-g bg-g2"></div>

<aside class="side" id="side">
  <div class="side-brand"><div class="side-logo">HR</div><span>myHR Analytics</span></div>
  <nav class="side-nav" id="snav"></nav>
  <div class="sec" id="sds"></div>
  <div class="side-foot">
    <div style="display:flex;align-items:center;gap:9px">
      <div style="width:30px;height:30px;border-radius:7px;background:linear-gradient(135deg,#f59e0b,#f97316);display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;color:#000;flex-shrink:0">JD</div>
      <div><div style="font-size:12.5px;font-weight:500">Jane Doe</div><div style="font-size:10.5px;color:var(--muted)">HR Analytics Lead</div></div>
    </div>
  </div>
</aside>

<div class="wrap" id="wrap">
  <header class="top" id="top"></header>
  <main class="dash" id="dash"></main>
</div>

<div class="ov" id="ov" onclick="if(event.target===this)closeMdl()">
  <div class="mdl" id="mc"></div>
</div>
<div class="tc" id="tc"></div>

<script>
// ================================================================
// API CONFIGURATION
// ================================================================
// Dashboard HTML is served from the same FastAPI app as the API
// endpoints, so relative URLs always resolve to the correct origin
// regardless of port (8000, 8001, etc.).
var API_BASE = '';

// ================================================================
// API CLIENT
// ================================================================
var apiStatus = 'loading';

function apiFetch(path, opts) {
  opts = opts || {};
  return fetch(API_BASE + path, { headers: { 'Content-Type': 'application/json' }, ...opts })
    .then(function(res) {
      if (!res.ok) return res.json().then(function(err) { throw new Error(err.detail || 'API Error ' + res.status); });
      return res.json();
    });
}

function apiPost(path, body) {
  return apiFetch(path, { method: 'POST', body: JSON.stringify(body) });
}

// ================================================================
// CONSTANTS
// ================================================================
var CC = ['#f59e0b','#06b6d4','#10b981','#f43f5e','#8b5cf6','#ec4899','#3b82f6','#84cc16','#f97316','#14b8a6'];
var CCB = CC.map(function(c) { return c + '22'; });

// ================================================================
// STATE — multiple queries array
// ================================================================
var datasetSchemas = {};
var datasetCounts = {};
var globalFilters = { department: null, location: null, status: null };
var CI = {};
var queries = [];
var activeQueryIndex = -1;

function currentQuery() {
  return queries[activeQueryIndex] || null;
}

function addQuery(q) {
  queries.push(q);
  activeQueryIndex = queries.length - 1;
}

function removeQuery(idx) {
  if (idx < 0 || idx >= queries.length) return;
  queries.splice(idx, 1);
  if (activeQueryIndex >= queries.length) {
    activeQueryIndex = queries.length - 1;
  }
  if (queries.length === 0) {
    activeQueryIndex = -1;
  }
  renderDash();
}

// ================================================================
// CHART RENDERING
// ================================================================
function renderC(cid, res, ct) {
  if (typeof Chart === 'undefined') {
    var el = document.getElementById(cid);
    if (el) el.innerHTML = '<div class="no-ch"><i class="fa-solid fa-chart-column"></i><span>Chart.js not loaded</span></div>';
    return null;
  }
  try {
    var cv = document.getElementById(cid); if (!cv) return null;
    if (CI[cid]) { CI[cid].destroy(); delete CI[cid]; }
    var labels = res.labels, ds = res.datasets;
    var isH = ct === 'hbar', at = isH ? 'bar' : ct === 'area' ? 'line' : ct;
    var cds = ds.map(function(d, i) {
      var c = CC[i % CC.length], bg = CCB[i % CCB.length];
      var o = { label: d.label, data: d.data, borderColor: c, backgroundColor: (at === 'line' || ct === 'area') ? bg : c, borderWidth: 2, pointRadius: at === 'line' ? 3 : 0, pointHoverRadius: at === 'line' ? 5 : 0, tension: .3, fill: ct === 'area' };
      if (['pie','doughnut','polarArea'].indexOf(at) >= 0) {
        o.backgroundColor = labels.map(function(_, j) { return CC[j % CC.length] + 'cc'; });
        o.borderColor = labels.map(function(_, j) { return CC[j % CC.length]; });
        o.borderWidth = 2;
      }
      return o;
    });
    var noScales = ['pie','doughnut','polarArea','radar'].indexOf(at) >= 0;
    var opts = {
      responsive: true, maintainAspectRatio: false, indexAxis: isH ? 'y' : 'x',
      plugins: {
        legend: { display: ds.length > 1 || ['pie','doughnut','polarArea'].indexOf(at) >= 0, labels: { color: '#5e5e78', font: { size: 11, family: 'DM Sans' }, boxWidth: 12, padding: 12 } },
        tooltip: { backgroundColor: '#1a1a2a', titleColor: '#e4e4ed', bodyColor: '#e4e4ed', borderColor: '#1f1f32', borderWidth: 1, padding: 10, titleFont: { family: 'Space Grotesk', weight: '600' }, bodyFont: { family: 'DM Sans' }, cornerRadius: 8 }
      },
      scales: noScales ? {} : {
        x: { grid: { color: 'rgba(255,255,255,.04)' }, ticks: { color: '#5e5e78', font: { size: 11, family: 'DM Sans' }, maxRotation: 45 } },
        y: { grid: { color: 'rgba(255,255,255,.04)' }, ticks: { color: '#5e5e78', font: { size: 11, family: 'DM Sans' } }, beginAtZero: true }
      },
      animation: { duration: 600, easing: 'easeOutQuart' }
    };
    if (at === 'radar') { opts.scales = { r: { grid: { color: 'rgba(255,255,255,.06)' }, angleLines: { color: 'rgba(255,255,255,.06)' }, pointLabels: { color: '#6b6b80', font: { size: 10 } }, ticks: { display: false }, beginAtZero: true } }; }
    CI[cid] = new Chart(cv, { type: at, data: { labels: labels, datasets: cds }, options: opts });
    return CI[cid];
  } catch (e) { console.error('Chart render error:', e); return null; }
}

// ================================================================
// DASHBOARD RENDERING
// ================================================================
function renderDash() {
  var d = document.getElementById('dash'); if (!d) return;
  var h = '';

  // API error banner
  if (apiStatus === 'error') {
    h += '<div class="err-banner"><i class="fa-solid fa-triangle-exclamation"></i><span>Cannot connect to API at ' + API_BASE + '/api/health</span><button onclick="boot()"><i class="fa-solid fa-arrows-rotate"></i> Retry</button></div>';
  }

  // Query banner (only if we have an active query)
  var aq = currentQuery();
  if (aq && aq.config) {
    h += '<div class="query-banner"><i class="fa-solid fa-bolt"></i><div class="qb-text"><div class="qb-label">Active Query</div>';
    if (aq.raw) {
      h += '<div class="qb-value">"' + escHtml(aq.raw) + '"</div>';
    }
    if (aq.interpretation) {
      h += '<div class="qb-interpretation">' + escHtml(aq.interpretation) + '</div>';
    }
    h += '</div></div>';
  }

  // KPIs
  h += '<div class="kpi-row" id="kpi-row">';
  if (apiStatus === 'loading') {
    for (var k = 0; k < 4; k++) h += '<div class="kpi"><div class="sk" style="width:38px;height:38px;border-radius:9px;margin-bottom:12px"></div><div class="sk" style="width:100px;height:10px;margin-bottom:8px"></div><div class="sk" style="width:80px;height:28px"></div><div class="sk" style="width:120px;height:10px;margin-top:10px"></div></div>';
  } else {
    h += '<div class="kpi"><div class="kpi-ic" style="background:rgba(245,158,11,.12);color:#f59e0b"><i class="fa-solid fa-users"></i></div><div class="kpi-lb">Total Employees</div><div class="kpi-vl" id="kpi-emp">--</div><div class="kpi-td up"><i class="fa-solid fa-arrow-up"></i> 8.2% vs last quarter</div><div style="position:absolute;top:0;right:0;width:70px;height:70px;border-radius:50%;filter:blur(35px);opacity:.12;background:#f59e0b"></div></div>';
    h += '<div class="kpi"><div class="kpi-ic" style="background:rgba(6,182,212,.12);color:#06b6d4"><i class="fa-solid fa-dollar-sign"></i></div><div class="kpi-lb">Average Salary</div><div class="kpi-vl" id="kpi-sal">--</div><div class="kpi-td up"><i class="fa-solid fa-arrow-up"></i> 3.5% vs last quarter</div><div style="position:absolute;top:0;right:0;width:70px;height:70px;border-radius:50%;filter:blur(35px);opacity:.12;background:#06b6d4"></div></div>';
    h += '<div class="kpi"><div class="kpi-ic" style="background:rgba(16,185,129,.12);color:#10b981"><i class="fa-solid fa-star"></i></div><div class="kpi-lb">Avg Performance</div><div class="kpi-vl" id="kpi-perf">--</div><div class="kpi-td up"><i class="fa-solid fa-arrow-up"></i> 0.3 vs last quarter</div><div style="position:absolute;top:0;right:0;width:70px;height:70px;border-radius:50%;filter:blur(35px);opacity:.12;background:#10b981"></div></div>';
    h += '<div class="kpi"><div class="kpi-ic" style="background:rgba(244,63,94,.12);color:#f43f5e"><i class="fa-solid fa-calendar-check"></i></div><div class="kpi-lb">Attendance Rate</div><div class="kpi-vl" id="kpi-att">--</div><div class="kpi-td dn"><i class="fa-solid fa-arrow-down"></i> 0.8% vs last month</div><div style="position:absolute;top:0;right:0;width:70px;height:70px;border-radius:50%;filter:blur(35px);opacity:.12;background:#f43f5e"></div></div>';
  }
  h += '</div>';

  // Chart area
  h += '<div class="cgrid">';
  if (queries.length === 0) {
    if (apiStatus !== 'loading') {
      // Empty state — no query provided
      h += '<div class="empty-st"><i class="fa-solid fa-link"></i>';
      h += '<h2>Open from LibreChat</h2>';
      h += '<p>This dashboard renders dynamically based on a query. Navigate here with a URL parameter to see results.</p>';
      h += '<div class="sub">Examples</div>';
      h += '<div style="display:flex;flex-direction:column;gap:8px;align-items:center">';
      h += '<code>dashboard.html?query=headcount by department</code>';
      h += '<code>dashboard.html?query=average salary by location</code>';
      h += '<code>dashboard.html?query=performance trend over quarters</code>';
      h += '</div>';
      h += '<p class="sub" style="margin-top:16px">Or pass a structured config:</p>';
      h += '<code style="font-size:10px;max-width:500px;word-break:break-all">dashboard.html?config=' + encodeURIComponent('{"dataset":"employees","dimensions":["department"],"metrics":[{"field":"id","agg":"count"}],"chart_type":"bar","sort":"desc"}') + '</code>';
      h += '</div>';
    } else {
      // Loading
      h += '<div class="ccard" style="grid-column:1/-1"><div class="ccard-b" style="height:340px"><div class="sk" style="width:100%;height:100%"></div></div></div>';
    }
  } else {
    // Render each query chart
    queries.forEach(function(q, idx) {
      if (!q.result || !q.result.labels || !q.result.labels.length) {
        if (!q.config) {
          h += '<div class="empty-st"><i class="fa-solid fa-filter-circle-xmark" style="color:var(--err)"></i><p>No data matches your query.</p><p class="sub">Try adjusting the filters above.</p></div>';
        } else {
          h += '<div class="ccard" style="grid-column:1/-1"><div class="ccard-b" style="height:340px"><div class="sk" style="width:100%;height:100%"></div></div></div>';
        }
        return;
      }
      var res = q.result;
      var cfg = q.config;
      var mi = (cfg.metrics || []).map(function(m) { return m.agg.toUpperCase() + ' ' + m.field; }).join(', ');
      var cid = 'chart-' + idx;
      h += '<div class="ccard sz-6"><div class="ccard-h"><div class="ccard-t"><span class="dot"></span>' + escHtml(q.title || ('Chart ' + (idx + 1))) + '</div><div class="ccard-a"><button onclick="refreshView(' + idx + ')" title="Refresh"><i class="fa-solid fa-arrows-rotate"></i></button><button onclick="showDT(' + idx + ')" title="View Data"><i class="fa-solid fa-table"></i></button><button onclick="expCSV(' + idx + ')" title="Export CSV"><i class="fa-solid fa-download"></i></button><button onclick="removeQuery(' + idx + ')" title="Remove chart"><i class="fa-solid fa-trash"></i></button></div></div><div class="ccard-b"><canvas id="' + cid + '"></canvas></div><div class="ccard-f"><span><i class="fa-solid fa-database"></i> ' + cfg.dataset + '</span><span><i class="fa-solid fa-layer-group"></i> ' + res.labels.length + ' groups</span><span><i class="fa-solid fa-calculator"></i> ' + mi + '</span></div></div>';
    });
  }
  h += '</div>';
  d.innerHTML = h;

  // Render chart canvases after DOM update
  queries.forEach(function(q, idx) {
    if (q.result && q.result.labels && q.result.labels.length) {
      requestAnimationFrame(function() {
        renderC('chart-' + idx, q.result, q.config.chart_type);
      });
    }
  });
}

function escHtml(s) {
  var div = document.createElement('div'); div.textContent = s; return div.innerHTML;
}

// Fetch KPIs
function fetchKPIs() {
  var params = [];
  if (globalFilters.department) params.push('department=' + encodeURIComponent(globalFilters.department));
  if (globalFilters.location) params.push('location=' + encodeURIComponent(globalFilters.location));
  if (globalFilters.status) params.push('status=' + encodeURIComponent(globalFilters.status));
  var qs = params.length ? '?' + params.join('&') : '';
  apiFetch('/api/kpis' + qs).then(function(data) {
    var e1 = document.getElementById('kpi-emp'); if (e1) e1.textContent = data.total_employees;
    var e2 = document.getElementById('kpi-sal'); if (e2) e2.textContent = '$' + (data.avg_salary / 1000).toFixed(0) + 'K';
    var e3 = document.getElementById('kpi-perf'); if (e3) e3.innerHTML = data.avg_performance + '<span style="font-size:15px;color:var(--muted);font-weight:400"> /5</span>';
    var e4 = document.getElementById('kpi-att'); if (e4) e4.innerHTML = data.attendance_rate + '<span style="font-size:15px;color:var(--muted);font-weight:400">%</span>';
  }).catch(function(e) { console.error('KPI fetch error:', e); });
}

// Refresh the current view (re-execute query with current filters)
function refreshView(idx) {
  idx = idx !== undefined ? idx : activeQueryIndex;
  if (idx < 0 || idx >= queries.length) return;
  var q = queries[idx];
  if (!q.config) return;
  q.config.global_department = globalFilters.department;
  q.config.global_location = globalFilters.location;
  q.config.global_status = globalFilters.status;
  // Show loading
  q.result = { labels: [], datasets: [] };
  renderDash();
  apiPost('/api/query', q.config).then(function(res) {
    q.result = res;
    renderDash();
    toast('Chart refreshed', 'ok');
  }).catch(function(e) {
    toast('Refresh failed: ' + e.message, 'er');
    renderDash();
  });
  fetchKPIs();
}

// ================================================================
// DATA TABLE & CSV
// ================================================================
function showDT(idx) {
  idx = idx !== undefined ? idx : activeQueryIndex;
  if (idx < 0 || idx >= queries.length) return;
  var q = queries[idx];
  if (!q.result) return;
  var r = q.result;
  var hd = ['Label'].concat(r.datasets.map(function(d) { return d.label; }));
  var rows = '';
  r.labels.forEach(function(lb, i) {
    rows += '<tr><td>' + lb + '</td>' + r.datasets.map(function(d) { return '<td>' + (typeof d.data[i] === 'number' ? d.data[i].toLocaleString() : d.data[i]) + '</td>'; }).join('') + '</tr>';
  });
  var h = '<div class="mdl-h"><h3><i class="fa-solid fa-table" style="color:var(--info);margin-right:7px"></i>' + escHtml(q.title || ('Chart ' + (idx + 1))) + ' — Data</h3><button class="mdl-x" onclick="closeMdl()"><i class="fa-solid fa-xmark"></i></button></div>';
  h += '<div class="mdl-b" style="padding:0"><div class="dt-w"><table class="dt"><thead><tr>' + hd.map(function(x) { return '<th>' + x + '</th>'; }).join('') + '</tr></thead><tbody>' + rows + '</tbody></table></div></div>';
  h += '<div class="mdl-f"><button class="tbtn" onclick="expCSV(' + idx + ')"><i class="fa-solid fa-download"></i> Export CSV</button><button class="tbtn" onclick="closeMdl()">Close</button></div>';
  openMdl(h, 'mdl-m');
}

function expCSV(idx) {
  idx = idx !== undefined ? idx : activeQueryIndex;
  if (idx < 0 || idx >= queries.length) return;
  var q = queries[idx];
  if (!q.result) return;
  var r = q.result;
  var hd = ['Label'].concat(r.datasets.map(function(d) { return d.label; }));
  var csv = hd.join(',') + '\n';
  r.labels.forEach(function(lb, i) {
    csv += '"' + lb + '",' + r.datasets.map(function(d) { return d.data[i]; }).join(',') + '\n';
  });
  var b = new Blob([csv], { type: 'text/csv' });
  var a = document.createElement('a');
  a.href = URL.createObjectURL(b);
  a.download = (q.title || ('chart-' + (idx + 1))).replace(/[^a-z0-9]/gi, '_') + '.csv';
  a.click();
  toast('CSV exported', 'ok');
}

// Dataset preview
function showDSPreview(k) {
  apiFetch('/api/datasets/' + k + '/data?limit=50').then(function(data) {
    if (!data.data || !data.data.length) { toast('No data found', 'inf'); return; }
    var fks = Object.keys(data.data[0]);
    var h = '<div class="mdl-h"><h3><i class="fa-solid ' + (datasetSchemas[k] || {}).icon + '" style="color:' + (datasetSchemas[k] || {}).color + ';margin-right:7px"></i>' + (datasetSchemas[k] || { label: k }).label + ' — Data Preview</h3><button class="mdl-x" onclick="closeMdl()"><i class="fa-solid fa-xmark"></i></button></div>';
    h += '<div class="mdl-b" style="padding:0"><div style="padding:10px 14px;font-size:11.5px;color:var(--muted);border-bottom:1px solid var(--bdr)">Showing ' + data.returned + ' of ' + data.total + ' records</div><div class="dt-w" style="max-height:480px"><table class="dt"><thead><tr>' + fks.map(function(f) { return '<th>' + f + '</th>'; }).join('') + '</tr></thead><tbody>' + data.data.map(function(row) { return '<tr>' + fks.map(function(f) { return '<td>' + (row[f] != null ? row[f] : '') + '</td>'; }).join('') + '</tr>'; }).join('') + '</tbody></table></div></div>';
    h += '<div class="mdl-f"><button class="tbtn" onclick="closeMdl()">Close</button></div>';
    openMdl(h, 'mdl-xl');
  }).catch(function(e) { toast('Failed to load data: ' + e.message, 'er'); });
}

// ================================================================
// SIDEBAR
// ================================================================
function renderSide() {
  document.getElementById('snav').innerHTML = '<a class="on"><i class="fa-solid fa-grid-2"></i>Dashboard</a>';
}

function updSideDS() {
  var dc = document.getElementById('sds');
  dc.innerHTML = '<div class="sec-t">Datasets</div>' + Object.keys(datasetSchemas).map(function(k) {
    var v = datasetSchemas[k];
    return '<div class="ds-item" onclick="showDSPreview(\'' + k + '\')"><i class="fa-solid ' + v.icon + '" style="color:' + v.color + '"></i><span>' + v.label + '</span><span class="cnt">' + (datasetCounts[k] || 0) + '</span></div>';
  }).join('');
}

// ================================================================
// TOPBAR
// ================================================================
function renderTop() {
  var t = document.getElementById('top');
  var connCls = apiStatus === 'ok' ? 'ok' : apiStatus === 'error' ? 'err' : 'ld';
  var connLbl = apiStatus === 'ok' ? 'API Connected' : apiStatus === 'error' ? 'API Offline' : 'Connecting...';
  var bcTitle = (currentQuery() && currentQuery().title) || 'Overview';
  t.innerHTML = '<button class="top-tog" onclick="togSide()"><i class="fa-solid fa-bars"></i></button><div class="bc"><span>Dashboard</span><i class="fa-solid fa-chevron-right" style="font-size:8px"></i><span class="cur">' + escHtml(bcTitle) + '</span></div><div class="top-f"><span class="conn ' + connCls + '" id="conn-badge"><span class="conn-dot"></span>' + connLbl + '</span><select class="fsel" id="f-dep" onchange="applyGF()"><option value="">All Departments</option>' + (window._depList || []).map(function(d) { return '<option value="' + d + '">' + d + '</option>'; }).join('') + '</select><select class="fsel" id="f-loc" onchange="applyGF()"><option value="">All Locations</option>' + (window._locList || []).map(function(d) { return '<option value="' + d + '">' + d + '</option>'; }).join('') + '</select><select class="fsel" id="f-sta" onchange="applyGF()"><option value="">All Statuses</option><option value="Active">Active</option><option value="On Leave">On Leave</option><option value="Probation">Probation</option></select><button class="tbtn" onclick="refreshView()" title="Refresh"><i class="fa-solid fa-arrows-rotate"></i></button><button class="tbtn" onclick="addChartPrompt()" title="Add chart"><i class="fa-solid fa-plus"></i> Add chart</button></div>';
}

function applyGF() {
  globalFilters.department = document.getElementById('f-dep').value || null;
  globalFilters.location = document.getElementById('f-loc').value || null;
  globalFilters.status = document.getElementById('f-sta').value || null;
  refreshView();
}

function togSide() { document.getElementById('side').classList.toggle('hide'); document.getElementById('wrap').classList.toggle('wide'); }

// ================================================================
// MODAL & TOAST
// ================================================================
function openMdl(html, cls) {
  var o = document.getElementById('ov'), c = document.getElementById('mc');
  c.className = 'mdl ' + (cls || 'mdl-m'); c.innerHTML = html; o.classList.add('on'); document.body.style.overflow = 'hidden';
}
function closeMdl() { document.getElementById('ov').classList.remove('on'); document.body.style.overflow = ''; }

function toast(msg, type) {
  var c = document.getElementById('tc'), t = document.createElement('div');
  t.className = 'toast ' + (type || 'inf');
  var ic = { ok: 'fa-check-circle', er: 'fa-exclamation-circle', inf: 'fa-info-circle' };
  var co = { ok: 'var(--ok)', er: 'var(--err)', inf: 'var(--info)' };
  t.innerHTML = '<i class="fa-solid ' + (ic[type] || ic.inf) + '" style="color:' + (co[type] || co.inf) + '"></i><span>' + msg + '</span>';
  c.appendChild(t);
  setTimeout(function() { t.style.opacity = '0'; t.style.transform = 'translateX(36px)'; t.style.transition = 'all .3s'; setTimeout(function() { t.remove(); }, 300); }, 2800);
}

// ================================================================
// URL PARAM PARSING
// ================================================================
function parseUrlParams() {
  var params = new URLSearchParams(window.location.search);
  var query = params.get('query');
  var config = params.get('config');
  var queriesParam = params.get('queries');
  if (queriesParam) {
    try {
      var parsed = JSON.parse(queriesParam);
      if (Array.isArray(parsed)) {
        return { type: 'queries', value: parsed };
      }
    } catch (e) {
      // Try pipe-separated
      var parts = queriesParam.split('|').map(function(s) { return s.trim(); }).filter(Boolean);
      if (parts.length) {
        return { type: 'queries', value: parts };
      }
    }
  }
  if (query) return { type: 'query', value: query };
  if (config) {
    try { return { type: 'config', value: JSON.parse(config) }; }
    catch (e) { console.error('Failed to parse config param:', e); return null; }
  }
  return null;
}

function addChartPrompt() {
  var q = prompt('Enter a query (e.g. "headcount by department") or structured JSON config:');
  if (!q) return;
  if (q.trim().startsWith('{')) {
    try {
      var cfg = JSON.parse(q);
      addQuery({ config: cfg, result: null, title: cfg.title || 'Custom Chart', interpretation: '' });
      activeQueryIndex = queries.length - 1;
      renderDash();
      apiPost('/api/query', cfg).then(function(res) {
        queries[activeQueryIndex].result = res;
        renderDash();
        toast('Chart added', 'ok');
      }).catch(function(e) {
        toast('Query failed: ' + e.message, 'er');
        renderDash();
      });
      fetchKPIs();
      return;
    } catch (e) {
      toast('Invalid JSON config', 'er');
      return;
    }
  }
  addQuery({ raw: q, config: null, result: null, title: q, interpretation: '' });
  activeQueryIndex = queries.length - 1;
  renderDash();
  apiPost('/api/ai-query', { query: q }).then(function(data) {
    queries[activeQueryIndex].config = data.config;
    queries[activeQueryIndex].result = data.result;
    queries[activeQueryIndex].title = data.title;
    queries[activeQueryIndex].interpretation = data.interpretation;
    renderDash();
    toast('Chart added', 'ok');
  }).catch(function(e) {
    toast('Query failed: ' + e.message, 'er');
    renderDash();
  });
  fetchKPIs();
}

// ================================================================
// BOOT SEQUENCE
// ================================================================
function boot() {
  apiStatus = 'loading';
  renderTop(); renderSide(); renderDash();

  apiFetch('/api/health').then(function(data) {
    apiStatus = 'ok';
    datasetCounts = data.datasets;
    return apiFetch('/api/datasets');
  }).then(function(schemas) {
    if (schemas) {
      schemas.forEach(function(s) {
        datasetSchemas[s.name] = { label: s.label, icon: s.icon, color: s.color, fields: {} };
        s.fields.forEach(function(f) { datasetSchemas[s.name].fields[f.name] = { type: f.type, label: f.label }; });
      });
    }
    window._depList = ['Engineering','Product','Design','Marketing','Sales','HR','Finance','Operations','Legal','Customer Success'];
    window._locList = ['New York','San Francisco','London','Berlin','Singapore','Tokyo','Sydney','Toronto'];
    renderTop(); renderSide(); updSideDS();

    // Parse URL and execute query
    var parsed = parseUrlParams();
    if (!parsed) {
      apiStatus = 'ok';
      renderDash();
      fetchKPIs();
      return;
    }

    if (parsed.type === 'queries') {
      // Multiple queries — execute each. /api/query returns Chart.js-ready
      // data directly (no config wrapper), so use the original qVal as the
      // config. Catch per-query failures so one bad chart doesn't kill the
      // whole dashboard.
      var qList = parsed.value;
      var promises = qList.map(function(qVal) {
        var p = (typeof qVal === 'string')
          ? apiPost('/api/ai-query', { query: qVal })
          : apiPost('/api/query', qVal);
        return p.catch(function(e) {
          console.error('Dashboard query failed:', e.message);
          return null;
        });
      });
      return Promise.all(promises).then(function(results) {
        results.forEach(function(data, i) {
          if (!data) return; // failed query — skip
          var qVal = qList[i];
          if (typeof qVal === 'string') {
            addQuery({ raw: qVal, config: data.config, result: data.result, title: data.title, interpretation: data.interpretation });
          } else {
            var schema = datasetSchemas[qVal.dataset] || {};
            var flds = schema.fields || {};
            var dimL = (qVal.dimensions || []).map(function(d) { return flds[d] ? flds[d].label : d; });
            var metL = (qVal.metrics || []).map(function(m) { return m.agg.toUpperCase() + ' ' + (flds[m.field] ? flds[m.field].label : m.field); });
            var title = qVal.title || (dimL.length ? metL.join(', ') + ' by ' + dimL.join(', ') : metL.join(', '));
            addQuery({ config: qVal, result: data, title: title, interpretation: '' });
          }
        });
        if (queries.length > 0) {
          document.title = queries[0].title + ' — myHR Analytics';
        }
        renderDash();
        fetchKPIs();
      });
    }

    if (parsed.type === 'query') {
      // Natural language — call /api/ai-query
      addQuery({ raw: parsed.value, config: null, result: null, title: parsed.value, interpretation: '' });
      document.title = parsed.value + ' — myHR Analytics';
      return apiPost('/api/ai-query', { query: parsed.value }).then(function(data) {
        queries[activeQueryIndex].config = data.config;
        queries[activeQueryIndex].result = data.result;
        queries[activeQueryIndex].title = data.title;
        queries[activeQueryIndex].interpretation = data.interpretation;
        document.title = data.title + ' — myHR Analytics';
        renderDash(); fetchKPIs();
      });
    } else {
      // Structured config — call /api/query
      var cfg = parsed.value;
      var schema = datasetSchemas[cfg.dataset] || {};
      var flds = schema.fields || {};
      var dimL = (cfg.dimensions || []).map(function(d) { return flds[d] ? flds[d].label : d; });
      var metL = (cfg.metrics || []).map(function(m) { return m.agg.toUpperCase() + ' ' + (flds[m.field] ? flds[m.field].label : m.field); });
      var title = dimL.length ? metL.join(', ') + ' by ' + dimL.join(', ') : metL.join(', ');
      addQuery({ config: cfg, result: null, title: title, interpretation: '' });
      document.title = title + ' — myHR Analytics';
      return apiPost('/api/query', cfg).then(function(res) {
        queries[activeQueryIndex].result = res;
        renderDash(); fetchKPIs();
      });
    }
  }).catch(function(e) {
    console.error('Boot failed:', e);
    apiStatus = 'error';
    window._depList = []; window._locList = [];
    renderTop(); renderSide(); updSideDS(); renderDash();
  });
}

// ================================================================
// START
// ================================================================
try { boot(); } catch (e) {
  console.error('Boot error:', e);
  document.getElementById('dash').innerHTML = '<div class="empty-st"><i class="fa-solid fa-triangle-exclamation" style="color:var(--err)"></i><h2>Initialization Error</h2><p class="sub">' + e.message + '</p></div>';
}
</script>
</body>
</html>"""

# DAB manager for data fetching (shared with the rest of the agent).
_dab_manager = DABTenantClientManager()

# Default tenant used by the dashboard data endpoints.
_DEFAULT_TENANT = os.getenv("DATASET_DEFAULT_TENANT", "RDEMOROCKFORT")


def _get_dab_client(tenant_id: str = _DEFAULT_TENANT):
    """Return a cached DAB client for the requested tenant."""
    return _dab_manager.get_client(tenant_id, role="HRMS_HR")


def _unwrap_dab_result(result: Any) -> Any:
    """Normalize a DAB MCP tool result to plain Python."""
    if isinstance(result, list):
        return result
    if not isinstance(result, dict):
        return result
    if result.get("isError"):
        msg = result.get("message", "DAB error")
        content = result.get("content", [])
        if content and isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict) and first.get("type") == "text":
                msg = first.get("text", msg)
        raise RuntimeError(msg)
    content = result.get("content")
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict) and first.get("type") == "text":
            text = first.get("text", "")
            if isinstance(text, str) and text.strip():
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict) and "result" in parsed:
                        return parsed["result"]
                    return parsed
                except (json.JSONDecodeError, ValueError):
                    return {"raw_text": text}
    if "result" in result:
        return result["result"]
    return result


def _extract_rows(result: Any) -> List[Dict[str, Any]]:
    """Normalize DAB tool result to a flat row list."""
    unwrapped = _unwrap_dab_result(result)
    if isinstance(unwrapped, list):
        return [r for r in unwrapped if isinstance(r, dict)]
    if isinstance(unwrapped, dict):
        for key in ("value", "data", "items"):
            val = unwrapped.get(key)
            if isinstance(val, list):
                return [r for r in val if isinstance(r, dict)]
    return []


def _extract_items(result: Any) -> List[Dict[str, Any]]:
    """Best-effort extraction of list-of-dicts from a DAB/MCP result."""
    unwrapped = _unwrap_dab_result(result)
    if isinstance(unwrapped, list):
        return [item for item in unwrapped if isinstance(item, dict)]
    if isinstance(unwrapped, dict):
        for key in ("entities", "databases", "items", "result", "data", "value"):
            val = unwrapped.get(key)
            if isinstance(val, list):
                return [item for item in val if isinstance(item, dict)]
    return []


# ------------------------------------------------------------------
# FastAPI router for the dynamic dashboard
# ------------------------------------------------------------------

dashboard_router = APIRouter(tags=["dynamic-dashboard"])


@dashboard_router.get("/dashboard.html", include_in_schema=False)
async def serve_dashboard() -> HTMLResponse:
    """Serve the single-query dynamic dashboard."""
    return HTMLResponse(content=_DASHBOARD_HTML)


@dashboard_router.get("/api/health")
async def api_health() -> JSONResponse:
    """Health check with dataset counts."""
    try:
        client = _get_dab_client()
        entities = client.describe_entities()
        items = _extract_items(entities)
        counts: Dict[str, int] = {}
        for ent in items:
            name = ent.get("name") or ent.get("entity") or ""
            if name:
                counts[name] = ent.get("recordCount", ent.get("count", 0))
        return JSONResponse({"datasets": counts, "status": "ok"})
    except Exception as exc:
        logger.warning("Dynamic dashboard /api/health failed: %s", exc)
        return JSONResponse({"datasets": {}, "status": "error", "detail": str(exc)})


@dashboard_router.get("/api/datasets")
async def api_list_datasets() -> JSONResponse:
    """List available datasets with schema metadata."""
    try:
        client = _get_dab_client()
        entities = client.describe_entities()
        items = _extract_items(entities)
        datasets = []
        for ent in items:
            name = ent.get("name") or ent.get("entity") or ""
            if not name:
                continue
            fields = ent.get("fields") or ent.get("columns") or []
            if not fields:
                try:
                    sample = client.call_tool("read_records", {"entity": name, "first": 1})
                    rows = _extract_rows(sample)
                    if rows:
                        sample_row = rows[0]
                        fields = [
                            {"name": k, "type": _infer_type(v), "label": k.replace("_", " ").title()}
                            for k, v in sample_row.items()
                        ]
                except Exception:
                    pass
            datasets.append({
                "name": name,
                "label": ent.get("displayName") or ent.get("description") or name.replace("_", " ").title(),
                "icon": "fa-solid fa-table",
                "color": "#f59e0b",
                "fields": fields,
            })
        return JSONResponse(datasets)
    except Exception as exc:
        logger.error("Dynamic dashboard /api/datasets failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@dashboard_router.post("/api/query")
async def api_query(config: Dict[str, Any]) -> JSONResponse:
    """Execute a structured chart config and return Chart.js-ready data."""
    try:
        result = await _execute_query_config(config)
        return JSONResponse(result)
    except Exception as exc:
        logger.error("Dynamic dashboard /api/query failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))


@dashboard_router.post("/api/ai-query")
async def api_ai_query(request: Dict[str, Any]) -> JSONResponse:
    """Parse a natural-language query, execute it, and return chart data."""
    query_text = request.get("query", "")
    if not query_text:
        raise HTTPException(status_code=400, detail="Missing 'query' field")

    try:
        config = _parse_natural_language_query(query_text)
        result = await _execute_query_config(config)
        return JSONResponse({
            "config": config,
            "result": result,
            "title": config.get("title", query_text),
            "interpretation": f"Parsed as {config.get('chart_type', 'bar')} chart of {config.get('metrics', ['count'])[0]} by {', '.join(config.get('dimensions', []))}",
        })
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Dynamic dashboard /api/ai-query failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))


@dashboard_router.get("/api/kpis")
async def api_kpis(
    department: Optional[str] = Query(None),
    location: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
) -> JSONResponse:
    """Return KPI summary numbers."""
    try:
        client = _get_dab_client()
        entity = "V_EMP"

        # Build OData filter from global filters
        filters = _build_odata_filter(department=department, location=location, status=status)

        # Total employees — count(*) -> DAB alias "count"
        emp_args = {"entity": entity, "function": "count", "field": "*"}
        if filters:
            emp_args["filter"] = filters
        emp_result = client.call_tool("aggregate_records", emp_args)
        emp_rows = _extract_rows(emp_result)
        total_employees = emp_rows[0].get("count", 0) if emp_rows else 0

        # Average salary (approximate — use SALARY or similar field if present)
        salary_field = _detect_salary_field(client, entity)
        avg_salary = 0
        if salary_field:
            sal_args = {"entity": entity, "function": "avg", "field": salary_field}
            if filters:
                sal_args["filter"] = filters
            sal_result = client.call_tool("aggregate_records", sal_args)
            sal_rows = _extract_rows(sal_result)
            if sal_rows:
                # DAB alias: "{function}_{field}"
                avg_salary = float(
                    sal_rows[0].get(f"avg_{salary_field}", sal_rows[0].get("avg", 0)) or 0
                )

        # Average performance (approximate)
        perf_field = _detect_performance_field(client, entity)
        avg_performance = 0.0
        if perf_field:
            perf_args = {"entity": entity, "function": "avg", "field": perf_field}
            if filters:
                perf_args["filter"] = filters
            perf_result = client.call_tool("aggregate_records", perf_args)
            perf_rows = _extract_rows(perf_result)
            if perf_rows:
                avg_performance = float(
                    perf_rows[0].get(f"avg_{perf_field}", perf_rows[0].get("avg", 0)) or 0
                )

        # Attendance rate (approximate — count active / total)
        attendance_rate = 0.0
        if total_employees > 0:
            active_args = {"entity": entity, "function": "count", "field": "*"}
            active_filter = _build_odata_filter(department=department, location=location, status="Active")
            if active_filter:
                active_args["filter"] = active_filter
            active_result = client.call_tool("aggregate_records", active_args)
            active_rows = _extract_rows(active_result)
            active_count = active_rows[0].get("count", 0) if active_rows else 0
            attendance_rate = round((active_count / total_employees) * 100, 1) if total_employees else 0.0

        return JSONResponse({
            "total_employees": total_employees,
            "avg_salary": avg_salary,
            "avg_performance": avg_performance,
            "attendance_rate": attendance_rate,
        })
    except Exception as exc:
        logger.error("Dynamic dashboard /api/kpis failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@dashboard_router.get("/api/datasets/{name}/data")
async def api_dataset_data(
    name: str,
    limit: int = Query(50, le=1000, ge=1),
    offset: int = Query(0, ge=0),
) -> JSONResponse:
    """Preview rows for a dataset."""
    try:
        client = _get_dab_client()
        result = client.call_tool("read_records", {
            "entity": name,
            "first": limit,
            "skip": offset,
        })
        rows = _extract_rows(result)
        return JSONResponse({
            "data": rows,
            "returned": len(rows),
            "total": len(rows),  # DAB does not always expose total count
        })
    except Exception as exc:
        logger.error("Dynamic dashboard /api/datasets/%s/data failed: %s", name, exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _infer_type(value: Any) -> str:
    """Map a Python value to a column type string."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        stripped = value.strip()
        if len(stripped) >= 10 and (stripped[4] == "-" or stripped[10] == "T"):
            try:
                import datetime
                datetime.datetime.fromisoformat(stripped.replace("Z", "+00:00"))
                return "datetime"
            except (ValueError, TypeError):
                pass
        return "string"
    return "string"


def _compute_tenure_band(date_str: str, reference_date: Optional[str] = None) -> str:
    """Compute tenure band from a date string (typically DATE_JOINED)."""
    try:
        import datetime
        ref = datetime.datetime.fromisoformat(reference_date.replace("Z", "+00:00")) if reference_date else datetime.datetime.now()
        dt = datetime.datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        days = (ref - dt).days
        years = days / 365.25
        if years < 0:
            return "Future"
        if years < 1:
            return "0-1y"
        if years < 2:
            return "1-2y"
        if years < 5:
            return "2-5y"
        return "5y+"
    except Exception:
        return "Unknown"


def _group_by_tenure_band(labels: List[str], data: List[float]) -> Dict[str, float]:
    """Group aggregated date labels into tenure bands and sum values."""
    bands: Dict[str, float] = {}
    for label, value in zip(labels, data):
        band = _compute_tenure_band(label)
        bands[band] = bands.get(band, 0.0) + (value if value is not None else 0.0)
    return bands


def _build_odata_filter(
    department: Optional[str] = None,
    location: Optional[str] = None,
    status: Optional[str] = None,
) -> str:
    """Build a simple OData filter string from dashboard global filters."""
    parts = []
    if department:
        parts.append(f"DEPARTMENT_CODE eq '{department}'")
    if location:
        parts.append(f"LOCATION_CODE eq '{location}'")
    if status:
        parts.append(f"EMPLOYEE_STATUS eq '{status}'")
    return " and ".join(parts) if parts else ""


def _detect_salary_field(client: Any, entity: str) -> Optional[str]:
    """Heuristically find a salary-like numeric field."""
    try:
        sample = client.call_tool("read_records", {"entity": entity, "first": 1})
        rows = _extract_rows(sample)
        if not rows:
            return None
        row = rows[0]
        for key in ("SALARY", "MONTHLY_SALARY", "BASE_SALARY", "GROSS_SALARY", "SALARY_AMOUNT", "HOURLY_RATE", "DAILY_RATE"):
            if key in row and isinstance(row[key], (int, float)):
                return key
        for key, val in row.items():
            if isinstance(val, (int, float)) and ("sal" in key.lower() or "rate" in key.lower()):
                return key
    except Exception:
        pass
    return None


def _detect_performance_field(client: Any, entity: str) -> Optional[str]:
    """Heuristically find a performance-like numeric field."""
    try:
        sample = client.call_tool("read_records", {"entity": entity, "first": 1})
        rows = _extract_rows(sample)
        if not rows:
            return None
        row = rows[0]
        for key in ("PERFORMANCE_RATING", "PERFORMANCE", "RATING", "SCORE"):
            if key in row and isinstance(row[key], (int, float)):
                return key
        for key, val in row.items():
            if isinstance(val, (int, float)) and ("perf" in key.lower() or "rating" in key.lower()):
                return key
    except Exception:
        pass
    return None


async def _execute_query_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a structured dashboard query config and return Chart.js data."""
    entity = config.get("dataset") or config.get("entity") or "V_EMP"
    dimensions = config.get("dimensions") or []
    metrics = config.get("metrics") or [{"field": "EMPLOYEE_NO", "agg": "count"}]
    chart_type = config.get("chart_type", "bar")
    sort = config.get("sort", "desc")

    # Apply global filters if present
    global_filters = _build_odata_filter(
        department=config.get("global_department"),
        location=config.get("global_location"),
        status=config.get("global_status"),
    )

    # Allow arbitrary OData filters from structured configs
    custom_filter = config.get("filter")
    if custom_filter:
        global_filters = (global_filters + " and " + custom_filter) if global_filters else custom_filter

    client = _get_dab_client()

    # If no dimensions, return a single big-number metric
    if not dimensions:
        metric = metrics[0]
        field = metric.get("field", "*")
        agg = metric.get("agg", "count")
        if agg == "count":
            field = "*"  # count(*) -> alias "count" (DAB ComputeAlias rule)
        args = {"entity": entity, "function": agg, "field": field}
        if global_filters:
            args["filter"] = global_filters
        result = client.call_tool("aggregate_records", args)
        rows = _extract_rows(result)
        # DAB alias: "count" for count(*), "{function}_{field}" otherwise
        alias = "count" if agg == "count" else f"{agg}_{field}"
        value = rows[0].get(alias, rows[0].get(agg, 0)) if rows else 0
        label = metric.get("label") or f"{agg.upper()} {field}"
        return {
            "labels": [label],
            "datasets": [{"label": label, "data": [value]}],
        }

    # Grouped query
    primary_metric = metrics[0]
    agg_field = primary_metric.get("field", "*")
    agg_func = primary_metric.get("agg", "count")
    if agg_func == "count":
        agg_field = "*"  # count(*) -> alias "count" (DAB ComputeAlias rule)

    # Handle derived TENURE_BAND dimension by grouping on DATE_JOINED and rebinning later
    use_tenure_band = "TENURE_BAND" in dimensions
    groupby_dimensions = [("DATE_JOINED" if d == "TENURE_BAND" else d) for d in dimensions]

    args: Dict[str, Any] = {
        "entity": entity,
        "function": agg_func,
        "field": agg_field,
        "groupby": groupby_dimensions,
    }
    if global_filters:
        args["filter"] = global_filters

    # Ordering: DAB aggregate_records 'orderby' is a direction string
    # ("asc"/"desc"), not an array of sort expressions. "desc" is the server
    # default, so only send "asc" explicitly.
    if sort == "asc":
        args["orderby"] = "asc"

    # Limit groups to keep charts readable
    args["first"] = 50

    result = client.call_tool("aggregate_records", args)
    rows = _extract_rows(result)

    if not rows:
        return {"labels": [], "datasets": []}

    # Build Chart.js structure
    labels = [" | ".join(str(row.get(d, "")) for d in groupby_dimensions) for row in rows]

    datasets = []
    for metric in metrics:
        m_field = metric.get("field", agg_field)
        m_agg = metric.get("agg", agg_func)
        m_label = metric.get("label") or f"{m_agg.upper()} {m_field}"
        # DAB ComputeAlias: count(*) -> "count"; otherwise "{function}_{field}"
        if m_agg == "count":
            m_alias = "count"
        else:
            m_alias = f"{m_agg}_{m_field}"
        data = []
        for row in rows:
            val = row.get(m_alias)
            if val is None:
                # Fallbacks for servers using different alias conventions
                val = row.get(m_agg, row.get(m_field, 0))
            data.append(float(val) if val is not None else 0.0)
        datasets.append({"label": m_label, "data": data})

    # Post-process tenure bands if requested
    if use_tenure_band and labels:
        banded_datasets = []
        for ds in datasets:
            bands = _group_by_tenure_band(labels, ds["data"])
            # Sort bands in logical order
            band_order = ["0-1y", "1-2y", "2-5y", "5y+", "Future", "Unknown"]
            sorted_labels = [b for b in band_order if b in bands]
            sorted_data = [bands.get(b, 0.0) for b in sorted_labels]
            banded_datasets.append({"label": ds["label"], "data": sorted_data})
        return {"labels": sorted_labels, "datasets": banded_datasets}

    return {"labels": labels, "datasets": datasets}


def _parse_natural_language_query(query: str) -> Dict[str, Any]:
    """Heuristic NL parser that converts a natural-language HR query into a chart config."""
    q = query.lower().strip()

    # Detect chart type hints
    chart_type = "bar"
    if "horizontal" in q or "hbar" in q:
        chart_type = "hbar"
    elif "line" in q or "trend" in q or "over time" in q:
        chart_type = "line"
    elif "area" in q:
        chart_type = "area"
    elif "pie" in q:
        chart_type = "pie"
    elif "doughnut" in q:
        chart_type = "doughnut"

    # Detect dimension keywords
    dimension_map = {
        "department": ["department_code", "department"],
        "branch": ["branch_code", "branch"],
        "company": ["company_code", "company"],
        "division": ["division_code"],
        "section": ["section_code"],
        "position": ["position_description", "position_code"],
        "gender": ["gender"],
        "nationality": ["nationality_code", "nationality"],
        "marital": ["marital_status"],
        "status": ["employee_status"],
        "location": ["location_code", "location"],
        "grade": ["grade_code", "grade"],
        "category": ["category_code", "category"],
        "pay country": ["pay_country", "pay_country"],
        "country": ["pay_country", "country"],
        "cost center": ["cost_center", "cost_center"],
        "profit center": ["profit_center", "profit_center"],
        "superior": ["superior_no", "superior"],
        "manager": ["superior_no", "superior"],
        "level": ["employee_level", "level_code", "level"],
        "confirmation": ["confirmation_status", "confirmation"],
        "resigned": ["date_resigned", "resigned"],
        "joined": ["date_joined", "joined"],
        "hire": ["date_joined", "hire_date", "hire"],
        "tenure": ["date_joined", "tenure"],
    }
    chosen_dim = None
    for kw, candidates in dimension_map.items():
        if kw in q:
            for cand in candidates:
                if cand.lower() in q:
                    chosen_dim = cand
                    break
        if chosen_dim:
            break

    # Detect metric intent
    count_intent = any(k in q for k in ("headcount", "count", "number of", "how many", "employees", "workforce", "population"))
    salary_intent = any(k in q for k in ("salary", "payroll", "compensation", "wage"))
    perf_intent = any(k in q for k in ("performance", "rating", "score"))

    if count_intent:
        metric_field = "EMPLOYEE_NO"
        metric_agg = "count"
    elif salary_intent:
        metric_field = "SALARY"
        metric_agg = "avg"
    elif perf_intent:
        metric_field = "PERFORMANCE_RATING"
        metric_agg = "avg"
    else:
        metric_field = "EMPLOYEE_NO"
        metric_agg = "count"

    # Detect entity from query
    entity = "V_EMP"
    for candidate in ("V_EMP", "employee_compensation", "V_TMS_OVERTIME", "codesetup"):
        if candidate.lower() in q:
            entity = candidate
            break

    # Detect sort direction
    sort = "desc"
    if "top" in q and ("5" in q or "10" in q):
        sort = "desc"
    elif "bottom" in q:
        sort = "asc"

    config: Dict[str, Any] = {
        "dataset": entity,
        "dimensions": [chosen_dim] if chosen_dim else [],
        "metrics": [{"field": metric_field, "agg": metric_agg}],
        "chart_type": chart_type,
        "sort": sort,
        "title": query,
    }
    return config


# ------------------------------------------------------------------
# Standalone runner (optional)
# ------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI

    _app = FastAPI(title="HR Dynamic Dashboard")
    _app.include_router(dashboard_router)

    @_app.get("/", include_in_schema=False)
    async def _root_redirect():
        return HTMLResponse(content=_DASHBOARD_HTML)

    print("Starting HR Dynamic Dashboard on http://0.0.0.0:8000")
    uvicorn.run(_app, host="0.0.0.0", port=8000)
