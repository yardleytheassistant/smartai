/* PeptideTrack — client-side research peptide tracker.
   All data lives in localStorage on this device. No network, no accounts. */

'use strict';

/* ---------------- Storage ---------------- */
const KEY = 'peptidetrack.v1';
const uid = () => Math.random().toString(36).slice(2, 10);

const DEFAULT_DB = {
  peptides: [],
  doses: [],
  supplies: [
    { id: uid(), name: 'Bacteriostatic water (30mL)', qty: 1, unit: 'vial', low: 1 },
    { id: uid(), name: 'Insulin syringes (U-100, 0.5mL)', qty: 100, unit: 'pcs', low: 20 },
    { id: uid(), name: 'Alcohol prep pads', qty: 100, unit: 'pcs', low: 20 },
    { id: uid(), name: 'Sharps container', qty: 1, unit: 'unit', low: 1 },
    { id: uid(), name: 'Gauze / band-aids', qty: 50, unit: 'pcs', low: 10 },
  ],
  settings: { syringeUnits: 100 }, // U-100 insulin syringe: 100 units = 1 mL
  version: 1,
};

function load() {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return structuredClone(DEFAULT_DB);
    const db = JSON.parse(raw);
    return { ...structuredClone(DEFAULT_DB), ...db };
  } catch (e) {
    return structuredClone(DEFAULT_DB);
  }
}
function save() { localStorage.setItem(KEY, JSON.stringify(DB)); }
let DB = load();

/* ---------------- Dosing math ----------------
   U-100 insulin syringe: 100 "units" = 1 mL.
   concentration (mg/mL)   = vialMg / bacWaterMl
   mcg per unit            = vialMg * 10 / bacWaterMl
   units for a dose (mcg)  = doseMcg * bacWaterMl / (10 * vialMg)
*/
function mcgPerUnit(vialMg, bacWaterMl, syringeUnits = 100) {
  if (!vialMg || !bacWaterMl) return 0;
  const mcgPerMl = vialMg * 1000 / bacWaterMl;
  return mcgPerMl / syringeUnits;
}
function unitsForDose(doseMcg, vialMg, bacWaterMl, syringeUnits = 100) {
  const mpu = mcgPerUnit(vialMg, bacWaterMl, syringeUnits);
  return mpu ? doseMcg / mpu : 0;
}
const fmt = (n, d = 1) => {
  if (!isFinite(n)) return '—';
  const r = Number(n.toFixed(d));
  return r.toString();
};

/* ---------------- Helpers ---------------- */
const $ = (sel, el = document) => el.querySelector(sel);
const el = (html) => { const t = document.createElement('template'); t.innerHTML = html.trim(); return t.content.firstElementChild; };
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

function fmtDate(ts) {
  const d = new Date(ts);
  return d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
}
function timeAgo(ts) {
  const s = (Date.now() - ts) / 1000;
  if (s < 60) return 'just now';
  if (s < 3600) return Math.floor(s / 60) + 'm ago';
  if (s < 86400) return Math.floor(s / 3600) + 'h ago';
  return Math.floor(s / 86400) + 'd ago';
}

function toast(msg) {
  const t = el(`<div class="toast">${esc(msg)}</div>`);
  document.body.appendChild(t);
  setTimeout(() => { t.style.opacity = '0'; t.style.transition = 'opacity .3s'; }, 1600);
  setTimeout(() => t.remove(), 2000);
}

const SITES = ['Abdomen L', 'Abdomen R', 'Thigh L', 'Thigh R', 'Love handle L', 'Love handle R', 'Delt L', 'Delt R'];

/* ---------------- Modal ---------------- */
function openModal(title, bodyEl, onMount) {
  const root = $('#modalRoot');
  const overlay = el(`<div class="modal-overlay"><div class="modal"><div class="modal-grip"></div><h2>${esc(title)}</h2></div></div>`);
  const modal = $('.modal', overlay);
  modal.appendChild(bodyEl);
  overlay.addEventListener('click', (e) => { if (e.target === overlay) closeModal(); });
  root.appendChild(overlay);
  if (onMount) onMount(modal);
}
function closeModal() { $('#modalRoot').innerHTML = ''; }

/* ---------------- Router ---------------- */
const TITLES = { home: 'Dashboard', calc: 'Dose Calculator', log: 'Injection Log', supplies: 'Supplies', peptides: 'Vials' };
let currentView = 'home';

function switchView(view) {
  currentView = view;
  document.querySelectorAll('.view').forEach(v => v.classList.add('hidden'));
  $('#view-' + view).classList.remove('hidden');
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.view === view));
  $('#viewTitle').textContent = TITLES[view];
  $('#main').scrollTop = 0;
  render();
}

document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => switchView(t.dataset.view)));
$('#settingsBtn').addEventListener('click', openSettings);

/* ---------------- Render dispatch ---------------- */
function render() {
  ({ home: renderHome, calc: renderCalc, log: renderLog, supplies: renderSupplies, peptides: renderPeptides }[currentView])();
}

/* ---------------- Vials view ---------------- */
function renderPeptides() {
  const v = $('#view-peptides');
  const list = DB.peptides.length ? DB.peptides.map(p => {
    const mpu = mcgPerUnit(p.vialMg, p.bacWaterMl, DB.settings.syringeUnits);
    const recon = p.bacWaterMl > 0;
    return `<div class="row" data-pid="${p.id}">
      <div class="row-main">
        <div class="row-title">${esc(p.name)}</div>
        <div class="row-sub">${p.vialMg} mg vial${recon ? ` · ${fmt(p.bacWaterMl,1)} mL BAC · ${fmt(mpu,1)} mcg/unit` : ' · not reconstituted'}</div>
      </div>
      <div class="row-right">${recon ? `<span class="row-badge badge-ok">ready</span>` : `<span class="row-badge badge-low">mix</span>`}</div>
    </div>`;
  }).join('') : `<div class="empty"><span class="em-icon">🧪</span>No vials yet.<br>Add a peptide vial to start.</div>`;

  v.innerHTML = `
    <div class="card"><h2>Peptide vials</h2><div id="pepList">${list}</div></div>
    <button class="btn" id="addPepBtn">＋ Add vial</button>
    <p class="disclaimer">For research documentation only. Nothing here is medical advice or a recommendation to administer any substance. Follow all applicable laws and your study protocol.</p>`;

  $('#addPepBtn').addEventListener('click', () => editPeptide());
  v.querySelectorAll('[data-pid]').forEach(r => r.addEventListener('click', () => editPeptide(r.dataset.pid)));
}

function editPeptide(id) {
  const p = id ? DB.peptides.find(x => x.id === id) : { name: '', vialMg: '', bacWaterMl: '', reconDate: '', notes: '' };
  const body = el(`<div>
    <label class="field"><span class="lbl">Peptide name</span><input id="f-name" placeholder="e.g. BPC-157" value="${esc(p.name)}"></label>
    <div class="inline-2">
      <label class="field"><span class="lbl">Vial amount (mg)</span><input id="f-mg" type="number" inputmode="decimal" placeholder="5" value="${esc(p.vialMg)}"></label>
      <label class="field"><span class="lbl">BAC water (mL)</span><input id="f-bac" type="number" inputmode="decimal" placeholder="2" value="${esc(p.bacWaterMl)}"></label>
    </div>
    <p class="hint">Leave BAC water blank until you reconstitute. Concentration updates automatically.</p>
    <div id="calcPreview"></div>
    <label class="field"><span class="lbl">Reconstitution date</span><input id="f-date" type="date" value="${esc(p.reconDate)}"></label>
    <label class="field"><span class="lbl">Notes / lot #</span><textarea id="f-notes" placeholder="Supplier, lot number, storage…">${esc(p.notes)}</textarea></label>
    <div class="btn-row">
      <button class="btn" id="saveP">Save vial</button>
      ${id ? '<button class="btn danger small" id="delP">Delete</button>' : ''}
    </div>
  </div>`);

  const preview = () => {
    const mg = parseFloat($('#f-mg', body).value);
    const bac = parseFloat($('#f-bac', body).value);
    const box = $('#calcPreview', body);
    if (mg > 0 && bac > 0) {
      const mpu = mcgPerUnit(mg, bac, DB.settings.syringeUnits);
      const conc = mg / bac;
      box.innerHTML = `<div class="result-box"><div class="big">${fmt(mpu,1)}</div><div class="unit">mcg per unit (on the syringe)</div><div class="sub">${fmt(conc,2)} mg/mL concentration</div></div><hr class="divider">`;
    } else { box.innerHTML = ''; }
  };
  openModal(id ? 'Edit vial' : 'Add vial', body, () => {
    $('#f-mg', body).addEventListener('input', preview);
    $('#f-bac', body).addEventListener('input', preview);
    preview();
    $('#saveP', body).addEventListener('click', () => {
      const name = $('#f-name', body).value.trim();
      if (!name) return toast('Name required');
      const rec = {
        id: p.id || uid(),
        name,
        vialMg: parseFloat($('#f-mg', body).value) || 0,
        bacWaterMl: parseFloat($('#f-bac', body).value) || 0,
        reconDate: $('#f-date', body).value,
        notes: $('#f-notes', body).value.trim(),
      };
      if (id) { DB.peptides = DB.peptides.map(x => x.id === id ? rec : x); }
      else DB.peptides.push(rec);
      save(); closeModal(); render(); toast('Saved');
    });
    if (id) $('#delP', body).addEventListener('click', () => {
      if (!confirm('Delete this vial?')) return;
      DB.peptides = DB.peptides.filter(x => x.id !== id);
      save(); closeModal(); render(); toast('Deleted');
    });
  });
}

/* ---------------- Calculator view ---------------- */
let calcState = { pid: '', dose: '' };
function renderCalc() {
  const v = $('#view-calc');
  const opts = DB.peptides.map(p => `<option value="${p.id}" ${p.id === calcState.pid ? 'selected' : ''}>${esc(p.name)} — ${p.vialMg}mg</option>`).join('');
  v.innerHTML = `
    <div class="card">
      <h2>Draw calculator</h2>
      ${DB.peptides.length ? `
      <label class="field"><span class="lbl">Peptide vial</span><select id="c-pep"><option value="">Select a vial…</option>${opts}</select></label>
      <div id="reconInfo"></div>
      <label class="field"><span class="lbl">Target dose (mcg)</span><input id="c-dose" type="number" inputmode="decimal" placeholder="250" value="${esc(calcState.dose)}"></label>
      <div id="c-result"></div>
      ` : `<div class="empty"><span class="em-icon">🧪</span>Add a vial first, then come back to calculate your draw.</div>`}
    </div>
    ${DB.peptides.length ? `<button class="btn secondary" id="c-logbtn">💉 Log this as an injection</button>` : ''}
    <div class="card">
      <h2>Quick converter</h2>
      <div class="inline-2">
        <label class="field"><span class="lbl">Vial (mg)</span><input id="q-mg" type="number" inputmode="decimal" placeholder="5"></label>
        <label class="field"><span class="lbl">BAC water (mL)</span><input id="q-bac" type="number" inputmode="decimal" placeholder="2"></label>
      </div>
      <label class="field"><span class="lbl">Dose (mcg)</span><input id="q-dose" type="number" inputmode="decimal" placeholder="250"></label>
      <div id="q-result"></div>
    </div>`;

  const compute = () => {
    const p = DB.peptides.find(x => x.id === $('#c-pep', v).value);
    calcState.pid = p ? p.id : '';
    calcState.dose = $('#c-dose', v).value;
    const info = $('#reconInfo', v);
    const res = $('#c-result', v);
    if (!p) { info.innerHTML = ''; res.innerHTML = ''; return; }
    if (!p.bacWaterMl) {
      info.innerHTML = `<p class="hint" style="color:var(--amber)">⚠ This vial isn't reconstituted yet. Set the BAC water volume on the Vials tab.</p>`;
      res.innerHTML = ''; return;
    }
    const mpu = mcgPerUnit(p.vialMg, p.bacWaterMl, DB.settings.syringeUnits);
    info.innerHTML = `<p class="hint">${fmt(p.vialMg/p.bacWaterMl,2)} mg/mL · ${fmt(mpu,1)} mcg per unit · vial ≈ ${fmt(p.vialMg*1000/mpu,0)} units total</p>`;
    const dose = parseFloat($('#c-dose', v).value);
    if (dose > 0) {
      const units = unitsForDose(dose, p.vialMg, p.bacWaterMl, DB.settings.syringeUnits);
      const ml = units / DB.settings.syringeUnits;
      const over = units > DB.settings.syringeUnits;
      res.innerHTML = `<div class="result-box"><div class="big">${fmt(units,1)}</div><div class="unit">units on the syringe</div><div class="sub">${fmt(ml,3)} mL · ${dose} mcg${over ? ' · ⚠ exceeds a 1 mL / 100-unit syringe' : ''}</div></div>`;
    } else res.innerHTML = '';
  };

  if (DB.peptides.length) {
    $('#c-pep', v).addEventListener('change', compute);
    $('#c-dose', v).addEventListener('input', compute);
    compute();
    $('#c-logbtn', v).addEventListener('click', () => {
      const p = DB.peptides.find(x => x.id === calcState.pid);
      const dose = parseFloat(calcState.dose);
      if (!p || !p.bacWaterMl || !dose) return toast('Pick a reconstituted vial and dose first');
      logDose(p.id, dose);
    });
  }

  const q = () => {
    const mg = parseFloat($('#q-mg', v).value), bac = parseFloat($('#q-bac', v).value), dose = parseFloat($('#q-dose', v).value);
    const box = $('#q-result', v);
    if (mg > 0 && bac > 0 && dose > 0) {
      const units = unitsForDose(dose, mg, bac, DB.settings.syringeUnits);
      box.innerHTML = `<div class="result-box"><div class="big">${fmt(units,1)}</div><div class="unit">units</div><div class="sub">${fmt(mcgPerUnit(mg,bac,DB.settings.syringeUnits),1)} mcg/unit · ${fmt(units/DB.settings.syringeUnits,3)} mL</div></div>`;
    } else box.innerHTML = '';
  };
  ['#q-mg', '#q-bac', '#q-dose'].forEach(s => $(s, v).addEventListener('input', q));
}

/* ---------------- Log view ---------------- */
function logDose(pid, presetDose) {
  const p = DB.peptides.find(x => x.id === pid);
  const lastSite = DB.doses[0]?.site;
  const nextSiteIdx = lastSite ? (SITES.indexOf(lastSite) + 1) % SITES.length : 0;
  const opts = DB.peptides.map(x => `<option value="${x.id}" ${x.id === pid ? 'selected' : ''}>${esc(x.name)}</option>`).join('');
  const siteOpts = SITES.map((s, i) => `<option value="${s}" ${i === nextSiteIdx ? 'selected' : ''}>${s}</option>`).join('');
  const nowLocal = new Date(Date.now() - new Date().getTimezoneOffset() * 60000).toISOString().slice(0, 16);
  const body = el(`<div>
    <label class="field"><span class="lbl">Peptide</span><select id="l-pep">${opts}</select></label>
    <label class="field"><span class="lbl">Dose (mcg)</span><input id="l-dose" type="number" inputmode="decimal" value="${esc(presetDose ?? '')}" placeholder="250"></label>
    <div id="l-units" class="hint"></div>
    <label class="field"><span class="lbl">Injection site (rotates)</span><select id="l-site">${siteOpts}</select></label>
    <label class="field"><span class="lbl">Date & time</span><input id="l-time" type="datetime-local" value="${nowLocal}"></label>
    <label class="field"><span class="lbl">Notes</span><textarea id="l-notes" placeholder="Observations, side effects…"></textarea></label>
    <button class="btn" id="l-save">💉 Save injection</button>
  </div>`);
  const upd = () => {
    const pp = DB.peptides.find(x => x.id === $('#l-pep', body).value);
    const dose = parseFloat($('#l-dose', body).value);
    const box = $('#l-units', body);
    if (pp && pp.bacWaterMl && dose > 0) {
      const u = unitsForDose(dose, pp.vialMg, pp.bacWaterMl, DB.settings.syringeUnits);
      box.textContent = `≈ ${fmt(u,1)} units on the syringe`;
    } else box.textContent = pp && !pp.bacWaterMl ? '⚠ vial not reconstituted' : '';
  };
  openModal('Log injection', body, () => {
    $('#l-pep', body).addEventListener('change', upd);
    $('#l-dose', body).addEventListener('input', upd);
    upd();
    $('#l-save', body).addEventListener('click', () => {
      const pp = DB.peptides.find(x => x.id === $('#l-pep', body).value);
      const dose = parseFloat($('#l-dose', body).value);
      if (!pp || !dose) return toast('Peptide and dose required');
      const units = pp.bacWaterMl ? unitsForDose(dose, pp.vialMg, pp.bacWaterMl, DB.settings.syringeUnits) : 0;
      DB.doses.unshift({
        id: uid(), peptideId: pp.id, peptideName: pp.name, doseMcg: dose,
        units: Number(units.toFixed(2)), site: $('#l-site', body).value,
        ts: new Date($('#l-time', body).value).getTime(), notes: $('#l-notes', body).value.trim(),
      });
      // auto-decrement a syringe + swab if tracked
      decSupply('syringe'); decSupply('prep');
      save(); closeModal(); render(); toast('Injection logged');
    });
  });
}

function decSupply(keyword) {
  const s = DB.supplies.find(x => x.name.toLowerCase().includes(keyword) && x.qty > 0);
  if (s) s.qty -= 1;
}

function renderLog() {
  const v = $('#view-log');
  const rows = DB.doses.length ? DB.doses.map(d => `
    <div class="row" data-did="${d.id}">
      <div class="row-main">
        <div class="row-title">${esc(d.peptideName)} · ${d.doseMcg} mcg</div>
        <div class="row-sub">${d.units ? d.units + ' units · ' : ''}${esc(d.site || '')} · ${fmtDate(d.ts)}</div>
        ${d.notes ? `<div class="row-sub">📝 ${esc(d.notes)}</div>` : ''}
      </div>
      <div class="row-right">${timeAgo(d.ts)}</div>
    </div>`).join('') : `<div class="empty"><span class="em-icon">💉</span>No injections logged yet.</div>`;
  v.innerHTML = `
    <button class="btn" id="logNewBtn" style="margin-bottom:14px">＋ Log injection</button>
    <div class="card"><h2>History (${DB.doses.length})</h2>${rows}</div>
    ${DB.doses.length ? '<button class="btn secondary small" id="exportBtn" style="width:auto">⬇ Export CSV</button>' : ''}`;
  $('#logNewBtn').addEventListener('click', () => { if (!DB.peptides.length) return toast('Add a vial first'); logDose(DB.peptides[0].id); });
  v.querySelectorAll('[data-did]').forEach(r => r.addEventListener('click', () => {
    const d = DB.doses.find(x => x.id === r.dataset.did);
    if (confirm(`Delete this log entry?\n${d.peptideName} ${d.doseMcg}mcg · ${fmtDate(d.ts)}`)) {
      DB.doses = DB.doses.filter(x => x.id !== r.dataset.did); save(); render(); toast('Deleted');
    }
  }));
  if (DB.doses.length) $('#exportBtn').addEventListener('click', exportCsv);
}

function exportCsv() {
  const head = 'datetime,peptide,dose_mcg,units,site,notes\n';
  const rows = DB.doses.map(d => [new Date(d.ts).toISOString(), d.peptideName, d.doseMcg, d.units, d.site, (d.notes || '').replace(/"/g, '""')]
    .map(x => `"${x}"`).join(',')).join('\n');
  const blob = new Blob([head + rows], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `peptide-log-${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  toast('CSV exported');
}

/* ---------------- Supplies view ---------------- */
function renderSupplies() {
  const v = $('#view-supplies');
  const rows = DB.supplies.length ? DB.supplies.map(s => {
    const state = s.qty <= 0 ? 'out' : (s.qty <= s.low ? 'low' : 'ok');
    const badge = { ok: 'badge-ok', low: 'badge-low', out: 'badge-out' }[state];
    const label = { ok: 'in stock', low: 'low', out: 'out' }[state];
    return `<div class="row">
      <div class="row-main">
        <div class="row-title">${esc(s.name)}</div>
        <div class="row-sub"><span class="row-badge ${badge}">${label}</span> ${s.qty} ${esc(s.unit)}</div>
      </div>
      <div class="row-right right-actions">
        <button class="btn secondary small" data-dec="${s.id}">−</button>
        <button class="btn secondary small" data-inc="${s.id}">＋</button>
        <button class="icon-btn" data-edit="${s.id}">✎</button>
      </div>
    </div>`;
  }).join('') : `<div class="empty"><span class="em-icon">📦</span>No supplies tracked.</div>`;
  v.innerHTML = `
    <div class="card"><h2>Inventory</h2>${rows}</div>
    <button class="btn" id="addSupBtn">＋ Add supply</button>
    <p class="disclaimer">A syringe and prep pad are auto-deducted each time you log an injection.</p>`;
  $('#addSupBtn').addEventListener('click', () => editSupply());
  v.querySelectorAll('[data-inc]').forEach(b => b.addEventListener('click', () => { const s = DB.supplies.find(x => x.id === b.dataset.inc); s.qty++; save(); render(); }));
  v.querySelectorAll('[data-dec]').forEach(b => b.addEventListener('click', () => { const s = DB.supplies.find(x => x.id === b.dataset.dec); if (s.qty > 0) s.qty--; save(); render(); }));
  v.querySelectorAll('[data-edit]').forEach(b => b.addEventListener('click', () => editSupply(b.dataset.edit)));
}

function editSupply(id) {
  const s = id ? DB.supplies.find(x => x.id === id) : { name: '', qty: '', unit: 'pcs', low: '' };
  const body = el(`<div>
    <label class="field"><span class="lbl">Item name</span><input id="s-name" value="${esc(s.name)}" placeholder="e.g. Bacteriostatic water"></label>
    <div class="inline-2">
      <label class="field"><span class="lbl">Quantity</span><input id="s-qty" type="number" inputmode="numeric" value="${esc(s.qty)}"></label>
      <label class="field"><span class="lbl">Unit</span><input id="s-unit" value="${esc(s.unit)}" placeholder="pcs / vial / mL"></label>
    </div>
    <label class="field"><span class="lbl">Low-stock alert at</span><input id="s-low" type="number" inputmode="numeric" value="${esc(s.low)}" placeholder="20"></label>
    <div class="btn-row">
      <button class="btn" id="s-save">Save</button>
      ${id ? '<button class="btn danger small" id="s-del">Delete</button>' : ''}
    </div>
  </div>`);
  openModal(id ? 'Edit supply' : 'Add supply', body, () => {
    $('#s-save', body).addEventListener('click', () => {
      const name = $('#s-name', body).value.trim();
      if (!name) return toast('Name required');
      const rec = { id: s.id || uid(), name, qty: parseFloat($('#s-qty', body).value) || 0, unit: $('#s-unit', body).value.trim() || 'pcs', low: parseFloat($('#s-low', body).value) || 0 };
      if (id) DB.supplies = DB.supplies.map(x => x.id === id ? rec : x); else DB.supplies.push(rec);
      save(); closeModal(); render(); toast('Saved');
    });
    if (id) $('#s-del', body).addEventListener('click', () => { if (confirm('Delete supply?')) { DB.supplies = DB.supplies.filter(x => x.id !== id); save(); closeModal(); render(); } });
  });
}

/* ---------------- Home / dashboard ---------------- */
function renderHome() {
  const v = $('#view-home');
  const totalDoses = DB.doses.length;
  const last = DB.doses[0];
  const weekAgo = Date.now() - 7 * 86400000;
  const thisWeek = DB.doses.filter(d => d.ts >= weekAgo).length;
  const readyVials = DB.peptides.filter(p => p.bacWaterMl > 0).length;
  const lowSupplies = DB.supplies.filter(s => s.qty <= s.low);

  const lowHtml = lowSupplies.length ? `<div class="card">
    <h2>⚠ Restock needed</h2>
    ${lowSupplies.map(s => `<div class="row"><div class="row-main"><div class="row-title">${esc(s.name)}</div><div class="row-sub">${s.qty} ${esc(s.unit)} left</div></div><div class="row-right"><span class="row-badge ${s.qty <= 0 ? 'badge-out' : 'badge-low'}">${s.qty <= 0 ? 'out' : 'low'}</span></div></div>`).join('')}
  </div>` : '';

  const recent = DB.doses.slice(0, 5).map(d => `<div class="row"><div class="row-main"><div class="row-title">${esc(d.peptideName)} · ${d.doseMcg} mcg</div><div class="row-sub">${esc(d.site || '')} · ${fmtDate(d.ts)}</div></div><div class="row-right">${timeAgo(d.ts)}</div></div>`).join('');

  v.innerHTML = `
    <div class="stat-grid">
      <div class="stat"><div class="num">${thisWeek}</div><div class="cap">Doses this week</div></div>
      <div class="stat"><div class="num">${totalDoses}</div><div class="cap">Total logged</div></div>
      <div class="stat"><div class="num">${readyVials}/${DB.peptides.length}</div><div class="cap">Vials ready</div></div>
      <div class="stat"><div class="num">${last ? timeAgo(last.ts) : '—'}</div><div class="cap">Last injection</div></div>
    </div>
    ${lowHtml}
    <button class="btn" id="h-log" style="margin:4px 0 14px">💉 Log an injection</button>
    <div class="card">
      <h2>Recent activity</h2>
      ${recent || '<div class="empty" style="padding:20px"><span class="em-icon">📋</span>Nothing logged yet.<br>Add a vial, then log your first dose.</div>'}
    </div>
    <div class="btn-row">
      <button class="btn secondary" id="h-calc">🧮 Calculator</button>
      <button class="btn secondary" id="h-vials">🧪 Vials</button>
    </div>
    <p class="disclaimer" style="margin-top:16px">PeptideTrack stores everything only on this device. This tool supports research record-keeping and dose math; it is not medical advice and does not endorse administering any substance. Handle materials safely and dispose of sharps properly.</p>`;

  $('#h-log').addEventListener('click', () => { if (!DB.peptides.length) { switchView('peptides'); toast('Add a vial first'); return; } logDose(DB.peptides[0].id); });
  $('#h-calc').addEventListener('click', () => switchView('calc'));
  $('#h-vials').addEventListener('click', () => switchView('peptides'));
}

/* ---------------- Settings ---------------- */
function openSettings() {
  const body = el(`<div>
    <label class="field"><span class="lbl">Insulin syringe type</span>
      <select id="set-syr">
        <option value="100" ${DB.settings.syringeUnits === 100 ? 'selected' : ''}>U-100 (100 units = 1 mL) — standard</option>
        <option value="50" ${DB.settings.syringeUnits === 50 ? 'selected' : ''}>U-50 marking (50 units = 0.5 mL)</option>
        <option value="40" ${DB.settings.syringeUnits === 40 ? 'selected' : ''}>U-40 (40 units = 1 mL)</option>
      </select>
    </label>
    <p class="hint">U-100 is the standard insulin syringe. Only change this if your syringes are marked differently.</p>
    <hr class="divider">
    <button class="btn secondary" id="set-export">⬇ Export all data (JSON backup)</button>
    <label class="field" style="margin-top:12px"><span class="lbl">Restore from backup</span><input id="set-import" type="file" accept="application/json"></label>
    <hr class="divider">
    <button class="btn danger" id="set-reset">Reset all data</button>
    <p class="disclaimer" style="margin-top:14px">Data lives in this browser's local storage. To keep it as a home-screen app: tap Share → Add to Home Screen in Safari. Export a backup regularly — clearing Safari data will erase it.</p>
  </div>`);
  openModal('Settings', body, () => {
    $('#set-syr', body).addEventListener('change', (e) => { DB.settings.syringeUnits = parseInt(e.target.value, 10); save(); toast('Saved'); });
    $('#set-export', body).addEventListener('click', () => {
      const blob = new Blob([JSON.stringify(DB, null, 2)], { type: 'application/json' });
      const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
      a.download = `peptidetrack-backup-${new Date().toISOString().slice(0, 10)}.json`; a.click();
    });
    $('#set-import', body).addEventListener('change', (e) => {
      const f = e.target.files[0]; if (!f) return;
      const r = new FileReader();
      r.onload = () => { try { DB = { ...structuredClone(DEFAULT_DB), ...JSON.parse(r.result) }; save(); closeModal(); render(); toast('Restored'); } catch { toast('Invalid file'); } };
      r.readAsText(f);
    });
    $('#set-reset', body).addEventListener('click', () => { if (confirm('Erase ALL vials, logs and supplies?')) { DB = structuredClone(DEFAULT_DB); save(); closeModal(); render(); toast('Reset'); } });
  });
}

/* ---------------- Boot ---------------- */
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => navigator.serviceWorker.register('sw.js').catch(() => {}));
}
switchView('home');
