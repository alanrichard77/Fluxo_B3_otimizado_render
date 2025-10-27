function ensureFigure(fig){
  if(!fig || !fig.data){ return {data:[], layout:{}}; }
  return fig;
}

async function fetchJSON(url, opts){
  const r = await fetch(url, opts);
  if(!r.ok){ throw new Error("Falha HTTP "+r.status); }
  return await r.json();
}

function fmtBRLbi(x){
  const sign = x < 0 ? "-" : "";
  const v = Math.abs(x);
  return `${sign}R$ ${v.toFixed(1).replace(".", ",")}bi`;
}

function cardTemplate(title, value, hint){
  return `
    <div class="card">
      <h4>${title}</h4>
      <div class="value">${fmtBRLbi(value)}</div>
      <div class="hint">${hint}</div>
    </div>
  `;
}

async function renderCards(){
  const el = document.getElementById("cards");
  el.innerHTML = "";
  try{
    const data = await fetchJSON("/api/cards");
    const c = data.cards || {};
    el.insertAdjacentHTML("beforeend", cardTemplate("Estrangeiro", c["Estrangeiro"]?.valor || 0, c["Estrangeiro"]?.texto || ""));
    el.insertAdjacentHTML("beforeend", cardTemplate("Institucional", c["Institucional"]?.valor || 0, c["Institucional"]?.texto || ""));
    el.insertAdjacentHTML("beforeend", cardTemplate("Pessoa Física", c["Pessoa Física"]?.valor || 0, c["Pessoa Física"]?.texto || ""));
    el.insertAdjacentHTML("beforeend", cardTemplate("Inst. Financeira", c["Inst. Financeira"]?.valor || 0, c["Inst. Financeira"]?.texto || ""));
    el.insertAdjacentHTML("beforeend", cardTemplate("Outros", c["Outros"]?.valor || 0, c["Outros"]?.texto || ""));
  }catch(e){
    el.innerHTML = `<div class="card"><h4>Erro</h4><div class="hint">${e.message}</div></div>`;
  }
}

async function renderCharts(){
  const now = new Date();
  const start = new Date(now.getTime() - 180*24*3600*1000).toISOString().slice(0,10);
  const end = now.toISOString().slice(0,10);
  const data = await fetchJSON(`/api/series?start=${start}&end=${end}`);
  Plotly.newPlot("fig_main", data.fig_main.data, data.fig_main.layout, {displaylogo:false, responsive:true});
  Plotly.newPlot("fig_daily", data.fig_daily.data, data.fig_daily.layout, {displaylogo:false, responsive:true});
  Plotly.newPlot("fig_leaders", data.fig_leaders.data, data.fig_leaders.layout, {displaylogo:false, responsive:true});
  Plotly.newPlot("fig_heat", data.fig_heat.data, data.fig_heat.layout, {displaylogo:false, responsive:true});
  document.getElementById("updateDate").textContent = new Date().toLocaleString("pt-BR");
}

document.getElementById("btnRefresh").addEventListener("click", async ()=>{
  try{
    await fetchJSON("/api/refresh", {method:"POST"});
    await renderCards();
    await renderCharts();
  }catch(e){ alert("Erro, "+e.message); }
});

(async function init(){
  await renderCards();
  await renderCharts();
})();

