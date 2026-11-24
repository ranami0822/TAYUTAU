/** DOM controls (tool palette, selected-source editor, simulation controls) and the HUD. */
import type { App, Preset } from "../app";
import type { SimStats } from "../gpu/gpuSolver";
import type { Tool } from "./interaction";

const TOOLS: { id: Tool; label: string; title: string }[] = [
  { id: "hot", label: "🔥 加熱", title: "水槽をタップ／クリックして熱源を置く" },
  { id: "cold", label: "❄️ 冷却", title: "水槽をタップ／クリックして冷却源を置く" },
  { id: "move", label: "✋ 移動", title: "熱源をドラッグして動かす（他のツールでもドラッグで動かせます）" },
  { id: "erase", label: "🗑 消す", title: "熱源をタップして消す（ダブルクリック／Delete キーでも消せます）" },
];

export function buildUI(app: App) {
  const root = document.getElementById("ui")!;
  root.innerHTML = `
  <div class="panel" id="panel">
    <div class="title">TAYUTAU <span class="sub">3D 流体・相変化</span></div>
    <div class="row tools">${TOOLS.map((t) => `<button data-tool="${t.id}" title="${t.title}">${t.label}</button>`).join("")}</div>
    <div class="sel" id="sel">
      <div class="row"><label>温度 <b id="selT"></b></label><input id="selTr" type="range" min="-60" max="600" step="5"></div>
      <div class="row"><label>大きさ <b id="selR"></b></label><input id="selRr" type="range" min="0.05" max="0.25" step="0.01"></div>
      <div class="row"><button id="selDel">この熱源を消す</button></div>
    </div>
    <div class="hint" id="hint"></div>
    <div class="row">
      <button id="pause">⏸ 一時停止</button>
      <select id="speed" title="1 フレームあたりのシミュレーション時間">
        <option value="1">×0.5</option><option value="2" selected>×1（実時間）</option><option value="4">×2</option><option value="8">×4</option>
      </select>
    </div>
    <div class="row">
      <select id="preset"><option value="water">水 20 °C</option><option value="cold-water">冷水 2 °C</option><option value="ice-block">氷の塊（−15 °C）</option></select>
      <button id="reset">リセット</button>
    </div>
    <div class="row"><label><input type="checkbox" id="tempView"> 温度を強調表示</label></div>
    <div class="row">
      <label for="quality" style="flex:0 0 auto">描画解像度</label>
      <select id="quality"><option value="0.75">高</option><option value="1" selected>標準</option><option value="1.5">軽量</option><option value="2">最軽量</option></select>
    </div>
  </div>
  <div class="hud" id="hud"></div>
  <div class="legend"><span>0 °C</span><div class="bar"></div><span>100 °C</span></div>`;

  const st = app.state;
  const q = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
  const toolBtns = Array.from(root.querySelectorAll<HTMLButtonElement>("button[data-tool]"));
  toolBtns.forEach((b) =>
    b.addEventListener("click", () => {
      st.tool = b.dataset.tool as Tool;
      refresh();
    }),
  );
  const L = Math.min(st.params.nx, st.params.ny, st.params.nz) * st.params.h;
  q<HTMLInputElement>("selTr").addEventListener("input", (e) => {
    app.updateSource(st.selected, { T: Number((e.target as HTMLInputElement).value) });
    refresh();
  });
  q<HTMLInputElement>("selRr").addEventListener("input", (e) => {
    app.updateSource(st.selected, { r: Number((e.target as HTMLInputElement).value) * L });
    refresh();
  });
  q("selDel").addEventListener("click", () => {
    app.removeSource(st.selected);
    refresh();
  });
  q("pause").addEventListener("click", () => {
    st.paused = !st.paused;
    refresh();
  });
  q<HTMLSelectElement>("speed").addEventListener("change", (e) => (st.stepsPerFrame = Number((e.target as HTMLSelectElement).value)));
  q("reset").addEventListener("click", () => app.reset(q<HTMLSelectElement>("preset").value as Preset));
  q<HTMLSelectElement>("quality").addEventListener("change", (e) =>
    app.scene.getEngine().setHardwareScalingLevel(Number((e.target as HTMLSelectElement).value)),
  );
  q<HTMLInputElement>("tempView").addEventListener("change", (e) => (st.viewMode = (e.target as HTMLInputElement).checked ? 1 : 0));

  function refresh() {
    toolBtns.forEach((b) => b.classList.toggle("on", b.dataset.tool === st.tool));
    const s = st.sources[st.selected];
    q("sel").style.display = s ? "" : "none";
    if (s) {
      q("selT").textContent = `${Math.round(s.T)} °C`;
      q("selR").textContent = `${(s.r * 1000).toFixed(1)} mm`;
      q<HTMLInputElement>("selTr").value = String(s.T);
      q<HTMLInputElement>("selRr").value = String(s.r / L);
    }
    q("pause").textContent = st.paused ? "▶ 再開" : "⏸ 一時停止";
    const hints: Record<Tool, string> = {
      hot: "水槽をタップすると、見えている内壁に熱源を置きます。",
      cold: "水槽をタップすると、見えている内壁に冷却源を置きます。",
      move: "熱源をドラッグして動かします。空いている所のドラッグで視点を回します。",
      erase: "熱源をタップすると消えます。",
    };
    q("hint").textContent = `${hints[st.tool]}（熱源 ${st.sources.length}/8）`;
  }

  function showStats(s: SimStats) {
    const bal = s.energy - (st.stats0 ?? s.energy - s.injected) - s.injected;
    q("hud").innerHTML = `
      <div>時間 <b>${s.time.toFixed(1)} s</b></div>
      <div>平均 <b>${s.meanT.toFixed(1)} °C</b>　最高 <b>${s.maxT.toFixed(0)} °C</b>　最低 <b>${s.minT.toFixed(0)} °C</b></div>
      <div>氷 <b>${((100 * s.iceMass) / s.totalMass).toFixed(1)} %</b>　蒸気 <b>${((100 * s.vaporMass) / s.totalMass).toFixed(2)} %</b></div>
      <div title="(全エネルギーの変化) − (熱源が出し入れした熱)。0 なら保存している">エネルギー収支 <b>${bal >= 0 ? "+" : ""}${bal.toFixed(2)} J</b></div>`;
  }

  refresh();
  return { refresh, showStats };
}
