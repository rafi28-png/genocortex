/**
 * GenoCortex v1.0 — Live Browser Genomics Workbench UI Controller
 * ===============================================================
 * Connects real FASTA/FASTQ(.gz) file uploads, live NCBI E-utilities accession
 * fetching, off-main-thread Web Worker execution (fpc-worker.js), 2-D genome
 * track canvas rendering, lazy-loaded offline 3-D Plotly WebGL surfaces &
 * constellations, and standard bioinformatics file downloads (.bed, .bedgraph,
 * .nwk, .tsv).
 */

const SURPRISE_SCALE = [
  [0.0, "#0d1b3a"],
  [0.2, "#104281"],
  [0.4, "#1c5cab"],
  [0.6, "#2a78d6"],
  [0.8, "#5598e7"],
  [1.0, "#cde2fb"],
];

const CATEGORICAL = [
  "#3987e5",
  "#d95926",
  "#1baf7a",
  "#9085e9",
  "#c98500",
  "#e66767",
  "#26b3c6",
  "#d55181",
];

let worker = null;
let reqCounter = 0;
const pending = new Map();
let plotlyPromise = null;

function getWorker() {
  if (!worker) {
    worker = new Worker("./fpc-worker.js", { type: "module" });
    worker.onmessage = (ev) => {
      const msg = ev.data || {};
      const entry = pending.get(msg.id);
      if (!entry) return;
      if (msg.type === "progress") {
        if (entry.onProgress) entry.onProgress(msg.stage, msg.current, msg.total);
        return;
      }
      pending.delete(msg.id);
      if (msg.type === "result") {
        entry.resolve(msg.result);
      } else {
        entry.reject(new Error(msg.message || "Worker computation failed"));
      }
    };
    worker.onerror = (err) => {
      for (const [, entry] of pending) {
        entry.reject(new Error(err.message || "Web Worker fatal error"));
      }
      pending.clear();
    };
  }
  return worker;
}

function runWorkerTask(task, payload, onProgress) {
  const w = getWorker();
  const id = ++reqCounter;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject, onProgress });
    w.postMessage({ id, task, payload });
  });
}

function loadPlotly() {
  if (window.Plotly) return Promise.resolve(window.Plotly);
  if (plotlyPromise) return plotlyPromise;
  plotlyPromise = new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "./vendor/plotly.min.js";
    script.async = true;
    script.onload = () => {
      if (window.Plotly) resolve(window.Plotly);
      else reject(new Error("Plotly loaded without global window.Plotly"));
    };
    script.onerror = () => {
      plotlyPromise = null;
      reject(new Error("Failed to load local ./vendor/plotly.min.js bundle"));
    };
    document.head.appendChild(script);
  });
  return plotlyPromise;
}

async function readUploadedFiles(fileList, labelByFilename = false) {
  const chunks = [];
  for (const file of Array.from(fileList)) {
    let text = "";
    if (file.name.toLowerCase().endsWith(".gz")) {
      if (typeof DecompressionStream === "undefined") {
        throw new Error("This browser does not support native .gz DecompressionStream; gunzip first.");
      }
      const ds = file.stream().pipeThrough(new DecompressionStream("gzip"));
      text = await new Response(ds).text();
    } else {
      text = await file.text();
    }
    text = text.trim();
    if (!text) continue;
    if (labelByFilename && text.startsWith(">")) {
      const stem = file.name.replace(/\.gz$/i, "").replace(/\.[^.]+$/, "");
      // Concatenate contigs in file under single >stem header for multi-file panels
      const seqBody = text
        .split(/\r?\n/)
        .filter((l) => !l.trim().startsWith(">"))
        .join("");
      chunks.push(`>${stem}\n${seqBody}`);
    } else {
      chunks.push(text);
    }
  }
  return chunks.join("\n");
}

async function fetchNcbiAccessions(accessionCsv, db = "nucleotide") {
  const ids = accessionCsv
    .split(/[\s,;]+/)
    .map((s) => s.trim())
    .filter(Boolean);
  if (ids.length === 0) {
    throw new Error("Enter at least one NCBI accession (e.g. NC_045512.2)");
  }
  const url =
    `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=${encodeURIComponent(db)}` +
    `&id=${encodeURIComponent(ids.join(","))}&rettype=fasta&retmode=text`;
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`NCBI efetch returned HTTP ${resp.status} for ${ids.join(", ")}`);
  }
  const text = (await resp.text()).trim();
  if (!text.startsWith(">")) {
    throw new Error(`NCBI returned non-FASTA response for ${ids.join(", ")}`);
  }
  return text;
}

function downloadBlob(filename, content, mime = "text/plain;charset=utf-8") {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// Deterministic offline synthetic biological fixtures (explicitly triggered ONLY via "Load Offline Sample" button)
function makeOfflineChimeraSample() {
  let state = 1337;
  const rnd = () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
  // Host chromosome segment: AT-biased structured Markov motifs (~38% GC)
  const hostMotifs = ["AATTGCAT", "ATGCGTTA", "TTAAGCAT", "CATTAAGC", "AATGCTTA", "TTGCATAA"];
  // Foreign pathogenicity island: GC-rich CpG/GGCC island (~72% GC)
  const donorMotifs = ["GCGCACGG", "CCGGCGCA", "GGCGCGCC", "CGCGTACG", "GCCGGCGC"];

  const buildHost = (nMotifs) => {
    const out = [];
    for (let i = 0; i < nMotifs; i++) {
      let m = hostMotifs[Math.floor(rnd() * hostMotifs.length)];
      if (rnd() < 0.25) {
        const pos = Math.floor(rnd() * m.length);
        m = m.slice(0, pos) + "ACGT"[Math.floor(rnd() * 4)] + m.slice(pos + 1);
      }
      out.push(m);
    }
    return out.join("");
  };
  const buildDonor = (nMotifs) => {
    const out = [];
    for (let i = 0; i < nMotifs; i++) {
      out.push(donorMotifs[Math.floor(rnd() * donorMotifs.length)]);
    }
    return out.join("");
  };

  const ref = buildHost(2400); // 19.2 kb reference
  const left = buildHost(1000); // 8.0 kb
  const island1 = buildDonor(220); // 1,760 bp island at 8,000..9,760
  const mid = buildHost(900); // 7.2 kb
  const island2 = buildDonor(180); // 1,440 bp island at 16,960..18,400
  const right = buildHost(700); // 5.6 kb
  return {
    ref: `>host_core_ref Synthetic core chromosome (38% GC)\n${ref}`,
    query: `>strain_query_chr1 Chimeric chromosome with 2 HGT islands (8000..9760 & 16960..18400)\n${left}${island1}${mid}${island2}${right}`,
  };
}

function makeOfflinePhyloSample() {
  let state = 2026;
  const rnd = () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
  const mutate = (seq, rate) => {
    const bases = "ACGT";
    const arr = seq.split("");
    for (let i = 0; i < arr.length; i++) {
      if (rnd() < rate) arr[i] = bases[Math.floor(rnd() * 4)];
    }
    return arr.join("");
  };
  let anc = "";
  const motifs = ["ACGTTGCA", "GCTATAGC", "TTCGAAGG", "CGGATCCA", "AAGCTTCG"];
  for (let i = 0; i < 500; i++) anc += motifs[Math.floor(rnd() * motifs.length)];

  const cladeA = mutate(anc, 0.08);
  const cladeB = mutate(anc, 0.14);
  const cladeC = mutate(anc, 0.24);
  const taxa = [
    ["Sarbeco_alpha", mutate(cladeA, 0.03)],
    ["Sarbeco_beta", mutate(cladeA, 0.035)],
    ["Merbeco_1", mutate(cladeB, 0.04)],
    ["Merbeco_2", mutate(cladeB, 0.045)],
    ["Embeco_out1", mutate(cladeC, 0.05)],
    ["Embeco_out2", mutate(cladeC, 0.055)],
  ];
  return taxa.map(([nm, s]) => `>${nm}\n${s}`).join("\n");
}

function makeOfflineClassifySample() {
  const sample = makeOfflinePhyloSample();
  const recs = sample
    .split(">")
    .filter(Boolean)
    .map((blk) => {
      const [h, ...rest] = blk.trim().split(/\n/);
      return { id: h.trim(), seq: rest.join("") };
    });
  const revComp = (s) =>
    s
      .split("")
      .reverse()
      .map((c) => ({ A: "T", T: "A", C: "G", G: "C" }[c] || "N"))
      .join("");

  const refs = recs.slice(0, 3);
  const reads = [
    [
      "read_01_fwd_Sarbeco",
      refs[0].seq.slice(400, 560),
      "I".repeat(160),
    ],
    [
      "read_02_revcomp_Sarbeco",
      revComp(refs[0].seq.slice(1200, 1360)),
      "I".repeat(150) + "##########",
    ],
    [
      "read_03_fwd_Merbeco",
      refs[1].seq.slice(800, 960),
      "H".repeat(160),
    ],
    [
      "read_04_revcomp_Embeco",
      revComp(refs[2].seq.slice(1800, 1960)),
      "I".repeat(160),
    ],
  ];
  return {
    refs: refs.map((r) => `>${r.id}\n${r.seq}`).join("\n"),
    queries: reads.map(([id, sq, ql]) => `@${id}\n${sq}\n+\n${ql}`).join("\n"),
  };
}

// ---------------------------------------------------------------------------
// 2-D Genome Surprise & Z-Score Canvas Renderer
// ---------------------------------------------------------------------------
function drawSurpriseTrackCanvas(canvas, track, islands, zThreshold) {
  if (!canvas || !track || !track.starts || track.starts.length === 0) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(320, Math.floor(rect.width || 800));
  const height = 210;
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  ctx.fillStyle = "#0e1013";
  ctx.fillRect(0, 0, width, height);

  const padL = 52;
  const padR = 18;
  const padT = 22;
  const padB = 32;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;

  const maxBp = track.ends[track.ends.length - 1] || 1;
  const zs = track.zScores.filter((v) => Number.isFinite(v));
  const minZ = Math.min(-2.0, ...zs);
  const maxZ = Math.max(zThreshold + 1.5, ...zs);
  const spanZ = Math.max(1e-6, maxZ - minZ);

  const xOf = (bp) => padL + (bp / maxBp) * plotW;
  const yOf = (z) => padT + plotH - ((z - minZ) / spanZ) * plotH;

  // Highlight called island spans
  for (const isl of islands) {
    if (isl.chrom !== track.chrom) continue;
    const x0 = xOf(isl.start);
    const x1 = Math.max(x0 + 2, xOf(isl.end));
    ctx.fillStyle = isl.significant ? "rgba(235, 104, 52, 0.24)" : "rgba(201, 133, 0, 0.16)";
    ctx.fillRect(x0, padT, x1 - x0, plotH);
  }

  // Horizontal grid lines & threshold line
  ctx.strokeStyle = "rgba(255,255,255,0.07)";
  ctx.lineWidth = 1;
  const yZero = yOf(0);
  ctx.beginPath();
  ctx.moveTo(padL, yZero);
  ctx.lineTo(padL + plotW, yZero);
  ctx.stroke();

  const yThresh = yOf(zThreshold);
  ctx.save();
  ctx.setLineDash([5, 4]);
  ctx.strokeStyle = "#eb6834";
  ctx.beginPath();
  ctx.moveTo(padL, yThresh);
  ctx.lineTo(padL + plotW, yThresh);
  ctx.stroke();
  ctx.restore();

  // Z-score trace
  ctx.strokeStyle = "#5598e7";
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  let started = false;
  for (let i = 0; i < track.starts.length; i++) {
    const z = track.zScores[i];
    if (!Number.isFinite(z)) continue;
    const xm = xOf(0.5 * (track.starts[i] + track.ends[i]));
    const ym = yOf(z);
    if (!started) {
      ctx.moveTo(xm, ym);
      started = true;
    } else {
      ctx.lineTo(xm, ym);
    }
  }
  ctx.stroke();

  // Axes & labels
  ctx.fillStyle = "#a8abb0";
  ctx.font = "11px system-ui, -apple-system, sans-serif";
  ctx.textAlign = "right";
  ctx.fillText(`z=${maxZ.toFixed(1)}`, padL - 6, padT + 8);
  ctx.fillText("z=0.0", padL - 6, Math.min(padT + plotH, Math.max(padT + 14, yZero + 4)));
  ctx.fillStyle = "#eb6834";
  ctx.fillText(`cut ${zThreshold.toFixed(1)}`, padL - 6, Math.max(padT + 10, yThresh - 3));

  ctx.fillStyle = "#a8abb0";
  ctx.textAlign = "left";
  ctx.fillText(`${track.chrom}: 0 bp`, padL, height - 10);
  ctx.textAlign = "center";
  ctx.fillText(`${Math.round(maxBp / 2).toLocaleString()} bp`, padL + plotW / 2, height - 10);
  ctx.textAlign = "right";
  ctx.fillText(`${maxBp.toLocaleString()} bp`, padL + plotW, height - 10);
}

// ---------------------------------------------------------------------------
// 3-D Plotly Renderers (Cognitive Terrain & Phylogeny Constellation)
// ---------------------------------------------------------------------------
async function renderCognitive3D(container, terrain, titleText) {
  const Plotly = await loadPlotly();
  const traces = [
    {
      type: "surface",
      x: terrain.xs,
      y: terrain.ys,
      z: terrain.zRows,
      colorscale: SURPRISE_SCALE,
      connectgaps: true,
      opacity: 0.97,
      colorbar: {
        title: { text: "surprise<br>(bits)", font: { color: "#c3c2b7", size: 11 } },
        tickfont: { color: "#c3c2b7", size: 10 },
        len: 0.65,
      },
      hovertemplate: "CGR-x %{x:.3f}<br>CGR-y %{y:.3f}<br>surprise %{z:.2f} bits<extra></extra>",
    },
  ];

  const pf = terrain.prototypes;
  const m = pf.x.length;
  const maxShown = 2500;
  let indices = Array.from({ length: m }, (_, i) => i);
  if (m > maxShown) {
    indices.sort((a, b) => pf.usage[b] - pf.usage[a]);
    indices = indices.slice(0, maxShown);
  }
  const grid = terrain.xs.length;
  const lift = 0.05 * Math.max(0.2, terrain.zMax - terrain.zMin);
  const maxUsage = indices.reduce((mx, idx) => Math.max(mx, pf.usage[idx]), 1e-9);

  for (let s = 0; s < terrain.symbols.length; s++) {
    const px = [];
    const py = [];
    const pz = [];
    const psize = [];
    for (const idx of indices) {
      if (pf.dominant[idx] !== s) continue;
      const gx = Math.min(grid - 1, Math.max(0, Math.floor(pf.x[idx] * grid)));
      const gy = Math.min(grid - 1, Math.max(0, Math.floor(pf.y[idx] * grid)));
      const cellZ = terrain.zRows[gy][gx];
      px.push(pf.x[idx]);
      py.push(pf.y[idx]);
      pz.push((cellZ !== null ? cellZ : terrain.zMin) + lift);
      psize.push(3.5 + 7.5 * (pf.usage[idx] / maxUsage));
    }
    if (px.length > 0) {
      traces.push({
        type: "scatter3d",
        mode: "markers",
        x: px,
        y: py,
        z: pz,
        name: `predicts '${terrain.symbols[s]}'`,
        marker: {
          size: psize,
          color: CATEGORICAL[s % CATEGORICAL.length],
          opacity: 0.92,
        },
        hovertemplate: `prototype → '${terrain.symbols[s]}'<br>CGR (%{x:.3f}, %{y:.3f})<extra></extra>`,
      });
    }
  }

  const axisStyle = (title) => ({
    title: { text: title, font: { color: "#c3c2b7", size: 11 } },
    backgroundcolor: "#16171a",
    gridcolor: "#2a2c30",
    zerolinecolor: "#2a2c30",
    showbackground: true,
    tickfont: { color: "#898781", size: 9 },
  });

  await Plotly.react(
    container,
    traces,
    {
      title: { text: titleText, font: { color: "#f4f5f7", size: 14 } },
      paper_bgcolor: "#0e1013",
      plot_bgcolor: "#0e1013",
      margin: { l: 0, r: 0, t: 38, b: 0 },
      legend: { font: { color: "#c3c2b7", size: 10 }, bgcolor: "rgba(0,0,0,0)" },
      scene: {
        xaxis: axisStyle("Chaos-Game x"),
        yaxis: axisStyle("Chaos-Game y"),
        zaxis: axisStyle("surprise (bits)"),
        camera: { eye: { x: 1.48, y: 1.48, z: 1.12 } },
        aspectmode: "cube",
      },
    },
    { responsive: true, displaylogo: false }
  );
}

async function renderPhylogeny3D(container, coords, names, mstEdges) {
  const Plotly = await loadPlotly();
  const ex = [];
  const ey = [];
  const ez = [];
  for (const [u, v] of mstEdges) {
    ex.push(coords[u][0], coords[v][0], null);
    ey.push(coords[u][1], coords[v][1], null);
    ez.push(coords[u][2] || 0, coords[v][2] || 0, null);
  }
  const traces = [
    {
      type: "scatter3d",
      mode: "lines",
      x: ex,
      y: ey,
      z: ez,
      line: { color: "#68707c", width: 3 },
      name: "MST lineage",
      hoverinfo: "skip",
    },
    {
      type: "scatter3d",
      mode: "markers+text",
      x: coords.map((c) => c[0]),
      y: coords.map((c) => c[1]),
      z: coords.map((c) => c[2] || 0),
      text: names,
      textposition: "top center",
      textfont: { color: "#e6e8ec", size: 10 },
      marker: {
        size: 8,
        color: names.map((_, i) => CATEGORICAL[i % CATEGORICAL.length]),
        line: { width: 1, color: "#0e1013" },
      },
      name: "Taxa",
      hovertemplate: "<b>%{text}</b><br>MDS (%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>",
    },
  ];

  const axisStyle = (title) => ({
    title: { text: title, font: { color: "#c3c2b7", size: 11 } },
    backgroundcolor: "#16171a",
    gridcolor: "#2a2c30",
    zerolinecolor: "#2a2c30",
    showbackground: true,
    tickfont: { color: "#898781", size: 9 },
  });

  await Plotly.react(
    container,
    traces,
    {
      title: {
        text: "3-D Cross-Entropy MDS Constellation + Minimum Spanning Tree",
        font: { color: "#f4f5f7", size: 14 },
      },
      paper_bgcolor: "#0e1013",
      plot_bgcolor: "#0e1013",
      margin: { l: 0, r: 0, t: 38, b: 0 },
      showlegend: false,
      scene: {
        xaxis: axisStyle("MDS-1"),
        yaxis: axisStyle("MDS-2"),
        zaxis: axisStyle("MDS-3"),
        camera: { eye: { x: 1.45, y: 1.45, z: 1.05 } },
        aspectmode: "cube",
      },
    },
    { responsive: true, displaylogo: false }
  );
}

// ---------------------------------------------------------------------------
// DOM Wiring
// ---------------------------------------------------------------------------
function initWorkbench() {
  const root = document.getElementById("gc-workbench");
  if (!root) return;

  const statusEl = document.getElementById("wb-status");
  const setStatus = (text, kind = "idle") => {
    statusEl.textContent = text;
    statusEl.dataset.state = kind;
  };

  // Tab switching
  const tabBtns = root.querySelectorAll("[data-wb-tab]");
  const panels = root.querySelectorAll("[data-wb-panel]");
  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.wbTab;
      tabBtns.forEach((b) => {
        const active = b.dataset.wbTab === target;
        b.classList.toggle("active", active);
        b.setAttribute("aria-selected", String(active));
      });
      panels.forEach((p) => {
        p.hidden = p.dataset.wbPanel !== target;
      });
    });
  });

  // ---------------- TAB 1: SCAN ----------------
  const scanQueryEl = document.getElementById("wb-scan-query");
  const scanRefEl = document.getElementById("wb-scan-ref");
  let lastScanDownloads = null;

  document.getElementById("wb-scan-query-file").addEventListener("change", async (e) => {
    try {
      if (!e.target.files.length) return;
      setStatus("Reading query sequence file...", "busy");
      scanQueryEl.value = await readUploadedFiles(e.target.files, false);
      setStatus(`Loaded ${e.target.files[0].name} (${scanQueryEl.value.length.toLocaleString()} chars)`, "ok");
    } catch (err) {
      setStatus(err.message, "err");
    }
  });

  document.getElementById("wb-scan-ref-file").addEventListener("change", async (e) => {
    try {
      if (!e.target.files.length) return;
      setStatus("Reading reference sequence file...", "busy");
      scanRefEl.value = await readUploadedFiles(e.target.files, false);
      setStatus(`Loaded reference ${e.target.files[0].name}`, "ok");
    } catch (err) {
      setStatus(err.message, "err");
    }
  });

  document.getElementById("wb-scan-ncbi-btn").addEventListener("click", async () => {
    const acc = document.getElementById("wb-scan-ncbi-id").value;
    const db = document.getElementById("wb-scan-alpha").value === "PROTEIN" ? "protein" : "nucleotide";
    try {
      setStatus(`Fetching ${acc} live from NCBI E-utilities...`, "busy");
      scanQueryEl.value = await fetchNcbiAccessions(acc, db);
      setStatus(`Fetched ${acc} from NCBI (${scanQueryEl.value.length.toLocaleString()} chars)`, "ok");
    } catch (err) {
      setStatus(`NCBI fetch failed: ${err.message}`, "err");
    }
  });

  document.getElementById("wb-scan-sample-btn").addEventListener("click", () => {
    const { ref, query } = makeOfflineChimeraSample();
    scanRefEl.value = ref;
    scanQueryEl.value = query;
    setStatus("Loaded 24 kb chimeric chromosome + 19.2 kb core reference", "ok");
  });

  document.getElementById("wb-scan-run").addEventListener("click", async () => {
    try {
      const zThreshold = Number(document.getElementById("wb-scan-z").value || 3.2);
      setStatus("Starting Web Worker island scan...", "busy");
      const res = await runWorkerTask(
        "scan",
        {
          alphabet: document.getElementById("wb-scan-alpha").value,
          order: Number(document.getElementById("wb-scan-order").value || 7),
          window: Number(document.getElementById("wb-scan-win").value || 500),
          step: Math.max(25, Math.floor(Number(document.getElementById("wb-scan-win").value || 500) / 4)),
          zThreshold,
          minPhred: Number(document.getElementById("wb-scan-phred").value || 0),
          queryText: scanQueryEl.value,
          refText: scanRefEl.value,
        },
        (stage, cur, tot) => setStatus(`Running ${stage} (${cur.toLocaleString()}/${tot.toLocaleString()})...`, "busy")
      );

      lastScanDownloads = res.downloads;
      document.getElementById("wb-scan-results").hidden = false;
      const mbps = (res.totalBp / Math.max(1, res.elapsedMs) / 1000).toFixed(2);
      const nSig = res.islands.filter((x) => x.significant).length;
      document.getElementById("wb-scan-metrics").innerHTML =
        `<span><strong>${res.totalBp.toLocaleString()}</strong> bp scanned</span>` +
        `<span><strong>${res.elapsedMs.toFixed(1)}</strong> ms (${mbps} Mb/s)</span>` +
        `<span><strong>${res.nPrototypes.toLocaleString()}</strong> CGR prototypes</span>` +
        `<span><strong>${nSig}</strong> FDR-significant islands (${res.islands.length} total)</span>`;

      drawSurpriseTrackCanvas(
        document.getElementById("wb-scan-canvas"),
        res.track,
        res.islands,
        zThreshold
      );

      const tbody = document.getElementById("wb-scan-tbody");
      if (res.islands.length === 0) {
        tbody.innerHTML = `<tr><td colspan="8" class="muted">No windows exceeded z ≥ ${zThreshold.toFixed(1)}.</td></tr>`;
      } else {
        tbody.innerHTML = res.islands
          .map(
            (isl) =>
              `<tr>` +
              `<td><code>${isl.chrom}</code></td>` +
              `<td>${isl.start.toLocaleString()}–${isl.end.toLocaleString()}</td>` +
              `<td>${isl.length.toLocaleString()} bp</td>` +
              `<td><strong>${isl.peakZ.toFixed(2)}</strong></td>` +
              `<td>${isl.meanBits.toFixed(3)} (${isl.deltaBits >= 0 ? "+" : ""}${isl.deltaBits.toFixed(3)})</td>` +
              `<td>${Number.isFinite(isl.gcContent) ? (isl.gcContent * 100).toFixed(1) + "%" : "NA"} ` +
              `(${Number.isFinite(isl.deltaGc) ? (isl.deltaGc >= 0 ? "+" : "") + (isl.deltaGc * 100).toFixed(1) + "%" : "NA"})</td>` +
              `<td>${isl.qValue.toExponential(2)}</td>` +
              `<td><span class="wb-badge ${isl.significant ? "sig" : "cand"}">${isl.significant ? "FDR ≤ 0.05" : "candidate"}</span></td>` +
              `</tr>`
          )
          .join("");
      }

      setStatus("Rendering live 3-D WebGL cognitive surface...", "busy");
      await renderCognitive3D(
        document.getElementById("wb-scan-3d"),
        res.terrain,
        `3-D Fractal Surprise Terrain — ${res.track.chrom}`
      );
      setStatus(`Completed scan of ${res.totalBp.toLocaleString()} bp in ${res.elapsedMs.toFixed(0)} ms`, "ok");
    } catch (err) {
      setStatus(err.message, "err");
    }
  });

  document.getElementById("wb-dl-bed").addEventListener("click", () => {
    if (lastScanDownloads) downloadBlob("genocortex_islands.bed", lastScanDownloads.bed);
  });
  document.getElementById("wb-dl-bedgraph").addEventListener("click", () => {
    if (lastScanDownloads) downloadBlob("genocortex_surprise.bedgraph", lastScanDownloads.bedgraph);
  });
  document.getElementById("wb-dl-scan-tsv").addEventListener("click", () => {
    if (lastScanDownloads) downloadBlob("genocortex_islands.tsv", lastScanDownloads.tsv);
  });

  // ---------------- TAB 2: TREE ----------------
  const treeFastaEl = document.getElementById("wb-tree-fasta");
  let lastTreeDownloads = null;

  document.getElementById("wb-tree-file").addEventListener("change", async (e) => {
    try {
      if (!e.target.files.length) return;
      setStatus("Reading FASTA file(s)...", "busy");
      const perFile = e.target.files.length > 1;
      treeFastaEl.value = await readUploadedFiles(e.target.files, perFile);
      setStatus(`Loaded ${e.target.files.length} file(s)`, "ok");
    } catch (err) {
      setStatus(err.message, "err");
    }
  });

  document.getElementById("wb-tree-ncbi-btn").addEventListener("click", async () => {
    const accs = document.getElementById("wb-tree-ncbi-ids").value;
    const db = document.getElementById("wb-tree-alpha").value === "PROTEIN" ? "protein" : "nucleotide";
    try {
      setStatus(`Fetching ${accs} live from NCBI...`, "busy");
      treeFastaEl.value = await fetchNcbiAccessions(accs, db);
      setStatus(`Fetched multi-FASTA from NCBI`, "ok");
    } catch (err) {
      setStatus(`NCBI fetch failed: ${err.message}`, "err");
    }
  });

  document.getElementById("wb-tree-sample-btn").addEventListener("click", () => {
    treeFastaEl.value = makeOfflinePhyloSample();
    setStatus("Loaded 6-taxon multi-FASTA lineage set (24 kb)", "ok");
  });

  document.getElementById("wb-tree-run").addEventListener("click", async () => {
    try {
      setStatus("Building cross-entropy distance matrix & Neighbour-Joining tree...", "busy");
      const res = await runWorkerTask(
        "tree",
        {
          alphabet: document.getElementById("wb-tree-alpha").value,
          order: Number(document.getElementById("wb-tree-order").value || 6),
          strandInvariant: document.getElementById("wb-tree-strand").checked,
          fastaText: treeFastaEl.value,
        },
        (stage) => setStatus(`Phylogeny: ${stage}...`, "busy")
      );
      lastTreeDownloads = res.downloads;
      document.getElementById("wb-tree-results").hidden = false;
      document.getElementById("wb-tree-newick").textContent = res.newick;

      const headHtml =
        "<tr><th>Taxon</th>" + res.names.map((n) => `<th><code>${n}</code></th>`).join("") + "</tr>";
      const rowsHtml = res.names
        .map(
          (nm, i) =>
            `<tr><td><strong><code>${nm}</code></strong></td>` +
            res.distanceMatrix[i].map((v) => `<td>${v.toFixed(4)}</td>`).join("") +
            `</tr>`
        )
        .join("");
      document.getElementById("wb-tree-matrix").innerHTML =
        `<thead>${headHtml}</thead><tbody>${rowsHtml}</tbody>`;

      setStatus("Rendering 3-D MDS constellation...", "busy");
      await renderPhylogeny3D(
        document.getElementById("wb-tree-3d"),
        res.coords,
        res.names,
        res.mstEdges
      );
      setStatus(
        `Built Neighbour-Joining tree for ${res.names.length} taxa (${res.totalBp.toLocaleString()} bp) in ${res.elapsedMs.toFixed(0)} ms`,
        "ok"
      );
    } catch (err) {
      setStatus(err.message, "err");
    }
  });

  document.getElementById("wb-dl-nwk").addEventListener("click", () => {
    if (lastTreeDownloads) downloadBlob("genocortex_tree.nwk", lastTreeDownloads.newick);
  });
  document.getElementById("wb-dl-dist").addEventListener("click", () => {
    if (lastTreeDownloads) downloadBlob("genocortex_distances.tsv", lastTreeDownloads.tsv);
  });

  // ---------------- TAB 3: CLASSIFY ----------------
  const clfRefsEl = document.getElementById("wb-clf-refs");
  const clfQueriesEl = document.getElementById("wb-clf-queries");
  let lastClfDownloads = null;

  document.getElementById("wb-clf-ref-file").addEventListener("change", async (e) => {
    try {
      if (!e.target.files.length) return;
      const perFile = e.target.files.length > 1;
      clfRefsEl.value = await readUploadedFiles(e.target.files, perFile);
      setStatus(`Loaded ${e.target.files.length} reference file(s)`, "ok");
    } catch (err) {
      setStatus(err.message, "err");
    }
  });

  document.getElementById("wb-clf-query-file").addEventListener("change", async (e) => {
    try {
      if (!e.target.files.length) return;
      clfQueriesEl.value = await readUploadedFiles(e.target.files, false);
      setStatus(`Loaded query reads from ${e.target.files[0].name}`, "ok");
    } catch (err) {
      setStatus(err.message, "err");
    }
  });

  document.getElementById("wb-clf-sample-btn").addEventListener("click", () => {
    const s = makeOfflineClassifySample();
    clfRefsEl.value = s.refs;
    clfQueriesEl.value = s.queries;
    setStatus("Loaded 3 reference genomes + 4 unoriented (+/-) FASTQ reads", "ok");
  });

  document.getElementById("wb-clf-run").addEventListener("click", async () => {
    try {
      setStatus("Training reference cortices & classifying reads...", "busy");
      const res = await runWorkerTask(
        "classify",
        {
          alphabet: document.getElementById("wb-clf-alpha").value,
          order: Number(document.getElementById("wb-clf-order").value || 7),
          minPhred: Number(document.getElementById("wb-clf-phred").value || 20),
          strandAware: document.getElementById("wb-clf-strand").checked,
          refText: clfRefsEl.value,
          queryText: clfQueriesEl.value,
        },
        (stage) => setStatus(`Classifier: ${stage}...`, "busy")
      );
      lastClfDownloads = res.downloads;
      document.getElementById("wb-clf-results").hidden = false;
      document.getElementById("wb-clf-tbody").innerHTML = res.calls
        .map(
          (c) =>
            `<tr>` +
            `<td><code>${c.id}</code></td>` +
            `<td>${c.length.toLocaleString()} bp</td>` +
            `<td><strong style="color:var(--accent)">${c.predictedClass}</strong></td>` +
            `<td><code>${c.strand}</code></td>` +
            `<td>${c.bitsPerBase.toFixed(4)}</td>` +
            `<td>${(c.posterior * 100).toFixed(1)}%</td>` +
            `<td>${c.runnerUp ?? "NA"} (+${c.deltaBits.toFixed(4)} bits)</td>` +
            `</tr>`
        )
        .join("");
      setStatus(
        `Classified ${res.calls.length} read(s) across ${res.classes.length} classes in ${res.elapsedMs.toFixed(0)} ms`,
        "ok"
      );
    } catch (err) {
      setStatus(err.message, "err");
    }
  });

  document.getElementById("wb-dl-clf").addEventListener("click", () => {
    if (lastClfDownloads) downloadBlob("genocortex_classifications.tsv", lastClfDownloads.tsv);
  });

  // ---------------- TAB 4: PREDICT ----------------
  const predTrainEl = document.getElementById("wb-pred-train");
  document.getElementById("wb-pred-sample-btn").addEventListener("click", () => {
    const { ref } = makeOfflineChimeraSample();
    predTrainEl.value = ref;
    document.getElementById("wb-pred-prefix").value = "AATTGCATATGCGTTA";
    setStatus("Loaded structured training sequence & seed prefix", "ok");
  });

  document.getElementById("wb-pred-run").addEventListener("click", async () => {
    try {
      setStatus("Learning sequence & predicting continuation...", "busy");
      const res = await runWorkerTask("predict", {
        alphabet: document.getElementById("wb-pred-alpha").value,
        order: Number(document.getElementById("wb-pred-order").value || 7),
        trainText: predTrainEl.value,
        prefix: document.getElementById("wb-pred-prefix").value,
        length: Number(document.getElementById("wb-pred-len").value || 120),
        sample: document.getElementById("wb-pred-stochastic").checked,
      });
      document.getElementById("wb-pred-results").hidden = false;
      document.getElementById("wb-pred-metrics").innerHTML =
        `<span>Trained on <strong>${res.trainBp.toLocaleString()}</strong> symbols</span>` +
        `<span><strong>${res.nPrototypes.toLocaleString()}</strong> prototypes</span>` +
        `<span>Cross-entropy: <strong>${res.meanBits.toFixed(3)}</strong> bits/symbol</span>` +
        `<span>Perplexity: <strong>${res.perplexity.toFixed(2)}</strong></span>`;

      const barsHtml = res.symbols
        .map((sym, i) => {
          const pct = (res.nextDist[i] * 100).toFixed(1);
          return (
            `<div class="wb-prob-row">` +
            `<code>${sym}</code>` +
            `<div class="wb-prob-track"><div class="wb-prob-fill" style="width:${pct}%"></div></div>` +
            `<span>${pct}%</span>` +
            `</div>`
          );
        })
        .join("");
      document.getElementById("wb-pred-bars").innerHTML = barsHtml;
      document.getElementById("wb-pred-seq").innerHTML =
        `<span class="muted">${res.prefix}</span><strong style="color:var(--accent-2)">${res.continuation}</strong>`;
      setStatus(`Generated ${res.continuation.length} symbols in ${res.elapsedMs.toFixed(0)} ms`, "ok");
    } catch (err) {
      setStatus(err.message, "err");
    }
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initWorkbench);
} else {
  initWorkbench();
}
