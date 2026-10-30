/**
 * GenoCortex v1.0 — Client-Side / Universal Fractal Predictive Cortex Engine
 * ==========================================================================
 *
 * Pure ES2022 Float64Array / Int32Array implementation of the Fractal Predictive
 * Cortex with exact adaptive power-of-2 spatial-grid k-NN acceleration (32x32,
 * 128x128, 256x256). Mathematically bit-aligned with the Python/C engine
 * (fpc/_fast.c, fpc/cortex.py, fpc/phylo.py, fpc/islands.py, fpc/tasks.py).
 *
 * Zero external runtime dependencies. Runs in browser Web Workers and Node.js.
 */

const LOG2E = Math.LOG2E;

// ---------------------------------------------------------------------------
// Statistical utilities: Cody / asymptotic erfc -> Gaussian upper tail normSf
// ---------------------------------------------------------------------------
export function erfc(x) {
  const z = Math.abs(x);
  if (z <= 0.46875) {
    const y = z * z;
    const p =
      ((((1.85781706184603153e-1 * y + 3.1611237438705656) * y +
        1.13864154151050156e2) *
        y +
        3.77485237685302021e2) *
        y +
        3.20937758913846947e3) *
      z;
    const q =
      (((y + 2.36012909523441209e1) * y + 2.44024637934444173e2) * y +
        1.28261652607737228e3) *
        y +
      2.84423683343917062e3;
    const erfVal = p / q;
    return x < 0 ? 1.0 + erfVal : 1.0 - erfVal;
  }
  if (z <= 4.0) {
    const p =
      (((((((2.15311535474403846e-8 * z + 5.64188496988670089e-1) * z +
        8.88314979438837594) *
        z +
        6.61191906371416295e1) *
        z +
        2.98635138197400131e2) *
        z +
        8.8195222124176909e2) *
        z +
        1.71204761263407058e3) *
        z +
        2.05107837782607147e3) *
        z +
      1.23033935479799725e3;
    const q =
      (((((((z + 1.57449261107098347e1) * z + 1.17693950891312499e2) * z +
        5.37181101862009858e2) *
        z +
        1.62138957456669019e3) *
        z +
        3.29079923573345963e3) *
        z +
        4.36261909014324716e3) *
        z +
        3.43936767414372164e3) *
        z +
      1.23033935480374942e3;
    const res = Math.exp(-z * z) * (p / q);
    return x < 0 ? 2.0 - res : res;
  }
  const y = 1.0 / (z * z);
  const p =
    ((((1.63153871373020978e-2 * y + 3.05326634961232344e-1) * y +
      3.60344899949804439e-1) *
      y +
      1.25781726111229246e-1) *
      y +
      1.60837851487422766e-2) *
      y +
    6.58749161529837803e-4;
  const q =
    ((((y + 2.56852019228982242) * y + 1.87295284992346047) * y +
      5.27905102951428412e-1) *
      y +
      6.05183413124413191e-2) *
      y +
    2.33520497626869185e-3;
  const r = (Math.exp(-z * z) / z) * (0.56418958354775629 - (y * p) / q);
  return x < 0 ? 2.0 - r : r;
}

export function normSf(z) {
  if (!Number.isFinite(z)) return NaN;
  return 0.5 * erfc(z * Math.SQRT1_2);
}

export function median(values) {
  const n = values.length;
  if (n === 0) return NaN;
  const copy = Float64Array.from(values);
  copy.sort();
  const mid = n >> 1;
  return n % 2 === 1 ? copy[mid] : 0.5 * (copy[mid - 1] + copy[mid]);
}

export function benjaminiHochberg(pValues, totalTests = null) {
  const k = pValues.length;
  if (k === 0) return new Float64Array(0);
  const m = Math.max(k, totalTests !== null ? Math.floor(totalTests) : k);
  const idx = Array.from({ length: k }, (_, i) => i);
  idx.sort((a, b) => pValues[a] - pValues[b] || a - b);
  const mono = new Float64Array(k);
  let runMin = 1.0;
  for (let r = k - 1; r >= 0; r--) {
    const raw = (pValues[idx[r]] * m) / (r + 1);
    if (raw < runMin) runMin = raw;
    mono[r] = runMin > 1.0 ? 1.0 : runMin;
  }
  const out = new Float64Array(k);
  for (let r = 0; r < k; r++) {
    out[idx[r]] = mono[r];
  }
  return out;
}

// ---------------------------------------------------------------------------
// Alphabet + O(N) 256-byte ASCII lookup + Reverse Complement
// ---------------------------------------------------------------------------
export class Alphabet {
  constructor(name, symbols, wildcard = "N", complements = null) {
    this.name = name;
    this.symbols = symbols.toUpperCase();
    this.wildcard = wildcard.toUpperCase();
    this.size = this.symbols.length;
    this.wildcardId = this.size;
    this.fullSize = this.size + 1;
    this.complements = complements;
    this.isComplementary = complements !== null;

    this._lut = new Uint8Array(256);
    this._lut.fill(this.wildcardId);
    for (let i = 0; i < this.size; i++) {
      const ch = this.symbols[i];
      this._lut[ch.charCodeAt(0)] = i;
      this._lut[ch.toLowerCase().charCodeAt(0)] = i;
    }

    this._compLut = new Int32Array(this.fullSize);
    for (let i = 0; i < this.fullSize; i++) this._compLut[i] = this.wildcardId;
    if (complements) {
      for (let i = 0; i < this.size; i++) {
        const compChar = complements[this.symbols[i]];
        const compIdx = this.symbols.indexOf(compChar);
        this._compLut[i] = compIdx >= 0 ? compIdx : this.wildcardId;
      }
    }
  }

  encode(seq) {
    const n = seq.length;
    const out = new Int32Array(n);
    const lut = this._lut;
    const wild = this.wildcardId;
    for (let i = 0; i < n; i++) {
      const code = seq.charCodeAt(i);
      out[i] = code < 256 ? lut[code] : wild;
    }
    return out;
  }

  decode(ids) {
    const chars = new Array(ids.length);
    for (let i = 0; i < ids.length; i++) {
      const id = ids[i];
      chars[i] = id >= 0 && id < this.size ? this.symbols[id] : this.wildcard;
    }
    return chars.join("");
  }

  reverseComplementIds(ids) {
    if (!this.isComplementary) {
      throw new Error(`Alphabet ${this.name} does not define reverse complements`);
    }
    const n = ids.length;
    const out = new Int32Array(n);
    const clut = this._compLut;
    const wild = this.wildcardId;
    for (let i = 0; i < n; i++) {
      const id = ids[n - 1 - i];
      out[i] = id >= 0 && id < this.fullSize ? clut[id] : wild;
    }
    return out;
  }

  reverseComplement(seq) {
    return this.decode(this.reverseComplementIds(this.encode(seq)));
  }
}

export const DNA = new Alphabet("DNA", "ACGT", "N", { A: "T", T: "A", C: "G", G: "C" });
export const RNA = new Alphabet("RNA", "ACGU", "N", { A: "U", U: "A", C: "G", G: "C" });
export const PROTEIN = new Alphabet("PROTEIN", "ACDEFGHIKLMNPQRSTVWY", "X", null);

export function getAlphabet(name) {
  const key = String(name || "DNA").trim().toUpperCase();
  if (key === "DNA" || key === "NT" || key === "NUCLEOTIDE") return DNA;
  if (key === "RNA") return RNA;
  if (key === "PROTEIN" || key === "AA" || key === "AMINO") return PROTEIN;
  throw new Error(`Unknown alphabet: ${name}`);
}

// ---------------------------------------------------------------------------
// Chaos Game Embedding
// ---------------------------------------------------------------------------
export class ChaosGameEmbedding {
  constructor(alphabet, contraction = null) {
    this.alphabet = alphabet;
    const a = alphabet.size;
    if (contraction === null || contraction === undefined) {
      this.r = a <= 4 ? 0.5 : Math.min(0.5, Math.sin(Math.PI / a));
    } else {
      this.r = Number(contraction);
    }
    this.centreX = 0.5;
    this.centreY = 0.5;
    this.allVertices = this._buildVertices(a);
  }

  _buildVertices(a) {
    const verts = new Float64Array((a + 1) * 2);
    if (a === 4) {
      // Exact match to fpc/cgr.py: [[0,0], [0,1], [1,1], [1,0]]
      verts[0] = 0.0; verts[1] = 0.0;
      verts[2] = 0.0; verts[3] = 1.0;
      verts[4] = 1.0; verts[5] = 1.0;
      verts[6] = 1.0; verts[7] = 0.0;
    } else {
      for (let k = 0; k < a; k++) {
        const theta = Math.PI / 2.0 - (2.0 * Math.PI * k) / a;
        verts[2 * k] = 0.5 + 0.5 * Math.cos(theta);
        verts[2 * k + 1] = 0.5 + 0.5 * Math.sin(theta);
      }
    }
    verts[2 * a] = 0.5;
    verts[2 * a + 1] = 0.5;
    return verts;
  }

  embed(ids) {
    const T = ids.length;
    const P = new Float64Array(T * 2);
    if (T === 0) return P;
    const verts = this.allVertices;
    const nVerts = this.alphabet.fullSize;
    const wildIdx = nVerts - 1;
    const r = this.r;
    const omr = 1.0 - r;
    let px = this.centreX;
    let py = this.centreY;
    for (let t = 0; t < T; t++) {
      let s = ids[t];
      if (s < 0 || s >= nVerts) s = wildIdx;
      px = omr * px + r * verts[2 * s];
      py = omr * py + r * verts[2 * s + 1];
      P[2 * t] = px;
      P[2 * t + 1] = py;
    }
    return P;
  }

  contextPoints(ids) {
    const T = ids.length;
    const C = new Float64Array(T * 2);
    if (T === 0) return C;
    const P = this.embed(ids);
    C[0] = this.centreX;
    C[1] = this.centreY;
    if (T > 1) {
      C.set(P.subarray(0, (T - 1) * 2), 2);
    }
    return C;
  }
}

// ---------------------------------------------------------------------------
// Adaptive Power-of-2 Spatial Grid Index (identical to fpc/_fast.c)
// ---------------------------------------------------------------------------
function chooseGridN(targetPrototypes) {
  if (targetPrototypes <= 4096) return 32;
  if (targetPrototypes <= 32768) return 128;
  return 256;
}

function cellCoord(v, gridN) {
  let c = Math.floor(v * gridN);
  if (c < 0) c = 0;
  else if (c >= gridN) c = gridN - 1;
  return c;
}

function minCellDist2(cx, cy, gx, gy, invGridN) {
  const x0 = gx * invGridN;
  const x1 = (gx + 1) * invGridN;
  const y0 = gy * invGridN;
  const y1 = (gy + 1) * invGridN;
  let dx = 0.0;
  let dy = 0.0;
  if (cx < x0) dx = x0 - cx;
  else if (cx > x1) dx = cx - x1;
  if (cy < y0) dy = y0 - cy;
  else if (cy > y1) dy = cy - y1;
  return dx * dx + dy * dy;
}

class SpatialGrid {
  constructor(capacity, targetPrototypes = capacity) {
    this.gridN = chooseGridN(targetPrototypes);
    this.invGridN = 1.0 / this.gridN;
    this.head = new Int32Array(this.gridN * this.gridN);
    this.next = new Int32Array(capacity);
    this.prev = new Int32Array(capacity);
    this.cell = new Int32Array(capacity);
    this.clear();
  }

  ensureCapacity(need) {
    if (need <= this.next.length) return;
    const cap = Math.max(need, this.next.length * 2, 64);
    const next = new Int32Array(cap);
    const prev = new Int32Array(cap);
    const cell = new Int32Array(cap);
    next.set(this.next);
    prev.set(this.prev);
    cell.set(this.cell);
    this.next = next;
    this.prev = prev;
    this.cell = cell;
  }

  clear() {
    this.head.fill(-1);
  }

  insert(i, x, y) {
    const gn = this.gridN;
    const c = cellCoord(y, gn) * gn + cellCoord(x, gn);
    const h = this.head[c];
    this.prev[i] = -1;
    this.next[i] = h;
    if (h >= 0) this.prev[h] = i;
    this.head[c] = i;
    this.cell[i] = c;
  }

  move(i, x, y) {
    const gn = this.gridN;
    const newC = cellCoord(y, gn) * gn + cellCoord(x, gn);
    const oldC = this.cell[i];
    if (newC === oldC) return;
    const p = this.prev[i];
    const n = this.next[i];
    if (p >= 0) this.next[p] = n;
    else this.head[oldC] = n;
    if (n >= 0) this.prev[n] = p;

    const h = this.head[newC];
    this.prev[i] = -1;
    this.next[i] = h;
    if (h >= 0) this.prev[h] = i;
    this.head[newC] = i;
    this.cell[i] = newC;
  }

  rebuild(X, m) {
    this.clear();
    for (let i = 0; i < m; i++) {
      this.insert(i, X[2 * i], X[2 * i + 1]);
    }
  }

  knn(X, m, cx, cy, kReq, outIdx, outD2) {
    const k = Math.min(kReq, m);
    if (k <= 0) return 0;

    let count = 0;
    let worstPos = 0;
    let worstD2 = Infinity;

    const pushCandidate = (idx, d2) => {
      if (count < k) {
        outIdx[count] = idx;
        outD2[count] = d2;
        count++;
        if (count === k) {
          worstPos = 0;
          worstD2 = outD2[0];
          for (let j = 1; j < k; j++) {
            if (outD2[j] > worstD2 || (outD2[j] === worstD2 && outIdx[j] > outIdx[worstPos])) {
              worstD2 = outD2[j];
              worstPos = j;
            }
          }
        }
      } else if (d2 < worstD2 || (d2 === worstD2 && idx < outIdx[worstPos])) {
        outIdx[worstPos] = idx;
        outD2[worstPos] = d2;
        worstPos = 0;
        worstD2 = outD2[0];
        for (let j = 1; j < k; j++) {
          if (outD2[j] > worstD2 || (outD2[j] === worstD2 && outIdx[j] > outIdx[worstPos])) {
            worstD2 = outD2[j];
            worstPos = j;
          }
        }
      }
    };

    if (m <= 48) {
      for (let i = 0; i < m; i++) {
        const dx = X[2 * i] - cx;
        const dy = X[2 * i + 1] - cy;
        pushCandidate(i, dx * dx + dy * dy);
      }
      return count;
    }

    const gn = this.gridN;
    const invGn = this.invGridN;
    const gx0 = cellCoord(cx, gn);
    const gy0 = cellCoord(cy, gn);
    const maxRing = Math.max(gx0, gn - 1 - gx0, gy0, gn - 1 - gy0);

    const scanCell = (gx, gy) => {
      if (count === k && minCellDist2(cx, cy, gx, gy, invGn) > worstD2) return;
      let idx = this.head[gy * gn + gx];
      while (idx >= 0) {
        const dx = X[2 * idx] - cx;
        const dy = X[2 * idx + 1] - cy;
        pushCandidate(idx, dx * dx + dy * dy);
        idx = this.next[idx];
      }
    };

    for (let ring = 0; ring <= maxRing; ring++) {
      if (ring > 0 && count === k) {
        const xMin = gx0 * invGn;
        const xMax = (gx0 + 1) * invGn;
        const yMin = gy0 * invGn;
        const yMax = (gy0 + 1) * invGn;
        const span = (ring - 1) * invGn;
        let minRingD = Infinity;
        if (gx0 - ring >= 0) minRingD = Math.min(minRingD, cx - xMin + span);
        if (gx0 + ring < gn) minRingD = Math.min(minRingD, xMax - cx + span);
        if (gy0 - ring >= 0) minRingD = Math.min(minRingD, cy - yMin + span);
        if (gy0 + ring < gn) minRingD = Math.min(minRingD, yMax - cy + span);
        if (minRingD * minRingD > worstD2) break;
      }

      if (ring === 0) {
        scanCell(gx0, gy0);
      } else {
        const xLo = Math.max(0, gx0 - ring);
        const xHi = Math.min(gn - 1, gx0 + ring);

        if (gy0 - ring >= 0) {
          for (let gx = xLo; gx <= xHi; gx++) scanCell(gx, gy0 - ring);
        }
        if (gy0 + ring < gn) {
          for (let gx = xLo; gx <= xHi; gx++) scanCell(gx, gy0 + ring);
        }
        for (let gy = Math.max(0, gy0 - ring + 1); gy <= Math.min(gn - 1, gy0 + ring - 1); gy++) {
          if (gx0 - ring >= 0) scanCell(gx0 - ring, gy);
          if (gx0 + ring < gn) scanCell(gx0 + ring, gy);
        }
      }
    }
    return count;
  }
}

// ---------------------------------------------------------------------------
// SurpriseTrack
// ---------------------------------------------------------------------------
export class SurpriseTrack {
  constructor(bits) {
    this.bits = bits instanceof Float64Array ? bits : Float64Array.from(bits);
    let nValid = 0;
    let total = 0.0;
    for (let i = 0; i < this.bits.length; i++) {
      const v = this.bits[i];
      if (!Number.isNaN(v)) {
        nValid++;
        total += v;
      }
    }
    this.nValid = nValid;
    this.totalBits = total;
    this.meanBits = nValid > 0 ? total / nValid : NaN;
    this.perplexity = Number.isFinite(this.meanBits) ? Math.pow(2.0, this.meanBits) : NaN;
  }

  smoothed(window = 25) {
    const w = Math.max(1, Math.floor(window));
    const n = this.bits.length;
    const out = new Float64Array(n);
    const csB = new Float64Array(n + 1);
    const csV = new Int32Array(n + 1);
    for (let i = 0; i < n; i++) {
      const b = this.bits[i];
      if (!Number.isNaN(b)) {
        csB[i + 1] = csB[i] + b;
        csV[i + 1] = csV[i] + 1;
      } else {
        csB[i + 1] = csB[i];
        csV[i + 1] = csV[i];
      }
    }
    // Match np.convolve(..., np.ones(w), mode="same")
    const leftHalf = Math.floor((w - 1) / 2);
    const rightHalf = w - 1 - leftHalf;
    // Wait: in np.convolve(a, ones(w), mode="same") for even w:
    // start index is (w - 1) // 2 in full convolution; full[k] sums a[j] for k - (w-1) <= j <= k.
    // At output index i, full index is k = i + Math.floor((w - 1) / 2) when w is odd;
    // Actually for np.convolve(n, w, 'same'), start = Math.floor((Math.min(n,w) - 1) / 2)?
    // Let's check exact formula: full[k] = sum_{m=0..w-1} a[k - m].
    // mode='same' slices full[start : start + n] where start = Math.floor((w - 1) / 2) for odd w, and Math.floor(w / 2) - 1?
    // Wait: in NumPy, start = (min_len - 1) // 2? No, in NumPy _centered(arr, newshape):
    // currshape = n + w - 1; startind = (currshape - n) // 2 = (w - 1) // 2!
    // Yes! startind is ALWAYS Math.floor((w - 1) / 2)!
    // So output[i] = full[i + startind] = sum_{j = i + startind - (w - 1) .. i + startind} a[j]!
    const startInd = Math.floor((w - 1) / 2);
    for (let i = 0; i < n; i++) {
      const lo = Math.max(0, i + startInd - (w - 1));
      const hi = Math.min(n, i + startInd + 1);
      const cnt = csV[hi] - csV[lo];
      out[i] = cnt > 0 ? (csB[hi] - csB[lo]) / cnt : NaN;
    }
    return out;
  }

  anomalies(window = 25, zThreshold = 4.0, alpha = 0.05) {
    const s = this.smoothed(window);
    const finiteVals = [];
    for (let i = 0; i < s.length; i++) {
      if (Number.isFinite(s[i])) finiteVals.push(s[i]);
    }
    if (finiteVals.length < 3) return [];
    const med = median(finiteVals);
    const absDev = new Float64Array(finiteVals.length);
    for (let i = 0; i < finiteVals.length; i++) absDev[i] = Math.abs(finiteVals[i] - med);
    const mad = median(absDev);
    let scale = 1.4826 * mad;
    if (!(mad > 0)) {
      let mean = 0;
      for (const v of finiteVals) mean += v;
      mean /= finiteVals.length;
      let v2 = 0;
      for (const v of finiteVals) v2 += (v - mean) * (v - mean);
      scale = Math.sqrt(v2 / finiteVals.length) || 1.0;
    }

    const n = s.length;
    const z = new Float64Array(n);
    const above = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      if (Number.isFinite(s[i])) {
        const zi = (s[i] - med) / scale;
        z[i] = zi;
        if (zi >= zThreshold) above[i] = 1;
      } else {
        z[i] = NaN;
      }
    }

    const nTests = Math.max(1, Math.floor(finiteVals.length / Math.max(1, window)));
    const pCut = alpha / nTests;
    const list = [];
    let i = 0;
    while (i < n) {
      if (!above[i]) {
        i++;
        continue;
      }
      let j = i;
      while (j + 1 < n && above[j + 1]) j++;
      let peakPos = i;
      let peakZ = -Infinity;
      let sumB = 0;
      let cntB = 0;
      for (let t = i; t <= j; t++) {
        if (z[t] > peakZ) {
          peakZ = z[t];
          peakPos = t;
        }
        if (!Number.isNaN(this.bits[t])) {
          sumB += this.bits[t];
          cntB++;
        }
      }
      const pVal = normSf(peakZ);
      list.push({
        start: i,
        end: j,
        peakPos,
        peakZ,
        meanBits: cntB > 0 ? sumB / cntB : NaN,
        pValue: pVal,
        significant: pVal <= pCut,
        qValue: 1.0,
        fdrSignificant: false,
      });
      i = j + 1;
    }

    if (list.length > 0) {
      const qVals = benjaminiHochberg(
        list.map((a) => a.pValue),
        Math.max(nTests, list.length)
      );
      for (let k = 0; k < list.length; k++) {
        list[k].qValue = qVals[k];
        list[k].fdrSignificant = qVals[k] <= alpha;
      }
    }
    return list;
  }
}

// ---------------------------------------------------------------------------
// FractalPredictiveCortex
// ---------------------------------------------------------------------------
export class FractalPredictiveCortex {
  constructor(alphabet = DNA, options = {}) {
    this.alphabet = alphabet;
    this.a = alphabet.size;
    this.order = options.order !== undefined ? Math.floor(options.order) : 8;
    this.embedding = new ChaosGameEmbedding(alphabet, options.contraction ?? null);
    this.r = this.embedding.r;

    this.distThreshold = Math.pow(this.r, this.order);
    this.sigma = Math.max(this.distThreshold, 1e-9);
    this.nNeighbors = options.nNeighbors !== undefined ? Math.floor(options.nNeighbors) : 8;
    this.alpha = options.alpha !== undefined ? Number(options.alpha) : 0.5;
    this.backgroundWeight =
      options.backgroundWeight !== undefined ? Number(options.backgroundWeight) : 0.02;
    this.learnRate = options.learnRate !== undefined ? Number(options.learnRate) : 0.02;
    this.surpriseGrowth =
      options.surpriseGrowth !== undefined ? Boolean(options.surpriseGrowth) : true;
    this.maxPrototypes =
      options.maxPrototypes !== undefined ? Math.floor(options.maxPrototypes) : 6000;
    this.pruneInterval =
      options.pruneInterval !== undefined ? Math.floor(options.pruneInterval) : 4000;
    this.usageDecay = options.usageDecay !== undefined ? Number(options.usageDecay) : 0.9997;

    this._cap = Math.max(64, Math.min(this.maxPrototypes + 1, 2048));
    this._m = 0;
    this.X = new Float64Array(this._cap * 2);
    this.N = new Float64Array(this._cap * this.a);
    this.usage = new Float64Array(this._cap);
    this.globalCounts = new Float64Array(this.a);
    this.globalCounts.fill(this.alpha);

    this._grid = new SpatialGrid(this._cap, this.maxPrototypes);
    this._steps = 0;
    this._surpN = 0;
    this._surpMean = 0.0;
    this._surpM2 = 0.0;
  }

  get nPrototypes() {
    return this._m;
  }

  _ensureCapacity(need) {
    if (need <= this._cap) return;
    const newCap = Math.max(64, this._cap * 2, need);
    const X = new Float64Array(newCap * 2);
    const N = new Float64Array(newCap * this.a);
    const U = new Float64Array(newCap);
    X.set(this.X.subarray(0, this._m * 2));
    N.set(this.N.subarray(0, this._m * this.a));
    U.set(this.usage.subarray(0, this._m));
    this.X = X;
    this.N = N;
    this.usage = U;
    this._cap = newCap;
    this._grid.ensureCapacity(newCap);
  }

  _prune() {
    if (this._m <= this.maxPrototypes) return;
    const m = this._m;
    const order = Array.from({ length: m }, (_, idx) => idx);
    // Exact match to np.argsort(usage[:m], kind="mergesort")[::-1][:max_prototypes]
    order.sort((ia, ib) => {
      const diff = this.usage[ib] - this.usage[ia];
      if (diff !== 0) return diff;
      return ib - ia;
    });
    const keep = order.slice(0, this.maxPrototypes);
    keep.sort((ia, ib) => ia - ib);
    const k = keep.length;
    const a = this.a;
    for (let dst = 0; dst < k; dst++) {
      const src = keep[dst];
      if (dst !== src) {
        this.X[2 * dst] = this.X[2 * src];
        this.X[2 * dst + 1] = this.X[2 * src + 1];
        for (let s = 0; s < a; s++) {
          this.N[dst * a + s] = this.N[src * a + s];
        }
        this.usage[dst] = this.usage[src];
      }
    }
    this._m = k;
    this._grid.rebuild(this.X, k);
  }

  predictDist(cx, cy, nnIdxBuf = null, nnD2Buf = null, outP = null) {
    const a = this.a;
    const p = outP || new Float64Array(a);
    let gcSum = 0.0;
    for (let s = 0; s < a; s++) gcSum += this.globalCounts[s];
    const invGc = 1.0 / gcSum;

    if (this._m === 0) {
      for (let s = 0; s < a; s++) p[s] = this.globalCounts[s] * invGc;
      return { p, nearest: -1, nearestDist: Infinity };
    }

    const idxBuf = nnIdxBuf || new Int32Array(Math.max(1, this.nNeighbors));
    const d2Buf = nnD2Buf || new Float64Array(Math.max(1, this.nNeighbors));
    const k = this._grid.knn(this.X, this._m, cx, cy, this.nNeighbors, idxBuf, d2Buf);

    let nearest = idxBuf[0];
    let minD2 = d2Buf[0];
    for (let j = 1; j < k; j++) {
      if (d2Buf[j] < minD2 || (d2Buf[j] === minD2 && idxBuf[j] < nearest)) {
        minD2 = d2Buf[j];
        nearest = idxBuf[j];
      }
    }
    const nearestDist = Math.sqrt(minD2);
    const inv2Sig2 = 1.0 / (2.0 * this.sigma * this.sigma);

    let wSum = 0.0;
    p.fill(0.0);
    for (let j = 0; j < k; j++) {
      const w = Math.exp(-d2Buf[j] * inv2Sig2);
      wSum += w;
      const base = idxBuf[j] * a;
      for (let s = 0; s < a; s++) {
        p[s] += w * this.N[base + s];
      }
    }

    if (wSum <= 0.0) {
      for (let s = 0; s < a; s++) p[s] = this.globalCounts[s] * invGc;
      return { p, nearest, nearestDist };
    }

    let denom = 0.0;
    for (let s = 0; s < a; s++) denom += p[s];
    const invProtoDen = 1.0 / (denom + a * this.alpha);
    const lam = this.backgroundWeight;
    const oml = 1.0 - lam;
    let pSum = 0.0;
    for (let s = 0; s < a; s++) {
      const pProto = (p[s] + this.alpha) * invProtoDen;
      const bg = this.globalCounts[s] * invGc;
      const val = oml * pProto + lam * bg;
      p[s] = val;
      pSum += val;
    }
    const invPSum = 1.0 / pSum;
    for (let s = 0; s < a; s++) p[s] *= invPSum;
    return { p, nearest, nearestDist };
  }

  predictNext(prefixIds) {
    if (prefixIds.length === 0) {
      return this.predictDist(this.embedding.centreX, this.embedding.centreY).p;
    }
    const P = this.embedding.embed(prefixIds);
    const last = (prefixIds.length - 1) * 2;
    return this.predictDist(P[last], P[last + 1]).p;
  }

  fit(ids, { epochs = 1, update = true, doubleStrand = false, onProgress = null } = {}) {
    const rcIds =
      doubleStrand && update && this.alphabet.isComplementary && ids.length > 0
        ? this.alphabet.reverseComplementIds(ids)
        : null;
    let track = null;
    const epCount = Math.max(1, Math.floor(epochs));
    for (let ep = 0; ep < epCount; ep++) {
      track = this._stream(ids, update, onProgress);
      if (rcIds !== null) {
        this._stream(rcIds, true, null);
      }
    }
    return track;
  }

  score(ids, onProgress = null) {
    const T = ids.length;
    if (T === 0) return new SurpriseTrack(new Float64Array(0));
    if (this._m === 0) return this._stream(ids, false, onProgress);

    const C = this.embedding.contextPoints(ids);
    const bits = new Float64Array(T);
    bits.fill(NaN);
    const a = this.a;
    const idxBuf = new Int32Array(Math.max(1, this.nNeighbors));
    const d2Buf = new Float64Array(Math.max(1, this.nNeighbors));
    const pBuf = new Float64Array(a);

    for (let t = 0; t < T; t++) {
      const s = ids[t];
      if (s >= 0 && s < a) {
        this.predictDist(C[2 * t], C[2 * t + 1], idxBuf, d2Buf, pBuf);
        const ps = Math.max(1e-18, Math.min(1.0, pBuf[s]));
        bits[t] = -Math.log(ps) * LOG2E;
      }
      if (onProgress && (t & 32767) === 32767) {
        onProgress(t + 1, T);
      }
    }
    return new SurpriseTrack(bits);
  }

  _stream(ids, update, onProgress = null) {
    const T = ids.length;
    const bits = new Float64Array(T);
    bits.fill(NaN);
    if (T === 0) return new SurpriseTrack(bits);

    this._ensureCapacity(Math.max(64, this.maxPrototypes + 1, this._m + 1));
    const C = this.embedding.contextPoints(ids);
    const a = this.a;
    const idxBuf = new Int32Array(Math.max(1, this.nNeighbors));
    const d2Buf = new Float64Array(Math.max(1, this.nNeighbors));
    const pBuf = new Float64Array(a);
    const decayBlock =
      this.pruneInterval > 0 ? Math.pow(this.usageDecay, this.pruneInterval) : 1.0;

    for (let t = 0; t < T; t++) {
      const s = ids[t];
      const cx = C[2 * t];
      const cy = C[2 * t + 1];
      const { p, nearest, nearestDist } = this.predictDist(cx, cy, idxBuf, d2Buf, pBuf);

      if (s >= 0 && s < a) {
        const ps = p[s];
        const b = ps > 0.0 ? -Math.log(ps) * LOG2E : 60.0;
        bits[t] = b;

        if (update) {
          this._surpN++;
          const d = b - this._surpMean;
          this._surpMean += d / this._surpN;
          this._surpM2 += d * (b - this._surpMean);
          this.globalCounts[s] += 1.0;

          const empty = this._m === 0;
          const far = nearestDist > this.distThreshold;
          let highSurp = false;
          if (this.surpriseGrowth && this._surpN >= 50) {
            const variance = this._surpM2 / Math.max(1, this._surpN - 1);
            highSurp = b > this._surpMean + 2.0 * Math.sqrt(variance);
          }
          const shocked = highSurp && nearestDist > 0.5 * this.distThreshold;
          const canGrow = this._m < this.maxPrototypes;

          if (empty || ((far || shocked) && canGrow)) {
            this._ensureCapacity(this._m + 1);
            const idx = this._m;
            this.X[2 * idx] = cx;
            this.X[2 * idx + 1] = cy;
            const base = idx * a;
            for (let j = 0; j < a; j++) this.N[base + j] = 0.0;
            this.N[base + s] = 1.0;
            this.usage[idx] = 1.0;
            this._grid.insert(idx, cx, cy);
            this._m++;
          } else {
            this.N[nearest * a + s] += 1.0;
            this.usage[nearest] += 1.0;
            if (this.learnRate > 0.0) {
              const nx = this.X[2 * nearest] + this.learnRate * (cx - this.X[2 * nearest]);
              const ny =
                this.X[2 * nearest + 1] + this.learnRate * (cy - this.X[2 * nearest + 1]);
              this.X[2 * nearest] = nx;
              this.X[2 * nearest + 1] = ny;
              this._grid.move(nearest, nx, ny);
            }
          }
        }
      }

      if (update) {
        this._steps++;
        if (this.pruneInterval > 0 && this._steps % this.pruneInterval === 0) {
          for (let i = 0; i < this._m; i++) this.usage[i] *= decayBlock;
          this._prune();
        }
      }

      if (onProgress && (t & 32767) === 32767) {
        onProgress(t + 1, T);
      }
    }
    return new SurpriseTrack(bits);
  }

  crossEntropy(ids, strandAware = false) {
    const fwd = this.score(ids).meanBits;
    if (strandAware && this.alphabet.isComplementary && ids.length > 0) {
      const rev = this.score(this.alphabet.reverseComplementIds(ids)).meanBits;
      if (Number.isFinite(rev) && (!Number.isFinite(fwd) || rev < fwd)) {
        return rev;
      }
    }
    return fwd;
  }

  crossEntropyWithStrand(ids) {
    const fwd = this.score(ids).meanBits;
    if (!this.alphabet.isComplementary || ids.length === 0) {
      return { bits: fwd, strand: "+" };
    }
    const rev = this.score(this.alphabet.reverseComplementIds(ids)).meanBits;
    if (Number.isFinite(rev) && (!Number.isFinite(fwd) || rev < fwd)) {
      return { bits: rev, strand: "-" };
    }
    return { bits: fwd, strand: "+" };
  }

  generate(prefixIds, length, { sample = false, temperature = 1.0 } = {}) {
    const out = Array.from(prefixIds);
    for (let step = 0; step < length; step++) {
      const p = Float64Array.from(this.predictNext(Int32Array.from(out)));
      let nxt = 0;
      if (sample) {
        if (temperature !== 1.0) {
          const temp = Math.max(1e-6, temperature);
          let maxLp = -Infinity;
          for (let s = 0; s < this.a; s++) {
            p[s] = Math.log(Math.max(1e-12, p[s])) / temp;
            if (p[s] > maxLp) maxLp = p[s];
          }
          let sum = 0;
          for (let s = 0; s < this.a; s++) {
            p[s] = Math.exp(p[s] - maxLp);
            sum += p[s];
          }
          for (let s = 0; s < this.a; s++) p[s] /= sum;
        }
        const u = Math.random();
        let acc = 0;
        nxt = this.a - 1;
        for (let s = 0; s < this.a; s++) {
          acc += p[s];
          if (u <= acc) {
            nxt = s;
            break;
          }
        }
      } else {
        let best = -Infinity;
        for (let s = 0; s < this.a; s++) {
          if (p[s] > best) {
            best = p[s];
            nxt = s;
          }
        }
      }
      out.push(nxt);
    }
    return Int32Array.from(out);
  }

  prototypeFrame() {
    const m = this._m;
    const x = new Float64Array(m);
    const y = new Float64Array(m);
    const usage = new Float64Array(m);
    const dominant = new Int32Array(m);
    const entropy = new Float64Array(m);
    const a = this.a;
    for (let i = 0; i < m; i++) {
      x[i] = this.X[2 * i];
      y[i] = this.X[2 * i + 1];
      usage[i] = this.usage[i];
      let sum = 0;
      let bestS = 0;
      let bestC = -Infinity;
      for (let s = 0; s < a; s++) {
        const c = this.N[i * a + s] + this.alpha;
        sum += c;
        if (c > bestC) {
          bestC = c;
          bestS = s;
        }
      }
      dominant[i] = bestS;
      let ent = 0;
      for (let s = 0; s < a; s++) {
        const ps = (this.N[i * a + s] + this.alpha) / sum;
        if (ps > 0) ent -= ps * Math.log(ps) * LOG2E;
      }
      entropy[i] = ent;
    }
    return { x, y, usage, dominant, entropy };
  }
}

// ---------------------------------------------------------------------------
// FASTA & FASTQ Parser with Phred Quality Masking
// ---------------------------------------------------------------------------
export function parseSequences(text, { minPhred = 0, wildcard = "N", phredOffset = 33 } = {}) {
  const raw = String(text || "").trim();
  if (!raw) throw new Error("Sequence input is empty");

  const lines = raw.split(/\r?\n/);
  let firstIdx = 0;
  while (firstIdx < lines.length && !lines[firstIdx].trim()) firstIdx++;
  if (firstIdx >= lines.length) throw new Error("Sequence input is empty");

  const lead = lines[firstIdx].trim()[0];
  if (lead === ">") {
    const records = [];
    let header = null;
    let chunks = [];
    for (let i = firstIdx; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line) continue;
      if (line.startsWith(">")) {
        if (header !== null) {
          const id = header.split(/\s+/)[0] || header;
          records.push({ id, description: header, seq: chunks.join("").toUpperCase(), qual: null });
        }
        header = line.slice(1).trim();
        if (!header) throw new Error("Malformed FASTA: empty '>' header");
        chunks = [];
      } else {
        chunks.push(line);
      }
    }
    if (header !== null) {
      const id = header.split(/\s+/)[0] || header;
      records.push({ id, description: header, seq: chunks.join("").toUpperCase(), qual: null });
    }
    if (records.length === 0) throw new Error("No FASTA records found");
    return records;
  }

  if (lead === "@") {
    const records = [];
    let idx = firstIdx;
    while (idx < lines.length) {
      while (idx < lines.length && !lines[idx].trim()) idx++;
      if (idx >= lines.length) break;
      const hLine = lines[idx++].trim();
      if (!hLine.startsWith("@")) {
        throw new Error(`Malformed FASTQ header: ${hLine.slice(0, 32)}`);
      }
      const desc = hLine.slice(1).trim();
      const id = desc.split(/\s+/)[0] || "read";

      const seqChunks = [];
      let sawPlus = false;
      while (idx < lines.length) {
        const s = lines[idx++].trim();
        if (!s) continue;
        if (s.startsWith("+")) {
          sawPlus = true;
          break;
        }
        seqChunks.push(s);
      }
      if (!sawPlus) throw new Error(`Malformed FASTQ record ${id}: missing '+' separator`);
      let seq = seqChunks.join("").toUpperCase();

      const qualChunks = [];
      let qLen = 0;
      while (qLen < seq.length && idx < lines.length) {
        const q = lines[idx++].trim();
        if (!q) continue;
        qualChunks.push(q);
        qLen += q.length;
      }
      const qual = qualChunks.join("");
      if (qual.length !== seq.length) {
        throw new Error(
          `Malformed FASTQ record ${id}: quality length (${qual.length}) != sequence length (${seq.length})`
        );
      }
      if (minPhred > 0) {
        const cutoff = phredOffset + minPhred;
        const chars = seq.split("");
        const wChar = wildcard[0] || "N";
        for (let k = 0; k < chars.length; k++) {
          if (qual.charCodeAt(k) < cutoff) chars[k] = wChar;
        }
        seq = chars.join("");
      }
      records.push({ id, description: desc, seq, qual });
    }
    if (records.length === 0) throw new Error("No FASTQ records found");
    return records;
  }

  // Raw pasted sequence fallback (only valid IUPAC letters / whitespace)
  const cleaned = lines
    .map((l) => l.trim())
    .join("")
    .replace(/\s+/g, "")
    .toUpperCase();
  if (!/^[A-Z*.-]+$/.test(cleaned)) {
    throw new Error(
      "Unrecognised sequence format: provide FASTA ('>'), FASTQ ('@'), or raw IUPAC letters"
    );
  }
  return [{ id: "query_1", description: "query_1", seq: cleaned, qual: null }];
}

// ---------------------------------------------------------------------------
// Genomic Island Scanner + BED / BedGraph / TSV Exporters
// ---------------------------------------------------------------------------
export function defaultMaxPrototypes(alphabet, order) {
  const approx = Math.pow(alphabet.size, Math.min(Math.floor(order), 9));
  return Math.max(8192, Math.min(65536, Math.floor(1.5 * approx)));
}

function windowShannonH0(ids, alphabet, starts, ends) {
  const n = ids.length;
  const wCount = starts.length;
  const h0 = new Float64Array(wCount);
  if (wCount === 0 || n === 0) return h0;
  const a = alphabet.size;
  const cs = new Int32Array(n + 1);
  const counts = new Float64Array(wCount * a);
  const totals = new Float64Array(wCount);

  for (let sym = 0; sym < a; sym++) {
    cs[0] = 0;
    for (let i = 0; i < n; i++) {
      cs[i + 1] = cs[i] + (ids[i] === sym ? 1 : 0);
    }
    for (let w = 0; w < wCount; w++) {
      const c = cs[ends[w]] - cs[starts[w]];
      counts[w * a + sym] = c;
      totals[w] += c;
    }
  }

  for (let w = 0; w < wCount; w++) {
    const tot = totals[w];
    if (tot <= 0) continue;
    let ent = 0.0;
    for (let sym = 0; sym < a; sym++) {
      const c = counts[w * a + sym];
      if (c > 0) {
        const p = c / tot;
        ent -= p * Math.log(p) * LOG2E;
      }
    }
    h0[w] = ent;
  }
  return h0;
}

function callIslandsFromBits(
  records,
  encodedList,
  bitsList,
  alphabet,
  {
    window = 1000,
    step = 250,
    zThreshold = 3.5,
    alpha = 0.05,
    mergeGap = 2000,
    minLength = null,
    normalizeEntropy = true,
  } = {}
) {
  const win = Math.max(10, Math.floor(window));
  const stp = Math.max(1, Math.floor(step));
  const effMinLen = minLength !== null ? Math.floor(minLength) : win;
  const isNuc = alphabet.size === 4 && alphabet.symbols.includes("G") && alphabet.symbols.includes("C");
  const gId = isNuc ? alphabet.symbols.indexOf("G") : -1;
  const cId = isNuc ? alphabet.symbols.indexOf("C") : -1;

  const rawWindows = [];
  const allValidScores = [];
  const allValidWinBits = [];
  let totalValidBases = 0;
  const csums = [];

  for (let r = 0; r < records.length; r++) {
    const rec = records[r];
    const ids = encodedList[r];
    const bits = bitsList[r];
    const n = bits.length;

    const csB = new Float64Array(n + 1);
    const csV = new Int32Array(n + 1);
    const csGC = isNuc ? new Int32Array(n + 1) : null;
    const csCan = isNuc ? new Int32Array(n + 1) : null;

    for (let i = 0; i < n; i++) {
      const b = bits[i];
      if (!Number.isNaN(b)) {
        csB[i + 1] = csB[i] + b;
        csV[i + 1] = csV[i] + 1;
        totalValidBases++;
      } else {
        csB[i + 1] = csB[i];
        csV[i + 1] = csV[i];
      }
      if (isNuc) {
        const s = ids[i];
        csGC[i + 1] = csGC[i] + (s === gId || s === cId ? 1 : 0);
        csCan[i + 1] = csCan[i] + (s >= 0 && s < 4 ? 1 : 0);
      }
    }
    csums.push({ csB, csV, csGC, csCan });

    const startsArr = [];
    const endsArr = [];
    if (n > 0 && n <= win) {
      startsArr.push(0);
      endsArr.push(n);
    } else if (n > win) {
      for (let s = 0; s <= n - win; s += stp) {
        startsArr.push(s);
        endsArr.push(s + win);
      }
      if (startsArr[startsArr.length - 1] + win < n) {
        startsArr.push(n - win);
        endsArr.push(n);
      }
    }

    const starts = Int32Array.from(startsArr);
    const ends = Int32Array.from(endsArr);
    const wBits = new Float64Array(starts.length);
    const h0 = normalizeEntropy ? windowShannonH0(ids, alphabet, starts, ends) : null;
    const wScore = new Float64Array(starts.length);

    for (let w = 0; w < starts.length; w++) {
      const vc = csV[ends[w]] - csV[starts[w]];
      const minV = Math.max(1, (ends[w] - starts[w]) >> 1);
      if (vc >= minV) {
        const mb = (csB[ends[w]] - csB[starts[w]]) / vc;
        wBits[w] = mb;
        const sc = normalizeEntropy ? mb - h0[w] : mb;
        wScore[w] = sc;
        allValidWinBits.push(mb);
        allValidScores.push(sc);
      } else {
        wBits[w] = NaN;
        wScore[w] = NaN;
      }
    }
    rawWindows.push({ chrom: rec.id, starts, ends, wBits, wScore });
  }

  if (allValidScores.length === 0) {
    return {
      islands: [],
      windowTracks: rawWindows.map((rw) => ({
        chrom: rw.chrom,
        starts: rw.starts,
        ends: rw.ends,
        meanBits: rw.wBits,
        zScores: new Float64Array(rw.wBits.length).fill(NaN),
      })),
    };
  }

  const medScore = median(allValidScores);
  const absDev = allValidScores.map((v) => Math.abs(v - medScore));
  const mad = median(absDev);
  let scale = 1.4826 * mad;
  if (!(mad > 0)) {
    let mean = 0;
    for (const v of allValidScores) mean += v;
    mean /= allValidScores.length;
    let v2 = 0;
    for (const v of allValidScores) v2 += (v - mean) * (v - mean);
    scale = Math.sqrt(v2 / allValidScores.length) || 1.0;
  }
  const medBits = median(allValidWinBits);

  const windowTracks = [];
  const islands = [];

  for (let r = 0; r < rawWindows.length; r++) {
    const { chrom, starts, ends, wBits, wScore } = rawWindows[r];
    const z = new Float64Array(wScore.length);
    const flagged = [];
    for (let w = 0; w < wScore.length; w++) {
      if (Number.isFinite(wScore[w])) {
        z[w] = (wScore[w] - medScore) / scale;
        if (z[w] >= zThreshold) flagged.push(w);
      } else {
        z[w] = NaN;
      }
    }
    windowTracks.push({ chrom, starts, ends, meanBits: wBits, zScores: z });
    if (flagged.length === 0) continue;

    const { csB, csV, csGC, csCan } = csums[r];
    const chromGC =
      isNuc && csCan[csCan.length - 1] > 0
        ? csGC[csGC.length - 1] / csCan[csCan.length - 1]
        : NaN;

    const groups = [];
    let g0 = flagged[0];
    let g1 = flagged[0];
    for (let k = 1; k < flagged.length; k++) {
      const fi = flagged[k];
      if (starts[fi] - ends[g1] <= mergeGap) {
        g1 = fi;
      } else {
        groups.push([g0, g1]);
        g0 = fi;
        g1 = fi;
      }
    }
    groups.push([g0, g1]);

    for (const [w0, w1] of groups) {
      const istart = starts[w0];
      const iend = ends[w1];
      if (iend - istart < effMinLen) continue;

      let bestW = w0;
      let peakZ = -Infinity;
      for (let w = w0; w <= w1; w++) {
        if (z[w] > peakZ) {
          peakZ = z[w];
          bestW = w;
        }
      }
      const peakPos = (starts[bestW] + ends[bestW]) >> 1;
      const vCnt = csV[iend] - csV[istart];
      const islBits = vCnt > 0 ? (csB[iend] - csB[istart]) / vCnt : NaN;
      const deltaBits = Number.isFinite(islBits) ? islBits - medBits : NaN;

      let gcContent = NaN;
      let deltaGc = NaN;
      if (isNuc) {
        const canCnt = csCan[iend] - csCan[istart];
        if (canCnt > 0) {
          gcContent = (csGC[iend] - csGC[istart]) / canCnt;
          deltaGc = gcContent - chromGC;
        }
      }

      islands.push({
        chrom,
        start: istart,
        end: iend,
        length: iend - istart,
        peakPos,
        peakZ,
        meanBits: islBits,
        deltaBits,
        gcContent,
        deltaGc,
        pValue: normSf(peakZ),
        qValue: 1.0,
        significant: false,
      });
    }
  }

  if (islands.length > 0) {
    const nIndep = Math.max(1, Math.floor(totalValidBases / win));
    const qVals = benjaminiHochberg(
      islands.map((isl) => isl.pValue),
      Math.max(nIndep, islands.length)
    );
    for (let i = 0; i < islands.length; i++) {
      islands[i].qValue = qVals[i];
      islands[i].significant = qVals[i] <= alpha;
    }
  }
  return { islands, windowTracks };
}

export function scanAgainstReference(
  refRecords,
  queryRecords,
  {
    alphabet = DNA,
    order = 7,
    maxPrototypes = null,
    doubleStrand = true,
    window = 1000,
    step = 250,
    zThreshold = 3.5,
    alpha = 0.05,
    mergeGap = 2000,
    minLength = null,
    normalizeEntropy = true,
    onProgress = null,
  } = {}
) {
  const effMax = maxPrototypes ?? defaultMaxPrototypes(alphabet, order);
  const cortex = new FractalPredictiveCortex(alphabet, { order, maxPrototypes: effMax });
  for (let i = 0; i < refRecords.length; i++) {
    const ids = alphabet.encode(refRecords[i].seq);
    cortex.fit(ids, {
      epochs: 1,
      doubleStrand: doubleStrand && alphabet.isComplementary,
      onProgress: onProgress
        ? (done, tot) => onProgress("training reference", done, tot)
        : null,
    });
  }
  const qEncoded = queryRecords.map((r) => alphabet.encode(r.seq));
  const qBits = qEncoded.map((ids) =>
    cortex.score(ids, onProgress ? (done, tot) => onProgress("scanning query", done, tot) : null).bits
  );
  const { islands, windowTracks } = callIslandsFromBits(queryRecords, qEncoded, qBits, alphabet, {
    window,
    step,
    zThreshold,
    alpha,
    mergeGap,
    minLength,
    normalizeEntropy,
  });
  return { islands, windowTracks, cortex, qEncoded, qBits };
}

export function scanSelfIslands(
  queryRecords,
  {
    alphabet = DNA,
    order = 7,
    maxPrototypes = null,
    doubleStrand = true,
    nFolds = 8,
    guardBp = 25000,
    window = 1000,
    step = 250,
    zThreshold = 3.5,
    alpha = 0.05,
    mergeGap = 2000,
    minLength = null,
    normalizeEntropy = true,
    onProgress = null,
  } = {}
) {
  const effMax = maxPrototypes ?? defaultMaxPrototypes(alphabet, order);
  const qEncoded = queryRecords.map((r) => alphabet.encode(r.seq));
  const spacerLen = Math.max(1, order);
  let concatLen = 0;
  for (let i = 0; i < qEncoded.length; i++) {
    if (i > 0) concatLen += spacerLen;
    concatLen += qEncoded[i].length;
  }
  if (concatLen === 0) throw new Error("Query sequence contains 0 bases");

  const concatIds = new Int32Array(concatLen);
  concatIds.fill(alphabet.wildcardId);
  const offsets = [];
  let cursor = 0;
  for (let i = 0; i < qEncoded.length; i++) {
    if (i > 0) cursor += spacerLen;
    const s = cursor;
    concatIds.set(qEncoded[i], cursor);
    cursor += qEncoded[i].length;
    offsets.push([s, cursor]);
  }

  const effFolds = Math.max(2, Math.min(nFolds, Math.max(2, Math.floor(concatLen / Math.max(200, window * 2)))));
  const effGuard = Math.min(guardBp, Math.floor(concatLen / (2 * effFolds)));
  const oofBits = new Float64Array(concatLen);
  oofBits.fill(NaN);
  const ctxPad = Math.max(1, order);

  let visCortex = null;
  for (let f = 0; f < effFolds; f++) {
    const fStart = Math.floor((f * concatLen) / effFolds);
    const fEnd = Math.floor(((f + 1) * concatLen) / effFolds);
    if (fEnd <= fStart) continue;

    if (onProgress) onProgress(`cross-fold ${f + 1}/${effFolds}`, f + 1, effFolds);
    const exclStart = Math.max(0, fStart - effGuard);
    const exclEnd = Math.min(concatLen, fEnd + effGuard);

    const cortex = new FractalPredictiveCortex(alphabet, { order, maxPrototypes: effMax });
    if (exclStart > 0) {
      cortex.fit(concatIds.subarray(0, exclStart), {
        epochs: 1,
        doubleStrand: doubleStrand && alphabet.isComplementary,
      });
    }
    if (exclEnd < concatLen) {
      cortex.fit(concatIds.subarray(exclEnd), {
        epochs: 1,
        doubleStrand: doubleStrand && alphabet.isComplementary,
      });
    }
    const scoreStart = Math.max(0, fStart - ctxPad);
    const scored = cortex.score(concatIds.subarray(scoreStart, fEnd)).bits;
    oofBits.set(scored.subarray(fStart - scoreStart), fStart);
    if (f === effFolds - 1) visCortex = cortex;
  }

  const qBits = offsets.map(([s, e]) => oofBits.slice(s, e));
  const { islands, windowTracks } = callIslandsFromBits(queryRecords, qEncoded, qBits, alphabet, {
    window,
    step,
    zThreshold,
    alpha,
    mergeGap,
    minLength,
    normalizeEntropy,
  });
  return { islands, windowTracks, cortex: visCortex, qEncoded, qBits };
}

export function formatBed(islands, onlySignificant = false) {
  const lines = [];
  let idx = 0;
  for (const isl of islands) {
    idx++;
    if (onlySignificant && !isl.significant) continue;
    const dgc = Number.isFinite(isl.deltaGc)
      ? `_dGC${isl.deltaGc >= 0 ? "+" : ""}${isl.deltaGc.toFixed(3)}`
      : "";
    const tag = isl.significant ? "_FDR" : "_cand";
    const name = `island_${idx}_z${isl.peakZ.toFixed(2)}${dgc}${tag}`;
    const score = Math.min(1000, Math.max(0, Math.round(isl.peakZ * 100)));
    lines.push(`${isl.chrom}\t${isl.start}\t${isl.end}\t${name}\t${score}\t.`);
  }
  return lines.join("\n") + (lines.length ? "\n" : "");
}

export function formatBedGraph(windowTracks, metric = "z_score") {
  const lines = [
    `track type=bedGraph name="GenoCortex_Surprise" description="GenoCortex ${metric}"`,
  ];
  for (const wt of windowTracks) {
    const vals = metric === "bits" ? wt.meanBits : wt.zScores;
    for (let i = 0; i < wt.starts.length; i++) {
      const v = vals[i];
      if (!Number.isFinite(v)) continue;
      const s = wt.starts[i];
      const e = i + 1 < wt.starts.length ? wt.starts[i + 1] : wt.ends[i];
      if (e > s) lines.push(`${wt.chrom}\t${s}\t${e}\t${v.toFixed(4)}`);
    }
  }
  return lines.join("\n") + "\n";
}

export function formatIslandsTsv(islands) {
  const lines = [
    "chrom\tstart\tend\tlength\tpeak_pos\tpeak_z\tmean_bits\tdelta_bits\tgc_content\tdelta_gc\tp_value\tq_value\tsignificant",
  ];
  for (const isl of islands) {
    const gcS = Number.isFinite(isl.gcContent) ? isl.gcContent.toFixed(4) : "NA";
    const dgcS = Number.isFinite(isl.deltaGc)
      ? `${isl.deltaGc >= 0 ? "+" : ""}${isl.deltaGc.toFixed(4)}`
      : "NA";
    const dbS = `${isl.deltaBits >= 0 ? "+" : ""}${isl.deltaBits.toFixed(4)}`;
    lines.push(
      `${isl.chrom}\t${isl.start}\t${isl.end}\t${isl.length}\t${isl.peakPos}\t` +
        `${isl.peakZ.toFixed(3)}\t${isl.meanBits.toFixed(4)}\t${dbS}\t` +
        `${gcS}\t${dgcS}\t${isl.pValue.toExponential(3)}\t${isl.qValue.toExponential(3)}\t` +
        `${isl.significant}`
    );
  }
  return lines.join("\n") + "\n";
}

// ---------------------------------------------------------------------------
// Strand-Aware Classifier
// ---------------------------------------------------------------------------
export class FPCClassifier {
  constructor(
    alphabet = DNA,
    { order = 7, epochs = 1, strandAware = null, doubleStrandTrain = false, maxPrototypes = null } = {}
  ) {
    this.alphabet = alphabet;
    this.order = order;
    this.epochs = epochs;
    this.strandAware = strandAware === null ? alphabet.isComplementary : Boolean(strandAware);
    this.doubleStrandTrain = Boolean(doubleStrandTrain);
    this.maxPrototypes = maxPrototypes ?? defaultMaxPrototypes(alphabet, order);
    this.classes = [];
    this.models = new Map();
  }

  fit(recordsWithLabels) {
    const byLabel = new Map();
    for (const item of recordsWithLabels) {
      if (!byLabel.has(item.label)) byLabel.set(item.label, []);
      byLabel.get(item.label).push(item.seq);
    }
    this.classes = Array.from(byLabel.keys()).sort();
    this.models.clear();
    for (const lab of this.classes) {
      const cortex = new FractalPredictiveCortex(this.alphabet, {
        order: this.order,
        maxPrototypes: this.maxPrototypes,
      });
      for (const seq of byLabel.get(lab)) {
        cortex.fit(this.alphabet.encode(seq), {
          epochs: this.epochs,
          doubleStrand: this.doubleStrandTrain && this.alphabet.isComplementary,
        });
      }
      this.models.set(lab, cortex);
    }
    return this;
  }

  classifyRecords(records) {
    const nCls = this.classes.length;
    if (nCls === 0) throw new Error("Classifier has no trained classes");
    const checkRc = this.strandAware && this.alphabet.isComplementary;
    const calls = [];

    for (const rec of records) {
      const ids = this.alphabet.encode(rec.seq);
      const rcIds = checkRc && ids.length > 0 ? this.alphabet.reverseComplementIds(ids) : null;
      const scores = new Float64Array(nCls);
      const strands = new Array(nCls);

      for (let j = 0; j < nCls; j++) {
        const model = this.models.get(this.classes[j]);
        const fwd = model.score(ids).meanBits;
        if (rcIds !== null) {
          const rev = model.score(rcIds).meanBits;
          if (Number.isFinite(rev) && (!Number.isFinite(fwd) || rev < fwd)) {
            scores[j] = rev;
            strands[j] = "-";
            continue;
          }
        }
        scores[j] = fwd;
        strands[j] = "+";
      }

      let anyFinite = false;
      for (let j = 0; j < nCls; j++) {
        if (Number.isFinite(scores[j])) anyFinite = true;
      }
      if (!anyFinite) {
        calls.push({
          id: rec.id,
          length: rec.seq.length,
          predictedClass: "UNCLASSIFIED",
          strand: ".",
          bitsPerBase: NaN,
          posterior: 1.0 / Math.max(1, nCls),
          runnerUp: null,
          deltaBits: 0.0,
        });
        continue;
      }

      let maxLogit = -Infinity;
      const logits = new Float64Array(nCls);
      for (let j = 0; j < nCls; j++) {
        logits[j] = Number.isFinite(scores[j]) ? -scores[j] : -1e9;
        if (logits[j] > maxLogit) maxLogit = logits[j];
      }
      let pSum = 0;
      const probs = new Float64Array(nCls);
      for (let j = 0; j < nCls; j++) {
        probs[j] = Math.exp(logits[j] - maxLogit);
        pSum += probs[j];
      }
      for (let j = 0; j < nCls; j++) probs[j] /= pSum || 1.0;

      const order = Array.from({ length: nCls }, (_, j) => j).sort((a, b) => {
        const sa = Number.isFinite(scores[a]) ? scores[a] : Infinity;
        const sb = Number.isFinite(scores[b]) ? scores[b] : Infinity;
        return sa - sb || a - b;
      });
      const bestJ = order[0];
      const secJ = nCls >= 2 && Number.isFinite(scores[order[1]]) ? order[1] : -1;

      calls.push({
        id: rec.id,
        length: rec.seq.length,
        predictedClass: this.classes[bestJ],
        strand: strands[bestJ],
        bitsPerBase: scores[bestJ],
        posterior: probs[bestJ],
        runnerUp: secJ >= 0 ? this.classes[secJ] : null,
        deltaBits: secJ >= 0 ? scores[secJ] - scores[bestJ] : 0.0,
      });
    }
    return calls;
  }
}

export function formatClassificationTsv(calls) {
  const lines = [
    "id\tlength\tpredicted_class\tstrand\tbits_per_base\tposterior\trunner_up\tdelta_bits",
  ];
  for (const c of calls) {
    const bpb = Number.isFinite(c.bitsPerBase) ? c.bitsPerBase.toFixed(4) : "NA";
    const db = Number.isFinite(c.deltaBits) ? c.deltaBits.toFixed(4) : "0.0000";
    lines.push(
      `${c.id}\t${c.length}\t${c.predictedClass}\t${c.strand}\t${bpb}\t` +
        `${c.posterior.toFixed(4)}\t${c.runnerUp ?? "NA"}\t${db}`
    );
  }
  return lines.join("\n") + "\n";
}

// ---------------------------------------------------------------------------
// Phylogenetics: Distance Matrix, Saitou-Nei NJ, Newick, 3-D MDS & MST
// ---------------------------------------------------------------------------
function compareInt32Arrays(a, b) {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    if (a[i] !== b[i]) return a[i] - b[i];
  }
  return a.length - b.length;
}

export function distanceMatrix(
  sequences,
  { alphabet = DNA, order = 6, epochs = 1, strandInvariant = false, maxPrototypes = null, onProgress = null } = {}
) {
  const n = sequences.length;
  const effMax = maxPrototypes ?? defaultMaxPrototypes(alphabet, order);
  const useSym = Boolean(strandInvariant && alphabet.isComplementary);
  const fwdIds = sequences.map((s) => alphabet.encode(s));
  const rcIds = useSym ? fwdIds.map((x) => alphabet.reverseComplementIds(x)) : null;

  const cortices = new Array(n);
  for (let i = 0; i < n; i++) {
    if (onProgress) onProgress(`learning taxon ${i + 1}/${n}`, i + 1, n * 2);
    const c = new FractalPredictiveCortex(alphabet, { order, maxPrototypes: effMax });
    if (useSym && rcIds) {
      const cmp = compareInt32Arrays(fwdIds[i], rcIds[i]);
      const first = cmp <= 0 ? fwdIds[i] : rcIds[i];
      const second = cmp <= 0 ? rcIds[i] : fwdIds[i];
      for (let ep = 0; ep < Math.max(1, epochs); ep++) {
        c.fit(first, { epochs: 1 });
        c.fit(second, { epochs: 1 });
      }
    } else {
      c.fit(fwdIds[i], { epochs });
    }
    cortices[i] = c;
  }

  const ce = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    if (onProgress) onProgress(`cross-scoring ${i + 1}/${n}`, n + i + 1, n * 2);
    for (let j = 0; j < n; j++) {
      const cf = cortices[i].crossEntropy(fwdIds[j]);
      if (useSym && rcIds) {
        const cr = cortices[i].crossEntropy(rcIds[j]);
        ce[i][j] = 0.5 * (cf + cr);
      } else {
        ce[i][j] = cf;
      }
    }
  }

  const D = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const d = 0.5 * (ce[i][j] - ce[j][j] + (ce[j][i] - ce[i][i]));
      const val = Math.max(0.0, d);
      D[i][j] = val;
      D[j][i] = val;
    }
  }
  return D;
}

function formatNewickLabel(name) {
  const clean = String(name || "unnamed").trim();
  if (/[\s():;,'[\]]/.test(clean)) {
    return `'${clean.replace(/'/g, "_")}'`;
  }
  return clean;
}

function nodeToNewickBody(node, precision = 6) {
  let body = "";
  if (node.children && node.children.length > 0) {
    body = "(" + node.children.map((c) => nodeToNewickBody(c, precision)).join(",") + ")";
    if (node.name) body += formatNewickLabel(node.name);
  } else {
    body = formatNewickLabel(node.name);
  }
  if (node.branchLength !== null && node.branchLength !== undefined) {
    body += `:${node.branchLength.toFixed(precision)}`;
  }
  return body;
}

export function neighborJoining(D, names, { clampNegative = true, precision = 6 } = {}) {
  const n = D.length;
  if (n !== names.length) throw new Error("Distance matrix and names length mismatch");
  if (n === 0) throw new Error("Cannot build tree from 0 taxa");
  if (n === 1) {
    const root = { name: String(names[0]), branchLength: 0.0, children: [] };
    return { root, newick: nodeToNewickBody(root, precision) + ";" };
  }

  let dist = Array.from({ length: n }, (_, i) =>
    Float64Array.from({ length: n }, (__, j) => (i === j ? 0.0 : 0.5 * (D[i][j] + D[j][i])))
  );

  if (n === 2) {
    const half = clampNegative ? Math.max(0.0, 0.5 * dist[0][1]) : 0.5 * dist[0][1];
    const root = {
      name: null,
      branchLength: null,
      children: [
        { name: String(names[0]), branchLength: half, children: [] },
        { name: String(names[1]), branchLength: half, children: [] },
      ],
    };
    return { root, newick: nodeToNewickBody(root, precision) + ";" };
  }

  let nodes = names.map((nm) => ({ name: String(nm), branchLength: null, children: [] }));

  while (nodes.length > 3) {
    const m = nodes.length;
    const r = new Float64Array(m);
    for (let i = 0; i < m; i++) {
      let s = 0;
      for (let j = 0; j < m; j++) s += dist[i][j];
      r[i] = s;
    }

    let bestI = 0;
    let bestJ = 1;
    let bestQ = Infinity;
    for (let i = 0; i < m; i++) {
      for (let j = i + 1; j < m; j++) {
        const q = (m - 2.0) * dist[i][j] - r[i] - r[j];
        if (q < bestQ) {
          bestQ = q;
          bestI = i;
          bestJ = j;
        }
      }
    }

    const dij = dist[bestI][bestJ];
    let deltaI = 0.5 * dij + (r[bestI] - r[bestJ]) / (2.0 * (m - 2.0));
    let deltaJ = dij - deltaI;
    if (clampNegative) {
      deltaI = Math.max(0.0, deltaI);
      deltaJ = Math.max(0.0, deltaJ);
    }

    const childI = nodes[bestI];
    const childJ = nodes[bestJ];
    childI.branchLength = deltaI;
    childJ.branchLength = deltaJ;
    const parent = { name: null, branchLength: null, children: [childI, childJ] };

    const keep = [];
    for (let k = 0; k < m; k++) {
      if (k !== bestI && k !== bestJ) keep.push(k);
    }

    const nextM = m - 1;
    const nextDist = Array.from({ length: nextM }, () => new Float64Array(nextM));
    for (let a = 0; a < keep.length; a++) {
      for (let b = 0; b < keep.length; b++) {
        nextDist[a][b] = dist[keep[a]][keep[b]];
      }
      let dNew = 0.5 * (dist[bestI][keep[a]] + dist[bestJ][keep[a]] - dij);
      if (clampNegative) dNew = Math.max(0.0, dNew);
      nextDist[a][nextM - 1] = dNew;
      nextDist[nextM - 1][a] = dNew;
    }

    nodes = keep.map((k) => nodes[k]);
    nodes.push(parent);
    dist = nextDist;
  }

  let b0 = 0.5 * (dist[0][1] + dist[0][2] - dist[1][2]);
  let b1 = 0.5 * (dist[0][1] + dist[1][2] - dist[0][2]);
  let b2 = 0.5 * (dist[0][2] + dist[1][2] - dist[0][1]);
  if (clampNegative) {
    b0 = Math.max(0.0, b0);
    b1 = Math.max(0.0, b1);
    b2 = Math.max(0.0, b2);
  }
  nodes[0].branchLength = b0;
  nodes[1].branchLength = b1;
  nodes[2].branchLength = b2;

  const root = { name: null, branchLength: null, children: [nodes[0], nodes[1], nodes[2]] };
  return { root, newick: nodeToNewickBody(root, precision) + ";" };
}

export function formatDistanceTsv(D, names, precision = 6) {
  const lines = ["taxon\t" + names.join("\t")];
  for (let i = 0; i < names.length; i++) {
    const vals = [];
    for (let j = 0; j < names.length; j++) vals.push(D[i][j].toFixed(precision));
    lines.push(`${names[i]}\t${vals.join("\t")}`);
  }
  return lines.join("\n") + "\n";
}

// Classical 3-D MDS via symmetric Jacobi eigendecomposition + Prim's MST
export function mdsEmbedding(D, dims = 3) {
  const n = D.length;
  if (n === 0) return [];
  const D2 = Array.from({ length: n }, (_, i) =>
    Float64Array.from({ length: n }, (__, j) => D[i][j] * D[i][j])
  );
  const rowMean = new Float64Array(n);
  let grandMean = 0;
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) s += D2[i][j];
    rowMean[i] = s / n;
    grandMean += s;
  }
  grandMean /= n * n;

  // B = -0.5 * J * D^2 * J
  const A = Array.from({ length: n }, () => new Float64Array(n));
  const V = Array.from({ length: n }, (_, i) => {
    const row = new Float64Array(n);
    row[i] = 1.0;
    return row;
  });
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const val = -0.5 * (D2[i][j] - rowMean[i] - rowMean[j] + grandMean);
      A[i][j] = val;
      A[j][i] = val;
    }
  }

  // Jacobi rotations for symmetric matrix A
  const maxSweeps = 60;
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let maxOff = 0.0;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const a = Math.abs(A[p][q]);
        if (a > maxOff) maxOff = a;
      }
    }
    if (maxOff < 1e-13) break;

    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = A[p][q];
        if (Math.abs(apq) < 1e-15) continue;
        const app = A[p][p];
        const aqq = A[q][q];
        const tau = (aqq - app) / (2.0 * apq);
        const t =
          tau >= 0
            ? 1.0 / (tau + Math.sqrt(1.0 + tau * tau))
            : -1.0 / (-tau + Math.sqrt(1.0 + tau * tau));
        const c = 1.0 / Math.sqrt(1.0 + t * t);
        const s = t * c;

        A[p][p] = app - t * apq;
        A[q][q] = aqq + t * apq;
        A[p][q] = 0.0;
        A[q][p] = 0.0;

        for (let r = 0; r < n; r++) {
          if (r !== p && r !== q) {
            const arp = A[r][p];
            const arq = A[r][q];
            A[r][p] = A[p][r] = c * arp - s * arq;
            A[r][q] = A[q][r] = s * arp + c * arq;
          }
          const vrp = V[r][p];
          const vrq = V[r][q];
          V[r][p] = c * vrp - s * vrq;
          V[r][q] = s * vrp + c * vrq;
        }
      }
    }
  }

  const eigOrder = Array.from({ length: n }, (_, i) => i).sort((a, b) => A[b][b] - A[a][a]);
  const coords = Array.from({ length: n }, () => new Float64Array(dims));
  for (let d = 0; d < Math.min(dims, n); d++) {
    const col = eigOrder[d];
    const scale = Math.sqrt(Math.max(0.0, A[col][col]));
    for (let i = 0; i < n; i++) {
      coords[i][d] = V[i][col] * scale;
    }
  }
  return coords;
}

export function minimumSpanningTree(D) {
  const n = D.length;
  if (n < 2) return [];
  const inTree = new Uint8Array(n);
  const minDist = new Float64Array(n).fill(Infinity);
  const parent = new Int32Array(n).fill(-1);
  inTree[0] = 1;
  for (let j = 1; j < n; j++) {
    minDist[j] = D[0][j];
    parent[j] = 0;
  }
  const edges = [];
  for (let step = 1; step < n; step++) {
    let u = -1;
    let best = Infinity;
    for (let j = 0; j < n; j++) {
      if (!inTree[j] && minDist[j] < best) {
        best = minDist[j];
        u = j;
      }
    }
    if (u < 0) break;
    inTree[u] = 1;
    edges.push([parent[u], u, D[parent[u]][u]]);
    for (let v = 0; v < n; v++) {
      if (!inTree[v] && D[u][v] < minDist[v]) {
        minDist[v] = D[u][v];
        parent[v] = u;
      }
    }
  }
  return edges;
}

// ---------------------------------------------------------------------------
// 3-D Cognitive Map Terrain Builder (for live Plotly 3-D surface)
// ---------------------------------------------------------------------------
export function buildCognitiveTerrain(cortex, ids, grid = 72) {
  const P = cortex.embedding.embed(ids);
  const track = cortex.score(ids);
  const bits = track.bits;
  const acc = new Float64Array(grid * grid);
  const cnt = new Int32Array(grid * grid);

  for (let t = 0; t < bits.length; t++) {
    const b = bits[t];
    if (Number.isNaN(b)) continue;
    const gx = Math.min(grid - 1, Math.max(0, Math.floor(P[2 * t] * grid)));
    const gy = Math.min(grid - 1, Math.max(0, Math.floor(P[2 * t + 1] * grid)));
    const idx = gy * grid + gx;
    acc[idx] += b;
    cnt[idx] += 1;
  }

  // Separable 3x3 NaN-aware Gaussian smoothing (1-2-1 kernel)
  const raw = new Float64Array(grid * grid).fill(NaN);
  for (let i = 0; i < grid * grid; i++) {
    if (cnt[i] > 0) raw[i] = acc[i] / cnt[i];
  }

  const zRows = [];
  const xs = Array.from({ length: grid }, (_, i) => (i + 0.5) / grid);
  const ys = Array.from({ length: grid }, (_, i) => (i + 0.5) / grid);
  let zMin = Infinity;
  let zMax = -Infinity;

  for (let gy = 0; gy < grid; gy++) {
    const row = new Array(grid);
    for (let gx = 0; gx < grid; gx++) {
      let num = 0;
      let den = 0;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = gy + dy;
        if (yy < 0 || yy >= grid) continue;
        const wy = dy === 0 ? 2 : 1;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = gx + dx;
          if (xx < 0 || xx >= grid) continue;
          const v = raw[yy * grid + xx];
          if (!Number.isNaN(v)) {
            const w = wy * (dx === 0 ? 2 : 1);
            num += w * v;
            den += w;
          }
        }
      }
      if (den > 0) {
        const val = num / den;
        row[gx] = val;
        if (val < zMin) zMin = val;
        if (val > zMax) zMax = val;
      } else {
        row[gx] = null;
      }
    }
    zRows.push(row);
  }
  if (!Number.isFinite(zMin)) {
    zMin = 0;
    zMax = 2;
  }

  const pf = cortex.prototypeFrame();
  return { xs, ys, zRows, zMin, zMax, prototypes: pf, symbols: cortex.alphabet.symbols };
}
