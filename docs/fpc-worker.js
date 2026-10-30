/**
 * GenoCortex v1.0 — Web Worker Runner
 * ===================================
 * Executes live sequence island scanning, Neighbour-Joining phylogenetics,
 * strand-aware MDL read classification, and predictive perplexity evaluation
 * off the browser main thread with real-time progress telemetry.
 */

import {
  getAlphabet,
  parseSequences,
  FractalPredictiveCortex,
  FPCClassifier,
  defaultMaxPrototypes,
  scanAgainstReference,
  scanSelfIslands,
  formatBed,
  formatBedGraph,
  formatIslandsTsv,
  formatClassificationTsv,
  distanceMatrix,
  neighborJoining,
  formatDistanceTsv,
  mdsEmbedding,
  minimumSpanningTree,
  buildCognitiveTerrain,
} from "./fpc-engine.js";

function postProgress(id, stage, current, total) {
  self.postMessage({ type: "progress", id, stage, current, total });
}

self.onmessage = (event) => {
  const { id, task, payload } = event.data || {};
  const t0 = performance.now();

  try {
    if (task === "scan") {
      const alphabet = getAlphabet(payload.alphabet || "DNA");
      const queryRecords = parseSequences(payload.queryText, {
        minPhred: payload.minPhred || 0,
        wildcard: alphabet.wildcard,
      });
      const order = Number(payload.order || 7);
      const window = Number(payload.window || 500);
      const step = Number(payload.step || 125);
      const zThreshold = Number(payload.zThreshold || 3.2);
      const alpha = Number(payload.alpha || 0.05);
      const mergeGap = Number(payload.mergeGap || 1000);

      let scanOut;
      if (payload.refText && String(payload.refText).trim().length > 0) {
        const refRecords = parseSequences(payload.refText, {
          minPhred: payload.minPhred || 0,
          wildcard: alphabet.wildcard,
        });
        scanOut = scanAgainstReference(refRecords, queryRecords, {
          alphabet,
          order,
          window,
          step,
          zThreshold,
          alpha,
          mergeGap,
          onProgress: (stage, cur, tot) => postProgress(id, stage, cur, tot),
        });
      } else {
        scanOut = scanSelfIslands(queryRecords, {
          alphabet,
          order,
          window,
          step,
          zThreshold,
          alpha,
          mergeGap,
          onProgress: (stage, cur, tot) => postProgress(id, stage, cur, tot),
        });
      }

      postProgress(id, "building 3-D cognitive terrain", 1, 1);
      // Use first contig (capped at 250 kb for smooth 60fps WebGL terrain rendering)
      const visIds = scanOut.qEncoded[0].subarray(0, Math.min(scanOut.qEncoded[0].length, 250000));
      const terrain = buildCognitiveTerrain(scanOut.cortex, visIds, 64);

      const firstTrack = scanOut.windowTracks[0] || {
        chrom: queryRecords[0].id,
        starts: new Int32Array(0),
        ends: new Int32Array(0),
        meanBits: new Float64Array(0),
        zScores: new Float64Array(0),
      };
      const totalBp = queryRecords.reduce((acc, r) => acc + r.seq.length, 0);
      const elapsedMs = performance.now() - t0;

      self.postMessage({
        type: "result",
        id,
        task,
        result: {
          elapsedMs,
          totalBp,
          nContigs: queryRecords.length,
          nPrototypes: scanOut.cortex.nPrototypes,
          islands: scanOut.islands,
          track: {
            chrom: firstTrack.chrom,
            starts: Array.from(firstTrack.starts),
            ends: Array.from(firstTrack.ends),
            meanBits: Array.from(firstTrack.meanBits),
            zScores: Array.from(firstTrack.zScores),
          },
          terrain: {
            xs: terrain.xs,
            ys: terrain.ys,
            zRows: terrain.zRows,
            zMin: terrain.zMin,
            zMax: terrain.zMax,
            symbols: terrain.symbols,
            prototypes: {
              x: Array.from(terrain.prototypes.x),
              y: Array.from(terrain.prototypes.y),
              usage: Array.from(terrain.prototypes.usage),
              dominant: Array.from(terrain.prototypes.dominant),
            },
          },
          downloads: {
            bed: formatBed(scanOut.islands, false),
            bedgraph: formatBedGraph(scanOut.windowTracks, "z_score"),
            tsv: formatIslandsTsv(scanOut.islands),
          },
        },
      });
      return;
    }

    if (task === "tree") {
      const alphabet = getAlphabet(payload.alphabet || "DNA");
      const records = parseSequences(payload.fastaText, {
        minPhred: payload.minPhred || 0,
        wildcard: alphabet.wildcard,
      });
      if (records.length < 2) {
        throw new Error(`Provide at least 2 FASTA/FASTQ records for phylogenetics (got ${records.length})`);
      }
      const names = records.map((r) => r.id);
      const seqs = records.map((r) => r.seq);
      const order = Number(payload.order || 6);
      const strandInvariant = payload.strandInvariant !== false && alphabet.isComplementary;

      const D = distanceMatrix(seqs, {
        alphabet,
        order,
        strandInvariant,
        onProgress: (stage, cur, tot) => postProgress(id, stage, cur, tot),
      });
      const { newick } = neighborJoining(D, names);
      const coords = mdsEmbedding(D, 3);
      const mstEdges = minimumSpanningTree(D);
      const totalBp = records.reduce((acc, r) => acc + r.seq.length, 0);
      const elapsedMs = performance.now() - t0;

      self.postMessage({
        type: "result",
        id,
        task,
        result: {
          elapsedMs,
          totalBp,
          names,
          lengths: records.map((r) => r.seq.length),
          newick,
          distanceMatrix: D.map((row) => Array.from(row)),
          coords: coords.map((row) => Array.from(row)),
          mstEdges,
          downloads: {
            newick: newick + "\n",
            tsv: formatDistanceTsv(D, names),
          },
        },
      });
      return;
    }

    if (task === "classify") {
      const alphabet = getAlphabet(payload.alphabet || "DNA");
      const refRecords = parseSequences(payload.refText, {
        minPhred: payload.minPhred || 0,
        wildcard: alphabet.wildcard,
      });
      if (refRecords.length < 2) {
        throw new Error("Provide at least 2 reference FASTA records (each '>' record ID is a class label)");
      }
      const queryRecords = parseSequences(payload.queryText, {
        minPhred: payload.minPhred || 0,
        wildcard: alphabet.wildcard,
      });
      const order = Number(payload.order || 7);
      postProgress(id, "training reference cortices", 1, 2);

      const clf = new FPCClassifier(alphabet, {
        order,
        strandAware: payload.strandAware !== false,
        doubleStrandTrain: Boolean(payload.doubleStrandTrain),
      });
      clf.fit(refRecords.map((r) => ({ label: r.id, seq: r.seq })));

      postProgress(id, "scoring queries on +/- strands", 2, 2);
      const calls = clf.classifyRecords(queryRecords);
      const elapsedMs = performance.now() - t0;
      const totalBp = queryRecords.reduce((acc, r) => acc + r.seq.length, 0);

      self.postMessage({
        type: "result",
        id,
        task,
        result: {
          elapsedMs,
          totalBp,
          classes: clf.classes,
          calls,
          downloads: {
            tsv: formatClassificationTsv(calls),
          },
        },
      });
      return;
    }

    if (task === "predict") {
      const alphabet = getAlphabet(payload.alphabet || "DNA");
      const trainRecords = parseSequences(payload.trainText, {
        minPhred: payload.minPhred || 0,
        wildcard: alphabet.wildcard,
      });
      const order = Number(payload.order || 7);
      const cortex = new FractalPredictiveCortex(alphabet, {
        order,
        maxPrototypes: defaultMaxPrototypes(alphabet, order),
      });
      let trainBp = 0;
      let totalBits = 0;
      let validBp = 0;
      for (const r of trainRecords) {
        const ids = alphabet.encode(r.seq);
        const tr = cortex.fit(ids);
        trainBp += ids.length;
        totalBits += tr.totalBits;
        validBp += tr.nValid;
      }

      const rawPrefix = String(payload.prefix || "").trim().toUpperCase();
      const prefix =
        rawPrefix ||
        trainRecords[0].seq.slice(Math.max(0, trainRecords[0].seq.length - 24));
      const prefixIds = alphabet.encode(prefix);
      const nextDist = Array.from(cortex.predictNext(prefixIds));
      const genLen = Math.min(2000, Math.max(1, Number(payload.length || 120)));
      const genIds = cortex.generate(prefixIds, genLen, {
        sample: Boolean(payload.sample),
        temperature: Number(payload.temperature || 1.0),
      });
      const fullSeq = alphabet.decode(genIds);
      const continuation = fullSeq.slice(prefixIds.length);
      const meanBits = validBp > 0 ? totalBits / validBp : NaN;
      const elapsedMs = performance.now() - t0;

      self.postMessage({
        type: "result",
        id,
        task,
        result: {
          elapsedMs,
          trainBp,
          nPrototypes: cortex.nPrototypes,
          meanBits,
          perplexity: Number.isFinite(meanBits) ? Math.pow(2.0, meanBits) : NaN,
          symbols: alphabet.symbols.split(""),
          nextDist,
          prefix: alphabet.decode(prefixIds),
          continuation,
        },
      });
      return;
    }

    throw new Error(`Unknown worker task: ${task}`);
  } catch (err) {
    self.postMessage({
      type: "error",
      id,
      task,
      message: err && err.message ? err.message : String(err),
    });
  }
};
