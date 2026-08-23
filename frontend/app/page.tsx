"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
const FLAGSHIP = "What architecture was proposed to reduce database load, who explained it, and where was the corresponding diagram shown?";

type Evidence = { evidence_id: string; origin: "direct" | "graph_expanded"; final_score: number; selection_role?: string; node: {
  id: string; source_id: string; modality: string; content: string | null; start_sec: number | null;
  end_sec: number | null; speaker: string | null; page_number: number | null; frame_path: string | null;
  entities: string[]; confidence: number | null; provenance: Record<string, unknown> } };
type GraphResult = { answer: string; evidence_bundle: Evidence[]; latency_ms: number; connections: Array<{
  from_id: string; to_id: string; relationship_type: string; confidence: number; explanation: string;
  signal_breakdown: Record<string, number | null> }> };
type BaselineResult = { answer: string; latency_ms: number; results: Array<{
  chunk_id: string; text: string; source_filename: string; score?: number | null }> };
type EvidenceNode = Evidence["node"];
type EvidenceEdge = { neighbor_id: string; relationship_type: string; confidence: number;
  signal_breakdown: { temporal: number | null; entity: number | null; semantic: number | null };
  explanation: string };
type EvidenceDetail = { node: EvidenceNode; connected_edges: EvidenceEdge[]; neighbors: EvidenceNode[] };
type DemoStatus = { status: string; source_count: number; evidence_count: number; edge_count: number;
  active_retrieval_policy: string; evaluation_summary: { baseline: Record<string, number>; evidence_graph: Record<string, number> };
  sources: Array<{ id: string; filename: string; media_type: string;
  evidence_count: number; processing_status: string }> };

function clock(value: number | null) {
  if (value == null) return "";
  return `${Math.floor(value / 60)}:${Math.floor(value % 60).toString().padStart(2, "0")}`;
}
function mediaUrl(path: string | null) {
  return path ? `${API}/media/${path.replace(/^data[\\/]/, "").replaceAll("\\", "/")}` : null;
}
function safeScore(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "—";
}
function signal(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(2) : "—";
}

export default function Home() {
  const [question, setQuestion] = useState(FLAGSHIP);
  const [mode, setMode] = useState<"graph" | "baseline">("graph");
  const [demo, setDemo] = useState<DemoStatus | null>(null);
  const [graph, setGraph] = useState<GraphResult | null>(null);
  const [baseline, setBaseline] = useState<BaselineResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detail, setDetail] = useState<EvidenceDetail | null>(null);
  const [error, setError] = useState("");

  async function loadDemo() {
    setError("");
    try {
      const response = await fetch(`${API}/api/demo/load`);
      if (!response.ok) throw new Error("Demo corpus is unavailable.");
      setDemo(await response.json());
    } catch (cause) { setError(cause instanceof Error ? cause.message : "Could not load the demo corpus."); }
  }
  useEffect(() => { void loadDemo(); }, []);

  async function ask(event: FormEvent) {
    event.preventDefault(); setLoading(true); setError("");
    try {
      const endpoint = mode === "graph" ? "/api/query" : "/api/baseline/query";
      const response = await fetch(`${API}${endpoint}`, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, top_k: 5, ...(mode === "graph" ? { graph_hops: 1 } : {}) }) });
      if (!response.ok) throw new Error("The query could not be completed.");
      if (mode === "graph") setGraph(await response.json()); else setBaseline(await response.json());
    } catch (cause) { setError(cause instanceof Error ? cause.message : "Query failed."); }
    finally { setLoading(false); }
  }
  async function inspectEvidence(evidenceId: string) {
    setDetailLoading(true); setError("");
    try {
      const response = await fetch(`${API}/api/evidence/${encodeURIComponent(evidenceId)}`);
      if (!response.ok) throw new Error("Evidence details could not be loaded.");
      setDetail(await response.json());
    } catch (cause) { setError(cause instanceof Error ? cause.message : "Evidence lookup failed."); }
    finally { setDetailLoading(false); }
  }
  const evidence = useMemo(() => graph?.evidence_bundle.slice(0, 5) ?? [], [graph]);

  return <main>
    <header className="topbar"><a className="brand" href="#top"><span className="mark">E</span>EvidenceGraph</a>
      <div className="status"><span className="pulse" />{demo?.status === "ready" ? "Cached demo ready" : "Connecting"}</div></header>

    <section className="hero" id="top"><div><p className="eyebrow">Multimodal evidence retrieval</p>
      <h1>Ask once. See every source behind the answer.</h1>
      <p className="lede">Speech, frames, images, OCR, and documents stay connected through explicit temporal and entity relationships—with provenance intact.</p></div>
      <div className="corpus-card"><div className="corpus-heading"><span>Prepared corpus</span><b>{demo?.active_retrieval_policy ?? "V1.5"}</b></div>
        <div className="stats"><span><strong>{demo?.source_count ?? "—"}</strong> sources</span><span><strong>{demo?.evidence_count ?? "—"}</strong> evidence</span><span><strong>{demo?.edge_count ?? "—"}</strong> links</span></div>
        <button className="quiet-button" onClick={() => void loadDemo()}>Reload cached dataset</button></div></section>

    <section className="workspace"><form className="ask-panel" onSubmit={ask}>
      <div className="mode-switch" role="group" aria-label="Retrieval system"><button type="button" className={mode === "graph" ? "active" : ""} onClick={() => setMode("graph")}>EvidenceGraph</button><button type="button" className={mode === "baseline" ? "active" : ""} onClick={() => setMode("baseline")}>Text baseline</button></div>
      <label htmlFor="question">Question</label><textarea id="question" value={question} onChange={(event) => setQuestion(event.target.value)} rows={4} />
      <div className="form-footer"><span>{mode === "graph" ? "MiniLM + CLIP + entities + graph" : "MiniLM text chunks only"}</span><button className="ask-button" disabled={loading || !question.trim()}>{loading ? "Tracing evidence…" : "Trace the answer"}</button></div>
      {error && <p className="error" role="alert">{error}</p>}</form>
      <aside className="source-panel"><div className="section-title"><span>Sources</span><small>{demo?.source_count ?? 0} ready</small></div>
        <div className="source-list">{demo?.sources.map((source) => <div className="source-row" key={source.id}><span className={`source-icon ${source.media_type}`}>{source.media_type.slice(0, 1).toUpperCase()}</span><div><b>{source.filename}</b><small>{source.media_type} · {source.evidence_count} observations</small></div><span className="ready-dot" /></div>)}</div></aside></section>

    {demo && <section className="scoreboard"><div><p className="eyebrow">Frozen V1.5 evaluation</p><h2>Traceability is the advantage.</h2><p>The text baseline retrieves more exact groups; EvidenceGraph preserves substantially richer provenance and visual context.</p></div>
      <div className="metric"><span>Recall@5</span><b>{demo.evaluation_summary.evidence_graph.evidence_recall_at_5.toFixed(3)}</b><small>Baseline {demo.evaluation_summary.baseline.evidence_recall_at_5.toFixed(3)}</small></div>
      <div className="metric"><span>Complete</span><b>{demo.evaluation_summary.evidence_graph.complete_evidence_rate.toFixed(3)}</b><small>Baseline {demo.evaluation_summary.baseline.complete_evidence_rate.toFixed(3)}</small></div>
      <div className="metric highlight"><span>Provenance</span><b>{demo.evaluation_summary.evidence_graph.provenance_completeness.toFixed(3)}</b><small>Baseline {demo.evaluation_summary.baseline.provenance_completeness.toFixed(3)}</small></div></section>}

    {mode === "graph" && graph && <section className="results"><div className="answer-card"><p className="eyebrow">Provenance-preserving answer</p><p>{graph.answer.split("\n")[0]}</p><small>Deterministic · {graph.latency_ms.toFixed(1)} ms · {evidence.length} displayed observations</small></div>
      <div className="section-title"><span>Evidence chain</span><small>Direct retrieval + connected evidence</small></div><div className="evidence-grid">{evidence.map((item) => {
        const node = item.node; const image = mediaUrl(node.frame_path); return <button type="button" className="evidence-card" aria-label={`Inspect ${node.modality} evidence ${node.id}`} onClick={() => void inspectEvidence(node.id)} key={node.id}>{image && <img src={image} alt={`Evidence from ${String(node.provenance.file ?? node.source_id)}`} />}<div className="card-body"><div className="card-meta"><span className={`modality ${node.modality}`}>{node.modality.replace("_", " ")}</span><span>{item.origin === "direct" ? "Direct" : "Graph connected"}</span></div><p>{node.content || "Visual evidence from the source media."}</p><div className="provenance"><b>{String(node.provenance.file ?? node.source_id)}</b>{node.speaker && <span>Speaker {node.speaker}</span>}{node.start_sec != null && <span>{clock(node.start_sec)}–{clock(node.end_sec)}</span>}{node.page_number && <span>Page {node.page_number}</span>}</div></div></button>;
      })}</div><div className="connections"><div className="section-title"><span>Why these connect</span><small>Explicit evidence edges</small></div>{graph.connections.slice(0, 6).map((edge, index) => <div className="edge" key={`${edge.from_id}-${edge.to_id}-${index}`}><span className="edge-type">{edge.relationship_type.replace("_", " ")}</span><p>{edge.explanation}</p><b>{Math.round(edge.confidence * 100)}%</b></div>)}</div></section>}

    {mode === "baseline" && baseline && <section className="results"><div className="answer-card baseline"><p className="eyebrow">Text-only baseline</p><p>{baseline.answer.split("\n")[0]}</p><small>Fixed text chunks · {baseline.latency_ms.toFixed(1)} ms</small></div><div className="evidence-grid">{baseline.results.map((item) => <article className="evidence-card text-card" key={item.chunk_id}><div className="card-body"><div className="card-meta"><span className="modality pdf_text">text chunk</span><span>{safeScore(item.score)} match</span></div><p>{item.text}</p><div className="provenance"><b>{item.source_filename}</b></div></div></article>)}</div></section>}

    {(detail || detailLoading) && <div className="explorer-backdrop" role="presentation" onClick={() => setDetail(null)}><aside className="explorer" role="dialog" aria-modal="true" aria-label="Evidence Explorer" onClick={(event) => event.stopPropagation()}>
      <div className="explorer-header"><div><p className="eyebrow">Evidence Explorer</p><h2>Selected evidence</h2></div><button type="button" aria-label="Close Evidence Explorer" onClick={() => setDetail(null)}>×</button></div>
      {detailLoading && !detail ? <p>Loading evidence…</p> : detail && <><section className="selected-evidence"><span className={`modality ${detail.node.modality}`}>{detail.node.modality.replace("_", " ")}</span><p>{detail.node.content || "Visual evidence from the source media."}</p><div className="detail-grid"><span><small>Source</small>{String(detail.node.provenance.file ?? detail.node.source_id)}</span>{detail.node.speaker && <span><small>Speaker</small>{detail.node.speaker}</span>}{detail.node.start_sec != null && <span><small>Timestamp</small>{clock(detail.node.start_sec)}–{clock(detail.node.end_sec)}</span>}{detail.node.page_number && <span><small>Page</small>{detail.node.page_number}</span>}{detail.node.frame_path && <span><small>Frame path</small>{detail.node.frame_path}</span>}<span><small>Evidence confidence</small>{signal(detail.node.confidence)}</span></div></section>
        <section className="connected-evidence"><div className="section-title"><span>Connected evidence</span><small>{detail.connected_edges.length} persisted edges</small></div>{detail.connected_edges.map((edge) => { const neighbor = detail.neighbors.find((node) => node.id === edge.neighbor_id); return <article className="connection-detail" key={edge.neighbor_id}><div className="connection-heading"><span className="edge-type">{edge.relationship_type}</span><b>Confidence: {signal(edge.confidence)}</b></div><p className="neighbor-preview"><strong>{neighbor?.modality.replace("_", " ") ?? "Evidence"}</strong> · {neighbor?.content || String(neighbor?.provenance.file ?? edge.neighbor_id)}</p><dl><div><dt>Temporal</dt><dd>{signal(edge.signal_breakdown.temporal)}</dd></div><div><dt>Entity</dt><dd>{signal(edge.signal_breakdown.entity)}</dd></div><div><dt>Semantic</dt><dd>{signal(edge.signal_breakdown.semantic)}</dd></div></dl><small>Why connected</small><p>{edge.explanation}</p></article>; })}</section></>}
    </aside></div>}
  </main>;
}
