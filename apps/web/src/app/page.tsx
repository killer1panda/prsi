"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Activity,
  Network,
  Image as ImageIcon,
  MessageSquare,
  ShieldAlert,
  Terminal,
  Zap,
  BrainCircuit,
  DatabaseZap,
  RadioTower,
  Search,
  Loader2,
  Wifi,
  WifiOff,
  AlertTriangle,
} from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  Radar,
  ScatterChart,
  Scatter,
  ZAxis
} from "recharts";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "doom_dev_key";

const temporalData = [
  { time: "00:00", score: 45 },
  { time: "04:00", score: 52 },
  { time: "08:00", score: 38 },
  { time: "12:00", score: 65 },
  { time: "16:00", score: 72 },
  { time: "20:00", score: 89 },
  { time: "24:00", score: 94 },
];

const defaultRadarData = [
  { subject: "Toxicity", A: 0, fullMark: 100 },
  { subject: "Outrage", A: 0, fullMark: 100 },
  { subject: "Panic", A: 0, fullMark: 100 },
  { subject: "Cynicism", A: 0, fullMark: 100 },
  { subject: "Adversarial", A: 0, fullMark: 100 },
  { subject: "Irony", A: 0, fullMark: 100 },
];

const graphNodes = Array.from({ length: 20 }).map(() => ({
  x: Math.random() * 100,
  y: Math.random() * 100,
  z: Math.random() * 200 + 50,
  threat: Math.random() > 0.7 ? "high" : "low"
}));

function getRiskColor(score: number): string {
  if (score >= 80) return "text-rose-500";
  if (score >= 60) return "text-amber-500";
  if (score >= 40) return "text-yellow-500";
  return "text-emerald-500";
}

function getRiskBadgeStyle(level: string): string {
  const l = (level || "").toLowerCase();
  if (l.includes("critical") || l.includes("extreme")) return "bg-rose-950 text-rose-300 border-rose-500/50";
  if (l.includes("high")) return "bg-amber-950 text-amber-300 border-amber-500/50";
  if (l.includes("medium")) return "bg-yellow-950 text-yellow-300 border-yellow-500/50";
  return "bg-emerald-950 text-emerald-300 border-emerald-500/50";
}

interface AnalysisResult {
  doom_score: number;
  risk_level: string;
  confidence: number;
  sentiment: { compound: number; overall: string };
  emoji_analysis: Record<string, number | boolean>;
  word_attribution: Array<{ word: string; attribution: number; direction: string }>;
  counterfactual_rewrites: Array<{ variant_id: number; rewritten_text: string; doom_score: number; doom_reduction: number }>;
}

interface LiveEvent {
  id: number;
  text: string;
  source: string;
  doom_score: number;
  risk_level: string;
  timestamp: number;
}

const LiveScoreDisplay = ({ score }: { score: number }) => {
  return (
    <div className={`text-7xl font-black tracking-tighter flex items-baseline ${getRiskColor(score)}`}>
      {score.toFixed(1)}
      <span className="text-2xl text-zinc-500 ml-2">/ 100</span>
    </div>
  );
};

const LiveFeedPanel = React.memo(function LiveFeedPanel() {
  const [events, setEvents] = useState<LiveEvent[]>([]);
  const [sseStatus, setSseStatus] = useState<"connecting" | "connected" | "disconnected">("connecting");
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    const es = new EventSource(`${API_BASE}/events`);
    esRef.current = es;

    es.onopen = () => setSseStatus("connected");

    es.onmessage = (e) => {
      try {
        const data: LiveEvent = JSON.parse(e.data);
        setEvents((prev) => [data, ...prev].slice(0, 8));
      } catch { /* ignore malformed */ }
    };

    es.onerror = () => {
      setSseStatus("disconnected");
      es.close();
      // Retry after 8 seconds
      setTimeout(() => {
        if (esRef.current === es) {
          setSseStatus("connecting");
          const newEs = new EventSource(`${API_BASE}/events`);
          esRef.current = newEs;
          newEs.onopen = () => setSseStatus("connected");
          newEs.onmessage = es.onmessage;
          newEs.onerror = es.onerror;
        }
      }, 8000);
    };

    return () => {
      es.close();
      esRef.current = null;
    };
  }, []);

  return (
    <Card className="bg-zinc-900/50 border-rose-900/20 shadow-2xl">
      <CardHeader className="pb-2 border-b border-zinc-800">
        <div className="flex justify-between items-center">
          <CardTitle className="text-sm font-medium text-zinc-400 flex items-center gap-2">
            <RadioTower className="w-4 h-4 text-rose-500 animate-pulse" />
            Live Social Threat Feed
          </CardTitle>
          <div className="flex items-center gap-1.5 text-xs">
            {sseStatus === "connected" ? (
              <><Wifi className="w-3 h-3 text-emerald-400" /><span className="text-emerald-400">LIVE</span></>
            ) : sseStatus === "connecting" ? (
              <><Loader2 className="w-3 h-3 text-amber-400 animate-spin" /><span className="text-amber-400">CONNECTING</span></>
            ) : (
              <><WifiOff className="w-3 h-3 text-zinc-500" /><span className="text-zinc-500">OFFLINE</span></>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-0 max-h-[280px] overflow-y-auto">
        {events.length === 0 ? (
          <div className="p-6 text-center text-zinc-600 text-xs">
            {sseStatus === "connecting" ? "Connecting to live feed..." : "No events yet. API offline."}
          </div>
        ) : (
          <div className="divide-y divide-zinc-800/50">
            {events.map((ev) => (
              <div key={ev.id} className="p-3 hover:bg-zinc-800/20 transition-colors flex items-start gap-3">
                <span className={`text-lg font-black font-mono shrink-0 ${getRiskColor(ev.doom_score)}`}>
                  {ev.doom_score.toFixed(0)}
                </span>
                <div className="flex-1 min-w-0">
                  <p className="text-xs text-zinc-300 truncate">{ev.text.slice(0, 90)}</p>
                  <p className="text-[10px] text-zinc-600 mt-0.5">{ev.source}</p>
                </div>
                <Badge variant="outline" className={`text-[9px] px-1 py-0 shrink-0 ${getRiskBadgeStyle(ev.risk_level)}`}>
                  {ev.risk_level.replace("RISK_LEVEL_", "")}
                </Badge>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
});

const ThreatAnalyzer = React.memo(function ThreatAnalyzer({ onResult }: { onResult: (r: AnalysisResult) => void }) {
  const [inputText, setInputText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<AnalysisResult | null>(null);

  const handleAnalyze = useCallback(async () => {
    if (!inputText.trim()) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const res = await fetch(`${API_BASE}/analyze/explain`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${API_KEY}`,
        },
        body: JSON.stringify({ text: inputText }),
      });
      if (!res.ok) {
        const errBody = await res.text();
        throw new Error(`API error ${res.status}: ${errBody.slice(0, 100)}`);
      }
      const data: AnalysisResult = await res.json();
      setResult(data);
      onResult(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, [inputText, onResult]);

  return (
    <Card className="bg-zinc-900/50 border-rose-900/20 shadow-2xl">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium text-zinc-400 flex items-center gap-2">
          <Search className="w-4 h-4 text-rose-500" />
          Live Threat Analyzer
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <textarea
          className="w-full h-24 bg-zinc-800 border border-zinc-700 rounded p-3 text-sm text-zinc-200 font-mono resize-none focus:outline-none focus:border-rose-500/60 placeholder:text-zinc-600"
          placeholder="Paste social media text, tweet, or post to analyze..."
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && e.ctrlKey) handleAnalyze(); }}
        />
        <button
          onClick={handleAnalyze}
          disabled={loading || !inputText.trim()}
          className="w-full py-2 px-4 bg-rose-600 hover:bg-rose-700 disabled:bg-zinc-700 disabled:text-zinc-500 text-white text-xs font-bold uppercase tracking-widest rounded transition-colors flex items-center justify-center gap-2"
        >
          {loading ? <><Loader2 className="w-3 h-3 animate-spin" /> Analyzing...</> : <><Zap className="w-3 h-3" /> Analyze Threat</>}
        </button>

        {error && (
          <div className="flex items-center gap-2 text-xs text-amber-400 bg-amber-950/30 border border-amber-800/30 rounded p-2">
            <AlertTriangle className="w-3 h-3 shrink-0" />
            {error}
          </div>
        )}

        {result && (
          <div className="space-y-3 pt-1">
            {/* Score + Risk */}
            <div className="flex items-center justify-between">
              <span className={`text-4xl font-black font-mono ${getRiskColor(result.doom_score)}`}>
                {result.doom_score.toFixed(1)}
              </span>
              <div className="text-right">
                <Badge variant="outline" className={`text-xs ${getRiskBadgeStyle(result.risk_level)}`}>
                  {result.risk_level.replace("RISK_LEVEL_", "")}
                </Badge>
                <p className="text-[10px] text-zinc-500 mt-1">{result.sentiment.overall} sentiment</p>
              </div>
            </div>

            {/* Emoji Analysis */}
            {result.emoji_analysis && (result.emoji_analysis.emoji_count as number) > 0 && (
              <div className="bg-zinc-800/60 rounded p-2 space-y-1">
                <p className="text-[10px] text-zinc-500 uppercase tracking-wider mb-1">Emoji Emotional Swing</p>
                <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-xs">
                  <span className="text-zinc-400">Emoji Count: <strong className="text-zinc-200">{result.emoji_analysis.emoji_count as number}</strong></span>
                  <span className="text-zinc-400">Outrage: <strong className="text-rose-400">{((result.emoji_analysis.outrage_score as number) * 100).toFixed(0)}%</strong></span>
                  <span className="text-zinc-400">Panic: <strong className="text-amber-400">{((result.emoji_analysis.panic_score as number) * 100).toFixed(0)}%</strong></span>
                  <span className="text-zinc-400">Positive: <strong className="text-emerald-400">{((result.emoji_analysis.positive_score as number) * 100).toFixed(0)}%</strong></span>
                  {result.emoji_analysis.irony_flag && (
                    <span className="col-span-2 text-purple-400 font-bold">⚠ Irony/Sarcasm Detected</span>
                  )}
                </div>
              </div>
            )}

            {/* Word Attribution */}
            {result.word_attribution && result.word_attribution.length > 0 && (
              <div className="bg-zinc-800/60 rounded p-2">
                <p className="text-[10px] text-zinc-500 uppercase tracking-wider mb-1.5">Token Attribution (Top 5)</p>
                <div className="flex flex-wrap gap-1">
                  {result.word_attribution.map((w, i) => (
                    <span
                      key={i}
                      className={`text-xs px-1.5 py-0.5 rounded font-mono ${
                        w.direction === "negative"
                          ? "bg-rose-950/60 text-rose-300"
                          : w.direction === "positive"
                          ? "bg-emerald-950/60 text-emerald-300"
                          : "bg-zinc-700/60 text-zinc-400"
                      }`}
                      title={`Attribution: ${w.attribution}`}
                    >
                      {w.word}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Counterfactual Rewrites */}
            {result.counterfactual_rewrites && result.counterfactual_rewrites.length > 0 && (
              <div className="bg-zinc-800/60 rounded p-2 space-y-2">
                <p className="text-[10px] text-zinc-500 uppercase tracking-wider">Counterfactual De-escalations</p>
                {result.counterfactual_rewrites.map((cf) => (
                  <div key={cf.variant_id} className="border border-zinc-700/50 rounded p-2 space-y-1">
                    <p className="text-xs text-zinc-300">{cf.rewritten_text}</p>
                    <div className="flex gap-3 text-[10px]">
                      <span className="text-emerald-400">Score: {cf.doom_score.toFixed(1)}</span>
                      <span className="text-zinc-500">Reduction: -{cf.doom_reduction.toFixed(1)}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
});

export default function ThreatIntelligenceDashboard() {
  const [globalScore, setGlobalScore] = useState(47.3);
  const [radarData, setRadarData] = useState(defaultRadarData);

  // Idle random jitter when no analysis running
  useEffect(() => {
    const interval = setInterval(() => {
      setGlobalScore((prev) => {
        const delta = (Math.random() - 0.5) * 1.5;
        return Math.min(100, Math.max(0, prev + delta));
      });
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  const handleAnalysisResult = useCallback((result: AnalysisResult) => {
    // Update global doom score and radar from analysis result
    setGlobalScore(result.doom_score);
    const em = result.emoji_analysis || {};
    const tox = 1 - (result.sentiment?.compound ?? 0);
    setRadarData([
      { subject: "Toxicity", A: Math.round(tox * 100), fullMark: 100 },
      { subject: "Outrage", A: Math.round(((em.outrage_score as number) || 0) * 100), fullMark: 100 },
      { subject: "Panic", A: Math.round(((em.panic_score as number) || 0) * 100), fullMark: 100 },
      { subject: "Cynicism", A: Math.round(((em.cynicism_score as number) || 0) * 100), fullMark: 100 },
      { subject: "Adversarial", A: result.doom_score > 70 ? 80 : 25, fullMark: 100 },
      { subject: "Irony", A: em.irony_flag ? 90 : 10, fullMark: 100 },
    ]);
  }, []);

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-50 font-mono flex flex-col selection:bg-rose-500/30">
      {/* Top Navbar */}
      <header className="sticky top-0 z-50 flex h-16 items-center gap-4 border-b border-rose-900/30 bg-zinc-950/80 px-6 backdrop-blur-md">
        <div className="flex items-center gap-2 text-rose-500">
          <BrainCircuit className="h-6 w-6" />
          <span className="text-lg font-bold tracking-tighter">PRSI // DOOM-INDEX</span>
        </div>
        <div className="ml-auto flex items-center gap-4 text-xs">
          <Badge variant="outline" className="border-rose-500/50 text-rose-400 bg-rose-950/20">
            <RadioTower className="w-3 h-3 mr-1 animate-pulse" /> Live SSE Stream
          </Badge>
          <Badge variant="outline" className="border-zinc-700 text-zinc-400">
            <DatabaseZap className="w-3 h-3 mr-1" /> Neo4j Connected
          </Badge>
          <div className="flex items-center gap-2 pl-4 border-l border-zinc-800">
            <span className="text-zinc-500">OP_ID:</span>
            <span className="font-bold text-zinc-300">ADMIN-01</span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 grid grid-cols-12 gap-6 p-6">

        {/* Left Column */}
        <div className="col-span-8 flex flex-col gap-6">

          {/* Charts Row */}
          <div className="grid grid-cols-3 gap-6">
            <Card className="col-span-2 bg-zinc-900/50 border-rose-900/20 shadow-2xl">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-zinc-400 flex items-center gap-2">
                  <Network className="w-4 h-4 text-rose-500" />
                  GNN Entity Topology
                </CardTitle>
              </CardHeader>
              <CardContent className="h-[250px]">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                    <XAxis type="number" dataKey="x" hide />
                    <YAxis type="number" dataKey="y" hide />
                    <ZAxis type="number" dataKey="z" range={[50, 400]} />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#09090b', borderColor: '#3f3f46' }} />
                    <Scatter data={graphNodes.filter(n => n.threat === 'high')} fill="#f43f5e" />
                    <Scatter data={graphNodes.filter(n => n.threat === 'low')} fill="#3f3f46" />
                  </ScatterChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card className="col-span-1 bg-zinc-900/50 border-rose-900/20 shadow-2xl">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-zinc-400 flex items-center gap-2">
                  <ShieldAlert className="w-4 h-4 text-rose-500" />
                  Risk Vectors
                </CardTitle>
              </CardHeader>
              <CardContent className="h-[250px]">
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart cx="50%" cy="50%" outerRadius="70%" data={radarData}>
                    <PolarGrid stroke="#27272a" />
                    <PolarAngleAxis dataKey="subject" tick={{ fill: '#a1a1aa', fontSize: 9 }} />
                    <Radar name="Threat" dataKey="A" stroke="#f43f5e" fill="#f43f5e" fillOpacity={0.3} />
                  </RadarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          {/* Live Threat Analyzer */}
          <ThreatAnalyzer onResult={handleAnalysisResult} />

          {/* Live Feed */}
          <LiveFeedPanel />
        </div>

        {/* Right Column */}
        <div className="col-span-4 flex flex-col gap-6">
          {/* Global Doom Index */}
          <Card className="bg-zinc-900/50 border-rose-900/50 shadow-2xl overflow-hidden relative">
            <div className="absolute top-0 right-0 p-4">
              <Zap className="w-6 h-6 text-rose-500 animate-pulse" />
            </div>
            <CardHeader>
              <CardTitle className="text-sm font-medium text-zinc-400 uppercase tracking-widest">
                Global Doom Index
              </CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col items-center justify-center py-6">
              <LiveScoreDisplay score={globalScore} />
              <p className={`text-sm mt-3 font-medium uppercase tracking-widest text-center ${
                globalScore >= 80 ? "text-rose-400/80" : globalScore >= 60 ? "text-amber-400/80" : "text-zinc-500"
              }`}>
                {globalScore >= 80 ? "Critical Threshold Exceeded" : globalScore >= 60 ? "Elevated Risk Detected" : "Monitoring Active"}
              </p>
            </CardContent>
          </Card>

          {/* 24H Trend */}
          <Card className="bg-zinc-900/50 border-rose-900/20 shadow-2xl">
            <CardHeader>
              <CardTitle className="text-sm font-medium text-zinc-400 flex items-center gap-2">
                <Activity className="w-4 h-4 text-rose-500" />
                24H Temporal Trend
              </CardTitle>
            </CardHeader>
            <CardContent className="h-[200px] p-0 pl-2">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={temporalData} margin={{ top: 20, right: 20, bottom: 20, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                  <XAxis dataKey="time" stroke="#52525b" fontSize={10} tickLine={false} axisLine={false} />
                  <YAxis stroke="#52525b" fontSize={10} tickLine={false} axisLine={false} />
                  <Tooltip contentStyle={{ backgroundColor: '#09090b', borderColor: '#3f3f46' }} />
                  <Line type="monotone" dataKey="score" stroke="#f43f5e" strokeWidth={3} dot={{ r: 4, fill: '#f43f5e', strokeWidth: 0 }} />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* System Status */}
          <Card className="bg-zinc-900/50 border-rose-900/20 shadow-2xl">
            <CardHeader>
              <CardTitle className="text-sm font-medium text-zinc-400">System Status</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-xs text-zinc-500">FastAPI Inference Server</span>
                <Badge className="bg-emerald-500/10 text-emerald-400 border-emerald-500/20">ONLINE</Badge>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-zinc-500">Kafka MSK Cluster</span>
                <Badge className="bg-emerald-500/10 text-emerald-400 border-emerald-500/20">ONLINE</Badge>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-zinc-500">Adversarial Generator</span>
                <Badge className="bg-emerald-500/10 text-emerald-400 border-emerald-500/20">ONLINE</Badge>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-zinc-500">Neo4j Graph Store</span>
                <Badge className="bg-emerald-500/10 text-emerald-400 border-emerald-500/20">ONLINE</Badge>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-zinc-500">/analyze/explain</span>
                <Badge className="bg-emerald-500/10 text-emerald-400 border-emerald-500/20">ACTIVE</Badge>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-zinc-500">/events SSE Feed</span>
                <Badge className="bg-emerald-500/10 text-emerald-400 border-emerald-500/20">ACTIVE</Badge>
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
}
