"use client";

import React, { useState, useEffect } from "react";
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
  RadioTower
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

const temporalData = [
  { time: "00:00", score: 45 },
  { time: "04:00", score: 52 },
  { time: "08:00", score: 38 },
  { time: "12:00", score: 65 },
  { time: "16:00", score: 72 },
  { time: "20:00", score: 89 },
  { time: "24:00", score: 94 },
];

const radarData = [
  { subject: "Toxicity", A: 85, fullMark: 100 },
  { subject: "Hate Speech", A: 65, fullMark: 100 },
  { subject: "Misinfo", A: 92, fullMark: 100 },
  { subject: "Causal Outrage", A: 99, fullMark: 100 },
  { subject: "Adversarial", A: 45, fullMark: 100 },
  { subject: "Meme Threat", A: 78, fullMark: 100 },
];

const graphNodes = Array.from({ length: 20 }).map(() => ({
  x: Math.random() * 100,
  y: Math.random() * 100,
  z: Math.random() * 200 + 50,
  threat: Math.random() > 0.7 ? "high" : "low"
}));

const LiveScoreDisplay = () => {
  const [liveScore, setLiveScore] = useState(94.2);

  useEffect(() => {
    const interval = setInterval(() => {
      setLiveScore((prev) => {
        const delta = (Math.random() - 0.5) * 2;
        return Math.min(100, Math.max(0, prev + delta));
      });
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="text-7xl font-black text-rose-500 tracking-tighter flex items-baseline">
      {liveScore.toFixed(1)}
      <span className="text-2xl text-rose-900 ml-2">/ 100</span>
    </div>
  );
};

export default function ThreatIntelligenceDashboard() {
  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-50 font-mono flex flex-col selection:bg-rose-500/30">
      {/* Top Navbar */}
      <header className="sticky top-0 z-50 flex h-16 items-center gap-4 border-b border-rose-900/30 bg-zinc-950/80 px-6 backdrop-blur-md">
        <div className="flex items-center gap-2 text-rose-500">
          <BrainCircuit className="h-6 w-6" />
          <h1 className="text-lg font-bold tracking-tighter">PRSI // DOOM-INDEX</h1>
        </div>
        <div className="ml-auto flex items-center gap-4 text-xs">
          <Badge variant="outline" className="border-rose-500/50 text-rose-400 bg-rose-950/20">
            <RadioTower className="w-3 h-3 mr-1 animate-pulse" /> Live Kafka Stream
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
      <main className="flex-1 grid grid-cols-12 gap-6 p-6 overflow-hidden">
        
        {/* Left Column: Multimodal Stream & Graph */}
        <div className="col-span-8 flex flex-col gap-6">
          
          <div className="grid grid-cols-3 gap-6">
            <Card className="col-span-2 bg-zinc-900/50 border-rose-900/20 shadow-2xl">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-zinc-400 flex items-center gap-2">
                  <Network className="w-4 h-4 text-rose-500" />
                  GNN Entity Topology
                </CardTitle>
              </CardHeader>
              <CardContent className="h-[300px]">
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
                  Multimodal Risk Vectors
                </CardTitle>
              </CardHeader>
              <CardContent className="h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart cx="50%" cy="50%" outerRadius="70%" data={radarData}>
                    <PolarGrid stroke="#27272a" />
                    <PolarAngleAxis dataKey="subject" tick={{ fill: '#a1a1aa', fontSize: 10 }} />
                    <Radar name="Threat" dataKey="A" stroke="#f43f5e" fill="#f43f5e" fillOpacity={0.3} />
                  </RadarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          <Card className="flex-1 bg-zinc-900/50 border-rose-900/20 shadow-2xl flex flex-col min-h-0">
            <CardHeader className="pb-4 border-b border-zinc-800">
              <div className="flex justify-between items-center">
                <CardTitle className="text-sm font-medium text-zinc-400 flex items-center gap-2">
                  <Terminal className="w-4 h-4 text-rose-500" />
                  Inference Output Stream
                </CardTitle>
                <div className="flex gap-2">
                  <Badge variant="outline" className="border-rose-500/50 text-rose-400">EXTREME RISK</Badge>
                </div>
              </div>
            </CardHeader>
            <CardContent
              className="flex-1 overflow-y-auto p-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-500 focus-visible:ring-inset"
              tabIndex={0}
              role="region"
              aria-label="Inference Output Stream"
            >
              <div className="divide-y divide-zinc-800/50">
                
                {/* Stream Item 1 */}
                <div className="p-4 hover:bg-zinc-800/20 transition-colors flex gap-4">
                  <div className="w-32 h-24 bg-zinc-800 rounded flex items-center justify-center border border-rose-900/50 shrink-0 relative overflow-hidden">
                     <ImageIcon className="w-8 h-8 text-zinc-600 absolute" />
                     {/* Bounding box mock */}
                     <div className="absolute top-2 left-2 w-16 h-12 border-2 border-rose-500 bg-rose-500/10"></div>
                  </div>
                  <div className="flex-1 space-y-2">
                    <div className="flex justify-between items-start">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-bold text-rose-500">PREDICTION_ID: XF-992-A</span>
                        <Badge variant="secondary" className="bg-rose-950 text-rose-300 text-[10px]">MEME_DETECTED</Badge>
                      </div>
                      <span className="text-xs text-zinc-500">Just now</span>
                    </div>
                    <p className="text-sm text-zinc-300">
                      <MessageSquare className="w-3 h-3 inline mr-2 text-zinc-500"/>
                      &quot;The global elites are collapsing the network tomorrow.&quot;
                    </p>
                    <div className="flex gap-4 text-xs text-zinc-500">
                      <span>Causal Outrage: <strong className="text-rose-400">0.98</strong></span>
                      <span>Toxicity: <strong className="text-amber-400">0.76</strong></span>
                      <span>Source: <strong className="text-zinc-300">t.me/dark_intel</strong></span>
                    </div>
                  </div>
                </div>

                {/* Stream Item 2 */}
                <div className="p-4 hover:bg-zinc-800/20 transition-colors flex gap-4">
                  <div className="w-32 h-24 bg-zinc-800 rounded flex items-center justify-center border border-zinc-700 shrink-0 relative overflow-hidden">
                     <ImageIcon className="w-8 h-8 text-zinc-600 absolute" />
                  </div>
                  <div className="flex-1 space-y-2">
                    <div className="flex justify-between items-start">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-bold text-amber-500">PREDICTION_ID: XF-991-B</span>
                        <Badge variant="secondary" className="bg-amber-950 text-amber-300 text-[10px]">MISINFO_SUSPECTED</Badge>
                      </div>
                      <span className="text-xs text-zinc-500">2m ago</span>
                    </div>
                    <p className="text-sm text-zinc-300">
                      <MessageSquare className="w-3 h-3 inline mr-2 text-zinc-500"/>
                      &quot;They don&apos;t want you to know the truth about the water supply.&quot;
                    </p>
                    <div className="flex gap-4 text-xs text-zinc-500">
                      <span>Causal Outrage: <strong className="text-rose-400">0.82</strong></span>
                      <span>Toxicity: <strong className="text-zinc-400">0.41</strong></span>
                      <span>Source: <strong className="text-zinc-300">x.com/anon_user</strong></span>
                    </div>
                  </div>
                </div>

              </div>
            </CardContent>
          </Card>
        </div>

        {/* Right Column: Global Doom Index */}
        <div className="col-span-4 flex flex-col gap-6">
          <Card className="bg-zinc-900/50 border-rose-900/50 shadow-2xl overflow-hidden relative">
            <div className="absolute top-0 right-0 p-4">
              <Zap className="w-6 h-6 text-rose-500 animate-pulse" />
            </div>
            <CardHeader>
              <CardTitle className="text-sm font-medium text-zinc-400 uppercase tracking-widest">
                Global Doom Index
              </CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col items-center justify-center py-8">
              <LiveScoreDisplay />
              <p className="text-sm text-rose-400/80 mt-4 font-medium uppercase tracking-widest text-center">
                Critical Threshold Exceeded
              </p>
            </CardContent>
          </Card>

          <Card className="flex-1 bg-zinc-900/50 border-rose-900/20 shadow-2xl">
            <CardHeader>
              <CardTitle className="text-sm font-medium text-zinc-400 flex items-center gap-2">
                <Activity className="w-4 h-4 text-rose-500" />
                24H Temporal Trend
              </CardTitle>
            </CardHeader>
            <CardContent className="h-[250px] p-0 pl-2">
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

          <Card className="bg-zinc-900/50 border-rose-900/20 shadow-2xl">
            <CardHeader>
              <CardTitle className="text-sm font-medium text-zinc-400">System Status</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-xs text-zinc-500">Triton Inference Server</span>
                <Badge className="bg-emerald-500/10 text-emerald-400 border-emerald-500/20 flex items-center gap-1.5"><span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>ONLINE</Badge>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-zinc-500">Kafka MSK Cluster</span>
                <Badge className="bg-emerald-500/10 text-emerald-400 border-emerald-500/20 flex items-center gap-1.5"><span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" style={{ animationDelay: '150ms' }}></span>ONLINE</Badge>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-zinc-500">Adversarial Generator</span>
                <Badge className="bg-emerald-500/10 text-emerald-400 border-emerald-500/20 flex items-center gap-1.5"><span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" style={{ animationDelay: '300ms' }}></span>ONLINE</Badge>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-zinc-500">Neo4j Graph Store</span>
                <Badge className="bg-emerald-500/10 text-emerald-400 border-emerald-500/20 flex items-center gap-1.5"><span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" style={{ animationDelay: '450ms' }}></span>ONLINE</Badge>
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
}
