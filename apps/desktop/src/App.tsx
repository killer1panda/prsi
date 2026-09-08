import { useState, useEffect } from "react";
import "./App.css";

const API_BASE = "http://localhost:8000";
const TOKEN = "dummy_token";

function Dashboard() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const loadDashboard = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API_BASE}/dashboard/leaderboard`, {
        headers: { Authorization: `Bearer ${TOKEN}` },
      });
      if (!res.ok) throw new Error("Failed to load dashboard data");
      const json = await res.json();
      setData(json);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDashboard();
  }, []);

  return (
    <div className="view-container">
      <h2>Dashboard & Leaderboard</h2>
      <button onClick={loadDashboard} disabled={loading}>Refresh</button>
      {loading && <p>Loading...</p>}
      {error && <p className="error">{error}</p>}
      {data && (
        <pre className="data-display">{JSON.stringify(data, null, 2)}</pre>
      )}
    </div>
  );
}

function Analyzer() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleAnalyze = async () => {
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const res = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${TOKEN}`,
        },
        body: JSON.stringify({ text, source: "reddit" }),
      });
      if (!res.ok) throw new Error("Analysis failed");
      const json = await res.json();
      setResult(json);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="view-container">
      <h2>Analyzer</h2>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Enter text to analyze..."
        rows={6}
      />
      <button onClick={handleAnalyze} disabled={loading || !text.trim()}>
        {loading ? "Analyzing..." : "Analyze Threat"}
      </button>
      {error && <p className="error">{error}</p>}
      {result && (
        <pre className="data-display">{JSON.stringify(result, null, 2)}</pre>
      )}
    </div>
  );
}

function AttackSimulator() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSimulate = async () => {
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const res = await fetch(`${API_BASE}/attack/simulate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${TOKEN}`,
        },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) throw new Error("Simulation failed");
      const json = await res.json();
      setResult(json);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="view-container">
      <h2>Attack Simulator</h2>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Enter attack vector..."
        rows={6}
      />
      <button onClick={handleSimulate} disabled={loading || !text.trim()}>
        {loading ? "Simulating..." : "Simulate Attack"}
      </button>
      {error && <p className="error">{error}</p>}
      {result && (
        <pre className="data-display">{JSON.stringify(result, null, 2)}</pre>
      )}
    </div>
  );
}

function App() {
  const [activeTab, setActiveTab] = useState("dashboard");

  return (
    <div className="app-layout">
      <nav className="sidebar">
        <h1>DOOM-INDEX</h1>
        <button
          className={activeTab === "dashboard" ? "active" : ""}
          onClick={() => setActiveTab("dashboard")}
        >
          Dashboard
        </button>
        <button
          className={activeTab === "analyzer" ? "active" : ""}
          onClick={() => setActiveTab("analyzer")}
        >
          Analyzer
        </button>
        <button
          className={activeTab === "attack" ? "active" : ""}
          onClick={() => setActiveTab("attack")}
        >
          Attack Simulator
        </button>
      </nav>
      <main className="content">
        {activeTab === "dashboard" && <Dashboard />}
        {activeTab === "analyzer" && <Analyzer />}
        {activeTab === "attack" && <AttackSimulator />}
      </main>
    </div>
  );
}

export default App;
