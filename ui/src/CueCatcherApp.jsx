import { useState, useEffect, useRef, useCallback } from "react";
import DebugPanel from "./DebugPanel";
import Dashboard from "./Dashboard";
import SkeletonOverlay from "./SkeletonOverlay";

// ── Configuration ─────────────────────────────────────────────
const DEFAULT_COMM_BUTTONS = [
  { id: "want", label: "I want", icon: "🤲", color: "#f59e0b", phrase: "I want something" },
  { id: "stop", label: "Stop", icon: "✋", color: "#ef4444", phrase: "Stop please" },
  { id: "more", label: "More", icon: "🔄", color: "#22c55e", phrase: "More please" },
];

const EXTRA_BUTTONS = [
  { id: "help", label: "Help", icon: "🆘", color: "#8b5cf6", phrase: "I need help" },
  { id: "eat", label: "Eat", icon: "🍽️", color: "#f97316", phrase: "I'm hungry" },
  { id: "drink", label: "Drink", icon: "🥤", color: "#06b6d4", phrase: "I want a drink" },
  { id: "outside", label: "Outside", icon: "🌳", color: "#10b981", phrase: "I want to go outside" },
  { id: "play", label: "Play", icon: "🎈", color: "#ec4899", phrase: "I want to play" },
  { id: "hug", label: "Hug", icon: "🤗", color: "#a855f7", phrase: "I want a hug" },
  { id: "tired", label: "Tired", icon: "😴", color: "#6366f1", phrase: "I'm tired" },
  { id: "hurt", label: "Hurts", icon: "🩹", color: "#dc2626", phrase: "Something hurts" },
  { id: "happy", label: "Happy", icon: "😊", color: "#fbbf24", phrase: "I'm happy" },
  { id: "no", label: "No", icon: "🚫", color: "#b91c1c", phrase: "No, I don't want that" },
  { id: "yes", label: "Yes", icon: "👍", color: "#16a34a", phrase: "Yes please" },
  { id: "music", label: "Music", icon: "🎵", color: "#7c3aed", phrase: "I want music" },
];

const CONFIDENCE_COLORS = { high: "#22c55e", medium: "#eab308", low: "#f97316" };
const INTENT_ICONS = { request: "🤲", reject: "🚫", social: "💛", regulate: "🌊", explore: "🔍" };
const INTENT_LABELS = { request: "Requesting", reject: "Rejecting", social: "Social", regulate: "Self-regulating" };
function getConfLevel(c) { return c >= 0.7 ? "high" : c >= 0.5 ? "medium" : "low"; }

// ── Main App ──────────────────────────────────────────────────
export default function CueCatcherApp() {
  const [connected, setConnected] = useState(false);
  const [sessionActive, setSessionActive] = useState(false);
  const [interpretations, setInterpretations] = useState([]);
  const [currentInterp, setCurrentInterp] = useState(null);
  const [fps, setFps] = useState(0);
  const [frameCount, setFrameCount] = useState(0);
  const [view, setView] = useState("live"); // live | dashboard | settings
  const [commButtons, setCommButtons] = useState(DEFAULT_COMM_BUTTONS);
  const [commMode, setCommMode] = useState(false);
  const [boardSize, setBoardSize] = useState(3);
  const [lastSpoken, setLastSpoken] = useState(null);
  const [speaking, setSpeaking] = useState(false);
  const [serverUrl, setServerUrl] = useState("ws://127.0.0.1:8084/ws/stream");
  const [ttsEnabled, setTtsEnabled] = useState(true);
  const [childState, setChildState] = useState("idle");

  // Debug
  const [debugMode, setDebugMode] = useState(true);
  const [lastDetections, setLastDetections] = useState(null);

  const wsRef = useRef(null);
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const frameTimesRef = useRef([]);
  const audioCtxRef = useRef(null);
  const loopRef = useRef(null);
  const mountedRef = useRef(true);

  // Refs for closures (fix stale state in setTimeout/callbacks)
  const sessionActiveRef = useRef(false);
  const ttsEnabledRef = useRef(true);
  useEffect(() => { sessionActiveRef.current = sessionActive; }, [sessionActive]);
  useEffect(() => { ttsEnabledRef.current = ttsEnabled; }, [ttsEnabled]);

  // ── WebSocket ──
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN ||
        wsRef.current?.readyState === WebSocket.CONNECTING) return;
    const ws = new WebSocket(serverUrl);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => { setConnected(true); };
    ws.onclose = () => { setConnected(false); if (mountedRef.current) setTimeout(connect, 3000); };
    ws.onerror = () => {};
    ws.onmessage = (e) => {
      if (typeof e.data === "string") { return; }
      if (!(e.data instanceof ArrayBuffer)) return;
      const bytes = new Uint8Array(e.data);
      if (bytes.length === 0) return;
      if (bytes[0] === 0x53) { if (ttsEnabledRef.current) playAudio(bytes.slice(1)); return; }
      try {
        const result = JSON.parse(new TextDecoder().decode(bytes));
        setLastDetections(result);  // feed debug panel
        if (result.interpretation) {
          setCurrentInterp(result.interpretation);
          setInterpretations(prev => [result.interpretation, ...prev].slice(0, 50));
        }
        if (result.detections?.child_state) setChildState(result.detections.child_state);
      } catch {}
    };
    wsRef.current = ws;
  }, [serverUrl]);

  // ── Camera ──
  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user", frameRate: { ideal: 30 } },
        audio: { sampleRate: 16000, channelCount: 1, echoCancellation: false, noiseSuppression: false },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => { startFrameLoop(); startAudioCapture(stream); };
      }
    } catch (err) { console.error("Camera:", err); }
  }, []);

  const startFrameLoop = () => {
    const video = videoRef.current;
    if (!video) return;
    const canvas = document.createElement("canvas");
    canvas.width = 1280; canvas.height = 720;
    const ctx = canvas.getContext("2d");
    let sending = false;
    const loop = () => {
      if (!mountedRef.current) return;
      const ws = wsRef.current;
      if (sessionActiveRef.current && ws?.readyState === WebSocket.OPEN && video.readyState >= 2 && !sending) {
        sending = true;
        ctx.drawImage(video, 0, 0, 1280, 720);
        canvas.toBlob((blob) => {
          if (!blob || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) { sending = false; return; }
          blob.arrayBuffer().then((buf) => {
            const msg = new Uint8Array(buf.byteLength + 1);
            msg[0] = 0x56; msg.set(new Uint8Array(buf), 1);
            try { wsRef.current.send(msg); } catch {}
            const now = performance.now();
            frameTimesRef.current = frameTimesRef.current.filter(t => now - t < 1000);
            frameTimesRef.current.push(now);
            setFps(frameTimesRef.current.length);
            setFrameCount(c => c + 1);
            sending = false;
          });
        }, "image/jpeg", 0.7);
      }
      loopRef.current = setTimeout(loop, 100);
    };
    loop();
  };

  const startAudioCapture = (stream) => {
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      audioCtxRef.current = ctx;
      const src = ctx.createMediaStreamSource(stream);
      const proc = ctx.createScriptProcessor(8192, 1, 1);
      proc.onaudioprocess = (e) => {
        if (!sessionActiveRef.current) return;
        const ws = wsRef.current;
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        const f32 = e.inputBuffer.getChannelData(0);
        const i16 = new Int16Array(f32.length);
        for (let i = 0; i < f32.length; i++) i16[i] = Math.max(-32768, Math.min(32767, Math.round(f32[i] * 32768)));
        const msg = new Uint8Array(i16.buffer.byteLength + 1);
        msg[0] = 0x41; msg.set(new Uint8Array(i16.buffer), 1);
        try { ws.send(msg); } catch {}
      };
      src.connect(proc); proc.connect(ctx.destination);
    } catch {}
  };

  const playAudio = (wavBytes) => {
    try {
      const url = URL.createObjectURL(new Blob([wavBytes], { type: "audio/wav" }));
      const a = new Audio(url); a.play().catch(() => {}); a.onended = () => URL.revokeObjectURL(url);
    } catch {}
  };

  const speakButton = (btn) => {
    setSpeaking(true); setLastSpoken(btn);
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: "speak", text: btn.phrase, button_id: btn.id }));
      wsRef.current.send(JSON.stringify({ action: "log_comm", type: "button_press", button_id: btn.id, phrase: btn.phrase, timestamp: Date.now() }));
    }
    setTimeout(() => setSpeaking(false), 2000);
  };

  const toggleSession = () => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const next = !sessionActive;
    ws.send(JSON.stringify({ action: next ? "start" : "stop" }));
    setSessionActive(next);
  };

  const sendFeedback = (id, fb) => {
    if (wsRef.current?.readyState === WebSocket.OPEN)
      wsRef.current.send(JSON.stringify({ action: "feedback", interpretation_id: id, feedback: fb }));
    setInterpretations(prev => prev.map(i => i.id === id ? { ...i, _fb: fb } : i));
  };

  useEffect(() => {
    mountedRef.current = true;
    connect(); startCamera();
    return () => { mountedRef.current = false; if (loopRef.current) clearTimeout(loopRef.current);
      wsRef.current?.close(); streamRef.current?.getTracks().forEach(t => t.stop()); audioCtxRef.current?.close().catch(() => {}); };
  }, []);

  // ── AAC Full-screen ──
  if (commMode) {
    const buttons = boardSize <= 3 ? commButtons.slice(0, 3) : boardSize <= 6 ? [...commButtons, ...EXTRA_BUTTONS].slice(0, 6) : [...commButtons, ...EXTRA_BUTTONS].slice(0, 12);
    return (
      <div style={s.aacFullscreen}>
        <div style={s.aacTopBar}>
          <button onClick={() => setCommMode(false)} style={s.aacBackBtn}>← Back</button>
          <span style={s.aacTitle}>Talk</span>
          <div style={s.aacSizeControls}>
            {[3, 6, 12].map(n => <button key={n} onClick={() => setBoardSize(n)} style={{ ...s.aacSizeBtn, ...(boardSize === n ? s.aacSizeBtnActive : {}) }}>{n}</button>)}
          </div>
        </div>
        {lastSpoken && speaking && (
          <div style={s.aacSpokenFeedback}><span style={{ fontSize: 28 }}>{lastSpoken.icon}</span><span style={{ fontSize: 18, fontWeight: 600, color: "#e2e8f0" }}>{lastSpoken.phrase}</span></div>
        )}
        <div style={{ ...s.aacGrid, gridTemplateColumns: boardSize <= 3 ? "1fr" : boardSize <= 6 ? "1fr 1fr" : "repeat(3, 1fr)" }}>
          {buttons.map(btn => (
            <button key={btn.id} onClick={() => speakButton(btn)} style={{ ...s.aacButton, background: btn.color, boxShadow: lastSpoken?.id === btn.id && speaking ? `0 0 0 6px ${btn.color}44` : `0 4px 16px ${btn.color}33` }}>
              <span style={{ fontSize: boardSize <= 3 ? 72 : boardSize <= 6 ? 56 : 40, filter: "drop-shadow(0 3px 6px rgba(0,0,0,0.35))" }}>{btn.icon}</span>
              <span style={{ fontWeight: 800, color: "#fff", marginTop: 8, fontSize: boardSize <= 3 ? 32 : boardSize <= 6 ? 24 : 18, textShadow: "0 2px 4px rgba(0,0,0,0.35)" }}>{btn.label}</span>
            </button>
          ))}
        </div>
      </div>
    );
  }

  // ── Main UI ──
  return (
    <div style={s.container}>
      {/* Header */}
      <header style={s.header}>
        <div style={s.hdrLeft}>
          <span style={{ fontSize: 22 }}>🧭</span>
          <h1 style={{ fontSize: 16, fontWeight: 700, margin: 0, letterSpacing: 2 }}>CueCatcher</h1>
          <span style={{ width: 8, height: 8, borderRadius: "50%", marginLeft: 6, background: connected ? "#22c55e" : "#ef4444" }} />
        </div>
        <div style={s.hdrRight}>
          <button onClick={() => setDebugMode(!debugMode)} style={{ background: debugMode ? "#3b82f6" : "#334155", border: "none", color: "#fff", fontSize: 10, padding: "2px 8px", borderRadius: 4, cursor: "pointer" }}>
            {debugMode ? "🔧 DBG" : "🔧"}
          </button>
          {sessionActive && <span style={{ ...s.stat, color: "#22c55e" }}>● LIVE</span>}
          <span style={s.stat}>{fps} fps</span>
          <span style={s.stat}>#{frameCount}</span>
        </div>
      </header>

      {/* Nav — 4 tabs */}
      <nav style={s.nav}>
        {[
          { id: "live", label: "📹 Live" },
          { id: "talk", label: "💬 Talk" },
          { id: "dashboard", label: "📊 Dashboard" },
          { id: "settings", label: "⚙️ Settings" },
        ].map(v => (
          <button key={v.id}
            onClick={() => v.id === "talk" ? setCommMode(true) : setView(v.id)}
            style={{ ...s.navBtn, ...(view === v.id && v.id !== "talk" ? s.navActive : {}), ...(v.id === "talk" ? s.navTalk : {}) }}>
            {v.label}
          </button>
        ))}
      </nav>

      {/* Content */}
      <main style={s.main}>

        {/* ── LIVE VIEW ── */}
        {view === "live" && (
          <div style={s.liveWrap}>
            <div style={s.videoBox}>
              <video ref={videoRef} autoPlay playsInline muted style={s.video} />

              <SkeletonOverlay detections={lastDetections?.detections} />

              {/* Debug overlay */}
              {debugMode && <DebugPanel data={lastDetections} fps={fps} frameCount={frameCount} />}

              {/* Interpretation overlay (shows above debug when triggered) */}
              {currentInterp && !debugMode && (
                <div style={{ ...s.overlay, borderColor: CONFIDENCE_COLORS[getConfLevel(currentInterp.confidence)] }}>
                  <div style={s.ovHeader}>
                    <span style={{ fontSize: 18 }}>{INTENT_ICONS[currentInterp.intent] || "❓"}</span>
                    <span style={{ fontWeight: 600, fontSize: 13 }}>{INTENT_LABELS[currentInterp.intent]}</span>
                    <span style={{ ...s.ovBadge, background: CONFIDENCE_COLORS[getConfLevel(currentInterp.confidence)] }}>{Math.round(currentInterp.confidence * 100)}%</span>
                  </div>
                  <p style={s.ovDesc}>{currentInterp.description}</p>
                  {currentInterp.alternatives?.[0] && <p style={s.ovAlt}>Also possible: {currentInterp.alternatives[0]}</p>}
                </div>
              )}

              <button onClick={toggleSession} style={{ ...s.sessBtn, background: sessionActive ? "#ef4444" : "#22c55e" }}>
                {sessionActive ? "⏹ Stop" : "▶ Start"}
              </button>
            </div>

            {/* Mini AAC board */}
            <div style={s.miniBoard}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                <span style={{ fontSize: 12, fontWeight: 600, color: "#94a3b8", textTransform: "uppercase", letterSpacing: 1 }}>Quick Talk</span>
                <button onClick={() => setCommMode(true)} style={s.expandBtn}>Expand ↗</button>
              </div>
              <div style={s.miniBoardGrid}>
                {commButtons.slice(0, 3).map(btn => (
                  <button key={btn.id} onClick={() => speakButton(btn)} style={{ ...s.miniBtn, background: btn.color, boxShadow: lastSpoken?.id === btn.id && speaking ? `0 0 0 4px ${btn.color}55` : `0 2px 8px ${btn.color}33` }}>
                    <span style={{ fontSize: 28, filter: "drop-shadow(0 2px 4px rgba(0,0,0,0.3))" }}>{btn.icon}</span>
                    <span style={{ fontSize: 13, fontWeight: 700, color: "#fff", marginTop: 4, textShadow: "0 1px 3px rgba(0,0,0,0.3)" }}>{btn.label}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Interpretation feed */}
            <div style={{ padding: 12 }}>
              <h3 style={s.feedTitle}>Recent</h3>
              {interpretations.slice(0, 8).map(i => <InterpCard key={i.id} i={i} onFb={sendFeedback} />)}
              {interpretations.length === 0 && <p style={s.empty}>{sessionActive ? "Analyzing... waiting for signals." : "Press ▶ Start to begin."}</p>}
            </div>
          </div>
        )}

        {/* ── DASHBOARD VIEW ── */}
        {view === "dashboard" && <Dashboard />}

        {/* ── SETTINGS VIEW ── */}
        {view === "settings" && (
          <SettingsView
            serverUrl={serverUrl} onUrlChange={setServerUrl}
            ttsEnabled={ttsEnabled} onTtsToggle={setTtsEnabled}
            commButtons={commButtons} onButtonsChange={setCommButtons}
          />
        )}
      </main>
    </div>
  );
}

// ── Interpretation Card ──
function InterpCard({ i, onFb }) {
  const lv = getConfLevel(i.confidence);
  return (
    <div style={{ ...s.card, borderLeft: `4px solid ${CONFIDENCE_COLORS[lv]}`, ...(i._fb === "confirmed" ? { background: "#0f2a1a" } : i._fb === "rejected" ? { background: "#2a0f0f" } : {}) }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <span>{INTENT_ICONS[i.intent] || "❓"}</span>
        <span style={{ fontWeight: 600, fontSize: 12 }}>{INTENT_LABELS[i.intent]}</span>
        <span style={{ marginLeft: "auto", padding: "1px 6px", borderRadius: 8, fontSize: 10, fontWeight: 700, color: "#000", background: CONFIDENCE_COLORS[lv] }}>{Math.round(i.confidence * 100)}%</span>
      </div>
      <p style={{ margin: "3px 0", fontSize: 12, color: "#cbd5e1", lineHeight: 1.4 }}>{i.description}</p>
      {!i._fb ? (
        <div style={{ display: "flex", gap: 6, marginTop: 6 }}>
          <button onClick={() => onFb(i.id, "confirmed")} style={s.fbBtn}>✅ Correct</button>
          <button onClick={() => onFb(i.id, "rejected")} style={{ ...s.fbBtn, borderColor: "#7f1d1d" }}>❌ Wrong</button>
        </div>
      ) : <span style={{ fontSize: 11, color: "#64748b", marginTop: 4, display: "block" }}>{i._fb === "confirmed" ? "✅ Confirmed" : "❌ Rejected"}</span>}
    </div>
  );
}

// ── Settings ──
function SettingsView({ serverUrl, onUrlChange, ttsEnabled, onTtsToggle, commButtons, onButtonsChange }) {
  const [voiceFile, setVoiceFile] = useState(null);
  const uploadVoice = async () => {
    if (!voiceFile) return;
    const fd = new FormData(); fd.append("file", voiceFile);
    const base = serverUrl.replace("ws://", "http://").replace("/ws/stream", "");
    await fetch(`${base}/api/voice/upload`, { method: "POST", body: fd }).catch(() => {});
  };
  const allButtons = [...DEFAULT_COMM_BUTTONS, ...EXTRA_BUTTONS];
  return (
    <div style={{ padding: 14, color: "#e2e8f0" }}>
      <h3 style={s.setTitle}>Connection</h3>
      <label style={{ fontSize: 12, color: "#94a3b8", display: "block", marginBottom: 4 }}>Server URL</label>
      <input style={s.input} value={serverUrl} onChange={e => onUrlChange(e.target.value)} />

      <h3 style={s.setTitle}>Voice Cloning</h3>
      <p style={s.setDesc}>Upload 5–25 seconds of a parent speaking naturally.</p>
      <input type="file" accept="audio/*" onChange={e => setVoiceFile(e.target.files?.[0])} style={{ marginBottom: 6, fontSize: 12 }} />
      {voiceFile && <button onClick={uploadVoice} style={s.uploadBtn}>🎤 Upload Voice</button>}

      <h3 style={s.setTitle}>Speech Output</h3>
      <label style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 13, padding: "6px 0" }}>
        <span>Speak interpretations aloud</span>
        <button onClick={() => onTtsToggle(!ttsEnabled)} style={{ ...s.toggle, background: ttsEnabled ? "#22c55e" : "#475569" }}>
          <div style={{ width: 20, height: 20, borderRadius: "50%", background: "#fff", position: "absolute", top: 2, transition: "transform 0.2s", transform: ttsEnabled ? "translateX(24px)" : "translateX(2px)" }} />
        </button>
      </label>

      <h3 style={s.setTitle}>Communication Buttons</h3>
      <p style={s.setDesc}>Choose 3 for the main screen. Full board always in Talk tab.</p>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 6, marginBottom: 12 }}>
        {allButtons.map(btn => {
          const active = commButtons.some(b => b.id === btn.id);
          return (
            <button key={btn.id} onClick={() => {
              if (active) { if (commButtons.length > 1) onButtonsChange(commButtons.filter(b => b.id !== btn.id)); }
              else if (commButtons.length < 3) onButtonsChange([...commButtons, btn]);
            }} style={{ display: "flex", flexDirection: "column", alignItems: "center", padding: "10px 4px", borderRadius: 10, cursor: "pointer", background: active ? btn.color : "#1e293b", opacity: active ? 1 : 0.5, border: active ? `2px solid ${btn.color}` : "2px solid #334155", position: "relative" }}>
              <span style={{ fontSize: 28 }}>{btn.icon}</span>
              <span style={{ fontSize: 12, marginTop: 4, color: "#e2e8f0" }}>{btn.label}</span>
              {active && <span style={{ position: "absolute", top: 4, right: 4, fontSize: 10, background: "#fff", color: "#000", borderRadius: "50%", width: 16, height: 16, display: "flex", alignItems: "center", justifyContent: "center", fontWeight: 700 }}>✓</span>}
            </button>
          );
        })}
      </div>

      <h3 style={s.setTitle}>Privacy</h3>
      <p style={s.setDesc}>All processing happens locally. No data leaves your network.</p>
    </div>
  );
}

// ── Styles ──
const s = {
  container: { fontFamily: "'DM Sans', -apple-system, sans-serif", background: "#0f172a", color: "#e2e8f0", minHeight: "100vh", display: "flex", flexDirection: "column", maxWidth: 900, margin: "0 auto" },
  header: { display: "flex", justifyContent: "space-between", alignItems: "center", padding: "10px 16px", background: "#1e293b", borderBottom: "1px solid #334155" },
  hdrLeft: { display: "flex", alignItems: "center", gap: 8 },
  hdrRight: { display: "flex", gap: 10, alignItems: "center" },
  stat: { fontSize: 11, color: "#94a3b8" },
  nav: { display: "flex", background: "#1e293b", borderBottom: "1px solid #334155" },
  navBtn: { flex: 1, padding: "10px 0", border: "none", background: "transparent", color: "#94a3b8", fontSize: 13, cursor: "pointer", borderBottom: "2px solid transparent" },
  navActive: { color: "#e2e8f0", borderBottom: "2px solid #3b82f6" },
  navTalk: { background: "linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)", color: "#fff", fontWeight: 700, fontSize: 14, letterSpacing: 0.5 },
  main: { flex: 1, overflow: "auto" },
  liveWrap: { display: "flex", flexDirection: "column" },
  videoBox: { position: "relative", background: "#000", aspectRatio: "16/9" },
  video: { width: "100%", height: "100%", objectFit: "cover" },
  overlay: { position: "absolute", bottom: 56, left: 10, right: 10, background: "rgba(15,23,42,0.92)", backdropFilter: "blur(8px)", borderRadius: 12, padding: "10px 14px", border: "2px solid", zIndex: 50 },
  ovHeader: { display: "flex", alignItems: "center", gap: 6 },
  ovBadge: { marginLeft: "auto", padding: "2px 8px", borderRadius: 12, fontSize: 11, fontWeight: 700, color: "#000" },
  ovDesc: { margin: "4px 0 0", fontSize: 12, color: "#cbd5e1", lineHeight: 1.4 },
  ovAlt: { margin: "2px 0 0", fontSize: 10, color: "#64748b", fontStyle: "italic" },
  sessBtn: { position: "absolute", top: 10, right: 10, padding: "8px 18px", borderRadius: 20, border: "none", color: "#fff", fontWeight: 700, fontSize: 13, cursor: "pointer", zIndex: 200 },
  miniBoard: { padding: "8px 12px 12px", background: "#1e293b", borderBottom: "1px solid #334155" },
  expandBtn: { background: "none", border: "1px solid #475569", color: "#94a3b8", fontSize: 11, padding: "3px 10px", borderRadius: 6, cursor: "pointer" },
  miniBoardGrid: { display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8 },
  miniBtn: { display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "12px 4px", borderRadius: 14, border: "none", cursor: "pointer", minHeight: 70 },
  feedTitle: { fontSize: 12, fontWeight: 600, margin: "0 0 8px", color: "#64748b", textTransform: "uppercase", letterSpacing: 1 },
  empty: { color: "#475569", fontSize: 13, textAlign: "center", padding: 32 },
  card: { background: "#1e293b", borderRadius: 10, padding: "10px 12px", marginBottom: 6 },
  fbBtn: { flex: 1, padding: "7px 0", borderRadius: 8, border: "1px solid #334155", background: "#0f172a", color: "#e2e8f0", fontSize: 12, cursor: "pointer" },
  setTitle: { fontSize: 15, fontWeight: 600, margin: "18px 0 6px", color: "#e2e8f0" },
  setDesc: { fontSize: 12, color: "#94a3b8", lineHeight: 1.5, margin: "0 0 10px" },
  input: { width: "100%", padding: "9px 11px", borderRadius: 8, border: "1px solid #334155", background: "#0f172a", color: "#e2e8f0", fontSize: 13, boxSizing: "border-box" },
  uploadBtn: { padding: "9px 18px", borderRadius: 8, border: "none", background: "#3b82f6", color: "#fff", fontWeight: 600, fontSize: 13, cursor: "pointer", marginTop: 6 },
  toggle: { width: 48, height: 24, borderRadius: 12, border: "none", cursor: "pointer", position: "relative" },
  aacFullscreen: { position: "fixed", inset: 0, background: "#0f172a", display: "flex", flexDirection: "column", zIndex: 9999 },
  aacTopBar: { display: "flex", alignItems: "center", justifyContent: "space-between", padding: "8px 14px", background: "#1e293b", borderBottom: "1px solid #334155" },
  aacBackBtn: { background: "none", border: "1px solid #475569", color: "#e2e8f0", fontSize: 14, padding: "6px 14px", borderRadius: 8, cursor: "pointer" },
  aacTitle: { fontSize: 18, fontWeight: 700, color: "#e2e8f0" },
  aacSizeControls: { display: "flex", gap: 4 },
  aacSizeBtn: { width: 32, height: 32, borderRadius: 8, border: "1px solid #475569", background: "transparent", color: "#94a3b8", fontSize: 14, fontWeight: 700, cursor: "pointer" },
  aacSizeBtnActive: { background: "#3b82f6", color: "#fff", borderColor: "#3b82f6" },
  aacSpokenFeedback: { display: "flex", alignItems: "center", justifyContent: "center", gap: 12, padding: "10px 16px", background: "#1e293b", borderBottom: "1px solid #334155" },
  aacGrid: { flex: 1, display: "grid", gap: 10, padding: 10, alignContent: "stretch" },
  aacButton: { display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", borderRadius: 20, border: "none", cursor: "pointer", minHeight: 0 },
};
