import { useState, useEffect, useCallback } from "react";

/*
  CueCatcher Longitudinal Dashboard
  
  Shows communication trends over days and weeks — the patterns
  invisible to real-time observation that reveal developmental progress.
  
  Sections:
    1. Trend summary (this week vs last week)
    2. Daily episode chart (bar chart)  
    3. Hourly heatmap (when does communication happen?)
    4. Session history with replay/export
    5. Learned patterns (behavior dictionary)
*/

const API_BASE = ""; // relative to current host

function useFetch(url) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    setLoading(true);
    fetch(`${API_BASE}${url}`).then(r => r.json()).then(d => { setData(d); setLoading(false); }).catch(() => setLoading(false));
  }, [url]);
  return { data, loading };
}

export default function Dashboard() {
  const [tab, setTab] = useState("trends"); // trends | sessions | patterns
  const { data: summary, loading: summaryLoading } = useFetch("/api/dashboard/summary?days=30");
  const { data: daily } = useFetch("/api/dashboard/daily?days=14");
  const { data: hourly } = useFetch("/api/dashboard/hourly?days=7");
  const { data: sessions } = useFetch("/api/sessions?limit=20");
  const { data: patterns } = useFetch("/api/dashboard/patterns");

  return (
    <div style={ds.wrap}>
      {/* Tab bar */}
      <div style={ds.tabs}>
        {[
          { id: "trends", label: "📈 Trends" },
          { id: "sessions", label: "📹 Sessions" },
          { id: "patterns", label: "🧠 Patterns" },
        ].map(t => (
          <button key={t.id} onClick={() => setTab(t.id)}
            style={{ ...ds.tab, ...(tab === t.id ? ds.tabActive : {}) }}>
            {t.label}
          </button>
        ))}
      </div>

      {tab === "trends" && (
        <div style={ds.section}>
          {/* Trend Cards */}
          {summary && <TrendCards summary={summary} />}

          {/* Daily Chart */}
          {daily?.days?.length > 0 && (
            <>
              <h3 style={ds.sectionTitle}>Daily Communication</h3>
              <DailyChart days={daily.days} />
            </>
          )}

          {/* Hourly Heatmap */}
          {hourly?.hours && Object.keys(hourly.hours).length > 0 && (
            <>
              <h3 style={ds.sectionTitle}>When Does She Communicate?</h3>
              <HourlyHeatmap hours={hourly.hours} />
            </>
          )}

          {summaryLoading && <p style={ds.loading}>Loading dashboard...</p>}
          {!summaryLoading && !summary?.total_sessions && (
            <div style={ds.emptyState}>
              <span style={{ fontSize: 48 }}>📊</span>
              <p>No session data yet. Start observing to see trends here.</p>
            </div>
          )}
        </div>
      )}

      {tab === "sessions" && (
        <div style={ds.section}>
          <h3 style={ds.sectionTitle}>Session History</h3>
          {sessions?.sessions?.length > 0 ? (
            sessions.sessions.map((s, i) => <SessionCard key={i} session={s} />)
          ) : (
            <p style={ds.loading}>No recorded sessions.</p>
          )}
        </div>
      )}

      {tab === "patterns" && (
        <div style={ds.section}>
          <h3 style={ds.sectionTitle}>Learned Communication Patterns</h3>
          <p style={ds.desc}>
            These patterns are learned from your feedback. The more you confirm or
            reject interpretations, the smarter the system gets.
          </p>
          {patterns?.patterns?.length > 0 ? (
            patterns.patterns.map((p, i) => <PatternCard key={i} pattern={p} />)
          ) : (
            <div style={ds.emptyState}>
              <span style={{ fontSize: 48 }}>🧠</span>
              <p>No patterns learned yet. Confirm or reject interpretations during sessions to teach the system.</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Trend Cards ───────────────────────────────────────────────
function TrendCards({ summary }) {
  const trend = summary.communication_trend;
  const trendIcon = trend === "increasing" ? "📈" : trend === "decreasing" ? "📉" : "➡️";
  const trendColor = trend === "increasing" ? "#22c55e" : trend === "decreasing" ? "#f97316" : "#94a3b8";

  return (
    <div style={ds.cardGrid}>
      <div style={ds.trendCard}>
        <div style={{ ...ds.trendValue, color: trendColor }}>{trendIcon}</div>
        <div style={ds.trendLabel}>
          {trend === "increasing" ? "Communication increasing" :
           trend === "decreasing" ? "Communication decreasing" : "Communication stable"}
        </div>
        <div style={ds.trendDetail}>
          This week: {summary.recent_7d_episodes || 0} episodes
          {summary.previous_7d_episodes ? ` (prev: ${summary.previous_7d_episodes})` : ""}
        </div>
      </div>

      <div style={ds.trendCard}>
        <div style={ds.trendValue}>{summary.highest_comm_level || 1}</div>
        <div style={ds.trendLabel}>Highest Comm Level</div>
        <div style={ds.trendDetail}>
          {summary.highest_comm_level >= 3 ? "Directed communication! 🎉" :
           summary.highest_comm_level >= 2 ? "Intentional behavior" : "Pre-intentional"}
        </div>
      </div>

      <div style={ds.trendCard}>
        <div style={{ ...ds.trendValue, color: summary.gaze_alternation_count > 0 ? "#22c55e" : "#94a3b8" }}>
          {summary.gaze_alternation_count || 0}
        </div>
        <div style={ds.trendLabel}>Gaze Alternations</div>
        <div style={ds.trendDetail}>
          {summary.gaze_alternation_count > 0
            ? "Joint attention detected — this is significant!"
            : "No gaze alternation yet"}
        </div>
      </div>

      <div style={ds.trendCard}>
        <div style={ds.trendValue}>{summary.total_sessions || 0}</div>
        <div style={ds.trendLabel}>Sessions (30 days)</div>
      </div>
    </div>
  );
}

// ── Daily Chart ───────────────────────────────────────────────
function DailyChart({ days }) {
  const maxEps = Math.max(...days.map(d => d.total_episodes), 1);

  return (
    <div style={ds.chartWrap}>
      <div style={ds.barChart}>
        {days.map((d, i) => {
          const height = (d.total_episodes / maxEps) * 100;
          const date = new Date(d.date);
          const label = date.toLocaleDateString([], { weekday: "short", day: "numeric" });
          const hasGaze = d.gaze_alternation > 0;

          return (
            <div key={i} style={ds.barCol}>
              <div style={ds.barValue}>{d.total_episodes}</div>
              <div style={{
                ...ds.bar,
                height: `${Math.max(height, 2)}%`,
                background: hasGaze
                  ? "linear-gradient(180deg, #22c55e 0%, #3b82f6 100%)"
                  : "linear-gradient(180deg, #3b82f6 0%, #6366f1 100%)",
              }}>
                {hasGaze && <div style={ds.barGazeDot} title="Gaze alternation detected">✨</div>}
              </div>
              <div style={ds.barLabel}>{label}</div>
            </div>
          );
        })}
      </div>
      <div style={ds.chartLegend}>
        <span style={ds.legendItem}><span style={{ ...ds.legendDot, background: "#3b82f6" }} /> Episodes</span>
        <span style={ds.legendItem}><span style={{ ...ds.legendDot, background: "#22c55e" }} /> With gaze alternation</span>
      </div>
    </div>
  );
}

// ── Hourly Heatmap ────────────────────────────────────────────
function HourlyHeatmap({ hours }) {
  const maxCount = Math.max(...Object.values(hours), 1);

  return (
    <div style={ds.heatmapWrap}>
      <div style={ds.heatmapGrid}>
        {Array.from({ length: 24 }, (_, h) => {
          const count = hours[h] || 0;
          const intensity = count / maxCount;
          const label = h === 0 ? "12a" : h < 12 ? `${h}a` : h === 12 ? "12p" : `${h - 12}p`;

          return (
            <div key={h} style={ds.heatmapCell}>
              <div style={{
                ...ds.heatmapBlock,
                background: count > 0
                  ? `rgba(59, 130, 246, ${0.15 + intensity * 0.85})`
                  : "rgba(51, 65, 85, 0.3)",
              }}>
                {count > 0 && <span style={ds.heatmapCount}>{count}</span>}
              </div>
              <span style={ds.heatmapLabel}>{label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Session Card ──────────────────────────────────────────────
function SessionCard({ session }) {
  const [expanded, setExpanded] = useState(false);
  const date = new Date(session.started_at);

  const exportSession = () => {
    const url = `${API_BASE}/api/sessions/${session.session_id}/export?format=csv`;
    window.open(url, "_blank");
  };

  return (
    <div style={ds.sessCard}>
      <div style={ds.sessHeader} onClick={() => setExpanded(!expanded)}>
        <div>
          <div style={ds.sessDate}>
            {date.toLocaleDateString([], { weekday: "long", month: "short", day: "numeric" })}
          </div>
          <div style={ds.sessTime}>
            {date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
            {" · "}
            {session.duration_minutes} min
            {" · "}
            {session.total_episodes || 0} episodes
            {session.button_presses > 0 && ` · ${session.button_presses} button presses`}
          </div>
        </div>
        <span style={ds.sessChevron}>{expanded ? "▲" : "▼"}</span>
      </div>

      {expanded && (
        <div style={ds.sessBody}>
          {/* State breakdown */}
          {session.state_durations && Object.keys(session.state_durations).length > 0 && (
            <div style={ds.sessStates}>
              <div style={ds.sessStatesLabel}>States:</div>
              {Object.entries(session.state_durations).map(([state, secs]) => (
                <span key={state} style={ds.sessStateChip}>
                  {state}: {Math.round(secs)}s
                </span>
              ))}
            </div>
          )}

          {/* Button usage */}
          {session.button_breakdown && Object.keys(session.button_breakdown).length > 0 && (
            <div style={ds.sessStates}>
              <div style={ds.sessStatesLabel}>Buttons used:</div>
              {Object.entries(session.button_breakdown).map(([btn, count]) => (
                <span key={btn} style={{ ...ds.sessStateChip, background: "#312e81" }}>
                  {btn}: {count}×
                </span>
              ))}
            </div>
          )}

          {/* Actions */}
          <div style={ds.sessActions}>
            <button onClick={exportSession} style={ds.sessActionBtn}>
              📄 Export CSV (for therapist)
            </button>
            {session.has_video && (
              <button onClick={() => window.open(`${API_BASE}/api/sessions/${session.session_id}/video`)}
                style={ds.sessActionBtn}>
                🎬 Watch Video
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Pattern Card ──────────────────────────────────────────────
function PatternCard({ pattern }) {
  const accuracy = pattern.accuracy || 0;
  const total = (pattern.confirmed || 0) + (pattern.rejected || 0);

  return (
    <div style={ds.patternCard}>
      <div style={ds.patternHeader}>
        <span style={ds.patternName}>{pattern.name?.replace(/_/g, " ")}</span>
        <span style={{
          ...ds.patternAccuracy,
          color: accuracy >= 80 ? "#22c55e" : accuracy >= 50 ? "#eab308" : "#94a3b8",
        }}>
          {accuracy}% accurate
        </span>
      </div>
      <p style={ds.patternDesc}>{pattern.description}</p>
      <div style={ds.patternStats}>
        <span>✅ {pattern.confirmed || 0}</span>
        <span>❌ {pattern.rejected || 0}</span>
        <span>Total: {total}</span>
        {pattern.last_seen && (
          <span>Last: {new Date(pattern.last_seen).toLocaleDateString()}</span>
        )}
      </div>
    </div>
  );
}

// ── Styles ────────────────────────────────────────────────────
const ds = {
  wrap: { padding: 12, fontFamily: "'DM Sans', -apple-system, sans-serif", color: "#e2e8f0" },

  tabs: { display: "flex", gap: 0, marginBottom: 14, borderBottom: "1px solid #334155" },
  tab: {
    flex: 1, padding: "9px 0", border: "none", background: "transparent",
    color: "#94a3b8", fontSize: 13, cursor: "pointer", borderBottom: "2px solid transparent",
  },
  tabActive: { color: "#e2e8f0", borderBottom: "2px solid #3b82f6" },

  section: {},
  sectionTitle: { fontSize: 14, fontWeight: 600, margin: "18px 0 8px", color: "#94a3b8", textTransform: "uppercase", letterSpacing: 1 },
  desc: { fontSize: 12, color: "#64748b", lineHeight: 1.5, margin: "0 0 12px" },
  loading: { color: "#64748b", fontSize: 13, textAlign: "center", padding: 24 },
  emptyState: { textAlign: "center", padding: 40, color: "#64748b", fontSize: 14 },

  // Trend cards
  cardGrid: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 16 },
  trendCard: { background: "#1e293b", borderRadius: 12, padding: "14px 16px" },
  trendValue: { fontSize: 28, fontWeight: 800, color: "#e2e8f0", lineHeight: 1 },
  trendLabel: { fontSize: 12, fontWeight: 600, color: "#94a3b8", marginTop: 4 },
  trendDetail: { fontSize: 11, color: "#64748b", marginTop: 4, lineHeight: 1.4 },

  // Bar chart
  chartWrap: { background: "#1e293b", borderRadius: 12, padding: 14, marginBottom: 12 },
  barChart: { display: "flex", alignItems: "flex-end", gap: 4, height: 120 },
  barCol: { flex: 1, display: "flex", flexDirection: "column", alignItems: "center", height: "100%" },
  barValue: { fontSize: 10, color: "#94a3b8", marginBottom: 2 },
  bar: { width: "100%", borderRadius: "4px 4px 0 0", position: "relative", minHeight: 2, transition: "height 0.3s" },
  barGazeDot: { position: "absolute", top: -14, left: "50%", transform: "translateX(-50%)", fontSize: 10 },
  barLabel: { fontSize: 9, color: "#64748b", marginTop: 4, textAlign: "center" },
  chartLegend: { display: "flex", gap: 12, marginTop: 10, justifyContent: "center" },
  legendItem: { fontSize: 10, color: "#94a3b8", display: "flex", alignItems: "center", gap: 4 },
  legendDot: { width: 8, height: 8, borderRadius: 2, display: "inline-block" },

  // Heatmap
  heatmapWrap: { background: "#1e293b", borderRadius: 12, padding: 14, marginBottom: 12 },
  heatmapGrid: { display: "grid", gridTemplateColumns: "repeat(12, 1fr)", gap: 3 },
  heatmapCell: { display: "flex", flexDirection: "column", alignItems: "center" },
  heatmapBlock: { width: "100%", aspectRatio: "1", borderRadius: 4, display: "flex", alignItems: "center", justifyContent: "center" },
  heatmapCount: { fontSize: 9, fontWeight: 700, color: "#fff" },
  heatmapLabel: { fontSize: 8, color: "#64748b", marginTop: 2 },

  // Session cards
  sessCard: { background: "#1e293b", borderRadius: 10, marginBottom: 6, overflow: "hidden" },
  sessHeader: { display: "flex", justifyContent: "space-between", alignItems: "center", padding: "10px 14px", cursor: "pointer" },
  sessDate: { fontSize: 14, fontWeight: 600 },
  sessTime: { fontSize: 11, color: "#94a3b8", marginTop: 2 },
  sessChevron: { fontSize: 11, color: "#64748b" },
  sessBody: { padding: "0 14px 12px", borderTop: "1px solid #334155" },
  sessStates: { display: "flex", flexWrap: "wrap", gap: 4, marginTop: 8, alignItems: "center" },
  sessStatesLabel: { fontSize: 11, color: "#64748b", marginRight: 4 },
  sessStateChip: { fontSize: 10, padding: "2px 8px", background: "#334155", borderRadius: 8, color: "#94a3b8" },
  sessActions: { display: "flex", gap: 8, marginTop: 10 },
  sessActionBtn: {
    flex: 1, padding: "8px 0", borderRadius: 8, border: "1px solid #334155",
    background: "#0f172a", color: "#e2e8f0", fontSize: 12, cursor: "pointer",
  },

  // Pattern cards
  patternCard: { background: "#1e293b", borderRadius: 10, padding: "10px 14px", marginBottom: 6 },
  patternHeader: { display: "flex", justifyContent: "space-between", alignItems: "center" },
  patternName: { fontSize: 14, fontWeight: 600, textTransform: "capitalize" },
  patternAccuracy: { fontSize: 12, fontWeight: 700 },
  patternDesc: { fontSize: 12, color: "#cbd5e1", margin: "4px 0", lineHeight: 1.4 },
  patternStats: { display: "flex", gap: 12, fontSize: 11, color: "#64748b", marginTop: 4 },
};
