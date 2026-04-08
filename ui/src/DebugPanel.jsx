/**
 * CueCatcher Debug Panel
 * 
 * Drop-in component that shows real-time detection data overlaid on the video.
 * Shows: pose confidence, head direction, expression, child state, actions, fps.
 * 
 * Usage in CueCatcherApp.jsx:
 *   import DebugPanel from './DebugPanel'
 *   // Inside the videoBox div, after <video>:
 *   {debugMode && <DebugPanel data={lastDetections} fps={fps} frameCount={frameCount} />}
 */

import { useState } from "react";

export default function DebugPanel({ data, fps, frameCount }) {
  const [expanded, setExpanded] = useState(true);

  if (!data) {
    return (
      <div style={d.panel}>
        <div style={d.header} onClick={() => setExpanded(!expanded)}>
          <span style={d.title}>🔧 DEBUG</span>
          <span style={d.waiting}>Waiting for data...</span>
        </div>
      </div>
    );
  }

  const det = data.detections || data;

  // Extract key values
  const poseConf = det.person_confidence ?? 0;
  const headYaw = det.head_yaw ?? 0;
  const headPitch = det.head_pitch ?? 0;
  const gazeTarget = det.gaze_target ?? "—";
  const lookingAtCam = det.looking_at_camera ?? false;
  const expression = det.calibrated_expression || det.expression || "—";
  const exprConf = det.expression_confidence ?? 0;
  const mouth = det.mouth_openness ?? 0;
  const smile = det.smile_score ?? 0;
  const faceDetected = det.face_detected ?? false;
  const childState = det.child_state ?? "idle";
  const stateConf = det.child_state_confidence ?? 0;
  const stateDur = det.child_state_duration_s ?? 0;
  const stateChanged = det.state_changed ?? false;
  const isVocal = det.is_vocalization ?? false;
  const vocalClass = det.vocalization_class ?? "—";
  const vocalConf = det.vocalization_confidence ?? 0;
  const pitch = det.pitch_hz ?? 0;
  const energy = det.energy_db ?? -60;
  const actions = det.actions_detected ?? [];
  const episodes = det.new_episodes ?? [];
  const gazeAlt = det.gaze_alternation;

  // Interpretation (if present)
  const interp = data.interpretation;

  const stateColors = {
    idle: "#64748b", attending: "#eab308", communicating: "#22c55e",
    distressed: "#ef4444", regulating: "#8b5cf6", withdrawn: "#475569",
    transitioning: "#f97316", engaged: "#06b6d4",
  };

  return (
    <div style={{ ...d.panel, maxHeight: expanded ? 600 : 36 }}>
      <div style={d.header} onClick={() => setExpanded(!expanded)}>
        <span style={d.title}>🔧 DEBUG</span>
        <span style={d.fps}>{fps} fps</span>
        <span style={d.frameNum}>#{frameCount}</span>
        <span style={{
          ...d.stateChip,
          background: stateColors[childState] || "#64748b",
        }}>{childState}</span>
        <span style={d.expandArrow}>{expanded ? "▼" : "▲"}</span>
      </div>

      {expanded && (
        <div style={d.body}>
          {/* Row 1: Pose + Gaze */}
          <div style={d.row}>
            <div style={d.section}>
              <div style={d.label}>POSE</div>
              <div style={d.val}>
                <Bar value={poseConf} label="conf" />
                {det.pose_keypoints ? (
                  <span style={d.ok}>✅ {det.num_keypoints || "?"} kpts</span>
                ) : (
                  <span style={d.warn}>❌ no pose</span>
                )}
              </div>
            </div>

            <div style={d.section}>
              <div style={d.label}>GAZE</div>
              <div style={d.val}>
                <span style={d.mono}>yaw: {headYaw.toFixed(1)}° pitch: {headPitch.toFixed(1)}°</span>
              </div>
              <div style={d.val}>
                <span style={d.tag}>{gazeTarget}</span>
                {lookingAtCam && <span style={d.ok}>👀 at camera</span>}
              </div>
              {gazeAlt?.detected && (
                <div style={d.val}>
                  <span style={{ ...d.tag, background: "#166534" }}>
                    ⚡ GAZE ALT ({gazeAlt.shift_count} shifts)
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Row 2: Face + Audio */}
          <div style={d.row}>
            <div style={d.section}>
              <div style={d.label}>FACE</div>
              {faceDetected ? (
                <>
                  <div style={d.val}>
                    <span style={d.tag}>{expression}</span>
                    <span style={d.dim}>{(exprConf * 100).toFixed(0)}%</span>
                  </div>
                  <div style={d.val}>
                    <Bar value={mouth} label="mouth" color="#f59e0b" />
                    <Bar value={smile} label="smile" color="#22c55e" />
                  </div>
                </>
              ) : (
                <span style={d.warn}>❌ no face</span>
              )}
            </div>

            <div style={d.section}>
              <div style={d.label}>AUDIO</div>
              <div style={d.val}>
                {isVocal ? (
                  <>
                    <span style={{ ...d.tag, background: "#7c2d12" }}>{vocalClass}</span>
                    <span style={d.dim}>{(vocalConf * 100).toFixed(0)}%</span>
                  </>
                ) : (
                  <span style={d.dim}>silence</span>
                )}
              </div>
              <div style={d.val}>
                <span style={d.mono}>pitch: {pitch.toFixed(0)}Hz</span>
                <span style={d.mono}>energy: {energy.toFixed(0)}dB</span>
              </div>
            </div>
          </div>

          {/* Row 3: State */}
          <div style={d.row}>
            <div style={d.section}>
              <div style={d.label}>STATE</div>
              <div style={d.val}>
                <span style={{
                  ...d.stateChip,
                  background: stateColors[childState] || "#64748b",
                  fontSize: 14,
                  padding: "3px 12px",
                }}>
                  {childState}
                </span>
                <span style={d.dim}>{(stateConf * 100).toFixed(0)}%</span>
                <span style={d.dim}>{stateDur.toFixed(0)}s</span>
                {stateChanged && <span style={d.ok}>🔄 changed!</span>}
              </div>
            </div>
          </div>

          {/* Row 4: Actions */}
          {actions.length > 0 && (
            <div style={d.row}>
              <div style={d.section}>
                <div style={d.label}>ACTIONS</div>
                {actions.map((a, i) => (
                  <div key={i} style={d.val}>
                    <span style={{ ...d.tag, background: "#4c1d95" }}>{a.action}</span>
                    <span style={d.dim}>{(a.confidence * 100).toFixed(0)}%</span>
                    <span style={d.dim}>{a.comm_signal}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Row 5: Episodes */}
          {episodes.length > 0 && (
            <div style={d.row}>
              <div style={d.section}>
                <div style={d.label}>NEW EPISODES</div>
                {episodes.map((e, i) => (
                  <div key={i} style={d.val}>
                    <span style={{ ...d.tag, background: "#065f46" }}>{e.type}</span>
                    <span style={d.dim}>{e.duration_ms}ms</span>
                    <span style={d.dim}>{(e.confidence * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Row 6: Interpretation */}
          {interp && (
            <div style={{ ...d.row, background: "#14532d" }}>
              <div style={d.section}>
                <div style={d.label}>💡 INTERPRETATION</div>
                <div style={d.val}>
                  <span style={{ ...d.tag, background: "#166534", fontSize: 13 }}>
                    {interp.intent}: {interp.description}
                  </span>
                </div>
                <div style={d.val}>
                  <span style={d.dim}>
                    conf: {(interp.confidence * 100).toFixed(0)}% | 
                    level: {interp.comm_level} | 
                    speak: {interp.should_speak ? "YES" : "no"}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// Mini bar chart component
function Bar({ value, label, color = "#3b82f6" }) {
  const pct = Math.min(100, Math.max(0, (value || 0) * 100));
  return (
    <div style={d.barWrap}>
      <span style={d.barLabel}>{label}</span>
      <div style={d.barTrack}>
        <div style={{ ...d.barFill, width: `${pct}%`, background: color }} />
      </div>
      <span style={d.barVal}>{pct.toFixed(0)}</span>
    </div>
  );
}

// ── Styles ────────────────────────────────────────────────────
const d = {
  panel: {
    position: "absolute", bottom: 0, left: 0, right: 0,
    background: "rgba(0,0,0,0.88)", backdropFilter: "blur(4px)",
    fontFamily: "'JetBrains Mono', 'Consolas', monospace",
    fontSize: 11, color: "#e2e8f0",
    overflow: "hidden", transition: "max-height 0.2s",
    zIndex: 100, borderTop: "1px solid #334155",
  },
  header: {
    display: "flex", alignItems: "center", gap: 8,
    padding: "6px 10px", cursor: "pointer",
    borderBottom: "1px solid #1e293b",
  },
  title: { fontWeight: 700, fontSize: 12 },
  fps: { color: "#22c55e", fontWeight: 600 },
  frameNum: { color: "#64748b" },
  expandArrow: { marginLeft: "auto", color: "#64748b", fontSize: 10 },
  waiting: { color: "#64748b", fontStyle: "italic" },

  body: { padding: "4px 0", maxHeight: 500, overflow: "auto" },
  row: {
    display: "flex", gap: 0, borderBottom: "1px solid #1e293b",
    padding: "4px 10px",
  },
  section: { flex: 1, minWidth: 0 },
  label: { fontSize: 9, fontWeight: 700, color: "#64748b", textTransform: "uppercase", letterSpacing: 1, marginBottom: 2 },
  val: { display: "flex", alignItems: "center", gap: 6, marginBottom: 2, flexWrap: "wrap" },

  tag: {
    padding: "1px 6px", borderRadius: 4, background: "#334155",
    fontSize: 10, fontWeight: 600, whiteSpace: "nowrap",
  },
  stateChip: {
    padding: "1px 8px", borderRadius: 6, fontSize: 10,
    fontWeight: 700, color: "#fff",
  },
  mono: { fontFamily: "monospace", fontSize: 10, color: "#94a3b8" },
  dim: { color: "#64748b", fontSize: 10 },
  ok: { color: "#22c55e", fontSize: 10 },
  warn: { color: "#f97316", fontSize: 10 },

  // Bar chart
  barWrap: { display: "flex", alignItems: "center", gap: 4, minWidth: 80 },
  barLabel: { fontSize: 9, color: "#64748b", width: 32, textAlign: "right" },
  barTrack: { flex: 1, height: 6, background: "#1e293b", borderRadius: 3, overflow: "hidden" },
  barFill: { height: "100%", borderRadius: 3, transition: "width 0.15s" },
  barVal: { fontSize: 9, color: "#94a3b8", width: 20 },
};
