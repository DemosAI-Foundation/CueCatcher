/**
 * SkeletonOverlay - draws pose skeleton, gaze arrow, and expression
 * directly on a canvas overlaid on the video.
 * 
 * Usage: Place inside the videoBox div, after <video>
 *   <SkeletonOverlay detections={lastDetections?.detections} width={1280} height={720} />
 */

import { useRef, useEffect } from "react";

// MediaPipe 33-keypoint skeleton connections
const SKELETON_CONNECTIONS = [
  // Torso
  [11, 12], [11, 23], [12, 24], [23, 24],
  // Left arm
  [11, 13], [13, 15],
  // Right arm
  [12, 14], [14, 16],
  // Left leg
  [23, 25], [25, 27],
  // Right leg
  [24, 26], [26, 28],
  // Face
  [0, 1], [0, 4], [1, 2], [2, 3], [4, 5], [5, 6],
  // Shoulders to ears
  [11, 0], [12, 0],
  // Hands (wrist to pinky/index if available)
  [15, 17], [15, 19], [15, 21],
  [16, 18], [16, 20], [16, 22],
  // Feet
  [27, 29], [27, 31],
  [28, 30], [28, 32],
];

// Keypoint colors by body region
function getKeypointColor(idx) {
  if (idx <= 10) return "#60a5fa";  // face - blue
  if (idx <= 16) return "#34d399";  // arms - green
  if (idx <= 22) return "#a78bfa";  // hands - purple
  if (idx <= 28) return "#fbbf24";  // legs - yellow
  return "#f87171";                  // feet - red
}

function getBoneColor(i, j) {
  if (i <= 10 || j <= 10) return "rgba(96, 165, 250, 0.6)";
  if (i <= 16 || j <= 16) return "rgba(52, 211, 153, 0.6)";
  if (i <= 22 || j <= 22) return "rgba(167, 139, 250, 0.5)";
  if (i <= 28 || j <= 28) return "rgba(251, 191, 36, 0.6)";
  return "rgba(248, 113, 113, 0.5)";
}

export default function SkeletonOverlay({ detections, width = 1280, height = 720 }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !detections) return;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, width, height);

    const kp = detections.pose_keypoints;
    if (!kp || kp.length === 0) return;

    // MediaPipe keypoints are normalized (0-1), scale to canvas
    const isNormalized = kp[0][0] <= 1.5 && kp[0][1] <= 1.5;
    const sx = isNormalized ? width : 1;
    const sy = isNormalized ? height : 1;

    // ── Draw bones ──
    for (const [i, j] of SKELETON_CONNECTIONS) {
      if (i >= kp.length || j >= kp.length) continue;
      const [x1, y1, c1] = kp[i];
      const [x2, y2, c2] = kp[j];
      if (c1 < 0.3 || c2 < 0.3) continue;

      ctx.beginPath();
      ctx.moveTo(x1 * sx, y1 * sy);
      ctx.lineTo(x2 * sx, y2 * sy);
      ctx.strokeStyle = getBoneColor(i, j);
      ctx.lineWidth = 3;
      ctx.lineCap = "round";
      ctx.stroke();
    }

    // ── Draw keypoints ──
    for (let i = 0; i < kp.length && i < 33; i++) {
      const [x, y, conf] = kp[i];
      if (conf < 0.3) continue;

      const px = x * sx;
      const py = y * sy;
      const radius = i <= 10 ? 3 : 5;

      // Outer glow
      ctx.beginPath();
      ctx.arc(px, py, radius + 2, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(0, 0, 0, 0.4)";
      ctx.fill();

      // Inner dot
      ctx.beginPath();
      ctx.arc(px, py, radius, 0, Math.PI * 2);
      ctx.fillStyle = getKeypointColor(i);
      ctx.fill();
    }

    // ── Draw gaze direction arrow ──
    const headYaw = detections.head_yaw || 0;
    const headPitch = detections.head_pitch || 0;
    const nose = kp[0];
    if (nose && nose[2] > 0.3) {
      const noseX = nose[0] * sx;
      const noseY = nose[1] * sy;

      // Arrow from nose in head direction
      const arrowLen = 60;
      const yawRad = (headYaw * Math.PI) / 180;
      const pitchRad = (headPitch * Math.PI) / 180;
      const endX = noseX + Math.sin(yawRad) * arrowLen;
      const endY = noseY + Math.sin(pitchRad) * arrowLen;

      // Arrow shaft
      ctx.beginPath();
      ctx.moveTo(noseX, noseY);
      ctx.lineTo(endX, endY);
      ctx.strokeStyle = "#f59e0b";
      ctx.lineWidth = 3;
      ctx.stroke();

      // Arrow head
      const angle = Math.atan2(endY - noseY, endX - noseX);
      ctx.beginPath();
      ctx.moveTo(endX, endY);
      ctx.lineTo(endX - 12 * Math.cos(angle - 0.4), endY - 12 * Math.sin(angle - 0.4));
      ctx.moveTo(endX, endY);
      ctx.lineTo(endX - 12 * Math.cos(angle + 0.4), endY - 12 * Math.sin(angle + 0.4));
      ctx.strokeStyle = "#f59e0b";
      ctx.lineWidth = 2.5;
      ctx.stroke();

      // Gaze target label
      const target = detections.gaze_target || "";
      if (target && target !== "unknown") {
        ctx.font = "bold 12px sans-serif";
        ctx.fillStyle = "#f59e0b";
        ctx.fillText(`👁 ${target}`, endX + 8, endY - 4);
      }
    }

    // ── Draw face expression label ──
    if (detections.face_detected) {
      const expr = detections.calibrated_expression || detections.expression || "neutral";
      const conf = detections.expression_confidence || 0;
      if (nose && nose[2] > 0.3 && expr !== "neutral") {
        const labelX = nose[0] * sx;
        const labelY = nose[1] * sy - 40;

        // Background
        ctx.font = "bold 13px sans-serif";
        const textWidth = ctx.measureText(expr).width;
        ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
        ctx.beginPath();
        ctx.roundRect(labelX - textWidth / 2 - 6, labelY - 12, textWidth + 12, 20, 6);
        ctx.fill();

        // Text
        ctx.fillStyle = expr === "happy" ? "#22c55e" : expr === "distress" ? "#ef4444" : "#e2e8f0";
        ctx.textAlign = "center";
        ctx.fillText(expr, labelX, labelY + 3);
        ctx.textAlign = "start";
      }
    }

    // ── Draw bounding box ──
    const bbox = detections.person_bbox;
    if (bbox && bbox.length === 4) {
      const [bx1, by1, bx2, by2] = bbox;
      const bsx = isNormalized ? sx : 1;
      const bsy = isNormalized ? sy : 1;
      ctx.strokeStyle = "rgba(59, 130, 246, 0.3)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.strokeRect(bx1 * bsx, by1 * bsy, (bx2 - bx1) * bsx, (by2 - by1) * bsy);
      ctx.setLineDash([]);
    }

    // ── Draw child state badge ──
    const state = detections.child_state;
    if (state && state !== "idle") {
      const stateColors = {
        attending: "#eab308", communicating: "#22c55e", distressed: "#ef4444",
        regulating: "#8b5cf6", withdrawn: "#475569", transitioning: "#f97316",
      };
      const color = stateColors[state] || "#94a3b8";

      ctx.font = "bold 14px sans-serif";
      const sw = ctx.measureText(state.toUpperCase()).width;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.roundRect(10, 10, sw + 20, 28, 6);
      ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.fillText(state.toUpperCase(), 20, 30);
    }

    // ── Draw action indicators ──
    const actions = detections.actions_detected || [];
    actions.forEach((action, i) => {
      if (typeof action !== "object") return;
      const label = action.action || "";
      const conf = action.confidence || 0;

      ctx.font = "bold 12px sans-serif";
      const aw = ctx.measureText(`⚡ ${label}`).width;
      const ay = 48 + i * 28;

      ctx.fillStyle = "rgba(139, 92, 246, 0.8)";
      ctx.beginPath();
      ctx.roundRect(10, ay, aw + 16, 22, 5);
      ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.fillText(`⚡ ${label} (${(conf * 100).toFixed(0)}%)`, 18, ay + 16);
    });

  }, [detections, width, height]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={{
        position: "absolute",
        top: 0, left: 0,
        width: "100%", height: "100%",
        pointerEvents: "none",
        zIndex: 10,
      }}
    />
  );
}
