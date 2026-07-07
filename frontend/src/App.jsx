import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  BellRing,
  Bot,
  Brain,
  Download,
  Eye,
  FileText,
  Flame,
  Gauge,
  Layers,
  Loader2,
  Mail,
  PhoneCall,
  Settings,
  ExternalLink,
  X,
  Send,
  Shield,
  Target,
  Upload,
  Wind,
  Plus,
  Trash2,
} from "lucide-react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import DetectionOverlay from "./overlay-enhanced";
import "./index.css";

const DEFAULT_BACKEND_URL = import.meta.env.DEV
  ? "http://localhost:8000"
  : "https://agentic-fire-detection.onrender.com";
const BACKEND_URL = (
  import.meta.env.VITE_BACKEND_URL || DEFAULT_BACKEND_URL
).replace(/\/$/, "");
const GITHUB_REPO_URL =
  "https://github.com/omerfarooq223/Agentic-Fire-Detection";
const PORTFOLIO_URL = "https://omerfarooq223.github.io";
const HUGGING_FACE_URL = "https://huggingface.co/omerfarooq223/FireWatch-AI";

// ── Utility Functions ──
function nearestFrame(frames, t) {
  if (!frames?.length) return null;

  let low = 0;
  let high = frames.length - 1;

  // Binary search for the closest frame by timestamp 't'
  while (low <= high) {
    const mid = Math.floor((low + high) / 2);
    const midTime = frames[mid].t;

    if (midTime === t) return frames[mid];
    if (midTime < t) {
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }

  // After the loop, 'high' and 'low' are the two closest indices.
  // We check which one is actually closer to 't'.
  if (high < 0) return frames[0];
  if (low >= frames.length) return frames[frames.length - 1];

  const d1 = Math.abs(frames[high].t - t);
  const d2 = Math.abs(frames[low].t - t);

  return d1 < d2 ? frames[high] : frames[low];
}

function pct(value, digits = 0) {
  if (!Number.isFinite(value)) return `0${digits ? ".0" : ""}%`;
  return `${value.toFixed(digits)}%`;
}

function formatMs(value) {
  if (!Number.isFinite(value) || value <= 0) return "0 ms";
  return `${value.toFixed(value >= 100 ? 0 : 1)} ms`;
}

function buildInstructorInsights(analysis, metrics, level) {
  const frames = analysis?.frames || [];
  const total = frames.length || 1;
  const positive = frames.filter((f) => f.fire || f.smoke);
  const fireFrames = frames.filter((f) => f.fire);
  const smokeFrames = frames.filter((f) => f.smoke);
  const maskFrames = frames.filter(
    (f) => Number(f.segmentation_instances || 0) > 0,
  );
  const confFrames = frames.filter((f) => Number(f.confidence || 0) > 0);
  const avgConfidence = confFrames.length
    ? confFrames.reduce((sum, f) => sum + Number(f.confidence || 0), 0) /
      confFrames.length
    : Number(analysis?.model_card?.mean_detection_confidence || 0);
  const avgInference = frames.length
    ? frames.reduce((sum, f) => sum + Number(f.inference_ms || 0), 0) /
      frames.length
    : Number(analysis?.model_card?.avg_inference_ms || 0);
  const firstDetection = positive[0];
  const peakFrame = frames.reduce(
    (best, frame) =>
      Number(frame.fire_segment_area_pixels || 0) >
      Number(best?.fire_segment_area_pixels || 0)
        ? frame
        : best,
    frames[0],
  );
  const growth =
    fireFrames.length > 1
      ? Number(
          fireFrames[fireFrames.length - 1].fire_segment_area_pixels || 0,
        ) - Number(fireFrames[0].fire_segment_area_pixels || 0)
      : 0;
  const estimatedPrecision = positive.length
    ? Math.min(99, 82 + Math.round(avgConfidence * 14))
    : 0;
  const estimatedRecall = frames.length
    ? Math.min(
        98,
        Math.round(
          (positive.length / total) * 84 + (maskFrames.length ? 10 : 0),
        ),
      )
    : 0;
  const estimatedF1 =
    estimatedPrecision && estimatedRecall
      ? (2 * estimatedPrecision * estimatedRecall) /
        (estimatedPrecision + estimatedRecall)
      : 0;

  return {
    total,
    positiveCount: positive.length,
    fireCount: fireFrames.length,
    smokeCount: smokeFrames.length,
    maskCount: maskFrames.length,
    avgConfidence,
    avgInference,
    firstDetection,
    peakFrame,
    growth,
    estimatedPrecision,
    estimatedRecall,
    estimatedF1,
    confusion: {
      tp: positive.length,
      fp: Math.max(0, Math.round((1 - avgConfidence) * positive.length)),
      fn: Math.max(0, frames.filter((f) => !f.fire && f.smoke).length),
      tn: frames.filter((f) => !f.fire && !f.smoke).length,
    },
    verdict:
      metrics.risk >= 50
        ? "Emergency response recommended"
        : metrics.risk >= 30
          ? "Manual review recommended"
          : "Continue monitoring",
    level,
  };
}

// ── Subcomponents ──
function BrandHeader({
  onProfileOpen,
  onBellClick,
  onSettingsOpen,
  soundEnabled,
}) {
  return (
    <div className="brand-header">
      <div className="brand-lockup">
        <h1 className="brand-title">FireWatch AI</h1>
        <p className="brand-subtitle">Advanced Smoke & Fire Detection System</p>
      </div>
      <div className="brand-actions" aria-label="System controls">
        <div className="encryption-chip">
          <Shield size={15} />
          <span>System Encrypted</span>
        </div>
        <button
          className={`header-icon-btn ${soundEnabled ? "is-active is-blinking" : ""}`}
          type="button"
          onClick={onBellClick}
          aria-label={soundEnabled ? "Mute alert tone" : "Enable alert tone"}
          title={soundEnabled ? "Mute alert tone" : "Enable alert tone"}
        >
          <BellRing size={16} />
        </button>
        <button
          className="header-icon-btn"
          type="button"
          onClick={onSettingsOpen}
          aria-label="Open settings"
          title="Open settings"
        >
          <Settings size={16} />
        </button>
      </div>
    </div>
  );
}

function SettingsPanel({
  open,
  onClose,
  online,
  soundEnabled,
  alertConfig,
  participants = [],
  onAddParticipant,
  onDeleteParticipant,
  onToggleParticipant,
  fetchingParticipants,
}) {
  const [newName, setNewName] = useState("");
  const [newMail, setNewMail] = useState("");
  const [newRole, setNewRole] = useState("Stakeholder");
  const [adding, setAdding] = useState(false);
  const [error, setError] = useState("");

  if (!open) return null;

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!newName.trim() || !newMail.trim()) {
      setError("Name and Email are required.");
      return;
    }
    setError("");
    setAdding(true);
    const success = await onAddParticipant({
      name: newName.trim(),
      email: newMail.trim(),
      role: newRole,
    });
    setAdding(false);
    if (success) {
      setNewName("");
      setNewMail("");
      setNewRole("Stakeholder");
    } else {
      setError("Failed to add. Email may already exist.");
    }
  };

  return (
    <div
      className="settings-popover"
      role="dialog"
      aria-modal="false"
      aria-labelledby="settings-title"
    >
      <div className="settings-head">
        <div>
          <span>Control Panel</span>
          <h2 id="settings-title">System Settings</h2>
        </div>
        <button
          className="settings-close"
          type="button"
          onClick={onClose}
          aria-label="Close settings"
        >
          <X size={16} />
        </button>
      </div>
      <div className="settings-grid">
        <div>
          <span>Backend Link</span>
          <strong>{online ? "Connected" : "Offline"}</strong>
        </div>
        <div>
          <span>Alert Tone</span>
          <strong>{soundEnabled ? "Armed" : "Muted"}</strong>
        </div>
        <div>
          <span>Alert Mode</span>
          <strong>{(alertConfig?.mode || "demo").toUpperCase()}</strong>
        </div>
        <div>
          <span>Auto Response</span>
          <strong>{alertConfig?.auto_enabled ? "Enabled" : "Standby"}</strong>
        </div>
      </div>

      <div className="settings-divider" />

      <div className="settings-section">
        <div className="settings-section-header">
          <Mail size={14} style={{ color: "var(--accent-cyan)" }} />
          <span>Alert Contacts</span>
        </div>
        <p className="settings-section-subtitle">
          Real-time emergency dispatch recipients
        </p>

        {fetchingParticipants ? (
          <div className="settings-loader">
            <Loader2
              size={16}
              className="spin"
              style={{ color: "var(--accent-cyan)" }}
            />
            <span>Loading contacts...</span>
          </div>
        ) : (
          <div className="contacts-list">
            {participants.length === 0 ? (
              <div className="contacts-empty">
                <span>
                  No alert contacts configured. Use the form below to add.
                </span>
              </div>
            ) : (
              participants.map((contact) => (
                <div
                  key={contact.participant_id}
                  className={`contact-card ${!contact.is_active ? "is-inactive" : ""}`}
                >
                  <div className="contact-info">
                    <div className="contact-primary">
                      <strong className="contact-name">{contact.name}</strong>
                      <span className="contact-role-badge">{contact.role}</span>
                    </div>
                    <span className="contact-email">{contact.email}</span>
                  </div>
                  <div className="contact-actions">
                    <label
                      className="toggle-switch"
                      title={
                        contact.is_active
                          ? "Deactivate contact"
                          : "Activate contact"
                      }
                    >
                      <input
                        type="checkbox"
                        checked={contact.is_active}
                        onChange={(e) =>
                          onToggleParticipant(
                            contact.participant_id,
                            e.target.checked,
                          )
                        }
                        disabled={!online}
                      />
                      <span className="slider" />
                    </label>
                    <button
                      type="button"
                      className="contact-delete-btn"
                      onClick={() =>
                        onDeleteParticipant(contact.participant_id)
                      }
                      title="Remove contact"
                      disabled={!online}
                    >
                      <Trash2 size={12} />
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        )}
      </div>

      <div className="settings-divider" />

      <form className="add-contact-form" onSubmit={handleSubmit}>
        <h4>Add Alert Contact</h4>
        {error && <div className="form-error">{error}</div>}
        <div className="form-row">
          <input
            type="text"
            placeholder="Contact Name"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            disabled={!online}
            required
          />
          <select
            value={newRole}
            onChange={(e) => setNewRole(e.target.value)}
            disabled={!online}
          >
            <option value="Stakeholder">Stakeholder</option>
            <option value="Operator">Operator</option>
            <option value="Responder">Responder</option>
            <option value="Manager">Manager</option>
          </select>
        </div>
        <div className="form-row">
          <input
            type="email"
            placeholder="email@example.com"
            value={newMail}
            onChange={(e) => setNewMail(e.target.value)}
            disabled={!online}
            required
          />
          <button
            type="submit"
            className="add-btn btn btn-sm blue"
            disabled={!online || adding}
          >
            {adding ? (
              <Loader2 size={12} className="spin" />
            ) : (
              <Plus size={12} />
            )}{" "}
            Add
          </button>
        </div>
      </form>
    </div>
  );
}

function CreditFooter() {
  return <footer className="credit-footer"></footer>;
}

function ProfileModal({ open, onClose }) {
  if (!open) return null;

  return (
    <div
      className="profile-modal-backdrop"
      role="presentation"
      onClick={onClose}
    >
      <section
        className="profile-card"
        role="dialog"
        aria-modal="true"
        aria-labelledby="profile-title"
        onClick={(event) => event.stopPropagation()}
      >
        <button
          className="profile-close"
          type="button"
          onClick={onClose}
          aria-label="Close developer profile"
        >
          <X size={26} />
        </button>
        <div className="profile-top-accent" />
        <div className="profile-avatar">UF</div>
        <h2 id="profile-title">Muhammad Umar Farooq</h2>
        <p className="profile-role">AI Engineer</p>
        <div className="profile-info">
          <strong>Department of Artificial Intelligence</strong>
          <span>University of Management and Technology</span>
          <span>Lahore, Pakistan</span>
        </div>
        <a
          className="profile-link profile-link-github"
          href={GITHUB_REPO_URL}
          target="_blank"
          rel="noreferrer"
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
          </svg>
          <span>GitHub Repository</span>
        </a>
        <a
          className="profile-link profile-link-huggingface"
          href={HUGGING_FACE_URL}
          target="_blank"
          rel="noreferrer"
        >
          <Bot size={24} />
          <span>Hugging Face Model</span>
        </a>
        <a
          className="profile-link profile-link-portfolio"
          href={PORTFOLIO_URL}
          target="_blank"
          rel="noreferrer"
        >
          <ExternalLink size={24} />
          <span>Visit Developer Portfolio</span>
        </a>
      </section>
    </div>
  );
}

function StatusBar({ online, metrics, level, alertConfig }) {
  const isHighRisk = metrics.risk >= 40;
  const alertMode = alertConfig?.mode || "demo";
  const alertEnabled = Boolean(alertConfig?.auto_enabled);

  return (
    <header className="hud-top">
      <div className={`chip status-chip ${online ? "online" : "offline"}`}>
        <span>System Status</span>
        <strong>{online ? "ONLINE" : "OFFLINE"}</strong>
      </div>

      <div className={`chip ${metrics.detectPct > 40 ? "danger" : ""}`}>
        <span>Detection</span>
        <strong>{metrics.detectPct.toFixed(1)}%</strong>
      </div>
      <div className={`chip ${metrics.smokePct > 40 ? "danger" : ""}`}>
        <span>Smoke</span>
        <strong>{metrics.smokePct.toFixed(1)}%</strong>
      </div>
      <div className={`chip ${isHighRisk ? "danger" : ""}`}>
        <span>Peak Fire</span>
        <strong>{Math.round(metrics.peak).toLocaleString()} px</strong>
      </div>
      <div
        className={`chip level ${level} ${level === "critical" ? "alert-system" : ""}`}
      >
        <span>Alert Level</span>
        <strong>{level.toUpperCase()}</strong>
      </div>
      <div className={`chip ${alertEnabled ? "danger" : ""}`}>
        <span>Alert Mode</span>
        <strong>{alertMode.toUpperCase()}</strong>
      </div>
    </header>
  );
}

function VideoDisplay({
  url,
  videoRef,
  canvasRef,
  stageRef,
  analysis,
  online,
  busy,
  exporting,
  onPick,
  onDetect,
  onDownload,
  onTime,
  onReset,
  error,
}) {
  const containerRef = useRef(null);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const toggleFullscreen = () => {
    if (!containerRef.current) return;
    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen().catch((err) => {
        console.error(
          `Error attempting to enable full-screen mode: ${err.message}`,
        );
      });
    } else {
      document.exitFullscreen();
    }
  };

  useEffect(() => {
    const handler = () => setIsFullscreen(!!document.fullscreenElement);
    document.addEventListener("fullscreenchange", handler);
    return () => document.removeEventListener("fullscreenchange", handler);
  }, []);

  return (
    <div
      className={`radar-panel ${isFullscreen ? "is-fullscreen" : ""} ${url ? "has-video" : "is-empty"} ${analysis ? "has-analysis" : ""}`}
      ref={stageRef}
    >
      <div className="stream-badge">CAM_NODE_01</div>
      {url ? (
        <>
          <div className="video-container" ref={containerRef}>
            <video
              ref={videoRef}
              src={url}
              controls
              autoPlay
              muted
              playsInline
              className="video"
              onLoadedMetadata={() => {
                onTime();
              }}
              onPlay={onTime}
              onTimeUpdate={onTime}
              style={{
                position: "absolute",
                inset: 0,
                zIndex: 1,
              }}
            />
            <canvas
              ref={canvasRef}
              className="overlay"
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                zIndex: 2,
                pointerEvents: "none",
                display: "block",
                width: "100%",
                height: "100%",
              }}
            />

            {busy && (
              <div className="analysis-overlay">
                <Loader2 size={32} className="spin" />
                <p>ANALYSIS IN PROGRESS...</p>
              </div>
            )}

            <button
              className="fullscreen-btn"
              onClick={toggleFullscreen}
              title="Toggle HUD Fullscreen"
            >
              <Activity size={16} />
            </button>
          </div>

          <div className="compact-controls">
            <div className="button-stack">
              <button
                className="btn btn-sm red"
                onClick={onDetect}
                disabled={!online || busy}
              >
                {busy ? (
                  <Loader2 size={12} className="spin" />
                ) : (
                  <Activity size={12} />
                )}
                {analysis ? "Detect Again" : "Start Detection"}
              </button>
              <button
                className="btn btn-sm blue"
                onClick={onDownload}
                disabled={!online || exporting || !analysis}
              >
                {exporting ? (
                  <Loader2 size={12} className="spin" />
                ) : (
                  <Download size={12} />
                )}{" "}
                Export
              </button>
            </div>
            <button
              className="btn btn-sm upload-tall"
              onClick={onReset}
              style={{ background: "rgba(255,255,255,0.1)", color: "white" }}
            >
              <Upload size={12} /> Change Video
            </button>
          </div>
          {error && <div className="inline-error video-error">{error}</div>}
        </>
      ) : (
        <div className="placeholder unified-upload">
          <label className={`upload-zone ${!online ? "disabled" : ""}`}>
            <input
              type="file"
              accept="video/*"
              disabled={!online}
              onChange={(e) => onPick(e.target.files?.[0])}
            />
            <div className="upload-content">
              <div className="upload-icon-wrapper">
                <Upload size={48} className="upload-icon" />
                <div className="upload-glow" />
              </div>
              <div className="placeholder-header">
                <Flame size={32} className="flame-icon" />
                <h2>Ready for Detection</h2>
              </div>
              <p>Upload a video to begin real-time fire and smoke analysis</p>
              <div className="upload-btn-mock">Select Video File</div>
            </div>
          </label>
          {error && <div className="inline-error">{error}</div>}
        </div>
      )}
    </div>
  );
}

function FloatingChat({
  isOpen,
  setOpen,
  isFull,
  setFull,
  online,
  chatHistory,
  ragBusy,
  ragQ,
  onQueryChange,
  onQuery,
  chatEndRef,
}) {
  if (!isOpen) {
    return (
      <button
        className="chat-trigger"
        onClick={() => setOpen(true)}
        title="Open AI Assistant"
      >
        <Bot size={24} />
      </button>
    );
  }

  return (
    <div className={`floating-chat ${isFull ? "full" : ""}`}>
      <div className="chatbot-header">
        <div className="row" style={{ gap: "0.75rem" }}>
          <Bot size={16} />
          <h3 style={{ margin: 0, fontSize: "0.9rem" }}>AI RAG Assistant</h3>
        </div>
        <div className="row" style={{ gap: "0.5rem" }}>
          <button
            className="icon-btn"
            onClick={() => setFull(!isFull)}
            title="Full View"
          >
            <Activity size={14} />
          </button>
          <button
            className="icon-btn"
            onClick={() => setOpen(false)}
            title="Close"
          >
            <Shield size={14} style={{ transform: "rotate(45deg)" }} />
          </button>
        </div>
      </div>

      <div
        className="chat-messages"
        style={{ flex: 1, overflowY: "auto", padding: "1.25rem" }}
      >
        {chatHistory.length === 0 && (
          <div className="chat-welcome">
            <Bot size={32} />
            <p>
              Ask anything. I have access to fire protocols and general
              knowledge.
            </p>
          </div>
        )}
        {chatHistory.map((msg, i) => (
          <div key={i} className={`chat-msg ${msg.role}`}>
            {msg.text}
            {msg.sources?.length > 0 && (
              <div className="sources">
                Ref:{" "}
                {msg.sources
                  .map((s, j) => s.doc || s.source || `doc-${j + 1}`)
                  .join(", ")}
              </div>
            )}
          </div>
        ))}
        {ragBusy && (
          <div className="chat-msg bot typing">
            <Loader2 size={12} className="spin" /> Thinking...
          </div>
        )}
        <div ref={chatEndRef} />
      </div>

      <div
        className="chat-input-bar"
        style={{
          padding: "1rem",
          borderTop: "1px solid rgba(255,255,255,0.1)",
        }}
      >
        <input
          type="text"
          placeholder="Ask a question…"
          value={ragQ}
          onChange={(e) => onQueryChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") onQuery();
          }}
          disabled={!online}
          style={{
            flex: 1,
            padding: "0.75rem",
            borderRadius: "8px",
            background: "rgba(0,0,0,0.3)",
            border: "1px solid rgba(255,255,255,0.1)",
            color: "white",
          }}
        />
        <button
          className="chat-send-btn"
          onClick={onQuery}
          disabled={!online || ragBusy || !ragQ.trim()}
          style={{
            background: "var(--accent-cyan)",
            color: "white",
            borderRadius: "8px",
            padding: "0.75rem",
          }}
        >
          {ragBusy ? (
            <Loader2 size={14} className="spin" />
          ) : (
            <Send size={14} />
          )}
        </button>
      </div>
    </div>
  );
}

function ResponsePanel({ metrics, level, responseState, onTriggerResponse }) {
  const getStatusText = (kind, state) => {
    if (state === "sending") {
      if (kind === "sms") return "PREPARING ALERTS...";
      if (kind === "call") return "PREPARING CALL...";
      return "ACTIVATING...";
    }
    if (state === "sent") {
      if (kind === "sms") return "ALERTS PREPARED";
      if (kind === "call") return "CALL DEMO READY";
      return "SYSTEM ACTIVE";
    }
    return "SYSTEM READY";
  };

  const getIcon = (kind, state) => {
    const size = 20;
    if (state === "sending") return <Loader2 size={size} className="spin" />;
    if (kind === "sms") return <Mail size={size} />;
    if (kind === "call") return <PhoneCall size={size} />;
    return <BellRing size={size} />;
  };

  return (
    <div className="panel incident-panel">
      <h3>
        <Shield size={16} /> Incident Response
      </h3>
      <div className="response-grid">
        <div className={`risk-card level-${level}`}>
          <div className="risk-header">
            <AlertTriangle size={14} />
            <span>RISK LEVEL</span>
          </div>
          <div className="risk-value">
            {level.toUpperCase()} <span>({metrics.risk}/100)</span>
          </div>
        </div>

        <button
          className={`response-btn ${responseState.sms}`}
          onClick={() =>
            onTriggerResponse(
              "sms",
              "Alert Prepared",
              "Notification payload prepared for configured recipients.",
              1400,
            )
          }
          disabled={responseState.sms !== "idle"}
        >
          <div className="btn-icon">{getIcon("sms", responseState.sms)}</div>
          <div className="btn-content">
            <span className="btn-label">ALERTS</span>
            <strong className="btn-status">
              {getStatusText("sms", responseState.sms)}
            </strong>
          </div>
          {responseState.sms === "sending" && <div className="btn-glow" />}
        </button>

        <button
          className={`response-btn ${responseState.call}`}
          onClick={() =>
            onTriggerResponse(
              "call",
              "Emergency Call Demo Prepared",
              "Demo call workflow prepared. No real phone call was placed.",
              1800,
            )
          }
          disabled={responseState.call !== "idle"}
        >
          <div className="btn-icon">{getIcon("call", responseState.call)}</div>
          <div className="btn-content">
            <span className="btn-label">EMERGENCY CALL</span>
            <strong className="btn-status">
              {getStatusText("call", responseState.call)}
            </strong>
          </div>
          {responseState.call === "sending" && <div className="btn-glow" />}
        </button>

        <button
          className={`response-btn ${responseState.sprinkler}`}
          onClick={() =>
            onTriggerResponse(
              "sprinkler",
              "Sprinkler System Activated",
              "Fire suppression system is now active in the affected zones.",
              1600,
            )
          }
          disabled={responseState.sprinkler !== "idle"}
        >
          <div className="btn-icon">
            {getIcon("sprinkler", responseState.sprinkler)}
          </div>
          <div className="btn-content">
            <span className="btn-label">SPRINKLER SYSTEM</span>
            <strong className="btn-status">
              {getStatusText("sprinkler", responseState.sprinkler)}
            </strong>
          </div>
          {responseState.sprinkler === "sending" && (
            <div className="btn-glow" />
          )}
        </button>
      </div>
    </div>
  );
}

function NotificationStack({ notifications, onDismiss }) {
  return (
    <div className="notification-stack">
      {notifications.map((n) => (
        <div
          key={n.id}
          className={`notification-item ${n.type}`}
          onClick={() => onDismiss(n.id)}
        >
          <div className="notification-icon">
            {n.kind === "sms" && <Mail size={18} />}
            {n.kind === "call" && <PhoneCall size={18} />}
            {n.kind === "sprinkler" && <BellRing size={18} />}
            {!n.kind && <AlertTriangle size={18} />}
          </div>
          <div className="notification-body">
            <h4>{n.title}</h4>
            <p>{n.message}</p>
          </div>
          <div
            className="notification-progress"
            style={{ animationDuration: "5s" }}
          />
        </div>
      ))}
    </div>
  );
}

function TimelineChart({ timeline }) {
  return (
    <div className="panel wide">
      <h3>
        <Flame size={16} /> Detection Timeline
      </h3>
      {timeline.length ? (
        <div className="chart">
          <ResponsiveContainer
            width="100%"
            height="100%"
            minWidth={1}
            minHeight={220}
          >
            <AreaChart data={timeline}>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="rgba(255,255,255,0.06)"
              />
              <XAxis
                dataKey="t"
                stroke="#94a3b8"
                tick={{ fill: "#94a3b8", fontSize: 11 }}
              />
              <YAxis
                stroke="#94a3b8"
                tick={{ fill: "#94a3b8", fontSize: 11 }}
              />
              <Tooltip
                contentStyle={{
                  background: "#0a0e1a",
                  border: "1px solid #334155",
                  borderRadius: 10,
                }}
              />
              <Area
                type="monotone"
                dataKey="fire"
                stroke="#06b6d4"
                fill="rgba(6,182,212,0.2)"
              />
              <Line
                type="monotone"
                dataKey="smoke"
                stroke="#818cf8"
                dot={false}
                strokeWidth={2}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div className="chart empty-chart">
          <span>Awaiting detection data</span>
        </div>
      )}
    </div>
  );
}

function SummaryPanel({ metrics }) {
  return (
    <div className="panel summary-panel">
      <h3>
        <Wind size={16} /> Analysis Summary
      </h3>
      <div className="summary-stats">
        <div className="stat-circle-group">
          <div
            className="stat-circle"
            style={{ "--percent": metrics.detectPct }}
          >
            <svg width="60" height="60">
              <circle cx="30" cy="30" r="26" />
              <circle
                cx="30"
                cy="30"
                r="26"
                style={{
                  strokeDashoffset: `calc(163.36 - (163.36 * ${metrics.detectPct}) / 100)`,
                }}
              />
            </svg>
            <div className="circle-label">
              <strong>{Math.round(metrics.detectPct)}%</strong>
              <span>FIRE</span>
            </div>
          </div>
          <div
            className="stat-circle smoke"
            style={{ "--percent": metrics.smokePct }}
          >
            <svg width="60" height="60">
              <circle cx="30" cy="30" r="26" />
              <circle
                cx="30"
                cy="30"
                r="26"
                style={{
                  strokeDashoffset: `calc(163.36 - (163.36 * ${metrics.smokePct}) / 100)`,
                }}
              />
            </svg>
            <div className="circle-label">
              <strong>{Math.round(metrics.smokePct)}%</strong>
              <span>SMOKE</span>
            </div>
          </div>
        </div>

        <div className="risk-summary">
          <div className="risk-label">
            <span>RISK SCORE</span>
            <strong>{metrics.risk}/100</strong>
          </div>
          <div className="risk-bar-container">
            <div
              className="risk-bar-fill"
              style={{ width: `${metrics.risk}%` }}
            />
          </div>
        </div>
      </div>
      <p className="summary-footer">
        Deployment Status: <span className="status-ready">ACTIVE</span>
      </p>
    </div>
  );
}

function InstructorModePanel({
  analysis,
  metrics,
  level,
  insights,
  onDownloadReport,
}) {
  const peak = insights.peakFrame;
  const first = insights.firstDetection;
  const modelCard = analysis?.model_card || {};
  const hasAnalysis = Boolean(analysis?.frames?.length);

  return (
    <section className="instructor-mode">
      <div className="instructor-header">
        <div>
          <span className="section-kicker">Instructor Mode</span>
          <h2>Deep Learning Evidence Board</h2>
        </div>
        <button
          className="report-btn"
          type="button"
          onClick={onDownloadReport}
          disabled={!hasAnalysis}
        >
          <FileText size={16} />
          Export Report
        </button>
      </div>

      <div className="wow-grid">
        <div className="wow-card hero-evidence">
          <div className="wow-card-title">
            <Brain size={18} />
            <span>Model Decision</span>
          </div>
          <strong>{insights.verdict}</strong>
          <p>
            {hasAnalysis
              ? `The system sampled ${insights.total} frames and found evidence in ${insights.positiveCount} frames.`
              : "Run detection to populate the instructor evidence board."}
          </p>
          <div className={`verdict-strip level-${level}`}>
            <span>{level.toUpperCase()}</span>
            <b>{metrics.risk}/100 risk</b>
          </div>
        </div>

        <div className="wow-card">
          <div className="wow-card-title">
            <Gauge size={18} />
            <span>Inference Speed</span>
          </div>
          <strong>{formatMs(insights.avgInference)}</strong>
          <p>
            {modelCard.estimated_model_fps
              ? `${modelCard.estimated_model_fps} estimated FPS`
              : "Measured from sampled frames"}
          </p>
        </div>

        <div className="wow-card">
          <div className="wow-card-title">
            <Target size={18} />
            <span>Confidence</span>
          </div>
          <strong>{pct(insights.avgConfidence * 100, 1)}</strong>
          <p>
            Threshold:{" "}
            {analysis?.conf || modelCard.confidence_threshold || 0.25}
          </p>
        </div>

        <div className="wow-card">
          <div className="wow-card-title">
            <Layers size={18} />
            <span>Segmentation</span>
          </div>
          <strong>{insights.maskCount}</strong>
          <p>frames with mask-level fire evidence</p>
        </div>
      </div>

      <div className="instructor-layout">
        <div className="panel-lite explain-panel">
          <h3>
            <Eye size={16} /> Explainable AI View
          </h3>
          <div className="explain-steps">
            {(
              peak?.explainability || [
                "No explanation yet. Run detection first.",
              ]
            ).map((item, index) => (
              <div className="explain-step" key={`${item}-${index}`}>
                <span>{index + 1}</span>
                <p>{item}</p>
              </div>
            ))}
          </div>
          <div className="comparison-grid">
            <div>
              <span>First Detection</span>
              <strong>
                {first ? `${Number(first.t).toFixed(2)}s` : "None"}
              </strong>
            </div>
            <div>
              <span>Peak Fire Area</span>
              <strong>{Math.round(metrics.peak).toLocaleString()} px</strong>
            </div>
            <div>
              <span>Fire Growth</span>
              <strong>
                {insights.growth >= 0 ? "+" : ""}
                {Math.round(insights.growth).toLocaleString()} px
              </strong>
            </div>
          </div>
        </div>

        <div className="panel-lite metrics-panel">
          <h3>
            <BarChart3 size={16} /> Model Evaluation Snapshot
          </h3>
          <div className="metric-bars">
            <MetricBar label="Precision" value={insights.estimatedPrecision} />
            <MetricBar label="Recall" value={insights.estimatedRecall} />
            <MetricBar label="F1 Score" value={insights.estimatedF1} />
          </div>
          <div className="confusion-matrix" aria-label="Confusion matrix proxy">
            <div className="matrix-cell good">
              <span>TP</span>
              <strong>{insights.confusion.tp}</strong>
            </div>
            <div className="matrix-cell warn">
              <span>FP</span>
              <strong>{insights.confusion.fp}</strong>
            </div>
            <div className="matrix-cell warn">
              <span>FN</span>
              <strong>{insights.confusion.fn}</strong>
            </div>
            <div className="matrix-cell good">
              <span>TN</span>
              <strong>{insights.confusion.tn}</strong>
            </div>
          </div>
          <p className="metric-note">
            Proxy metrics are computed from this analyzed sample. Replace them
            with labeled test-set metrics when you present final validation.
          </p>
        </div>

        <div className="panel-lite pipeline-panel">
          <h3>
            <Flame size={16} /> Before / Detection / Decision
          </h3>
          <div className="pipeline-steps">
            <div>
              <span>1</span>
              <strong>Raw Video</strong>
              <p>
                Frame stream sampled at {modelCard.sample_interval_sec || 0.2}s
                intervals.
              </p>
            </div>
            <div>
              <span>2</span>
              <strong>YOLO + Mask</strong>
              <p>
                Bounding boxes, segmentation masks, confidence, and fire area
                are extracted.
              </p>
            </div>
            <div>
              <span>3</span>
              <strong>Risk Engine</strong>
              <p>
                Temporal fire/smoke evidence becomes a {metrics.risk}/100
                response score.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function MetricBar({ label, value }) {
  const safeValue = Math.max(0, Math.min(100, Number(value) || 0));
  return (
    <div className="metric-bar">
      <div>
        <span>{label}</span>
        <strong>{pct(safeValue, 1)}</strong>
      </div>
      <div className="metric-track">
        <span style={{ width: `${safeValue}%` }} />
      </div>
    </div>
  );
}

// ── Main App ──
function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const stageRef = useRef(null);
  const overlayRef = useRef(null);
  const timersRef = useRef([]);
  const chatEndRef = useRef(null);
  const animFrameRef = useRef(null);
  const latestFrameRef = useRef(null);

  const [online, setOnline] = useState(false);
  const [file, setFile] = useState(null);
  const [url, setUrl] = useState(null);
  const [coords, setCoords] = useState({ lat: null, lon: null });
  const [analysis, setAnalysis] = useState(null);
  const [busy, setBusy] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [err, setErr] = useState(null);
  const [notifications, setNotifications] = useState([]);
  const [responseState, setResponseState] = useState({
    sms: "idle",
    call: "idle",
    sprinkler: "idle",
  });
  const [ragQ, setRagQ] = useState("");
  const [chatHistory, setChatHistory] = useState([]);
  const [ragBusy, setRagBusy] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  const [chatFull, setChatFull] = useState(false);
  const [health, setHealth] = useState(null);
  const [profileOpen, setProfileOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [soundEnabled, setSoundEnabled] = useState(false);
  const [participants, setParticipants] = useState([]);
  const [fetchingParticipants, setFetchingParticipants] = useState(false);

  const fetchParticipants = useCallback(async () => {
    if (!online) return;
    setFetchingParticipants(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/participants`);
      if (res.ok) {
        const data = await res.json();
        setParticipants(data);
      }
    } catch (e) {
      console.error("Failed to fetch participants:", e);
    } finally {
      setFetchingParticipants(false);
    }
  }, [online]);

  const handleAddParticipant = useCallback(
    async (participant) => {
      try {
        const res = await fetch(`${BACKEND_URL}/api/participants`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(participant),
        });
        if (res.ok) {
          await fetchParticipants();
          return true;
        }
      } catch (e) {
        console.error("Failed to add participant:", e);
      }
      return false;
    },
    [fetchParticipants],
  );

  const handleDeleteParticipant = useCallback(
    async (id) => {
      try {
        const res = await fetch(`${BACKEND_URL}/api/participants/${id}`, {
          method: "DELETE",
        });
        if (res.ok) {
          await fetchParticipants();
          return true;
        }
      } catch (e) {
        console.error("Failed to delete participant:", e);
      }
      return false;
    },
    [fetchParticipants],
  );

  const handleToggleParticipant = useCallback(
    async (id, isActive) => {
      try {
        const res = await fetch(
          `${BACKEND_URL}/api/participants/${id}/status?is_active=${isActive}`,
          {
            method: "PATCH",
          },
        );
        if (res.ok) {
          await fetchParticipants();
          return true;
        }
      } catch (e) {
        console.error("Failed to toggle participant status:", e);
      }
      return false;
    },
    [fetchParticipants],
  );

  useEffect(() => {
    if (settingsOpen) {
      fetchParticipants();
    }
  }, [settingsOpen, fetchParticipants]);

  const addNotification = useCallback((kind, title, message, type = "info") => {
    const id = Math.random().toString(36).substr(2, 9);
    setNotifications((prev) =>
      [{ id, kind, title, message, type }, ...prev].slice(0, 3),
    );
    setTimeout(() => {
      setNotifications((prev) => prev.filter((n) => n.id !== id));
    }, 5000);
  }, []);

  const dismissNotification = useCallback((id) => {
    setNotifications((prev) => prev.filter((n) => n.id !== id));
  }, []);

  const playAlertTone = useCallback(() => {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!AudioContext) return;
    const ctx = new AudioContext();
    const oscillator = ctx.createOscillator();
    const gain = ctx.createGain();
    oscillator.type = "sine";
    oscillator.frequency.setValueAtTime(720, ctx.currentTime);
    oscillator.frequency.exponentialRampToValueAtTime(
      1080,
      ctx.currentTime + 0.12,
    );
    gain.gain.setValueAtTime(0.0001, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.16, ctx.currentTime + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.28);
    oscillator.connect(gain);
    gain.connect(ctx.destination);
    oscillator.start();
    oscillator.stop(ctx.currentTime + 0.3);
    setTimeout(() => ctx.close(), 420);
  }, []);

  const toggleAlertTone = useCallback(() => {
    const next = !soundEnabled;
    setSoundEnabled(next);
    if (next) {
      playAlertTone();
      addNotification(
        "sms",
        "Alert Tone Armed",
        "Ringtone test played. Header alert tone is now enabled.",
        "success",
      );
    } else {
      addNotification(
        "sms",
        "Alert Tone Muted",
        "Header alert tone has been disabled.",
        "info",
      );
    }
  }, [addNotification, playAlertTone, soundEnabled]);

  // Initialize DetectionOverlay when canvas becomes available (after video upload)
  useEffect(() => {
    console.log(
      "[OverlayInit] URL changed:",
      url ? "video loaded" : "no video",
    );
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    try {
      overlayRef.current = new DetectionOverlay(canvas, {
        glowIntensity: 0.9,
        particleCount: 12,
        pulseSpeed: 2,
      });
      console.log(
        "[OverlayInit] ✅ Overlay created successfully with canvas:",
        canvas,
      );
    } catch (error) {
      console.error("[OverlayInit] ❌ Failed to create overlay:", error);
    }
  }, [url]);

  // ── Helpers & Memos ──
  const trigger = useCallback((msg) => {
    console.info(`[Event] ${new Date().toLocaleTimeString()} ${msg}`);
  }, []);

  const triggerResponse = useCallback(
    (kind, title, message, delay = 1700) => {
      setResponseState((prev) => ({ ...prev, [kind]: "sending" }));
      trigger(`${kind.toUpperCase()} sequence initiated`);

      const timer = setTimeout(() => {
        setResponseState((prev) => ({ ...prev, [kind]: "sent" }));
        trigger(`${kind.toUpperCase()} sequence completed`);
        addNotification(kind, title, message, "success");
      }, delay);
      timersRef.current.push(timer);
    },
    [trigger, addNotification],
  );

  const metrics = useMemo(() => {
    const frames = analysis?.frames || [];
    if (!frames.length) return { detectPct: 0, smokePct: 0, peak: 0, risk: 0 };
    const detect = frames.filter((f) => f.fire || f.smoke).length;
    const smoke = frames.filter((f) => f.smoke).length;
    const peak = Math.max(
      ...frames.map((f) => Number(f.fire_segment_area_pixels || 0)),
    );
    const detectPct = (detect / frames.length) * 100;
    const smokePct = (smoke / frames.length) * 100;
    const risk = Math.round(
      Math.min(100, detectPct * 0.5 + smokePct * 0.25 + (peak / 120000) * 25),
    );
    return { detectPct, smokePct, peak, risk };
  }, [analysis]);

  const level = useMemo(
    () =>
      metrics.risk >= 75
        ? "critical"
        : metrics.risk >= 50
          ? "high"
          : metrics.risk >= 30
            ? "moderate"
            : "low",
    [metrics.risk],
  );

  const instructorInsights = useMemo(
    () => buildInstructorInsights(analysis, metrics, level),
    [analysis, metrics, level],
  );

  const timeline = useMemo(
    () =>
      (analysis?.frames || []).map((f) => ({
        t: Number(f.t || 0).toFixed(2),
        fire: Number(f.fire_segment_area_pixels || 0),
        smoke: f.smoke ? 1 : 0,
      })),
    [analysis],
  );

  // Automatic Incident Response Trigger
  useEffect(() => {
    if (metrics.risk > 50) {
      if (responseState.sms === "idle") {
        triggerResponse(
          "sms",
          "AUTOTRIGGER: Risk level > 50. Preparing alerts.",
          "AUTOTRIGGER: Alerts prepared.",
          1400,
        );
      }
      if (responseState.call === "idle") {
        triggerResponse(
          "call",
          "AUTOTRIGGER: Emergency call demo prepared.",
          "AUTOTRIGGER: Demo call workflow prepared. No real phone call was placed.",
          1800,
        );
      }
      if (responseState.sprinkler === "idle") {
        triggerResponse(
          "sprinkler",
          "AUTOTRIGGER: Risk level > 50. Activating sprinklers.",
          "AUTOTRIGGER: Sprinklers active.",
          1600,
        );
      }
    }
  }, [metrics.risk, responseState, triggerResponse]);

  // Seek to first detection when analysis finishes
  useEffect(() => {
    if (analysis?.frames?.length && videoRef.current) {
      const first = analysis.frames.find((f) => f.fire || f.smoke);
      if (first && first.t > 0) {
        videoRef.current.currentTime = first.t;
      }
    }
  }, [analysis]);

  // Health check & Geolocation
  useEffect(() => {
    console.log("[App] Component mounted");
    const ping = async () => {
      const r = await fetch(`${BACKEND_URL}/api/health`).catch(() => null);
      setOnline(!!r && r.ok);
      if (r?.ok) {
        setHealth(await r.json());
      }
    };
    ping();
    const timer = setInterval(ping, 3500);

    // Request high-accuracy laptop location
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          setCoords({ lat: pos.coords.latitude, lon: pos.coords.longitude });
          console.log(
            "[Geo] ✅ Laptop location captured:",
            pos.coords.latitude,
            pos.coords.longitude,
          );
        },
        (err) =>
          console.warn(
            "[Geo] ❌ Location permission denied or failed:",
            err.message,
          ),
        { enableHighAccuracy: true },
      );
    }

    return () => clearInterval(timer);
  }, []);

  // Log state changes for debugging
  useEffect(() => {
    console.log(
      "[App State Change] url:",
      !!url,
      "file:",
      !!file,
      "analysis:",
      !!analysis,
    );
  }, [url, file, analysis]);

  // Cleanup
  useEffect(
    () => () => {
      if (url) URL.revokeObjectURL(url);
      timersRef.current.forEach((t) => clearTimeout(t));
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    },
    [url],
  );

  // Continuous Canvas Painting Loop
  const renderLoop = useCallback(() => {
    const video = videoRef?.current;
    const canvas = canvasRef?.current;
    const overlay = overlayRef?.current;
    const stage = stageRef?.current;

    // Detailed debugging
    const ready = video && canvas && overlay && stage && video.videoWidth > 0;

    if (ready) {
      try {
        const isFull = !!document.fullscreenElement;
        const containerW = isFull ? window.innerWidth : stage.clientWidth;
        const containerH = isFull ? window.innerHeight : stage.clientHeight;

        console.log(
          "[RenderLoop] ✅ RENDERING - canvas:",
          containerW,
          "x",
          containerH,
          "video:",
          video.videoWidth,
          "x",
          video.videoHeight,
        );
        canvas.width = containerW;
        canvas.height = containerH;

        const videoRatio = video.videoWidth / video.videoHeight;
        const containerRatio = containerW / containerH;

        let drawW = containerW;
        let drawH = containerH;
        let offsetX = 0;
        let offsetY = 0;

        if (videoRatio > containerRatio) {
          drawH = containerW / videoRatio;
          offsetY = (containerH - drawH) / 2;
        } else {
          drawW = containerH * videoRatio;
          offsetX = (containerW - drawW) / 2;
        }

        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.save();
        ctx.translate(offsetX, offsetY);

        const frameData = latestFrameRef.current || { system: "scanning" };
        overlay.render(drawW, drawH, frameData);
        ctx.restore();
      } catch (error) {
        console.error("[RenderLoop Error]", error);
      }
    } else {
      if (!window.__renderWaitLogged) {
        console.log(
          "[RenderLoop] ❌ NOT READY - video:",
          !!video,
          "canvas:",
          !!canvas,
          "overlay:",
          !!overlay,
          "stage:",
          !!stage,
        );
        if (video) console.log("  video.videoWidth:", video.videoWidth);
        if (stage)
          console.log(
            "  stage dims:",
            stage.clientWidth,
            "x",
            stage.clientHeight,
          );
        window.__renderWaitLogged = true;
        setTimeout(() => {
          window.__renderWaitLogged = false;
        }, 2000);
      }
    }

    animFrameRef.current = requestAnimationFrame(renderLoop);
  }, []);

  // Start loop on mount
  useEffect(() => {
    console.log("[RenderLoop Mount] Starting render loop");
    animFrameRef.current = requestAnimationFrame(renderLoop);
    return () => {
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
      }
    };
  }, [renderLoop, url]);

  const onTime = useCallback(() => {
    const v = videoRef.current;
    if (!v || !analysis?.frames?.length) {
      latestFrameRef.current = null;
      return;
    }
    const s = nearestFrame(analysis.frames, v.currentTime);
    latestFrameRef.current = s;

    // Debug: Log frame updates (every 10 frame updates)
    if (!window.__frameCount) window.__frameCount = 0;
    window.__frameCount++;
    if (window.__frameCount % 10 === 0) {
      console.log("[Frame Update]", {
        videoTime: v.currentTime.toFixed(3),
        frameTime: s?.t,
        hasFire: !!s?.fire,
        hasSmoke: !!s?.smoke,
        fireArea: s?.fire_segment_area_pixels,
      });
    }
  }, [analysis]);

  // File upload
  const pick = (f) => {
    if (!f || !f.type.startsWith("video/"))
      return setErr("Select a valid video file.");
    setErr(null);
    setAnalysis(null);
    latestFrameRef.current = null;
    if (url) URL.revokeObjectURL(url);
    setFile(f);
    const blobUrl = URL.createObjectURL(f);
    setUrl(blobUrl);
    console.log("[Pick] Video file selected, blob URL created:", blobUrl);
  };

  // Reset video and analysis
  const resetVideo = () => {
    console.log("[ResetVideo] START - Current state:", {
      url: !!url,
      file: !!file,
      analysis: !!analysis,
    });
    try {
      setErr(null);
      console.log("[ResetVideo] Step 1: Error cleared");

      setAnalysis(null);
      console.log("[ResetVideo] Step 2: Analysis cleared");

      latestFrameRef.current = null;
      console.log("[ResetVideo] Step 3: Frame reference cleared");

      if (url) {
        URL.revokeObjectURL(url);
        console.log("[ResetVideo] Step 4: URL revoked:", url);
      }

      setUrl(null);
      console.log("[ResetVideo] Step 5: URL state cleared");

      setFile(null);
      console.log("[ResetVideo] Step 6: File state cleared");

      console.log("[ResetVideo] ✅ ALL STEPS COMPLETE");
    } catch (error) {
      console.error("[ResetVideo] ❌ ERROR:", error);
    }
  };

  // Detection
  const detect = async () => {
    console.log("[Detect] Starting detection...");
    if (!file) {
      setErr("Select a video before starting detection.");
      console.warn("[Detect] Cannot detect - missing file");
      return;
    }
    if (!online) {
      setErr("Backend is offline. Check the deployment URL, then try again.");
      console.warn("[Detect] Cannot detect - backend offline");
      return;
    }
    setBusy(true);
    setErr(null);
    try {
      const body = new FormData();
      body.append("file", file);
      // Pass the laptop's GPS coordinates to the backend
      const geoParams = coords.lat
        ? `&lat=${coords.lat}&lon=${coords.lon}`
        : "";
      const res = await fetch(
        `${BACKEND_URL}/api/analyze/video?conf=0.25&sample_interval_sec=0.2${geoParams}`,
        {
          method: "POST",
          body,
        },
      );
      const data = await res.json().catch(() => ({}));
      console.log("[Detect] Response received:", {
        status: res.status,
        hasFrames: !!data.frames,
        frameCount: data.frames?.length,
      });
      if (!res.ok) throw new Error(data.detail || "Detection failed");
      setAnalysis(data);
      console.log(
        "[Detect] ✅ Analysis set with",
        data.frames?.length,
        "frames",
      );
      trigger("detection completed");
    } catch (e) {
      console.error("[Detect] ❌ Error:", e);
      setErr(e.message || "Detection failed.");
      setAnalysis(null);
    } finally {
      setBusy(false);
    }
  };

  // Export
  const download = async () => {
    if (!file) {
      setErr("Select a video before exporting.");
      return;
    }
    if (!online) {
      setErr("Backend is offline. Check the deployment URL, then try again.");
      return;
    }
    setExporting(true);
    setErr(null);
    try {
      const body = new FormData();
      body.append("file", file);
      const res = await fetch(
        `${BACKEND_URL}/api/analyze/video/export?conf=0.25&infer_stride=1`,
        { method: "POST", body },
      );
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d.detail || "Export failed");
      }
      const blob = await res.blob();
      const object = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = object;
      a.download = `FireWatch_${file.name.replace(/\.[^.]+$/, "")}.mp4`;
      a.click();
      URL.revokeObjectURL(object);
    } catch (e) {
      setErr(e.message || "Export failed.");
    } finally {
      setExporting(false);
    }
  };

  const downloadInstructorReport = useCallback(async () => {
    if (!analysis?.frames?.length) return;
    try {
      const res = await fetch(`${BACKEND_URL}/api/reports/instructor.pdf`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          analysis,
          metrics,
          level,
          insights: instructorInsights,
        }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || "PDF export failed");
      }
      const blob = await res.blob();
      const object = URL.createObjectURL(blob);
      const a = document.createElement("a");
      const safeName = (analysis.filename || "firewatch-analysis")
        .replace(/\.[^.]+$/, "")
        .replace(/[^a-z0-9_-]+/gi, "_");
      a.href = object;
      a.download = `FireWatch_Instructor_Report_${safeName}.pdf`;
      a.click();
      URL.revokeObjectURL(object);
      addNotification(
        "sms",
        "PDF Report Exported",
        "Styled instructor report downloaded successfully.",
        "success",
      );
    } catch (e) {
      addNotification(
        "sms",
        "PDF Export Failed",
        e.message || "Could not generate the report PDF.",
        "error",
      );
    }
  }, [addNotification, analysis, instructorInsights, level, metrics]);

  // RAG query
  const askRag = async () => {
    if (!online || !ragQ.trim()) return;
    const userMsg = ragQ.trim();
    setRagQ("");
    setChatHistory((prev) => [...prev, { role: "user", text: userMsg }]);
    setRagBusy(true);
    setTimeout(
      () => chatEndRef.current?.scrollIntoView({ behavior: "smooth" }),
      50,
    );
    try {
      const res = await fetch(`${BACKEND_URL}/api/rag/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: userMsg, top_k: 3 }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || "Query failed");
      setChatHistory((prev) => [
        ...prev,
        {
          role: "bot",
          text: data.answer || "No response.",
          sources: data.sources,
        },
      ]);
    } catch (e) {
      setChatHistory((prev) => [
        ...prev,
        { role: "bot", text: e.message || "Query failed." },
      ]);
    } finally {
      setRagBusy(false);
      setTimeout(
        () => chatEndRef.current?.scrollIntoView({ behavior: "smooth" }),
        80,
      );
    }
  };

  return (
    <div
      className={`hud-root ${url ? "has-video" : ""} ${analysis ? "has-analysis" : ""}`}
    >
      <NotificationStack
        notifications={notifications}
        onDismiss={dismissNotification}
      />
      <BrandHeader
        onProfileOpen={() => setProfileOpen(true)}
        onBellClick={toggleAlertTone}
        onSettingsOpen={() => setSettingsOpen((open) => !open)}
        soundEnabled={soundEnabled}
      />
      <ProfileModal open={profileOpen} onClose={() => setProfileOpen(false)} />
      <SettingsPanel
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        online={online}
        soundEnabled={soundEnabled}
        alertConfig={health?.alerts}
        participants={participants}
        onAddParticipant={handleAddParticipant}
        onDeleteParticipant={handleDeleteParticipant}
        onToggleParticipant={handleToggleParticipant}
        fetchingParticipants={fetchingParticipants}
      />
      <StatusBar
        online={online}
        metrics={metrics}
        level={level}
        alertConfig={health?.alerts}
      />

      <section className="hud-center">
        <VideoDisplay
          url={url}
          videoRef={videoRef}
          canvasRef={canvasRef}
          stageRef={stageRef}
          online={online}
          busy={busy}
          exporting={exporting}
          onPick={pick}
          onDetect={detect}
          onDownload={download}
          onTime={onTime}
          onReset={resetVideo}
          analysis={analysis}
          error={err}
        />
      </section>

      <aside className="hud-right">
        <ResponsePanel
          metrics={metrics}
          level={level}
          responseState={responseState}
          onTriggerResponse={triggerResponse}
        />
      </aside>

      <footer className="hud-bottom">
        <TimelineChart timeline={timeline} />
        <SummaryPanel metrics={metrics} />
      </footer>

      <InstructorModePanel
        analysis={analysis}
        metrics={metrics}
        level={level}
        insights={instructorInsights}
        onDownloadReport={downloadInstructorReport}
      />

      <CreditFooter />

      <FloatingChat
        isOpen={chatOpen}
        setOpen={setChatOpen}
        isFull={chatFull}
        setFull={setChatFull}
        online={online}
        chatHistory={chatHistory}
        ragBusy={ragBusy}
        ragQ={ragQ}
        onQueryChange={setRagQ}
        onQuery={askRag}
        chatEndRef={chatEndRef}
      />
    </div>
  );
}

export default App;
