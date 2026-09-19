import React, { useState, useEffect, useMemo, useCallback } from "react";

/**
 * OdomAI — single-file React front end
 * -------------------------------------------------
 * Beyond the odometer. Know your car's true value.
 *
 * Wires up to the existing Flask API (see /health, /metadata, /predict):
 *   GET  {API_BASE}/metadata   -> manufacturer/model options (from vehicles_clean.csv)
 *   POST {API_BASE}/predict    -> { predicted_price, confidence? }
 *
 * A future SQL-backed history endpoint (e.g. GET {API_BASE}/history) can
 * replace the local `readingHistory` state below — see the clearly marked
 * spot in the "Reading history" section.
 *
 * Drop this file in as e.g. src/App.jsx in the Vite + React project.
 */

// ---------------------------------------------------------------------------
// Brand tokens
// ---------------------------------------------------------------------------
const COLORS = {
  purple: "#534AB7", // OdomPurple — primary
  dial: "#AFA9EC", // Dial — secondary accent
  gaugeLight: "#EEEDFE", // Gauge light — app background
  night: "#0F0B26", // Night mode — dark bg / ink
  green: "#1D9E75", // Value green — high confidence / positive value
  red: "#C24B4B", // low confidence — muted, not alarmist
  white: "#FFFFFF",
};

const TAGLINE = "Beyond the odometer. Know your car's true value.";
const API_BASE = "https://odomai.onrender.com";

const CURRENT_YEAR = new Date().getFullYear();
const MIN_YEAR = 1980;
const MAX_YEAR = CURRENT_YEAR + 1;
const MAX_MILES = 400000;

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

// The /metadata contract may evolve; normalize a couple of likely shapes so
// the UI doesn't break if the backend's key names shift.
function normalizeMetadata(raw) {
  if (!raw || typeof raw !== "object") return { manufacturers: [], modelsByManufacturer: {}, fuelTypes: [], transmissions: [] };

  const manufacturers =
    raw.manufacturers || raw.makes || raw.manufacturer_list || Object.keys(raw.models_by_make || raw.modelsByManufacturer || {});

  const modelsByManufacturer =
    raw.models_by_manufacturer || raw.modelsByManufacturer || raw.models_by_make || raw.models || {};

  // Fallback defaults match app.py's hardcoded lists
  const fuelTypes = raw.fuel_types || raw.fuelTypes || ["gas", "diesel", "hybrid", "electric"];
  const transmissions = raw.transmissions || ["automatic", "manual"];

  return {
    manufacturers: Array.isArray(manufacturers) ? manufacturers : [],
    modelsByManufacturer: modelsByManufacturer && typeof modelsByManufacturer === "object" ? modelsByManufacturer : {},
    fuelTypes: Array.isArray(fuelTypes) ? fuelTypes : [],
    transmissions: Array.isArray(transmissions) ? transmissions : [],
  };
}

// Deterministic placeholder confidence, used only until /predict returns a
// real `confidence` field. Keeps the dial meaningful (not random each render)
// without pretending to be a trained estimate.
function placeholderConfidence({ year, miles }) {
  const age = CURRENT_YEAR - Number(year || CURRENT_YEAR);
  const ageScore = Math.max(0, 100 - age * 4);
  const mileageScore = Math.max(0, 100 - (Number(miles || 0) / MAX_MILES) * 100);
  return Math.round(Math.min(96, Math.max(35, ageScore * 0.5 + mileageScore * 0.5)));
}

function formatCurrency(n) {
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(n);
}

function formatMiles(n) {
  return new Intl.NumberFormat("en-US").format(n);
}

// Interpolate between red and green for the dial fill + needle color.
function confidenceColor(pct) {
  const t = Math.max(0, Math.min(100, pct)) / 100;
  const hex = (c) => Math.round(c).toString(16).padStart(2, "0");
  const from = { r: 0xc2, g: 0x4b, b: 0x4b }; // red
  const to = { r: 0x1d, g: 0x9e, b: 0x75 }; // green
  const r = from.r + (to.r - from.r) * t;
  const g = from.g + (to.g - from.g) * t;
  const b = from.b + (to.b - from.b) * t;
  return `#${hex(r)}${hex(g)}${hex(b)}`;
}

// ---------------------------------------------------------------------------
// Logo — gauge/dial wordmark lockup, matches the three brand variants
// ---------------------------------------------------------------------------
function Logo({ variant = "light", size = 28 }) {
  const ink = variant === "dark" ? COLORS.white : variant === "brand" ? COLORS.purple : COLORS.night;
  const needle = variant === "dark" ? COLORS.dial : COLORS.purple;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      <svg width={size} height={size} viewBox="0 0 40 40" aria-hidden="true">
        <circle cx="20" cy="20" r="18" fill="none" stroke={variant === "dark" ? COLORS.dial : COLORS.gaugeLight} strokeWidth="3" />
        <path
          d="M20 20 L28 12"
          stroke={needle}
          strokeWidth="2.5"
          strokeLinecap="round"
        />
        <circle cx="20" cy="20" r="3" fill={needle} />
      </svg>
      <span
        style={{
          fontFamily: "'Fraunces', Georgia, serif",
          fontWeight: 600,
          fontSize: size * 0.75,
          color: ink,
          letterSpacing: "-0.01em",
        }}
      >
        Odom<span style={{ color: variant === "brand" ? ink : COLORS.purple }}>AI</span>
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Confidence dial — semicircular gauge, red -> green, needle at pct
// ---------------------------------------------------------------------------
function ConfidenceDial({ pct }) {
  const clamped = Math.max(0, Math.min(100, pct));
  const angle = -90 + (clamped / 100) * 180; // -90deg (left) to +90deg (right)
  const color = confidenceColor(clamped);

  // Arc geometry (semicircle, r=80, centered at 100,100)
  const r = 80;
  const cx = 100;
  const cy = 100;
  const describeArc = (startAngle, endAngle) => {
    const toRad = (deg) => ((deg - 180) * Math.PI) / 180;
    const start = { x: cx + r * Math.cos(toRad(startAngle)), y: cy + r * Math.sin(toRad(startAngle)) };
    const end = { x: cx + r * Math.cos(toRad(endAngle)), y: cy + r * Math.sin(toRad(endAngle)) };
    const largeArc = endAngle - startAngle > 180 ? 1 : 0;
    return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 1 ${end.x} ${end.y}`;
  };

  const needleRad = ((angle - 90) * Math.PI) / 180;
  const needleLen = r - 14;
  const nx = cx + needleLen * Math.sin(needleRad) * -1;
  const ny = cy - needleLen * Math.cos(needleRad) * -1;
  // simpler: compute needle tip directly from angle where 0 = pointing up
  const tipX = cx + needleLen * Math.sin((angle * Math.PI) / 180);
  const tipY = cy - needleLen * Math.cos((angle * Math.PI) / 180);

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6 }}>
      <svg width="220" height="130" viewBox="0 0 200 115">
        <defs>
          <linearGradient id="odomai-gauge-grad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor={COLORS.red} />
            <stop offset="50%" stopColor="#D9A441" />
            <stop offset="100%" stopColor={COLORS.green} />
          </linearGradient>
        </defs>
        <path d={describeArc(0, 180)} fill="none" stroke={COLORS.gaugeLight} strokeWidth="14" strokeLinecap="round" />
        <path
          d={describeArc(0, 180)}
          fill="none"
          stroke="url(#odomai-gauge-grad)"
          strokeWidth="14"
          strokeLinecap="round"
          opacity="0.9"
        />
        <line x1={cx} y1={cy} x2={tipX} y2={tipY} stroke={COLORS.night} strokeWidth="3" strokeLinecap="round" />
        <circle cx={cx} cy={cy} r="6" fill={COLORS.night} />
      </svg>
      <div style={{ fontFamily: "'Fraunces', Georgia, serif", fontSize: 28, fontWeight: 600, color }}>
        {clamped}%
      </div>
      <div style={{ fontSize: 12, letterSpacing: "0.02em", color: COLORS.night, opacity: 0.6 }}>
        AI confidence reading
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main app
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Confidence dial — semicircular gauge, red -> green, with a visible pointer
// ---------------------------------------------------------------------------
function ConfidenceDial({ pct }) {
  const clamped = Math.max(0, Math.min(100, pct));
  const angle = -90 + (clamped / 100) * 180; // -90deg (left) to +90deg (right)
  const color = confidenceColor(clamped);

  const r = 80;
  const cx = 100;
  const cy = 100;
  const describeArc = (startAngle, endAngle) => {
    const toRad = (deg) => ((deg - 180) * Math.PI) / 180;
    const start = { x: cx + r * Math.cos(toRad(startAngle)), y: cy + r * Math.sin(toRad(startAngle)) };
    const end = { x: cx + r * Math.cos(toRad(endAngle)), y: cy + r * Math.sin(toRad(endAngle)) };
    const largeArc = endAngle - startAngle > 180 ? 1 : 0;
    return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 1 ${end.x} ${end.y}`;
  };

  // Needle: base pivot + a triangular pointer tip, so it reads as an arrow
  const needleLen = r - 10;
  const tipX = cx + needleLen * Math.sin((angle * Math.PI) / 180);
  const tipY = cy - needleLen * Math.cos((angle * Math.PI) / 180);

  // Small perpendicular offset to build a triangle (arrowhead) at the tip
  const perpAngle = angle + 90;
  const baseWidth = 5;
  const baseLx = cx + baseWidth * Math.sin((perpAngle * Math.PI) / 180);
  const baseLy = cy - baseWidth * Math.cos((perpAngle * Math.PI) / 180);
  const baseRx = cx - baseWidth * Math.sin((perpAngle * Math.PI) / 180);
  const baseRy = cy + baseWidth * Math.cos((perpAngle * Math.PI) / 180);

  // A marker dot exactly on the arc at the current position, for extra clarity
  const markerR = r;
  const markerX = cx + markerR * Math.sin((angle * Math.PI) / 180);
  const markerY = cy - markerR * Math.cos((angle * Math.PI) / 180);

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6 }}>
      <svg width="220" height="130" viewBox="0 0 200 115">
        <defs>
          <linearGradient id="odomai-gauge-grad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor={COLORS.red} />
            <stop offset="50%" stopColor="#D9A441" />
            <stop offset="100%" stopColor={COLORS.green} />
          </linearGradient>
        </defs>

        {/* Track */}
        <path d={describeArc(0, 180)} fill="none" stroke={COLORS.gaugeLight} strokeWidth="14" strokeLinecap="round" />
        {/* Gradient scale */}
        <path
          d={describeArc(0, 180)}
          fill="none"
          stroke="url(#odomai-gauge-grad)"
          strokeWidth="14"
          strokeLinecap="round"
          opacity="0.9"
        />

        {/* Marker dot on the arc showing exact position */}
        <circle cx={markerX} cy={markerY} r="7" fill={COLORS.white} stroke={color} strokeWidth="3" />

        {/* Needle shaft */}
        <line x1={cx} y1={cy} x2={tipX} y2={tipY} stroke={COLORS.night} strokeWidth="3" strokeLinecap="round" />
        {/* Arrowhead at the tip, pointing along the needle direction */}
        <polygon
          points={`${tipX},${tipY} ${baseLx},${baseLy} ${baseRx},${baseRy}`}
          fill={COLORS.night}
        />

        {/* Pivot */}
        <circle cx={cx} cy={cy} r="6" fill={COLORS.night} />
      </svg>
      <div style={{ fontFamily: "'Fraunces', Georgia, serif", fontSize: 28, fontWeight: 600, color }}>
        {clamped}%
      </div>
      <div style={{ fontSize: 12, letterSpacing: "0.02em", color: COLORS.night, opacity: 0.85, fontWeight: 500 }}>
        AI confidence reading
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------
const FONT_IMPORT = `@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');`;

const styles = {
  page: {
    minHeight: "100vh",
    background: COLORS.gaugeLight,
    color: COLORS.night,
    fontFamily: "'Inter', system-ui, sans-serif",
    display: "flex",
    flexDirection: "column",
  },
  header: {
    padding: "28px 24px 20px",
    display: "flex",
    flexDirection: "column",
    gap: 8,
    maxWidth: 760,
    margin: "0 auto",
    width: "100%",
    boxSizing: "border-box",
  },
  headerTop: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    width: "100%",
  },
  nav: {
    display: "flex",
    gap: 8,
    background: COLORS.white,
    padding: 4,
    borderRadius: 20,
    border: `1px solid ${COLORS.dial}33`,
  },
  navButton: {
    padding: "6px 14px",
    borderRadius: 16,
    border: "none",
    background: "transparent",
    color: COLORS.night,
    opacity: 0.7,
    fontSize: 14,
    fontWeight: 500,
    cursor: "pointer",
  },
  navActive: {
    padding: "6px 14px",
    borderRadius: 16,
    border: "none",
    background: COLORS.gaugeLight,
    color: COLORS.night,
    fontSize: 14,
    fontWeight: 600,
    cursor: "default",
  },
  tagline: {
    margin: 0,
    fontFamily: "'Fraunces', Georgia, serif",
    fontSize: 20,
    color: COLORS.night,
    opacity: 0.85,
  },
  main: {
    flex: 1,
    maxWidth: 760,
    margin: "0 auto",
    width: "100%",
    padding: "0 24px 48px",
    display: "flex",
    flexDirection: "column",
    gap: 20,
    boxSizing: "border-box",
  },
  card: {
    background: COLORS.white,
    borderRadius: 16,
    padding: 24,
    boxShadow: "0 1px 2px rgba(15,11,38,0.06)",
    border: `1px solid ${COLORS.dial}33`,
  },
  cardTitle: {
    margin: "0 0 16px",
    fontFamily: "'Fraunces', Georgia, serif",
    fontSize: 20,
    fontWeight: 600,
  },
  formGrid: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: 16,
  },
  field: { display: "flex", flexDirection: "column", gap: 6 },
  label: { fontSize: 13, color: COLORS.night, opacity: 0.7 },
  select: {
    padding: "10px 12px",
    borderRadius: 10,
    border: `1px solid ${COLORS.dial}66`,
    background: COLORS.gaugeLight,
    fontSize: 15,
    color: COLORS.night,
  },
  input: {
    padding: "10px 12px",
    borderRadius: 10,
    border: `1px solid ${COLORS.dial}66`,
    background: COLORS.gaugeLight,
    fontSize: 15,
    color: COLORS.night,
  },
  inputError: { border: `1px solid ${COLORS.red}` },
  errorText: { color: COLORS.red, fontSize: 12 },
  warnText: { color: COLORS.red, fontSize: 13, marginTop: 12 },
  mutedText: { color: COLORS.night, opacity: 0.6, fontSize: 14 },
  button: {
    marginTop: 20,
    width: "100%",
    padding: "14px 20px",
    borderRadius: 12,
    border: "none",
    background: COLORS.purple,
    color: COLORS.white,
    fontSize: 16,
    fontWeight: 600,
    cursor: "pointer",
  },
  resultCard: {
    background: COLORS.night,
    borderRadius: 16,
    padding: 28,
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    flexWrap: "wrap",
    gap: 20,
  },
  priceBlock: { display: "flex", flexDirection: "column", gap: 4 },
  priceLabel: { color: COLORS.dial, fontSize: 13 },
  price: {
    fontFamily: "'Fraunces', Georgia, serif",
    fontSize: 48,
    fontWeight: 700,
    color: COLORS.white,
    lineHeight: 1.05,
  },
  priceSub: { color: COLORS.dial, fontSize: 14 },
  historyCard: {
    background: COLORS.white,
    borderRadius: 16,
    padding: 24,
    border: `1px solid ${COLORS.dial}33`,
  },
  historyTableWrap: { overflowX: "auto" },
  refreshButton: {
    padding: "6px 12px",
    borderRadius: 8,
    border: `1px solid ${COLORS.dial}`,
    background: COLORS.white,
    color: COLORS.night,
    fontSize: 13,
    fontWeight: 500,
    cursor: "pointer",
  },
  historyTable: { width: "100%", borderCollapse: "collapse", fontSize: 14 },
  th: {
    textAlign: "left",
    padding: "8px 10px",
    color: COLORS.night,
    opacity: 0.6,
    fontWeight: 500,
    borderBottom: `1px solid ${COLORS.dial}55`,
  },
  td: {
    padding: "10px 10px",
    borderBottom: `1px solid ${COLORS.dial}33`,
  },
  footer: {
    padding: "20px 24px 32px",
    maxWidth: 760,
    margin: "0 auto",
    width: "100%",
    display: "flex",
    alignItems: "center",
    gap: 10,
    boxSizing: "border-box",
  },
  footerText: { fontSize: 13, opacity: 0.55 },
};
