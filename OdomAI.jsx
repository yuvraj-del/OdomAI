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
export default function OdomAI() {
  const [metadata, setMetadata] = useState({ manufacturers: [], modelsByManufacturer: {}, fuelTypes: [], transmissions: [] });
  const [metadataError, setMetadataError] = useState(null);
  const [loadingMetadata, setLoadingMetadata] = useState(true);

  const [manufacturer, setManufacturer] = useState("");
  const [model, setModel] = useState("");
  const [fuel, setFuel] = useState("");
  const [transmission, setTransmission] = useState("");
  const [year, setYear] = useState("");
  const [miles, setMiles] = useState("");

  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState(null); // { price, confidence }
  const [apiError, setApiError] = useState(null);

  // Reading history — local for now. Swap the setter below for a fetch to a
  // future GET {API_BASE}/history endpoint backed by the SQL history table,
  // and drop the localStorage persistence once the backend owns this.
  const [readingHistory, setReadingHistory] = useState(() => {
    try {
      const saved = window.localStorage?.getItem("odomai_history");
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        setLoadingMetadata(true);
        const res = await fetch(`${API_BASE}/metadata`);
        if (!res.ok) throw new Error(`Metadata request failed (${res.status})`);
        const raw = await res.json();
        if (!cancelled) setMetadata(normalizeMetadata(raw));
      } catch (err) {
        if (!cancelled) setMetadataError(err.message || "Couldn't load make/model list.");
      } finally {
        if (!cancelled) setLoadingMetadata(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const models = useMemo(() => {
    if (!manufacturer) return [];
    return metadata.modelsByManufacturer[manufacturer] || [];
  }, [manufacturer, metadata]);

  const yearError =
    year !== "" && (!/^\d+$/.test(String(year)) || Number(year) < MIN_YEAR || Number(year) > MAX_YEAR)
      ? `Enter a year between ${MIN_YEAR} and ${MAX_YEAR}.`
      : null;

  const milesError =
    miles !== "" && (!/^\d+$/.test(String(miles)) || Number(miles) < 0 || Number(miles) > MAX_MILES)
      ? `Enter mileage between 0 and ${formatMiles(MAX_MILES)}.`
      : null;

  const canSubmit =
    manufacturer && model && fuel && transmission && year !== "" && miles !== "" && !yearError && !milesError && !submitting;

  const handleManufacturerChange = (e) => {
    setManufacturer(e.target.value);
    setModel(""); // reset dependent dropdown — model list is restricted per manufacturer
    setResult(null);
  };

  const handleYearChange = (e) => {
    const v = e.target.value.replace(/[^\d]/g, "").slice(0, 4);
    setYear(v);
  };

  const handleMilesChange = (e) => {
    const v = e.target.value.replace(/[^\d]/g, "").slice(0, 6);
    setMiles(v);
  };

  const calculate = useCallback(async () => {
    if (!canSubmit) return;
    setSubmitting(true);
    setApiError(null);
    setResult(null);

    const payload = {
      manufacturer,
      model,
      fuel,
      transmission,
      year: Number(year),
      odometer: Number(miles),
    };

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(`Prediction request failed (${res.status})`);
      const data = await res.json();

      const price = data.predicted_price ?? data.price;
      // Real confidence isn't shipped by the backend yet — fall back to a
      // deterministic placeholder so the dial still reads meaningfully.
      const confidence = typeof data.confidence === "number" ? data.confidence : placeholderConfidence(payload);

      if (typeof price !== "number") throw new Error("No price came back from the model.");

      const reading = { ...payload, price, confidence, timestamp: new Date().toISOString() };
      setResult(reading);

      setReadingHistory((prev) => {
        const next = [reading, ...prev].slice(0, 8);
        try {
          window.localStorage?.setItem("odomai_history", JSON.stringify(next));
        } catch {
          /* localStorage unavailable — history just won't persist */
        }
        return next;
      });
    } catch (err) {
      setApiError(err.message || "Something went wrong getting a reading.");
    } finally {
      setSubmitting(false);
    }
  }, [canSubmit, manufacturer, model, year, miles]);

  return (
    <div style={styles.page}>
      <style>{FONT_IMPORT}</style>

      <header style={styles.header}>
        <Logo variant="light" size={30} />
        <p style={styles.tagline}>{TAGLINE}</p>
      </header>

      <main style={styles.main}>
        <section style={styles.card}>
          <h2 style={styles.cardTitle}>Take a reading</h2>

          <div style={styles.formGrid}>
            <label style={styles.field}>
              <span style={styles.label}>Manufacturer</span>
              <select
                style={styles.select}
                value={manufacturer}
                onChange={handleManufacturerChange}
                disabled={loadingMetadata}
              >
                <option value="">{loadingMetadata ? "Loading…" : "Select manufacturer"}</option>
                {metadata.manufacturers.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </label>

            <label style={styles.field}>
              <span style={styles.label}>Model</span>
              <select
                style={styles.select}
                value={model}
                onChange={(e) => setModel(e.target.value)}
                disabled={!manufacturer}
              >
                <option value="">{manufacturer ? "Select model" : "Pick a manufacturer first"}</option>
                {models.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </label>

            <label style={styles.field}>
              <span style={styles.label}>Fuel Type</span>
              <select
                style={styles.select}
                value={fuel}
                onChange={(e) => setFuel(e.target.value)}
                disabled={metadata.fuelTypes.length === 0}
              >
                <option value="">Select fuel type</option>
                {metadata.fuelTypes.map((f) => (
                  <option key={f} value={f}>
                    {f.charAt(0).toUpperCase() + f.slice(1)}
                  </option>
                ))}
              </select>
            </label>

            <label style={styles.field}>
              <span style={styles.label}>Transmission</span>
              <select
                style={styles.select}
                value={transmission}
                onChange={(e) => setTransmission(e.target.value)}
                disabled={metadata.transmissions.length === 0}
              >
                <option value="">Select transmission</option>
                {metadata.transmissions.map((t) => (
                  <option key={t} value={t}>
                    {t.charAt(0).toUpperCase() + t.slice(1)}
                  </option>
                ))}
              </select>
            </label>

            <label style={styles.field}>
              <span style={styles.label}>Year</span>
              <input
                style={{ ...styles.input, ...(yearError ? styles.inputError : {}) }}
                type="number"
                inputMode="numeric"
                placeholder={`e.g. ${CURRENT_YEAR - 3}`}
                value={year}
                onChange={handleYearChange}
              />
              {yearError && <span style={styles.errorText}>{yearError}</span>}
            </label>

            <label style={styles.field}>
              <span style={styles.label}>Miles</span>
              <input
                style={{ ...styles.input, ...(milesError ? styles.inputError : {}) }}
                type="number"
                inputMode="numeric"
                placeholder="e.g. 42000"
                value={miles}
                onChange={handleMilesChange}
              />
              {milesError && <span style={styles.errorText}>{milesError}</span>}
            </label>
          </div>

          {metadataError && <p style={styles.warnText}>{metadataError} — you can still type once the list loads.</p>}

          <button style={{ ...styles.button, opacity: canSubmit ? 1 : 0.5 }} onClick={calculate} disabled={!canSubmit}>
            {submitting ? "Reading the gauges…" : "Calculate price"}
          </button>

          {apiError && <p style={styles.errorText}>{apiError}</p>}
        </section>

        {result && (
          <section style={styles.resultCard}>
            <div style={styles.priceBlock}>
              <span style={styles.priceLabel}>Estimated value</span>
              <span style={styles.price}>{formatCurrency(result.price)}</span>
              <span style={styles.priceSub}>
                {result.year} {result.manufacturer} {result.model} · {formatMiles(result.odometer)} mi · {result.fuel} · {result.transmission}
              </span>
            </div>
            <ConfidenceDial pct={result.confidence} />
          </section>
        )}

        <section style={styles.historyCard}>
          <h2 style={styles.cardTitle}>Reading history</h2>
          {readingHistory.length === 0 ? (
            <p style={styles.mutedText}>
              Past readings will show up here. Once the backend's SQL history table is exposed through an
              endpoint, this panel is the spot to swap in.
            </p>
          ) : (
            <div style={styles.historyTableWrap}>
              <table style={styles.historyTable}>
                <thead>
                  <tr>
                    <th style={styles.th}>Vehicle</th>
                    <th style={styles.th}>Year</th>
                    <th style={styles.th}>Miles</th>
                    <th style={styles.th}>Price</th>
                    <th style={styles.th}>Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {readingHistory.map((r, i) => (
                    <tr key={i}>
                      <td style={styles.td}>
                        {r.manufacturer} {r.model}
                      </td>
                      <td style={styles.td}>{r.year}</td>
                      <td style={styles.td}>{formatMiles(r.odometer)}</td>
                      <td style={styles.td}>{formatCurrency(r.price)}</td>
                      <td style={{ ...styles.td, color: confidenceColor(r.confidence), fontWeight: 600 }}>
                        {r.confidence}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </main>

      <footer style={styles.footer}>
        <Logo variant="light" size={18} />
        <span style={styles.footerText}>Gauges, not guesswork.</span>
      </footer>
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
