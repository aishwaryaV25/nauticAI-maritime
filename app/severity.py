SEVERITY_MAP = {
    # ── Subsea Infrastructure ─────────────────────────────────────────────
    "leakage":          ("Critical", "#FF4444", "🚨 Immediate intervention — active leak detected"),
    "anomaly":          ("Critical", "#FF4444", "🚨 Unknown anomaly — flag for immediate review"),
    "pipe_coupling":    ("High",     "#FF8800", "⚠️  Check seal integrity — schedule inspection"),
    "flange":           ("High",     "#FF8800", "⚠️  Inspect flange seal and bolt integrity"),
    "anode":            ("Medium",   "#FFCC00", "🔍 Check sacrificial anode depletion level"),
    "bend_restrictor":  ("Medium",   "#FFCC00", "🔍 Check for fatigue cracking at bend"),
    "biofouling":       ("Medium",   "#FFCC00", "🔍 Monitor — re-inspect in 90 days"),
    "pipeline":         ("Low",      "#44BB44", "✅ Structural element — log condition"),
    "concrete":         ("Low",      "#44BB44", "✅ Foundation element — check displacement"),
    "buoy":             ("Low",      "#44BB44", "✅ Navigation aid — check mooring"),
    # ── Hull Classes ──────────────────────────────────────────────────────
    "bilge_keel":       ("Medium",   "#FFCC00", "🔍 Inspect for corrosion and structural damage"),
    "draft_mark":       ("Low",      "#44BB44", "✅ Draft mark visible — log reading"),
    "hull":             ("Medium",   "#FFCC00", "🔍 Check hull coating and corrosion level"),
    "propeller":        ("High",     "#FF8800", "⚠️  Inspect propeller for cavitation damage"),
    "ropeguard":        ("Medium",   "#FFCC00", "🔍 Check ropeguard integrity"),
    "rudder":           ("High",     "#FF8800", "⚠️  Inspect rudder bearings and pintles"),
    "sea_chest":        ("High",     "#FF8800", "⚠️  Check sea chest grating for blockage"),
    "thruster_blades":  ("High",     "#FF8800", "⚠️  Inspect thruster blades for damage"),
    "thruster_grating": ("Medium",   "#FFCC00", "🔍 Check grating for marine growth blockage"),
}

def get_severity(label: str, conf: float):
    key = label.lower().replace("-", "_").replace(" ", "_")
    sev, color, action = SEVERITY_MAP.get(key, ("Medium", "#FFCC00", "🔍 Monitor and log finding"))
    if sev == "Low" and conf > 0.85:
        sev, color, action = "Medium", "#FFCC00", "🔍 High confidence — monitor closely"
    return sev, color, action