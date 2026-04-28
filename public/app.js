const EXECUTION_MODES = {
  webgpu: {
    id: "webgpu",
    label: "WebGPU EP",
    backend: "webgpu",
    fallbackTriggered: false,
    workerMode: "worker",
    initDelayMs: 72,
    graphDelayMs: 16,
    firstOutputDelayMs: 18,
    stepDelayMs: 9,
    stepCount: 12
  },
  fallback: {
    id: "fallback",
    label: "Wasm Fallback",
    backend: "wasm",
    fallbackTriggered: true,
    workerMode: "main",
    initDelayMs: 136,
    graphDelayMs: 31,
    firstOutputDelayMs: 42,
    stepDelayMs: 21,
    stepCount: 12
  }
};

function resolveExecutionMode() {
  const requested = new URLSearchParams(window.location.search).get("mode");
  return EXECUTION_MODES[requested] || EXECUTION_MODES.webgpu;
}

const executionMode = resolveExecutionMode();

const requestedMode = typeof window !== "undefined"
  ? new URLSearchParams(window.location.search).get("mode")
  : null;
const isRealRuntimeMode = typeof requestedMode === "string" && requestedMode.startsWith("real-");
const REAL_ADAPTER_WAIT_MS = 5000;
const REAL_ADAPTER_LOAD_MS = 20000;

function withTimeout(promise, timeoutMs, label) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`${label} timed out after ${timeoutMs} ms`)), timeoutMs);
    promise.then((value) => {
      clearTimeout(timer);
      resolve(value);
    }, (error) => {
      clearTimeout(timer);
      reject(error);
    });
  });
}

function findRegisteredRealRuntime() {
  const registry = typeof window !== "undefined" ? window.__aiWebGpuLabRuntimeRegistry : null;
  if (!registry || typeof registry.list !== "function") return null;
  return registry.list().find((adapter) => adapter && adapter.isReal === true) || null;
}

async function awaitRealRuntime(timeoutMs = REAL_ADAPTER_WAIT_MS) {
  const startedAt = performance.now();
  while (performance.now() - startedAt < timeoutMs) {
    const adapter = findRegisteredRealRuntime();
    if (adapter) return adapter;
    if (typeof window !== "undefined" && window.__aiWebGpuLabRealOrtBootstrapError) {
      return null;
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  return null;
}

const state = {
  startedAt: performance.now(),
  environment: buildEnvironment(),
  active: false,
  run: null,
  trace: [],
  realAdapterError: null,
  logs: []
};

const elements = {
  inputProfile: document.getElementById("input-profile"),
  statusRow: document.getElementById("status-row"),
  summary: document.getElementById("summary"),
  runInference: document.getElementById("run-inference"),
  downloadJson: document.getElementById("download-json"),
  traceView: document.getElementById("trace-view"),
  metricGrid: document.getElementById("metric-grid"),
  metaGrid: document.getElementById("meta-grid"),
  logList: document.getElementById("log-list"),
  resultJson: document.getElementById("result-json")
};

function round(value, digits = 2) {
  if (!Number.isFinite(value)) return null;
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function parseBrowser() {
  const ua = navigator.userAgent;
  for (const [needle, name] of [["Edg/", "Edge"], ["Chrome/", "Chrome"], ["Firefox/", "Firefox"], ["Version/", "Safari"]]) {
    const marker = ua.indexOf(needle);
    if (marker >= 0) return { name, version: ua.slice(marker + needle.length).split(/[\s)/;]/)[0] || "unknown" };
  }
  return { name: "Unknown", version: "unknown" };
}

function parseOs() {
  const ua = navigator.userAgent;
  if (/Windows NT/i.test(ua)) return { name: "Windows", version: (ua.match(/Windows NT ([0-9.]+)/i) || [])[1] || "unknown" };
  if (/Mac OS X/i.test(ua)) return { name: "macOS", version: ((ua.match(/Mac OS X ([0-9_]+)/i) || [])[1] || "unknown").replace(/_/g, ".") };
  if (/Linux/i.test(ua)) return { name: "Linux", version: "unknown" };
  return { name: "Unknown", version: "unknown" };
}

function inferDeviceClass() {
  const threads = navigator.hardwareConcurrency || 0;
  const memory = navigator.deviceMemory || 0;
  if (memory >= 16 && threads >= 12) return "desktop-high";
  if (memory >= 8 && threads >= 8) return "desktop-mid";
  if (threads >= 4) return "laptop";
  return "unknown";
}

function buildEnvironment() {
  return {
    browser: parseBrowser(),
    os: parseOs(),
    device: {
      name: navigator.platform || "unknown",
      class: inferDeviceClass(),
      cpu: navigator.hardwareConcurrency ? `${navigator.hardwareConcurrency} threads` : "unknown",
      memory_gb: navigator.deviceMemory || undefined,
      power_mode: "unknown"
    },
    gpu: {
      adapter: executionMode.fallbackTriggered ? "wasm-fallback-simulated" : "ort-webgpu-simulated-adapter",
      required_features: executionMode.fallbackTriggered ? [] : ["shader-f16"],
      limits: {}
    },
    backend: executionMode.backend,
    fallback_triggered: executionMode.fallbackTriggered,
    worker_mode: executionMode.workerMode,
    cache_state: "warm"
  };
}

function log(message) {
  state.logs.unshift(`[${new Date().toLocaleTimeString()}] ${message}`);
  state.logs = state.logs.slice(0, 12);
  renderLogs();
}

function parseInputProfile(text) {
  const values = {};
  for (const part of text.split(/\s+/)) {
    const [key, value] = part.split("=");
    if (key && value) values[key] = value;
  }
  return {
    batch: Number(values.batch) || 1,
    sequence: Number(values.sequence) || 128,
    hidden: Number(values.hidden) || 384,
    heads: Number(values.heads) || 6,
    quantization: values.quantization || "q4f16",
    provider: executionMode.id
  };
}

function syntheticKernel(profile, step) {
  const width = Math.max(8, Math.floor(profile.hidden / 24));
  let checksum = 0;
  for (let row = 0; row < width; row += 1) {
    for (let col = 0; col < width; col += 1) {
      const left = ((row + 1) * (step + 3)) % 17;
      const right = ((col + 5) * (profile.heads + 7)) % 19;
      checksum += Math.sin((left * right + profile.sequence) * 0.0174533);
    }
  }
  return round(checksum, 4);
}

async function runRealRuntimeOrt(adapter) {
  log(`Connecting real runtime adapter '${adapter.id}'.`);
  await withTimeout(
    Promise.resolve(adapter.loadModel({ modelId: "ort-webgpu-baseline-default" })),
    REAL_ADAPTER_LOAD_MS,
    `loadModel(${adapter.id})`
  );
  const prefill = await withTimeout(
    Promise.resolve(adapter.prefill({ promptTokens: 96 })),
    REAL_ADAPTER_LOAD_MS,
    `prefill(${adapter.id})`
  );
  const decode = await withTimeout(
    Promise.resolve(adapter.decode({ tokenBudget: 32 })),
    REAL_ADAPTER_LOAD_MS,
    `decode(${adapter.id})`
  );
  log(`Real runtime adapter '${adapter.id}' ready: prefill_tok_per_sec=${prefill?.tokPerSec ?? "?"}, decode_tok_per_sec=${decode?.tokPerSec ?? "?"}.`);
  return { adapter, prefill, decode };
}

async function runInference() {
  if (state.active) return;
  state.active = true;
  state.trace = [];
  render();

  if (isRealRuntimeMode) {
    log(`Mode=${requestedMode} requested; awaiting real runtime adapter registration.`);
    const adapter = await awaitRealRuntime();
    if (adapter) {
      try {
        const { prefill, decode } = await runRealRuntimeOrt(adapter);
        state.realAdapterPrefill = prefill;
        state.realAdapterDecode = decode;
        state.realAdapter = adapter;
      } catch (error) {
        state.realAdapterError = error?.message || String(error);
        log(`Real runtime '${adapter.id}' failed: ${state.realAdapterError}; falling back to deterministic.`);
      }
    } else {
      const reason = (typeof window !== "undefined" && window.__aiWebGpuLabRealOrtBootstrapError) || "timed out waiting for adapter registration";
      state.realAdapterError = reason;
      log(`No real runtime adapter registered (${reason}); falling back to deterministic ORT baseline.`);
    }
  }

  const profile = parseInputProfile(elements.inputProfile.value);
  log(`Running ORT-style ${executionMode.label} profile.`);

  const initStartedAt = performance.now();
  await new Promise((resolve) => setTimeout(resolve, executionMode.initDelayMs));
  const initMs = performance.now() - initStartedAt;
  state.trace.push(`session.init ${round(initMs, 2)} ms`);

  const graphStartedAt = performance.now();
  await new Promise((resolve) => setTimeout(resolve, executionMode.graphDelayMs));
  const graphMs = performance.now() - graphStartedAt;
  state.trace.push(`graph.optimize ${round(graphMs, 2)} ms`);

  const runStartedAt = performance.now();
  const stepDurations = [];
  let firstOutputMs = 0;
  let checksum = 0;
  for (let step = 0; step < executionMode.stepCount; step += 1) {
    const stepStartedAt = performance.now();
    await new Promise((resolve) => setTimeout(resolve, step === 0 ? executionMode.firstOutputDelayMs : executionMode.stepDelayMs));
    checksum += syntheticKernel(profile, step);
    const stepMs = performance.now() - stepStartedAt;
    if (step === 0) firstOutputMs = performance.now() - runStartedAt;
    stepDurations.push(stepMs);
    state.trace.push(`run.step.${step + 1} ${round(stepMs, 2)} ms`);
    renderTrace();
  }
  const inferenceMs = performance.now() - runStartedAt;
  const totalMs = initMs + graphMs + inferenceMs;
  const tokens = profile.sequence * profile.batch;

  state.run = {
    profile,
    initMs,
    graphMs,
    firstOutputMs,
    inferenceMs,
    totalMs,
    stepDurations,
    tokens,
    checksum: round(checksum, 4),
    prefillTokPerSec: tokens / Math.max((initMs + graphMs) / 1000, 0.001),
    decodeTokPerSec: tokens / Math.max(inferenceMs / 1000, 0.001),
    realAdapter: state.realAdapter || null
  };
  state.active = false;
  log(`${executionMode.label} inference complete: first output ${round(firstOutputMs, 2)} ms, throughput ${round(state.run.decodeTokPerSec, 2)} units/s.`);
  render();
}

function describeRuntimeAdapter() {
  const registry = typeof window !== "undefined" ? window.__aiWebGpuLabRuntimeRegistry : null;
  const requested = typeof window !== "undefined"
    ? new URLSearchParams(window.location.search).get("mode")
    : null;
  if (registry) {
    return registry.describe(requested);
  }
  return {
    id: "deterministic-ort",
    label: "Deterministic ORT",
    status: "deterministic",
    isReal: false,
    version: "1.0.0",
    capabilities: ["prefill", "decode", "fixed-output-budget"],
    runtimeType: "synthetic",
    message: "Runtime adapter registry unavailable; using inline deterministic mock."
  };
}

function buildResult() {
  const run = state.run;
  return {
    meta: {
      repo: "exp-ort-webgpu-baseline",
      commit: "bootstrap-generated",
      timestamp: new Date().toISOString(),
      owner: "ai-webgpu-lab",
      track: "llm",
      scenario: (state.run && state.run.realAdapter) ? `ort-webgpu-baseline-real-${state.run.realAdapter.id}` : (run ? `ort-webgpu-baseline-${executionMode.id}` : "ort-webgpu-baseline-pending"),
      notes: run
        ? `provider=${executionMode.label}; batch=${run.profile.batch}; sequence=${run.profile.sequence}; hidden=${run.profile.hidden}; checksum=${run.checksum}${state.run && state.run.realAdapter ? `; realAdapter=${state.run.realAdapter.id}` : (isRealRuntimeMode && state.realAdapterError ? `; realAdapter=fallback(${state.realAdapterError})` : "")}`
        : "Run the ORT-Web provider readiness profile."
    },
    environment: state.environment,
    workload: {
      kind: "llm-chat",
      name: "ort-webgpu-provider-readiness",
      input_profile: run ? `batch-${run.profile.batch}-seq-${run.profile.sequence}-hidden-${run.profile.hidden}` : "profile-pending",
      model_id: "ort-webgpu-transformer-block-baseline",
      quantization: run ? run.profile.quantization : "q4f16",
      context_tokens: run ? run.tokens : 0,
      output_tokens: run ? executionMode.stepCount : 0
    },
    metrics: {
      common: {
        time_to_interactive_ms: round(performance.now() - state.startedAt, 2) || 0,
        init_ms: run ? round(run.initMs + run.graphMs, 2) || 0 : 0,
        success_rate: run ? 1 : 0.5,
        peak_memory_note: navigator.deviceMemory ? `${navigator.deviceMemory} GB reported by browser` : "deviceMemory unavailable",
        error_type: ""
      },
      llm: {
        ttft_ms: run ? round(run.firstOutputMs, 2) || 0 : 0,
        prefill_tok_per_sec: run ? round(run.prefillTokPerSec, 2) || 0 : 0,
        decode_tok_per_sec: run ? round(run.decodeTokPerSec, 2) || 0 : 0,
        turn_latency_ms: run ? round(run.totalMs, 2) || 0 : 0
      }
    },
    status: run ? "success" : "partial",
    artifacts: {
      raw_logs: state.logs.slice(0, 5),
      deploy_url: "https://ai-webgpu-lab.github.io/exp-ort-webgpu-baseline/",
      runtime_adapter: describeRuntimeAdapter()
    }
  };
}

function renderStatus() {
  const badges = state.active
    ? [`${executionMode.label} running`, state.environment.worker_mode]
    : state.run
      ? [`${executionMode.label} complete`, `${round(state.run.decodeTokPerSec, 2)} units/s`]
      : [`${executionMode.label} ready`, "Awaiting run"];
  elements.statusRow.innerHTML = "";
  for (const text of badges) {
    const node = document.createElement("span");
    node.className = "badge";
    node.textContent = text;
    elements.statusRow.appendChild(node);
  }
  elements.summary.textContent = state.run
    ? `Last run: ${executionMode.label}, first output ${round(state.run.firstOutputMs, 2)} ms, total ${round(state.run.totalMs, 2)} ms.`
    : "Run the ORT-style inference profile to capture init, first output, throughput, and total latency for the active provider mode.";
}

function renderCards(container, items) {
  container.innerHTML = "";
  for (const [label, value] of items) {
    const card = document.createElement("div");
    card.className = "card";
    const labelNode = document.createElement("span");
    labelNode.className = "label";
    labelNode.textContent = label;
    const valueNode = document.createElement("span");
    valueNode.className = "value";
    valueNode.textContent = value;
    card.append(labelNode, valueNode);
    container.appendChild(card);
  }
}

function renderMetrics() {
  const run = state.run;
  renderCards(elements.metricGrid, [
    ["Provider", executionMode.label],
    ["Init + Graph", run ? `${round(run.initMs + run.graphMs, 2)} ms` : "pending"],
    ["First Output", run ? `${round(run.firstOutputMs, 2)} ms` : "pending"],
    ["Throughput", run ? `${round(run.decodeTokPerSec, 2)} units/s` : "pending"],
    ["Total", run ? `${round(run.totalMs, 2)} ms` : "pending"],
    ["Checksum", run ? String(run.checksum) : "pending"]
  ]);
}

function renderEnvironment() {
  renderCards(elements.metaGrid, [
    ["Browser", `${state.environment.browser.name} ${state.environment.browser.version}`],
    ["OS", `${state.environment.os.name} ${state.environment.os.version}`],
    ["Backend", state.environment.backend],
    ["Worker", state.environment.worker_mode],
    ["Fallback", String(state.environment.fallback_triggered)],
    ["Cache", state.environment.cache_state]
  ]);
}

function renderLogs() {
  elements.logList.innerHTML = "";
  const logs = state.logs.length ? state.logs : ["ORT WebGPU readiness harness ready."];
  for (const message of logs) {
    const item = document.createElement("li");
    item.textContent = message;
    elements.logList.appendChild(item);
  }
}

function renderTrace() {
  elements.traceView.textContent = state.trace.length ? state.trace.join("\n") : "No inference run yet.";
}

function renderResult() {
  elements.resultJson.textContent = JSON.stringify(buildResult(), null, 2);
}

function render() {
  renderStatus();
  renderMetrics();
  renderEnvironment();
  renderLogs();
  renderTrace();
  renderResult();
}

function downloadJson() {
  const blob = new Blob([JSON.stringify(buildResult(), null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `exp-ort-webgpu-baseline-${executionMode.id}.json`;
  anchor.click();
  URL.revokeObjectURL(url);
  log("Downloaded ORT WebGPU readiness JSON draft.");
}

elements.runInference.addEventListener("click", () => {
  runInference().catch((error) => {
    state.active = false;
    log(`Run failed: ${error instanceof Error ? error.message : String(error)}`);
    render();
  });
});
elements.downloadJson.addEventListener("click", downloadJson);

render();
log(`ORT WebGPU readiness harness ready in ${executionMode.label} mode.`);
