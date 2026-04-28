// Real ONNX Runtime Web integration sketch for exp-ort-webgpu-baseline.
//
// Gated by ?mode=real-ort. Default deterministic harness path is untouched.
// `loadOrtFromCdn` is parameterized so tests can inject a stub.

const DEFAULT_ORT_VERSION = "1.20.0";
const DEFAULT_ORT_CDN = (version) => `https://esm.sh/onnxruntime-web@${version}`;
const DEFAULT_MODEL_URL = "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx";

export async function loadOrtFromCdn({ version = DEFAULT_ORT_VERSION } = {}) {
  const ort = await import(/* @vite-ignore */ DEFAULT_ORT_CDN(version));
  if (!ort || !ort.InferenceSession || typeof ort.InferenceSession.create !== "function") {
    throw new Error("onnxruntime-web module did not expose InferenceSession.create");
  }
  return { ort, InferenceSession: ort.InferenceSession, Tensor: ort.Tensor };
}

export function buildRealOrtAdapter({
  ort,
  InferenceSession,
  Tensor,
  version = DEFAULT_ORT_VERSION,
  modelUrl = DEFAULT_MODEL_URL,
  executionProviders = ["webgpu"]
}) {
  if (!InferenceSession || typeof InferenceSession.create !== "function") {
    throw new Error("buildRealOrtAdapter requires InferenceSession.create");
  }
  const slug = modelUrl.replace(/[^A-Za-z0-9]/g, "-").toLowerCase().slice(-48);
  const id = `ort-webgpu-${slug}-${version.replace(/[^0-9]/g, "")}`;
  let session = null;

  return {
    id,
    label: `ONNX Runtime Web ${version} (${executionProviders.join("/")})`,
    version,
    capabilities: ["prefill", "decode", "ort-session", "fixed-output-budget"],
    loadType: "async",
    backendHint: executionProviders[0] === "webgpu" ? "webgpu" : "wasm",
    isReal: true,
    async loadRuntime({ providers = executionProviders } = {}) {
      session = await InferenceSession.create(modelUrl, { executionProviders: providers });
      return session;
    },
    async prefill(_runtime, prompt) {
      const startedAt = performance.now();
      const tokens = (prompt && prompt.tokens) || (typeof prompt === "string" ? prompt.split(/\s+/).filter(Boolean).length : 0);
      const inputs = (prompt && prompt.inputs) || null;
      const prefillMs = performance.now() - startedAt;
      return { promptTokens: tokens, prefillMs, inputs };
    },
    async decode(activeSession, prefillResult, outputTokenBudget = 1) {
      const target = activeSession || session;
      if (!target || typeof target.run !== "function") {
        throw new Error("real ort adapter requires loadRuntime() before decode()");
      }
      const inputs = (prefillResult && prefillResult.inputs) || {};
      const startedAt = performance.now();
      const output = await target.run(inputs);
      const decodeMs = performance.now() - startedAt;
      const outputs = output && typeof output === "object" ? Object.keys(output) : [];
      let dimensions = 0;
      if (outputs.length > 0) {
        const first = output[outputs[0]];
        dimensions = first && first.dims && first.dims[first.dims.length - 1] ? first.dims[first.dims.length - 1] : 0;
      }
      return {
        tokens: outputTokenBudget,
        decodeMs,
        outputCount: outputs.length,
        dimensions,
        ttftMs: decodeMs,
        decodeTokPerSec: outputTokenBudget / Math.max(decodeMs / 1000, 0.001)
      };
    }
  };
}

export async function connectRealOrt({
  registry = typeof window !== "undefined" ? window.__aiWebGpuLabRuntimeRegistry : null,
  loader = loadOrtFromCdn,
  version = DEFAULT_ORT_VERSION,
  modelUrl = DEFAULT_MODEL_URL,
  executionProviders = ["webgpu"]
} = {}) {
  if (!registry) {
    throw new Error("runtime registry not available");
  }
  const { ort, InferenceSession, Tensor } = await loader({ version });
  if (!InferenceSession) {
    throw new Error("loaded module is missing InferenceSession");
  }
  const adapter = buildRealOrtAdapter({ ort, InferenceSession, Tensor, version, modelUrl, executionProviders });
  registry.register(adapter);
  return { adapter, ort, InferenceSession };
}

if (typeof window !== "undefined" && window.location && typeof window.location.search === "string") {
  const params = new URLSearchParams(window.location.search);
  if (params.get("mode") === "real-ort" && !window.__aiWebGpuLabRealOrtBootstrapping) {
    window.__aiWebGpuLabRealOrtBootstrapping = true;
    connectRealOrt().catch((error) => {
      console.warn(`[real-ort] bootstrap failed: ${error.message}`);
      window.__aiWebGpuLabRealOrtBootstrapError = error.message;
    });
  }
}
