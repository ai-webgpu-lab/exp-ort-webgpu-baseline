# Results

## 1. 실험 요약
- 저장소: exp-ort-webgpu-baseline
- 커밋 해시: 93a915c
- 실험 일시: 2026-05-20T15:44:44.863Z -> 2026-05-20T15:44:49.908Z
- 담당자: ai-webgpu-lab
- 실험 유형: `llm`
- 상태: `success`

## 2. 질문
- ORT-Web style provider baseline에서 WebGPU와 Wasm fallback 메타데이터를 같은 포맷으로 기록할 수 있는가
- 같은 입력 프로필에서 first output latency와 throughput 차이가 재현되는가
- 실제 ONNX Runtime Web integration 전에 provider/run protocol을 고정할 수 있는가

## 3. 실행 환경
### 브라우저
- 이름: Chrome
- 버전: 147.0.7727.15

### 운영체제
- OS: Linux
- 버전: unknown

### 디바이스
- 장치명: Linux x86_64
- device class: `desktop-high`
- CPU: 16 threads
- 메모리: 32 GB
- 전원 상태: `unknown`

### GPU / 실행 모드
- adapter: ort-webgpu-simulated-adapter, wasm-fallback-simulated
- backend: `webgpu, wasm`
- fallback triggered: `false, true`
- worker mode: `worker, main`
- cache state: `warm`
- required features: ["shader-f16"], []
- limits snapshot: {}

## 4. 워크로드 정의
- 시나리오 이름: ORT WebGPU Provider, ORT Wasm Fallback
- 입력 프로필: batch-1-seq-128-hidden-384
- 데이터 크기: provider=WebGPU EP; batch=1; sequence=128; hidden=384; checksum=-248.4516; automation=playwright-chromium, provider=Wasm Fallback; batch=1; sequence=128; hidden=384; checksum=-248.4516; automation=playwright-chromium, provider=WebGPU EP; batch=1; sequence=128; hidden=384; checksum=-248.4516; realAdapter=fallback(adapter.loadModel is not a function); automation=playwright-chromium
- dataset: -
- model_id 또는 renderer: ort-webgpu-transformer-block-baseline
- 양자화/정밀도: q4f16
- resolution: -
- context_tokens: 128
- output_tokens: 12

## 5. 측정 지표
### 공통
- time_to_interactive_ms: 399 ~ 1141.9 ms
- init_ms: 88.3 ~ 167.5 ms
- success_rate: 1
- peak_memory_note: 32 GB reported by browser
- error_type: -

### LLM / Benchmark
- ttft_ms: 18.2 ~ 42.3 ms
- prefill_tok_per_sec: 764.18 ~ 1449.6 tok/s
- decode_tok_per_sec: 458.45 ~ 1002.35 tok/s
- turn_latency_ms: 216 ~ 446.7 ms
- backends: webgpu, wasm
- fallback states: false, true

## 6. 결과 표
| Run | Scenario | Backend | Cache | Mean | P95 | Notes |
|---|---|---:|---:|---:|---:|---|
| 1 | ORT WebGPU Provider | webgpu | warm | 937.73 | 18.5 | prefill=1430.17 tok/s, metric=decode tok/s / TTFT ms |
| 2 | ORT Wasm Fallback | wasm | warm | 458.45 | 42.3 | prefill=764.18 tok/s, metric=decode tok/s / TTFT ms |
| 3 | ORT WebGPU Provider | webgpu | warm | 1002.35 | 18.2 | prefill=1449.6 tok/s, metric=decode tok/s / TTFT ms |

## 7. 관찰
- ORT provider readiness baseline은 backend=webgpu, worker_mode=worker로 기록됐다.
- WebGPU provider throughput=937.73 units/s, fallback throughput=458.45 units/s였다.
- playwright-chromium로 수집된 automation baseline이며 headless=true, browser=Chromium 147.0.7727.15.
- 실제 runtime/model/renderer 교체 전 deterministic harness 결과이므로, 절대 성능보다 보고 경로와 재현성 확인에 우선 의미가 있다.

## 8. WebGPU vs Fallback
- provider readiness: webgpu=ORT WebGPU Provider, fallback=ORT Wasm Fallback
- throughput units/s: webgpu=937.73, fallback=458.45, delta=+479.28
- first output: webgpu=18.5 ms, fallback=42.3 ms, delta=-23.8 ms

## 9. Real Adapter vs Deterministic
- adapter: real=ort-webgpu-va-all-minilm-l6-v2-resolve-main-onnx-model-onnx-1200, deterministic=deterministic-mock
- adapter_run: real=connected, deterministic=deterministic
- success_rate: real=1, deterministic=1

## 10. 결론
- ORT-Web provider readiness가 WebGPU와 Wasm fallback pair를 raw JSON과 RESULTS.md 양쪽에 남기게 됐다.
- 다음 단계는 deterministic transformer-block fixture를 실제 ONNX Runtime Web session/load/run path로 교체하는 것이다.
- real model asset, execution provider 설정, wasm/webgpu package build 경로를 같은 결과 포맷에 연결해야 한다.

## 11. 첨부
- 스크린샷: ./reports/screenshots/01-ort-webgpu-provider.png, ./reports/screenshots/02-ort-wasm-fallback.png, ./reports/screenshots/10-ort-webgpu-baseline-real-ort.png
- 로그 파일: ./reports/logs/01-ort-webgpu-provider.log, ./reports/logs/02-ort-wasm-fallback.log, ./reports/logs/10-ort-webgpu-baseline-real-ort.log
- raw json: ./reports/raw/01-ort-webgpu-provider.json, ./reports/raw/02-ort-wasm-fallback.json, ./reports/raw/10-ort-webgpu-baseline-real-ort.json
- 배포 URL: https://ai-webgpu-lab.github.io/exp-ort-webgpu-baseline/
- 관련 이슈/PR: -
