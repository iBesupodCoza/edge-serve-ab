import http from "k6/http";
import { check, sleep, group } from "k6";
import { Rate, Trend } from "k6/metrics";

// Treat 2xx + 304 + 429 as "expected" (not counted in http_req_failed)
// so we can push the system and still track true failures separately.
http.setResponseCallback(http.expectedStatuses({ min: 200, max: 299 }, 304, 429));

export let options = {
  tags: { test: "infer" },
  thresholds: {
    // Only unexpected failures (e.g., 5xx, 4xx other than 429) should be <5%
    "http_req_failed{expected_response:true}": ["rate<0.05"],

    // SLOs on successful/expected responses only
    "http_req_duration{expected_response:true}": ["p(95)<1200"],

    // Custom rates
    infer_2xx: ["rate>0.10"],      // prove we got some success at each stage
    infer_5xx: ["rate<0.01"],      // real errors must be rare
    // At high RPS we *expect* throttling; tune this bound to your target envelope
    infer_429: ["rate<0.85"],

    // Optional: ensure our functional checks mostly pass
    "checks{group:::POST /v1/infer}": ["rate>0.95"],
  },
  scenarios: {
    ramp_infer: {
      executor: "ramping-arrival-rate",
      startRate: 5,
      timeUnit: "1s",
      preAllocatedVUs: 20,
      maxVUs: 200,
      stages: [
        { target: 20, duration: "30s" },
        { target: 60, duration: "1m" },
        { target: 100, duration: "1m" },
        { target: 0, duration: "15s" },
      ],
    },
  },
};

const SUCC_2XX = new Rate("infer_2xx");  // change to Rate
const RATE_429 = new Rate("infer_429");  // change to Rate
const RATE_5XX = new Rate("infer_5xx");  // change to Rate
const LAT = new Trend("infer_latency_ms");

const BASE = __ENV.BASE_URL || "http://localhost:8080";

// 1x1 transparent PNG (base64)
const TINY_PNG_B64 =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAA" +
  "AAC0lEQVR42mP8/58BAgMDAwP9PwAAH4YB2u0XxW8AAAAASUVORK5CYII=";

export default function () {
  group("POST /v1/infer", () => {
    const payload = JSON.stringify({ image_b64: TINY_PNG_B64, img_size: 224 });
    const params = {
      headers: {
        "Content-Type": "application/json",
        "X-Request-ID": `k6-${__VU}-${__ITER}`,
      },
    };

    const res = http.post(`${BASE}/v1/infer`, payload, params);

    // functional checks (keep loose so throttled responses don’t fail this)
    const ok =
      check(res, {
        "status 200 or 429": (r) => r.status === 200 || r.status === 429,
        // accept any of these trace header conventions
        "has trace id header":
          (r) =>
            (r.headers["Trace-Id"] || r.headers["X-Trace-Id"] || r.headers["TraceId"] || "").length > 0,
      }) || false;

    SUCC_2XX.add(res.status >= 200 && res.status < 300);
    RATE_429.add(res.status === 429);
    RATE_5XX.add(res.status >= 500);
    LAT.add(res.timings.duration);
  });

  sleep(0.2);
}

export function handleSummary(data) {
  return { "/results/infer-summary.json": JSON.stringify(data, null, 2) };
}
