import http from "k6/http";
import { check, sleep, group } from "k6";
import { Counter, Trend } from "k6/metrics";

export let options = {
  tags: { test: "healthz" },
  thresholds: {
    http_req_failed: ["rate<0.01"],
    http_req_duration: ["p(95)<250"],
  },
  scenarios: {
    ramp_healthz: {
      executor: "ramping-arrival-rate",
      startRate: 5,
      timeUnit: "1s",
      preAllocatedVUs: 10,
      maxVUs: 50,
      stages: [
        { target: 20, duration: "30s" },
        { target: 50, duration: "1m" },
        { target: 0, duration: "10s" },
      ],
    },
  },
};

const OK = new Counter("healthz_ok");
const LAT = new Trend("healthz_latency_ms");
const BASE = __ENV.BASE_URL || "http://localhost:8080";

export default function () {
  group("GET /healthz", () => {
    const res = http.get(`${BASE}/healthz`);
    const good =
      check(res, {
        "status is 200": (r) => r.status === 200,
        "body ok==true": (r) => {
          try {
            return JSON.parse(r.body).ok === true;
          } catch (e) {
            return false;
          }
        },
      }) || false;

    if (good) OK.add(1);
    LAT.add(res.timings.duration);
  });

  sleep(0.2);
}

// Save JSON summary (no external HTML report)
export function handleSummary(data) {
  return {
    "/results/healthz-summary.json": JSON.stringify(data, null, 2),
  };
}
