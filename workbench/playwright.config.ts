import { defineConfig, devices } from "@playwright/test";

const externalBaseUrl = process.env.PLAYWRIGHT_BASE_URL;
const frontendPort = 5190;
const backendPort = 8019;

export default defineConfig({
  testDir: "./e2e",
  outputDir: "./test-results",
  timeout: 90_000,
  expect: { timeout: 30_000 },
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["line"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL: externalBaseUrl ?? `http://127.0.0.1:${frontendPort}`,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: externalBaseUrl
    ? undefined
    : [
        {
          command: `python -m uvicorn panorai_workbench.api:app --app-dir backend --host 127.0.0.1 --port ${backendPort}`,
          cwd: ".",
          env: {
            PYTHONPATH: `backend${process.platform === "win32" ? ";" : ":"}..`,
            PANORAI_WORKBENCH_HOME:
              process.env.PANORAI_WORKBENCH_HOME ?? "/tmp/panorai-workbench-e2e",
          },
          url: `http://127.0.0.1:${backendPort}/api/v1/health/ready`,
          reuseExistingServer: !process.env.CI,
          timeout: 120_000,
        },
        {
          command: `npm run dev -- --port ${frontendPort}`,
          cwd: ".",
          env: {
            PANORAI_WORKBENCH_API_URL: `http://127.0.0.1:${backendPort}`,
          },
          url: `http://127.0.0.1:${frontendPort}`,
          reuseExistingServer: !process.env.CI,
          timeout: 120_000,
        },
      ],
});
