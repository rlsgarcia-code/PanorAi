import { expect, test, type Page } from "@playwright/test";

function failOnBrowserErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("pageerror", (error) => errors.push(error.message));
  return errors;
}

test("goal discovery starts from a public ERP and prepares without executing", async ({
  page,
  request,
}) => {
  const errors = failOnBrowserErrors(page);
  const runsBefore = await request.get("/api/v1/runs?limit=100");
  const beforeCount = (await runsBefore.json()).runs.length;

  await page.goto("/");
  await expect(page.getByRole("heading", { name: "O que posso fazer com esta imagem?" })).toBeVisible();
  await expect(page.locator("#inputName")).toHaveText("Poly Haven Studio");
  await expect(page.locator("#discoveryImage")).toHaveAttribute("src", /blob:/);
  await expect(
    page.getByRole("button", { name: /Analisar máscaras e resultados/ }),
  ).toContainText("falta contexto");

  await page.getByRole("searchbox", { name: "Buscar um problema" }).fill("encontrar pontos");
  await expect(page.getByRole("button", { name: /Detectar e descrever features/ })).toBeVisible();
  await page.getByRole("button", { name: /Detectar e descrever features/ }).click();
  await page.getByRole("button", { name: /Montar pipeline/ }).click();
  await expect(page.getByText("Nenhuma execução foi iniciada.")).toBeVisible();
  await page.getByRole("link", { name: /Abrir bancada para revisar/ }).click();
  await expect(page.locator("#inputState")).toHaveText("entrada preparada · revise antes de executar");

  const runsAfter = await request.get("/api/v1/runs?limit=100");
  expect((await runsAfter.json()).runs).toHaveLength(beforeCount);
  expect(errors).toEqual([]);
});

test("discovery handoff keeps the current Workbench visual system in Runner", async ({
  page,
}) => {
  const errors = failOnBrowserErrors(page);

  await page.goto("/");
  await page.getByRole("button", { name: /Projetar e navegar/ }).click();
  await page.getByRole("button", { name: /Montar pipeline/ }).click();
  await page.getByRole("link", { name: /Revisar e editar no Runner/ }).click();

  await expect(page).toHaveURL(/\/runner\.html/);
  await expect(
    page.getByText("Receita preparada · Abrir uma vista gnomônica"),
  ).toBeVisible();
  await expect(page.getByRole("link", { name: "O que posso fazer?" })).toBeVisible();
  await expect(page.getByRole("link", { name: "Catálogo técnico" })).toBeVisible();
  await expect(page.getByRole("link", { name: "Multi View" })).toBeVisible();

  const tokens = await page.locator(".runner-shell").evaluate((element) => {
    const root = getComputedStyle(document.documentElement);
    const eyebrow = getComputedStyle(
      element.querySelector<HTMLElement>(".eyebrow")!,
    );
    const body = getComputedStyle(document.body);
    return {
      accent: root.getPropertyValue("--accent").trim(),
      accentText: eyebrow.color,
      surface: body.backgroundColor,
    };
  });
  expect(tokens).toEqual({
    accent: "#ffc857",
    accentText: "rgb(255, 200, 87)",
    surface: "rgb(9, 11, 9)",
  });
  expect(errors).toEqual([]);
});

test("typed pair context unlocks relative pose and preserves the technical catalog", async ({ page }) => {
  const errors = failOnBrowserErrors(page);
  await page.goto("/");
  await page.getByRole("button", { name: "Par público + pose" }).click();
  const pairGoal = page.getByRole("button", { name: /Relacionar duas vistas/ });
  await expect(pairGoal).toContainText("pronto agora");
  await pairGoal.click();
  await expect(page.getByRole("heading", { name: "Estimar pose relativa esférica" })).toBeVisible();

  await page.getByRole("button", { name: "Catálogo técnico" }).click();
  await expect(page.locator("#technicalView")).toBeVisible();
  await expect(page.locator(".component-row").first()).toBeVisible();
  expect(errors).toEqual([]);
});

test("all workspaces load their public default inputs", async ({ page }) => {
  const errors = failOnBrowserErrors(page);

  await page.goto("/single-view.html?experiment=filters-edges");
  await expect(page.locator("#inputState")).toHaveText("exemplo CC0 carregado");
  await expect(page.locator("#fileName")).toContainText("trocar ERP");

  await page.goto("/two-view.html?experiment=relative-pose");
  await expect(page.locator("#nameA")).toHaveText("studio-known-pose-a.png");
  await expect(page.locator("#nameB")).toHaveText("studio-known-pose-b.png");
  await expect(page.locator("#runPair")).toBeEnabled();

  await page.goto("/multi-view.html");
  await expect(page.locator("#revision")).toHaveText("Exemplo CC0 · ainda não salvo");
  await expect(page.locator("#view-a option")).toHaveCount(2);
  await expect(page.locator("#view-b option")).toHaveCount(2);

  expect(errors).toEqual([]);
});

test("catalog-derived Single View route opens the specialized planar workspace", async ({
  page,
}) => {
  const errors = failOnBrowserErrors(page);
  await page.goto(
    "/single-view.html?experiment=known-height-planar-mapping",
  );
  await expect(page).toHaveURL(/\/known-height-planar\.html$/);
  await expect(
    page.getByRole("heading", { name: "Referência métrica" }),
  ).toBeVisible();
  expect(errors).toEqual([]);
});

test("real default scene gates height, alignment, and segmentation in order", async ({
  page,
}) => {
  const errors = failOnBrowserErrors(page);
  await page.goto("/known-height-planar.html");

  const height = page.getByLabel("Altura medida (m) · obrigatória");
  await expect(height).toBeVisible();
  await expect(page.locator("#status")).toContainText("Descoberta pronta", {
    timeout: 90_000,
  });
  await expect(page.locator("#alignmentStep")).toHaveClass(/locked/);
  await expect(page.locator("#segmentationStep")).toHaveClass(/locked/);

  await height.fill("1.62");
  await page.getByLabel("Incerteza (m) · opcional").fill("0.02");
  await page.getByRole("button", { name: "Usar esta altura" }).click();

  await expect(page.locator("#heightState")).toHaveText("1.62 m informada");
  await expect(page.locator("#alignmentStep")).not.toHaveClass(/locked/);
  await expect(page.locator("#segmentationStep")).toHaveClass(/locked/);
  expect(errors).toEqual([]);
});

test("analytic planar reference executes the public pipeline and opens 3D", async ({
  page,
}) => {
  const errors = failOnBrowserErrors(page);
  await page.goto("/known-height-planar.html");
  await page.getByRole("button", { name: "Solução analítica" }).click();

  const validate = page.getByRole("button", { name: "Validar alinhamento" });
  await expect(validate).toBeEnabled();
  await validate.click();

  await expect(page.locator("#previewState")).toHaveText(
    "Resolvido · publicável",
  );
  await expect(page.locator("#resolved")).toHaveText("3");
  await expect(page.locator("#pixels")).not.toHaveText("0");
  await expect(page.getByText("Nuvem métrica · point-cloud")).toBeVisible();

  await page.getByRole("button", { name: "Preview 3D" }).click();
  await expect(page.locator("#previewDock")).not.toHaveClass(/collapsed/);
  expect(errors).toEqual([]);
});
