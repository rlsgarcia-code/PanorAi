export interface ClosedFillOptions {
  support?: Uint8Array | null;
  excluded?: Uint8Array | null;
  positive?: Uint8Array | null;
  /** A contour spanning every longitude needs an explicit visible polar side. */
  pole?: "north" | "south";
}

/**
 * Fill enclosed background components on the native ERP grid in place.
 * The shortest periodic longitude ROI avoids treating the seam as a gap.
 * Four-connected background is flooded from the ROI exterior; foreground
 * strokes are the barriers. Support and negative seeds restrict publication,
 * not topology. Open contours therefore never acquire a fabricated closing edge.
 * Existing foreground is preserved. Returns the number of newly filled pixels.
 */
export function fillClosedContours(
  values: Uint8Array,
  shape: [number, number],
  options: ClosedFillOptions = {},
): number {
  const [height, width] = shape,
    size = height * width;
  if (
    !Number.isInteger(height) ||
    !Number.isInteger(width) ||
    height < 1 ||
    width < 1 ||
    !Number.isSafeInteger(size) ||
    size > 160000000 ||
    values.length !== size
  )
    throw new Error("Grade de preenchimento inválida.");
  for (const mask of [options.support, options.excluded, options.positive])
    if (mask && mask.length !== size)
      throw new Error("Máscara auxiliar incompatível.");
  let minY = height,
    maxY = -1;
  const columns = new Uint8Array(width);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++)
      if (values[y * width + x]) {
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
        columns[x] = 1;
      }
  if (maxY < 0) return 0;
  let gap = 0,
    gapEnd = 0,
    run = 0;
  for (let i = 0; i < 2 * width; i++) {
    run = columns[i % width] ? 0 : run + 1;
    if (run > gap) {
      gap = run;
      gapEnd = i;
    }
  }
  if (!gap && !options.pole) return 0; // no unambiguous spherical outside
  const start = (((gapEnd + 1) % width) + width - 1) % width;
  const roiWidth = gap ? Math.min(width, width - gap + 2) : width;
  let y0 = Math.max(0, minY - 1),
    y1 = Math.min(height - 1, maxY + 1);
  if (!gap && options.pole === "north") y0 = 0;
  if (!gap && options.pole === "south") y1 = height - 1;
  const roiHeight = y1 - y0 + 1,
    count = roiWidth * roiHeight;
  const state = new Uint8Array(count),
    queue = new Uint32Array(count);
  for (let y = 0; y < roiHeight; y++)
    for (let x = 0; x < roiWidth; x++)
      state[y * roiWidth + x] = values[(y + y0) * width + ((start + x) % width)]
        ? 1
        : 0;
  let head = 0,
    tail = 0;
  const visit = (index: number) => {
    if (!state[index]) {
      state[index] = 2;
      queue[tail++] = index;
    }
  };
  if (gap || options.pole !== "north")
    for (let x = 0; x < roiWidth; x++) visit(x);
  if (gap || options.pole !== "south")
    for (let x = 0; x < roiWidth; x++) visit((roiHeight - 1) * roiWidth + x);
  if (gap)
    for (let y = 0; y < roiHeight; y++) {
      visit(y * roiWidth);
      visit(y * roiWidth + roiWidth - 1);
    }
  if (!tail) return 0;
  const wrap = roiWidth === width;
  while (head < tail) {
    const index = queue[head++],
      x = index % roiWidth,
      y = Math.floor(index / roiWidth);
    if (x > 0) visit(index - 1);
    else if (wrap) visit(index + roiWidth - 1);
    if (x < roiWidth - 1) visit(index + 1);
    else if (wrap) visit(index - roiWidth + 1);
    if (y > 0) visit(index - roiWidth);
    if (y < roiHeight - 1) visit(index + roiWidth);
  }
  let filled = 0;
  for (let y = 0; y < roiHeight; y++)
    for (let x = 0; x < roiWidth; x++) {
      if (state[y * roiWidth + x]) continue;
      const index = (y + y0) * width + ((start + x) % width);
      if (
        (options.support && !options.support[index]) ||
        options.excluded?.[index]
      )
        continue;
      values[index] = 1;
      if (options.positive) options.positive[index] = 1;
      filled++;
    }
  return filled;
}
