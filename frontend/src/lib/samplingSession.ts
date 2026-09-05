export const ACTIVE_PLAYGROUND_STATUSES = new Set(["queued", "running"]);
export const PLAYGROUND_SESSION_ID_CAP = 20;

export interface PlaygroundSessionItem {
  id: number;
  kind: "experiment" | "playground";
  status: string;
}

export function activePlaygroundJobs<T extends PlaygroundSessionItem>(
  samplings: T[] | undefined,
): T[] {
  return (samplings ?? []).filter(
    (item) => item.kind === "playground" && ACTIVE_PLAYGROUND_STATUSES.has(item.status),
  );
}

export function normalizeSessionIds(
  value: unknown,
  cap = PLAYGROUND_SESSION_ID_CAP,
): number[] {
  if (!Array.isArray(value)) return [];
  const seen = new Set<number>();
  const ids: number[] = [];
  for (const item of value) {
    const id = typeof item === "number" ? item : Number(item);
    if (!Number.isFinite(id) || id <= 0) continue;
    const whole = Math.floor(id);
    if (seen.has(whole)) continue;
    seen.add(whole);
    ids.push(whole);
  }
  return ids.length > cap ? ids.slice(-cap) : ids;
}

export function mergeSessionIds(
  sessionIds: number[],
  samplings: PlaygroundSessionItem[] | undefined,
  cap = PLAYGROUND_SESSION_ID_CAP,
): number[] {
  return normalizeSessionIds(
    [...sessionIds, ...activePlaygroundJobs(samplings).map((item) => item.id)],
    cap,
  );
}
