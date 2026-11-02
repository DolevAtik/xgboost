/** File layout of public/snapshot/, shared by scripts/snapshot.ts and the static client. */

const slug = (s: string) => s.replace(/[^A-Za-z0-9._-]+/g, "_");

export const snapshotPredictPath = (id: string, models: string[]) =>
  `predict/${slug(id)}__${models.join("-")}.json`;

export const snapshotDrivePath = (token: string, serial: string) => `drives/${slug(token)}/${slug(serial)}.json`;
