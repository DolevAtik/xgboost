import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));

/** The XGBOOST project root: models/, results/, demo/ and webapp/ live here. */
export const PROJECT_ROOT = path.resolve(here, "..", "..");
export const RESULTS_DIR = path.join(PROJECT_ROOT, "results");
export const DEMO_DIR = path.join(PROJECT_ROOT, "demo");
export const MODELS_DIR = path.join(PROJECT_ROOT, "models");
export const DIST_DIR = path.resolve(here, "..", "dist");

export const HOST = process.env.HOST ?? "127.0.0.1";
export const PORT = Number(process.env.API_PORT ?? 4000);
export const FLASK_URL = (process.env.FLASK_URL ?? "http://127.0.0.1:5000").replace(/\/$/, "");

/** Same ceiling the Flask app enforces (webapp/app.py MAX_CONTENT_LENGTH). */
export const MAX_UPLOAD_BYTES = 400 * 1024 * 1024;

/** The evaluation files the story reads. Prefix of results/w2022x_*.{csv,json}. */
export const RESULTS_PREFIX = "w2022x";
