import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const root = new URL("../", import.meta.url);

async function source() {
  return readFile(new URL("app/page.tsx", root), "utf8");
}

test("production route renders with product metadata", async () => {
  const workerUrl = new URL("dist/server/index.js", root);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  const response = await worker.fetch(new Request("http://localhost/"), {
    ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) },
  }, { waitUntil() {}, passThroughOnException() {} });
  assert.equal(response.status, 200);
  assert.match(await response.text(), /EvidenceGraph/);
});

test("evidence cards use the canonical detail endpoint and accessible interaction", async () => {
  const page = await source();
  assert.match(page, /fetch\(`\$\{API\}\/api\/evidence\/\$\{encodeURIComponent\(evidenceId\)\}`\)/);
  assert.match(page, /aria-label=\{`Inspect \$\{node\.modality\} evidence \$\{node\.id\}`\}/);
  assert.match(page, /onClick=\{\(\) => void inspectEvidence\(node\.id\)\}/);
  assert.match(page, /Connected evidence/);
});

test("explorer renders backend relationship fields without inventing semantic scores", async () => {
  const page = await source();
  assert.match(page, /edge\.relationship_type/);
  assert.match(page, /signal\(edge\.confidence\)/);
  assert.match(page, /signal\(edge\.signal_breakdown\.temporal\)/);
  assert.match(page, /signal\(edge\.signal_breakdown\.entity\)/);
  assert.match(page, /signal\(edge\.signal_breakdown\.semantic\)/);
  assert.match(page, /edge\.explanation/);
  assert.match(page, /return typeof value === "number" && Number\.isFinite\(value\) \? value\.toFixed\(2\) : "—"/);
});

test("baseline scores are defensive and production logic contains no corpus IDs", async () => {
  const page = await source();
  assert.match(page, /score\?: number \| null/);
  assert.match(page, /safeScore\(item\.score\)/);
  assert.match(page, /Number\.isFinite\(value\)/);
  assert.doesNotMatch(page, /\b(?:ev|src)_[0-9a-z_]+/i);
  assert.match(page, /demo\?\.sources\.map/);
});
