# Judge Demo

1. Start the cached API and React UI using the commands in `README.md`.
2. Confirm the header says **Cached demo ready** and shows 4 sources, 333 evidence nodes, and 451 relationships.
3. Run the prefilled flagship question in **EvidenceGraph** mode.
4. Point out direct versus graph-connected cards, speaker/timestamps, frame/image provenance, PDF pages, and **Why these connect**.
5. Switch to **Text baseline** and run the identical question.
6. Use the frozen V1.5 metric strip to explain the honest tradeoff: baseline has higher exact recall, while EvidenceGraph has substantially stronger provenance completeness and visual evidence.

The answer formatter is deterministic and uses no external LLM. CLIP is optional; set `DISABLE_CLIP=1` before starting the API to demonstrate the text/entity plus graph fallback.
