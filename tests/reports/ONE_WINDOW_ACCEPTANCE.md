# One-window real-model acceptance: NOT PASSED

Attempted 2026-10-23, using the new configurable runner and the staged InternVL2-8B checkpoint.

Scope: one RoadAccidents011 window, original eight frames, 448-square inputs, baseline/no intervention, BF16, unchanged VERA prompt and greedy generation (maximum 1024 new tokens). No mock outputs were substituted.

- MPS attempt from the agent process: failed before checkpoint loading because MPS is inaccessible in this execution context. Artifacts: tests/artifacts/real_model/run 3/output/.
- CPU attempt: checkpoint loaded and video preprocessing succeeded; vision forward completed and the request entered language generation. Timed out at the configured 300-second wall limit with zero complete responses. Artifacts: tests/artifacts/real_model_cpu/run 1/output/.
- CPU test report: tests/reports/real-model/20261023T211659848082Z/.

Conclusion: the new runner has not passed real-model one-window acceptance on this Mac within the five-minute test budget. This does not establish that generation can never complete; it is an unsuccessful bounded execution test. The earlier vanilla run 7 did return a real strict label 0 for the same opening window after 5838.89 seconds (~97.3 minutes), but that older output is not counted as a pass for the new adapter.

No further heavy run is left running by these tests. No push was performed. The test budget, backbone, input count, precision and response requirements were not relaxed to manufacture a pass.
