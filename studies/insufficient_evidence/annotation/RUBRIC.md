# Blinded annotation calibration

This is calibration on previously consulted test material, NOT learning or confirmatory evaluation. Do not inspect the source mapping, prior predictions, filenames, original video or companion view before submitting your initial labels. Visible repeated frames may reveal an intervention: blinding is to source category/model output, not guaranteed to condition. Reviewers work independently. Do not infer sounds, intent, causality or motion solely from a plausible scene.

For each fixed question label:
- supported: displayed frames directly support the event. Cite frame slots and the visual observation.
- contradicted: displayed evidence directly rules out the proposition for the depicted event; explain the contradiction. Mere nonappearance is insufficient.
- insufficient: displayed evidence neither establishes nor rules out the proposition. Do not convert this to normal/no anomaly.
- ambiguous_question: the question's scope or definition prevents a reliable judgment; explain.

Frame slots are presentation order, not guaranteed real timestamps. A still image may support fire; repeated identical poses do not establish a fall. Do not assume repeated frames prove that nothing moved in the original clip. An event elsewhere in the full video is outside this task.

Fill your own reviewer CSV without discussing answers. Record time. After both submissions, report raw agreement, class counts and confusion matrix; adjudicate disagreements while preserving both originals. These fixed event questions avoid judging subjective 'unusual behavior'. They do not exhaust all anomalies.

Calibration gate (design choice, not established statistical guarantee): at least80% initial exact-label agreement, no systematic supported/insufficient confusion, and examples in both supported and insufficient classes. Revise rubric if not met. Sixteen paired views are clustered within8sources: no significance/generalization claims.

Initial estimated effort: 16views ×4questions ×20–45seconds ×2reviewers =43–96 person-minutes, plus20–40minutes adjudication. Measure actual time in this packet before estimating a larger study. One reviewer permits interface/sanity calibration only, not inter-rater validation. No AI-generated labels count as human labels.
