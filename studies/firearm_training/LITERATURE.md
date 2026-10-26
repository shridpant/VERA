# Literature check: the variants of Training 1 (2026-10-26)

A bounded check, not a systematic review: 9 web searches on 2026-10-26 00:00 ET, read from
abstracts, project pages and search summaries. Nothing was downloaded beyond those pages. It
asked whether the two planned variants are new and how they relate to known work.

## Contents

- [Findings](#findings)
- [What this means for the variants](#what-this-means-for-the-variants)
- [Sources](#sources)

## Findings

- **VERA** (CVPR 2025) learns at most five guiding questions with a learner and an optimizer VLM
  from coarse labels. The searches found no follow-up that changes its optimizer.
- **Confusion-driven concept generation is established for images:**
  - **ESCHER** (Sehgal et al., CVPR 2025) evolves a concept library. It finds the class pairs a
    concept-bottleneck classifier confuses, using a VLM critic, and an LLM writes new
    discriminative concepts for them.
  - **FuDD** (Esfandiarpoor & Bach, ICLR 2024) re-classifies among the classes a first pass
    found ambiguous, using LLM-written differential descriptions. Most of its gain comes from
    describing the differences between the most ambiguous classes.
  - **LLM-Mutate** (Chiquier, Mall & Vondrick, ECCV 2024) evolves the attributes of an
    interpretable classifier; the LLM sees past attributes with their scores.
  - **CAPT** (CVPR 2026) learns continuous prompts from confusion pairs.
  - **Yan et al.** (ICCV 2023) select a few descriptive attributes from a large LLM-generated
    pool.
- **Prompt optimisation with scores and errors:**
  - OPRO shows the optimizer past prompts with their scores.
  - ProTeGi turns errors into "textual gradients" and uses beam search.
  - APE proposes candidates and keeps the best.
  - **MPO** (2025) extends prompt optimisation to multimodal tasks, video included.
  - A 2026 preprint argues that text-only optimizers are "blind" to the visual evidence.
- **Questions or concepts for video:**
  - A temporal concept bottleneck (2025) discovers concepts with a VLM for video
    classification.
  - The searches found no method that learns a bank of yes/no questions which a frozen
    generative VLM proposes, answers and is scored by, inside VERA's procedure.

## What this means for the variants

- **VERA, fixed** combines known ideas: balanced larger batches; OPRO's score history;
  accept-if-not-worse, as in APE and ProTeGi's selection; and a short-answer constraint. It is
  an engineering variant, not a new method.
- **Confusion-pair question bank** adapts ESCHER and FuDD (confusion-targeted discriminative
  concepts) to VERA's setting:
  - video;
  - one frozen generative VLM as proposer, answerer and scorer;
  - natural-language yes/no questions scored by the model's own answers;
  - VERA's explanation-style output kept.

  This bounded check found no paper with that combination. It is still reported as an
  adaptation, not a new method, and any wider claim needs a systematic search.

## Sources

- VERA: https://arxiv.org/abs/2412.01095 ; https://openaccess.thecvf.com/content/CVPR2025/supplemental/Ye_VERA_Explainable_Video_CVPR_2025_supplemental.pdf
- ESCHER: https://arxiv.org/pdf/2504.00185 ; https://openaccess.thecvf.com/content/CVPR2025/html/Sehgal_Self-Evolving_Visual_Concept_Library_using_Vision-Language_Critics_CVPR_2025_paper.html ; https://trishullab.github.io/escher-web/
- FuDD: https://arxiv.org/pdf/2311.07593 ; https://proceedings.iclr.cc/paper_files/paper/2024/file/17eca81974d2fd34f458916e0bfd820d-Paper-Conference.pdf
- LLM-Mutate: https://arxiv.org/pdf/2404.09941 ; https://llm-mutate.cs.columbia.edu
- CAPT: https://arxiv.org/abs/2603.02557
- Learning concise and descriptive attributes: https://arxiv.org/abs/2308.03685
- Attribute-based visual reprogramming: https://arxiv.org/abs/2501.13982
- MPO: https://bytez.com/docs/arxiv/2510.09201/paper
- TP-Eval: https://arxiv.org/pdf/2410.18071
- Are prompt optimizers blind?: https://arxiv.org/html/2607.24354v1
- Temporal concept bottleneck for video: https://arxiv.org/pdf/2509.20899
