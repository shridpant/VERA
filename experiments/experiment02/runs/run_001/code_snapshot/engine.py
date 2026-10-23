"""GPU engine for experiment02: one frozen InternVL2-8B (BF16) shared by three prompt variants.

Generation always goes through the pinned VERA wrapper (VideoAnomalyDetectionModelInference.forward ->
InternVL batch_chat), exactly as experiment01. The auxiliary score never generates: it lets VERA/batch_chat build
the label-only query, intercepts the token ids/pixels that would be passed to generate(), and runs one
teacher-forced forward through the same embedding-injection code path as InternVLChatModel.generate().
"""
import json
import sys
import time

import attr_lib as al

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Engine:
    def __init__(self, cfg, eval_manifest_path):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.cfg = cfg
        src = al.ROOT / cfg["source"]["checkout"]
        sys.path.insert(0, str(src))
        import generate_initial_score as gis
        self.gis = gis
        if "".join(q + "\n" for q in cfg["prompt"]["guiding_questions"]) != gis.new_qs:
            raise RuntimeError("config guiding questions differ from the pinned source")
        instruct = (src / cfg["prompt"]["learner_instruct_file"]).read_text(encoding="utf-8")
        self.prompt_parts = {
            "original": (instruct, gis.new_qs),
            "nosound": (instruct, al.nosound_questions(cfg["prompt"]["guiding_questions"])),
            "label_only": (al.label_only_instruct(instruct), gis.new_qs),
        }
        # transform exactly as the VERA dataset builds it
        self.dataset = gis.Video_Instruct_Dataset_Inference(vis_root=str(al.ROOT / cfg["frames"]["cache_root"]) + "/",
                                                            ann_root=str(eval_manifest_path))
        self.transform = self.dataset.transform

        torch.manual_seed(cfg["seed"])
        t0 = time.time()
        model_dir = al.ROOT / cfg["model"]["local_dir"]
        self.model = AutoModel.from_pretrained(str(model_dir), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                               trust_remote_code=True, use_flash_attn=cfg["model"]["use_flash_attn"]).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        self.wrappers = {}
        for name, (ins, qs) in self.prompt_parts.items():
            self.wrappers[name] = gis.VideoAnomalyDetectionModelInference(
                model=self.model, tokenizer=self.tok, optimizer_instruct=None, model_instruct=ins, new_qs=qs,
                generation_config=dict(cfg["generation"])).eval().cuda()
        self.load_s = time.time() - t0
        self.attention = {
            "vision_use_flash_attn": bool(self.model.vision_model.encoder.layers[0].attn.use_flash_attn),
            "llm_attn_implementation": getattr(self.model.language_model.config, "attn_implementation", None),
            "llm_attention_class": type(self.model.language_model.model.layers[0].attention).__name__}

        self._orig_generate = self.model.generate
        self._mode = "generate"
        self._cap = {}

        def dispatch(*a, **kw):  # instrumentation: record the exact inputs; in capture mode do not generate
            self._cap = dict(kw)
            if self._mode == "capture":
                return torch.zeros((1, 1), dtype=torch.long, device=kw["input_ids"].device)
            out = self._orig_generate(*a, **kw)
            self._cap["n_new_tokens"] = int(out.shape[1])
            return out
        self.model.generate = dispatch
        self.candidates = None
        self.counts = {"generation_calls": 0, "score_calls": 0, "alignment_score_forwards": 0,
                       "alignment_generate_calls": 0, "gradient_calls": 0}
        self._verified_cases = set()
        self._baseline_hashes = {}

    # ------------------------------------------------------------ inputs

    def pixel_values(self, case, mask, spec):
        """Upstream pixel path; `spec` is applied to the slots in `mask` after dynamic_preprocess."""
        self.verify_case_frames(case)
        pv, slot_hashes = al.slot_tensors(self.gis, self.transform, al.ROOT / case["frames_dir"],
                                          case["frame_indices"], mask, spec)
        if not mask:
            self._baseline_hashes[case["case_id"]] = slot_hashes
        return pv.to(self.torch.bfloat16).cuda(), slot_hashes

    def verify_case_frames(self, case):
        """The JPEGs read now must be the ones hashed into cases.json (the frame cache is shared)."""
        if case["case_id"] in self._verified_cases:
            return
        fdir = al.ROOT / case["frames_dir"]
        now_sha = [al.sha256_file(fdir / f"{i:06d}.jpg") for i in case["frame_indices"]]
        if now_sha != case["frame_sha256"]:
            raise RuntimeError(f"{case['case_id']}: frame files differ from cases.json hashes")
        self._verified_cases.add(case["case_id"])

    def slot_identity(self, case, mask, slot_hashes):
        """True iff exactly the masked slots differ from this case's unperturbed slot tensors."""
        if case["case_id"] not in self._baseline_hashes:
            _, self._baseline_hashes[case["case_id"]] = al.slot_tensors(
                self.gis, self.transform, al.ROOT / case["frames_dir"], case["frame_indices"], (), {"family": "none"})
        b = self._baseline_hashes[case["case_id"]]
        return all((slot_hashes[i] != b[i]) == (i in mask) for i in range(al.N_SLOTS))

    # ------------------------------------------------------------ generation (original / nosound / label_only)

    def generate(self, variant, case, mask, spec):
        torch = self.torch
        pv, slot_hashes = self.pixel_values(case, mask, spec)
        w = self.wrappers[variant]
        self._mode = "generate"
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        labels = w.forward(pv, [1] * al.N_SLOTS)
        torch.cuda.synchronize()
        dt = time.time() - t0
        self.counts["generation_calls"] += 1
        raw = w.last_responses[0]
        ids = self._cap["input_ids"][0].cpu()
        if variant == "label_only":
            label, status = al.parse_label_only(raw)
        else:
            label, status = al.vp.strict_parse_label(raw)
        return {
            "raw_response": raw, "label": label, "parse_status": status,
            "label_upstream_root_parser": labels[0] if isinstance(labels, list) and len(labels) == 1 else None,
            "question_sha256": al.sha256_bytes(w.last_questions[0].encode()),
            "model_input_sha256": al.sha256_bytes(ids.numpy().tobytes()), "model_input_tokens": int(ids.shape[0]),
            "slot_tensor_sha256": slot_hashes, "slot_identity_ok": self.slot_identity(case, mask, slot_hashes),
            "n_new_tokens": self._cap.get("n_new_tokens"),
            "effective_generation_config": dict(w.generation_config), "runtime_s": round(dt, 3),
            "peak_gpu_mem_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "mentions_sound": al.vp.mentions_sound(raw), "sound_sentences": al.sound_sentences(raw),
            "claim_flags": al.claim_flags(raw)}

    # ------------------------------------------------------------ auxiliary label-only score

    def capture_label_only_inputs(self, pv):
        self._mode = "capture"
        try:
            self.wrappers["label_only"].forward(pv, [1] * al.N_SLOTS)
        finally:
            self._mode = "generate"
        cap = self._cap
        return cap["pixel_values"], cap["input_ids"], cap["attention_mask"], self.wrappers["label_only"].last_questions[0]

    def setup_candidates(self, question, captured_ids):
        """Rebuild the batch_chat query text, confirm it tokenizes to the captured ids, tokenize '0'/'1' after it."""
        torch = self.torch
        mod = sys.modules[type(self.model).__module__]
        tpl = mod.get_conv_template(self.model.template)
        tpl.system_message = self.model.system_message
        tpl.append_message(tpl.roles[0], question)
        tpl.append_message(tpl.roles[1], None)
        query = tpl.get_prompt()
        query = query.replace("<image>", "<img>" + "<IMG_CONTEXT>" * self.model.num_image_token * al.N_SLOTS + "</img>", 1)
        q_ids = self.tok(query, return_tensors="pt").input_ids[0]
        info = {"query_reconstruction_matches_captured_ids": bool(torch.equal(q_ids, captured_ids[0].cpu())),
                "boundary": "start of the assistant reply, right after '<|im_start|>assistant\\n'",
                "query_tail_repr": repr(query[-60:]), "candidates": {}}
        for lab in ("0", "1"):
            full = self.tok(query + lab, return_tensors="pt").input_ids[0]
            info["candidates"][lab] = {
                "prefix_stable": bool(torch.equal(full[:len(q_ids)], q_ids)),
                "token_ids": full[len(q_ids):].tolist(),
                "tokens": self.tok.convert_ids_to_tokens(full[len(q_ids):].tolist()),
                "standalone_token_ids": self.tok(lab, add_special_tokens=False).input_ids}
        info["single_token_candidates"] = all(len(c["token_ids"]) == 1 for c in info["candidates"].values())
        info["ok"] = info["query_reconstruction_matches_captured_ids"] and all(
            c["prefix_stable"] and c["token_ids"] for c in info["candidates"].values())
        self.candidates = info
        return info

    def lm_logits(self, pv, ids, am):
        """InternVLChatModel.generate's embedding injection, then one full forward (no cache). Returns float logits."""
        torch = self.torch
        m = self.model
        vit_embeds = m.extract_feature(pv)
        input_embeds = m.language_model.get_input_embeddings()(ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C).clone()
        selected = ids.reshape(B * N) == m.img_context_token_id
        if int(selected.sum()) != vit_embeds.shape[0] * vit_embeds.shape[1]:
            raise RuntimeError("image-token count does not match visual embeddings")
        input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
        input_embeds = input_embeds.reshape(B, N, C)
        out = m.language_model(inputs_embeds=input_embeds, attention_mask=am, use_cache=False, return_dict=True)
        return out.logits[0].float()

    def score_from_inputs(self, pv, ids, am):
        torch = self.torch
        c = self.candidates["candidates"]
        with torch.no_grad():
            if self.candidates["single_token_candidates"]:
                logits = self.lm_logits(pv, ids, am)[-1]
                logp = torch.log_softmax(logits, -1)
                lp = {lab: float(logp[c[lab]["token_ids"][0]]) for lab in ("0", "1")}
                lp["raw_logit_0"] = float(logits[c["0"]["token_ids"][0]])
                lp["raw_logit_1"] = float(logits[c["1"]["token_ids"][0]])
                top = torch.topk(logp, 5)
                top5 = [(self.tok.convert_ids_to_tokens(int(i)), int(i), round(float(v.exp()), 6))
                        for v, i in zip(top.values, top.indices)]
            else:  # full continuation log-likelihood of each candidate's token sequence
                lp, top5 = {}, None
                for lab in ("0", "1"):
                    ext = torch.tensor([c[lab]["token_ids"]], device=ids.device)
                    ids2 = torch.cat([ids, ext], 1)
                    am2 = torch.cat([am, torch.ones_like(ext)], 1)
                    logits = self.lm_logits(pv, ids2, am2)
                    logp = torch.log_softmax(logits, -1)
                    n0 = ids.shape[1]
                    lp[lab] = float(sum(logp[n0 - 1 + j, t] for j, t in enumerate(c[lab]["token_ids"])))
                logits = None
        return lp, top5, logits

    def score(self, case, mask, spec):
        torch = self.torch
        pv0, slot_hashes = self.pixel_values(case, mask, spec)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        pv, ids, am, question = self.capture_label_only_inputs(pv0)
        if self.candidates is None:
            self.setup_candidates(question, ids)
        if not self.candidates["ok"]:
            raise RuntimeError("candidate tokenization / query reconstruction check failed; refusing to score")
        if not self.candidates["single_token_candidates"]:
            raise RuntimeError("multi-token candidates: out of scope for contract label_only_first_token_v1 (see config)")
        lp, top5, _ = self.score_from_inputs(pv, ids, am)
        torch.cuda.synchronize()
        dt = time.time() - t0
        self.counts["score_calls"] += 1
        S = lp["1"] - lp["0"]
        l0, l1 = lp["raw_logit_0"], lp["raw_logit_1"]
        return {"S": S, "logp_0": lp["0"], "logp_1": lp["1"], "p_0": float(torch.tensor(lp["0"]).exp()),
                "p_1": float(torch.tensor(lp["1"]).exp()),
                "candidate_mass": float(torch.tensor(lp["0"]).exp() + torch.tensor(lp["1"]).exp()),
                "top5_next_tokens": top5, "finite": bool(torch.isfinite(torch.tensor(S))),
                "question_sha256": al.sha256_bytes(question.encode()),
                "model_input_sha256": al.sha256_bytes(ids[0].cpu().numpy().tobytes()),
                "model_input_tokens": int(ids.shape[1]), "slot_tensor_sha256": slot_hashes,
                "slot_identity_ok": self.slot_identity(case, mask, slot_hashes),
                "logit_0": l0, "logit_1": l1,
                "S_resolution": max(al.bf16_spacing(l0), al.bf16_spacing(l1)),
                "candidate_token_ids": {k: v["token_ids"] for k, v in self.candidates["candidates"].items()},
                "scoring_contract": al.SCORING_CONTRACT_VERSION, "runtime_s": round(dt, 3),
                "peak_gpu_mem_allocated_bytes": int(torch.cuda.max_memory_allocated())}

    def alignment_check(self, case):
        """Compare the scorer's boundary logits with generate()'s own first-step scores on the same inputs."""
        torch = self.torch
        pv0, _ = self.pixel_values(case, (), {"family": "none"})
        pv, ids, am, question = self.capture_label_only_inputs(pv0)
        if self.candidates is None:
            self.setup_candidates(question, ids)
        lp, top5, logits = self.score_from_inputs(pv, ids, am)
        self.counts["alignment_score_forwards"] += 1
        eos = self.tok.convert_tokens_to_ids("<|im_end|>")
        with torch.no_grad():
            out = self._orig_generate(pixel_values=pv, input_ids=ids, attention_mask=am, max_new_tokens=1,
                                      do_sample=False, num_beams=1, eos_token_id=eos,
                                      output_scores=True, return_dict_in_generate=True)
        self.counts["alignment_generate_calls"] += 1
        g = out.scores[0][0].float()
        glogp = torch.log_softmax(g, -1)
        c = self.candidates["candidates"]
        res = {"case_id": case["case_id"], "S_scorer": lp["1"] - lp["0"]}
        if logits is not None:
            res["max_abs_logit_diff"] = float((g - logits).abs().max())
            res["argmax_scorer"] = int(logits.argmax())
        res["argmax_generate_step0"] = int(g.argmax())
        res["argmax_generate_token"] = self.tok.convert_ids_to_tokens(int(g.argmax()))
        res["generated_token"] = self.tok.convert_ids_to_tokens(int(out.sequences[0, -1]))
        if self.candidates["single_token_candidates"]:
            res["S_generate_step0"] = float(glogp[c["1"]["token_ids"][0]] - glogp[c["0"]["token_ids"][0]])
            res["abs_S_diff"] = abs(res["S_generate_step0"] - res["S_scorer"])
        return res

    def effective_question(self, variant):
        """The exact question string VERA builds for 8 frames (no model call)."""
        ins, qs = self.prompt_parts[variant]
        prefix = "".join(f"Frame{i + 1}: <image>\n" for i in range(al.N_SLOTS))
        q = ins.replace("$Data", prefix)
        flag = "Based on the analysis above, please conclude your answer to 'Is there any anomaly in the video?' in 'Yes, there is an anomaly' or 'No, there is no anomaly'."
        i = q.find(flag)
        return q[:i] + qs + q[i:]

    def summary(self):
        return {"model_load_s": round(self.load_s, 1), "attention": self.attention, "counts": dict(self.counts),
                "candidates": self.candidates}


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False)
