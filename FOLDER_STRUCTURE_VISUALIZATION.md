# 📂 프로젝트 폴더 구조 상세도

## 🌳 전체 디렉토리 트

```
/root/IR/

 📁 finetune/                                    # 파인튜닝 파이프라인
   ├── 🔵 1_generate_qa.py                        # Stage 1: QA 생
   ├── 🟢 2_mine_negatives_v3.py                  # Stage 2: Hard Negative Mining
   ├── 🟡 3_run_train_v3.sh                       # Stage 3: BGE-M3 학습
   ├── 📊 1_generate_qa.log                       # QA 생성 로그
   ├── 📊 3_run_train.log                         # v1 학습 로그 (268 steps)
   └── 📊 train_v2.log                            # v2 학     로그 (402 steps)

 📁 data/                                        # 데이터 디렉토리
   ├── 📄 corpus.jsonl                            # 원본 문서 (4,272개)
   ├── 📄 synthetic_qa_solar.jsonl                # 생성 QA (12,816개)
   ├── 📄 train_data_v3.jsonl                     # 학습 데이터 (12,816개)
   ├── 📄 test.jsonl                              # 평가 질문 (220개)
   └── ...

 📁 finetuned_bge_m3/                            # v1 파인튜닝 모델
   ├── 🏆 model.safetensors                       # 2.27GB 모델 가중치
   ├── ⚙️ config.json                             # 모델 설정
   ├── 📝 tokenizer_config.json                   # 토크나이저 설정
   ├── 📝 tokenizer.json                          # 토크나이저
   ├── 📝 special_tokens_map.json                 # 특수 토큰
   └── 📝 training_args.bin                       # 학습 인자

 📁 finetuned_bge_m3_v2/                         # v2 파인튜닝 모델 (402 steps)
   ├── 🏆 model.safetensors                       # 2.27GB
   └── ... (동일 구조)

 📁 finetuned_bge_m3_v3/                         # v3 파인튜닝 모델 (최종, 12K)
   ├── 🏆 model.safetensors                       # 2.27GB
 ... (동일 구조)   └

 📄 eval_rag.py                                  # 메인 평가 스크립트
 📄 eval_rag_finetuned.py                        # 파인튜닝 모델 평가
 📄 eval_finetuned_v9.log                        # v9 평가 로그
 📄 eval_rag_finetuned.log                       # 파인튜닝 평가 로그

 📄 submission_surgical_v1.csv                   # 현재 최고 (MAP 0.9470)
 📄 submission_54_bge_m3_sota.csv                # v1 평가 (206KB)
 📄 submission_55_bge_m3_sota.csv                # v2 평가 (175KB)
 📄 submission_56_bge_m3_sota_v3.csv             # v3 평가 (178KB)
 📄 submission_57_bge_m3_sota_v4.csv             # 파라미터 조정 (183KB)
 📄 submission_58_bge_m3_sota_v5.csv             # 파라미터 조정 (176KB)
 📄 submission_59_bge_m3_sota_v6.csv             # 파라조정 (179KB)
 📄 submission_60_bge_m3_sota_v7.csv             # 파라미터 조정 (188KB)
 📄 submission_61_bge_m3_solar_sota.csv          # Solar 통합 (309KB)
 📄 submission_88_ready_bge_m3_*.csv             # 최종 제출 (107KB)
 📄 submission_bge_m3_finetuned.csv              # 기본 평가 (415KB)
 📄 submission_bge_m3_finetuned_v9.csv           # v9 평가 (391KB)
 ... (20+ 더 많은 submission 파일)

 📄 SYNTHETIC_FINETUNING_COMPREHENSIVE_REPORT.md # 종합 보고서
 📄 FINETUNING_WORKFLOW_SUMMARY.md              # 워크플로우 요약
 📄 LEADERBOARD_SUBMISSION_HISTORY.md           # 리더보드 이력

 ... (기타 분석 및 실험 )
```

---

## 🔍 주요 디렉토리 설명

### 1. `/finetune/` - 파인튜닝 파이프라인
**목적**: 합성 데이터 생성 및 모델 학습 자동화

```
finetune/
 1_generate_qa.py          # Solar Pro 2로 QA 생성
 2_mine_negatives_v3.py    # BM25+Dense+Reranker로 Hard Negatives
 3_run_train_v3.sh         # BGE-M3 Contrastive Learning
```

**워크플로*:
```
Documents → QA Generation → Hard Negative Mining → Model Training
```

---

### 2. `/data/` - 데이터 디렉토리
**목적**: 원본 문서, 생성 데이터, 학습 데이터 저장

```
data/
 corpus.jsonl              # 4,272 documents
 synthetic_qa_solar.jsonl  # 12,816 QA pairs (3 Q per doc)
 train_data_v3.jsonl       # 12,816 samples (1 pos + 7 neg)
 test.jsonl                # 220 evaluation queries
```

**데이터 변환**:
```
4,272 docs → 12,816 QA → 102,528 doc-query pairs
```

---

### 3. `/finetuned_bge_m3_*` - 파인튜닝 .env .git .gitignore .last_v16_log .last_v16_out .vscode ANALYSIS_COLLEAGUE_CODE.md ANALYSIS_FINAL_RESULT.md ANALYSIS_SCORE_DROP.md BGE_M3_SOTA_OPTIMIZATION_REPORT_FINAL.md BGE_M3_SOTA_OPTIMIZATION_REPORT_LAST.md EMBEDDING_LLM_REPORT_20251224_193623.md EXPERIMENT_SUMMARY.md FINAL_REPORT.md FINETUNING_WORKFLOW_SUMMARY.md FULL_CONFIG_REPORT_20251224_194055.md FULL_CONFIG_REPORT_20251224_194637.md FULL_CONFIG_REPORT_20251224_194959.md FULL_CONFIG_REPORT_20251224_195400.md FULL_CONFIG_REPORT_20251224_195838.md GATING_STRATEGY_COMPREHENSIVE_REPORT.md GRID_SEARCH_LEADERBOARD.md LEADERBOARD_SUBMISSION_HISTORY.md NEXT_METHODS_AFTER_MAP08765.md OPTIMIZATION_STRATEGY.md PHASE_3_FAILURE_ANALYSIS.md PHASE_4_ANALYSIS_AND_STRATEGY.md PHASE_5_RECOVERY_PLAN.md PHASE_7_REPORT.md PHASE_7_SUMMARY.md README.md ROOT_CAUSE_ANALYSIS.md Report SOLAR_PRO2_OPTIMIZATION_REPORT.md SYNTHETIC_FINETUNING_COMPREHENSIVE_REPORT.md __pycache__ ab_llm_tiebreak_bge_m3.py ab_precompute.log analyze_742_failure.py analyze_742_testing.py analyze_all_strategies.py analyze_empty_cases.py analyze_ensemble_weights.py analyze_gating.py analyze_hyde_effect.py analyze_hyde_impact.py analyze_low_gaps_with_solar.py analyze_missed_ids.py analyze_reranker_effect.py analyze_top_diffs.py analyze_top_diffs_v2.py analyze_v9_v3_diff.py analyze_weight_tuning_success.py artifacts auditing auto_tuning.py baseline best_7008.ipynb best_9174.ipynb bge_m3_run.log bge_m3_run_final.log bge_m3_run_final_v2.log bge_m3_run_sota_v2.log bge_m3_run_v2.log bge_m3_run_v3.log bge_m3_run_v4.log bge_m3_run_v5.log bge_m3_run_v6.log build_final_union_rerank.py build_v17_conservative.py cache cache_search_results.py cache_step1.log check_271_303.py check_changes.py check_empty_queries.py check_final_changes.py check_gemini_models.py check_nogating.py check_v5_gaps.py cleanup_v3.py compare_all_top.py compare_gating_vs_submission.py compare_phase1_vs_planA.py compare_phase2_vs_planA.py compare_results.py compare_submissions.py compare_subs_clean.py compare_subs_sota_vs_best.py compare_top_submissions.py compare_v2_surg.py compare_v3_v9.py compare_v9_final.py compare_v9_final_v2.py compare_v9_final_v3.py compare_v9_final_v4.py compare_v9_final_v5.py compare_v9_v15.py comprehensive_experiment_analysis.py confidence_optimization_results.json consensus_rerank.py conservative_strike.py convert_v9_to_csv.py create_master.py create_v10_sota.py create_v11_sota.py create_v12_submission.py create_v13_submission.py create_v14_submission.py create_v15_submission.py data deep_scan.py deep_scan_v2.py detailed_experiment_comparison.py elasticsearch-8.8.0 ensemble_base_ft.py ensemble_final.py ensemble_run.log es_setup.py es_setup.py.backup es_setup_old.py eval_\[7\,4\,2\]_full.log eval_\[7\,4\,2\]_log.txt eval_finetuned_v9.log eval_rag.py eval_rag.py.bak eval_rag_bge_m3.py eval_rag_bge_m3_base.py eval_rag_bge_m3_v2.py eval_rag_bge_m3_v3.py eval_rag_bge_m3_v4.py eval_rag_bge_m3_v5.py eval_rag_bge_m3_v6.py eval_rag_bge_m3_v7.py eval_rag_bge_m3_v8_recovery.py eval_rag_e5_base.py eval_rag_e5_ensemble.py eval_rag_e5_final.py eval_rag_e5_hybrid.py eval_rag_e5_multi.py eval_rag_e5_repro.py eval_rag_e5_sota.py eval_rag_e5_ultimate.py eval_rag_final_strategy.py eval_rag_finetuned.log eval_rag_finetuned.py eval_rag_finetuned_v9.py eval_rag_no_gating.py eval_rag_rerank_ensemble.py eval_rag_topk60.py eval_rag_v11_full_solar.py eval_rag_v16_gemini_rerank.py eval_rag_v2_final.py eval_rag_v3_ensemble.py eval_rag_v8_v5_queries.py eval_rag_v9_sota.py eval_rag_weight552.py eval_rag_weighted_rrf.py eval_v3.log eval_v3_ensemble.log eval_v3_fixed.log eval_v3_fixed_2.log evaluation_gating_v2.log evaluation_with_gating.log experiment_cp100_20251223_080055.log experiment_topk80_20251223_063042.log experiment_topk80_20251223_063501.log experiment_topk80_20251223_063621.log experiment_topk80_20251223_063621.pid experiment_topk80_run.log experiments fast_alpha_sweep.py fill_empty.py fill_empty_v2.py final_comprehensive_report.py final_strategy.log final_strategy.py final_summary.py final_surgical_check.py finalize_submission.py find_v9_v3_diffs.py finetune finetuned_bge_m3 finetuned_bge_m3_v2 finetuned_bge_m3_v3 fix_v9_order.py gemini_indexing.log gemini_run.log gemini_run.pid generate_candidates.py generate_final_challenge.py generate_final_last_chance.py generate_final_surgical.py generate_final_surgical_v2.py generate_hybrid_s33gating_wrrf.py generate_qa.log generate_super_hybrid.py generate_synthetic_qa.py gpt4o_run.log grid_search.py grid_search_cached.py grid_search_results.json grid_search_step2.log hyde_evaluation.log hyde_planA.log hyde_test.log inspect_v11_changes.py inspect_v9_choices.py judge_decisions.json judge_mismatches.py judge_report.json judge_results.json last_mq120_submission_log.txt last_mq120_submission_path.txt last_mq_submission_log.txt last_mq_submission_path.txt list_empty.py log_ab_baseline.jsonl log_ab_gpt4o_gap015.jsonl log_ab_gpt4o_sample.jsonl log_ab_gpt4o_sample2.jsonl log_ab_solar_gap0.05.jsonl log_ab_solar_gap0.10.jsonl log_ab_solar_gap0.20.jsonl log_ab_solar_gap015.jsonl log_ab_solar_sample.jsonl log_ab_solar_sample2.jsonl main.py main_eval_final.log main_eval_solar.log main_eval_solar.pid main_eval_solar_v2.log main_eval_solar_v2.pid main_eval_solar_v3.log main_eval_solar_v3.pid main_reranker.log main_reranker_optimized.log main_run.log main_run_improved.log merge_v9_v3.py mine_v2.log models optimize_confidence.py optuna_search.py phase2_tuning.log phase_2_1_evaluation.log phase_3_1_test.log phase_3_full_evaluation.log phase_4a_evaluation.log phase_4b_evaluation.log phase_4c_evaluation.log phase_4d_evaluation.log phase_4d_nogating_evaluation.log phase_4d_topk60_evaluation.log phase_5_evaluation.log phase_6a_evaluation.log phase_6a_evaluation_v2.log phase_6a_final.log phase_6b1_evaluation.log phase_7_evaluation.log phase_7_evaluation_real.log phase_7_new.log phase_8_evaluation.log phase_9_evaluation.log pipeline_v3.log precision_strike.py prepare_judge.py prepare_v12_candidates.py progress.log requirements.txt rerank_ensemble.log result_gate result_gem result_multi retrieval run_bge_m3_sota_20251229_023154.log run_bge_m3_sota_env.sh run_eval_742.sh run_judge.py run_rrf_k20_20251224_060251.log run_rrf_k20_20251224_060339.log run_rrf_k20_mq_cp120_upstageHeavy_20251224_172309.log run_rrf_k20_mq_tk120_cp120_20251224_071802.log run_rrf_k20_mq_tk120_cp120_20251224_080338.log run_rrf_k20_mq_tk120_cp120_20251224_082326.log run_rrf_k20_mq_tk120_cp120_upstageOnly_20251224_165428.log run_rrf_k20_mq_tk80_cp80_20251224_071819.log run_rrf_k20_mq_tk80_cp80_20251224_072844.log run_rrf_k20_mq_tk80_cp80_dense3_upstage2048_20251224_154415.log run_rrf_mq_20251225_010454.log run_single_eval.py run_strategy_20251224_202157.log run_strategy_v2_20251225_002516.log run_tests.sh run_tk100_cp80_20251223_152050.log run_tk100_cp80_20251223_152141.log run_tk100_cp80_20251224_023753.log run_tuning_grid.sh run_v2_final.log run_v2_final.sh run_v3_pipeline.sh run_v7_solar.log scripts search_results_cache.json snapshot_submission.py solar_diff_analysis.json solar_gating_audit.json solar_low_gap_improvements.json strategy_a_evaluation.log submission.csv submission_18\(14\).csv submission_19.csv submission_20.csv submission_38_ready_rrf_k20_mq_tk80_cp80_dense3_20251224_114800.csv submission_39_ready_rrf_k20_mq_tk80_cp80_dense3_upstage2048_20251224_154415.csv submission_40_ready_rrf_k20_mq_tk120_cp120_upstageOnly_20251224_165428.csv submission_41_ready_rrf_k20_mq_cp120_upstageHeavy_20251224_172309.csv submission_42_strategy_tk100_cp100_h300_mq_20251224_202157.csv submission_43_strategy_v2_tk100_cp100_h300_mq_20251225_002516.csv submission_44_rrf_k30_mq_tk100_cp100_20251225_010454.csv submission_45_hybrid_s33gating_wrrf_search.csv submission_46_final_strategy.csv submission_47_e5_final.csv submission_48_e5_hybrid.csv submission_49_e5_sota.csv submission_50_e5_solar_pro.csv submission_51_e5_gemini.csv submission_52_e5_ultimate.csv submission_53_e5_super_ensemble.csv submission_54_bge_m3_sota.csv submission_55_bge_m3_sota.csv submission_56_bge_m3_sota_v3.csv submission_57_bge_m3_sota_v4.csv submission_58_bge_m3_sota_v5.csv submission_59_bge_m3_sota_v6.csv submission_60_bge_m3_sota_v7.csv submission_61_bge_m3_solar_sota.csv submission_62_v8_v5_queries_solar_tiebreak.csv submission_63_v9_sota.csv submission_64_v12_sota.csv submission_65_v13_sota.csv submission_66_v14_sota.csv submission_67_v15_sota.csv submission_68_v16_gemini_rerank_20251227_130830.csv submission_69_v17_conservative_from_v9_20251227_145004.csv submission_70_v17_safe3_from_v9_20251227_150049.csv submission_71_v17_attack5_from_v9_20251227_150049.csv submission_72_final_union_rerank_v18.csv submission_73_ensemble_base0.7_ft0.3.csv submission_74_ensemble_base0.5_ft0.5.csv submission_75_ensemble_base0.8_ft0.2.csv submission_76_v2_final_rerank.csv submission_77_final_ensemble_v9_v2.csv submission_78_final_v2_precision.csv submission_79.csv submission_80_v3_final_rerank.csv submission_81_v3_final.csv submission_82_surgical_v1.csv submission_83_final_0.95_break.csv submission_84_final_0.95_break_v2.csv submission_85_final_0.95_master.csv submission_86_candidate_B_id271.csv submission_87_candidate_D_id271_id303.csv submission_88_ready_bge_m3_sota_20251229_023154.csv submission_89_grid_v2_mq_off_20251229_025014.csv submission_90_final_challenge_0.95.csv submission_91_final_surgical_v2_id270_only.csv submission_92_final_last_chance.csv submission_93_grid_v3_tk200_20251229_025014.csv submission_ab_baseline.csv submission_ab_gpt4o_gap015.csv submission_ab_gpt4o_sample.csv submission_ab_gpt4o_sample2.csv submission_ab_solar_gap0.05.csv submission_ab_solar_gap0.10.csv submission_ab_solar_gap0.20.csv submission_ab_solar_gap015.csv submission_ab_solar_sample.csv submission_ab_solar_sample2.csv submission_backup_old.csv submission_backup_phase6b.csv submission_baseline_map08765_20251223_063042.csv submission_before_cp100_20251223_080055.csv submission_before_reranker.csv submission_before_topk80_20251223_063621.csv submission_best_9174.csv submission_best_9273.csv submission_best_9394.csv submission_best_map08765.csv submission_bge_m3_base_simple.csv submission_bge_m3_finetuned.csv submission_bge_m3_finetuned_v9.csv submission_bge_m3_sota.csv submission_bge_m3_sota_v3.csv submission_bge_m3_sota_v4.csv submission_bge_m3_sota_v5.csv submission_bge_m3_sota_v6.csv submission_bge_m3_sota_v7.csv submission_bge_m3_v2_ft.csv submission_candidate_A_surgical.csv submission_candidate_B_id271.csv submission_candidate_C_id303.csv submission_candidate_D_id271_id303.csv submission_conservative_strike.csv submission_cp100_20251223_104822.csv submission_diffs.json submission_e5_base.csv submission_e5_final.csv submission_e5_gemini.csv submission_e5_gpt4o.csv submission_e5_hybrid.csv submission_e5_multi.csv submission_e5_repro.csv submission_e5_solar_pro.csv submission_e5_sota.csv submission_e5_super_ensemble.csv submission_e5_ultimate.csv submission_ensemble_base0.5_ft0.5.csv submission_ensemble_base0.7_ft0.3.csv submission_ensemble_base0.8_ft0.2.csv submission_final_0.95_break.csv submission_final_0.95_break_v2.csv submission_final_0.95_master.csv submission_final_challenge_0.95.csv submission_final_ensemble_v9_v2.csv submission_final_strategy.csv submission_final_surgical_hybrid_0.95.csv submission_final_surgical_v2_id270_only.csv submission_final_union_rerank_4sources.csv submission_final_union_rerank_v18.csv submission_final_v2_precision.csv submission_grid_v1_llm_on_20251229_025014.csv submission_grid_v2_mq_off_20251229_025014.csv submission_grid_v3_tk200_20251229_025014.csv submission_hybrid_s33gating_wrrf_search.csv submission_hyde_v1.csv submission_nogating.csv submission_old.csv submission_old_0.csv submission_partial_before_solar_fullrun_20251222_230943.csv submission_phase7_failed.csv submission_planA.csv submission_pre_topk80_20251223_063501.csv submission_precision_strike.csv submission_ready_5_tk100_cp80_20251223_152141.csv submission_ready_bge_m3_sota_20251229_023154.csv submission_ready_rrf_k20_mq_cp120_upstageHeavy_20251224_172309.csv submission_ready_rrf_k20_mq_tk120_cp120_20251224_071802.csv submission_ready_rrf_k20_mq_tk120_cp120_20251224_080338.csv submission_ready_rrf_k20_mq_tk120_cp120_20251224_082326.csv submission_ready_rrf_k20_mq_tk120_cp120_upstageOnly_20251224_165428.csv submission_ready_rrf_k20_mq_tk80_cp80_20251224_071819.csv submission_ready_rrf_k20_mq_tk80_cp80_20251224_072844.csv submission_ready_rrf_k20_mq_tk80_cp80_dense3_20251224_114800.csv submission_ready_rrf_k20_mq_tk80_cp80_dense3_upstage2048_20251224_154415.csv submission_ready_rrf_k20_tk80_cp80_20251224_060339.csv submission_rerank_ensemble_v1.csv submission_reranker.csv submission_snapshot.json submission_solar_final_sota.csv submission_solar_mq_tiebreak_v7.csv submission_solar_precheck_backup_20251222_191832.csv submission_solar_v2_scienceonly_20251223_000954.csv submission_submitted_07697_20251222_234512.csv submission_super_hybrid_final.csv submission_super_hybrid_final_v2.csv submission_surgical_v1.csv submission_topk60.csv submission_ultimate_ensemble_v1.csv submission_ultimate_strike.csv submission_v11_sota.csv submission_v12_sota.csv submission_v13_sota.csv submission_v14_sota.csv submission_v15_sota.csv submission_v16_gemini_rerank_20251227_130830.csv submission_v16_gemini_rerank_smoke.csv submission_v17_attack5_from_v9_20251227_150036.csv submission_v17_attack5_from_v9_20251227_150049.csv submission_v17_conservative_from_v9_20251227_145004.csv submission_v17_safe3_from_v9_20251227_150036.csv submission_v17_safe3_from_v9_20251227_150049.csv submission_v2_final_rerank.csv submission_v3_ensemble.csv submission_v3_final.csv submission_v3_final_rerank.csv submission_v3_v9_rrf_64.csv submission_v3_v9_rrf_82.csv submission_v8_recovery_recovery.csv submission_v8_v5_queries_solar_tiebreak.csv submission_v9_sota.csv submission_weighted_rrf.csv surgical_strike.py test_alpha_on_diffs.py test_configs.py test_embedding_change.py test_gemini_rerank.py test_hyde_eval.py test_hyde_quality.py test_parameter_tuning.py test_phase_3_1.py test_solar_v7.py test_v2_scores.py train_v2.log tuning_6_3_1.log ultimate_ensemble.py ultimate_run.log ultimate_strike.py upstage_index_20251224_144509.log upstage_index_full.pid upstage_index_full_20251224_145149.log upstage_index_full_20251224_150842.log v12_candidates_data.json v16_gemini_rerank_20251227_130830.log v16_gemini_rerank_resume_20251227_132912.log v16_gemini_rerank_resume_20251227_133454.log v16_gemini_rerank_resume_20251227_135442.log v16_gemini_rerank_resume_20251227_140859.log v16_gemini_rerank_resume_20251227_141006.log v16_gemini_rerank_resume_20251227_141102.log v16_gemini_rerank_resume_20251227_141131_30270.log v16_gemini_rerank_resume_20251227_141429_8676.log v5_score_gaps.json v7_fixed.log v9_v3_diffs.json verify_hybrid.py wait_then_generate.pid wait_then_generate_20251224_152314.log weighted_rrf_log.txt 
**목적**: 학습된 임베딩 모델 저장

```
finetuned_bge_m3_v3/
 model.safetensors         # 2.27GB XLM-RoBERTa weights
 config.json               # Model configuration
 tokenizer*.json           # Tokenizer files
 training_args.bin         # Training arguments
```

**모델 버전**:
- **v1**: 4,272 samples, 2 epochs, 268 steps (초기)
- **v2**: 4,272 samples, 2+ epochs, 402 steps (개선)
- **v3**: 12,816 samples, 5 epochs, ~1000+ steps (최종)

---

### 4. `/submission_*` - 제출 파일
**목적**: 리더보드 평가 결과 저장

```
submission_*.csv 패턴:
 submission_54-61_bge_m3_*.csv    # v1-v3 평가 (8개)
 submission_88_*.csv              # 최종 제출
 submission_bge_m3_finetuned*.csv # 다양한 평가 (2개)
 ... (총 20+ 파일)
```

**제출 전략**:
- 각 파일은 서로 다른 파라미터 조합 테스트
- Hard Voting: [6,3,1], [7,4,2], [5,3,1] 등
- HyDE: Full, Sparse Only, None
- Reranker: Top-5, Top-10, Top-20

---

## 📊 파일 크기 및 통계

### 모델 파일
```
finetuned_bge_m3/           2.27GB
finetuned_bge_m3_v2/        2.27GB
finetuned_bge_m3_v3/        2.27GB

.env .git .gitignore .last_v16_log .last_v16_out .vscode ANALYSIS_COLLEAGUE_CODE.md ANALYSIS_FINAL_RESULT.md ANALYSIS_SCORE_DROP.md BGE_M3_SOTA_OPTIMIZATION_REPORT_FINAL.md BGE_M3_SOTA_OPTIMIZATION_REPORT_LAST.md EMBEDDING_LLM_REPORT_20251224_193623.md EXPERIMENT_SUMMARY.md FINAL_REPORT.md FINETUNING_WORKFLOW_SUMMARY.md FULL_CONFIG_REPORT_20251224_194055.md FULL_CONFIG_REPORT_20251224_194637.md FULL_CONFIG_REPORT_20251224_194959.md FULL_CONFIG_REPORT_20251224_195400.md FULL_CONFIG_REPORT_20251224_195838.md GATING_STRATEGY_COMPREHENSIVE_REPORT.md GRID_SEARCH_LEADERBOARD.md LEADERBOARD_SUBMISSION_HISTORY.md NEXT_METHODS_AFTER_MAP08765.md OPTIMIZATION_STRATEGY.md PHASE_3_FAILURE_ANALYSIS.md PHASE_4_ANALYSIS_AND_STRATEGY.md PHASE_5_RECOVERY_PLAN.md PHASE_7_REPORT.md PHASE_7_SUMMARY.md README.md ROOT_CAUSE_ANALYSIS.md Report SOLAR_PRO2_OPTIMIZATION_REPORT.md SYNTHETIC_FINETUNING_COMPREHENSIVE_REPORT.md __pycache__ ab_llm_tiebreak_bge_m3.py ab_precompute.log analyze_742_failure.py analyze_742_testing.py analyze_all_strategies.py analyze_empty_cases.py analyze_ensemble_weights.py analyze_gating.py analyze_hyde_effect.py analyze_hyde_impact.py analyze_low_gaps_with_solar.py analyze_missed_ids.py analyze_reranker_effect.py analyze_top_diffs.py analyze_top_diffs_v2.py analyze_v9_v3_diff.py analyze_weight_tuning_success.py artifacts auditing auto_tuning.py baseline best_7008.ipynb best_9174.ipynb bge_m3_run.log bge_m3_run_final.log bge_m3_run_final_v2.log bge_m3_run_sota_v2.log bge_m3_run_v2.log bge_m3_run_v3.log bge_m3_run_v4.log bge_m3_run_v5.log bge_m3_run_v6.log build_final_union_rerank.py build_v17_conservative.py cache cache_search_results.py cache_step1.log check_271_303.py check_changes.py check_empty_queries.py check_final_changes.py check_gemini_models.py check_nogating.py check_v5_gaps.py cleanup_v3.py compare_all_top.py compare_gating_vs_submission.py compare_phase1_vs_planA.py compare_phase2_vs_planA.py compare_results.py compare_submissions.py compare_subs_clean.py compare_subs_sota_vs_best.py compare_top_submissions.py compare_v2_surg.py compare_v3_v9.py compare_v9_final.py compare_v9_final_v2.py compare_v9_final_v3.py compare_v9_final_v4.py compare_v9_final_v5.py compare_v9_v15.py comprehensive_experiment_analysis.py confidence_optimization_results.json consensus_rerank.py conservative_strike.py convert_v9_to_csv.py create_master.py create_v10_sota.py create_v11_sota.py create_v12_submission.py create_v13_submission.py create_v14_submission.py create_v15_submission.py data deep_scan.py deep_scan_v2.py detailed_experiment_comparison.py elasticsearch-8.8.0 ensemble_base_ft.py ensemble_final.py ensemble_run.log es_setup.py es_setup.py.backup es_setup_old.py eval_\[7\,4\,2\]_full.log eval_\[7\,4\,2\]_log.txt eval_finetuned_v9.log eval_rag.py eval_rag.py.bak eval_rag_bge_m3.py eval_rag_bge_m3_base.py eval_rag_bge_m3_v2.py eval_rag_bge_m3_v3.py eval_rag_bge_m3_v4.py eval_rag_bge_m3_v5.py eval_rag_bge_m3_v6.py eval_rag_bge_m3_v7.py eval_rag_bge_m3_v8_recovery.py eval_rag_e5_base.py eval_rag_e5_ensemble.py eval_rag_e5_final.py eval_rag_e5_hybrid.py eval_rag_e5_multi.py eval_rag_e5_repro.py eval_rag_e5_sota.py eval_rag_e5_ultimate.py eval_rag_final_strategy.py eval_rag_finetuned.log eval_rag_finetuned.py eval_rag_finetuned_v9.py eval_rag_no_gating.py eval_rag_rerank_ensemble.py eval_rag_topk60.py eval_rag_v11_full_solar.py eval_rag_v16_gemini_rerank.py eval_rag_v2_final.py eval_rag_v3_ensemble.py eval_rag_v8_v5_queries.py eval_rag_v9_sota.py eval_rag_weight552.py eval_rag_weighted_rrf.py eval_v3.log eval_v3_ensemble.log eval_v3_fixed.log eval_v3_fixed_2.log evaluation_gating_v2.log evaluation_with_gating.log experiment_cp100_20251223_080055.log experiment_topk80_20251223_063042.log experiment_topk80_20251223_063501.log experiment_topk80_20251223_063621.log experiment_topk80_20251223_063621.pid experiment_topk80_run.log experiments fast_alpha_sweep.py fill_empty.py fill_empty_v2.py final_comprehensive_report.py final_strategy.log final_strategy.py final_summary.py final_surgical_check.py finalize_submission.py find_v9_v3_diffs.py finetune finetuned_bge_m3 finetuned_bge_m3_v2 finetuned_bge_m3_v3 fix_v9_order.py gemini_indexing.log gemini_run.log gemini_run.pid generate_candidates.py generate_final_challenge.py generate_final_last_chance.py generate_final_surgical.py generate_final_surgical_v2.py generate_hybrid_s33gating_wrrf.py generate_qa.log generate_super_hybrid.py generate_synthetic_qa.py gpt4o_run.log grid_search.py grid_search_cached.py grid_search_results.json grid_search_step2.log hyde_evaluation.log hyde_planA.log hyde_test.log inspect_v11_changes.py inspect_v9_choices.py judge_decisions.json judge_mismatches.py judge_report.json judge_results.json last_mq120_submission_log.txt last_mq120_submission_path.txt last_mq_submission_log.txt last_mq_submission_path.txt list_empty.py log_ab_baseline.jsonl log_ab_gpt4o_gap015.jsonl log_ab_gpt4o_sample.jsonl log_ab_gpt4o_sample2.jsonl log_ab_solar_gap0.05.jsonl log_ab_solar_gap0.10.jsonl log_ab_solar_gap0.20.jsonl log_ab_solar_gap015.jsonl log_ab_solar_sample.jsonl log_ab_solar_sample2.jsonl main.py main_eval_final.log main_eval_solar.log main_eval_solar.pid main_eval_solar_v2.log main_eval_solar_v2.pid main_eval_solar_v3.log main_eval_solar_v3.pid main_reranker.log main_reranker_optimized.log main_run.log main_run_improved.log merge_v9_v3.py mine_v2.log models optimize_confidence.py optuna_search.py phase2_tuning.log phase_2_1_evaluation.log phase_3_1_test.log phase_3_full_evaluation.log phase_4a_evaluation.log phase_4b_evaluation.log phase_4c_evaluation.log phase_4d_evaluation.log phase_4d_nogating_evaluation.log phase_4d_topk60_evaluation.log phase_5_evaluation.log phase_6a_evaluation.log phase_6a_evaluation_v2.log phase_6a_final.log phase_6b1_evaluation.log phase_7_evaluation.log phase_7_evaluation_real.log phase_7_new.log phase_8_evaluation.log phase_9_evaluation.log pipeline_v3.log precision_strike.py prepare_judge.py prepare_v12_candidates.py progress.log requirements.txt rerank_ensemble.log result_gate result_gem result_multi retrieval run_bge_m3_sota_20251229_023154.log run_bge_m3_sota_env.sh run_eval_742.sh run_judge.py run_rrf_k20_20251224_060251.log run_rrf_k20_20251224_060339.log run_rrf_k20_mq_cp120_upstageHeavy_20251224_172309.log run_rrf_k20_mq_tk120_cp120_20251224_071802.log run_rrf_k20_mq_tk120_cp120_20251224_080338.log run_rrf_k20_mq_tk120_cp120_20251224_082326.log run_rrf_k20_mq_tk120_cp120_upstageOnly_20251224_165428.log run_rrf_k20_mq_tk80_cp80_20251224_071819.log run_rrf_k20_mq_tk80_cp80_20251224_072844.log run_rrf_k20_mq_tk80_cp80_dense3_upstage2048_20251224_154415.log run_rrf_mq_20251225_010454.log run_single_eval.py run_strategy_20251224_202157.log run_strategy_v2_20251225_002516.log run_tests.sh run_tk100_cp80_20251223_152050.log run_tk100_cp80_20251223_152141.log run_tk100_cp80_20251224_023753.log run_tuning_grid.sh run_v2_final.log run_v2_final.sh run_v3_pipeline.sh run_v7_solar.log scripts search_results_cache.json snapshot_submission.py solar_diff_analysis.json solar_gating_audit.json solar_low_gap_improvements.json strategy_a_evaluation.log submission.csv submission_18\(14\).csv submission_19.csv submission_20.csv submission_38_ready_rrf_k20_mq_tk80_cp80_dense3_20251224_114800.csv submission_39_ready_rrf_k20_mq_tk80_cp80_dense3_upstage2048_20251224_154415.csv submission_40_ready_rrf_k20_mq_tk120_cp120_upstageOnly_20251224_165428.csv submission_41_ready_rrf_k20_mq_cp120_upstageHeavy_20251224_172309.csv submission_42_strategy_tk100_cp100_h300_mq_20251224_202157.csv submission_43_strategy_v2_tk100_cp100_h300_mq_20251225_002516.csv submission_44_rrf_k30_mq_tk100_cp100_20251225_010454.csv submission_45_hybrid_s33gating_wrrf_search.csv submission_46_final_strategy.csv submission_47_e5_final.csv submission_48_e5_hybrid.csv submission_49_e5_sota.csv submission_50_e5_solar_pro.csv submission_51_e5_gemini.csv submission_52_e5_ultimate.csv submission_53_e5_super_ensemble.csv submission_54_bge_m3_sota.csv submission_55_bge_m3_sota.csv submission_56_bge_m3_sota_v3.csv submission_57_bge_m3_sota_v4.csv submission_58_bge_m3_sota_v5.csv submission_59_bge_m3_sota_v6.csv submission_60_bge_m3_sota_v7.csv submission_61_bge_m3_solar_sota.csv submission_62_v8_v5_queries_solar_tiebreak.csv submission_63_v9_sota.csv submission_64_v12_sota.csv submission_65_v13_sota.csv submission_66_v14_sota.csv submission_67_v15_sota.csv submission_68_v16_gemini_rerank_20251227_130830.csv submission_69_v17_conservative_from_v9_20251227_145004.csv submission_70_v17_safe3_from_v9_20251227_150049.csv submission_71_v17_attack5_from_v9_20251227_150049.csv submission_72_final_union_rerank_v18.csv submission_73_ensemble_base0.7_ft0.3.csv submission_74_ensemble_base0.5_ft0.5.csv submission_75_ensemble_base0.8_ft0.2.csv submission_76_v2_final_rerank.csv submission_77_final_ensemble_v9_v2.csv submission_78_final_v2_precision.csv submission_79.csv submission_80_v3_final_rerank.csv submission_81_v3_final.csv submission_82_surgical_v1.csv submission_83_final_0.95_break.csv submission_84_final_0.95_break_v2.csv submission_85_final_0.95_master.csv submission_86_candidate_B_id271.csv submission_87_candidate_D_id271_id303.csv submission_88_ready_bge_m3_sota_20251229_023154.csv submission_89_grid_v2_mq_off_20251229_025014.csv submission_90_final_challenge_0.95.csv submission_91_final_surgical_v2_id270_only.csv submission_92_final_last_chance.csv submission_93_grid_v3_tk200_20251229_025014.csv submission_ab_baseline.csv submission_ab_gpt4o_gap015.csv submission_ab_gpt4o_sample.csv submission_ab_gpt4o_sample2.csv submission_ab_solar_gap0.05.csv submission_ab_solar_gap0.10.csv submission_ab_solar_gap0.20.csv submission_ab_solar_gap015.csv submission_ab_solar_sample.csv submission_ab_solar_sample2.csv submission_backup_old.csv submission_backup_phase6b.csv submission_baseline_map08765_20251223_063042.csv submission_before_cp100_20251223_080055.csv submission_before_reranker.csv submission_before_topk80_20251223_063621.csv submission_best_9174.csv submission_best_9273.csv submission_best_9394.csv submission_best_map08765.csv submission_bge_m3_base_simple.csv submission_bge_m3_finetuned.csv submission_bge_m3_finetuned_v9.csv submission_bge_m3_sota.csv submission_bge_m3_sota_v3.csv submission_bge_m3_sota_v4.csv submission_bge_m3_sota_v5.csv submission_bge_m3_sota_v6.csv submission_bge_m3_sota_v7.csv submission_bge_m3_v2_ft.csv submission_candidate_A_surgical.csv submission_candidate_B_id271.csv submission_candidate_C_id303.csv submission_candidate_D_id271_id303.csv submission_conservative_strike.csv submission_cp100_20251223_104822.csv submission_diffs.json submission_e5_base.csv submission_e5_final.csv submission_e5_gemini.csv submission_e5_gpt4o.csv submission_e5_hybrid.csv submission_e5_multi.csv submission_e5_repro.csv submission_e5_solar_pro.csv submission_e5_sota.csv submission_e5_super_ensemble.csv submission_e5_ultimate.csv submission_ensemble_base0.5_ft0.5.csv submission_ensemble_base0.7_ft0.3.csv submission_ensemble_base0.8_ft0.2.csv submission_final_0.95_break.csv submission_final_0.95_break_v2.csv submission_final_0.95_master.csv submission_final_challenge_0.95.csv submission_final_ensemble_v9_v2.csv submission_final_strategy.csv submission_final_surgical_hybrid_0.95.csv submission_final_surgical_v2_id270_only.csv submission_final_union_rerank_4sources.csv submission_final_union_rerank_v18.csv submission_final_v2_precision.csv submission_grid_v1_llm_on_20251229_025014.csv submission_grid_v2_mq_off_20251229_025014.csv submission_grid_v3_tk200_20251229_025014.csv submission_hybrid_s33gating_wrrf_search.csv submission_hyde_v1.csv submission_nogating.csv submission_old.csv submission_old_0.csv submission_partial_before_solar_fullrun_20251222_230943.csv submission_phase7_failed.csv submission_planA.csv submission_pre_topk80_20251223_063501.csv submission_precision_strike.csv submission_ready_5_tk100_cp80_20251223_152141.csv submission_ready_bge_m3_sota_20251229_023154.csv submission_ready_rrf_k20_mq_cp120_upstageHeavy_20251224_172309.csv submission_ready_rrf_k20_mq_tk120_cp120_20251224_071802.csv submission_ready_rrf_k20_mq_tk120_cp120_20251224_080338.csv submission_ready_rrf_k20_mq_tk120_cp120_20251224_082326.csv submission_ready_rrf_k20_mq_tk120_cp120_upstageOnly_20251224_165428.csv submission_ready_rrf_k20_mq_tk80_cp80_20251224_071819.csv submission_ready_rrf_k20_mq_tk80_cp80_20251224_072844.csv submission_ready_rrf_k20_mq_tk80_cp80_dense3_20251224_114800.csv submission_ready_rrf_k20_mq_tk80_cp80_dense3_upstage2048_20251224_154415.csv submission_ready_rrf_k20_tk80_cp80_20251224_060339.csv submission_rerank_ensemble_v1.csv submission_reranker.csv submission_snapshot.json submission_solar_final_sota.csv submission_solar_mq_tiebreak_v7.csv submission_solar_precheck_backup_20251222_191832.csv submission_solar_v2_scienceonly_20251223_000954.csv submission_submitted_07697_20251222_234512.csv submission_super_hybrid_final.csv submission_super_hybrid_final_v2.csv submission_surgical_v1.csv submission_topk60.csv submission_ultimate_ensemble_v1.csv submission_ultimate_strike.csv submission_v11_sota.csv submission_v12_sota.csv submission_v13_sota.csv submission_v14_sota.csv submission_v15_sota.csv submission_v16_gemini_rerank_20251227_130830.csv submission_v16_gemini_rerank_smoke.csv submission_v17_attack5_from_v9_20251227_150036.csv submission_v17_attack5_from_v9_20251227_150049.csv submission_v17_conservative_from_v9_20251227_145004.csv submission_v17_safe3_from_v9_20251227_150036.csv submission_v17_safe3_from_v9_20251227_150049.csv submission_v2_final_rerank.csv submission_v3_ensemble.csv submission_v3_final.csv submission_v3_final_rerank.csv submission_v3_v9_rrf_64.csv submission_v3_v9_rrf_82.csv submission_v8_recovery_recovery.csv submission_v8_v5_queries_solar_tiebreak.csv submission_v9_sota.csv submission_weighted_rrf.csv surgical_strike.py test_alpha_on_diffs.py test_configs.py test_embedding_change.py test_gemini_rerank.py test_hyde_eval.py test_hyde_quality.py test_parameter_tuning.py test_phase_3_1.py test_solar_v7.py test_v2_scores.py train_v2.log tuning_6_3_1.log ultimate_ensemble.py ultimate_run.log ultimate_strike.py upstage_index_20251224_144509.log upstage_index_full.pid upstage_index_full_20251224_145149.log upstage_index_full_20251224_150842.log v12_candidates_data.json v16_gemini_rerank_20251227_130830.log v16_gemini_rerank_resume_20251227_132912.log v16_gemini_rerank_resume_20251227_133454.log v16_gemini_rerank_resume_20251227_135442.log v16_gemini_rerank_resume_20251227_140859.log v16_gemini_rerank_resume_20251227_141006.log v16_gemini_rerank_resume_20251227_141102.log v16_gemini_rerank_resume_20251227_141131_30270.log v16_gemini_rerank_resume_20251227_141429_8676.log v5_score_gaps.json v7_fixed.log v9_v3_diffs.json verify_hybrid.py wait_then_generate.pid wait_then_generate_20251224_152314.log weighted_rrf_log.txt         크기:               6.81GB
```

### 데이터 파일
```
corpus.jsonl                ~10MB   (4,272 docs)
synthetic_qa_solar.jsonl    ~15MB   (12,816 QA)
train_data_v3.jsonl         ~150MB  (12,816 samples × 8 docs)

크기:             ~175MB
```

### 제출 파일
```
submission_*.csv            48KB ~ 440KB (평균 ~180KB)
20+ 파일                 ~4MB
```

---

## 
| 항목 | 수량 | 크기 |
|------|------|------|
| **원본 문서** | 4,272개 | ~10MB |
| **생성 QA** | 12,816개 | ~15MB |
| **학습 샘플** | 12,816개 | ~150MB |
| **파인튜닝 모델** | 3개 | 6.81GB |
| **제출 파일** | 20+ | ~4MB |
| **총 디스크 사용량** | - | ~7.5GB |

---

## 🚀  순서

### 1단계: 환경 설정
```bash
cd /root/IR
pip install -r requirements.txt
```

### 2단계: QA 생성
```bash
cd finetune
python 1_generate_qa.py
# → data/synthetic_qa_solar.jsonl 생성
```

### 3단계: Hard Negative Mining
```bash
python 2_mine_negatives_v3.py
 data/train_data_v3.jsonl 생성
```

### 4단계: 모델 학습
```bash
bash 3_run_train_v3.sh
# → finetuned_bge_m3_v3/ 생성
```

### 5단계: 평가
```bash
cd ..
python eval_rag_finetuned.py
# → submission_*.csv 생성
```

---

## 📁 주요 파일 상세

### `finetune/1_generate_qa.py`
**목적**: Solar Pro 2 API로 문서당 3개 질문 생성

**입력**:
- `data/corpus.jsonl` (4,272 docs)

**출력**:
- `data/synthetic_qa_solar.jsonl` (12,816 QA pairs)

**프로세**:
```python
for each document:
    context = document[:1000]  # 1000자 제한
    questions = solar_pro_2.generate(
        prompt="문서를 읽고 3개의 질문 생성",
        context=context
    )
    save_qa_pair(docid, questions, content)
```

---

### `finetune/2_mine_negatives_v3.py`
**목적**: Hybrid Retrieval로 Hard Negatives 7개 추출

**입력**:
- `data/synthetic_qa_solar.jsonl` (12,816 QA pairs)

**출력**:
- `data/train_data_v3.jsonl` (12,816 samples)

**프로세스**:
```python
for each qa_pair:
    # 1. BM25 Sparse Search
    bm25_candidates = elasticsearch.search(query, top_k=50)
    
    # 2. Dense Search
    dense_candidates = faiss.search(query_embedding, top_k=50)
    
    # 3. Pool Merge
    pool = merge_and_dedupe(bm25_candidates, dense_candidates)
    
    # 4. Reranker
    reranked = bge_reranker.rerank(query, pool)
    hard_negatives = reranked[:7]
    
    save_training_sample(query, positive_doc, hard_negatives)
```

---

### `finetune/3_run_train_v3.sh`
**목적**: BGE-M3 Contrastive Learning 실행

**입력**:
- `data/train_data_v3.jsonl` (12,816 samples)
- Base Model: `BAAI/bge-m3`

**출력**:
- `finetuned_bge_m3_v3/` (2.27GB model)

**하이퍼파라미터**:
```bash
--num_train_epochs 5
--per_device_train_batch_size 2
--gradient_accumulation_steps 16  # effective batch = 32
--learning_rate 1e-5
--temperature 0.02
--fp16
```

---

### `eval_rag_finetuned.py`
**목적**:.env .git .gitignore .last_v16_log .last_v16_out .vscode ANALYSIS_COLLEAGUE_CODE.md ANALYSIS_FINAL_RESULT.md ANALYSIS_SCORE_DROP.md BGE_M3_SOTA_OPTIMIZATION_REPORT_FINAL.md BGE_M3_SOTA_OPTIMIZATION_REPORT_LAST.md EMBEDDING_LLM_REPORT_20251224_193623.md EXPERIMENT_SUMMARY.md FINAL_REPORT.md FINETUNING_WORKFLOW_SUMMARY.md FULL_CONFIG_REPORT_20251224_194055.md FULL_CONFIG_REPORT_20251224_194637.md FULL_CONFIG_REPORT_20251224_194959.md FULL_CONFIG_REPORT_20251224_195400.md FULL_CONFIG_REPORT_20251224_195838.md GATING_STRATEGY_COMPREHENSIVE_REPORT.md GRID_SEARCH_LEADERBOARD.md LEADERBOARD_SUBMISSION_HISTORY.md NEXT_METHODS_AFTER_MAP08765.md OPTIMIZATION_STRATEGY.md PHASE_3_FAILURE_ANALYSIS.md PHASE_4_ANALYSIS_AND_STRATEGY.md PHASE_5_RECOVERY_PLAN.md PHASE_7_REPORT.md PHASE_7_SUMMARY.md README.md ROOT_CAUSE_ANALYSIS.md Report SOLAR_PRO2_OPTIMIZATION_REPORT.md SYNTHETIC_FINETUNING_COMPREHENSIVE_REPORT.md __pycache__ ab_llm_tiebreak_bge_m3.py ab_precompute.log analyze_742_failure.py analyze_742_testing.py analyze_all_strategies.py analyze_empty_cases.py analyze_ensemble_weights.py analyze_gating.py analyze_hyde_effect.py analyze_hyde_impact.py analyze_low_gaps_with_solar.py analyze_missed_ids.py analyze_reranker_effect.py analyze_top_diffs.py analyze_top_diffs_v2.py analyze_v9_v3_diff.py analyze_weight_tuning_success.py artifacts auditing auto_tuning.py baseline best_7008.ipynb best_9174.ipynb bge_m3_run.log bge_m3_run_final.log bge_m3_run_final_v2.log bge_m3_run_sota_v2.log bge_m3_run_v2.log bge_m3_run_v3.log bge_m3_run_v4.log bge_m3_run_v5.log bge_m3_run_v6.log build_final_union_rerank.py build_v17_conservative.py cache cache_search_results.py cache_step1.log check_271_303.py check_changes.py check_empty_queries.py check_final_changes.py check_gemini_models.py check_nogating.py check_v5_gaps.py cleanup_v3.py compare_all_top.py compare_gating_vs_submission.py compare_phase1_vs_planA.py compare_phase2_vs_planA.py compare_results.py compare_submissions.py compare_subs_clean.py compare_subs_sota_vs_best.py compare_top_submissions.py compare_v2_surg.py compare_v3_v9.py compare_v9_final.py compare_v9_final_v2.py compare_v9_final_v3.py compare_v9_final_v4.py compare_v9_final_v5.py compare_v9_v15.py comprehensive_experiment_analysis.py confidence_optimization_results.json consensus_rerank.py conservative_strike.py convert_v9_to_csv.py create_master.py create_v10_sota.py create_v11_sota.py create_v12_submission.py create_v13_submission.py create_v14_submission.py create_v15_submission.py data deep_scan.py deep_scan_v2.py detailed_experiment_comparison.py elasticsearch-8.8.0 ensemble_base_ft.py ensemble_final.py ensemble_run.log es_setup.py es_setup.py.backup es_setup_old.py eval_\[7\,4\,2\]_full.log eval_\[7\,4\,2\]_log.txt eval_finetuned_v9.log eval_rag.py eval_rag.py.bak eval_rag_bge_m3.py eval_rag_bge_m3_base.py eval_rag_bge_m3_v2.py eval_rag_bge_m3_v3.py eval_rag_bge_m3_v4.py eval_rag_bge_m3_v5.py eval_rag_bge_m3_v6.py eval_rag_bge_m3_v7.py eval_rag_bge_m3_v8_recovery.py eval_rag_e5_base.py eval_rag_e5_ensemble.py eval_rag_e5_final.py eval_rag_e5_hybrid.py eval_rag_e5_multi.py eval_rag_e5_repro.py eval_rag_e5_sota.py eval_rag_e5_ultimate.py eval_rag_final_strategy.py eval_rag_finetuned.log eval_rag_finetuned.py eval_rag_finetuned_v9.py eval_rag_no_gating.py eval_rag_rerank_ensemble.py eval_rag_topk60.py eval_rag_v11_full_solar.py eval_rag_v16_gemini_rerank.py eval_rag_v2_final.py eval_rag_v3_ensemble.py eval_rag_v8_v5_queries.py eval_rag_v9_sota.py eval_rag_weight552.py eval_rag_weighted_rrf.py eval_v3.log eval_v3_ensemble.log eval_v3_fixed.log eval_v3_fixed_2.log evaluation_gating_v2.log evaluation_with_gating.log experiment_cp100_20251223_080055.log experiment_topk80_20251223_063042.log experiment_topk80_20251223_063501.log experiment_topk80_20251223_063621.log experiment_topk80_20251223_063621.pid experiment_topk80_run.log experiments fast_alpha_sweep.py fill_empty.py fill_empty_v2.py final_comprehensive_report.py final_strategy.log final_strategy.py final_summary.py final_surgical_check.py finalize_submission.py find_v9_v3_diffs.py finetune finetuned_bge_m3 finetuned_bge_m3_v2 finetuned_bge_m3_v3 fix_v9_order.py gemini_indexing.log gemini_run.log gemini_run.pid generate_candidates.py generate_final_challenge.py generate_final_last_chance.py generate_final_surgical.py generate_final_surgical_v2.py generate_hybrid_s33gating_wrrf.py generate_qa.log generate_super_hybrid.py generate_synthetic_qa.py gpt4o_run.log grid_search.py grid_search_cached.py grid_search_results.json grid_search_step2.log hyde_evaluation.log hyde_planA.log hyde_test.log inspect_v11_changes.py inspect_v9_choices.py judge_decisions.json judge_mismatches.py judge_report.json judge_results.json last_mq120_submission_log.txt last_mq120_submission_path.txt last_mq_submission_log.txt last_mq_submission_path.txt list_empty.py log_ab_baseline.jsonl log_ab_gpt4o_gap015.jsonl log_ab_gpt4o_sample.jsonl log_ab_gpt4o_sample2.jsonl log_ab_solar_gap0.05.jsonl log_ab_solar_gap0.10.jsonl log_ab_solar_gap0.20.jsonl log_ab_solar_gap015.jsonl log_ab_solar_sample.jsonl log_ab_solar_sample2.jsonl main.py main_eval_final.log main_eval_solar.log main_eval_solar.pid main_eval_solar_v2.log main_eval_solar_v2.pid main_eval_solar_v3.log main_eval_solar_v3.pid main_reranker.log main_reranker_optimized.log main_run.log main_run_improved.log merge_v9_v3.py mine_v2.log models optimize_confidence.py optuna_search.py phase2_tuning.log phase_2_1_evaluation.log phase_3_1_test.log phase_3_full_evaluation.log phase_4a_evaluation.log phase_4b_evaluation.log phase_4c_evaluation.log phase_4d_evaluation.log phase_4d_nogating_evaluation.log phase_4d_topk60_evaluation.log phase_5_evaluation.log phase_6a_evaluation.log phase_6a_evaluation_v2.log phase_6a_final.log phase_6b1_evaluation.log phase_7_evaluation.log phase_7_evaluation_real.log phase_7_new.log phase_8_evaluation.log phase_9_evaluation.log pipeline_v3.log precision_strike.py prepare_judge.py prepare_v12_candidates.py progress.log requirements.txt rerank_ensemble.log result_gate result_gem result_multi retrieval run_bge_m3_sota_20251229_023154.log run_bge_m3_sota_env.sh run_eval_742.sh run_judge.py run_rrf_k20_20251224_060251.log run_rrf_k20_20251224_060339.log run_rrf_k20_mq_cp120_upstageHeavy_20251224_172309.log run_rrf_k20_mq_tk120_cp120_20251224_071802.log run_rrf_k20_mq_tk120_cp120_20251224_080338.log run_rrf_k20_mq_tk120_cp120_20251224_082326.log run_rrf_k20_mq_tk120_cp120_upstageOnly_20251224_165428.log run_rrf_k20_mq_tk80_cp80_20251224_071819.log run_rrf_k20_mq_tk80_cp80_20251224_072844.log run_rrf_k20_mq_tk80_cp80_dense3_upstage2048_20251224_154415.log run_rrf_mq_20251225_010454.log run_single_eval.py run_strategy_20251224_202157.log run_strategy_v2_20251225_002516.log run_tests.sh run_tk100_cp80_20251223_152050.log run_tk100_cp80_20251223_152141.log run_tk100_cp80_20251224_023753.log run_tuning_grid.sh run_v2_final.log run_v2_final.sh run_v3_pipeline.sh run_v7_solar.log scripts search_results_cache.json snapshot_submission.py solar_diff_analysis.json solar_gating_audit.json solar_low_gap_improvements.json strategy_a_evaluation.log submission.csv submission_18\(14\).csv submission_19.csv submission_20.csv submission_38_ready_rrf_k20_mq_tk80_cp80_dense3_20251224_114800.csv submission_39_ready_rrf_k20_mq_tk80_cp80_dense3_upstage2048_20251224_154415.csv submission_40_ready_rrf_k20_mq_tk120_cp120_upstageOnly_20251224_165428.csv submission_41_ready_rrf_k20_mq_cp120_upstageHeavy_20251224_172309.csv submission_42_strategy_tk100_cp100_h300_mq_20251224_202157.csv submission_43_strategy_v2_tk100_cp100_h300_mq_20251225_002516.csv submission_44_rrf_k30_mq_tk100_cp100_20251225_010454.csv submission_45_hybrid_s33gating_wrrf_search.csv submission_46_final_strategy.csv submission_47_e5_final.csv submission_48_e5_hybrid.csv submission_49_e5_sota.csv submission_50_e5_solar_pro.csv submission_51_e5_gemini.csv submission_52_e5_ultimate.csv submission_53_e5_super_ensemble.csv submission_54_bge_m3_sota.csv submission_55_bge_m3_sota.csv submission_56_bge_m3_sota_v3.csv submission_57_bge_m3_sota_v4.csv submission_58_bge_m3_sota_v5.csv submission_59_bge_m3_sota_v6.csv submission_60_bge_m3_sota_v7.csv submission_61_bge_m3_solar_sota.csv submission_62_v8_v5_queries_solar_tiebreak.csv submission_63_v9_sota.csv submission_64_v12_sota.csv submission_65_v13_sota.csv submission_66_v14_sota.csv submission_67_v15_sota.csv submission_68_v16_gemini_rerank_20251227_130830.csv submission_69_v17_conservative_from_v9_20251227_145004.csv submission_70_v17_safe3_from_v9_20251227_150049.csv submission_71_v17_attack5_from_v9_20251227_150049.csv submission_72_final_union_rerank_v18.csv submission_73_ensemble_base0.7_ft0.3.csv submission_74_ensemble_base0.5_ft0.5.csv submission_75_ensemble_base0.8_ft0.2.csv submission_76_v2_final_rerank.csv submission_77_final_ensemble_v9_v2.csv submission_78_final_v2_precision.csv submission_79.csv submission_80_v3_final_rerank.csv submission_81_v3_final.csv submission_82_surgical_v1.csv submission_83_final_0.95_break.csv submission_84_final_0.95_break_v2.csv submission_85_final_0.95_master.csv submission_86_candidate_B_id271.csv submission_87_candidate_D_id271_id303.csv submission_88_ready_bge_m3_sota_20251229_023154.csv submission_89_grid_v2_mq_off_20251229_025014.csv submission_90_final_challenge_0.95.csv submission_91_final_surgical_v2_id270_only.csv submission_92_final_last_chance.csv submission_93_grid_v3_tk200_20251229_025014.csv submission_ab_baseline.csv submission_ab_gpt4o_gap015.csv submission_ab_gpt4o_sample.csv submission_ab_gpt4o_sample2.csv submission_ab_solar_gap0.05.csv submission_ab_solar_gap0.10.csv submission_ab_solar_gap0.20.csv submission_ab_solar_gap015.csv submission_ab_solar_sample.csv submission_ab_solar_sample2.csv submission_backup_old.csv submission_backup_phase6b.csv submission_baseline_map08765_20251223_063042.csv submission_before_cp100_20251223_080055.csv submission_before_reranker.csv submission_before_topk80_20251223_063621.csv submission_best_9174.csv submission_best_9273.csv submission_best_9394.csv submission_best_map08765.csv submission_bge_m3_base_simple.csv submission_bge_m3_finetuned.csv submission_bge_m3_finetuned_v9.csv submission_bge_m3_sota.csv submission_bge_m3_sota_v3.csv submission_bge_m3_sota_v4.csv submission_bge_m3_sota_v5.csv submission_bge_m3_sota_v6.csv submission_bge_m3_sota_v7.csv submission_bge_m3_v2_ft.csv submission_candidate_A_surgical.csv submission_candidate_B_id271.csv submission_candidate_C_id303.csv submission_candidate_D_id271_id303.csv submission_conservative_strike.csv submission_cp100_20251223_104822.csv submission_diffs.json submission_e5_base.csv submission_e5_final.csv submission_e5_gemini.csv submission_e5_gpt4o.csv submission_e5_hybrid.csv submission_e5_multi.csv submission_e5_repro.csv submission_e5_solar_pro.csv submission_e5_sota.csv submission_e5_super_ensemble.csv submission_e5_ultimate.csv submission_ensemble_base0.5_ft0.5.csv submission_ensemble_base0.7_ft0.3.csv submission_ensemble_base0.8_ft0.2.csv submission_final_0.95_break.csv submission_final_0.95_break_v2.csv submission_final_0.95_master.csv submission_final_challenge_0.95.csv submission_final_ensemble_v9_v2.csv submission_final_strategy.csv submission_final_surgical_hybrid_0.95.csv submission_final_surgical_v2_id270_only.csv submission_final_union_rerank_4sources.csv submission_final_union_rerank_v18.csv submission_final_v2_precision.csv submission_grid_v1_llm_on_20251229_025014.csv submission_grid_v2_mq_off_20251229_025014.csv submission_grid_v3_tk200_20251229_025014.csv submission_hybrid_s33gating_wrrf_search.csv submission_hyde_v1.csv submission_nogating.csv submission_old.csv submission_old_0.csv submission_partial_before_solar_fullrun_20251222_230943.csv submission_phase7_failed.csv submission_planA.csv submission_pre_topk80_20251223_063501.csv submission_precision_strike.csv submission_ready_5_tk100_cp80_20251223_152141.csv submission_ready_bge_m3_sota_20251229_023154.csv submission_ready_rrf_k20_mq_cp120_upstageHeavy_20251224_172309.csv submission_ready_rrf_k20_mq_tk120_cp120_20251224_071802.csv submission_ready_rrf_k20_mq_tk120_cp120_20251224_080338.csv submission_ready_rrf_k20_mq_tk120_cp120_20251224_082326.csv submission_ready_rrf_k20_mq_tk120_cp120_upstageOnly_20251224_165428.csv submission_ready_rrf_k20_mq_tk80_cp80_20251224_071819.csv submission_ready_rrf_k20_mq_tk80_cp80_20251224_072844.csv submission_ready_rrf_k20_mq_tk80_cp80_dense3_20251224_114800.csv submission_ready_rrf_k20_mq_tk80_cp80_dense3_upstage2048_20251224_154415.csv submission_ready_rrf_k20_tk80_cp80_20251224_060339.csv submission_rerank_ensemble_v1.csv submission_reranker.csv submission_snapshot.json submission_solar_final_sota.csv submission_solar_mq_tiebreak_v7.csv submission_solar_precheck_backup_20251222_191832.csv submission_solar_v2_scienceonly_20251223_000954.csv submission_submitted_07697_20251222_234512.csv submission_super_hybrid_final.csv submission_super_hybrid_final_v2.csv submission_surgical_v1.csv submission_topk60.csv submission_ultimate_ensemble_v1.csv submission_ultimate_strike.csv submission_v11_sota.csv submission_v12_sota.csv submission_v13_sota.csv submission_v14_sota.csv submission_v15_sota.csv submission_v16_gemini_rerank_20251227_130830.csv submission_v16_gemini_rerank_smoke.csv submission_v17_attack5_from_v9_20251227_150036.csv submission_v17_attack5_from_v9_20251227_150049.csv submission_v17_conservative_from_v9_20251227_145004.csv submission_v17_safe3_from_v9_20251227_150036.csv submission_v17_safe3_from_v9_20251227_150049.csv submission_v2_final_rerank.csv submission_v3_ensemble.csv submission_v3_final.csv submission_v3_final_rerank.csv submission_v3_v9_rrf_64.csv submission_v3_v9_rrf_82.csv submission_v8_recovery_recovery.csv submission_v8_v5_queries_solar_tiebreak.csv submission_v9_sota.csv submission_weighted_rrf.csv surgical_strike.py test_alpha_on_diffs.py test_configs.py test_embedding_change.py test_gemini_rerank.py test_hyde_eval.py test_hyde_quality.py test_parameter_tuning.py test_phase_3_1.py test_solar_v7.py test_v2_scores.py train_v2.log tuning_6_3_1.log ultimate_ensemble.py ultimate_run.log ultimate_strike.py upstage_index_20251224_144509.log upstage_index_full.pid upstage_index_full_20251224_145149.log upstage_index_full_20251224_150842.log v12_candidates_data.json v16_gemini_rerank_20251227_130830.log v16_gemini_rerank_resume_20251227_132912.log v16_gemini_rerank_resume_20251227_133454.log v16_gemini_rerank_resume_20251227_135442.log v16_gemini_rerank_resume_20251227_140859.log v16_gemini_rerank_resume_20251227_141006.log v16_gemini_rerank_resume_20251227_141102.log v16_gemini_rerank_resume_20251227_141131_30270.log v16_gemini_rerank_resume_20251227_141429_8676.log v5_score_gaps.json v7_fixed.log v9_v3_diffs.json verify_hybrid.py wait_then_generate.pid wait_then_generate_20251224_152314.log weighted_rrf_log.txt  평가 및 제출 파일 생성 

**입력**:
- `finetuned_bge_m3_v3/` (학습된 모델)
- `data/test.jsonl` (220 queries)

**출력**:
- `submission_*.csv` (220 rows)

**프로세스**:
```python
# 1. Load fine-tuned model
model = load_finetuned_bge_m3("finetuned_bge_m3_v3")

# 2. Build index
index = build_faiss_index(corpus, model)

# 3. Evaluate
for query in test_queries:
    # HyDE expansion
    hyde_query = gemini_hyde(query)
    
    # Sparse + Dense retrieval
    bm25_results = bm25_search(hyde_query)
    dense_results = faiss_search(hyde_query, model, index)
    
    # Hard Voting
    voted = hard_vote(bm25_results, dense_results, weights=[6,3,1])
    
    # Reranker
    final = rerank(query, voted[:20], top_k=5)
    
    save_submission(query_id, final)
```

---

## 🎯 파일 역할 매핑

| 파일 | 역할 | 입력 | 출력 |
|------|------|------|------|
| `1_generate_qa.py` | QA 생성 | corpus.jsonl | synthetic_qa_solar.jsonl |
| `2_mine_negatives_v3.py` | Hard Negative | synthetic_qa_solar.jsonl | train_data_v3.jsonl |
| `3_run_train_v3.sh` | 모델 학습 | train_data_v3.jsonl | finetuned_bge_m3_v3/ |
| `eval_rag_finetuned.py` | 평가 | test.jsonl + model | submission_*.csv |

---

## 💡 파일 명명 규칙

### Submission 파일
```
submission_{번호}_{모델}_{버}_{특징}.csv

:
- submission_54_bge_m3_sota.csv          # 54번 제출, bge_m3, sota 설정
- submission_56_bge_m3_sota_v3.csv       # v3 모델 사용
- submission_61_bge_m3_solar_sota.csv    # Solar 통합
- submission_88_ready_bge_m3_*.csv       # 최종 제출 (88번)
```

### 모델 디렉토리
```
finetuned_bge_m3_{버전}/

:
- finetuned_bge_m3/           # v1 (초기)
- finetuned_bge_m3_v2/        # v2 (개선)
- finetuned_bge_m3_v3/        # v3 (최종)
```

### 데이터 파일
```
{목적}_{버전}.jsonl

:
- corpus.jsonl                # 원본 (버전 없음)
- synthetic_qa_solar.jsonl    # Solar로 생성
- train_data_v3.jsonl         # v3 학습 데이터
```

---

## 📚 관련 문

- **종합 보고서**: [SYNTHETIC_FINETUNING_COMPREHENSIVE_REPORT.md](SYNTHETIC_FINETUNING_COMPREHENSIVE_REPORT.md)
- **워크플로우 요약**: [FINETUNING_WORKFLOW_SUMMARY.md](FINETUNING_WORKFLOW_SUMMARY.md)
- **리더�**: [LEADERBOARD_SUBMISSION_HISTORY.md](LEADERBOARD_SUBMISSION_HISTORY.md)�드 

---

**작성일**: 2025년 12 29일  
**버전**: v1.0  
**문서 유형**: 폴더 구조 시각화
