# Graph Report - MLIC  (2026-09-30)

## Corpus Check
- 57 files · ~311,426 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 36 file(s) not represented in the graph (top: .log 34, (none) 2)

## Summary
- 564 nodes · 1009 edges · 30 communities (18 shown, 12 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 114 edges (avg confidence: 0.85)
- Token cost: 95,872 input · 0 output

## Community Hubs (Navigation)
- Training Entry & Datamodules
- Model Package Imports
- MLIC++ Architecture Diagram
- Attention Config & Datasets
- SSP Structure Losses
- Residual & Attention Blocks
- Codec Utilities
- MLICPlusPlus Model Core
- RD Loss Module
- Positional Peak Losses
- Parameter & Patch Helpers
- Kodak RD Benchmark
- Depth Smoothing & Loss Checks
- Image Quality Metrics
- Dynamic Loss Weighting
- Tecnick RD Benchmark
- Spectral & Rate Reference
- Training Resume Logic
- Gradient Term Monitor
- CLIC Pro RD Benchmark
- Fixed Weight SSP Loss
- LR Schedulers
- Structure Loss Mixin
- Launch Script
- Test Script
- MS-SSIM Train Script
- Train Script
- Warmup Script
- TensorBoard Script

## God Nodes (most connected - your core abstractions)
1. `MLICPlusPlus` - 20 edges
2. `main()` - 18 edges
3. `MLIC++` - 17 edges
4. `Kodak Rate-Distortion Curve (PSNR vs bpp)` - 14 edges
5. `DynamicLossWeightingSSPLoss` - 12 edges
6. `main()` - 11 edges
7. `BD-Rate Benchmark vs VTM-17.0 Intra` - 11 edges
8. `Tecnick Rate-Distortion Curve (PSNR vs bpp)` - 11 edges
9. `StructureLossMixin` - 10 edges
10. `FixedWeightSSPLoss` - 10 edges

## Surprising Connections (you probably didn't know these)
- `EntropyParametersEX` --implements--> `Entropy Parameters g_ep`  [AMBIGUOUS]
  MLIC/modules/transform/entropy.py → assets/arch.png
- `MLICPlusPlus` --implements--> `MLIC++ Architecture Diagram`  [INFERRED]
  MLIC/models/mlicpp.py → assets/arch.png
- `MLICPlusPlus` --implements--> `MLIC++ (Ours, retrained)`  [INFERRED]
  MLIC/models/mlicpp.py → assets/clicp.png
- `MLICPlusPlus` --implements--> `MLIC++ (Ours)`  [INFERRED]
  MLIC/models/mlicpp.py → assets/kodak.png
- `MLICPlusPlus` --implements--> `MLIC++ (Ours, retrained)`  [INFERRED]
  MLIC/models/mlicpp.py → assets/tecnick.png

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **MLIC++ multi-reference context modules** — mlic_attention_config_channel_context_attention, mlic_attention_config_local_context_attention, mlic_attention_config_global_inter_context_attention, mlic_attention_config_global_intra_context_attention, readme_multi_reference_entropy_model [INFERRED 0.85]
- **MLIC-Train-100K composition** — readme_mlic_train_100k, readme_div2k, readme_flicker2k, readme_clic_train, readme_coco, readme_imagenet, train_list_training_image_list [EXTRACTED 1.00]
- **BD-rate evaluation on Kodak/Tecnick/CLIC** — readme_bd_rate_benchmark, readme_kodak, readme_tecnick, readme_clic_pro_valid, readme_vtm_17_0_intra [EXTRACTED 1.00]
- **MEM++ four context types** — assets_arch_channel_wise_context, assets_arch_local_spatial_context, assets_arch_intra_global_context, assets_arch_inter_global_context [EXTRACTED 1.00]
- **Hyperprior path** — assets_arch_hyper_analysis, assets_arch_factorized_entropy_model, assets_arch_hyper_synthesis, assets_arch_entropy_parameters [EXTRACTED 1.00]
- **MLIC family variants** — assets_clicp_mlic, assets_clicp_mlic_plus, assets_clicp_mlic_plus_plus [INFERRED 0.85]
- **Classical codec baselines** — assets_kodak_vtm_17, assets_kodak_bpg, assets_kodak_jpeg2000, assets_kodak_jpeg [INFERRED 0.85]
- **MLIC family variants** — assets_kodak_mlic, assets_kodak_mlic_plus, assets_kodak_mlic_plus_plus [INFERRED 0.85]
- **MLIC family variants** — assets_tecnick_mlic, assets_tecnick_mlic_plus, assets_tecnick_mlic_plus_plus [INFERRED 0.85]

## Communities (30 total, 12 thin omitted)

### Community 0 - "Training Entry & Datamodules"
Cohesion: 0.05
Nodes (20): test_options(), train_options(), model_config(), RateDistortionLoss, main(), compute_total_bits(), log_compression_rate_before_training(), main() (+12 more)

### Community 1 - "Model Package Imports"
Cohesion: 0.05
Nodes (9): ResidualBottleneck, AnalysisTransform, ChannelWiseEntropyParameters, EntropyParameters, EntropyParametersEX, CorrectLatentResidualPrediction, LatentResidualPrediction, LatentResidualPredictionOld (+1 more)

### Community 2 - "MLIC++ Architecture Diagram"
Cohesion: 0.06
Nodes (27): MLIC++ Architecture Diagram, Analysis Transform g_a, Arithmetic Encoder/Decoder (AE/AD), Channel-wise Context, Entropy Parameters g_ep, Factorized Entropy Model, Hyper Analysis h_a, Hyper Synthesis h_s (+19 more)

### Community 3 - "Attention Config & Datasets"
Cohesion: 0.07
Nodes (35): enable_channel_context flag, enable_global_inter_context flag, enable_global_intra_context flag, enable_local_context flag, MLICPlusPlus Model Configuration, BD-Rate Benchmark vs VTM-17.0 Intra, CLIC Pro Valid, CLIC Train (+27 more)

### Community 4 - "SSP Structure Losses"
Cohesion: 0.11
Nodes (16): denormalize(), _depth_view(), half_window_levels(), local_extrema_pos_loss(), matched_extrema_pos_loss(), metre_windows(), _multiscale_relief(), prominence_recall_loss() (+8 more)

### Community 5 - "Residual & Attention Blocks"
Cohesion: 0.08
Nodes (9): Residual Block (downsampling), conv1x1(), AttentionBlock, __init__(), ResidualBlock, ResidualBlockUpsample, ResidualBlockWithStride, HyperAnalysis (+1 more)

### Community 6 - "Codec Utilities"
Cohesion: 0.09
Nodes (12): compress_one_image(), decompress_one_image(), AverageMeter, Config, filesize(), read_body(), read_bytes(), read_uints() (+4 more)

### Community 7 - "MLICPlusPlus Model Core"
Cohesion: 0.18
Nodes (13): MLICPlusPlus, ckbd_anchor(), ckbd_anchor_sequeeze(), ckbd_anchor_unsequeeze(), ckbd_merge(), ckbd_nonanchor(), ckbd_nonanchor_sequeeze(), ckbd_nonanchor_unsequeeze() (+5 more)

### Community 8 - "RD Loss Module"
Cohesion: 0.11
Nodes (4): diff_mask(), differentiable_extrema_mask(), estimate_optimal_weights(), HeteroscedasticSSPLoss

### Community 9 - "Positional Peak Losses"
Cohesion: 0.19
Nodes (6): curvature_weighted_loss(), position_and_value_loss(), soft_peak_localization_loss(), wasserstein_peak_alignment_loss(), weighted_deriv_loss(), weighted_mse_loss()

### Community 10 - "Parameter & Patch Helpers"
Cohesion: 0.13
Nodes (7): cal_params(), calc_params(), find_named_buffer(), find_named_module(), get_scale_table(), _update_registered_buffer(), update_registered_buffers()

### Community 11 - "Kodak RD Benchmark"
Cohesion: 0.14
Nodes (15): Kodak Rate-Distortion Curve (PSNR vs bpp), Bit-rate [bpp], BPG, Cheng'20 (CVPR'20), Kodak dataset, ELIC (CVPR'22), JPEG, JPEG2000 (+7 more)

### Community 12 - "Depth Smoothing & Loss Checks"
Cohesion: 0.21
Nodes (4): _affine_from_norm_stats(), check_loss_dict(), depth_smoothing_matrix(), resolve_significant_depth_idx()

### Community 13 - "Image Quality Metrics"
Cohesion: 0.19
Nodes (4): compute_metrics(), compute_psnr(), get_f1_score(), get_min_max_idx()

### Community 15 - "Tecnick RD Benchmark"
Cohesion: 0.18
Nodes (12): Tecnick Rate-Distortion Curve (PSNR vs bpp), Bit-rate [bpp], Tecnick dataset, JPEG2000, LIC-TCM Large (CVPR'23), Minnen'20 (ICIP'20), MLIC (Ours), MLIC+ (Ours) (+4 more)

### Community 16 - "Spectral & Rate Reference"
Cohesion: 0.17
Nodes (4): HomoscedasticSSPLoss, power_spectrum(), rate_reference_bits_per_profile(), spectral_loss()

### Community 17 - "Training Resume Logic"
Cohesion: 0.20
Nodes (3): criterion_state(), restore_criterion_state(), _weights_from_train_log()

### Community 19 - "CLIC Pro RD Benchmark"
Cohesion: 0.24
Nodes (10): CLIC Professional Valid Rate-Distortion Curve (PSNR vs bpp), Bit-rate [bpp], Cheng'20, CLIC Professional Valid dataset, LIC-TCM Large (CVPR'23), MLIC (Ours), MLIC+ (Ours), MLIC++ (Ours, retrained) (+2 more)

## Ambiguous Edges - Review These
- `EntropyParametersEX` → `Entropy Parameters g_ep`  [AMBIGUOUS]
  assets/arch.png · relation: implements

## Knowledge Gaps
- **52 isolated node(s):** `tensorboard.sh script`, `test.sh script`, `PYTHONPATH`, `train.sh script`, `PYTHONPATH` (+47 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 224 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `EntropyParametersEX` and `Entropy Parameters g_ep`?**
  _Edge tagged AMBIGUOUS (relation: implements) - confidence is low._
- **Why does `MLICPlusPlus` connect `MLICPlusPlus Model Core` to `Training Entry & Datamodules`, `Model Package Imports`, `MLIC++ Architecture Diagram`, `Attention Config & Datasets`, `Parameter & Patch Helpers`, `Kodak RD Benchmark`, `Tecnick RD Benchmark`, `CLIC Pro RD Benchmark`?**
  _High betweenness centrality (0.296) - this node is a cross-community bridge._
- **Why does `main()` connect `Training Entry & Datamodules` to `MLICPlusPlus Model Core`, `Dynamic Loss Weighting`, `Spectral & Rate Reference`, `Training Resume Logic`, `Gradient Term Monitor`, `Fixed Weight SSP Loss`, `LR Schedulers`?**
  _High betweenness centrality (0.140) - this node is a cross-community bridge._
- **Why does `MLICPlusPlus Model Configuration` connect `Attention Config & Datasets` to `MLICPlusPlus Model Core`?**
  _High betweenness centrality (0.132) - this node is a cross-community bridge._
- **Are the 9 inferred relationships involving `MLICPlusPlus` (e.g. with `MLICPlusPlus Model Configuration` and `MLIC++ Architecture Diagram`) actually correct?**
  _`MLICPlusPlus` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `main()` (e.g. with `train_options()` and `DynamicLossWeightingSSPLoss`) actually correct?**
  _`main()` has 15 INFERRED edges - model-reasoned connections that need verification._
- **What connects `tensorboard.sh script`, `test.sh script`, `PYTHONPATH` to the rest of the system?**
  _52 weakly-connected nodes found - possible documentation gaps or missing edges._