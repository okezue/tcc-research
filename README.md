# tcc-research

Code and experimental results for Fisher-geometric privacy of hidden-state release in decoder-only transformers: an optimal noise-injection defense against adaptive retrieval attackers.

The repository supports a single research question: given that transformer hidden states are injective in the prompt (Nikolaou et al., 2026), how should a defender add Gaussian noise to a released activation so that a Bayes-optimal attacker cannot recover the prompt, while the downstream next-token distribution is preserved?

## Headline results

- **Closed-form optimal defense against an adaptive attacker.** The Mahalanobis-optimal noise covariance is $\Sigma^\star_{\mathrm{Mah}} = (\kappa/\mathrm{tr}(C^{1/2}))\, F_\lambda^{-1/2} C^{1/2} F_\lambda^{-1/2}$ where $C = F_\lambda^{1/2} \Sigma_\delta F_\lambda^{1/2}$, $F$ is the hidden-state Fisher, and $\Sigma_\delta$ is the margin-direction covariance. Derived as a trace minimization with a scalar utility budget.
- **Predictive scalar.** $G_{\mathrm{Mah}} = \mathrm{tr}(F_\lambda)\,\mathrm{tr}(\Sigma_\delta) / [\mathrm{tr}(C^{1/2})]^2$ forecasts the gain over isotropic noise against a covariance-aware attacker. Measured in the range 1.7–9.3 across GPT-2 Small, Mistral-7B, Phi-2, Qwen3-14B, DeepSeek-R1-14B.
- **The 13× Pareto defense is a Euclidean-attacker artifact.** Under plain $\ell_2$ retrieval, generalized-eigen noise on the top-128 eigenvectors of $(\Sigma_\delta, F)$ suppresses attack success by 13× on Mistral-7B. Under the adaptive Mahalanobis attacker the noise lives entirely in a rank-128 subspace the attacker can project out, and attack success goes back to 100% at every noise level.
- **A simple mechanism that survives.** $\Sigma_{\mathrm{diag}} = \sigma^2\,\mathrm{diag}(1/F_{ii})$ — diagonal, full-rank, coordinate-weighted by inverse Fisher — is the only Gaussian release that achieves worst-over-attackers top-1 ≤ 0.001 at $\sigma = 5$ across every (model, layer) point we tested. It is also the unique diagonal covariance with equal per-coordinate first-order KL cost.
- **Empty middle.** Across 1,536 (mechanism, $\sigma$, model, layer) cells, zero achieve both t1-agreement ≥ 0.5 with the clean model and worst-attacker top-1 ≤ 0.5.
- **Training-time DP does not help inference release.** DP-SGD fine-tuning at $\varepsilon \in \{2, 4, 8\}$ leaves held-out retrieval top-1 at 1.000.

Numbers in the paper's tables are reproducible from the JSONs under `artifacts/`.

## For researchers

### Layout
- `two_channel/mahalanobis_defense.py` — solves the Mahalanobis-optimal $\Sigma^\star_{\mathrm{Mah}}$ and computes $G_{\mathrm{Mah}}$, $G_{\mathrm{Euc}}$.
- `two_channel/mahalanobis_attacker.py` — Bayes-optimal retrieval with $(q-c)^\top (\Sigma + \tau I)^{-1} (q-c)$; $\tau$ tuned on held-out.
- `two_channel/rdp_accountant.py` — Rényi-DP accountant $\varepsilon_\alpha(\Sigma) = (\alpha/2)\sup_{\Delta} \Delta^\top \Sigma^{-1} \Delta$ over an adjacency set.
- `two_channel/adjacency_builder.py` — constructs the empirical adjacency $\mathcal{A}$ per model.
- `two_channel/sdp_worst_case.py` — cvxpy SDP for reduced-basis worst-case DP covariance.
- `two_channel/learned_inverter.py` — 6-layer decoder with activation cross-attention; attack upper bound for learned inversion.
- `two_channel/compute_subspace.py` — empirical Fisher $F$ and margin covariance $\Sigma_\delta$ at an arbitrary layer.
- `two_channel/exp_mahalanobis.py` — full defense × attacker × mechanism sweep (the script that produces `artifacts/mahalanobis/*.json`).
- `two_channel/exp_matched_eps.py`, `exp_rdp_curves.py` — matched-$\varepsilon$ and RDP curves.
- `two_channel/exp_dp_sgd.py` — DP-LoRA fine-tune + inference-time retrieval eval (Opacus).
- `two_channel/exp_learned_inverter.py` — learned-inverter training (GPT-2 Small, Mistral-7B).
- `two_channel/exp_isotropy_check.py` — fixed-projector isotropy measurement.
- `two_channel/exp_layer_sweep.py`, `exp_scaling_points.py` — per-layer and per-model sweeps.
- `artifacts/mahalanobis/runpod_L_*/` — per-layer defense sweeps for each model (primary result files).
- `artifacts/matched_eps/`, `artifacts/rdp/` — DP accounting.
- `artifacts/learned_inverter/` — GPT-2 Small and Mistral-7B inverter training/eval.
- `artifacts/dp_sgd/` — DP-SGD negative baseline.
- `artifacts/sdp/` — worst-case SDP (GPT-2).
- `artifacts/isotropy/` — fixed-projector isotropy checks.
- `artifacts/layer_sweep/` — GPT-2 layer sweep.

### Reproducing a result
Closed-form Mahalanobis defense on Mistral-7B, layer 16:
```
python -m two_channel.exp_mahalanobis --model mistralai/Mistral-7B-v0.1 --layer 16 --k 128 --n_cal 500 --n_bank 50000 --n_query 2000
```
Produces `artifacts/mahalanobis/mahalanobis_mistralai_Mistral-7B-v0.1.json` with $G_{\mathrm{Euc}}$, $G_{\mathrm{Mah}}$, the top generalized eigenvalues, and the full 9-mechanism × 6-$\sigma$ × 3-attacker sweep. End-to-end wall-clock on an H100 is ≈45 minutes.

Matched-$\varepsilon$ curves:
```
python -m two_channel.exp_matched_eps --model mistralai/Mistral-7B-v0.1 --layer 16 --k 128
```

DP-SGD baseline (GPT-2 Small):
```
python -m two_channel.exp_dp_sgd --eps_target 2 --delta 1e-6
```
Produces `artifacts/dp_sgd/dp_sgd_eps2.json`.

Full reproduction of every paper number takes ≈170 GPU-hours on a mix of H100 (for 7B/14B defense sweeps and Mistral learned-inverter training) and A10G (for GPT-2 sweeps and DP-SGD).

## For practitioners

If you are caching or transmitting hidden-state activations from a decoder-only transformer and want a Gaussian release that blocks a retrieval-based inversion attacker, the short answer is:

1. Compute $F_{ii}$, the diagonal of the empirical Fisher, on a calibration set of prompts at your chosen layer. $O(d)$ memory, one forward+backward pass per calibration prefix.
2. Release $\tilde h = h + \xi$, $\xi \sim \mathcal{N}(0, \sigma^2\,\mathrm{diag}(1/F_{ii}))$ for a noise scale $\sigma$ chosen by your utility budget. A minimal snippet:
   ```
   from two_channel.compute_subspace import fisher_diagonal
   F_diag = fisher_diagonal(model, calibration_prefixes, layer=ell)
   noise_cov = sigma**2 / F_diag        # shape (d,)
   released = h + noise_cov.sqrt() * torch.randn_like(h)
   ```
3. What you get: attack success ≤ 0.001 at $\sigma = 5$ on every model and layer we tested. The attacker's Mahalanobis score cannot beat isotropic on a full-rank diagonal-Fisher release, and the per-coordinate noise budget is invariant across layers because noise mass $\sigma^2 d$ absorbs variation in $\mathrm{tr}(F)$.
4. What you do *not* get: utility preservation. At $\sigma = 5$ the next-token distribution is badly corrupted (KL ≥ 5 nats on every model). The empty-middle result says that no Gaussian mechanism we tested achieves both t1-agreement ≥ 0.5 and attack top-1 ≤ 0.5 simultaneously. If your utility budget is tighter, pick a smaller $\sigma$ and accept a proportionally stronger attacker.
5. What *not* to do: do not use a rank-$k$ defense (gen-eigen, complement, random projection). They all collapse to 100% attack success under an adaptive Mahalanobis attacker because their covariance is singular. They only help against a naive $\ell_2$ attacker.
6. DP-SGD fine-tuning does not substitute for inference-time noise — our DP-SGD baseline at $\varepsilon \in \{2, 4, 8\}$ leaves fresh-prompt retrieval at 100%.

## Install
```
pip install -r requirements.txt
```
Requires PyTorch (with CUDA for 7B/14B runs), Hugging Face `transformers` + `datasets`, `opacus` (for the DP-SGD baseline), `cvxpy` (for the SDP baseline), `faiss-cpu` (optional, for larger retrieval banks).

## License
MIT.
