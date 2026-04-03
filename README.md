# Entity-Wan-Move: Entity-Centric Motion Control for Video Generation

<div align="center">

[![Paper](https://img.shields.io/badge/ArXiv-Paper-brown)](https://arxiv.org/abs/2512.08765)
[![Our-Research](https://img.shields.io/badge/Research-Entity--Centric-blueviolet)](https://github.com/yeahyyyyx/Entity-Wan-Move)
[![Model](https://img.shields.io/badge/Model-Wan--Move--14B-yellow)](https://huggingface.co/Ruihang/Wan-Move-14B-480P)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE.txt)

</div>

## 💡 TLDR: Beyond Pixels, Towards Entities

**Entity-Wan-Move** is an advanced research fork of [Wan-Move](https://github.com/ali-vilab/Wan-Move). While the original Wan-Move achieves SOTA motion control via latent trajectory guidance, it lacks explicit identity modeling, often leading to **identity swapping** or **drift** in complex multi-object scenes.

**Our Core Innovation:** We extend the Diffusion Transformer (DiT) by introducing persistent **Entity Tokens**. Instead of just guiding "where pixels move," we explicitly model "who is moving" by binding trajectories directly to decoupleable entity states.

---

## 🚨 Motivation: Why Entity-Centric?

Current trajectory-based models (including the baseline Wan-Move) treat motion as a spatial bias. This leads to fundamental failures:
- ❌ **Identity Swapping:** When two trajectories cross, the model often swaps their visual identities.
- ❌ **Vanishing under Occlusion:** When an object is occluded, the spatial guidance loses its "anchor," causing the object to disappear or morph.
- ❌ **Coupled Control:** Hard to steer one object without affecting the appearance of its neighbors.

**We argue that motion control without identity modeling is fundamentally incomplete.**

---

## 🧠 Method Overview (Proposed)

We transform the Video DiT from a pixel-pusher into an **Entity-State Machine**:

### 1. Entity Token Injection
We introduce $K$ learnable **Entity Slots** into the Transformer sequence. 
- Input sequence: $[E_1, ..., E_K, Z_{video}]$
- These tokens serve as "Identity Anchors" that persist across the entire denoising process.

### 2. Trajectory-to-Entity Binding
Unlike spatial injection, we encode motion trajectories $\tau_i$ and inject them exclusively into their corresponding Entity Tokens $E_i$.
- $E_i \leftarrow E_i + \text{MLP}(\tau_i)$

### 3. Entity–Latent Interaction (ELI)
Through modified Self-Attention or Cross-Attention, video patches learn to query their respective Entity Tokens to maintain identity consistency.
- $Z \leftarrow \text{Attn}(Q=Z, K=E, V=E)$

### 4. Dynamic State Update
Entity tokens are treated as **Stateful Variables** that evolve over time (via GRU or Attention updates), capturing the temporal dynamics of each specific object.

---

## 🧪 Research Roadmap & TODO

- [ ] **Phase 1: Environment & Baseline**
    - [x] Establish AutoDL environment and GitHub sync.
    - [ ] Successfully reproduce baseline Wan-Move 14B inference.
- [ ] **Phase 2: Architecture Surgery**
    - [ ] Implement Entity Token Injection in `WanModel`.
    - [ ] Modify `WanAttention` for Entity-Latent Interaction.
    - [ ] Develop the Trajectory-to-Entity Binding module.
- [ ] **Phase 3: Evaluation & Benchmarking**
    - [ ] Cross-trajectory consistency test (Identity Preservation).
    - [ ] Occlusion recovery analysis.
    - [ ] Comparative study against original Wan-Move on MoveBench.

---

## 🚀 Quickstart (Research Environment)

### Installation
```bash
git clone [https://github.com/yeahyyyyx/Entity-Wan-Move.git](https://github.com/yeahyyyyx/Entity-Wan-Move.git)
cd Entity-Wan-Move
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
