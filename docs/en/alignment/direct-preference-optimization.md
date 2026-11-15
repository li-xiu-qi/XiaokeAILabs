# Direct Preference Optimization

A preference pair contains a prompt and two answers, one rated more highly than the other. DPO trains a model to change their relative likelihood in the preferred direction compared with a frozen reference model. The reference appears in the objective; it does not guarantee safe or useful outputs.

For each answer, compute its conditional log probability under both the trainable and reference models. Subtract the rejected score from the preferred score for each model, then optimize a scaled logistic loss on the difference between those two margins. The method does not require a separate reward model or generation during each update. See the [original DPO paper](https://arxiv.org/abs/2305.18290) for the derivation and assumptions.

The [experiment](../../../experiments/dpo-preference-alignment/README.en.md) uses public model-scored preference pairs, a small chat model and a LoRA adapter. It evaluates the likelihood of fixed candidate answers on the dataset's separate test split. This is not a direct measurement of generated answer quality or real user preference. Training and test examples share the data source; near duplicates and pretraining overlap remain possible. A small, single-seed run does not support broad conclusions.

One GPU run trained on 256 pairs for 64 updates and evaluated 128 pairs. Before and after training, 71 pairs matched the dataset preference and no individual pair changed predicted direction. Roughly half of the answers were truncated, and summed token log probabilities are sensitive to answer length. This result does not establish better generated answers or preference judgment. The experiment README gives the environment, truncation counts and metrics.

## References

- Rafailov et al., [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290), 2023.
- Cui et al., [UltraFeedback: Boosting Language Models with High-quality Feedback](https://arxiv.org/abs/2310.01377), 2023.
