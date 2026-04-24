from flwr.serverapp.strategy import FedXgbBagging
import numpy as np
import xgboost as xgb
from sklearn.metrics import confusion_matrix
from flwr.serverapp.strategy.strategy_utils import aggregate_bagging

class AttentionWeightedFedXgbBagging(FedXgbBagging):
    """FedXgbBagging with per-round attention weighting.

    Overrides ``aggregate_train`` to intercept each client's tree update,
    evaluate it on a trusted server-side validation set, and tag it with
    an attention weight alpha_i = softmax(DR_i * (1 - FPR_i)).

    The standard bagging aggregation still runs (via ``super()``) so that
    clients receive the collaborative global model for the next round.
    The ``weighted_global_bag`` accumulates (full_model, alpha) entries
    across all rounds
    and is used for the final attention-weighted soft-voting inference.
    """

    def __init__(self, server_val_X, server_val_y, **kwargs):
        super().__init__(**kwargs)
        self.server_val_X = server_val_X
        self.server_val_y = server_val_y
        # Accumulates {round, full_model, alpha} across all rounds
        self.weighted_global_bag: list[dict] = []
        self.round_attention_log: list[dict] = []

    def aggregate_train(self, server_round, replies):
        """Intercept client trees, grade them, compute attention, then bag."""

        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        if not valid_replies:
            return None, None

        reply_contents = [msg.content for msg in valid_replies]
        array_record_key = next(iter(reply_contents[0].array_records.keys()))

        # Step 1: Evaluate each client's contribution on server holdout
            # In round 1, clients send the full model. In rounds 2+, clients
            # send only the last-N trees (residual corrections).  To evaluate
            # properly we reconstruct each client's complete model by combining
            # the current global model (self.current_bst) with the client's
            # new trees via aggregate_bagging.
        val_dmatrix = xgb.DMatrix(self.server_val_X, label=self.server_val_y)
        client_scores = []
        valid_full_models = []
        round_client_metrics = []

        for content in reply_contents:
            tree_bytes = content[array_record_key]["0"].numpy().tobytes()

            try:
                # Reconstruct this client's complete model:
                # global_model (from previous round) + client's new trees
                if self.current_bst is not None and self.current_bst != b"":
                    full_model_bytes = aggregate_bagging(
                        self.current_bst, tree_bytes
                    )
                else:
                    # Round 1 - client's trees ARE the full model
                    full_model_bytes = tree_bytes

                temp_bst = xgb.Booster()
                temp_bst.load_model(bytearray(full_model_bytes))
                y_probs = temp_bst.predict(val_dmatrix)
                y_pred = (y_probs > 0.5).astype(int)

                cm = confusion_matrix(
                    self.server_val_y, y_pred, labels=[0, 1]
                )
                tn, fp, fn, tp = cm.ravel()
                dr_i = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                fpr_i = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            except Exception as e:
                print(f"[AttentionFedXgb] WARNING: Dropping client update due to exception: {e}")
                continue

            score = dr_i * (1.0 - fpr_i)
            client_scores.append(score)
            valid_full_models.append(full_model_bytes)
            round_client_metrics.append({
                'dr': dr_i, 'fpr': fpr_i, 'score': score,
            })

        if not client_scores:
            print(
                f"[AttentionFedXgb] Round {server_round} - no valid client updates for attention; "
                "skipping attention log for this round"
            )
            # Continue standard FedXgbBagging aggregation for training progress.
            return super().aggregate_train(server_round, replies)

        # Step 2: Softmax attention weights
        scores_np = np.array(client_scores)
        exp_s = np.exp(scores_np - np.max(scores_np))  # numerical stability
        alpha = exp_s / exp_s.sum()

        entropy = -np.sum(alpha * np.log(alpha + 1e-12))

        print(f"[AttentionFedXgb] Round {server_round} - "
              f"alpha = {np.array2string(alpha, precision=4)}, "
              f"entropy = {entropy:.4f}")

        # We store the full model (global + client trees), not just the
        # tree fragments, so that inference uses complete boosters.
        for i, full_bytes in enumerate(valid_full_models):
            self.weighted_global_bag.append({
                'round': server_round,
                'full_model': full_bytes,
                'alpha': float(alpha[i]),
            })

        self.round_attention_log.append({
            'round': server_round,
            'alpha': alpha.tolist(),
            'entropy': float(entropy),
            'client_metrics': round_client_metrics,
        })

        # Standard bagging still happens for boosting
        return super().aggregate_train(server_round, replies)


def attention_weighted_inference(client_entries, X_data):
    """Weighted inference across full per-client models in margin space.

    Each entry contains a complete XGBoost model ('full_model') and weight
    ('alpha'). We aggregate raw margins, then apply sigmoid once:
    p(x) = sigmoid(sum(alpha_i * margin_i(x)) / sum(alpha_i)).
    """
    dmatrix = xgb.DMatrix(X_data)
    weighted_margin = np.zeros(len(X_data), dtype=np.float64)
    total_alpha = 0.0

    for entry in client_entries:
        bst = xgb.Booster()
        bst.load_model(bytearray(entry['full_model']))
        margins = bst.predict(dmatrix, output_margin=True)
        weighted_margin += entry['alpha'] * margins
        total_alpha += entry['alpha']

    if total_alpha > 0:
        weighted_margin /= total_alpha

    # Numerically stable sigmoid on aggregated margin
    weighted_margin = np.clip(weighted_margin, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-weighted_margin))