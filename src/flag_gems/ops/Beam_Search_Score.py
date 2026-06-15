import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def beam_search_score_kernel(log_probs, beam_scores):
    # Beam Search Score: Add cumulative log probability with new token log probability
    # This is the core operation in beam search: new_score = old_score + log_prob(token)
    return log_probs + beam_scores


def Beam_Search_Score(log_probs, beam_scores):
    """Compute Beam Search scores by adding cumulative scores with log probabilities.

    In Beam Search, each candidate's score is computed by adding the cumulative
    log probability with the log probability of the new token.

    Args:
        log_probs: Log probabilities for new tokens, shape [batch_size, vocab_size]
        beam_scores: Cumulative beam scores, shape [batch_size] or [batch_size, 1]

    Returns:
        Updated beam scores with shape [batch_size, vocab_size]
    """
    logger.debug("GEMS Beam_Search_Score")

    # Handle broadcasting: beam_scores needs to be broadcast to match log_probs
    log_probs_ndim = log_probs.dim()
    beam_scores_ndim = beam_scores.dim()

    # If beam_scores has fewer dimensions, we need to expand it
    if beam_scores_ndim < log_probs_ndim:
        # Get the broadcast shape
        # beam_scores shape should be [batch_size] and we need to add a dimension
        # at the end for broadcasting with [batch_size, vocab_size]
        if beam_scores_ndim == 1:
            # [batch_size] -> [batch_size, 1]
            beam_scores = beam_scores.unsqueeze(-1)

    return beam_search_score_kernel(log_probs, beam_scores)


def Beam_Search_Score_(log_probs, beam_scores):
    """In-place version of Beam_Search_Score.

    Computes Beam Search scores in-place, modifying log_probs directly.
    """
    logger.debug("GEMS Beam_Search_Score_")

    log_probs_ndim = log_probs.dim()
    beam_scores_ndim = beam_scores.dim()

    if beam_scores_ndim < log_probs_ndim:
        if beam_scores_ndim == 1:
            beam_scores = beam_scores.unsqueeze(-1)

    return beam_search_score_kernel(log_probs, beam_scores, out0=log_probs)