"""Regression tests for Chen and Cherry smoothing method 5.

method5 used to average the precision fractions themselves. Chen and Cherry
average the matched n-gram counts and keep each order's denominator.
"""

from nltk.translate.bleu_score import SmoothingFunction, modified_precision, sentence_bleu


def test_method5_averages_match_counts_not_precisions():
    references = [["a", "b", "c", "d"]]
    hypothesis = ["a", "b", "x", "y"]
    precisions = [modified_precision(references, hypothesis, n) for n in range(1, 5)]
    assert [p.numerator for p in precisions] == [2, 1, 0, 0]
    assert [p.denominator for p in precisions] == [4, 3, 2, 1]

    smoothed = SmoothingFunction().method5(list(precisions), references, hypothesis)

    # counts [2, 1, 0, 0], virtual order-0 count = 2 + 1
    # n=1: (3 + 2 + 1) / 3 / 4 = 0.5
    # n=2: (2 + 1 + 0) / 3 / 3 = 1/3
    # n=3: (1 + 0 + 0) / 3 / 2 = 1/6
    # n=4: (1/3 + 0 + 0) / 3 / 1 = 1/9
    assert smoothed[0] == 0.5
    assert abs(smoothed[1] - 1 / 3) < 1e-12
    assert abs(smoothed[2] - 1 / 6) < 1e-12
    assert abs(smoothed[3] - 1 / 9) < 1e-12

    # Averaging the fractions themselves (the old behavior) yields 7/9, not 1/2.
    old_first = ((precisions[0] + 1) + precisions[0] + precisions[1]) / 3
    assert abs(float(old_first) - 7 / 9) < 1e-12
    assert smoothed[0] != float(old_first)


def test_method5_extra_order_follows_weight_length():
    references = [["a", "b", "c", "d", "e"]]
    hypothesis = ["a", "b", "c", "x"]
    precisions = [modified_precision(references, hypothesis, n) for n in range(1, 3)]
    trigram = modified_precision(references, hypothesis, 3)
    fivegram = modified_precision(references, hypothesis, 5)
    assert trigram.numerator != fivegram.numerator

    smoothed = SmoothingFunction().method5(list(precisions), references, hypothesis)
    # Last order's n+1 count is the trigram match, not a hardcoded 5-gram (0).
    # counts feed forward: start at 3+1, then unigram, then bigram.
    after_unigram = (4 + precisions[0].numerator + precisions[1].numerator) / 3
    expected_bigram = (after_unigram + precisions[1].numerator + trigram.numerator) / 3
    expected_bigram /= precisions[1].denominator
    assert abs(smoothed[1] - expected_bigram) < 1e-12
    wrong = (after_unigram + precisions[1].numerator + fivegram.numerator) / 3
    wrong /= precisions[1].denominator
    assert abs(smoothed[1] - wrong) > 1e-9

    score = sentence_bleu(
        references,
        hypothesis,
        weights=(0.5, 0.5),
        smoothing_function=SmoothingFunction().method5,
    )
    assert score > 0
