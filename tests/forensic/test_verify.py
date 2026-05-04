"""Tests for the General Impostors (GI) authorship-verification method."""

from __future__ import annotations

import numpy as np
import pytest

from bitig.features import FeatureMatrix
from bitig.forensic import GeneralImpostors


def _make_fm(X: np.ndarray, doc_ids: list[str], feature_names: list[str]) -> FeatureMatrix:  # noqa: N803
    return FeatureMatrix(
        X=np.asarray(X, dtype=float),
        document_ids=doc_ids,
        feature_names=feature_names,
        feature_type="synthetic",
    )


def _synthetic_universe(
    seed: int = 0,
    n_known: int = 8,
    n_impostors: int = 20,
    n_features: int = 60,
) -> tuple[FeatureMatrix, FeatureMatrix, FeatureMatrix, FeatureMatrix]:
    """Build Q, K (same-author), Q2 (different-author), and an impostor pool.

    The candidate author has an idiosyncratic preference profile over a subset of features;
    impostors draw from a different profile. A same-author Q is drawn from the candidate's
    profile with noise; a different-author Q2 is drawn from the impostor profile.
    """
    rng = np.random.default_rng(seed)
    feature_names = [f"f{i}" for i in range(n_features)]

    candidate_profile = rng.dirichlet(alpha=np.ones(n_features) * 0.5)
    impostor_profile = rng.dirichlet(alpha=np.ones(n_features) * 0.5)

    def _sample(profile: np.ndarray, n: int) -> np.ndarray:
        rows = rng.multinomial(1000, profile, size=n).astype(float)
        rows /= rows.sum(axis=1, keepdims=True)
        return rows

    known_X = _sample(candidate_profile, n_known)
    impostor_X = np.vstack(
        [
            _sample(impostor_profile, n_impostors - 1),
            _sample(candidate_profile, 1) * 0.9 + 0.05 * _sample(impostor_profile, 1),
        ]
    )
    q_same = _sample(candidate_profile, 1)
    q_diff = _sample(impostor_profile, 1)

    return (
        _make_fm(q_same, ["Q_same"], feature_names),
        _make_fm(known_X, [f"K{i}" for i in range(n_known)], feature_names),
        _make_fm(q_diff, ["Q_diff"], feature_names),
        _make_fm(impostor_X, [f"I{i}" for i in range(n_impostors)], feature_names),
    )


class TestGeneralImpostorsDiscrimination:
    """GI should score same-author pairs higher than different-author pairs."""

    def test_same_author_scores_higher_than_different_author(self) -> None:
        q_same, known, q_diff, impostors = _synthetic_universe(seed=0)
        gi = GeneralImpostors(n_iterations=100, seed=42)
        score_same = gi.verify(questioned=q_same, known=known, impostors=impostors).values["score"]
        score_diff = gi.verify(questioned=q_diff, known=known, impostors=impostors).values["score"]
        assert score_same > score_diff
        # The discrimination signal should be large on this synthetic corpus.
        assert score_same - score_diff > 0.3

    def test_score_is_bounded_0_1(self) -> None:
        q_same, known, _, impostors = _synthetic_universe(seed=1)
        gi = GeneralImpostors(n_iterations=50, seed=7)
        result = gi.verify(questioned=q_same, known=known, impostors=impostors)
        assert 0.0 <= result.values["score"] <= 1.0


class TestGeneralImpostorsReproducibility:
    def test_same_seed_produces_identical_score(self) -> None:
        q_same, known, _, impostors = _synthetic_universe(seed=2)
        gi_a = GeneralImpostors(n_iterations=50, seed=123)
        gi_b = GeneralImpostors(n_iterations=50, seed=123)
        score_a = gi_a.verify(questioned=q_same, known=known, impostors=impostors).values["score"]
        score_b = gi_b.verify(questioned=q_same, known=known, impostors=impostors).values["score"]
        assert score_a == score_b

    def test_different_seeds_produce_different_scores(self) -> None:
        q_same, known, _, impostors = _synthetic_universe(seed=3)
        scores = {
            seed: GeneralImpostors(n_iterations=50, seed=seed)
            .verify(questioned=q_same, known=known, impostors=impostors)
            .values["score"]
            for seed in (1, 17, 999)
        }
        # At least two scores should differ — identical across all seeds would indicate
        # the random sampling is being ignored.
        assert len(set(scores.values())) > 1


class TestGeneralImpostorsResultContract:
    def test_result_carries_full_provenance_params(self) -> None:
        q, known, _, impostors = _synthetic_universe(seed=4)
        gi = GeneralImpostors(n_iterations=30, feature_subsample_rate=0.4, seed=5)
        result = gi.verify(questioned=q, known=known, impostors=impostors)
        assert result.method_name == "general_impostors"
        for key in (
            "n_iterations",
            "feature_subsample_rate",
            "feature_sample_size",
            "impostor_sample_size",
            "similarity",
            "aggregate",
            "seed",
            "n_features",
            "n_known",
            "n_impostors",
        ):
            assert key in result.params, f"missing param {key!r}"
        assert result.values["questioned_id"] == "Q_same"
        assert result.values["wins"] == int(result.values["wins"])
        assert result.values["wins"] <= result.values["total_iterations"]


class TestGeneralImpostorsInputValidation:
    def test_questioned_must_have_exactly_one_row(self) -> None:
        _, known, _, impostors = _synthetic_universe(seed=0)
        gi = GeneralImpostors(n_iterations=10, seed=0)
        with pytest.raises(ValueError, match="exactly 1 row"):
            gi.verify(questioned=known, known=known, impostors=impostors)

    def test_impostors_pool_must_be_at_least_two(self) -> None:
        q, known, _, impostors = _synthetic_universe(seed=0)
        tiny_pool = _make_fm(impostors.X[:1], [impostors.document_ids[0]], impostors.feature_names)
        gi = GeneralImpostors(n_iterations=10, seed=0)
        with pytest.raises(ValueError, match="at least 2"):
            gi.verify(questioned=q, known=known, impostors=tiny_pool)

    def test_feature_space_mismatch_raises(self) -> None:
        q, known, _, impostors = _synthetic_universe(seed=0)
        # Re-label the impostor features so names don't match.
        mismatched = _make_fm(
            impostors.X,
            impostors.document_ids,
            [f"z{i}" for i in range(impostors.X.shape[1])],
        )
        gi = GeneralImpostors(n_iterations=10, seed=0)
        with pytest.raises(ValueError, match="feature_names"):
            gi.verify(questioned=q, known=known, impostors=mismatched)

    def test_minmax_similarity_rejects_negative_features(self) -> None:
        q, known, _, impostors = _synthetic_universe(seed=0)
        # Inject a negative value into the candidate's features.
        bad_known_X = known.X.copy()
        bad_known_X[0, 0] = -0.01
        bad_known = _make_fm(bad_known_X, known.document_ids, known.feature_names)
        gi = GeneralImpostors(n_iterations=10, seed=0, similarity="minmax")
        with pytest.raises(ValueError, match="non-negative"):
            gi.verify(questioned=q, known=bad_known, impostors=impostors)

    def test_invalid_constructor_args_raise(self) -> None:
        with pytest.raises(ValueError, match="n_iterations"):
            GeneralImpostors(n_iterations=0)
        with pytest.raises(ValueError, match="feature_subsample_rate"):
            GeneralImpostors(feature_subsample_rate=0.0)
        with pytest.raises(ValueError, match="feature_subsample_rate"):
            GeneralImpostors(feature_subsample_rate=1.5)
        with pytest.raises(ValueError, match="impostor_sample_size"):
            GeneralImpostors(impostor_sample_size=0)
        with pytest.raises(ValueError, match="similarity"):
            GeneralImpostors(similarity="bogus")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="aggregate"):
            GeneralImpostors(aggregate="bogus")  # type: ignore[arg-type]


class TestGeneralImpostorsAggregateAndSimilarity:
    def test_centroid_and_nearest_both_discriminate(self) -> None:
        q_same, known, q_diff, impostors = _synthetic_universe(seed=10)
        for aggregate in ("centroid", "nearest"):
            gi = GeneralImpostors(
                n_iterations=80,
                aggregate=aggregate,  # type: ignore[arg-type]
                seed=42,
            )
            score_same = gi.verify(questioned=q_same, known=known, impostors=impostors).values[
                "score"
            ]
            score_diff = gi.verify(questioned=q_diff, known=known, impostors=impostors).values[
                "score"
            ]
            assert score_same > score_diff, f"aggregate={aggregate!r} failed to discriminate"

    def test_minmax_similarity_discriminates_on_nonnegative_features(self) -> None:
        q_same, known, q_diff, impostors = _synthetic_universe(seed=11)
        gi = GeneralImpostors(n_iterations=80, similarity="minmax", seed=42)
        score_same = gi.verify(questioned=q_same, known=known, impostors=impostors).values["score"]
        score_diff = gi.verify(questioned=q_diff, known=known, impostors=impostors).values["score"]
        assert score_same > score_diff
