import unittest

from frameledger.annotations import AnnotationError, evaluate_annotations, validate_annotations
from frameledger.models import Candidate


class AnnotationTests(unittest.TestCase):
    def test_one_candidate_cannot_match_two_ground_truth_items(self):
        annotations = {
            "must_keep": [
                {"timestamp": "00:00:10", "tolerance_seconds": 3, "reason": "A"},
                {"timestamp": "00:00:12", "tolerance_seconds": 3, "reason": "B"},
            ],
            "nice_to_keep": [],
            "avoid_duplicates": [],
        }
        candidates = [Candidate(0, 11.0, 1.0, ["test"])]
        metrics = evaluate_annotations(annotations, candidates)
        self.assertEqual(metrics["must_keep"]["matched"], 1)
        self.assertEqual(metrics["must_keep"]["recall"], 0.5)

    def test_asymmetric_acceptable_window(self):
        annotations = {
            "must_keep": [
                {
                    "timestamp": "00:01:05",
                    "acceptable_start": "00:01:02",
                    "acceptable_end": "00:01:05",
                }
            ],
            "nice_to_keep": [],
            "avoid_duplicates": [],
        }

        inside = evaluate_annotations(
            annotations,
            [Candidate(0, 62.0, 1.0, ["test"])],
        )
        outside = evaluate_annotations(
            annotations,
            [Candidate(0, 65.5, 1.0, ["test"])],
        )

        self.assertEqual(inside["must_keep"]["matched"], 1)
        self.assertEqual(outside["must_keep"]["matched"], 0)

    def test_duplicate_window_is_scored(self):
        annotations = {
            "must_keep": [],
            "nice_to_keep": [],
            "avoid_duplicates": [
                {"start": 5, "end": 20, "max_selected_frames": 1, "reason": "static"}
            ],
        }
        candidates = [
            Candidate(0, 10.0, 1.0, ["test"]),
            Candidate(1, 15.0, 1.0, ["test"]),
        ]
        metrics = evaluate_annotations(annotations, candidates)
        self.assertEqual(metrics["avoid_duplicates"]["passed"], 0)
        self.assertFalse(metrics["avoid_duplicates"]["items"][0]["passed"])

    def test_full_segment_review_scores_unlisted_selected_frames(self):
        annotations = {
            "review_status": "human_reviewed",
            "review": {
                "coverage": "exhaustive",
                "scope": "full_segment",
                "selection_policy": "terminal_states_only",
                "unlisted_states": "reject",
            },
            "must_keep": [
                {"timestamp": "00:00:10", "tolerance_seconds": 1},
                {"timestamp": "00:00:30", "tolerance_seconds": 1},
            ],
            "nice_to_keep": [],
            "avoid_duplicates": [],
        }
        candidates = [
            Candidate(0, 10.0, 1.0, ["test"]),
            Candidate(1, 12.0, 1.0, ["test"]),
            Candidate(2, 30.0, 1.0, ["test"]),
        ]

        quality = evaluate_annotations(annotations, candidates)["selection_quality"]

        self.assertIsNotNone(quality)
        self.assertEqual(quality["accepted_matches"], 2)
        self.assertEqual(quality["unlisted_selected_frames"], 1)
        self.assertAlmostEqual(quality["precision"], 2 / 3)
        self.assertEqual(quality["recall"], 1.0)
        self.assertAlmostEqual(quality["f1"], 0.8)

    def test_unlisted_rejection_requires_human_review(self):
        annotations = {
            "review_status": "draft",
            "review": {
                "coverage": "exhaustive",
                "scope": "full_segment",
                "unlisted_states": "reject",
            },
            "must_keep": [],
            "nice_to_keep": [],
            "avoid_duplicates": [],
        }

        with self.assertRaisesRegex(AnnotationError, "requires review_status=human_reviewed"):
            validate_annotations(
                annotations,
                video_name="example.mp4",
                start_seconds=0.0,
                duration_seconds=60.0,
            )

    def test_schema_v4_exhaustive_review_requires_source_review_attestation(self):
        annotations = {
            "schema_version": 4,
            "review_status": "human_reviewed",
            "review": {
                "coverage": "exhaustive",
                "scope": "full_segment",
                "unlisted_states": "reject",
                "confirmation": {
                    "confirmed_by": "user",
                    "labels_complete": True,
                    "plateau_windows_confirmed": True,
                },
            },
            "must_keep": [],
            "nice_to_keep": [],
            "avoid_duplicates": [],
        }

        with self.assertRaisesRegex(AnnotationError, "review.source_review"):
            validate_annotations(
                annotations,
                video_name="example.mp4",
                start_seconds=0.0,
                duration_seconds=60.0,
            )

    def test_schema_v4_exhaustive_review_requires_user_confirmation(self):
        annotations = {
            "schema_version": 4,
            "review_status": "human_reviewed",
            "review": {
                "coverage": "exhaustive",
                "scope": "full_segment",
                "unlisted_states": "reject",
                "source_review": {
                    "basis": "source_video",
                    "full_segment_watched": True,
                    "algorithm_outputs_used_as_labels": False,
                },
            },
            "must_keep": [],
            "nice_to_keep": [],
            "avoid_duplicates": [],
        }

        with self.assertRaisesRegex(AnnotationError, "review.confirmation"):
            validate_annotations(
                annotations,
                video_name="example.mp4",
                start_seconds=0.0,
                duration_seconds=60.0,
            )

    def test_schema_v4_exhaustive_review_accepts_required_attestations(self):
        annotations = {
            "schema_version": 4,
            "review_status": "human_reviewed",
            "review": {
                "coverage": "exhaustive",
                "scope": "full_segment",
                "unlisted_states": "reject",
                "source_review": {
                    "basis": "source_video",
                    "full_segment_watched": True,
                    "algorithm_outputs_used_as_labels": False,
                },
                "confirmation": {
                    "confirmed_by": "user",
                    "labels_complete": True,
                    "plateau_windows_confirmed": True,
                },
            },
            "must_keep": [],
            "nice_to_keep": [],
            "avoid_duplicates": [],
        }

        validate_annotations(
            annotations,
            video_name="example.mp4",
            start_seconds=0.0,
            duration_seconds=60.0,
        )

    def test_strategy_candidate_review_only_scores_bound_strategy(self):
        fingerprint = "a" * 64
        annotations = {
            "review_status": "human_reviewed",
            "review": {
                "coverage": "strategy_candidate_set",
                "scope": "selected_candidates",
                "strategy": "hybrid",
                "strategy_file_sha256": fingerprint,
                "reviewed_candidate_count": 2,
                "unlisted_candidates": "reject",
            },
            "must_keep": [{"timestamp": "00:00:10", "tolerance_seconds": 0.5}],
            "nice_to_keep": [],
            "avoid_duplicates": [],
        }
        candidates = [
            Candidate(0, 10.0, 1.0, ["test"]),
            Candidate(1, 12.0, 1.0, ["test"]),
        ]

        hybrid = evaluate_annotations(
            annotations,
            candidates,
            strategy_name="hybrid",
            candidate_set_sha256=fingerprint,
        )
        uniform = evaluate_annotations(
            annotations,
            candidates,
            strategy_name="uniform",
            candidate_set_sha256="b" * 64,
        )

        self.assertEqual(hybrid["selection_quality"]["review_basis"], "strategy_candidate_set")
        self.assertEqual(hybrid["selection_quality"]["precision"], 0.5)
        self.assertIsNone(hybrid["selection_quality"]["recall"])
        self.assertFalse(hybrid["selection_quality"]["source_segment_recall_assessable"])
        self.assertIsNone(uniform["selection_quality"])

    def test_strategy_candidate_review_rejects_candidate_set_drift(self):
        annotations = {
            "review_status": "human_reviewed",
            "review": {
                "coverage": "strategy_candidate_set",
                "scope": "selected_candidates",
                "strategy": "hybrid",
                "strategy_file_sha256": "a" * 64,
                "unlisted_candidates": "reject",
            },
            "must_keep": [],
            "nice_to_keep": [],
            "avoid_duplicates": [],
        }

        with self.assertRaisesRegex(AnnotationError, "strategy_file_sha256"):
            evaluate_annotations(
                annotations,
                [],
                strategy_name="hybrid",
                candidate_set_sha256="b" * 64,
            )

    def test_source_fingerprint_must_match_when_declared(self):
        annotations = {
            "source_sha256": "wrong",
            "must_keep": [],
            "nice_to_keep": [],
            "avoid_duplicates": [],
        }

        with self.assertRaisesRegex(AnnotationError, "source_sha256"):
            validate_annotations(
                annotations,
                video_name="example.mp4",
                start_seconds=0.0,
                duration_seconds=60.0,
                source_sha256="expected",
            )


if __name__ == "__main__":
    unittest.main()
