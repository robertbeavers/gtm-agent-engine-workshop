import importlib
import json
import os
import unittest
from copy import deepcopy
from unittest.mock import Mock, patch

from gtm_agent import data_service


class ProspectUpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
            cls.tools = importlib.import_module("gtm_agent.gtm_agent")

    def setUp(self):
        self.records = patch.object(data_service, "PROSPECTS", {
            "test-prospect": {
                "tech_stack": ["PostgreSQL"],
                "engagement_history": [],
                "account_details": [],
                "annual_revenue": 1000000,
            }
        })
        self.records.start()
        self.addCleanup(self.records.stop)
        self.cache = patch.object(data_service, "_PROFILES", {})
        self.cache.start()
        self.addCleanup(self.cache.stop)
        self.tracing = patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
        self.tracing.start()
        self.addCleanup(self.tracing.stop)

    def test_update_rebuilds_cached_profile_and_scores_updated_stack(self):
        before = self.tools.build_prospect_profile.invoke({"prospect_id": "test-prospect"})
        self.assertNotIn("Snowflake", before["prospect_profile"]["tech_stack"])
        updated = self.tools.update_prospect_info.invoke({
            "prospect_id": "test-prospect", "technology": "Snowflake"
        })
        self.assertTrue(updated["updated"])
        self.assertIn("Snowflake", data_service.fetch_tech_stack("test-prospect"))
        profile = self.tools.build_prospect_profile.invoke({"prospect_id": "test-prospect"})
        self.assertIn("Snowflake", profile["prospect_profile"]["tech_stack"])
        self.assertIn("Snowflake", data_service.get_profile_from_db("test-prospect")["prospect_profile"]["tech_stack"])
        self.assert_scoring_uses_stack(profile["prospect_profile"])

    def test_stale_fetch_does_not_replace_supplied_stack(self):
        profile = {"prospect_id": "test-prospect", "tech_stack": ["Snowflake"]}
        original = deepcopy(profile)
        with patch.object(data_service, "fetch_tech_stack", return_value=["PostgreSQL"]):
            self.assert_scoring_uses_stack(profile)
        self.assertEqual(profile, original)

    def assert_scoring_uses_stack(self, profile):
        offering = {
            "required_tech_stack": ["Snowflake"],
            "min_annual_revenue": 1000,
            "description": "Analytics",
        }
        scorer = Mock()
        scorer.invoke.return_value.model_dump.return_value = {"score": 90, "justification": "Has Snowflake"}
        with patch.object(self.tools, "_scoring_llm", scorer):
            result = self.tools.score_prospect.invoke({"prospect_profile": profile, "offering": offering})
        supplied = scorer.invoke.call_args.args[0][1]["content"].split("Prospect profile:\n", 1)[1]
        self.assertEqual(json.loads(supplied), profile)
        self.assertIn("Snowflake", json.loads(supplied)["tech_stack"])
        self.assertEqual(result, {"score": 90, "justification": "Has Snowflake"})

    def test_repeated_update_does_not_duplicate_technology(self):
        for _ in range(2):
            data_service.update_prospect_info("test-prospect", "Snowflake")
        self.assertEqual(data_service.fetch_tech_stack("test-prospect").count("Snowflake"), 1)

    def test_unknown_prospect_is_not_reported_as_updated(self):
        self.assertEqual(data_service.update_prospect_info("missing", "Snowflake"), {
            "updated": False, "found": False
        })


if __name__ == "__main__":
    unittest.main()
