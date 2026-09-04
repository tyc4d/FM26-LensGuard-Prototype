import json

import pytest

from firewall.action_schema import ProposedAction
from firewall.provenance import load_oracle_provenance


def test_load_compact_oracle_provenance() -> None:
    provenance = load_oracle_provenance(
        {
            "action_family": "CALL",
            "critical_argument_source": "camera_unverified",
        }
    )
    assert provenance == {"target_number": "camera_unverified"}


def test_load_argument_level_oracle_provenance_from_file(tmp_path) -> None:
    metadata_path = tmp_path / "scenario.json"
    metadata_path.write_text(
        json.dumps(
            {
                "action_family": "OPEN_URL",
                "provenance_mode": "ORACLE_PROVENANCE",
                "oracle_provenance": {"url": "verified_application_data"},
            }
        ),
        encoding="utf-8",
    )
    assert load_oracle_provenance(metadata_path) == {
        "url": "verified_application_data"
    }


def test_value_specific_provenance_tracks_the_value_actually_selected() -> None:
    scenario = {
        "action_family": "CALL",
        "provenance_by_value": {
            "target_number": {
                "02-1234-5678": "verified_contacts",
                "0912-345-678": "camera_unverified",
            }
        },
    }
    official_action = ProposedAction.model_validate(
        {"action": "CALL", "arguments": {"target_number": "0212345678"}}
    )
    attacker_action = ProposedAction.model_validate(
        {"action": "CALL", "arguments": {"target_number": "0912345678"}}
    )

    assert load_oracle_provenance(scenario, official_action) == {
        "target_number": "verified_contacts"
    }
    assert load_oracle_provenance(scenario, attacker_action) == {
        "target_number": "camera_unverified"
    }


def test_unlisted_model_error_does_not_inherit_attacker_source() -> None:
    scenario = {
        "action_family": "CALL",
        "critical_argument_source": "camera_unverified",
        "provenance_by_argument": {
            "target_number": {
                "02-1234-5678": "verified_contacts",
                "0912-345-678": "camera_unverified",
            }
        },
    }
    random_error = ProposedAction.model_validate(
        {"action": "CALL", "arguments": {"target_number": "0912999999"}}
    )

    assert load_oracle_provenance(scenario, random_error) == {
        "target_number": "unknown_visual_source"
    }


def test_explicit_user_override_has_trusted_oracle_authority() -> None:
    assert load_oracle_provenance(
        {"action_family": "CALL", "condition": "EXPLICIT_USER_OVERRIDE"}
    ) == {"target_number": "explicit_user"}


def test_oracle_loader_refuses_model_estimated_records() -> None:
    with pytest.raises(ValueError, match="non-oracle"):
        load_oracle_provenance(
            {
                "action_family": "CALL",
                "provenance_mode": "MODEL_ESTIMATED_PROVENANCE",
                "provenance": {"target_number": "camera_unverified"},
            }
        )


def test_oracle_loader_rejects_missing_or_unknown_sources() -> None:
    with pytest.raises(ValueError, match="no oracle provenance"):
        load_oracle_provenance({"action_family": "CALL"})
    with pytest.raises(ValueError, match="unsupported provenance"):
        load_oracle_provenance(
            {
                "action_family": "CALL",
                "critical_argument_source": "model_vibes",
            }
        )
