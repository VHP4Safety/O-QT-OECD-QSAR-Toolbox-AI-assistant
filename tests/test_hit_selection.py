from oqt_assistant.utils.hit_selection import rank_hits_by_quality


def test_rank_hits_prioritises_exact_cas_match():
    hits = [
        {
            "ChemId": "match",
            "Name": "Unrelated substance",
            "Cas": "50000",
            "SubstanceType": "MonoConstituent",
            "Smiles": "C",
        },
        {
            "ChemId": "name_match",
            "Name": "50000-substance",
            "Cas": "12345",
            "SubstanceType": "MonoConstituent",
            "Smiles": "C",
        },
    ]

    ranked = rank_hits_by_quality("50-00-0", hits)

    assert ranked[0]["ChemId"] == "match"
