"""The noise heuristic decides which chunks retrieval is allowed to return.

It was tuned against real chunk ids from this manual, and its false positives
were expensive: v1 flagged the coolant-capacity table, which is an actual answer.
These tests lock in both directions - junk caught, specs spared.
"""

from app.ingestion.quality import DOT_RATIO_LIMIT, dot_ratio, is_index_noise

# Real shapes from data/hilux_manual.pdf, trimmed.
TOC_LINE = "Side turn signal lights . . . . . . . . . . . . . . . . . . . . . P. 304"
SPECS_LINE = "Engine coolant capacity 1GR-FE 8.4 L (8.9 qt., 7.4 Imp.qt.) 2TR-FE 7.0 L"
PROSE = ("Check the engine oil level with the dipstick while the vehicle is "
         "parked on level ground and the engine is cold.")


def test_table_of_contents_leader_lines_are_noise():
    assert is_index_noise(TOC_LINE)


def test_specification_tables_are_not_noise():
    # Digit-heavy and full of periods from abbreviations, but it answers a real
    # question. An earlier dots+digits filter threw exactly this away.
    assert not is_index_noise(SPECS_LINE)


def test_prose_is_not_noise():
    assert not is_index_noise(PROSE)


def test_both_conditions_are_required():
    # High dot ratio alone is not enough - a dotted-leader run must be present -
    # and a leader run alone is not enough either.
    # Numbered fragments: one dot in three characters, but never two in a row,
    # so the leader regex finds nothing.
    dotty_no_leader = "1. 2. 3. 4. 5. 6. 7. 8."
    assert dot_ratio(dotty_no_leader) > DOT_RATIO_LIMIT
    assert not is_index_noise(dotty_no_leader)

    leader_in_prose = "Refer to the section on tires . . . then check the pressure " * 4
    assert dot_ratio(leader_in_prose) < DOT_RATIO_LIMIT
    assert not is_index_noise(leader_in_prose)


def test_empty_text_does_not_divide_by_zero():
    assert dot_ratio("") == 0.0
    assert not is_index_noise("")
