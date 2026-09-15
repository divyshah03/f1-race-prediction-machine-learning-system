from f1_predictor.data.loader import average_sector_times
from f1_predictor.features.engineer import build_feature_table, numeric_feature_columns


def test_build_feature_table_maps_target_from_historical_laps(sample_race_config, sample_historical_laps):
    sector_times = average_sector_times(sample_historical_laps)

    table = build_feature_table(
        sample_race_config, sector_times, sample_historical_laps, rain_probability=0.0, temperature=20.0
    )

    assert set(table["Driver"]) == {"VER", "NOR", "LEC"}
    indexed = table.set_index("Driver")
    assert indexed.loc["VER", "LapTime (s)"] == 90.5
    assert indexed.loc["NOR", "LapTime (s)"] == 92.5
    assert (table["RainProbability"] == 0.0).all()
    assert (table["Temperature"] == 20.0).all()


def test_wet_performance_adjusts_qualifying_time_only_when_raining(sample_race_config, sample_historical_laps):
    sample_race_config.wet_performance_factor = {"VER": 0.9, "NOR": 0.8, "LEC": 0.95}
    sector_times = average_sector_times(sample_historical_laps)

    dry = build_feature_table(
        sample_race_config, sector_times, sample_historical_laps, rain_probability=0.0, temperature=20.0
    )
    wet = build_feature_table(
        sample_race_config, sector_times, sample_historical_laps, rain_probability=0.9, temperature=20.0
    )

    assert dry.set_index("Driver").loc["VER", "QualifyingTime (s)"] == 80.0
    assert wet.set_index("Driver").loc["VER", "QualifyingTime (s)"] == 80.0 * 0.9


def test_qualifying_time_square_transform(sample_race_config, sample_historical_laps):
    sample_race_config.qualifying_time_transform = "square"
    sector_times = average_sector_times(sample_historical_laps)

    table = build_feature_table(sample_race_config, sector_times, sample_historical_laps, 0.0, 20.0)

    assert table.set_index("Driver").loc["VER", "QualifyingTime (s)"] == 80.0**2


def test_numeric_feature_columns_excludes_identifiers(sample_race_config, sample_historical_laps):
    sector_times = average_sector_times(sample_historical_laps)
    table = build_feature_table(sample_race_config, sector_times, sample_historical_laps, 0.0, 20.0)

    columns = numeric_feature_columns(table)

    assert "Driver" not in columns
    assert "circuit" not in columns
    assert "LapTime (s)" not in columns
    assert "QualifyingTime (s)" in columns
