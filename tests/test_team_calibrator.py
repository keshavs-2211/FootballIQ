from analytics.team import TeamCalibrator


def main():

    print("Creating TeamCalibrator...")

    calibrator = TeamCalibrator(
        minimum_observations=20
    )

    # Simulate observations from our current match.
    #
    # The actual jersey colors were:
    # Green = one team
    # White = another team
    # Yellow = goalkeeper
    #
    # We intentionally add Yellow as an observation to verify
    # that the calibrator itself does not assume specific colors.

    for _ in range(40):
        calibrator.observe("Green", 0.90)

    for _ in range(30):
        calibrator.observe("White", 0.95)

    for _ in range(5):
        calibrator.observe("Yellow", 0.90)

    print("\nObserved colors:")
    print(calibrator.color_counts)

    print("\nCalibrating teams...")

    mapping = calibrator.calibrate()

    if mapping is None:
        raise RuntimeError(
            "Team calibration failed."
        )

    print("\nTeam mapping:")
    print(mapping)

    print("\nTesting team assignment:")

    print(
        "Green ->",
        calibrator.get_team("Green", 0.90)
    )

    print(
        "White ->",
        calibrator.get_team("White", 0.95)
    )

    print(
        "Yellow ->",
        calibrator.get_team("Yellow", 0.90)
    )

    print(
        "Unknown ->",
        calibrator.get_team("Unknown", 0.90)
    )

    print("\nTeam calibrator test completed.")


if __name__ == "__main__":
    main()