from analytics.team_persistence import TeamPersistence


def main():

    print("Creating TeamPersistence...")
    persistence = TeamPersistence()

    # ---------------------------------------------------------
    # Test 1: Consistent Team A observations
    # ---------------------------------------------------------
    print("\nTest 1: Consistent Team A observations")

    for confidence in [0.90, 0.92, 0.95]:
        team = persistence.update(
            track_id=5,
            team="Team A",
            confidence=confidence,
        )

        print(
            f"Observation: Team A "
            f"confidence={confidence:.2f} "
            f"-> persisted={team}"
        )

    # ---------------------------------------------------------
    # Test 2: Temporary wrong Team B observation
    # ---------------------------------------------------------
    print("\nTest 2: Temporary Team B observation")

    team = persistence.update(
        track_id=5,
        team="Team B",
        confidence=0.60,
    )

    print(
        f"Observation: Team B "
        f"confidence=0.60 "
        f"-> persisted={team}"
    )

    # ---------------------------------------------------------
    # Test 3: Unknown should not overwrite Team A
    # ---------------------------------------------------------
    print("\nTest 3: Unknown observation")

    team = persistence.update(
        track_id=5,
        team="Unknown",
        confidence=0.0,
    )

    print(
        f"Observation: Unknown "
        f"-> persisted={team}"
    )

    # ---------------------------------------------------------
    # Display history
    # ---------------------------------------------------------
    print("\nTrack 5 history:")

    for observation in persistence.get_history(5):
        print(observation)

    # ---------------------------------------------------------
    # Test 4: Another track ID
    # ---------------------------------------------------------
    print("\nTest 4: Separate track")

    team = persistence.update(
        track_id=8,
        team="Team B",
        confidence=0.95,
    )

    print(
        f"Track 8 -> persisted={team}"
    )

    # ---------------------------------------------------------
    # Test 5: Invalid track ID
    # ---------------------------------------------------------
    print("\nTest 5: Invalid track ID")

    team = persistence.update(
        track_id=-1,
        team="Team A",
        confidence=0.95,
    )

    print(
        f"Track -1 -> persisted={team}"
    )

    # ---------------------------------------------------------
    # Test 6: History length
    # ---------------------------------------------------------
    print("\nTest 6: History length")

    for i in range(15):
        persistence.update(
            track_id=10,
            team="Team A",
            confidence=0.90,
        )

    history = persistence.get_history(10)

    print(
        f"Added 15 observations"
    )

    print(
        f"Stored observations: {len(history)}"
    )

    print(
        f"Maximum allowed: "
        f"{persistence.history_length}"
    )

    print("\nTeam persistence test completed.")


if __name__ == "__main__":
    main()