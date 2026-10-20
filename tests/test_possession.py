
from analytics.possession import PossessionTracker


def create_player(track_id, bbox, team):
    return {
        "track_id": track_id,
        "bbox": bbox,
        "team": team,
    }


def main():
    print("Creating PossessionTracker...")
    tracker = PossessionTracker()

    # Test 1: Ball unavailable
    print("\nTest 1: Ball unavailable")

    ball_state = {
        "ball_x": None,
        "ball_y": None,
        "status": "unavailable",
    }

    players = [
        create_player(1, [90, 90, 110, 150], "Team A")
    ]

    result = tracker.update(ball_state, players)
    print(result)

    # Test 2: Ball near player
    print("\nTest 2: Ball near player")

    ball_state = {
        "ball_x": 100,
        "ball_y": 150,
        "status": "detected",
    }

    players = [
        create_player(1, [90, 90, 110, 150], "Team A"),
        create_player(2, [400, 400, 430, 470], "Team B"),
    ]

    for frame in range(1, 4):
        result = tracker.update(ball_state, players)
        print(
            f"Frame {frame}: "
            f"possession_player={result['possession_player_id']} "
            f"team={result['possession_team']}"
        )

    # Test 3: Verify possession after three frames
    print("\nTest 3: Possession confirmation")

    if tracker.current_player_id == 1:
        print("PASS: Player 1 has possession.")
    else:
        print("FAIL: Player 1 did not receive possession.")

    # Test 4: Same player retains possession
    print("\nTest 4: Same player retains possession")

    for frame in range(4, 7):
        result = tracker.update(ball_state, players)
        print(
            f"Frame {frame}: "
            f"possession_player={result['possession_player_id']} "
            f"team={result['possession_team']}"
        )

    # Test 5: Possession switch
    print("\nTest 5: Possession switch")

    ball_state = {
        "ball_x": 415,
        "ball_y": 470,
        "status": "detected",
    }

    for frame in range(7, 10):
        result = tracker.update(ball_state, players)
        print(
            f"Frame {frame}: "
            f"possession_player={result['possession_player_id']} "
            f"team={result['possession_team']}"
        )

    if tracker.current_player_id == 2:
        print("PASS: Possession switched to Player 2.")
    else:
        print("FAIL: Possession did not switch correctly.")

    # Test 6: Unknown team
    print("\nTest 6: Unknown team")

    tracker.reset()

    ball_state = {
        "ball_x": 100,
        "ball_y": 150,
        "status": "detected",
    }

    players = [
        create_player(5, [90, 90, 110, 150], "Unknown")
    ]

    for frame in range(1, 4):
        result = tracker.update(ball_state, players)
        print(
            f"Frame {frame}: "
            f"possession_player={result['possession_player_id']} "
            f"team={result['possession_team']}"
        )

    # Test 7: has_ball flag
    print("\nTest 7: has_ball flag")

    if result["has_ball"].get(5) is True:
        print("PASS: has_ball=True for Player 5.")
    else:
        print("FAIL: has_ball flag is incorrect.")

    # Test 8: Distance calculation
    print("\nTest 8: Distance calculation")

    distance = tracker._distance((100, 100), (103, 104))
    print(f"Distance: {distance:.2f}")

    if abs(distance - 5.0) < 0.001:
        print("PASS: Distance calculation is correct.")
    else:
        print("FAIL: Distance calculation is incorrect.")

    print("\n==============================")
    print("Possession tracker test completed.")
    print("==============================")


def test_confirmation_with_130px_threshold():
    tracker = PossessionTracker()

    player = {
        "track_id": 40,
        "bbox": [1239.1, 760.0, 1292.4, 849.9],
        "team": "Team A",
    }

    ball_positions = [
        (1178.9, 908.0),
        (1161.6, 909.0),
        (1155.0, 909.0),
    ]

    print("\nTesting 130px confirmation behavior...")

    for index, (ball_x, ball_y) in enumerate(
        ball_positions, start=1
    ):
        result = tracker.update(
            {"ball_x": ball_x, "ball_y": ball_y},
            [player],
        )

        print(
            f"  Step {index}: "
            f"candidate={tracker.candidate_player_id}, "
            f"candidate_frames={tracker.candidate_frames}, "
            f"current={tracker.current_player_id}, "
            f"result={result['possession_player_id']}"
        )

    assert tracker.current_player_id == 40
    print("PASS: Possession confirmed within 130px.")


def test_candidate_survives_one_missing_ball_observation():
    tracker = PossessionTracker()

    player = {
        "track_id": 40,
        "bbox": [1239.1, 760.0, 1292.4, 849.9],
        "team": "Team A",
    }

    # Observation 1: candidate begins.
    tracker.update(
        {"ball_x": 1178.9, "ball_y": 908.0},
        [player],
    )

    print("\nTesting candidate persistence across a missing ball...")
    print("After first observation:")
    print("  Candidate:", tracker.candidate_player_id)
    print("  Candidate frames:", tracker.candidate_frames)

    assert tracker.candidate_player_id == 40
    assert tracker.candidate_frames == 1

    # Observation 2: ball is temporarily unavailable.
    tracker.update(
        {
            "ball_x": None,
            "ball_y": None,
            "status": "unavailable",
        },
        [player],
    )

    print("After missing ball observation:")
    print("  Candidate:", tracker.candidate_player_id)
    print("  Candidate frames:", tracker.candidate_frames)
    print("  Missed frames:", tracker.candidate_missed_frames)

    assert tracker.candidate_player_id == 40
    assert tracker.candidate_missed_frames == 1

    # Observation 3: ball returns near the same player.
    tracker.update(
        {"ball_x": 1161.6, "ball_y": 909.0},
        [player],
    )

    print("After ball returns:")
    print("  Candidate:", tracker.candidate_player_id)
    print("  Candidate frames:", tracker.candidate_frames)

    assert tracker.candidate_player_id == 40
    assert tracker.candidate_frames == 2

    print("PASS: Candidate survives one missing ball observation.")


def test_candidate_survives_one_out_of_range_observation():
    tracker = PossessionTracker()

    player = {
        "track_id": 40,
        "team": "Team A",
        "bbox": [100, 100, 140, 200],
    }

    # Ball is 100 px from the player's foot point.
    tracker.update(
        {"ball_x": 120, "ball_y": 100},
        [player],
    )

    assert tracker.candidate_player_id == 40
    assert tracker.candidate_frames == 1

    # Ball is now beyond the 130 px confirmation threshold.
    tracker.update(
        {"ball_x": 120, "ball_y": 340},
        [player],
    )

    assert tracker.candidate_player_id == 40
    assert tracker.candidate_missed_frames == 1

    print("PASS: Candidate survives one out-of-range observation.")


def test_clears_possession_when_current_player_disappears():
    tracker = PossessionTracker()

    player = {
        "track_id": 147,
        "team": "Team A",
        "bbox": [100, 100, 140, 200],
    }
    ball_state = {
        "ball_x": 120,
        "ball_y": 195,
    }

    # Three consecutive observations confirm possession.
    for _ in range(3):
        result = tracker.update(ball_state, [player])

    assert result["possession_player_id"] == 147
    assert result["possession_team"] == "Team A"
    assert result["has_ball"] == {147: True}

    # The player is no longer tracked in the next frame.
    result = tracker.update(ball_state, [])

    assert result["possession_player_id"] is None
    assert result["possession_team"] == "Unknown"
    assert result["has_ball"] == {}

if __name__ == "__main__":
    main()
    test_confirmation_with_130px_threshold()
    test_candidate_survives_one_missing_ball_observation()
    test_candidate_survives_one_out_of_range_observation()
