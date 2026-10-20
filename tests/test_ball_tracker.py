from tracker.ball_tracker import BallTracker


def main():

    print("Creating BallTracker...")

    tracker = BallTracker()

    # ---------------------------------------------------------
    # Test 1: Ball detected
    # ---------------------------------------------------------
    print("\nTest 1: Ball detected")

    detection = {
        "bbox": [100, 200, 120, 220],
        "confidence": 0.91,
    }

    state = tracker.update(
        detection,
        frame_index=0,
    )

    print(state)

    # ---------------------------------------------------------
    # Test 2: Ball moves
    # ---------------------------------------------------------
    print("\nTest 2: Ball movement")

    detection = {
        "bbox": [110, 205, 130, 225],
        "confidence": 0.88,
    }

    state = tracker.update(
        detection,
        frame_index=1,
    )

    print(state)

    # ---------------------------------------------------------
    # Test 3: Ball not detected
    # ---------------------------------------------------------
    print("\nTest 3: Ball unavailable")

    state = tracker.update(
        None,
        frame_index=2,
    )

    print(state)

    # ---------------------------------------------------------
    # Test 4: Player possession
    # ---------------------------------------------------------
    print("\nTest 4: Player possession")

    detection = {
        "bbox": [150, 250, 170, 270],
        "confidence": 0.95,
    }

    state = tracker.update(
        detection,
        frame_index=3,
    )

    state = tracker.update_possession(
        state,
        possession_player_id=12,
    )

    print(state)

    # ---------------------------------------------------------
    # Test 5: Ball center
    # ---------------------------------------------------------
    print("\nTest 5: Ball center")

    center = tracker.get_ball_center(
        detection
    )

    print(
        f"Ball center: {center}"
    )

    # ---------------------------------------------------------
    # Test 6: Distance
    # ---------------------------------------------------------
    print("\nTest 6: Distance")

    distance = tracker.distance(
        (100, 100),
        (103, 104),
    )

    print(
        f"Distance: {distance:.2f}"
    )

    print("\nBall tracker test completed.")


if __name__ == "__main__":
    main()