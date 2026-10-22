
"""FootballIQ V4 — Detection and Reference Upgrade.

Run:
    python upgrade_footballiq_v4.py

Includes:
- Lower YOLO detection thresholds
- 18 required reference images
- Referee exclusion before ByteTrack
- Three-reference goalkeeper identification
- Reference-aware ball detection
- Penalty-spot exclusion
- Label-free bird's-eye rendering
- Frontend updates

Original files are backed up before changes.
"""

from pathlib import Path
from datetime import datetime
import ast
import re
import shutil


ROOT = Path(__file__).resolve().parent

FILES = (
    "src/footballiq_web.py",
    "src/process_video.py",
    "src/track_ball.py",
    "src/render_refined_birdeye.py",
    "src/detector/jersey_classifier.py",
    "src/detection_settings.py",
    "src/offline_settings.py",
    "web/static/app.js",
    "web/templates/index.html",
    "web/static/styles.css",
)


# ============================================================
# HELPERS
# ============================================================

def change(text, old, new, description):
    """Replace a known section or recognize an existing update."""

    if old == new:
        return text

    count = text.count(old)

    if count == 1:
        return text.replace(old, new, 1)

    if count == 0 and new in text:
        return text

    raise RuntimeError(
        f"{description}: expected one matching section; "
        f"found {count}"
    )


def between(text, start, end, replacement, description):
    """Replace a section using two markers."""

    count = text.count(start)

    if count == 0:
        return text

    if count != 1:
        raise RuntimeError(
            f"{description}: ambiguous start marker"
        )

    begin = text.index(start)

    finish = text.find(
        end,
        begin + len(start)
    )

    if finish < 0:
        raise RuntimeError(
            f"{description}: end marker not found"
        )

    return (
        text[:begin]
        + replacement
        + text[finish:]
    )


# ============================================================
# 1. FLASK BACKEND
# ============================================================

def upgrade_web_backend(text):

    start = "REFERENCE_COUNT = 3\n"
    end = "LANDMARKS = {"

    header = '''REFERENCE_COUNT = 3

REQUIRED_ROLES = (
    "team_a",
    "team_b",
    "team_a_gk",
    "team_b_gk",
    "referee",
    "ball",
)

ROLE_STEMS = {
    role: f"{role}.png"
    for role in REQUIRED_ROLES
}


def _reference_filename(role, sample=1):

    if role not in ROLE_STEMS:
        raise ValueError("Unknown reference role")

    if not isinstance(sample, int) or not (
        1 <= sample <= REFERENCE_COUNT
    ):
        raise ValueError("Invalid reference sample")

    suffix = "" if sample == 1 else f"_{sample}"

    return f"{role}{suffix}.png"


'''

    if text.count(start) != 1 or text.count(end) != 1:
        raise RuntimeError(
            "Flask reference configuration changed. "
            "Cannot update safely."
        )

    text = between(
        text,
        start,
        end,
        header,
        "Flask reference configuration"
    )

    text = text.replace(
        "REFERENCE_COUNT if role in REQUIRED_ROLES else 1",
        "REFERENCE_COUNT"
    )

    text = text.replace(
        "Save all three reference images for Team A, Team B and Ball. Missing:",
        "Save 3 images each for Team A, Team B, both goalkeepers, referees and ball (18 total). Missing:"
    )

    text = text.replace(
        "Draw a crop around the jersey or ball",
        "Select the jersey, referee shirt or ball"
    )

    return text


# ============================================================
# 2. JERSEY CLASSIFIER
# ============================================================

def upgrade_classifier(text):

    if '"Referee": "referee"' not in text:

        text = change(
            text,
            '    "Team B GK": "team_b_gk",\n}',
            '    "Team B GK": "team_b_gk",\n'
            '    "Referee": "referee",\n}',
            "Referee reference role"
        )

    text = text.replace(
        "goalkeeper roles are optional",
        "all jersey roles require three references"
    )

    text = change(
        text,
        '    if "Team A" in current and "Team B" in current:',
        '    if all(len(current.get(name, [])) >= 3 for name in ROLE_FILES):',
        "Reference reuse"
    )

    text = text.replace(
        "Two outfield jerseys required; goalkeeper screenshots optional.",
        "Three photos required per team, goalkeeper and referee."
    )

    text = between(
        text,
        '            optional = name.endswith(" GK")\n',
        '            paths = []\n',
        "",
        "Remove optional goalkeeper selection"
    )

    text = between(
        text,
        '                if index > 1:\n'
        '                    answer = input(',
        '                while True:\n',
        "",
        "Require three offline images"
    )

    text = change(
        text,
        '        for required in ("Team A", "Team B"):',
        '        for required in ROLE_FILES:',
        "Required jersey roles"
    )

    text = change(
        text,
        '            if required not in self.references:\n'
        '                raise ValueError(f"Missing required jersey reference: {required}")',
        '            if len(self.references.get(required, [])) < REFERENCE_COUNT_PER_ROLE:\n'
        '                raise ValueError(f"Three valid jersey references needed for {required}")',
        "Validate three jersey references"
    )

    methods = '''    def _role_scores(self, frame, bbox):

        crops = self._torso_crops(frame, bbox)

        if not crops:
            return {}

        profiles = [
            self._profile(crop)
            for crop in crops
        ]

        return {
            role: max(
                self._score(profile, reference)
                for profile in profiles
                for reference in refs
            )
            for role, refs in self.references.items()
        }

    def is_referee(self, frame, bbox):
        """Exclude only strong referee appearance matches."""

        scores = self._role_scores(frame, bbox)

        referee = scores.get("Referee", 0.0)

        other = max(
            (
                score
                for role, score in scores.items()
                if role != "Referee"
            ),
            default=0.0
        )

        # Keep referee exclusion conservative.
        return (
            referee >= 0.63
            and referee - other >= 0.07
        )

'''

    if "    def is_referee(self, frame, bbox):" not in text:

        text = change(
            text,
            "    def classify(self, frame, bbox):\n",
            methods
            + "    def classify(self, frame, bbox):\n",
            "Insert referee classification"
        )

    text = between(
        text,
        '        crops = self._torso_crops(frame, bbox)\n'
        '        if not crops:\n'
        '            return unknown\n'
        '        profiles = [self._profile(crop) for crop in crops]\n'
        '        scores = {}\n',
        '        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)\n',
        '        scores = self._role_scores(frame, bbox)\n'
        '        if not scores:\n'
        '            return unknown\n',
        "Reuse reference similarity calculations"
    )

    text = change(
        text,
        '        if top < self.min_similarity or top - runner_up < self.min_margin:',
        '        if best == "Referee" or top < self.min_similarity or top - runner_up < self.min_margin:',
        "Exclude referee from team categories"
    )

    return text


# ============================================================
# 3. PLAYER PROCESSING
# ============================================================

def upgrade_player_pipeline(text):

    old = '''    if len(references.get("Team A", [])) < 3 or len(references.get("Team B", [])) < 3:
        raise ValueError(
            f"Three Team A and three Team B jersey crops required in references/{input_path.stem}/"
        )'''

    new = '''    required = (
        "Team A",
        "Team B",
        "Team A GK",
        "Team B GK",
        "Referee",
    )

    missing = [
        role
        for role in required
        if len(references.get(role, [])) < 3
    ]

    if missing:
        raise ValueError(
            "Three jersey references required for: "
            + ", ".join(missing)
        )'''

    text = change(
        text,
        old,
        new,
        "Required player, goalkeeper and referee references"
    )

    text = change(
        text,
        '        if not was_classified or yolo_class != "goalkeeper":\n'
        '            return',
        '        if not was_classified or yolo_class not in ("goalkeeper", "player"):\n'
        '            return',
        "Goalkeeper reference confirmation"
    )

    old = '''                trackable = [
                    d for d in detections
                    if d["class_name"] in ("player", "goalkeeper")
                    and (field_filter is None or field_filter.contains_bbox(d["bbox"], w, h))
                ]
                tracked = tracker.update(trackable)'''

    new = '''                trackable = []

                for detection in detections:

                    label = str(
                        detection["class_name"]
                    ).lower()

                    if (
                        label == "referee"
                        or label not in ("player", "goalkeeper")
                    ):
                        continue

                    if (
                        field_filter is not None
                        and not field_filter.contains_bbox(
                            detection["bbox"], w, h
                        )
                    ):
                        continue

                    if classifier.is_referee(
                        frame,
                        detection["bbox"]
                    ):
                        stats["referee_exclusions"] += 1
                        continue

                    trackable.append(detection)

                tracked = tracker.update(trackable)'''

    text = change(
        text,
        old,
        new,
        "Exclude referees before tracking"
    )

    text = change(
        text,
        '                ball_state = ball_tracker.update(ball, frame_index)',
        '                ball_state = None  # Ball overlay comes from reference-verified tracking.',
        "Suppress unverified ball annotations"
    )

    return text


# ============================================================
# 4. BALL TRACKING
# ============================================================

def upgrade_ball(text):

    # Lower the ball YOLO confidence.
    # Keep appearance matching separate.

    text = re.sub(
        r"(?m)^MIN_YOLO_CONFIDENCE\s*=\s*0\.13\s*$",
        "MIN_YOLO_CONFIDENCE = 0.09",
        text
    )

    if "MIN_YOLO_CONFIDENCE = 0.09" not in text:
        raise RuntimeError(
            "Cannot confirm ball YOLO threshold"
        )

    old = '''    if not (
        0 <= point[0] <= 105
        and 0 <= point[1] <= 68
    ):
        return False

    return True'''

    new = '''    if not (
        0 <= point[0] <= 105
        and 0 <= point[1] <= 68
    ):
        return False

    radius = max(
        0.0,
        float(
            setting(
                "BALL_PENALTY_EXCLUSION_METRES",
                2.0
            )
        )
    )

    if radius > 0:

        for sx, sy in (
            (11.0, 34.0),
            (94.0, 34.0),
        ):

            if np.hypot(
                point[0] - sx,
                point[1] - sy
            ) <= radius:
                return False

    return True'''

    start = text.index("def permitted(")

    end = text.index(
        "# ============================================================\n# 4.",
        start
    )

    permitted_section = text[start:end]

    if "BALL_PENALTY_EXCLUSION_METRES" not in permitted_section:

        text = change(
            text,
            old,
            new,
            "Penalty-spot exclusion"
        )

    text = change(
        text,
        '                    kalman.state is not None\n'
        '                    and frame_idx - kalman.last_measured',
        '                    kalman.state is not None\n'
        '                    and permitted(kalman.state[:2])\n'
        '                    and frame_idx - kalman.last_measured',
        "Reject predicted penalty-spot balls"
    )

    return text


# ============================================================
# 5. BIRD'S-EYE RENDERER
# ============================================================

def upgrade_birdeye(text):

    # Handles original and already label-free renderers.

    text = between(
        text,
        '            label = "GK" if role == "goalkeeper" else str(int(item["track_id"]))',
        '        if ball is not None:\n',
        "",
        "Remove tactical player labels"
    )

    text = between(
        text,
        '        cv2.rectangle(out, (0, 0), (WIDTH, 34),',
        '        return out\n',
        "",
        "Remove tactical banner"
    )

    old = (
        "            if 0 <= x <= 105 "
        "and 0 <= y <= 68:"
    )

    new = '''            radius = max(
                0.0,
                float(
                    getattr(
                        S,
                        "BALL_PENALTY_EXCLUSION_METRES",
                        2.0
                    )
                )
            )

            near_spot = (
                radius > 0
                and any(
                    np.hypot(
                        x - sx,
                        y - sy
                    ) <= radius
                    for sx, sy in (
                        (11.0, 34.0),
                        (94.0, 34.0),
                    )
                )
            )

            if (
                0 <= x <= 105
                and 0 <= y <= 68
                and not near_spot
            ):'''

    if "near_spot =" not in text:

        text = change(
            text,
            old,
            new,
            "Tactical penalty-spot filtering"
        )

    return text


# ============================================================
# 6. DETECTION SETTINGS — LOWER THRESHOLDS
# ============================================================

def upgrade_detector_settings(text):

    begin = "# FOOTBALLIQ_V4_LOWER_THRESHOLDS_BEGIN"
    end = "# FOOTBALLIQ_V4_LOWER_THRESHOLDS_END"

    if begin in text:

        text = re.sub(
            re.escape(begin)
            + r".*?"
            + re.escape(end),
            "",
            text,
            flags=re.S
        )

    override = '''
# FOOTBALLIQ_V4_LOWER_THRESHOLDS_BEGIN

# Lower detection profile.
# Review false positives after processing real match footage.

PRESETS["fast"].update(
    confidence=0.12,
    track_activation=0.19,
    jersey_min_similarity=0.50,
)

PRESETS["balanced"].update(
    confidence=0.09,
    track_activation=0.16,
    jersey_min_similarity=0.48,
)

PRESETS["accuracy"].update(
    confidence=0.08,
    track_activation=0.15,
    jersey_min_similarity=0.46,
)

# Refresh the active preset after updating the values.
SETTINGS = PRESETS[ACTIVE_PRESET].copy()

CLASS_MIN_CONFIDENCE.update(
    player=0.09,
    goalkeeper=0.09,
    referee=0.18,
    ball=0.10,
)

RECOVERY_MIN_CONFIDENCE = 0.11

# FOOTBALLIQ_V4_LOWER_THRESHOLDS_END
'''

    return text.rstrip() + "\n\n" + override


# ============================================================
# 7. OFFLINE BALL SETTINGS
# ============================================================

def upgrade_offline_settings(text):

    return change(
        text,
        "BALL_YOLO_CONF = 0.13",
        "BALL_YOLO_CONF = 0.09",
        "Ball YOLO runtime confidence"
    )


# ============================================================
# 8. WEBSITE JAVASCRIPT
# ============================================================

def upgrade_js(text):

    if (
        "referee:'Referee'" not in text
        and "referee: 'Referee'" not in text
    ):

        text = change(
            text,
            "const roleNames = {team_a:'Team A',team_b:'Team B',ball:'Ball',team_a_gk:'Team A GK',team_b_gk:'Team B GK'};",
            "const roleNames = {team_a:'Team A',team_b:'Team B',team_a_gk:'Team A goalkeeper',team_b_gk:'Team B goalkeeper',referee:'Referee',ball:'Ball'};",
            "Six website reference roles"
        )

    text = change(
        text,
        "const requiredRoles = ['team_a','team_b','ball'];",
        "const requiredRoles = ['team_a','team_b','team_a_gk','team_b_gk','referee','ball'];",
        "18 reference images"
    )

    text = text.replace(
        "Three reference photos each for Team A, Team B and Ball are required (9 total).",
        "Save 3 references for Team A, Team B, both goalkeepers, referees and ball (18 total)."
    )

    text = text.replace(
        "Draw a rectangle over the jersey or ball first.",
        "Select the jersey, referee shirt or ball first."
    )

    text = text.replace(
        "files.filter(f=>f.kind==='image'&&f.label!=='Analytics overview').map(imageCard).join('')",
        "files.filter(f=>f.kind==='image'&&['Team A heatmap','Team B heatmap','Team A positioning','Team B positioning'].includes(f.label)).map(imageCard).join('')"
    )

    labels = {
        "Visual intelligence":
            "Pitch activity and positioning",

        "Complete overview":
            "Match analysis overview",

        "Team geometry":
            "Team shape",

        "Individual heatmaps":
            "Player activity maps",

        "Explore a player track":
            "Select player track",

        "Match replay studio":
            "Match videos",

        "Distance by player track":
            "Player movement estimates",

        "Export your results":
            "Data exports",
    }

    for old, new in labels.items():
        text = text.replace(old, new)

    return text


# ============================================================
# 9. WEBSITE HTML
# ============================================================

def upgrade_html(text):

    if 'data-role="referee"' not in text:

        text = change(
            text,
            '<button class="role-button" data-role="team_b_gk">Team B GK</button>',
            '<button class="role-button" data-role="team_b_gk">Team B GK</button>'
            '<button class="role-button" data-role="referee">Referee</button>',
            "Referee role button"
        )

    text = text.replace(
        "REQUIRED REFERENCES — 9 PHOTOS",
        "REQUIRED REFERENCES — 18 PHOTOS"
    )

    text = text.replace(
        "Optional · 1 photo",
        "3 required photos"
    )

    text = text.replace(
        'id="status-team_a_gk">Not set',
        'id="status-team_a_gk">0 / 3'
    )

    text = text.replace(
        'id="status-team_b_gk">Not set',
        'id="status-team_b_gk">0 / 3'
    )

    if 'data-ref="referee"' not in text:

        anchor = (
            '<div class="reference-item" '
            'data-ref="team_b_gk">'
        )

        start = text.find(anchor)

        if start < 0:
            raise RuntimeError(
                "Cannot find goalkeeper reference counter"
            )

        end = text.find(
            "</div>",
            text.find("</div>", start) + len("</div>")
        )

        if end < 0:
            raise RuntimeError(
                "Cannot find end of goalkeeper counter"
            )

        end += len("</div>")

        referee_item = '''
                    <div class="reference-item" data-ref="referee">
                      <span class="ref-swatch" id="thumb-referee"></span>
                      <div>
                        <strong>Referee</strong>
                        <small>3 required photos</small>
                      </div>
                      <span class="ref-status" id="status-referee">0 / 3</span>
                    </div>'''

        text = (
            text[:end]
            + referee_item
            + text[end:]
        )

    text = text.replace(
        "Three team images feed FootballIQ's multi-reference jersey classifier. Ball crops are saved for a future ball detector upgrade; the current ball tracker does not yet use them.",
        "Three references are required for every role. Referee crops exclude match officials from player analytics; ball crops verify ball candidates."
    )

    labels = {
        "THE GAME, DECODED":
            "MATCH ANALYSIS",

        "Every match tells<br/>a <em>deeper story.</em>":
            "Football match<br/><em>analysis.</em>",

        "Build the full picture.":
            "FootballIQ",

        "From footage to insight.":
            "Local match analysis",

        "Finding the story<br/><em>inside your match.</em>":
            "Processing match<br/><em>footage.</em>",
    }

    for old, new in labels.items():
        text = text.replace(old, new)

    # Refresh cached frontend resources.
    text = re.sub(
        r"(/fi-assets/(?:app\.js|styles\.css)\?v=)\d+",
        r"\g<1>5",
        text
    )

    return text


# ============================================================
# 10. WEBSITE CSS
# ============================================================

def upgrade_css(text):

    marker = "/* FOOTBALLIQ_V4_REFERENCE_LAYOUT */"

    if marker in text:
        return text

    return text + """

/* FOOTBALLIQ_V4_REFERENCE_LAYOUT */

.role-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.role-button {
    white-space: nowrap;
}

.reference-list {
    display: grid;
    gap: 9px;
}

.reference-item {
    min-height: 60px;
}

.reference-preview {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 9px;
}

.preview-slot img {
    width: 100%;
    height: 75px;
    object-fit: cover;
}

.visual-card img {
    display: block;
    width: 100%;
    max-height: 480px;
    object-fit: contain;
}

.metrics-table td {
    font-variant-numeric: tabular-nums;
}

@media (max-width: 650px) {
    .role-button {
        flex: 1 1 auto;
    }
}
"""


# ============================================================
# REGISTER FILE UPDATES
# ============================================================

UPGRADES = {
    "src/footballiq_web.py":
        upgrade_web_backend,

    "src/process_video.py":
        upgrade_player_pipeline,

    "src/track_ball.py":
        upgrade_ball,

    "src/render_refined_birdeye.py":
        upgrade_birdeye,

    "src/detector/jersey_classifier.py":
        upgrade_classifier,

    "src/detection_settings.py":
        upgrade_detector_settings,

    "src/offline_settings.py":
        upgrade_offline_settings,

    "web/static/app.js":
        upgrade_js,

    "web/templates/index.html":
        upgrade_html,

    "web/static/styles.css":
        upgrade_css,
}


# ============================================================
# APPLY UPDATE
# ============================================================

def main():

    missing = [
        name
        for name in FILES
        if not (ROOT / name).is_file()
    ]

    if missing:
        raise SystemExit(
            "Run from the FootballIQ root folder. "
            "Missing: " + ", ".join(missing)
        )

    changes = {}

    # Validate all files before modifying the project.
    for name, transform in UPGRADES.items():

        path = ROOT / name

        original = path.read_text(
            encoding="utf-8"
        )

        updated = transform(original)

        if name.endswith(".py"):
            ast.parse(
                updated,
                filename=name
            )

        changes[name] = updated

        print("Validated:", name)

    changed = [
        name
        for name in FILES
        if changes[name] != (ROOT / name).read_text(
            encoding="utf-8"
        )
    ]

    if not changed:
        print("Everything is already updated.")
        return

    # Create a fresh backup each time.
    backup = ROOT / (
        "footballiq_backup_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    for name in changed:

        destination = backup / name

        destination.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        shutil.copy2(
            ROOT / name,
            destination
        )

    written = []

    try:

        for name in changed:

            (ROOT / name).write_text(
                changes[name],
                encoding="utf-8"
            )

            written.append(name)

    except Exception:

        # Restore original files if writing fails.
        for name in written:

            shutil.copy2(
                backup / name,
                ROOT / name
            )

        raise

    print()
    print("=" * 60)
    print("FOOTBALLIQ V4 UPDATE COMPLETE")
    print("=" * 60)

    print("Updated files:", len(changed))
    print("Backup:", backup)
    print()
    print("Balanced YOLO confidence: 0.09")
    print("Ball YOLO confidence: 0.09")
    print("Required references: 18")
    print("Referee filtering: enabled")
    print("Penalty-spot exclusion: enabled")
    print()
    print("Restart website using:")
    print("python run.py web")


if __name__ == "__main__":
    main()
