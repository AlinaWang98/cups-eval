from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


RELEASE_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = RELEASE_ROOT / "results"
TABLE_DIR = RESULTS_DIR / "tables"
SOURCE_TABLE_DIR = RESULTS_DIR / "source_tables"
CUPS_DIR = RESULTS_DIR / "cups"
FIG_DIR = RESULTS_DIR / "figures"

FONT_CANDIDATES = {
    "regular": [
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ],
    "bold": [
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ],
    "italic": [
        Path("/System/Library/Fonts/Supplemental/Arial Italic.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"),
    ],
}

CUPS_FIGURE_SCOPES = {
    "medical": "medical_glm_kimi",
    "nonmedical": "nonmedical_deepseek_flash",
}

INK = "#202A35"
MUTED = "#5C6670"
GRID = "#D9DEE5"
AXIS = "#3A424A"
BLUE = "#0077BB"
CYAN = "#33BBEE"
TEAL = "#009988"
ORANGE = "#EE7733"
RED = "#CC3311"
GREY = "#9AA3AD"
BLACK = "#000000"

METHOD_STYLE = {
    "Direct": (GREY, "circle"),
    "CoT": (BLUE, "square"),
    "Planner": (TEAL, "triangle"),
    "Debate": (ORANGE, "diamond"),
    "Oracle": (BLACK, "diamond"),
    "Rule-list": (CYAN, "square"),
    "CUPS": (RED, "circle"),
}


def font(size: int, *, bold: bool = False, italic: bool = False) -> ImageFont.ImageFont:
    style = "bold" if bold else "italic" if italic else "regular"
    for path in FONT_CANDIDATES[style]:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def draw_rotated_text(
    img: Image.Image,
    text: str,
    xy: tuple[int, int],
    fnt: ImageFont.ImageFont,
    fill: str = INK,
) -> None:
    dummy = Image.new("RGBA", (10, 10), (255, 255, 255, 0))
    dd = ImageDraw.Draw(dummy)
    w, h = text_size(dd, text, fnt)
    layer = Image.new("RGBA", (w + 10, h + 10), (255, 255, 255, 0))
    ld = ImageDraw.Draw(layer)
    ld.text((5, 5), text, font=fnt, fill=fill)
    layer = layer.rotate(90, expand=True)
    img.alpha_composite(layer, (xy[0] - layer.width // 2, xy[1] - layer.height // 2))


def marker(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    color: str,
    kind: str,
    size: int = 15,
    outline: str = BLACK,
) -> None:
    x = int(round(x))
    y = int(round(y))
    r = size // 2
    if kind == "circle":
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=outline, width=2)
    elif kind == "square":
        draw.rectangle((x - r, y - r, x + r, y + r), fill=color, outline=outline, width=2)
    elif kind == "triangle":
        draw.polygon([(x, y - r), (x - r, y + r), (x + r, y + r)], fill=color, outline=outline)
        draw.line([(x, y - r), (x - r, y + r), (x + r, y + r), (x, y - r)], fill=outline, width=2)
    elif kind == "diamond":
        draw.polygon([(x, y - r), (x - r, y), (x, y + r), (x + r, y)], fill=color, outline=outline)
        draw.line([(x, y - r), (x - r, y), (x, y + r), (x + r, y), (x, y - r)], fill=outline, width=2)


def draw_legend(draw: ImageDraw.ImageDraw, items: list[str], x: int, y: int, columns: int = 4) -> None:
    small = font(28)
    col_w = 260
    row_h = 42
    for i, item in enumerate(items):
        cx = x + (i % columns) * col_w
        cy = y + (i // columns) * row_h
        color, kind = METHOD_STYLE[item]
        marker(draw, cx, cy + 15, color, kind, 18)
        draw.text((cx + 24, cy), item, font=small, fill=INK)


def draw_axes(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xticks: list[float],
    yticks: list[float],
    xlabel: str,
    ylabel: str,
    *,
    xfmt=lambda x: f"{x:g}",
    yfmt=lambda y: f"{y:g}",
) -> tuple:
    left, top, right, bottom = rect
    tick_font = font(25)
    label_font = font(30)

    def sx(x: float) -> float:
        return left + (x - xlim[0]) / (xlim[1] - xlim[0]) * (right - left)

    def sy(y: float) -> float:
        return bottom - (y - ylim[0]) / (ylim[1] - ylim[0]) * (bottom - top)

    for y in yticks:
        yy = sy(y)
        draw.line((left, yy, right, yy), fill=GRID, width=2)
        draw.text((left - 14, yy), yfmt(y), anchor="rm", font=tick_font, fill=MUTED)
    for x in xticks:
        xx = sx(x)
        draw.line((xx, bottom, xx, bottom + 8), fill=AXIS, width=2)
        draw.text((xx, bottom + 18), xfmt(x), anchor="ma", font=tick_font, fill=MUTED)
    draw.line((left, top, left, bottom), fill=AXIS, width=3)
    draw.line((left, bottom, right, bottom), fill=AXIS, width=3)
    draw.text(((left + right) // 2, bottom + 66), xlabel, anchor="ma", font=label_font, fill=INK)
    draw_rotated_text(img, ylabel, (left - 95, (top + bottom) // 2), label_font, INK)
    return sx, sy


def save_figure(img: Image.Image, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rgb = img.convert("RGB")
    rgb.save(FIG_DIR / f"{stem}.png", dpi=(300, 300))
    rgb.save(FIG_DIR / f"{stem}.pdf", "PDF", resolution=300)


def pareto(points: list[dict]) -> list[dict]:
    ordered = sorted(points, key=lambda d: d["cost"])
    best = -1.0
    frontier = []
    for p in ordered:
        if p["acc"] > best + 1e-9:
            frontier.append(p)
            best = p["acc"]
    return frontier


def load_cups_points() -> dict[str, dict[str, float]]:
    df = pd.read_csv(CUPS_DIR / "cups_best_policy_tests.csv")
    points = {}
    for figure_scope, scope_id in CUPS_FIGURE_SCOPES.items():
        row = df[
            (df["scope_id"] == scope_id)
            & (df["evaluation_regime"] == "item_oof")
            & (df["selection_type"] == "best_accuracy")
            & (df["baseline"] == "cot")
        ].iloc[0]
        points[figure_scope] = {
            "accuracy": float(row["selected_accuracy"]),
            "cost_x_direct": float(row["cups_cost_multiplier"]),
            "relative_cost_per_correct": float(row["cups_relative_cost_per_correct"]),
        }
    return points


def figure_1_cost_accuracy() -> None:
    df = pd.read_csv(TABLE_DIR / "cross_domain_main_table.csv")
    method_map = {
        "direct": "Direct",
        "cot": "CoT",
        "planner_verifier": "Planner",
        "debate_2_judge": "Debate",
        "direct_first_oracle_any": "Oracle",
        "learned_rule_list": "Rule-list",
    }
    cups_points = load_cups_points()

    img = Image.new("RGBA", (2400, 1400), "white")
    draw = ImageDraw.Draw(img)
    draw.text((120, 48), "Cost-accuracy frontier", font=font(42, bold=True), fill=INK)
    draw.text(
        (120, 102),
        "Accuracy gains from always-on collaboration are shown against direct-normalized cost.",
        font=font(29),
        fill=MUTED,
    )
    draw_legend(draw, ["Direct", "CoT", "Planner", "Debate", "Oracle", "Rule-list", "CUPS"], 1240, 56, columns=4)

    panels = [
        ("medical", "Medical QA pooled", (170, 245, 1120, 1140)),
        ("nonmedical", "Non-medical pilots", (1320, 245, 2270, 1140)),
    ]
    for scope, title, rect in panels:
        left, top, right, bottom = rect
        draw.text((left, top - 64), title, font=font(34, bold=True), fill=INK)
        sx, sy = draw_axes(
            img,
            draw,
            rect,
            (0.75, 3.55),
            (0.66, 0.86),
            [1.0, 2.0, 3.0],
            [0.68, 0.72, 0.76, 0.80, 0.84],
            "Cost multiplier vs direct",
            "Accuracy",
            xfmt=lambda x: f"{x:.0f}x",
            yfmt=lambda y: f"{y:.2f}",
        )
        rows = df[df["scope"] == scope]
        pts = []
        for _, row in rows.iterrows():
            method = method_map.get(row["method"])
            if not method:
                continue
            pts.append({"method": method, "acc": float(row["accuracy"]), "cost": float(row["cost_x_direct"])})
        pts.append(
            {
                "method": "CUPS",
                "acc": cups_points[scope]["accuracy"],
                "cost": cups_points[scope]["cost_x_direct"],
            }
        )
        frontier = pareto(pts)
        if len(frontier) > 1:
            draw.line([(sx(p["cost"]), sy(p["acc"])) for p in frontier], fill="#6B7280", width=5)
        for p in pts:
            color, kind = METHOD_STYLE[p["method"]]
            x, y = sx(p["cost"]), sy(p["acc"])
            marker(draw, x, y, color, kind, 22)
            dx = 14 if p["cost"] < 2.8 else -132
            dy = -34 if p["method"] not in {"Planner", "Debate"} else 18
            draw.text((x + dx, y + dy), p["method"], font=font(24), fill=INK)
    save_figure(img, "figure_1_cost_accuracy_frontier")


def figure_2_capability_decay() -> None:
    rows = []
    with (TABLE_DIR / "math500_capability_gradient.csv").open() as f:
        for row in csv.DictReader(f):
            if row["direct"] == "TBD":
                continue
            rows.append(
                {
                    "model": row["model"].replace("-Preview", "").replace("DeepSeek v4 flash", "DeepSeek"),
                    "direct": float(row["direct"]),
                    "planner": float(row["planner_gain"]) * 100,
                    "debate": float(row["debate_gain"]) * 100 if row["debate_gain"] else None,
                    "oracle": float(row["oracle_gap"]) * 100,
                }
            )
    rows = sorted(rows, key=lambda r: r["direct"])
    series = [
        ("Planner gain", "planner", TEAL, "circle"),
        ("Debate gain", "debate", ORANGE, "square"),
        ("Oracle gap", "oracle", BLACK, "diamond"),
    ]
    img = Image.new("RGBA", (2200, 1400), "white")
    draw = ImageDraw.Draw(img)
    draw.text((120, 50), "Collaboration utility decays with direct accuracy", font=font(42, bold=True), fill=INK)
    draw.text(
        (120, 105),
        "MATH-500 model gradient; gains are measured relative to direct answering.",
        font=font(29),
        fill=MUTED,
    )
    rect = (180, 240, 2050, 1135)
    sx, sy = draw_axes(
        img,
        draw,
        rect,
        (0.40, 1.00),
        (-6, 24),
        [0.4, 0.6, 0.8, 1.0],
        [-6, 0, 6, 12, 18, 24],
        "Direct accuracy",
        "Gain vs direct (percentage points)",
        xfmt=lambda x: f"{x:.1f}",
        yfmt=lambda y: f"{y:.0f}",
    )
    draw.line((rect[0], sy(0), rect[2], sy(0)), fill="#707780", width=3)
    for label, key, color, kind in series:
        pts = [(sx(r["direct"]), sy(r[key])) for r in rows if r[key] is not None]
        draw.line(pts, fill=color, width=5)
        for r in rows:
            if r[key] is None:
                continue
            marker(draw, sx(r["direct"]), sy(r[key]), color, kind, 22)
    legend_items = [("Planner gain", TEAL, "circle"), ("Debate gain", ORANGE, "square"), ("Oracle gap", BLACK, "diamond")]
    lx = 1420
    for i, (label, color, kind) in enumerate(legend_items):
        y = 55 + i * 42
        marker(draw, lx, y + 15, color, kind, 20)
        draw.text((lx + 26, y), label, font=font(28), fill=INK)
    for r in rows:
        draw.text((sx(r["direct"]) + 14, sy(r["oracle"]) - 50), r["model"], font=font(24), fill=MUTED)
    save_figure(img, "figure_2_capability_decay")


def retained_oracle_gain(acc: float, direct: float, oracle: float) -> float:
    denom = oracle - direct
    if denom <= 0:
        return 0.0
    return (acc - direct) / denom * 100


def figure_3_rog_cpc() -> None:
    df = pd.read_csv(TABLE_DIR / "cross_domain_main_table.csv")
    method_map = {
        "cot": "CoT",
        "planner_verifier": "Planner",
        "debate_2_judge": "Debate",
        "learned_rule_list": "Rule-list",
    }
    cups_points = load_cups_points()
    img = Image.new("RGBA", (2400, 1400), "white")
    draw = ImageDraw.Draw(img)
    draw.text((120, 48), "Opportunity captured versus cost per correct", font=font(42, bold=True), fill=INK)
    draw.text(
        (120, 102),
        "Retained oracle gain is the share of direct-to-oracle accuracy headroom captured by a strategy.",
        font=font(29),
        fill=MUTED,
    )
    draw_legend(draw, ["CoT", "Planner", "Debate", "Rule-list", "CUPS"], 1440, 58, columns=3)
    panels = [
        ("medical", "Medical QA pooled", (170, 245, 1120, 1140)),
        ("nonmedical", "Non-medical pilots", (1320, 245, 2270, 1140)),
    ]
    for scope, title, rect in panels:
        left, top, _, _ = rect
        sub = df[df["scope"] == scope]
        direct = float(sub[sub["method"] == "direct"]["accuracy"].iloc[0])
        oracle = float(sub[sub["method"] == "direct_first_oracle_any"]["accuracy"].iloc[0])
        draw.text((left, top - 64), title, font=font(34, bold=True), fill=INK)
        sx, sy = draw_axes(
            img,
            draw,
            rect,
            (0.80, 4.40),
            (-10, 110),
            [1.0, 2.0, 3.0, 4.0],
            [0, 25, 50, 75, 100],
            "Relative cost per correct",
            "Retained oracle gain (%)",
            xfmt=lambda x: f"{x:.0f}",
            yfmt=lambda y: f"{y:.0f}",
        )
        draw.line((rect[0], sy(100), rect[2], sy(100)), fill="#71717A", width=3)
        draw.line((rect[0], sy(0), rect[2], sy(0)), fill="#71717A", width=2)
        pts = []
        for _, row in sub.iterrows():
            method = method_map.get(row["method"])
            if not method:
                continue
            pts.append(
                {
                    "method": method,
                    "x": float(row["relative_cost_per_correct"]),
                    "y": retained_oracle_gain(float(row["accuracy"]), direct, oracle),
                }
            )
        pts.append(
            {
                "method": "CUPS",
                "x": cups_points[scope]["relative_cost_per_correct"],
                "y": retained_oracle_gain(cups_points[scope]["accuracy"], direct, oracle),
            }
        )
        for p in pts:
            color, kind = METHOD_STYLE[p["method"]]
            x, y = sx(p["x"]), sy(p["y"])
            marker(draw, x, y, color, kind, 22)
            dy = -36 if p["method"] in {"CUPS", "CoT"} else 16
            draw.text((x + 14, y + dy), p["method"], font=font(24), fill=INK)
    save_figure(img, "figure_3_retained_oracle_gain")


def load_paired_stats() -> pd.DataFrame:
    paths = {
        "DeepSeek": SOURCE_TABLE_DIR / "paired_stats_math500_deepseek_v4_flash.csv",
        "GLM-5.2": SOURCE_TABLE_DIR / "paired_stats_math500_glm_5_2.csv",
        "MiniMax": SOURCE_TABLE_DIR / "paired_stats_math500_minimax_m3.csv",
        "Qwen": SOURCE_TABLE_DIR / "paired_stats_math500_qwen_3_7_max_preview.csv",
    }
    frames = []
    for model, path in paths.items():
        df = pd.read_csv(path)
        df["model"] = model
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def figure_4_rescue_harm() -> None:
    df = load_paired_stats()
    order = ["cot", "planner_verifier", "debate_2_judge"]
    labels = {"cot": "CoT", "planner_verifier": "Planner", "debate_2_judge": "Debate"}
    models = ["DeepSeek", "GLM-5.2", "MiniMax", "Qwen"]
    img = Image.new("RGBA", (2400, 1500), "white")
    draw = ImageDraw.Draw(img)
    draw.text((120, 48), "Rescue and harm relative to direct answering", font=font(42, bold=True), fill=INK)
    draw.text(
        (120, 102),
        "Positive bars show direct-wrong items rescued per 100 questions; negative bars show direct-correct items harmed.",
        font=font(29),
        fill=MUTED,
    )
    panels = [
        ("DeepSeek", (160, 245, 1110, 760)),
        ("GLM-5.2", (1300, 245, 2250, 760)),
        ("MiniMax", (160, 900, 1110, 1415)),
        ("Qwen", (1300, 900, 2250, 1415)),
    ]
    for model_name, rect in panels:
        left, top, right, bottom = rect
        draw.text((left, top - 52), model_name, font=font(34, bold=True), fill=INK)
        sx, sy = draw_axes(
            img,
            draw,
            rect,
            (-0.5, 2.5),
            (-15, 22),
            [0, 1, 2],
            [-10, 0, 10, 20],
            "",
            "Items per 100 questions",
            xfmt=lambda x: labels[order[int(x)]] if int(x) in range(3) else "",
            yfmt=lambda y: f"{y:.0f}",
        )
        zero = sy(0)
        draw.line((left, zero, right, zero), fill=AXIS, width=3)
        sub = df[df["model"] == model_name].set_index("strategy")
        for idx, strategy in enumerate(order):
            n = float(sub.loc[strategy, "n"])
            gains = float(sub.loc[strategy, "gains"]) / n * 100
            harms = float(sub.loc[strategy, "harms"]) / n * 100
            x = sx(idx)
            bar_w = 96
            draw.rectangle((x - bar_w, sy(gains), x - 8, zero), fill=TEAL, outline=BLACK, width=2)
            draw.rectangle((x + 8, zero, x + bar_w, sy(-harms)), fill=ORANGE, outline=BLACK, width=2)
            draw.text((x - bar_w / 2 - 4, sy(gains) - 34), f"{gains:.1f}", anchor="mm", font=font(24), fill=INK)
            draw.text((x + bar_w / 2 + 4, sy(-harms) + 34), f"{harms:.1f}", anchor="mm", font=font(24), fill=INK)
    marker(draw, 1390, 80, TEAL, "square", 22)
    draw.text((1424, 63), "Rescue", font=font(28), fill=INK)
    marker(draw, 1600, 80, ORANGE, "square", 22)
    draw.text((1634, 63), "Harm", font=font(28), fill=INK)
    save_figure(img, "figure_4_rescue_harm")


def main() -> None:
    figure_1_cost_accuracy()
    figure_2_capability_decay()
    figure_3_rog_cpc()
    figure_4_rescue_harm()
    for p in sorted(FIG_DIR.glob("figure_*.png")):
        print(p)


if __name__ == "__main__":
    main()
