"""Kaggriculture Replay Visualizer and Economic Dashboard.

This module provides data extraction, rendering engines, and animated visualization
for Kaggriculture game environment replays.

Attributes:
    EPISODE_STEPS (int): Total steps per episode simulation.
    OPPONENT (str): Baseline opponent agent name.
    SAVE_GIF (bool): Flag indicating whether to output animation to disk.
    GIF_PATH (Path): File path destination for saved animation.
"""

from __future__ import annotations

import json
import logging
import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeAlias

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from kaggle_environments import make
from main import agent

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Global Configuration Constants
EPISODE_STEPS: int = 720
OPPONENT: str = "starter"
SAVE_GIF: bool = False
GIF_PATH: Path = Path("kaggriculture_replay.gif")

# Domain Specific Type Aliases
Position: TypeAlias = Tuple[int, int]
TileData: TypeAlias = Optional[Dict[str, Any] | str]
ActionDict: TypeAlias = Dict[str, Any]


class ReplayExtractionError(Exception):
    """Raised when replay frame extraction fails or encounters malformed state."""


def _as_number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _orders_from_action(action: Any) -> List[List[Any]]:
    """Return market orders while tolerating malformed or empty actions."""
    if not isinstance(action, dict):
        return []
    orders = action.get("market", [])
    if isinstance(orders, list) and orders and isinstance(orders[0], str):
        return [orders]
    return [order for order in orders if isinstance(order, list)] if isinstance(orders, list) else []


def load_match_json(path: Path) -> Dict[str, Any]:
    """Load one benchmark turn log; TXT audit reports are never consulted."""
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict) or not isinstance(payload.get("turns"), list):
        raise ReplayExtractionError(f"Invalid match JSON: {path}")
    return payload


def render_match_json(path: Path, output: Optional[Path] = None) -> Path:
    """Render transaction and market telemetry from one match turn JSON."""
    payload = load_match_json(path)
    player_key = f"player_{payload.get('agent_index', 0)}"
    rows = [turn.get(player_key, {}) for turn in payload["turns"]]
    rows = [row for row in rows if isinstance(row, dict)]
    if not rows:
        raise ReplayExtractionError(f"No player rows in match JSON: {path}")

    turns = list(range(len(rows)))
    cash = [_as_number(row.get("cash")) for row in rows]
    transactions: List[Tuple[int, str, str, float, float]] = []
    price_history: Dict[str, List[float]] = defaultdict(list)
    for turn, row in enumerate(rows):
        market = row.get("market", {})
        prices = market.get("prices", {}) if isinstance(market, dict) else {}
        for item, price in prices.items():
            price_history[str(item)].append(_as_number(price))
        for order in _orders_from_action(row.get("action")):
            if not order:
                continue
            kind = str(order[0]).upper()
            item = str(order[1]).upper() if len(order) > 1 else ""
            quantity = _as_number(order[2], 1.0) if len(order) > 2 else 1.0
            unit_price = _as_number(prices.get(item))
            transactions.append((turn, kind, item, quantity, unit_price))

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    match_id = payload.get("match_id", path.stem)
    fig.suptitle(f"Match {match_id} | JSON transaction dashboard", fontsize=16, fontweight="bold")

    axes[0, 0].plot(turns, cash, color="#1769aa", linewidth=2)
    axes[0, 0].set_title("Kas aktual")
    axes[0, 0].set_xlabel("Turn")
    axes[0, 0].set_ylabel("$", rotation=0)
    axes[0, 0].grid(alpha=0.25)

    labels = [f"D{int(rows[t].get('day', 0)):02d} T{t}" for t, *_ in transactions]
    values = [quantity * price for _, kind, _, quantity, price in transactions]
    colors = ["#2e7d32" if kind == "SELL" else "#c62828" for _, kind, *_ in transactions]
    axes[0, 1].bar(range(len(values)), values, color=colors)
    axes[0, 1].set_title("Nilai transaksi per event")
    axes[0, 1].set_ylabel("Nilai ($)")
    axes[0, 1].set_xticks(range(len(labels)), labels, rotation=75, fontsize=7) if labels else None
    axes[0, 1].grid(axis="y", alpha=0.25)

    plotted_prices = False
    for item, prices in sorted(price_history.items()):
        if any(prices):
            axes[1, 0].plot(turns[:len(prices)], prices, label=item, linewidth=1.2)
            plotted_prices = True
    axes[1, 0].set_title("Harga pasar saat turn")
    axes[1, 0].set_xlabel("Turn")
    axes[1, 0].set_ylabel("Harga/unit ($)")
    if plotted_prices:
        axes[1, 0].legend(fontsize=7, ncol=2)
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].axis("off")
    lines = ["EVENT TIMELINE", ""]
    for turn, kind, item, quantity, price in transactions:
        day = int(rows[turn].get("day", 0))
        if kind == "HIRE":
            detail = f"{kind} ${price:,.0f}"
        elif kind == "BUY_LAND":
            detail = f"{kind} ${price:,.0f}"
        else:
            detail = f"{kind} {quantity:g} {item} @ ${price:,.2f}"
        lines.append(f"D{day:02d} T{turn:03d}  {detail}")
    if len(lines) == 2:
        lines.append("No market transactions recorded")
    axes[1, 1].text(0.02, 0.98, "\n".join(lines[-28:]), va="top", family="monospace", fontsize=8)

    destination = output or path.with_name(f"{path.stem}.png")
    fig.savefig(destination, dpi=140)
    plt.close(fig)
    return destination


@dataclass(frozen=True, slots=True)
class EventEntry:
    """Represents a discrete telemetry or game event log item.

    Attributes:
        step: In-game simulation step turn number.
        day: Current game day index.
        hour: Current game hour index.
        message: Informational description of the event.
    """

    step: int
    day: int
    hour: int
    message: str


@dataclass(slots=True)
class ReplayFrame:
    """Immutable snapshot of single-turn game state telemetry for visualization.

    Attributes:
        step: Global step counter (0..719).
        day: In-game day counter.
        hour: In-game hour turn (0..23).
        my_money: Agent wallet balance.
        enemy_money: Opponent wallet balance.
        farmer_pos: Primary farmer unit coordinates (x, y).
        hands_pos: List of hired hand coordinates [(x, y), ...].
        actions: Executed action dictionary during turn.
        tiles: 2D map state array of farm tiles.
        shed: Storage inventory counts.
        seeds: Seed inventory counts.
        market_prices: Per-item market transaction prices.
        market_inventory: Total available market resource volume.
        unlocked_shops: List of currently unlocked town shop names.
        economic_decisions: Internal diagnostic log dictionary.
    """

    step: int
    day: int
    hour: int
    my_money: float
    enemy_money: float
    farmer_pos: Position
    hands_pos: List[Position]
    actions: ActionDict
    tiles: List[List[TileData]]
    shed: Dict[str, int]
    seeds: Dict[str, int]
    market_prices: Dict[str, float]
    market_inventory: Dict[str, int]
    unlocked_shops: List[str]
    economic_decisions: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class ReplayDataCollector:
    """Extracts, parses, and enriches environment step data into ReplayFrame objects."""

    def __init__(self, agent_fn: Callable[[Dict[str, Any]], ActionDict] = agent) -> None:
        """Initializes the replay data collector.

        Args:
            agent_fn: Callable agent execution function taking an observation dict.
        """
        self.agent_fn = agent_fn
        self.seen_shops: Set[str] = set()
        self.event_logs: List[EventEntry] = []

    def extract_frames(self, env: Any) -> List[ReplayFrame]:
        """Extracts complete replay frames from a finished Kaggle Environment instance.

        Args:
            env: Evaluated Kaggle Environment instance.

        Returns:
            List of parsed ReplayFrame instances.

        Raises:
            ReplayExtractionError: If environment data structure is invalid or missing.
        """
        if not hasattr(env, "steps") or not env.steps:
            raise ReplayExtractionError("Environment execution steps are missing or empty.")

        frames: List[ReplayFrame] = []

        for step_no, raw_step in enumerate(env.steps):
            obs0 = self._extract_obs(raw_step, 0)
            if obs0 is None:
                continue

            obs1 = self._extract_obs(raw_step, 1)
            farms = obs0.get("farms", [{}, {}])
            farm0 = farms[0] if farms else {}
            enemy_money = self._extract_enemy_money(obs0, obs1)

            # Safeguard agent invocation for replay analysis
            try:
                actions = self.agent_fn(obs0)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Agent execution failed at step %d: %s", step_no, exc)
                actions = {"farmer": ["ERROR", str(exc)], "hands": [], "market": []}

            farmer_pos, hands_pos = self._extract_units(farm0)
            market_data = obs0.get("market", {})
            market_prices = market_data.get("prices", {})
            market_inv = market_data.get("inventory", {})

            town_data = obs0.get("town", {})
            current_shops: List[str] = town_data.get("unlocked_shops", [])

            day = obs0.get("day", 0)
            hour = obs0.get("hour", 0)
            self._track_shop_unlocks(current_shops, step_no, day, hour)

            frame = ReplayFrame(
                step=step_no,
                day=day,
                hour=hour,
                my_money=float(farm0.get("money", 0.0)),
                enemy_money=float(enemy_money),
                farmer_pos=farmer_pos,
                hands_pos=hands_pos,
                actions=actions,
                tiles=farm0.get("tiles", []),
                shed=obs0.get("private", {}).get("shed", {}),
                seeds=obs0.get("private", {}).get("seeds", {}),
                market_prices=market_prices,
                market_inventory=market_inv,
                unlocked_shops=list(current_shops),
                economic_decisions=obs0.get("economic_debug", {}),
            )
            frames.append(frame)

        return frames

    def _extract_obs(self, step_entry: Any, player_idx: int) -> Optional[Dict[str, Any]]:
        if isinstance(step_entry, list):
            if len(step_entry) <= player_idx:
                return None
            entry = step_entry[player_idx]
        else:
            entry = step_entry

        if not isinstance(entry, dict):
            return None

        obs = entry.get("observation")
        if isinstance(obs, dict):
            return obs
        if "farms" in entry and "market" in entry:
            return entry
        return None

    def _extract_enemy_money(
        self, obs0: Dict[str, Any], obs1: Optional[Dict[str, Any]]
    ) -> float:
        if obs1 is not None:
            farms1 = obs1.get("farms", [{}, {}])
            if len(farms1) > 1:
                return float(farms1[1].get("money", 0.0))
        try:
            return float(obs0["farms"][1].get("money", 0.0))
        except (IndexError, KeyError, TypeError):
            return 0.0

    def _extract_units(self, farm: Dict[str, Any]) -> Tuple[Position, List[Position]]:
        raw_farmer = farm.get("farmer", (0, 0))
        farmer: Position = (int(raw_farmer[0]), int(raw_farmer[1]))
        hands: List[Position] = [
            (int(h[0]), int(h[1])) for h in farm.get("hands", [])
        ]
        return farmer, hands

    def _track_shop_unlocks(
        self, current_shops: List[str], step: int, day: int, hour: int
    ) -> None:
        for shop in current_shops:
            if shop not in self.seen_shops:
                self.seen_shops.add(shop)
                self.event_logs.append(
                    EventEntry(step, day, hour, f"Unlocked Town Shop: {shop}")
                )


class FarmGridRenderer:
    """Renders 2D farm tile grid, spatial features, and unit markers."""

    TILE_COLORS: Dict[str, str] = {
        "LOCKED": "#424242",
        "WEED": "#9E9D24",
        "PLANT": "#66BB6A",
        "COOP": "#A1887F",
        "PASTURE": "#8D6E63",
        "EMPTY": "#E8F5E9",
    }

    def render(self, ax: Axes, frame: ReplayFrame) -> None:
        """Renders grid environment state onto target Matplotlib Axes.

        Args:
            ax: Targeted Matplotlib Axes object.
            frame: ReplayFrame snapshot to render.
        """
        ax.clear()
        tiles = frame.tiles
        height = len(tiles)
        width = max((len(row) for row in tiles), default=10) if height > 0 else 10

        for y in range(height):
            for x in range(width):
                tile = tiles[y][x] if x < len(tiles[y]) else "LOCKED"
                kind = self._get_kind(tile)
                color = self.TILE_COLORS.get(kind, "#E8F5E9")

                rect = Rectangle(
                    (x, y), 1, 1, facecolor=color, edgecolor="#BDBDBD", linewidth=0.6
                )
                ax.add_patch(rect)

                label = self._get_label(tile)
                if label:
                    ax.text(
                        x + 0.5,
                        y + 0.5,
                        label,
                        ha="center",
                        va="center",
                        fontsize=7,
                        fontweight="bold",
                        color="#212121" if kind != "LOCKED" else "#FFFFFF",
                    )

        # Draw Main Farmer Unit
        fx, fy = frame.farmer_pos
        ax.scatter(
            [fx + 0.5],
            [fy + 0.5],
            s=220,
            marker="o",
            facecolor="#1976D2",
            edgecolor="black",
            zorder=10,
        )
        ax.text(
            fx + 0.5,
            fy + 0.5,
            "F",
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            fontweight="bold",
            zorder=11,
        )

        # Draw Hired Farm Hands Units
        for i, (hx, hy) in enumerate(frame.hands_pos):
            ax.scatter(
                [hx + 0.5],
                [hy + 0.5],
                s=180,
                marker="s",
                facecolor="#F57C00",
                edgecolor="black",
                zorder=10,
            )
            ax.text(
                hx + 0.5,
                hy + 0.5,
                f"H{i}",
                ha="center",
                va="center",
                color="white",
                fontsize=7,
                fontweight="bold",
                zorder=11,
            )

        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_aspect("equal")
        ax.set_xticks(range(width + 1))
        ax.set_yticks(range(height + 1))
        ax.grid(True, linewidth=0.4, alpha=0.3)
        ax.set_title(
            f"Farm Layout | Day {frame.day:02d} | Hour {frame.hour:02d} | Step {frame.step:03d}",
            fontsize=10,
            fontweight="bold",
        )

    def _get_kind(self, tile: TileData) -> str:
        if tile is None:
            return "EMPTY"
        if isinstance(tile, str):
            return tile
        if isinstance(tile, dict):
            return str(tile.get("kind", "UNKNOWN"))
        return "UNKNOWN"

    def _get_label(self, tile: TileData) -> str:
        if not tile:
            return ""
        if isinstance(tile, str):
            return "X" if tile == "LOCKED" else tile[:2]
        if isinstance(tile, dict):
            kind = tile.get("kind", "?")
            if kind == "PLANT":
                return str(tile.get("crop", tile.get("seed", "W")))[:2]
            if kind in ("COOP", "PASTURE"):
                return str(tile.get("animal", kind))[:2]
            return "WD" if kind == "WEED" else str(kind)[:2]
        return "?"


class DashboardAnalyticsRenderer:
    """Renders financial charts, market metrics, and telemetry text blocks."""

    TRACKED_ITEMS: List[str] = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "EGG", "MILK"]

    def render_coin_history(
        self, ax: Axes, frames: List[ReplayFrame], current_idx: int
    ) -> None:
        """Renders current vs opponent wealth over time.

        Args:
            ax: Targeted Matplotlib Axes object.
            frames: Full dataset of replay frames up to current.
            current_idx: Active index bound for rendering.
        """
        ax.clear()
        sub_frames = frames[: current_idx + 1]
        steps = [f.step for f in sub_frames]
        my_money = [f.my_money for f in sub_frames]
        enemy_money = [f.enemy_money for f in sub_frames]

        ax.plot(steps, my_money, label="YOU", color="#388E3C", linewidth=2.0)
        ax.plot(
            steps,
            enemy_money,
            label=OPPONENT.upper(),
            color="#D32F2F",
            linestyle="--",
            linewidth=1.5,
        )

        ax.set_title("Coin Wealth Progression", fontsize=9, fontweight="bold")
        ax.set_ylabel("Coins ($)", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=7)

    def render_market_prices(
        self, ax: Axes, frames: List[ReplayFrame], current_idx: int
    ) -> None:
        """Renders market asset price movement trends over time.

        Args:
            ax: Targeted Matplotlib Axes object.
            frames: Full dataset of replay frames up to current.
            current_idx: Active index bound for rendering.
        """
        ax.clear()
        sub_frames = frames[: current_idx + 1]
        steps = [f.step for f in sub_frames]

        for item in self.TRACKED_ITEMS:
            prices = [f.market_prices.get(item, 0.0) for f in sub_frames]
            if any(p > 0 for p in prices):
                ax.plot(steps, prices, label=item, linewidth=1.2)

        ax.set_title("Market Price Dynamics", fontsize=9, fontweight="bold")
        ax.set_ylabel("Unit Price ($)", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=6, ncol=2)

    def render_telemetry_panel(
        self, ax: Axes, frame: ReplayFrame, events: List[EventEntry]
    ) -> None:
        """Renders text-based economic summary and event logs.

        Args:
            ax: Dedicated text panel Matplotlib Axes object.
            frame: Active ReplayFrame snapshot.
            events: List of recorded EventEntry domain objects.
        """
        ax.clear()
        ax.axis("off")

        margin = frame.my_money - frame.enemy_money
        lead_str = f"LEAD +${margin:,.0f}" if margin >= 0 else f"BEHIND -${abs(margin):,.0f}"

        lines = [
            "=== ECONOMIC SUMMARY ===",
            f"YOU: ${frame.my_money:,.0f}  |  {OPPONENT.upper()}: ${frame.enemy_money:,.0f}",
            f"Margin: ${margin:+,.0f} ({lead_str})",
            "",
            "=== SHED & SEEDS INVENTORY ===",
            f"Shed Storage : {sum(frame.shed.values())}/100 items {frame.shed}",
            f"Seed Reserve : {frame.seeds}",
            "",
            "=== TOWN SHOPS STATUS ===",
            f"Active Shops : {', '.join(frame.unlocked_shops) if frame.unlocked_shops else 'None'}",
            "",
            "=== RECENT EVENT LOGS ===",
        ]

        recent_events = [e for e in events if e.step <= frame.step][-4:]
        for ev in recent_events:
            lines.append(f"[D{ev.day:02d} H{ev.hour:02d}] {ev.message}")

        ax.text(
            0.02,
            0.98,
            "\n".join(lines),
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5", alpha=0.8),
        )


def main() -> None:
    """Main execution function initializing simulation and dashboard GUI."""
    parser = argparse.ArgumentParser(description="Visualize Kaggriculture JSON match logs")
    parser.add_argument("--json", type=Path, help="Path to match_XX_turns.json")
    parser.add_argument("--output", type=Path, help="PNG output path")
    args = parser.parse_args()
    if args.json:
        output = render_match_json(args.json, args.output)
        logger.info("Saved JSON match visualization to %s", output)
        return

    logger.info("Initializing Kaggriculture Multi-Panel Dashboard...")

    env = make("kaggriculture", configuration={"episodeSteps": EPISODE_STEPS}, debug=True)
    env.run([agent, OPPONENT])

    collector = ReplayDataCollector(agent)
    frames = collector.extract_frames(env)

    if not frames:
        raise ReplayExtractionError("Failed to extract frames from environment execution.")

    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(
        2, 3, width_ratios=[1.2, 1.0, 1.0], height_ratios=[1.0, 1.0]
    )

    ax_grid: Axes = fig.add_subplot(gs[:, 0])
    ax_coins: Axes = fig.add_subplot(gs[0, 1])
    ax_market: Axes = fig.add_subplot(gs[1, 1])
    ax_telemetry: Axes = fig.add_subplot(gs[:, 2])

    grid_renderer = FarmGridRenderer()
    dashboard_renderer = DashboardAnalyticsRenderer()

    def update(index: int) -> None:
        frame = frames[index]
        grid_renderer.render(ax_grid, frame)
        dashboard_renderer.render_coin_history(ax_coins, frames, index)
        dashboard_renderer.render_market_prices(ax_market, frames, index)
        dashboard_renderer.render_telemetry_panel(
            ax_telemetry, frame, collector.event_logs
        )

    anim = FuncAnimation(fig, update, frames=len(frames), interval=80, repeat=True)

    if SAVE_GIF:
        logger.info("Exporting animation output GIF to %s...", GIF_PATH)
        anim.save(str(GIF_PATH), writer="pillow", fps=12)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()