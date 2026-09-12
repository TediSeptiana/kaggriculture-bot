"""Automated Benchmark Suite with Unique Financial Audit Logging for Kaggriculture.

This module manages multi-match simulation runs against opponent baselines,
integrates transactional financial auditing, and exports replay logs for visualizer analysis.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from kaggle_environments import make

# Import Auditor with backward-compatibility fallback
try:
    from audit_logger import TransactionalAuditor as Auditor
except ImportError:
    from audit_logger import FinancialAuditor as Auditor  # type: ignore[no-redef]

from main import agent as my_agent

logger = logging.getLogger(__name__)


# ==========================================================================
# NEW: Turn-level logger (added for match-to-match divergence debugging)
# ==========================================================================
class TurnLogger:
    """Captures per-turn state snapshots for debugging match-to-match divergence.

    This class is purely additive: it does NOT interfere with the existing
    audit pipeline. It writes one JSON file per match, alongside the audit
    reports, so that the user can diff matches turn-by-turn (for example,
    to find the exact turn where Match 06 diverged from Match 07).
    """

    # Common observation keys across kaggriculture variants.
    # We use best-effort extraction so nothing breaks if the schema differs.
    CASH_KEYS: Sequence[str] = ("cash", "money", "balance", "funds", "gold")
    PRICE_KEYS: Sequence[str] = ("prices", "market", "market_prices", "price")
    DAY_KEYS: Sequence[str] = ("day", "turn", "step")
    SEED_KEYS: Sequence[str] = ("seeds", "seed_inventory")
    CROP_KEYS: Sequence[str] = ("farms", "farm", "crops", "plots")

    def __init__(self, log_dir: Union[str, Path]) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Safe attribute/dict access helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _try_get(obj: Any, key: str, default: Any = None) -> Any:
        try:
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)
        except Exception:
            return default

    @classmethod
    def _deep_find(
        cls,
        obj: Any,
        candidates: Sequence[str],
        max_depth: int = 3,
        _depth: int = 0,
    ) -> Any:
        """Best-effort recursive search for any of the candidate keys."""
        if _depth > max_depth or obj is None:
            return None
        if isinstance(obj, dict):
            for key in candidates:
                if key in obj:
                    return obj[key]
            for value in obj.values():
                found = cls._deep_find(value, candidates, max_depth, _depth + 1)
                if found is not None:
                    return found
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                found = cls._deep_find(value, candidates, max_depth, _depth + 1)
                if found is not None:
                    return found
        return None

    # ------------------------------------------------------------------
    # Per-step summary
    # ------------------------------------------------------------------
    def _summarize_step_state(self, step_state: Any) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}

        # Raw action is the single most useful field for divergence analysis:
        # HIRE / BUY_LAND / PLANT / SELL decisions live here.
        summary["action"] = self._try_get(step_state, "action")
        summary["reward"] = self._try_get(step_state, "reward")
        summary["status"] = self._try_get(step_state, "status")

        obs = self._try_get(step_state, "observation")
        if obs is not None:
            cash = self._deep_find(obs, self.CASH_KEYS)
            if cash is not None:
                summary["cash"] = cash

            day = self._deep_find(obs, self.DAY_KEYS)
            if day is not None:
                summary["day"] = day

            hands = self._deep_find(obs, self.HANDS_KEYS)
            if hands is not None:
                summary["hands"] = hands

            prices = self._deep_find(obs, self.PRICE_KEYS)
            if prices is not None:
                summary["prices"] = prices

            crops = self._deep_find(obs, self.CROP_KEYS)
            if crops is not None:
                summary["crops"] = crops

        return summary

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def log_match(
        self,
        env: Any,
        match_idx: int,
        agent_idx: int,
        opponent_name: str,
    ) -> Path:
        """Write a per-turn snapshot JSON for the given match."""
        turns: List[Dict[str, Any]] = []

        try:
            steps = list(env.steps)
        except Exception as exc:
            logger.error("Cannot access env.steps for match %d: %s", match_idx, exc)
            steps = []

        for turn_idx, step in enumerate(steps):
            turn_entry: Dict[str, Any] = {"turn": turn_idx}
            try:
                if isinstance(step, (list, tuple)):
                    for p_idx, p_state in enumerate(step):
                        turn_entry[f"player_{p_idx}"] = self._summarize_step_state(p_state)
                else:
                    turn_entry["raw"] = str(step)
            except Exception as exc:
                turn_entry["error"] = f"{type(exc).__name__}: {exc}"
            turns.append(turn_entry)

        out_path = self.log_dir / f"match_{match_idx:02d}_turns.json"
        payload = {
            "match_id": match_idx,
            "agent_index": agent_idx,
            "opponent": opponent_name,
            "total_turns": len(turns),
            "turns": turns,
        }

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
            logger.info("Saved turn log: %s", out_path)
        except OSError as exc:
            logger.error("Failed to save turn log %s: %s", out_path, exc)

        return out_path


# ==========================================================================
# NEW: Extended daily matrix writer
# ==========================================================================
class ExtendedDailyMatrixWriter:
    """Writes a supplementary daily-matrix text report per match.

    This is ADDITIVE. The main audit report generated by ``audit_logger.py``
    is not touched. This writer produces ``match_XX_extended.txt`` with extra
    columns so the user can inspect market prices, seed inventory, product
    inventory, and raw market actions per day.

    All extraction is best-effort and wrapped in try/except, so it can never
    crash the benchmark run even if the observation schema changes.
    """

    # Reuse TurnLogger key sets so the schema stays in sync
    CASH_KEYS: Sequence[str] = TurnLogger.CASH_KEYS
    DAY_KEYS: Sequence[str] = TurnLogger.DAY_KEYS
    SEED_KEYS: Sequence[str] = TurnLogger.SEED_KEYS

    # NOTE: we deliberately do NOT reuse TurnLogger.PRICE_KEYS here because
    # that tuple contains "market", which would resolve to the entire market
    # dict (inventory + prices) instead of just the price mapping.
    PRICE_KEYS: Sequence[str] = ("prices", "market_prices", "price")

    PRODUCT_KEYS: Sequence[str] = (
        "shed", "inventory", "products", "product_inventory",
        "harvest", "harvested", "storage",
    )

    # Fallback: how many env steps equal one in-game day
    STEPS_PER_DAY_FALLBACK: int = 24

    # Preferred display order and short labels for crop prices
    PRICE_ORDER: Sequence[Tuple[str, str]] = (
        ("WHEAT", "W"),
        ("CARROT", "C"),
        ("TOMATO", "T"),
        ("STRAWBERRY", "S"),
        ("MELON", "M"),
        ("FERTILIZER", "F"),
    )

    def __init__(self, log_dir: Union[str, Path]) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _fmt_money(v: Any) -> str:
        if v is None:
            return "-"
        try:
            return f"${float(v):,.2f}"
        except (TypeError, ValueError):
            return str(v)[:12]

    @staticmethod
    def _fmt_inv(v: Any) -> str:
        """Compact dict-of-crop -> qty representation, e.g. 'WHEAT:12'.

        Falls back to a short string if the value is not a dict.
        """
        if v is None:
            return "-"
        if isinstance(v, dict):
            parts = []
            for k, qty in v.items():
                try:
                    parts.append(f"{str(k)[:5]}:{int(qty)}")
                except (TypeError, ValueError):
                    parts.append(f"{str(k)[:5]}:?")
            return ",".join(parts) if parts else "{}"
        try:
            return f"{float(v):.0f}"
        except (TypeError, ValueError):
            return str(v)[:20]

    @classmethod
    def _fmt_prices(cls, v: Any) -> str:
        """Compact price string, e.g. 'W:25,C:35,T:60,S:120,M:250'.

        Uses short labels and a fixed order. Unknown keys are appended at
        the end so nothing is silently dropped.
        """
        if v is None:
            return "-"
        if not isinstance(v, dict):
            return str(v)[:30]

        seen = set()
        parts: List[str] = []
        for full_name, short in cls.PRICE_ORDER:
            if full_name in v:
                seen.add(full_name)
                try:
                    parts.append(f"{short}:{int(v[full_name])}")
                except (TypeError, ValueError):
                    parts.append(f"{short}:?")
        # Any extra keys not in PRICE_ORDER
        for k, val in v.items():
            if k in seen:
                continue
            try:
                parts.append(f"{str(k)[:3]}:{int(val)}")
            except (TypeError, ValueError):
                parts.append(f"{str(k)[:3]}:?")
        return ",".join(parts) if parts else "{}"

    @staticmethod
    def _safe_delta(a: Any, b: Any) -> Optional[float]:
        try:
            return float(a) - float(b)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _compact_action(a: Any, max_len: int = 60) -> str:
        """Compact representation of a market/farm action for logging."""
        try:
            if isinstance(a, dict):
                parts: List[str] = []
                for k, v in a.items():
                    if isinstance(v, (list, tuple)):
                        parts.append(f"{k}:{len(v)}")
                    else:
                        parts.append(str(k))
                s = "{" + ",".join(parts) + "}"
                return s[:max_len]
            return str(a)[:max_len]
        except Exception:
            return "?"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def write(
        self,
        env: Any,
        match_idx: int,
        agent_idx: int,
        opponent_name: str,
    ) -> Optional[Path]:
        """Write the extended daily matrix text file for the given match."""
        try:
            steps = list(env.steps)
        except Exception as exc:
            logger.error("Extended writer: cannot access env.steps: %s", exc)
            return None

        # --- Group steps into days ---
        day_buckets: Dict[Any, List[Any]] = {}
        for turn_idx, step in enumerate(steps):
            try:
                if isinstance(step, (list, tuple)) and len(step) > agent_idx:
                    state = step[agent_idx]
                else:
                    state = step

                obs = TurnLogger._try_get(state, "observation")
                day = TurnLogger._deep_find(obs, self.DAY_KEYS)

                if day is None:
                    day = turn_idx // self.STEPS_PER_DAY_FALLBACK

                day_buckets.setdefault(day, []).append(state)
            except Exception as exc:
                logger.debug("Extended writer: step %d skipped: %s", turn_idx, exc)

        # --- Compute per-day rows ---
        rows: List[Dict[str, Any]] = []
        for day in sorted(day_buckets.keys(), key=lambda d: (str(type(d)), d)):
            states = day_buckets[day]
            if not states:
                continue

            first = states[0]
            last = states[-1]
            first_obs = TurnLogger._try_get(first, "observation")
            last_obs = TurnLogger._try_get(last, "observation")

            open_cash = TurnLogger._deep_find(first_obs, self.CASH_KEYS)
            close_cash = TurnLogger._deep_find(last_obs, self.CASH_KEYS)
            seed_inv = TurnLogger._deep_find(last_obs, self.SEED_KEYS)
            prod_inv = TurnLogger._deep_find(last_obs, self.PRODUCT_KEYS)
            prices = TurnLogger._deep_find(last_obs, self.PRICE_KEYS)

            # Collect raw actions taken during the day
            actions: List[str] = []
            for s in states:
                a = TurnLogger._try_get(s, "action")
                if a:
                    actions.append(self._compact_action(a))

            rows.append({
                "day": day,
                "open_cash": open_cash,
                "close_cash": close_cash,
                "seed_inv": seed_inv,
                "prod_inv": prod_inv,
                "prices": prices,
                "actions": actions,
            })

        # --- Write text file ---
        out_path = self.log_dir / f"match_{match_idx:02d}_extended.txt"
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("=" * 160 + "\n")
                f.write(
                    f" EXTENDED DAILY MATRIX - MATCH {match_idx:02d} "
                    f"(player {agent_idx} vs {opponent_name})\n"
                )
                f.write("=" * 160 + "\n")
                f.write(
                    " Day | Open Cash   | Close Cash  | Cash Δ     | "
                    "Prices(W/C/T/S/M/F)                | "
                    "Seed Inv          | Product Inv       | Actions\n"
                )
                f.write("-" * 160 + "\n")

                for r in rows:
                    delta = self._safe_delta(r["close_cash"], r["open_cash"])
                    day_label = (
                        f"D{int(r['day']):02d}"
                        if isinstance(r["day"], (int, float))
                        else str(r["day"])[:4]
                    )

                    f.write(
                        f" {day_label:>3} | "
                        f"{self._fmt_money(r['open_cash']):>11} | "
                        f"{self._fmt_money(r['close_cash']):>11} | "
                        f"{self._fmt_money(delta):>10} | "
                        f"{self._fmt_prices(r['prices']):<33} | "
                        f"{self._fmt_inv(r['seed_inv']):<17} | "
                        f"{self._fmt_inv(r['prod_inv']):<17} | "
                        f"{' ; '.join(r['actions'])[:80]}\n"
                    )

                f.write("=" * 160 + "\n")
                f.write(
                    "NOTE:\n"
                    "  - 'Prices' shows end-of-day market prices. Format: W:25,C:35,T:60,S:120,M:250,F:100\n"
                    "  - 'Seed Inv' and 'Product Inv' are best-effort extractions from the observation.\n"
                    "  - If a column shows '-', the key was not found in the current observation schema.\n"
                )
                f.write("=" * 160 + "\n")

            logger.info("Saved extended daily matrix: %s", out_path)
            return out_path
        except OSError as exc:
            logger.error("Failed to save extended daily matrix: %s", exc)
            return None


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Immutable result metrics for a single match execution."""

    match_id: int
    agent_index: int
    agent_score: float
    opponent_score: float
    duration: float
    result_status: str  # "WIN", "LOSS", or "TIE"
    report_path: Path


@dataclass(slots=True)
class BenchmarkSummary:
    """Aggregated statistical metrics for a completed benchmark suite."""

    total_matches: int
    wins: int = 0
    losses: int = 0
    ties: int = 0
    agent_scores: List[float] = field(default_factory=list)
    opponent_scores: List[float] = field(default_factory=list)
    durations: List[float] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        """Calculates win rate percentage."""
        if self.total_matches == 0:
            return 0.0
        return (self.wins / self.total_matches) * 100.0

    @property
    def avg_agent_score(self) -> float:
        """Calculates mean agent reward score."""
        return statistics.mean(self.agent_scores) if self.agent_scores else 0.0

    @property
    def avg_opponent_score(self) -> float:
        """Calculates mean opponent reward score."""
        return statistics.mean(self.opponent_scores) if self.opponent_scores else 0.0

    @property
    def avg_duration(self) -> float:
        """Calculates mean match duration in seconds."""
        return statistics.mean(self.durations) if self.durations else 0.0


class MatchRunner:
    """Encapsulates Kaggle Environment match execution and score extraction."""

    DEFAULT_STEPS: int = 720

    def __init__(self, episode_steps: int = DEFAULT_STEPS, debug: bool = False) -> None:
        self.episode_steps = episode_steps
        self.debug = debug

    def run_match(
        self,
        match_idx: int,
        opponent: Union[str, Any],
        auditor: Auditor,
        turn_logger: Optional[TurnLogger] = None,                       # NEW (optional)
        extended_writer: Optional[ExtendedDailyMatrixWriter] = None,    # NEW (optional)
    ) -> Tuple[MatchResult, Dict[str, Any]]:
        """Executes a single environment match and audits the financial output."""
        start_time = time.perf_counter()

        # Alternate starting player order for fair evaluation
        if match_idx % 2 == 1:
            players = [my_agent, opponent]
            agent_idx = 0
        else:
            players = [opponent, my_agent]
            agent_idx = 1

        env = make("kaggriculture", configuration={"episodeSteps": self.episode_steps}, debug=self.debug)
        env.run(players)

        elapsed = time.perf_counter() - start_time

        opponent_name = str(opponent) if isinstance(opponent, str) else "Opponent"

        # ---- NEW: Write per-turn state log (wrapped so it can never crash the run) ----
        if turn_logger is not None:
            try:
                turn_logger.log_match(env, match_idx, agent_idx, opponent_name)
            except Exception as exc:
                logger.error("Turn logging failed for match %d: %s", match_idx, exc)

        # ---- NEW: Write extended daily matrix (also wrapped) ----
        if extended_writer is not None:
            try:
                extended_writer.write(env, match_idx, agent_idx, opponent_name)
            except Exception as exc:
                logger.error("Extended matrix writing failed for match %d: %s", match_idx, exc)

        # Run Transactional Financial Audit
        ledger = auditor.audit_environment(env, agent_idx, match_idx)
        report_path = auditor.generate_match_report(ledger, opponent_name)

        # Extract Final Reward Scores
        final_step = env.steps[-1]
        agent_score = float(final_step[agent_idx].get("reward") or 0.0)
        opp_score = float(final_step[1 - agent_idx].get("reward") or 0.0)

        # Determine Outcome Status
        if agent_score > opp_score:
            res_str = "WIN"
        elif agent_score < opp_score:
            res_str = "LOSS"
        else:
            res_str = "TIE"

        result = MatchResult(
            match_id=match_idx,
            agent_index=agent_idx,
            agent_score=agent_score,
            opponent_score=opp_score,
            duration=elapsed,
            result_status=res_str,
            report_path=report_path,
        )

        return result, env.toJSON()


class BenchmarkReporter:
    """Formats and displays console output and diagnostic reports."""

    @staticmethod
    def print_header(num_matches: int, opponent: str, log_dir: Path) -> None:
        """Prints benchmark suite execution header."""
        print("==================================================")
        print(f" Starting Financial Audit Benchmark ({num_matches} Matches)")
        print(f" Opponent Agent : '{opponent}'")
        print(f" Log Destination: {log_dir}/")
        print("==================================================")

    @staticmethod
    def print_match_result(result: MatchResult, total_matches: int) -> None:
        """Prints single match completion statistics."""
        print(
            f"Match {result.match_id:02d}/{total_matches:02d} | "
            f"Result: {result.result_status:<4} | "
            f"Score: ${result.agent_score:>8,.1f} | "
            f"Opponent: ${result.opponent_score:>8,.1f} | "
            f"Saved: {result.report_path.name}"
        )

    @staticmethod
    def print_summary(summary: BenchmarkSummary, log_dir: Path) -> None:
        """Prints comprehensive benchmark summary table."""
        print("\n==================================================")
        print(" BENCHMARK SUMMARY REPORT")
        print("==================================================")
        print(f"Total Matches Played : {summary.total_matches}")
        print(f"Win / Loss / Tie     : {summary.wins} W / {summary.losses} L / {summary.ties} T")
        print(f"Win Rate             : {summary.win_rate:.1f}%")
        print(f"Average Agent Money  : ${summary.avg_agent_score:,.2f}")
        print(f"Average Opponent     : ${summary.avg_opponent_score:,.2f}")
        print(f"Average Match Time   : {summary.avg_duration:.2f}s")
        print(f"Logs Directory       : {log_dir.resolve()}")
        print(f"Turn Logs            : match_XX_turns.json (per match)")
        print(f"Extended Matrices    : match_XX_extended.txt (per match)")
        print("==================================================")


class BenchmarkRunner:
    """Orchestrates benchmark runs, statistics gathering, and log export."""

    def __init__(self, base_log_dir: str = "logg", replay_export_path: str = "replay.json") -> None:
        self.base_log_dir = base_log_dir
        self.replay_export_path = Path(replay_export_path)

    def run(self, num_matches: int = 10, opponent: str = "starter") -> BenchmarkSummary:
        """Runs N-match benchmark suite and exports audit logs."""
        audit_folder = Auditor.create_unique_benchmark_dir(self.base_log_dir)
        auditor = Auditor(audit_folder)
        match_runner = MatchRunner()

        # NEW: instantiate loggers into the same folder as audit reports
        turn_logger = TurnLogger(audit_folder)
        extended_writer = ExtendedDailyMatrixWriter(audit_folder)

        BenchmarkReporter.print_header(num_matches, opponent, audit_folder)

        summary = BenchmarkSummary(total_matches=num_matches)
        last_replay: Optional[Dict[str, Any]] = None

        for match_idx in range(1, num_matches + 1):
            result, replay_json = match_runner.run_match(
                match_idx,
                opponent,
                auditor,
                turn_logger=turn_logger,
                extended_writer=extended_writer,
            )
            last_replay = replay_json

            # Record Statistics
            summary.agent_scores.append(result.agent_score)
            summary.opponent_scores.append(result.opponent_score)
            summary.durations.append(result.duration)

            if result.result_status == "WIN":
                summary.wins += 1
            elif result.result_status == "LOSS":
                summary.losses += 1
            else:
                summary.ties += 1

            BenchmarkReporter.print_match_result(result, num_matches)

        BenchmarkReporter.print_summary(summary, audit_folder)

        # Safely export last match replay for visualizer
        if last_replay:
            self._export_replay(last_replay)

        return summary

    def _export_replay(self, replay_data: Dict[str, Any]) -> None:
        """Exports the last match replay JSON safely."""
        try:
            with open(self.replay_export_path, "w", encoding="utf-8") as f:
                json.dump(replay_data, f, indent=2)
            logger.info("Exported final match replay to %s", self.replay_export_path)
        except OSError as exc:
            logger.error("Failed to save replay JSON: %s", exc)


def run_benchmark(num_matches: int = 10, opponent: str = "starter") -> None:
    """Public wrapper maintaining backward-compatibility with benchmark execution."""
    runner = BenchmarkRunner()
    runner.run(num_matches=num_matches, opponent=opponent)


if __name__ == "__main__":
    run_benchmark(num_matches= 50, opponent="starter")