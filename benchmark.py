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
        self, match_idx: int, opponent: Union[str, Any], auditor: Auditor
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

        # Run Transactional Financial Audit
        ledger = auditor.audit_environment(env, agent_idx, match_idx)
        opponent_name = str(opponent) if isinstance(opponent, str) else "Opponent"
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

        BenchmarkReporter.print_header(num_matches, opponent, audit_folder)

        summary = BenchmarkSummary(total_matches=num_matches)
        last_replay: Optional[Dict[str, Any]] = None

        for match_idx in range(1, num_matches + 1):
            result, replay_json = match_runner.run_match(match_idx, opponent, auditor)
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
    run_benchmark(num_matches=10, opponent="starter")