"""Automated Benchmark Suite for Kaggriculture Agent Execution."""

import json
import statistics
import time
from typing import Any, Dict, List
from kaggle_environments import make
from main import agent as my_agent


def run_benchmark(num_matches: int = 10, opponent: str = "starter") -> None:
    """Runs N-match simulation against target opponent and computes performance metrics."""
    print(f"==================================================")
    print(f" Starting Benchmark: {num_matches} Matches vs '{opponent}'")
    print(f"==================================================")

    wins = 0
    losses = 0
    ties = 0
    agent_rewards: List[float] = []
    opp_rewards: List[float] = []
    durations: List[float] = []

    last_replay: Dict[str, Any] = {}

    for match_idx in range(1, num_matches + 1):
        start_time = time.time()

        # Alternate starting player order for fairness
        if match_idx % 2 == 1:
            players = [my_agent, opponent]
            agent_idx = 0
        else:
            players = [opponent, my_agent]
            agent_idx = 1

        env = make("kaggriculture", configuration={"episodeSteps": 720}, debug=False)
        env.run(players)

        elapsed = time.time() - start_time
        durations.append(elapsed)

        final_step = env.steps[-1]
        agent_score = final_step[agent_idx]["reward"] if final_step[agent_idx]["reward"] is not None else 0
        opp_score = final_step[1 - agent_idx]["reward"] if final_step[1 - agent_idx]["reward"] is not None else 0

        agent_rewards.append(agent_score)
        opp_rewards.append(opp_score)

        if agent_score > opp_score:
            wins += 1
            res_str = "WIN"
        elif agent_score < opp_score:
            losses += 1
            res_str = "LOSS"
        else:
            ties += 1
            res_str = "TIE"

        print(
            f"Match {match_idx:02d}/{num_matches:02d} | Result: {res_str:<4} | "
            f"Agent Score: {agent_score:>8.1f} | Opponent Score: {opp_score:>8.1f} | "
            f"Time: {elapsed:.2f}s"
        )

        # Save the last match state for visualizer analysis
        if match_idx == num_matches:
            last_replay = env.toJSON()

    avg_agent = statistics.mean(agent_rewards)
    avg_opp = statistics.mean(opp_rewards)
    win_rate = (wins / num_matches) * 100

    print("\n==================================================")
    print(" BENCHMARK SUMMARY REPORT")
    print("==================================================")
    print(f"Total Matches Played : {num_matches}")
    print(f"Win / Loss / Tie     : {wins} W / {losses} L / {ties} T")
    print(f"Win Rate             : {win_rate:.1f}%")
    print(f"Average Agent Money  : ${avg_agent:,.2f}")
    print(f"Average Opponent     : ${avg_opp:,.2f}")
    print(f"Average Match Time   : {statistics.mean(durations):.2f}s")
    print("==================================================")

    # Export last game replay for analysis in visualizer.py
    with open("replay.json", "w", encoding="utf-8") as f:
        json.dump(last_replay, f, indent=2)
    print("Exported last match replay to 'replay.json' for analysis.")


if __name__ == "__main__":
    run_benchmark(num_matches=10, opponent="starter")