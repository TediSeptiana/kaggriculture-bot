"""Transaction-Level Financial Audit & Corporate Accounting System for Kaggriculture.

This module inspects raw step executions from Kaggle Environments to build exact
financial statements (Cash Flow Statement, Income Statement, Balance Sheet,
and Crop Performance Analysis).
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, TypeAlias

# Set up module logger
logger = logging.getLogger(__name__)

# Domain Type Aliases
Money: TypeAlias = float
ItemCount: TypeAlias = int


# ============================================================================
# Static Specifications
# ============================================================================

@dataclass(frozen=True, slots=True)
class CropSpecAudit:
    """Static crop specifications for accounting valuation."""

    seed_cost: float
    first_yield_day: int
    max_yield_day: int
    product_name: str


CROP_SPECS_AUDIT: Dict[str, CropSpecAudit] = {
    "WHEAT": CropSpecAudit(10.0, 2, 4, "WHEAT"),
    "CARROT": CropSpecAudit(20.0, 2, 3, "CARROT"),
    "TOMATO": CropSpecAudit(50.0, 8, 11, "TOMATO"),
    "STRAWBERRY": CropSpecAudit(100.0, 10, 16, "STRAWBERRY"),
    "MELON": CropSpecAudit(80.0, 10, 10, "MELON"),
}

_DEFAULT_CROP_SPEC = CropSpecAudit(10.0, 2, 4, "WHEAT")


# ============================================================================
# Ledger Dataclasses
# ============================================================================

@dataclass
class CropPerformanceLedger:
    """Tracks itemized financial performance per crop type."""

    crop_name: str
    seeds_bought: ItemCount = 0
    seed_expenditure: Money = 0.0
    units_sold: ItemCount = 0
    sales_revenue: Money = 0.0

    @property
    def gross_profit(self) -> Money:
        """Calculates Gross Profit = Revenue - Direct Seed Cost."""
        return self.sales_revenue - self.seed_expenditure

    @property
    def roi_percentage(self) -> float:
        """Calculates Return on Investment (ROI%)."""
        if self.seed_expenditure <= 0.0:
            return 0.0
        return (self.gross_profit / self.seed_expenditure) * 100.0

    @property
    def avg_sell_price(self) -> Money:
        """Calculates average realized price per unit sold."""
        if self.units_sold <= 0:
            return 0.0
        return self.sales_revenue / float(self.units_sold)


@dataclass
class DailyFinancialSnapshot:
    """Snapshot of daily cash flow movements and balance sheet totals."""

    day: int
    opening_cash: Money
    sales_revenue: Money = 0.0
    seed_cogs: Money = 0.0
    labor_opex: Money = 0.0
    land_capex: Money = 0.0
    closing_cash: Money = 0.0
    net_wealth: Money = 0.0
    shed_items_count: ItemCount = 0
    seed_items_count: ItemCount = 0


@dataclass
class FinancialLedger:
    """Complete financial ledger for a single match."""

    match_id: int
    agent_index: int
    starting_capital: Money = 3000.0
    ending_cash: Money = 0.0
    total_gross_revenue: Money = 0.0
    total_seed_cogs: Money = 0.0
    total_labor_opex: Money = 0.0
    total_land_capex: Money = 0.0
    unsold_inventory_value: Money = 0.0
    unsold_seed_value: Money = 0.0
    crop_breakdown: Dict[str, CropPerformanceLedger] = field(default_factory=dict)
    daily_snapshots: List[DailyFinancialSnapshot] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.crop_breakdown:
            for crop in CROP_SPECS_AUDIT:
                self.crop_breakdown[crop] = CropPerformanceLedger(crop_name=crop)


# ============================================================================
# Transactional Auditor
# ============================================================================

class TransactionalAuditor:
    """Engine that performs transaction-level auditing on Kaggle Environment replays."""

    # Fibonacci pricing multiplier base for HIRE (kept as-is to preserve original audit output)
    _HIRE_BASE_COST: Money = 100.0

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Directory helper
    # ------------------------------------------------------------------ #

    @classmethod
    def create_unique_benchmark_dir(cls, base_dir: str = "logg") -> Path:
        """Creates a uniquely named directory for audit outputs."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        unique_hash = hashlib.md5(f"{time.time()}".encode()).hexdigest()[:8]
        dir_name = f"benchmark_{timestamp}_{unique_hash}"
        folder_path = Path(base_dir) / dir_name
        folder_path.mkdir(parents=True, exist_ok=True)
        return folder_path

    # ------------------------------------------------------------------ #
    # Main audit entry-point
    # ------------------------------------------------------------------ #

    def audit_environment(
        self, env: Any, agent_index: int, match_number: int
    ) -> FinancialLedger:
        """Parses exact step transactions to compile an accurate Financial Ledger."""
        ledger = FinancialLedger(match_id=match_number, agent_index=agent_index)

        state = _AuditState(
            prev_cash=ledger.starting_capital,
            snapshot=DailyFinancialSnapshot(
                day=0, opening_cash=ledger.starting_capital
            ),
            prev_quads_count=1,
            prev_seeds_count={},
            prev_shed_count={},
        )

        for step_data in env.steps:
            if not isinstance(step_data, list) or len(step_data) <= agent_index:
                continue

            agent_step = step_data[agent_index]
            obs = agent_step.get("observation", {})
            action_data = agent_step.get("action", {})

            farms = obs.get("farms", [{}, {}])
            if not farms or len(farms) <= agent_index:
                continue

            my_farm = farms[agent_index]
            private = obs.get("private", {})
            market = obs.get("market", {})
            market_prices = market.get("prices", {})

            current_cash = float(my_farm.get("money", state.prev_cash))
            current_day = int(obs.get("day", 0))
            unlocked_quads = len(my_farm.get("unlocked_quadrants", ["NW"]))

            current_seeds = dict(private.get("seeds", {}))
            current_shed = dict(private.get("shed", {}))

            # --- Day rollover: flush previous snapshot and start a new one ---
            if current_day != state.snapshot.day:
                state.snapshot.closing_cash = state.prev_cash
                state.snapshot.net_wealth = (
                    state.prev_cash
                    + self._calculate_shed_value(state.prev_shed_count, market_prices)
                    + self._calculate_seed_value(state.prev_seeds_count)
                )
                ledger.daily_snapshots.append(state.snapshot)
                state.snapshot = DailyFinancialSnapshot(
                    day=current_day, opening_cash=current_cash
                )

            # --- Process explicit market orders for this turn ---
            market_orders = (
                action_data.get("market", [])
                if isinstance(action_data, dict)
                else []
            )

            for order in market_orders:
                self._process_market_order(
                    order=order,
                    ledger=ledger,
                    snapshot=state.snapshot,
                    market_prices=market_prices,
                    unlocked_quads=unlocked_quads,
                    hires_today=int(my_farm.get("hires_today", 1)),
                )

            # --- Fallback delta tracking if market actions are not in step log ---
            if not market_orders:
                self._apply_fallback_delta(
                    ledger=ledger,
                    snapshot=state.snapshot,
                    current_cash=current_cash,
                    prev_cash=state.prev_cash,
                    unlocked_quads=unlocked_quads,
                    prev_quads_count=state.prev_quads_count,
                )

            # --- Advance state ---
            state.prev_cash = current_cash
            state.prev_quads_count = unlocked_quads
            state.prev_seeds_count = current_seeds
            state.prev_shed_count = current_shed

        # --- Final day snapshot ---
        self._finalize_final_snapshot(env, ledger, state, agent_index)
        return ledger

    # ------------------------------------------------------------------ #
    # Order processing helpers
    # ------------------------------------------------------------------ #

    def _process_market_order(
        self,
        order: Any,
        ledger: FinancialLedger,
        snapshot: DailyFinancialSnapshot,
        market_prices: Dict[str, float],
        unlocked_quads: int,
        hires_today: int,
    ) -> None:
        """Dispatch a single market order to the appropriate handler."""
        if not isinstance(order, list) or not order:
            return

        order_type = str(order[0]).upper()

        if order_type == "BUY_SEED" and len(order) >= 3:
            self._handle_buy_seed(order, ledger, snapshot)
        elif order_type == "SELL" and len(order) >= 3:
            self._handle_sell(order, ledger, snapshot, market_prices)
        elif order_type == "BUY_LAND":
            self._handle_buy_land(ledger, snapshot, unlocked_quads)
        elif order_type == "HIRE":
            self._handle_hire(ledger, snapshot, hires_today)

    def _handle_buy_seed(
        self,
        order: list,
        ledger: FinancialLedger,
        snapshot: DailyFinancialSnapshot,
    ) -> None:
        crop = str(order[1]).upper()
        qty = int(order[2])
        spec = CROP_SPECS_AUDIT.get(crop, CROP_SPECS_AUDIT["WHEAT"])
        cost = spec.seed_cost * qty

        ledger.total_seed_cogs += cost
        snapshot.seed_cogs += cost

        if crop in ledger.crop_breakdown:
            ledger.crop_breakdown[crop].seeds_bought += qty
            ledger.crop_breakdown[crop].seed_expenditure += cost

    def _handle_sell(
        self,
        order: list,
        ledger: FinancialLedger,
        snapshot: DailyFinancialSnapshot,
        market_prices: Dict[str, float],
    ) -> None:
        item = str(order[1]).upper()
        qty = int(order[2])
        unit_price = float(market_prices.get(item, 0.0))
        revenue = unit_price * qty

        ledger.total_gross_revenue += revenue
        snapshot.sales_revenue += revenue

        if item in ledger.crop_breakdown:
            ledger.crop_breakdown[item].units_sold += qty
            ledger.crop_breakdown[item].sales_revenue += revenue

    def _handle_buy_land(
        self,
        ledger: FinancialLedger,
        snapshot: DailyFinancialSnapshot,
        unlocked_quads: int,
    ) -> None:
        # NOTE: preserved as-is from original implementation.
        land_cost = (
            1000.0
            if unlocked_quads == 2
            else (2000.0 if unlocked_quads == 3 else 4000.0)
        )
        ledger.total_land_capex += land_cost
        snapshot.land_capex += land_cost

    def _handle_hire(
        self,
        ledger: FinancialLedger,
        snapshot: DailyFinancialSnapshot,
        hires_today: int,
    ) -> None:
        # NOTE: preserved as-is from original implementation.
        from market import get_fibonacci_cost
        hire_cost = get_fibonacci_cost(hires_today, self._HIRE_BASE_COST)
        ledger.total_labor_opex += hire_cost
        snapshot.labor_opex += hire_cost

    def _apply_fallback_delta(
        self,
        ledger: FinancialLedger,
        snapshot: DailyFinancialSnapshot,
        current_cash: Money,
        prev_cash: Money,
        unlocked_quads: int,
        prev_quads_count: int,
    ) -> None:
        """Fallback: infer revenue/COGS from cash deltas when market log is missing."""
        cash_delta = current_cash - prev_cash
        if cash_delta > 0:
            ledger.total_gross_revenue += cash_delta
            snapshot.sales_revenue += cash_delta
        elif cash_delta < 0 and unlocked_quads == prev_quads_count:
            ledger.total_seed_cogs += abs(cash_delta)
            snapshot.seed_cogs += abs(cash_delta)

    # ------------------------------------------------------------------ #
    # Final snapshot finalization
    # ------------------------------------------------------------------ #

    def _finalize_final_snapshot(
        self,
        env: Any,
        ledger: FinancialLedger,
        state: "_AuditState",
        agent_index: int,
    ) -> None:
        snapshot = state.snapshot
        snapshot.closing_cash = state.prev_cash
        snapshot.shed_items_count = sum(state.prev_shed_count.values())
        snapshot.seed_items_count = sum(state.prev_seeds_count.values())

        last_obs = env.steps[-1][agent_index].get("observation", {})
        last_market = last_obs.get("market", {}).get("prices", {})

        ledger.ending_cash = state.prev_cash
        ledger.unsold_inventory_value = self._calculate_shed_value(
            state.prev_shed_count, last_market
        )
        ledger.unsold_seed_value = self._calculate_seed_value(
            state.prev_seeds_count
        )

        snapshot.net_wealth = (
            ledger.ending_cash
            + ledger.unsold_inventory_value
            + ledger.unsold_seed_value
        )
        ledger.daily_snapshots.append(snapshot)

    # ------------------------------------------------------------------ #
    # Valuation helpers
    # ------------------------------------------------------------------ #

    def _calculate_shed_value(
        self, shed: Dict[str, int], market_prices: Dict[str, float]
    ) -> Money:
        total_val = 0.0
        for item, count in shed.items():
            fallback_spec = CROP_SPECS_AUDIT.get(item, _DEFAULT_CROP_SPEC)
            price = float(
                market_prices.get(item, fallback_spec.seed_cost * 2)
            )
            total_val += price * count
        return total_val

    def _calculate_seed_value(self, seeds: Dict[str, int]) -> Money:
        total_val = 0.0
        for seed_type, count in seeds.items():
            cost = CROP_SPECS_AUDIT.get(seed_type, _DEFAULT_CROP_SPEC).seed_cost
            total_val += cost * count
        return total_val

    # ------------------------------------------------------------------ #
    # Report generation
    # ------------------------------------------------------------------ #

    def generate_match_report(
        self, ledger: FinancialLedger, opponent_name: str
    ) -> Path:
        """Writes audit statement with cash flow and crop breakdown to match_XX.txt."""
        file_path = self.output_dir / f"match_{ledger.match_id:02d}.txt"

        gross_profit = ledger.total_gross_revenue - ledger.total_seed_cogs
        operating_cash_flow = gross_profit - ledger.total_labor_opex
        net_wealth = (
            ledger.ending_cash
            + ledger.unsold_inventory_value
            + ledger.unsold_seed_value
        )
        net_profit = net_wealth - ledger.starting_capital
        overall_roi = (net_profit / ledger.starting_capital) * 100.0

        lines: List[str] = []
        lines.extend(self._build_header(ledger, opponent_name))
        lines.extend(
            self._build_cash_flow_section(
                ledger, gross_profit, operating_cash_flow
            )
        )
        lines.extend(self._build_wealth_section(ledger, net_wealth, net_profit, overall_roi))
        lines.extend(self._build_crop_section(ledger))
        lines.extend(self._build_diagnostics_section(ledger, net_wealth))
        lines.extend(self._build_daily_matrix_section(ledger))
        lines.append(
            "=========================================================================================="
        )

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return file_path

    # ---- Report section builders -------------------------------------- #

    def _build_header(
        self, ledger: FinancialLedger, opponent_name: str
    ) -> List[str]:
        return [
            "==========================================================================================",
            f"          TRANSACTIONAL AUDIT REPORT - MATCH {ledger.match_id:02d} vs '{opponent_name.upper()}'",
            "==========================================================================================",
            f"Match ID               : match_{ledger.match_id:02d}",
            f"Player Index           : Player {ledger.agent_index}",
            "",
        ]

    def _build_cash_flow_section(
        self,
        ledger: FinancialLedger,
        gross_profit: Money,
        operating_cash_flow: Money,
    ) -> List[str]:
        return [
            "1. STATEMENT OF CASH FLOWS & INCOME STATEMENT (LAPORAN ARUS KAS / LABA RUGI)",
            "------------------------------------------------------------------------------------------",
            f"  Initial Capital (Opening Cash)          : ${ledger.starting_capital:14,.2f}",
            f"  (+) Gross Sales Revenue (Penjualan)     : ${ledger.total_gross_revenue:14,.2f}",
            f"  (-) Cost of Goods Sold (Seed Purchases) : ${ledger.total_seed_cogs:14,.2f}",
            "  ----------------------------------------------------------------------------------------",
            f"  (=) GROSS PROFIT (Laba Kotor)           : ${gross_profit:14,.2f}",
            f"  (-) Labor OpEx (Biaya HIRE Farm Hand)   : ${ledger.total_labor_opex:14,.2f}",
            "  ----------------------------------------------------------------------------------------",
            f"  (=) OPERATING CASH FLOW (CFO)           : ${operating_cash_flow:14,.2f}",
            f"  (-) CapEx (BUY_LAND Investment)         : ${ledger.total_land_capex:14,.2f}",
            "  ----------------------------------------------------------------------------------------",
            f"  (=) CLOSING CASH BALANCE (Kas Akhir)    : ${ledger.ending_cash:14,.2f}",
            "",
        ]

    def _build_wealth_section(
        self,
        ledger: FinancialLedger,
        net_wealth: Money,
        net_profit: Money,
        overall_roi: float,
    ) -> List[str]:
        return [
            "2. ASSET & NET WEALTH EVALUATION (EVALUASI KEKAYAAN BERSIH)",
            "------------------------------------------------------------------------------------------",
            f"  Ending Cash Balance                     : ${ledger.ending_cash:14,.2f}",
            f"  Unsold Product Inventory Value (Shed)   : ${ledger.unsold_inventory_value:14,.2f}",
            f"  Unsold Seed Inventory Value             : ${ledger.unsold_seed_value:14,.2f}",
            "  ----------------------------------------------------------------------------------------",
            f"  TOTAL FINAL NET WEALTH (Kekayaan Akhir) : ${net_wealth:14,.2f}",
            f"  NET PROFIT                              : ${net_profit:14,.2f}",
            f"  OVERALL ROI                             : {overall_roi:14.2f}%",
            "",
        ]

    def _build_crop_section(self, ledger: FinancialLedger) -> List[str]:
        lines = [
            "3. CROP PERFORMANCE & CAPITAL ALLOCATION BREAKDOWN",
            "------------------------------------------------------------------------------------------",
            "  Crop Name  | Seeds Bought | Seed Cost ($) | Units Sold | Revenue ($) | Gross Profit | ROI %",
            "  ----------------------------------------------------------------------------------------",
        ]
        for crop, perf in ledger.crop_breakdown.items():
            lines.append(
                f"  {crop:<10} | {perf.seeds_bought:>12d} | ${perf.seed_expenditure:>11,.2f} | "
                f"{perf.units_sold:>10d} | ${perf.sales_revenue:>10,.2f} | "
                f"${perf.gross_profit:>11,.2f} | {perf.roi_percentage:>6.1f}%"
            )
        return lines

    def _build_diagnostics_section(
        self, ledger: FinancialLedger, net_wealth: Money
    ) -> List[str]:
        lines = [
            "",
            "4. FINANCIAL LEAKAGE & OPPORTUNITY COST DIAGNOSTICS",
            "------------------------------------------------------------------------------------------",
        ]

        if ledger.total_land_capex > 0 and net_wealth < 6000.0:
            lines.append("  [!] WARNING: CapEx Over-expansion Detected!")
            lines.append(
                f"      Land Purchases total ${ledger.total_land_capex:,.2f} "
                f"while final Net Wealth is low (${net_wealth:,.2f})."
            )
            lines.append(
                "      Recommendation: Delay BUY_LAND until active quadrant utilization exceeds 85%."
            )

        if ledger.total_labor_opex > 500.0:
            lines.append("  [!] WARNING: Excessive Labor Expenditure (OpEx Drag)!")
            lines.append(
                f"      Total HIRE expense is ${ledger.total_labor_opex:,.2f}."
            )
            lines.append(
                "      Recommendation: Restrict hiring to turns where unwatered crops/weeds > 6."
            )

        if ledger.unsold_inventory_value > 500.0:
            lines.append("  [!] WARNING: High Unrealized Liquidity (Inventory Drag)!")
            lines.append(
                f"      ${ledger.unsold_inventory_value:,.2f} worth of produce remains unsold in shed."
            )
            lines.append(
                "      Recommendation: Execute batch SELL orders starting from Day 26 to maximize final cash."
            )

        return lines

    def _build_daily_matrix_section(self, ledger: FinancialLedger) -> List[str]:
        lines = [
            "",
            "5. DAILY CASH FLOW & WEALTH PROGRESSION MATRIX",
            "------------------------------------------------------------------------------------------",
            "  Day | Open Cash   | Sales ($)  | Seed COGS  | Labor OpEx | Land CapEx | Close Cash  | Net Wealth",
            "  ----------------------------------------------------------------------------------------",
        ]
        for snap in ledger.daily_snapshots:
            lines.append(
                f"  D{snap.day:02d} | ${snap.opening_cash:>9,.2f} | ${snap.sales_revenue:>9,.2f} | "
                f"${snap.seed_cogs:>9,.2f} | ${snap.labor_opex:>9,.2f} | ${snap.land_capex:>9,.2f} | "
                f"${snap.closing_cash:>10,.2f} | ${snap.net_wealth:>9,.2f}"
            )
        return lines


# ============================================================================
# Internal State Container
# ============================================================================

@dataclass
class _AuditState:
    """Mutable accumulator used during a single audit pass."""

    prev_cash: Money
    snapshot: DailyFinancialSnapshot
    prev_quads_count: int
    prev_seeds_count: Dict[str, int]
    prev_shed_count: Dict[str, int]