"""Transaction-Level Financial Audit & Corporate Accounting System for Kaggriculture.

This module inspects raw step executions from Kaggle Environments to build exact
financial statements (Cash Flow Statement, Income Statement, Balance Sheet,
and Crop Performance Analysis).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypeAlias

# Set up module logger
logger = logging.getLogger(__name__)

# Domain Type Aliases
Money: TypeAlias = float
ItemCount: TypeAlias = int


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


class TransactionalAuditor:
    """Engine that performs transaction-level auditing on Kaggle Environment replays."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create_unique_benchmark_dir(cls, base_dir: str = "logg") -> Path:
        """Creates a uniquely named directory for audit outputs."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        unique_hash = hashlib.md5(f"{time.time()}".encode()).hexdigest()[:8]
        dir_name = f"benchmark_{timestamp}_{unique_hash}"
        folder_path = Path(base_dir) / dir_name
        folder_path.mkdir(parents=True, exist_ok=True)
        return folder_path

    def audit_environment(
        self, env: Any, agent_index: int, match_number: int
    ) -> FinancialLedger:
        """Parses exact step transactions to compile an accurate Financial Ledger."""
        ledger = FinancialLedger(match_id=match_number, agent_index=agent_index)

        prev_cash = ledger.starting_capital
        current_day_snapshot = DailyFinancialSnapshot(day=0, opening_cash=prev_cash)

        prev_quads_count = 1
        prev_seeds_count: Dict[str, int] = {}
        prev_shed_count: Dict[str, int] = {}

        for step_no, step_data in enumerate(env.steps):
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

            current_cash = float(my_farm.get("money", prev_cash))
            current_day = int(obs.get("day", 0))
            current_hour = int(obs.get("hour", 0))
            unlocked_quads = len(my_farm.get("unlocked_quadrants", ["NW"]))

            current_seeds = dict(private.get("seeds", {}))
            current_shed = dict(private.get("shed", {}))

            # Detect Day Rollover for Daily Cash Flow Reporting
            if current_day != current_day_snapshot.day:
                current_day_snapshot.closing_cash = prev_cash
                current_day_snapshot.net_wealth = (
                    prev_cash
                    + self._calculate_shed_value(prev_shed_count, market_prices)
                    + self._calculate_seed_value(prev_seeds_count)
                )
                ledger.daily_snapshots.append(current_day_snapshot)
                current_day_snapshot = DailyFinancialSnapshot(
                    day=current_day, opening_cash=current_cash
                )

            # --- 1. Transaction-Level Extraction from Market Actions ---
            market_orders = action_data.get("market", []) if isinstance(action_data, dict) else []

            for order in market_orders:
                if not isinstance(order, list) or not order:
                    continue

                order_type = str(order[0]).upper()

                if order_type == "BUY_SEED" and len(order) >= 3:
                    crop = str(order[1]).upper()
                    qty = int(order[2])
                    cost = CROP_SPECS_AUDIT.get(crop, CROP_SPECS_AUDIT["WHEAT"]).seed_cost * qty

                    ledger.total_seed_cogs += cost
                    current_day_snapshot.seed_cogs += cost

                    if crop in ledger.crop_breakdown:
                        ledger.crop_breakdown[crop].seeds_bought += qty
                        ledger.crop_breakdown[crop].seed_expenditure += cost

                elif order_type == "SELL" and len(order) >= 3:
                    item = str(order[1]).upper()
                    qty = int(order[2])
                    unit_price = float(market_prices.get(item, 0.0))
                    revenue = unit_price * qty

                    ledger.total_gross_revenue += revenue
                    current_day_snapshot.sales_revenue += revenue

                    if item in ledger.crop_breakdown:
                        ledger.crop_breakdown[item].units_sold += qty
                        ledger.crop_breakdown[item].sales_revenue += revenue

                elif order_type == "BUY_LAND":
                    # Land Expansion CapEx Calculation
                    land_cost = 1000.0 if unlocked_quads == 2 else (2000.0 if unlocked_quads == 3 else 4000.0)
                    ledger.total_land_capex += land_cost
                    current_day_snapshot.land_capex += land_cost

                elif order_type == "HIRE":
                    # Hire Labor OpEx Calculation
                    hires_today = int(my_farm.get("hires_today", 1))
                    # Fibonacci sequence pricing base cost
                    hire_cost = 100.0 * hires_today
                    ledger.total_labor_opex += hire_cost
                    current_day_snapshot.labor_opex += hire_cost

            # Fallback delta tracking if market actions are not explicitly in step log
            if not market_orders:
                cash_delta = current_cash - prev_cash
                if cash_delta > 0:
                    ledger.total_gross_revenue += cash_delta
                    current_day_snapshot.sales_revenue += cash_delta
                elif cash_delta < 0 and unlocked_quads == prev_quads_count:
                    ledger.total_seed_cogs += abs(cash_delta)
                    current_day_snapshot.seed_cogs += abs(cash_delta)

            prev_cash = current_cash
            prev_quads_count = unlocked_quads
            prev_seeds_count = current_seeds
            prev_shed_count = current_shed

        # Record Final Day Snapshot
        current_day_snapshot.closing_cash = prev_cash
        current_day_snapshot.shed_items_count = sum(prev_shed_count.values())
        current_day_snapshot.seed_items_count = sum(prev_seeds_count.values())

        last_obs = env.steps[-1][agent_index].get("observation", {})
        last_market = last_obs.get("market", {}).get("prices", {})

        ledger.ending_cash = prev_cash
        ledger.unsold_inventory_value = self._calculate_shed_value(prev_shed_count, last_market)
        ledger.unsold_seed_value = self._calculate_seed_value(prev_seeds_count)

        current_day_snapshot.net_wealth = (
            ledger.ending_cash
            + ledger.unsold_inventory_value
            + ledger.unsold_seed_value
        )
        ledger.daily_snapshots.append(current_day_snapshot)

        return ledger

    def _calculate_shed_value(self, shed: Dict[str, int], market_prices: Dict[str, float]) -> Money:
        total_val = 0.0
        for item, count in shed.items():
            price = float(market_prices.get(item, CROP_SPECS_AUDIT.get(item, CropSpecAudit(10.0, 2, 4, item)).seed_cost * 2))
            total_val += price * count
        return total_val

    def _calculate_seed_value(self, seeds: Dict[str, int]) -> Money:
        total_val = 0.0
        for seed_type, count in seeds.items():
            cost = CROP_SPECS_AUDIT.get(seed_type, CROP_SPECS_AUDIT["WHEAT"]).seed_cost
            total_val += cost * count
        return total_val

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

        lines = [
            "==========================================================================================",
            f"          TRANSACTIONAL AUDIT REPORT - MATCH {ledger.match_id:02d} vs '{opponent_name.upper()}'",
            "==========================================================================================",
            f"Match ID               : match_{ledger.match_id:02d}",
            f"Player Index           : Player {ledger.agent_index}",
            "",
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

        lines.extend([
            "",
            "4. FINANCIAL LEAKAGE & OPPORTUNITY COST DIAGNOSTICS",
            "------------------------------------------------------------------------------------------",
        ])

        # Diagnostic Rules based on audited numbers
        if ledger.total_land_capex > 0 and net_wealth < 6000.0:
            lines.append("  [!] WARNING: CapEx Over-expansion Detected!")
            lines.append(f"      Land Purchases total ${ledger.total_land_capex:,.2f} while final Net Wealth is low (${net_wealth:,.2f}).")
            lines.append("      Recommendation: Delay BUY_LAND until active quadrant utilization exceeds 85%.")

        if ledger.total_labor_opex > 500.0:
            lines.append("  [!] WARNING: Excessive Labor Expenditure (OpEx Drag)!")
            lines.append(f"      Total HIRE expense is ${ledger.total_labor_opex:,.2f}.")
            lines.append("      Recommendation: Restrict hiring to turns where unwatered crops/weeds > 6.")

        if ledger.unsold_inventory_value > 500.0:
            lines.append("  [!] WARNING: High Unrealized Liquidity (Inventory Drag)!")
            lines.append(f"      ${ledger.unsold_inventory_value:,.2f} worth of produce remains unsold in shed.")
            lines.append("      Recommendation: Execute batch SELL orders starting from Day 26 to maximize final cash.")

        lines.extend([
            "",
            "5. DAILY CASH FLOW & WEALTH PROGRESSION MATRIX",
            "------------------------------------------------------------------------------------------",
            "  Day | Open Cash   | Sales ($)  | Seed COGS  | Labor OpEx | Land CapEx | Close Cash  | Net Wealth",
            "  ----------------------------------------------------------------------------------------",
        ])

        for snap in ledger.daily_snapshots:
            lines.append(
                f"  D{snap.day:02d} | ${snap.opening_cash:>9,.2f} | ${snap.sales_revenue:>9,.2f} | "
                f"${snap.seed_cogs:>9,.2f} | ${snap.labor_opex:>9,.2f} | ${snap.land_capex:>9,.2f} | "
                f"${snap.closing_cash:>10,.2f} | ${snap.net_wealth:>9,.2f}"
            )

        lines.append("==========================================================================================")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return file_path