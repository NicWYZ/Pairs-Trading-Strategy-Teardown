"""
Transaction cost model: proportional costs in basis points of notional.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """
    Proportional transaction costs in basis points of traded notional.

    commission_bps : broker commission per side (bps of notional)
    slippage_bps   : half-spread + slippage per side (bps of notional)

    These are the dials the teardown turns: re-running under optimistic vs.
    pessimistic costs is how you show whether an edge is real or illusory.
    """

    commission_bps: float = 1.0
    slippage_bps: float = 5.0

    @property
    def rate(self) -> float:
        """
        Total cost as a fraction of traded notional (per side).
        """
        return (self.commission_bps + self.slippage_bps) / 1e4

    def cost(self, traded_notional):
        return self.rate * traded_notional
