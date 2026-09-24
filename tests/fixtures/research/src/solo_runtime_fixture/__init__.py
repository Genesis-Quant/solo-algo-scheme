from datetime import date

import pandas as pd
from pydantic import BaseModel

from scheme import Algo, Direction, DosVar, FactorParams, MarketOrder, ResearchContext
from scheme import Factor as BaseFactor


class Params(BaseModel):
    quantity: int = 100


class Context(ResearchContext[float]):
    sent: bool = False


class ModelAlgo(Algo[Params, Context]):
    def on_snapshot(self, msg: DosVar) -> None:
        if not self.ctx.sent:
            symbol = self.backtest.session.run(f"exec first(symbol) from {msg}")
            self.backtest.submit_order(
                MarketOrder(
                    symbol=symbol,
                    time=pd.Timestamp(self.backtest.time).to_pydatetime(),
                    direction=Direction.BUY_OPEN,
                    quantity=self.params.quantity,
                )
            )
            self.ctx.sent = True


class Factor(BaseFactor[FactorParams]):
    def compute(self, start: date, end: date) -> pd.DataFrame:
        dates = pd.bdate_range(start, pd.Timestamp(end) - pd.Timedelta(days=1))
        index = pd.MultiIndex.from_product(
            [dates, ["000001.XSHE", "600000.XSHG"]], names=["date", "symbol"]
        )
        return pd.DataFrame({"score": [1.0, 2.0] * len(dates)}, index=index)
