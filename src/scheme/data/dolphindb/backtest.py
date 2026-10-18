"""保留 Runtime 回测 API，仅将已知回测模块内联到当前会话。"""

import re
from collections.abc import Mapping
from functools import wraps
from inspect import signature
from typing import Any

from runtime.apps.backtest import api as _runtime

_SIGNATURE = signature(_runtime.run_backtest)
_FUNCTIONS = frozenset({
    "getParams", "getParam", "getIndustry", "getTradeDates", "getHistoryData",
    "getLastData", "getMinuteHistory", "order_target", "order_target_value",
    "backtest_order_quote",
    "read_stock_snapshots", "build_minute_bars", "build_stock_snapshot_message",
    "snapshot_callback",
})
# 仅区分不可改写的文本与标识符，不解析或重新生成 DOS。
_TOKENS = re.compile(
    r"(?P<comment>//[^\r\n]*|/\*[\s\S]*?(?:\*/|\Z))"
    r'''|(?P<literal>"(?:\\[\s\S]|[^"\\])*(?:"|\Z)|'(?:\\[\s\S]|[^'\\])*(?:'|\Z)|`[\w:]*)'''
    r"|(?P<name>[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)"
    r"|(?P<space>\s+)|(?P<other>[\s\S])"
)


def _code_tokens(script: str) -> list[re.Match[str]]:
    return [
        token for token in _TOKENS.finditer(script)
        if token.lastgroup not in {"comment", "space"}
    ]


def _rewrite_script(script: str) -> str:
    tokens = _code_tokens(script)
    for index, token in enumerate(tokens):
        if token.lastgroup == "name" and token.group() == "use":
            module = tokens[index + 1].group() if index + 1 < len(tokens) else ""
            if module != "query" or tokens[index + 1].lastgroup != "name":
                raise ValueError(f"Scheme 回测仅允许 use query，不支持 use {module}")

    def replace(token: re.Match[str]) -> str:
        value = token.group()
        if token.lastgroup == "name" and value.startswith("backtest::"):
            name = value.removeprefix("backtest::")
            if name not in _FUNCTIONS:
                raise ValueError(f"Scheme 未内联回测函数：{value}")
            return name
        return value

    return _TOKENS.sub(replace, script)


class _InlineBacktestSession:
    """仅拦截 Runtime 独立 bootstrap；结果对象继续持有该代理。"""

    def __init__(self, session: Any) -> None:
        self.raw_session = session
        self._loaded = False

    def run(self, script: str, *args: Any, **kwargs: Any) -> Any:
        bootstrap = script.strip().removesuffix(";").rstrip()
        if bootstrap == "use factor":
            return None
        if bootstrap == "use backtest":
            if not self._loaded:
                result = self.raw_session.run("""
                    def getParams() {
                        /* 返回 run_backtest 上传的只读策略参数字典。 */
                        return objByName("coreBacktestParams")
                    }

                    def getParam(key) {
                        /* 按名称读取一个策略参数；参数不存在时立即报错。 */
                        if (form(key) != 0 || !(type(key) in [STRING, SYMBOL])) {
                            throw "策略参数 key 必须是标量 STRING 或 SYMBOL"
                        }
                        parameterKey = string(key)
                        if (isNull(parameterKey) || parameterKey == "") {
                            throw "策略参数 key 不能为空"
                        }
                        parameters = getParams()
                        if (!(parameterKey in string(parameters.keys()))) {
                            throw "策略参数不存在：" + parameterKey
                        }
                        return parameters[parameterKey]
                    }

                    def getIndustry() {
                        /* 返回与 Factor 研究同源、使用 XSHG/XSHE 代码的行业字典。 */
                        return objByName("coreBacktestCodeToIndustry")
                    }

                    def getTradeDates() {
                        /* 返回当前回测实际回放的有序交易日期。 */
                        return date(objByName("coreBacktestTradeDates"))
                    }

                    def getHistoryData(context, msg, filter=true, start=NULL, end=NULL) {
                        /*
                        返回严格早于当前 onSnapshot 消息日期的 DSL 数据。

                        filter=true 读取 filters 后的数据，false 读取 filters 前的数据。
                        两张结果表由 run_backtest 在当前 DolphinDB 会话中生成；msg 只用于确定
                        当前回调日期。start 和 end 是可选闭区间边界，不会放宽严格历史边界。
                        合成快照字段使用 timestamp。
                        */
                        source = objByName(
                            iif(
                                filter,
                                "coreBacktestFilteredData",
                                "coreBacktestComputedData"
                            )
                        )
                        current_date = date(msg.timestamp[0])
                        if (form(start) != 0) {
                            throw "start 必须是 DATE 兼容的标量或 NULL"
                        }
                        if (form(end) != 0) {
                            throw "end 必须是 DATE 兼容的标量或 NULL"
                        }
                        hasStart = !isNull(start)
                        hasEnd = !isNull(end)
                        normalizedStart = date()
                        normalizedEnd = date()
                        if (hasStart) {
                            normalizedStart = date(start)
                            if (isNull(normalizedStart)) {
                                throw "start 必须是 DATE 兼容的标量或 NULL"
                            }
                        }
                        if (hasEnd) {
                            normalizedEnd = date(end)
                            if (isNull(normalizedEnd)) {
                                throw "end 必须是 DATE 兼容的标量或 NULL"
                            }
                        }
                        if (hasStart && hasEnd) {
                            if (normalizedStart > normalizedEnd) {
                                throw "start 不能晚于 end"
                            }
                            return select *
                            from source
                            where date(time) < current_date,
                                  date(time) >= normalizedStart,
                                  date(time) <= normalizedEnd
                        }
                        if (hasStart) {
                            return select *
                            from source
                            where date(time) < current_date,
                                  date(time) >= normalizedStart
                        }
                        if (hasEnd) {
                            return select *
                            from source
                            where date(time) < current_date,
                                  date(time) <= normalizedEnd
                        }
                        return select *
                        from source
                        where date(time) < current_date
                    }

                    def getLastData(context, msg, filter=true) {
                        // 返回严格早于当前消息日期的最后一个实际存在的 DSL 截面。
                        history = getHistoryData(context, msg, filter)
                        if (history.rows() == 0) return history
                        last_time = max(history.time)
                        return select *
                        from history
                        where time == last_time
                    }

                    def read_stock_snapshots(source, day, codes, begin_time, end_time, history=false) {
                        shCodes = symbol(strReplace(string(codes[endsWith(string(codes), ".XSHG")]), ".XSHG", ""))
                        szCodes = symbol(strReplace(string(codes[endsWith(string(codes), ".XSHE")]), ".XSHE", ""))
                        if (history) {
                            return select Market, SecurityID, TradeDate, TradeTime, LastPrice, TotalVolume, TotalAmount
                            from source where TradeDate=day, TradeTime>=begin_time, TradeTime<end_time,
                                ((Market="SH" && SecurityID in shCodes) || (Market="SZ" && SecurityID in szCodes))
                            order by TradeTime, Market, SecurityID
                        }
                        return select * from source where TradeDate=day, TradeTime>=begin_time, TradeTime<end_time,
                            ((Market="SH" && SecurityID in shCodes) || (Market="SZ" && SecurityID in szCodes))
                        order by TradeTime, Market, SecurityID
                    }

                    def build_minute_bars(raw) {
                        ordered = select symbol(string(SecurityID)+iif(Market=="SH", ".XSHG", ".XSHE")) as code,
                            TradeDate, TradeTime, LastPrice, TotalVolume, TotalAmount
                            from raw order by code, TradeDate, TradeTime
                        // 首条累计数作为基线；不能把此前未知的成交全算到第一根分钟线。
                        update ordered set volume=long(nullFill(deltas(TotalVolume),0)), amount=double(nullFill(deltas(TotalAmount),0)) context by code, TradeDate
                        if (any(ordered.volume<0) || any(ordered.amount < -0.000001)) {
                            throw "快照累计成交量或成交额在同一交易日内倒退"
                        }
                        inSession = select * from ordered where
                            ((TradeTime>=09:30:00.000 && TradeTime<=11:30:00.000) || (TradeTime>=13:00:00.000 && TradeTime<=15:00:00.000))
                        ends = long(ceil(double(long(inSession.TradeTime))/60000.0))*60000
                        ends = iif(inSession.TradeTime==09:30:00.000 || inSession.TradeTime==13:00:00.000, ends+60000, ends)
                        update inSession set time=timestamp(TradeDate)+ends
                        // 无效 LastPrice 不产生 OHLC，但不能因此丢掉该记录的累计成交增量。
                        totals = select sum(volume) as volume, sum(amount) as amount from inSession group by time,code
                        prices = select first(LastPrice) as open, max(LastPrice) as high, min(LastPrice) as low,
                            last(LastPrice) as close from inSession
                            where LastPrice>0, isValid(LastPrice), !isNanInf(LastPrice,true) group by time,code
                        bars = lj(prices,totals,`time`code)
                        return select time,code,open,high,low,close,volume,amount from bars order by time,code
                    }

                    def getMinuteHistory(context, msg, codes, count) {
                        if (form(count)!=0 || isNull(count) || count<=0 || long(count)!=count) {
                            throw "count 必须为正整数"
                        }
                        state = objByName("coreBacktestSnapshotState")
                        if (!state["enabled"]) throw "getMinuteHistory 仅适用于真实快照模式"
                        requested = symbol(distinct(string(codes)))
                        if (size(requested)==0 || any(!(requested in state["codes"]))) {
                            throw "codes 必须是当前回测股票范围内的非空 XSHG/XSHE 代码"
                        }
                        requested.sort!()
                        now = msg.timestamp[0]
                        day = date(now)
                        cutoff = timestamp(day)+long(floor(double(long(time(now)))/60000.0))*60000
                        empty = table(1:0,`time`code`open`high`low`close`volume`amount,[TIMESTAMP,SYMBOL,DOUBLE,DOUBLE,DOUBLE,DOUBLE,LONG,DOUBLE])
                        result = empty
                        cache = state["history"]
                        days = state["historyDates"]
                        days = reverse(days[days<=day])
                        usedKeys = array(STRING,0)
                        for (historyDay in days) {
                            // 当日只查询已经结束的分钟，避免预计算未来数据及未来数据错误干扰过去回调。
                            boundary = iif(historyDay==day, time(cutoff)+1, 23:59:59.999)
                            // 开盘整点属于下一根分钟线，暂不读取，保留此前记录作为差分基线。
                            if (historyDay==day && (time(cutoff)==09:30:00.000 || time(cutoff)==13:00:00.000)) {
                                boundary = time(cutoff)
                            }
                            key = string(historyDay)
                            usedKeys.append!(key)
                            if (key in cache) {
                                dayCache = cache[key]
                            } else {
                                dayCache = dict(SYMBOL,ANY)
                            }
                            begins = array(TIME,0,size(requested))
                            for (stockCode in requested) {
                                if (stockCode in dayCache) {
                                    entry = dayCache[stockCode]
                                } else {
                                    entry = dict(STRING,ANY)
                                    entry["boundary"] = 00:00:00.000
                                    entry["bars"] = empty[0:0]
                                }
                                if (entry["boundary"]>boundary) {
                                    // 回看较早消息时只重置该证券，不影响其他证券的读取进度。
                                    entry["boundary"] = 00:00:00.000
                                    entry["bars"] = empty[0:0]
                                    entry.erase!("previous")
                                }
                                begins.append!(entry["boundary"])
                                dayCache[stockCode] = entry
                            }
                            // 每只证券维护自己的游标；游标相同的证券仍一次批量查询。
                            for (begin in distinct(begins[begins<boundary])) {
                                batchCodes = requested[begins==begin]
                                raw = read_stock_snapshots(state["source"],historyDay,batchCodes,begin,boundary,true)
                                for (stockCode in batchCodes) {
                                    entry = dayCache[stockCode]
                                    if ("previous" in entry) raw.append!(entry["previous"])
                                }
                                added = build_minute_bars(raw)
                                // previous 只提供差分基线，不重复追加它所属的旧分钟。
                                lower = timestamp(historyDay)+long(begin)
                                added = select * from added where time>=lower, time<timestamp(historyDay)+long(boundary)
                                previous = select top 1 * from raw context by Market,SecurityID csort TradeTime desc
                                for (stockCode in batchCodes) {
                                    entry = dayCache[stockCode]
                                    bars = entry["bars"]
                                    bars.append!(select * from added where code=stockCode)
                                    entry["bars"] = bars
                                    entry["previous"] = select * from previous where
                                        symbol(string(SecurityID)+iif(Market=="SH",".XSHG",".XSHE"))=stockCode
                                    entry["boundary"] = boundary
                                    dayCache[stockCode] = entry
                                }
                            }
                            cache[key] = dayCache
                            for (stockCode in requested) {
                                entry = dayCache[stockCode]
                                bars = entry["bars"]
                                result.append!(select * from bars where time<=cutoff)
                            }
                            counts = select size(time) as n from result group by code
                            if (counts.rows()==size(requested) && all(counts.n>=count)) break
                        }
                        for (key in keys(cache)) {
                            if (!(key in usedKeys)) {
                                dayCache = cache[key]
                                // 只回收本次请求不再需要的旧日，保留其他证券的历史窗口。
                                erase!(dayCache,requested)
                                if (size(keys(dayCache))==0) {
                                    cache.erase!(key)
                                } else {
                                    cache[key] = dayCache
                                }
                            }
                        }
                        state["history"] = cache
                        if ("peakHistoryBytes" in state) {
                            cacheBytes = 0l
                            for (key in keys(cache)) {
                                dayCache = cache[key]
                                for (stockCode in keys(dayCache)) {
                                    entry = dayCache[stockCode]
                                    cacheBytes += memSize(entry["bars"])
                                    if ("previous" in entry) cacheBytes += memSize(entry["previous"])
                                }
                            }
                            state["peakHistoryBytes"] = max([state["peakHistoryBytes"],cacheBytes])
                        }
                        ordered = select * from result order by code, time desc
                        update ordered set ordinal=cumcount(time) context by code
                        return select time,code,open,high,low,close,volume,amount from ordered where ordinal<=count order by time,code
                    }

                    def backtest_order_quote(msg, stockCode) {
                        if ("coreBacktestSnapshotState" in objs(true).name) {
                            state = objByName("coreBacktestSnapshotState")
                            if (state["enabled"]) {
                                quotes = state["quotes"]
                                key = symbol([string(stockCode)])[0]
                                if (!(key in quotes)) throw "该证券当日尚无可用快照："+string(stockCode)
                                quote = quotes[key]
                                if (date(quote[0])!=date(msg.timestamp[0]) || quote[0]>msg.timestamp[0]) {
                                    throw "该证券没有当前时刻可见的当日报价："+string(stockCode)
                                }
                                return quote
                            }
                        }
                        index = find(symbol(string(msg.symbol)), stockCode)
                        if (index>=msg.rows()) throw "股票不在当前快照中"
                        return [msg.timestamp[index],msg.lastPrice[index],msg.bidPrice[0][index],msg.offerPrice[0][index],msg.bidQty[0][index],msg.offerQty[0][index]]
                    }

                    def order_target(mutable context, msg, stockCode, targetAmount, orderLabel="order_target") {
                        /*
                        将指定证券的多头持仓调整到目标股数。

                        targetAmount 是精确目标股数，允许卖出不足一手的剩余持仓。买单使用
                        卖一价，卖单使用买一价，确保限价单能够参与盘口撮合。
                        */
                        quote = backtest_order_quote(msg, stockCode)
                        if (isNull(targetAmount) || targetAmount < 0) throw "targetAmount 必须是非负数"
                        normalizedTargetAmount = long(targetAmount)
                        if (double(normalizedTargetAmount) != double(targetAmount)) throw "targetAmount 必须是整数"

                        currentPosition = Backtest::getPosition(context.engine, stockCode)
                        currentAmount = long(nullFill(currentPosition.longPosition.sum(), 0))
                        difference = normalizedTargetAmount - currentAmount
                        quantity = long(abs(difference))
                        if (quantity == 0) return NULL

                        if (difference > 0) {
                            orderPrice = double(quote[3])
                            if (isNull(quote[5]) || quote[5]<=0) throw "快照卖一没有可成交数量："+string(stockCode)
                            direction = 1
                        } else {
                            orderPrice = double(quote[2])
                            if (isNull(quote[4]) || quote[4]<=0) throw "快照买一没有可成交数量："+string(stockCode)
                            direction = 3
                        }
                        if (isNull(orderPrice) || orderPrice <= 0) throw "快照一档价格无效"
                        return Backtest::submitOrder(
                            context.engine,
                            (stockCode, msg.timestamp[0], 5, orderPrice, quantity, direction),
                            orderLabel
                        )
                    }

                    def order_target_value(mutable context, msg, stockCode, targetValue, orderLabel="order_target_value") {
                        /* 使用当前快照 lastPrice 将目标市值换算成目标股数并下单。 */
                        quote = backtest_order_quote(msg, stockCode)
                        if (isNull(targetValue) || targetValue < 0) throw "targetValue 必须是非负数"
                        lotSize = 100l
                        lastPrice = double(quote[1])
                        if (isNull(lastPrice) || lastPrice <= 0) throw "快照 lastPrice 无效"

                        currentPosition = Backtest::getPosition(context.engine, stockCode)
                        currentAmount = long(nullFill(currentPosition.longPosition.sum(), 0))
                        difference = double(targetValue) / lastPrice - currentAmount
                        adjustment = long(floor(abs(difference) / lotSize)) * long(lotSize)
                        targetAmount = iif(
                            targetValue == 0,
                            long(0),
                            iif(difference < 0, currentAmount - adjustment, currentAmount + adjustment)
                        )
                        return order_target(
                            context,
                            msg,
                            stockCode,
                            targetAmount,
                            orderLabel
                        )
                    }

                    def build_stock_snapshot_message(raw, reference) {
                        valid = select * from raw where isValid(LastPrice), !isNanInf(LastPrice,true), LastPrice>0
                        n = valid.rows()
                        if (n==0) return table(1:0,
                            `symbol`symbolSource`timestamp`lastPrice`upLimitPrice`downLimitPrice`totalBidQty`totalOfferQty`bidPrice`bidQty`offerPrice`offerQty`prevClosePrice,
                            [SYMBOL,SYMBOL,TIMESTAMP,DOUBLE,DOUBLE,DOUBLE,LONG,LONG,DOUBLE[],LONG[],DOUBLE[],LONG[],DOUBLE])
                        code = symbol(string(valid.SecurityID) + iif(valid.Market=="SH", ".XSHG", ".XSHE"))
                        quotes = table(code as code, valid.TradeDate as day,
                            timestamp(valid.TradeDate)+long(valid.TradeTime) as timestamp, double(valid.LastPrice) as lastPrice)
                        daily = select code, date(time) as day, pre_close, up_limit, down_limit from reference
                        quotes = lj(quotes, daily, `code`day)
                        for (field in `pre_close`up_limit`down_limit) {
                            invalid = isNull(quotes[field]) || isNanInf(double(quotes[field]),true) || quotes[field]<=0
                            if (any(invalid)) {
                                row = find(invalid, true)
                                throw "真实快照缺少有效参考价格：" + string(quotes.code[row]) + " " + string(quotes.day[row]) + " " + string(field)
                            }
                        }
                        // flatten(matrix) 按列展开，转置后每只证券连续存放五档。
                        bidMatrix = matrix(valid.BidPrice1, valid.BidPrice2, valid.BidPrice3, valid.BidPrice4, valid.BidPrice5)
                        bidQtyMatrix = matrix(valid.BidVolume1, valid.BidVolume2, valid.BidVolume3, valid.BidVolume4, valid.BidVolume5)
                        offerMatrix = matrix(valid.AskPrice1, valid.AskPrice2, valid.AskPrice3, valid.AskPrice4, valid.AskPrice5)
                        offerQtyMatrix = matrix(valid.AskVolume1, valid.AskVolume2, valid.AskVolume3, valid.AskVolume4, valid.AskVolume5)
                        bidPrices = double(flatten(transpose(bidMatrix)))
                        bidQuantities = long(flatten(transpose(bidQtyMatrix)))
                        offerPrices = double(flatten(transpose(offerMatrix)))
                        offerQuantities = long(flatten(transpose(offerQtyMatrix)))
                        bidValid = isValid(bidPrices) && !isNanInf(bidPrices,true) && bidPrices>0 && isValid(bidQuantities) && bidQuantities>0
                        offerValid = isValid(offerPrices) && !isNanInf(offerPrices,true) && offerPrices>0 && isValid(offerQuantities) && offerQuantities>0
                        ends = int(1..n)*5
                        return table(quotes.code as symbol,
                            symbol(iif(endsWith(string(quotes.code), ".XSHG"), "XSHG", "XSHE")) as symbolSource,
                            quotes.timestamp as timestamp, quotes.lastPrice as lastPrice,
                            double(quotes.up_limit) as upLimitPrice, double(quotes.down_limit) as downLimitPrice,
                            take(0l,n) as totalBidQty, take(0l,n) as totalOfferQty,
                            arrayVector(ends,iif(bidValid,bidPrices,0.0)) as bidPrice,
                            arrayVector(ends,iif(bidValid,bidQuantities,0l)) as bidQty,
                            arrayVector(ends,iif(offerValid,offerPrices,0.0)) as offerPrice,
                            arrayVector(ends,iif(offerValid,offerQuantities,0l)) as offerQty,
                            double(quotes.pre_close) as prevClosePrice)
                    }

                    def snapshot_callback(mutable context, msg, indicator, callback) {
                        state = objByName("coreBacktestSnapshotState")
                        day = date(msg.timestamp[0])
                        quotes = state["quotes"]
                        if (isNull(state["quoteDate"]) || state["quoteDate"]!=day) {
                            erase!(quotes,keys(quotes))
                            state["quoteDate"] = day
                        }
                        for (index in 0:msg.rows()) {
                            code = msg.symbol[index]
                            if (code in state["codes"]) {
                                quotes[code] = [msg.timestamp[index],msg.lastPrice[index],msg.bidPrice[0][index],msg.offerPrice[0][index],msg.bidQty[0][index],msg.offerQty[0][index]]
                            }
                        }
                        state["quotes"] = quotes
                        callback(context,msg,indicator)
                    }
                """, *args, **kwargs)
                self._loaded = True
                return result
            return None

        # Runtime 的固定调用协议；只展开代码 token，不触碰注释、字符串或插件命名空间。
        # 倒序替换保留原脚本中数据准备、日期过滤及基准拼接的顺序。
        tokens = _code_tokens(script)
        for index in range(len(tokens) - 1, -1, -1):
            token = tokens[index]
            if token.lastgroup != "name":
                continue
            name = token.group()
            if name == "backtest::build_backtest_message":
                call = re.match(
                    r"backtest::build_backtest_message\(\s*"
                    r"(?P<data>coreBacktestData|coreBacktestBenchmarkData)\s*,\s*"
                    r"(?P<adj>coreBacktestAdj|NULL)\s*,\s*"
                    r"(?P<spread>coreBacktestSyntheticSpread|0\.0)\s*\)",
                    script[token.start():],
                )
                if (
                    call is None or index < 2 or tokens[index - 1].group() != "="
                    or not re.fullmatch(r"[A-Za-z_]\w*", tokens[index - 2].group())
                ):
                    raise ValueError("不支持的 Runtime build_backtest_message 调用")
                target = tokens[index - 2].group()
                script = script[:tokens[index - 2].start()] + f"""
                    market_data = {call["data"]}
                    adj = {call["adj"]}
                    synthetic_spread = {call["spread"]}
                    /*
                    从完整股票范围的未筛选日线构造开盘、收盘单档合成快照。

                    每个交易日生成 09:30 和 15:00 两条快照，盘口只有一档，买卖价格分别
                    使用当日 open 和 close。盘口数量固定为十亿股/份，作为不会触发插件
                    整数溢出的近似无限流动性；盘口撮合比例固定为 100%，不使用当天结束后
                    才能确定的成交量。
                    adj 为 hfq 或 qfq 时使用
                    adj_factor 复权价格。synthetic_spread 表示合成买卖盘口的完整相对价差，
                    买一和卖一分别位于 lastPrice 的下方和上方一半价差处。
                    */
                    if (!isNull(adj) && !(adj in ["hfq", "qfq"])) {{
                        throw "adj 只能是 hfq、qfq 或 NULL"
                    }}
                    if (isNull(synthetic_spread) || synthetic_spread < 0 || synthetic_spread >= 1) {{
                        throw "synthetic_spread 必须位于 [0, 1)"
                    }}
                    adjustment = take(1.0, market_data.rows())
                    if (!isNull(adj)) {{
                        if (!("adj_factor" in columnNames(market_data))) {{
                            throw "复权行情缺少 adj_factor"
                        }}
                        adjustment = double(market_data.adj_factor)
                        if (adj == "qfq") {{
                            adjustmentHistory = select code, time, adj_factor
                                from market_data order by code, time
                            adjustmentBasis = select last(adj_factor) as basis
                                from adjustmentHistory group by code
                            basisByCode = dict(string(adjustmentBasis.code), double(adjustmentBasis.basis))
                            adjustment = adjustment / basisByCode[string(market_data.code)]
                        }}
                    }}

                    daily = select
                        code,
                        time,
                        open,
                        low,
                        high,
                        close,
                        double(
                            iif(
                                isNull(up_limit),
                                iif(
                                    high > round(pre_close * 1.1, 3),
                                    high,
                                    round(pre_close * 1.1, 3)
                                ),
                                up_limit
                            )
                        ) as upLimitPrice,
                        double(
                            iif(
                                isNull(down_limit),
                                iif(
                                    low < round(pre_close * 0.9, 3),
                                    low,
                                    round(pre_close * 0.9, 3)
                                ),
                                down_limit
                            )
                        ) as downLimitPrice,
                        pre_close as prevClosePrice
                    from market_data

                    for (column in `open`low`high`close`upLimitPrice`downLimitPrice`prevClosePrice) {{
                        values = double(daily[column])
                        if (!isNull(adj)) values = values * adjustment
                        replaceColumn!(daily, column, values)
                    }}
                    valid_rows = take(true, daily.rows())
                    for (column in `open`low`high`close`upLimitPrice`downLimitPrice`prevClosePrice) {{
                        valid_rows = valid_rows && !isNull(daily[column])
                    }}
                    daily = daily[valid_rows]
                    if (daily.rows() == 0) throw "无法从日线构造合成快照"

                    open_snapshot = select
                        code,
                        temporalAdd(temporalAdd(timestamp(time), 9, "h"), 30, "m") as timestamp,
                        open as lastPrice,
                        upLimitPrice,
                        downLimitPrice,
                        prevClosePrice
                    from daily
                    close_snapshot = select
                        code,
                        temporalAdd(timestamp(time), 15, "h") as timestamp,
                        close as lastPrice,
                        upLimitPrice,
                        downLimitPrice,
                        prevClosePrice
                    from daily
                    levels = unionAll(open_snapshot, close_snapshot)
                    levels.sortBy!(`timestamp`code)
                    ends = int(1..levels.rows())
                    unlimited_quantity = take(1000000000l, levels.rows())
                    codes = string(levels.code)
                    if (any(!(endsWith(codes, ".XSHE") || endsWith(codes, ".XSHG")))) {{
                        throw "回测证券代码必须使用 XSHG/XSHE 格式"
                    }}
                    message = table(
                        symbol(codes) as symbol,
                        symbol(iif(endsWith(codes, ".XSHE"), "XSHE", "XSHG")) as symbolSource,
                        levels.timestamp as timestamp,
                        double(levels.lastPrice) as lastPrice,
                        double(levels.upLimitPrice) as upLimitPrice,
                        double(levels.downLimitPrice) as downLimitPrice,
                        unlimited_quantity as totalBidQty,
                        unlimited_quantity as totalOfferQty,
                        arrayVector(ends, round(double(levels.lastPrice) * (1.0 - synthetic_spread / 2.0), 3)) as bidPrice,
                        arrayVector(ends, unlimited_quantity) as bidQty,
                        arrayVector(ends, round(double(levels.lastPrice) * (1.0 + synthetic_spread / 2.0), 3)) as offerPrice,
                        arrayVector(ends, unlimited_quantity) as offerQty,
                        double(levels.prevClosePrice) as prevClosePrice
                    )
                    {target} = message
                """ + script[token.start() + call.end():]
            elif name in {"backtest::run_backtest", "backtest::run_snapshot_backtest"}:
                snapshot = name == "backtest::run_snapshot_backtest"
                arguments = (
                    r"coreBacktestSnapshotTable\s*,\s*coreBacktestSnapshotReference\s*,\s*"
                    r"coreBacktestCodes\s*,\s*coreBacktestTradeDates\s*,\s*"
                    r"coreBacktestBenchmarkMessage"
                    if snapshot else r"(?P<message>[A-Za-z_]\w*)"
                )
                call = re.match(
                    re.escape(name) + r"\(\s*coreBacktestName\s*,\s*coreBacktestConfig\s*,\s*"
                    + arguments + r"\s*,\s*initialize\s*,\s*beforeTrading\s*,\s*onBar\s*,\s*"
                    r"onSnapshot\s*,\s*onOrder\s*,\s*onTrade\s*,\s*afterTrading\s*,\s*finalize\s*\)",
                    script[token.start():],
                )
                if (
                    call is None or index < 2 or tokens[index - 1].group() != "="
                    or tokens[index - 2].group() != "coreBacktestEngine"
                ):
                    raise ValueError("不支持的 Runtime 回放调用")
                if snapshot:
                    script = script[:tokens[index - 2].start()] + """
                        name = coreBacktestName
                        config = coreBacktestConfig
                        initialize_callback = initialize
                        before_trading_callback = beforeTrading
                        on_bar_callback = onBar
                        on_snapshot_callback = onSnapshot
                        on_order_callback = onOrder
                        on_trade_callback = onTrade
                        after_trading_callback = afterTrading
                        finalize_callback = finalize
                        source = coreBacktestSnapshotTable
                        reference = coreBacktestSnapshotReference
                        codes = coreBacktestCodes
                        days = coreBacktestTradeDates
                        benchmark_message = coreBacktestBenchmarkMessage
                        chunk_minutes = 5
                        if (size(days)==0) throw "真实快照回测区间没有交易日"
                        if (chunk_minutes<=0 || chunk_minutes>1440 || int(chunk_minutes)!=chunk_minutes) throw "回放块必须为 1 至 1440 个整分钟"
                        state = objByName("coreBacktestSnapshotState")
                        state["enabled"] = false
                        // 只聚合每天的覆盖与时间边界，不加载整段原始快照。缺失日期必须在
                        // 创建引擎、执行任意用户回调前一次性报出，回放时复用这些边界。
                        shCodes = symbol(strReplace(string(codes[endsWith(string(codes), ".XSHG")]), ".XSHG", ""))
                        szCodes = symbol(strReplace(string(codes[endsWith(string(codes), ".XSHE")]), ".XSHE", ""))
                        coverage = select min(TradeTime) as firstTime,max(TradeTime) as lastTime,
                            sum(isValid(LastPrice) && !isNanInf(LastPrice,true) && LastPrice>0) as validRows
                            from source where TradeDate in days,
                                ((Market="SH" && SecurityID in shCodes) || (Market="SZ" && SecurityID in szCodes))
                            group by TradeDate
                        missingDays = days[!(days in coverage.TradeDate)]
                        invalidDays = exec TradeDate from coverage where validRows=0 order by TradeDate
                        problems = array(STRING,0)
                        if (size(missingDays)>0) problems.append!("真实快照缺少交易日行情："+concat(string(missingDays),", "))
                        if (size(invalidDays)>0) problems.append!("真实快照缺少有效交易日行情："+concat(string(invalidDays),", "))
                        if (size(problems)>0) throw concat(problems,"；")
                        state["enabled"] = true
                        state["source"] = source
                        state["codes"] = symbol(codes)
                        state["quoteDate"] = date()
                        state["quotes"] = dict(SYMBOL,ANY)
                        state["history"] = dict(STRING,ANY)
                        state["peakHistoryBytes"] = 0l
                        diagnostics = table(1:0,`day`raw_rows`quote_rows`load_ms`replay_ms`peak_block_bytes`peak_history_bytes,[DATE,LONG,LONG,LONG,LONG,LONG,LONG])
                        state["diagnostics"] = diagnostics
                        state["historyDates"] = sort(distinct(date(reference.time)))
                        wrapped = snapshot_callback{,,,on_snapshot_callback}
                        config["dataType"] = 1
                        config["matchingMode"] = 1
                        config["frequency"] = 0
                        config["callbackForSnapshot"] = 0
                        config["msgAsPiecesOnSnapshot"] = true
                        config["matchingRatio"] = 0.0
                        config["orderBookMatchingRatio"] = 1.0
                        int_config_names = [
                            "dataType",
                            "matchingMode",
                            "frequency",
                            "latency",
                            "callbackForSnapshot",
                            "outputQueuePosition"
                        ]
                        for (config_name in int_config_names) {
                            if (config_name in config) {
                                config[config_name] = int(config[config_name])
                            }
                        }

                        engine = Backtest::createBacktestEngine(
                            name,
                            config,
                            ,
                            initialize_callback,
                            before_trading_callback,
                            on_bar_callback,
                            wrapped,
                            on_order_callback,
                            on_trade_callback,
                            after_trading_callback,
                            finalize_callback
                        )
                        try {
                            for (day in days) {
                                started = now()
                                rawCount = 0l
                                quoteCount = 0l
                                loadMs = 0l
                                replayMs = 0l
                                peakBlockBytes = 0l
                                state["peakHistoryBytes"] = 0l
                                dayReference = select code,time,pre_close,up_limit,down_limit from reference where date(time)=day
                                bounds = select firstTime,lastTime from coverage where TradeDate=day
                                benchmarkDay = select * from benchmark_message where date(timestamp)=day
                                firstMs = long(bounds.firstTime[0])
                                lastMs = long(bounds.lastTime[0])
                                if (benchmarkDay.rows()>0) {
                                    firstMs = min([firstMs,long(min(time(benchmarkDay.timestamp)))])
                                    lastMs = max([lastMs,long(max(time(benchmarkDay.timestamp)))])
                                }
                                blockMs = long(chunk_minutes)*60000
                                firstBlock = long(floor(double(firstMs)/blockMs))
                                lastBlock = long(floor(double(lastMs)/blockMs))
                                for (block in firstBlock..lastBlock) {
                                    fromMs = block*blockMs
                                    toMs = min([fromMs+long(chunk_minutes)*60000,86399999l])
                                    loadedAt = now()
                                    raw = read_stock_snapshots(source,day,codes,time(fromMs),time(toMs))
                                    loadMs += long(now()-loadedAt)
                                    rawCount += raw.rows()
                                    if (raw.rows()>0) {
                                        message = build_stock_snapshot_message(raw,dayReference)
                                    } else {
                                        message = benchmark_message[0:0]
                                    }
                                    quoteCount += message.rows()
                                    benchmark = select * from benchmarkDay where time(timestamp)>=time(fromMs),time(timestamp)<time(toMs)
                                    if (benchmark.rows()>0) message = unionAll(message,benchmark)
                                    if (message.rows()>0) {
                                        message.sortBy!(`timestamp`symbol)
                                        peakBlockBytes = max([peakBlockBytes,memSize(raw)+memSize(message)])
                                        replayAt = now()
                                        Backtest::appendQuotationMsg(engine,message)
                                        replayMs += long(now()-replayAt)
                                        lastMessage = message[(message.rows()-1):message.rows()]
                                    }
                                }
                                print("真实快照 "+string(day)+"：读取 "+string(rawCount)+" 条，回放 "+string(quoteCount)+" 条，剔除无效价格 "+string(rawCount-quoteCount)+" 条，耗时 "+string(now()-started)+"ms")
                                diagnostics.append!(table(day as day,rawCount as raw_rows,quoteCount as quote_rows,loadMs as load_ms,replayMs as replay_ms,peakBlockBytes as peak_block_bytes,state["peakHistoryBytes"] as peak_history_bytes))
                                state["diagnostics"] = diagnostics
                                print("按日性能：加载 "+string(loadMs)+"ms，回放 "+string(replayMs)+"ms，原始/适配块峰值 "+string(peakBlockBytes)+"B，分钟缓存峰值 "+string(state["peakHistoryBytes"])+"B")
                                if (quoteCount==0) throw "真实快照缺少有效交易日行情："+string(day)+"；不会自动缩短请求区间"
                            }
                            update lastMessage set symbol="END",timestamp=timestamp+1
                            Backtest::appendQuotationMsg(engine,lastMessage)
                        } catch (error) {
                            Backtest::dropBacktestEngine(engine)
                            state["enabled"] = false
                            throw error
                        }
                        coreBacktestEngine = engine
                    """ + script[token.start() + call.end():]
                else:
                    script = script[:tokens[index - 2].start()] + f"""
                        name = coreBacktestName
                        config = coreBacktestConfig
                        initialize_callback = initialize
                        before_trading_callback = beforeTrading
                        on_bar_callback = onBar
                        on_snapshot_callback = onSnapshot
                        on_order_callback = onOrder
                        on_trade_callback = onTrade
                        after_trading_callback = afterTrading
                        finalize_callback = finalize
                        message = {call["message"]}
                        /* 日线合成快照的原有回放入口。 */
                        if (message.rows() == 0) {{
                            throw "DSL 构造的回测 msg 表为空"
                        }}
                        config["dataType"] = 1
                        config["matchingMode"] = 1
                        config["frequency"] = 0
                        config["callbackForSnapshot"] = 0
                        config["msgAsPiecesOnSnapshot"] = true
                        config["matchingRatio"] = 0.0
                        config["orderBookMatchingRatio"] = 1.0
                        int_config_names = [
                            "dataType",
                            "matchingMode",
                            "frequency",
                            "latency",
                            "callbackForSnapshot",
                            "outputQueuePosition"
                        ]
                        for (config_name in int_config_names) {{
                            if (config_name in config) {{
                                config[config_name] = int(config[config_name])
                            }}
                        }}

                        engine = Backtest::createBacktestEngine(
                            name,
                            config,
                            ,
                            initialize_callback,
                            before_trading_callback,
                            on_bar_callback,
                            on_snapshot_callback,
                            on_order_callback,
                            on_trade_callback,
                            after_trading_callback,
                            finalize_callback
                        )
                        try {{
                            Backtest::appendQuotationMsg(engine, message)
                            // Flush the last timestamp batch using the snapshot protocol. END
                            // is a control message, not a quote and cannot provide liquidity.
                            lastRow = imax(message.timestamp)
                            endMessage = message[lastRow:(lastRow + 1)]
                            update endMessage set symbol="END", timestamp=timestamp+1
                            Backtest::appendQuotationMsg(engine, endMessage)
                        }} catch (error) {{
                            Backtest::dropBacktestEngine(engine)
                            throw error
                        }}
                        coreBacktestEngine = engine
                    """ + script[token.start() + call.end():]
            elif name == "backtest::standardize_return_summary":
                call = re.match(
                    r"backtest::standardize_return_summary\(\s*"
                    r"Backtest::getReturnSummary\(coreBacktestEngine\)\s*,\s*"
                    r"Backtest::getDailyTotalPortfolios\(coreBacktestEngine\)\s*,\s*"
                    r"coreBacktestAnnualTradingDays\s*,\s*coreBacktestRiskFreeRate\s*\)",
                    script[token.start():],
                )
                if call is None:
                    raise ValueError("不支持的 Runtime 收益汇总调用")
                script = script[:token.start()] + """
                    summary = Backtest::getReturnSummary(coreBacktestEngine)
                    daily_portfolios = Backtest::getDailyTotalPortfolios(coreBacktestEngine)
                    annual_trading_days = coreBacktestAnnualTradingDays
                    risk_free_rate = coreBacktestRiskFreeRate
                    result = summary
                    if (summary.rows() > 0 && daily_portfolios.rows() > 0) {
                        values = exec double(netValue) from daily_portfolios
                            where !isNull(netValue) order by tradeDate
                        if (size(values) > 0) {
                            result = select * from summary
                            trading_days = int(annual_trading_days)
                            annual_return = double(NULL)
                            if (last(values) > 0) {
                                annual_return = pow(
                                    last(values),
                                    double(trading_days) / size(values)
                                ) - 1
                            }

                            daily_returns = deltas(values) / prev(values)
                            daily_returns = daily_returns[!isNull(daily_returns)]
                            annual_volatility = double(NULL)
                            if (size(daily_returns) > 1) {
                                annual_volatility = stdp(daily_returns) * sqrt(trading_days)
                            }

                            sharpe_ratio = double(NULL)
                            if (
                                !isNull(annual_return) &&
                                !isNull(annual_volatility) &&
                                annual_volatility != 0
                            ) {
                                sharpe_ratio =
                                    (annual_return - double(risk_free_rate)) /
                                    annual_volatility
                            }

                            column_names = result.colNames()
                            row_count = result.rows()
                            if ("annualReturn" in column_names) {
                                replaceColumn!(
                                    result,
                                    `annualReturn,
                                    take(annual_return, row_count)
                                )
                            }
                            if ("annualVolatility" in column_names) {
                                replaceColumn!(
                                    result,
                                    `annualVolatility,
                                    take(annual_volatility, row_count)
                                )
                            }
                            if ("sharpeRatio" in column_names) {
                                replaceColumn!(
                                    result,
                                    `sharpeRatio,
                                    take(sharpe_ratio, row_count)
                                )
                            }
                            if ("drawdownRatio" in column_names) {
                                drawdown_ratio = take(double(NULL), row_count)
                                for (index in 0:row_count) {
                                    drawdown = result.maxDrawdown[index]
                                    if (
                                        !isNull(annual_return) &&
                                        !isNull(drawdown) &&
                                        drawdown != 0
                                    ) {
                                        drawdown_ratio[index] = annual_return / drawdown
                                    }
                                }
                                replaceColumn!(
                                    result,
                                    `drawdownRatio,
                                    drawdown_ratio
                                )
                            }
                        }
                    }
                    result
                """ + script[token.start() + call.end():]
        return self.raw_session.run(_rewrite_script(script), *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.raw_session, name)


@wraps(_runtime.run_backtest)
def backtest(*args: Any, **kwargs: Any) -> Any:
    bound = _SIGNATURE.bind(*args, **kwargs)
    # 在 Runtime 拼接函数源码前校验，不能把用户的 use factor 当作 bootstrap。
    callbacks = bound.arguments["callbacks"]
    scripts = list(callbacks.values()) if isinstance(callbacks, Mapping) else []
    scripts.append(bound.arguments.get("utils", ""))
    for script in scripts:
        if isinstance(script, str):
            _rewrite_script(script)

    session = bound.arguments.get("session")
    owns_session = session is None
    if owns_session:
        session = _runtime.create_session(max_time=_runtime.BACKTEST_SESSION_MAX_TIME)
    proxy = session if isinstance(session, _InlineBacktestSession) else _InlineBacktestSession(session)
    bound.arguments["session"] = proxy
    try:
        return _runtime.run_backtest(*bound.args, **bound.kwargs)
    except Exception:
        # Runtime 收到显式 session 后不再负责其异常生命周期。
        if owns_session:
            session.close()
        raise


__all__ = ["backtest"]
