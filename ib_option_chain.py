# -*- coding: utf-8 -*-
"""
based on notebook in /Python/ib_insync/notebooks/

searching for the methods in the ib api is useful to understand their operation and paraemeters
"""

from ib_insync import *
util.startLoop()

ib = IB()
IBGateway_code = 4001
TWS_code = 7497

ib.connect('127.0.0.1', TWS_code, clientId=12)


spx = Index('SPX', 'CBOE')
ib.qualifyContracts(spx)

ib.reqMarketDataType(4)

[ticker] = ib.reqTickers(spx)
spxValue = ticker.marketPrice()
spxValue

chains = ib.reqSecDefOptParams(spx.symbol, '', spx.secType, spx.conId)

chains_df = util.df(chains)

#look only at the first of these, SMART exchange, monthlies
chain = next(c for c in chains if c.tradingClass == 'SPX' and c.exchange == 'SMART')

strikes = [strike for strike in chain.strikes
        if strike % 20 == 0
        and spxValue - 20 < strike < spxValue + 20]

expirations = sorted(expir for expir in chain.expirations)[0:2]  #first 2 expiry only
rights = ['P', 'C']

contracts = [Option('SPX', expiration, strike, right, 'SMART', tradingClass='SPX')
        for right in rights
        for expiration in expirations
        for strike in strikes]

contracts = ib.qualifyContracts(*contracts)
len(contracts)

tickers = ib.reqTickers(*contracts)
tickers[0]

#the ib_insynch notebooks recommend using delayed data where no subscription:
ib.reqMarketDataType(4)
ib.disconnect()