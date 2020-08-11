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


wti = Contract(symbol='CL', exchange='NYMEX',localSymbol='CLZ0',secType='FUT')
ib.qualifyContracts(wti)

ib.reqMarketDataType(4)

[ticker] = ib.reqTickers(wti)
wtiValue = ticker.marketPrice()
wtiValue

chains = ib.reqSecDefOptParams(wti.symbol, wti.exchange, wti.secType, wti.conId)

chains_df = util.df(chains)

#look only at the first of these, monthlies
chain = next(c for c in chains if c.tradingClass == 'LO' )

strikes = [strike for strike in chain.strikes
        if strike % 5 == 0
        and wtiValue - 10 < strike < wtiValue + 10]

expirations = sorted(expir for expir in chain.expirations)[0:2]  #first 2 expiry only
rights = ['P', 'C']

#note FuturesOption not Option contract type (secType='FOP')
contracts = [FuturesOption('CL', expirations[0], strike, right, 'NYMEX')
        for right in rights
        for strike in strikes]

contracts = ib.qualifyContracts(*contracts)
len(contracts)

tickers = ib.reqTickers(*contracts)

tickers[0]   #despite the error that no live data is available, we still get delayed data
tickers[1]   #but it looks like we only get the first strike
tickers[6]
tickers = ib.reqTickers(contracts[3]) #that seems to work aftr a pause

#what if we try request each ticker consecutively?


#the ib_insynch notebooks recommend using delayed data where no subscription:
ib.reqMarketDataType(4)
ib.disconnect()