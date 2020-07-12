# -*- coding: utf-8 -*-
"""
https://github.com/erdewit/ib_insync
and in order for Spyder or iPython to use asyncio, I installed nest_asyncio:
https://github.com/erdewit/nest_asyncio

(CTL .  will restart kernel if you have already connected 
to gateway and connection refused)

"""

import nest_asyncio
nest_asyncio.apply()

from ib_insync import *
# util.startLoop()  # uncomment this line when in a notebook


ib = IB()
ib.connect('127.0.0.1', 4001, clientId=1)

contract = Forex('EURUSD')
bars = ib.reqHistoricalData(
    contract, endDateTime='', durationStr='30 D',
    barSizeSetting='1 hour', whatToShow='MIDPOINT', useRTH=True)

# convert to pandas dataframe:
df = util.df(bars)
print(df)

ib.disconnect(); print('disconnected')