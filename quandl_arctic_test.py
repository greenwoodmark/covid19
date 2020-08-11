# -*- coding: utf-8 -*-
"""
test quandl and arctic using README.md instructions in man-group/arctic github repo
"""
import quandl
quandl.ApiConfig.api_key = "5Kbhf3VcyvZcxRZwh6XN"

from arctic import Arctic

# Connect to Local MONGODB
store = Arctic('localhost')

# Create the library - defaults to VersionStore
store.initialize_library('NASDAQ')

# Access the library
library = store['NASDAQ']

# Load some data - maybe from Quandl
aapl = quandl.get("WIKI/AAPL", authtoken=quandl.ApiConfig.api_key)

# Store the data in the library
library.write('AAPL', aapl, metadata={'source': 'Quandl'})

# Reading the data
item = library.read('AAPL')
aapl = item.data
metadata = item.metadata