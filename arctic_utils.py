# -*- coding: utf-8 -*-
"""
test quandl and arctic using README.md instructions in man-group/arctic github repo
"""

import pandas as pd

#--------------------------------
def XLSX_to_arctic(filename_and_path, library_name, collection_name):
    """
    writes Excel file to a versionstore
    """
    from arctic import Arctic
    df = pd.read_excel(filename_and_path)
    store = Arctic('localhost')
    # Create the library - defaults to VersionStore
    store.initialize_library(library_name)
    library = store[library_name]
    library.write(collection_name, df, metadata={'source': 'SPS'})

    result = str(df.shape)+' written'
    return result
    
    
#--------------------------------
def quandl_arctic_test():
    """
    demonstrates a versionstore
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
    
#________________________________________________________________________________
if __name__ == "__main__":
    
    filename_and_path=r'C:\Users\Mark\Desktop\STATIC_table.xlsx'
    result = XLSX_to_arctic(filename_and_path)
