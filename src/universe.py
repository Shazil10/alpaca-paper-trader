import pandas as pd
import yfinance as yf
import requests
import io

# Configuration
MIN_PRICE = 10.0
MIN_DOLLAR_VOLUME = 10_000_000
OUTPUT_FILE = "universe.csv"

def get_candidates():
    """Scrapes S&P 500/400/600 and keeps Sector data."""
    sources = [
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
        "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"
    ]
    
    all_dfs = []
    headers = {'User-Agent': 'Mozilla/5.0'}
    print("Scraping indices...")

    for url in sources:
        try:
            response = requests.get(url, headers=headers)
            df = pd.read_html(io.StringIO(response.text))[0]
            
            # Standardize columns (Wiki tables vary slightly)
            # We need Symbol, Sector, and Sub-Industry
            cols = {'Symbol': 'Symbol', 'GICS Sector': 'Sector', 'GICS Sub-Industry': 'Industry'}
            df = df.rename(columns=cols)[cols.values()]
            
            all_dfs.append(df)
        except Exception as e:
            print(f"Failed to read {url}: {e}")

    # Combine and clean symbols (BRK.B -> BRK-B)
    full_df = pd.concat(all_dfs).drop_duplicates(subset='Symbol')
    full_df['Symbol'] = full_df['Symbol'].str.replace('.', '-', regex=False)
    
    return full_df

def filter_universe(df):
    """Filters the DataFrame by liquidity."""
    tickers = df['Symbol'].tolist()
    valid_tickers = []
    chunk_size = 300
    
    print(f"Filtering {len(tickers)} stocks...")

    for i in range(0, len(tickers), chunk_size):
        batch = tickers[i:i+chunk_size]
        try:
            data = yf.download(batch, period="5d", group_by='ticker', threads=True, progress=False)
            
            for symbol in batch:
                try:
                    stock_data = data[symbol]
                    if stock_data.empty: continue
                    
                    avg_price = stock_data['Close'].mean()
                    avg_vol = (stock_data['Close'] * stock_data['Volume']).mean()

                    if avg_price > MIN_PRICE and avg_vol > MIN_DOLLAR_VOLUME:
                        valid_tickers.append(symbol)
                except (KeyError, ValueError):
                    continue
        except Exception as e:
            print(f"Batch error: {e}")

    # Return only the rows that passed the filter
    return df[df['Symbol'].isin(valid_tickers)]

if __name__ == "__main__":
    candidates_df = get_candidates()
    final_df = filter_universe(candidates_df)
    
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Done. Saved {len(final_df)} stocks with Sector data to {OUTPUT_FILE}.")