import akshare as ak
import pandas as pd
import random
import os
import time

def download_sz500_subset(stock_count=50, start_date="20220101", end_date="20240101", save_dir="data"):
    
    components = ak.index_stock_cons("000905")
    stock_list = components["品种代码"].tolist()

    sampled_stocks = random.sample(stock_list, stock_count)
    pd.DataFrame({"selected_stocks": sampled_stocks}).to_csv("data/selected_stocks.csv", index=False)


    os.makedirs(save_dir, exist_ok=True)

    for code in sampled_stocks:
        try:
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )
            df.to_csv(f"{save_dir}/{code}.csv", index=False)
            print(f"Saved: {code}")
            time.sleep(1)  
        except Exception as e:
            print(f"Failed: {code} | {e}")

if __name__ == "__main__":
    download_sz500_subset()
