# %%
import sys
sys.path.append("/home/samael/github/FinNLP")
# %%
from finnlp.data_sources.news.sina_finance_date_range import Sina_Finance_Date_Range
# %%
config = {
    "use_proxy": "china_free",
    "max_retry": 5,
    "proxy_pages": 5,
}

# config = {
#     "max_retry": 5,
# }
# %%
start_date = "2022-01-01"
end_date = "2022-01-31"

# %%
news_downloader = Sina_Finance_Date_Range(config)
news_downloader.download_date_range_all(start_date, end_date)
news_downloader.gather_content()
# %%
df = news_downloader.dataframe
print(df.shape)
df.to_csv('/home/samael/Downloads/sina_news_2212.csv', sep='\x01', index=False)
# %%
