# %%
from wallhaven import api
from wallhaven.api import Wallhaven
from tqdm import tqdm
# %%
wallhaven = Wallhaven(api_key='2RCqfqJ3p6MPRtIOLoKkfUxCHPFtprTI')
# %%
wallhaven.params["categories"] = "001"
wallhaven.params["purity"] = '001'
wallhaven.params["sorting"] = "toplist"
wallhaven.params["topRange"] = "1M"
wallhaven.params["page"] = 1
results = wallhaven.search()

# %%
for wallpaper in tqdm(results.data):
    pic_info = wallhaven.get_wallpaper(wallpaper_id=wallpaper.id)
    prompt = '\x01'.join([x.name for x in pic_info.tags])
    with open('/home/samael/Pictures/SD_corpus/prompts.txt', 'a') as f:
        f.write(prompt + '\x02' + pic_info.id + '\n')
    wallpaper.save('/home/samael/Pictures/SD_corpus')
# %%
