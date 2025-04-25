import pandas as pd, json

df = pd.read_csv("C:/Users/janfa/OneDrive/桌面/教材/2025Spring/captions.txt", names=["image", "caption"])
grouped = df.groupby("image")["caption"].apply(list).reset_index()
json_data = [{"image": f"data/Images/{row['image']}", "caption": row["caption"]} for _, row in grouped.iterrows()]

with open("data/flickr8k.json", "w") as f:
    json.dump(json_data, f, indent=2)