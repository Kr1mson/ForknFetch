import pandas as pd
import json

paths = [
    'data/recipes_raw/recipes_raw_nosource_fn.json',
    'data/recipes_raw/recipes_raw_nosource_ar.json',
    'data/recipes_raw/recipes_raw_nosource_epi.json'
]

all_recipes = []

for path in paths:
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        all_recipes.extend(data.values())

df = pd.json_normalize(all_recipes)
df = df.dropna(subset=['title', 'ingredients', 'instructions']).reset_index(drop=True)

df['ingredients'] = df['ingredients'].apply(
    lambda x: ', '.join(x) if isinstance(x, list) else str(x)
)

df['instructions'] = df['instructions'].apply(
    lambda x: ' '.join(x) if isinstance(x, list) else str(x)
)
df['text'] = (
    df['title']
    + '. Ingredients: '
    + df['ingredients']
    + '. Instructions: '
    + df['instructions']
)

df['text'] = df['text'].str.replace('ADVERTISEMENT', '', regex=False)
df.to_csv('data/recipes_cleaned.csv', index=False)

print(f"Saved {len(df)} cleaned recipes.")