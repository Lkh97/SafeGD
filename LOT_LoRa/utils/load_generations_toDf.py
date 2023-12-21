import pandas as pd

def load_generations_toDf(path:str):
    df = pd.read_csv("path", sep=" ", header=None, 
                 names=["gens"])
    return df

    