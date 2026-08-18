import joblib, pandas as pd, pathlib
p = pathlib.Path('final_model.sav')
try:
    obj = joblib.load(p)
    print('Loaded as joblib object:', type(obj))
except Exception:
    try:
        df = pd.read_spss(p)
        print('Loaded as SPSS dataset; shape:', df.shape)
    except Exception as e:
        print('Could not load file:', e)