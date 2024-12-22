import pandas as pd


def descriptive_statistics(df):
    
    selected_columns = [
        "lntotal", "lnfinance", "lngdpp", "lnpop", "urban",
        "lnee", "industry", "trade", "fdiin", "repolicy", 
        "patent", "wind", "geo", "solar"
    ]

    stats = {
        "Variable": [],
        "Obs": [],
        "Mean": [],
        "SD": [],
        "Min": [],
        "Median": [],
        "Max": []
    }

    for column in selected_columns:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            stats["Variable"].append(column)
            stats["Obs"].append(df[column].count())
            stats["Mean"].append(round(df[column].mean(), 3))
            stats["SD"].append(round(df[column].std(), 3))
            stats["Min"].append(round(df[column].min(), 3))
            stats["Median"].append(round(df[column].median(), 3))
            stats["Max"].append(round(df[column].max(), 3))

    return pd.DataFrame(stats)


def xtbalance(df, id_var):
    
    year_list = range(df['year'].min(), df['year'].max() + 1)
    iso3_counts = df.groupby(id_var)['year'].nunique()
    valid_iso3 = iso3_counts[iso3_counts == len(year_list)].index

    filtered_data = df[df[id_var].isin(valid_iso3)]

    full_index = pd.MultiIndex.from_product([valid_iso3, year_list], names=[id_var, 'year'])

    balanced_data = filtered_data.set_index([id_var, 'year']).reindex(full_index).reset_index()

    return balanced_data