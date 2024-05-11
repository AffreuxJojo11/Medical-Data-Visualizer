import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['BMI'] = df['weight']/(df['height']/100)**2
df['overweight'] = df['BMI'].apply(lambda x : 1 if x>25 else 0)

# 3
df['cholesterol'] = df['cholesterol'].apply(lambda x: 1 if x>1 else 0)
df['gluc']        = df['gluc'].apply(lambda x: 1 if x>1 else 0)
df['smoke']       = df['smoke'].apply(lambda x: 1 if x>=1 else 0)
df['alco']        = df['alco'].apply(lambda x: 1 if x>=1 else 0)
df['active']      = df['active'].apply(lambda x: 1 if x>=1 else 0)
df['cardio']      = df['cardio'].apply(lambda x: 1 if x>=1 else 0)


# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol','gluc','smoke','alco','active','overweight'])


    # 6
    df_cat = df_cat.value_counts().to_frame()
    

    # 7
    df_cat.reset_index(inplace=True)
    df_cat.rename(columns={0:'total'}, inplace=True)


    # 8
    graph = sns.catplot(data=df_cat, x="variable", y="total",hue="value",
    col='cardio', kind="bar", order=['active','alco','cholesterol','gluc','overweight', 'smoke'])
    fig = graph.fig

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df.loc[(df['ap_lo'] <= df['ap_hi']) &
            (df['height'] >= df['height'].quantile(0.025)) &
            (df['height'] <= df['height'].quantile(0.975)) &
            (df['weight'] >= df['weight'].quantile(0.025)) &
            (df['weight'] <= df['weight'].quantile(0.975))]
    df_heat = df_heat.drop(columns='BMI')
    # 12
    corr = df_heat.corr()

    # 13
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True



    # 14
    fig, ax = plt.subplots()

    # 15
    fig.set_size_inches(10, 6)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', linewidth=.2, 
                          center=0.0, cbar_kws={"ticks":[-0.08,0.08,0.0,0.16,0.24]},
                          vmin=-0.2,vmax=0.3)

    # 16
    fig.savefig('heatmap.png')
    return fig
