"""Shared point-in-time fintech features for training and production inference."""
import numpy as np
import pandas as pd
H=30
def snapshot(raw,day,labeled=True):
    cut=pd.Timestamp(day)
    past=raw[raw.date<cut]
    recent=past[past.date>=cut-pd.Timedelta(days=180)].copy()
    g=recent.groupby('customer_id')
    x=g.agg(last=('date','max'),count_180=('value','size'),volume_180=('value','sum'),mean_180=('value','mean'),std_180=('value','std'),max_180=('value','max'),min_180=('value','min'),types_180=('type','nunique'),active_days_180=('date','nunique'))
    x['recency']=(cut-x.pop('last')).dt.days
    x['observed_tenure']=(cut-past.groupby('customer_id').date.min()).dt.days
    recent['weekend']=(recent.date.dt.dayofweek>=5).astype(float)
    recent['month_end']=(recent.date.dt.day>=25).astype(float)
    recent['negative']=(recent.amount<0).astype(float)
    recent['gap']=recent.groupby('customer_id').date.diff().dt.days
    g=recent.groupby('customer_id')
    for k in ['weekend','month_end','negative','gap']:
        x[k+'_mean']=g[k].mean()
    x['gap_std']=g.gap.std()
    for w in (7,14,30,60,90):
        q=recent[recent.date>=cut-pd.Timedelta(days=w)].groupby('customer_id')
        a=q.agg(**{f'count_{w}':('value','size'),f'volume_{w}':('value','sum'),f'mean_{w}':('value','mean'),f'std_{w}':('value','std'),f'active_days_{w}':('date','nunique'),f'types_{w}':('type','nunique')})
        x=x.join(a)
    x=x.fillna(0)
    for w in (7,14,30,60,90):
        x[f'count_share_{w}']=x[f'count_{w}']/x.count_180
        x[f'volume_share_{w}']=x[f'volume_{w}']/x.volume_180
    x['count_acceleration_30']=x.count_30/(1+(x.count_90-x.count_30)/2)
    x['volume_acceleration_30']=x.volume_30/(1+(x.volume_90-x.volume_30)/2)
    x['relative_amount_volatility']=x.std_180/(1+x.mean_180)
    x['transactions_per_active_day']=x.count_180/x.active_days_180
    x['as_of']=str(cut.date())
    if labeled:
        assert cut+pd.Timedelta(days=H-1)<=raw.date.max().normalize()
        f=raw[(raw.date>=cut)&(raw.date<cut+pd.Timedelta(days=H))].groupby('customer_id').value.agg(['size','sum'])
        x['future_count']=f['size'].reindex(x.index,fill_value=0)
        x['future_volume']=f['sum'].reindex(x.index,fill_value=0)
        x['inactive']=(x.future_count==0).astype(int)
    return x.reset_index()

