#数据使用快期3，写入净值与详细持仓
from __future__ import print_function, absolute_import
import riskfolio
import numpy as np
import pandas as pd
from time import time,sleep
from datetime import datetime,timedelta
from tqsdk import TqApi, TqAuth, TqKq, TargetPosTask
import warnings
warnings.filterwarnings("ignore")
start=time()
print('start')
lag_return=126
lag_var=126
target_var=0.15
risk_budget=np.array([0.25, 0.25, 0.25, 0.25])
assets=['CFFEX.IF','CFFEX.IC','CFFEX.IM','CFFEX.T','SHFE.AU','INE.SC','SHFE.RB','DCE.I','SHFE.CU','SHFE.AG','SHFE.AL','DCE.P','SHFE.NI','DCE.Y','DCE.M','SHFE.RU','CZCE.TA','CZCE.SA','CZCE.OI','CZCE.MA','CZCE.CF','CZCE.FG','DCE.V','SHFE.ZN','DCE.J','SHFE.FU','CZCE.SR','CZCE.AP','DCE.PP','DCE.EG','CZCE.RM','DCE.L','DCE.JM','DCE.A']
symbols=['KQ.m@'+item if item.split('.')[0]=='CFFEX' or item.split('.')[0]=='CZCE' else 'KQ.m@'+item.split('.')[0]+'.'+item.split('.')[1].lower() for item in assets]

def return_i():
    raw_returns=[]
    days=api.get_trading_calendar(start_dt=pd.to_datetime('2015-04-17'), end_dt=datetime.now())['trading'].sum()
    for symbol in symbols:
        quote=api.get_kline_serial(symbol,86400,days).dropna(axis=0)
        quote['datetime']=pd.to_datetime(quote['datetime'],unit='ns').dt.strftime('%Y-%m-%d')
        returns=np.divide(quote['close'].to_list()[1:],quote['close'].to_list()[:-1])-1
        raw_returns.append(pd.DataFrame(returns,index=quote['datetime'].to_list()[1:],columns=[quote['symbol'].to_list()[0]]))
    returns=pd.concat(raw_returns,axis=1,join='outer').sort_index().fillna(0)
    returns.index=[item.date() for item in pd.to_datetime(returns.index)]
    returns.columns=assets
    commodity_weights_raw=pd.read_excel('kuaiqi.xlsx',sheet_name='weights',usecols='A:M',nrows=30,skiprows=0).set_index('代码')
    commodity_weights=pd.DataFrame(np.zeros((returns.shape[0],commodity_weights_raw.shape[0])),index=returns.index,columns=commodity_weights_raw.index)
    for date in commodity_weights.index:
        commodity_weights.loc[date,:]=commodity_weights_raw.loc[:,date.year] if date.month>=6 else commodity_weights_raw.loc[:,date.year-1]
    all_returns=pd.DataFrame(np.zeros((returns.shape[0],4)),index=returns.index,columns=['股','债','商','金'])
    for date in returns.index:
        if date<pd.to_datetime('2022-07-25').date():
            all_returns.loc[date,'股']=1/2*returns.loc[date,'CFFEX.IF']+1/2*returns.loc[date,'CFFEX.IC']
        else:
            all_returns.loc[date,'股']=1/3*returns.loc[date,'CFFEX.IF']+1/3*returns.loc[date,'CFFEX.IC']+1/3*returns.loc[date,'CFFEX.IM']
    all_returns['债']=returns['CFFEX.T']
    all_returns['商']=(returns.iloc[:,5:].values*commodity_weights.values).sum(axis=1)
    all_returns['金']=returns['SHFE.AU']
    return [all_returns,commodity_weights]

def weight_i(returns, rm, method_cov):
    weight=returns.copy()
    prev_weights=np.ones((returns.shape[1],))/ returns.shape[1]
    for i in range(lag_return,returns.shape[0]):
        portfolio=riskfolio.Portfolio(returns.iloc[i-lag_return:i,:],sht=True)
        try:
            portfolio.assets_stats(method_mu='hist', method_cov=method_cov)
            if np.isnan(portfolio.cov.values).any() or np.isinf(portfolio.cov.values).any():
                print(f"Step {i} skipped due to invalid cov matrix.")
                weight.iloc[i,:]=prev_weights
                continue
            weights=portfolio.rp_optimization(model='Classic',rm=rm,b=risk_budget.reshape(-1,1))
            prev_weights=weights.T.values.flatten()
            weight.iloc[i,:]=prev_weights
        except Exception as e:
            print(f"Step {i} error: {e}")
            weight.iloc[i,:]=prev_weights
    return weight.iloc[lag_return:, :]

def final_weight(pre_leverage, post_leverage, rm, method_cov):
    all_returns=return_i()[0]
    adj_returns, volatility=all_returns.copy(), all_returns.copy()
    if pre_leverage:
        for col in all_returns.columns:
            var=[np.sqrt(np.var(all_returns[col])*252)]*lag_var
            for i in range(lag_var,all_returns.shape[0]):
                var.append(np.sqrt(np.var(all_returns[col].iloc[i-lag_var:i])*252))
            var=[float(x) for x in var]
            volatility[col]=var
            adj_returns[col]=all_returns[col]*target_var/np.array(var)
    current_returns=adj_returns.iloc[lag_return+1:,:] if pre_leverage else all_returns.iloc[lag_return+1:,:]
    current_weights=weight_i(adj_returns if pre_leverage else all_returns, rm=rm, method_cov=method_cov)
    current_weights=current_weights.iloc[0:-1,:].set_index(current_returns.index)
    total_returns=(current_returns.values*current_weights.values).sum(axis=1)
    nav=pd.DataFrame((1+total_returns).cumprod(), index=current_returns.index, columns=['净值'])
    
    if post_leverage:
        var=[np.sqrt(np.var(total_returns)*252)]*lag_var
        for i in range(lag_var,len(total_returns)):
            var.append(np.sqrt(np.var(total_returns[i-lag_var:i])*252))
        final_weights=pd.DataFrame(current_weights.T.values/var*target_var, columns=current_weights.index, index=current_weights.columns).T
        final_returns=(current_returns.values*final_weights.values).sum(axis=1)
        final_nav=pd.DataFrame((1+final_returns).cumprod(), index=current_returns.index, columns=['加杠杆净值'])
    else:
        final_weights,final_returns,final_nav=current_weights,total_returns,nav
    
    if pre_leverage:
        for col in final_weights.columns:
            final_weights[col]=final_weights[col]*target_var/np.array(volatility[col][lag_return+1:])
    print(final_weights)
    result=pd.DataFrame(np.zeros((len(symbols),1)),index=symbols)
    for i in range(0,3):
        result.iloc[i,0]=final_weights.iloc[-1,0]/3
    result.iloc[3,0]=final_weights.iloc[-1,1]
    result.iloc[4,0]=final_weights.iloc[-1,3]
    result.iloc[5:,0]=final_weights.iloc[-1,2]*return_i()[1].iloc[-1,:]
    return result

def main():
    print([account.get_account().pre_balance,account.get_account().balance])
    cld=api.get_trading_calendar(start_dt=datetime.now()-timedelta(days=3), end_dt=datetime.now())
    if cld.trading.to_list()[-1]:
        weights=final_weight(True, True, 'MV', 'hist')
        count=0
        while api.wait_update() and count<100:
            for item in symbols:
                try:
                    quote=api.get_quote(item)
                    mainforce=api.query_his_cont_quotes(item)
                except Exception as e:
                    print(e)
                else:
                    if api.is_changing(quote):
                        if mainforce.iloc[-1,1]!=mainforce.iloc[-2,1]:
                            print(f'{item}需要换月，先平仓{mainforce.iloc[-2,1]}，再开仓{mainforce.iloc[-1,1]}')
                            put=TargetPosTask(api, mainforce.iloc[-2,1])
                            put.set_target_volume(0)
                        volume=weights.loc[item,0]*account.get_account().static_balance/(quote.last_price*quote.volume_multiple)
                        call=TargetPosTask(api, quote.underlying_symbol)
                        call.set_target_volume(np.round(volume,0))
                        print(f'{quote.underlying_symbol}调仓至{np.round(volume, 0)}手，现在持仓{account.get_position(quote.underlying_symbol).volume_long}手')
            count=count+1
            sleep(1)
        blc_pst=[account.get_account().pre_balance]+[account.get_position(api.get_quote(item).underlying_symbol).pos for item in symbols]
        print(blc_pst)
        if datetime.now().hour>15:
            data=pd.read_excel('kuaiqi.xlsx',sheet_name='account',usecols='A:AJ')
            data['日期']=pd.to_datetime(data['日期'],errors='coerce')
            dat= data.sort_values('日期').reset_index(drop=True)
            index_pos = data['日期'].searchsorted(pd.Timestamp.now().normalize(),side='left')
            print(f'开始写入{pd.Timestamp.now().normalize()}的净值与持仓，写入第{index_pos+1}行')
            with pd.ExcelWriter('kuaiqi.xlsx',engine='openpyxl',mode='a',if_sheet_exists='overlay') as writer:
                pd.DataFrame(blc_pst).T.to_excel(writer,sheet_name='account',startrow=index_pos+1,startcol=1,index=None,header=None)
            print('已写入')

account=TqKq()
api=TqApi(account=account,auth=TqAuth("李嘉骥","all_weather_sim"))
#api=TqApi(account=account,auth=TqAuth("ljj_test","ljj_test"))
main()
api.close()
print('end')
end=time()
print(f'{end-start}s')
