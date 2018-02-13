
# coding: utf-8

# ## 摘要
# ----------
# 本文首先将基金进行分类筛选，然后在不同基金种类中，通过基金的历史业绩数据选择因子进行有效性测试。我们主要将因子分为3个大类，包括收益类因子，风险类因子、风险调整收益类因子。从测算的结果来
# 看，相同的因子对于不同类型的基金可能具有不同的影响效果，然而对于某一些因子来看，如收益类因子中的历史20个交易日净值收益对于每一类基金的平均IC都较高。
# <br>
# 最后我们用有效的因子对各类基金进行筛选，发现利用多因子选基的方法来构建FOF组合相比于全市场的平均水平均获得了不同幅度的超额收益。多因子选基对于FOF组合收益率具有一定的提升作用。

# In[ ]:

import numpy as np
import pandas as pd
from CAL.PyCAL import *
import datetime
import scipy.stats as st
from dateutil.parser import parse
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter 
cal = Calendar('China.SSE')


# ## 一、基金筛选以及分类
# ----
# 基金分类是基金评级的基础，只有将具有相近风险收益特征的基金放在一起比较，才能保证结果的有效性，因此进行因子测试之前，我们首先对于基金进行分类。这里我们拿取的是通联数据所有股票、债券与混合型基金。由于我们希望优选基金来获得更加的收益，因此我们剔除以被动管理模型进行管理的指数型基金。

# In[ ]:

fund_summary = DataAPI.FundGet(etfLof=u"",secID=u"",ticker=u"",category=u"E,H,B",idxID=u"",field=u"secID,secShortName,establishDate,category,indexFund",pandas="1")
#剔除指数型基金
fund_summary = fund_summary[fund_summary['indexFund'] != 'I']
#按照基金类型分组提取基金代码及基金成立日期
group_fund = fund_summary.groupby(['category']).apply(lambda x:x[['secID','establishDate']].set_index(['secID'])['establishDate'].to_dict())
#查看fund_summary内容
#print(fund_summary)
print(group_fund)


# In[ ]:

#提取混合型基金、股票型基金及债券型基金的仓位数据及市场规模数据
mix_hold_summary = DataAPI.FundHoldingsGet(reportDate=u"",secID=group_fund['H'].keys(),ticker=u"",beginDate=u"20140101",
endDate=u"",secType="",field=u"secID,reportDate,ratioInNa,marketValue",pandas="1")
print(mix_hold_summary)
bond_hold_summary = DataAPI.FundHoldingsGet(reportDate=u"",secID=group_fund['B'].keys(),ticker=u"",beginDate=u"20140101",
                                       endDate=u"",secType="",field=u"secID,reportDate,ratioInNa,marketValue",pandas="1")
eq_hold_summary = DataAPI.FundHoldingsGet(reportDate=u"",secID=group_fund['E'].keys(),ticker=u"",beginDate=u"20140101",
                                       endDate=u"",secType="",field=u"secID,reportDate,ratioInNa,marketValue",pandas="1")


# In[ ]:

def get_fund_describe_summary(fund_hold,group_fund,fund_type):
    
    fund_hold = fund_hold[fund_hold['reportDate'].apply(lambda x:x[5:]).isin(['03-31','06-30','09-30','12-31'])]
    fund_hold_percent = fund_hold.groupby(['reportDate','secID']).apply(lambda x:x['ratioInNa'].sum())
    fund_hold_marketValue = fund_hold.groupby(['reportDate','secID']).apply(lambda x:x['marketValue'].sum())
    fund_hold_summary = pd.concat([fund_hold_percent,fund_hold_marketValue],axis=1).rename(columns= {0:'证券仓位水平',1:'证券市值规模'})
    fund_hold_summary['establishDate'] = fund_hold_summary.index.get_level_values(1).map(lambda x:group_fund[fund_type][x])
    fund_hold_summary['establishDate'] = fund_hold_summary['establishDate'].map(lambda x:parse(x))
    fund_hold_summary['diff_days'] = fund_hold_summary.groupby(fund_hold_summary.index.get_level_values(0),group_keys=False).apply(lambda x:(parse(x.name) - x['establishDate']).map(lambda x:x.days))
    fund_hold_summary['基金净资产'] = fund_hold_summary['证券市值规模'] / fund_hold_summary['证券仓位水平'] *100
    return fund_hold_summary


# 由于FOF对于持有子基金的规模以及存续时间也有一定的要求，其正式指引中规定：除ETF联接基金外，基金中基金投资其他基金时，被投资基金的运作期限应当不少于1年，最近定期报告披露的基金净资产应当不低于1亿元。因此在筛选时，我们剔除存续时间不足一年，或者最新规模不足1亿人民币的基金。如下所示，我们得到最终每期的各类基金的基金池。

# In[ ]:

equity_fund_summary = get_fund_describe_summary(eq_hold_summary, group_fund,'E')
equity_select_summary = equity_fund_summary[(equity_fund_summary['基金净资产'] > 1e+08) & (equity_fund_summary['diff_days']>365)]


# In[ ]:

equity_select_summary.head()
#print(equity_select_summary[equity_select_summary['secID']=='040002.OFCN'])


# In[ ]:

bond_fund_summary = get_fund_describe_summary(bond_hold_summary, group_fund,'B')
bond_select_summary = bond_fund_summary[(bond_fund_summary['基金净资产'] > 1e+08) & (bond_fund_summary['diff_days']>365)]


# In[ ]:

bond_select_summary.tail()


# In[ ]:

mix_fund_summary = get_fund_describe_summary(mix_hold_summary, group_fund,'H')
mix_select_summary = mix_fund_summary[(mix_fund_summary['基金净资产'] > 1e+08) & (mix_fund_summary['diff_days']>365)]


# In[ ]:

mix_select_summary.head()
#print(mix_select_summary['establishDate'])
###至此以上，天相数据库完全可以实现


# ## 二、基金因子选取与计算
# ----------------
# 对于分类之后的基金池，我们对于每一个基金类别进行因子的测试，来寻找对于不同类别的基金，驱动其业绩收益的主要因素。
# <br>
# 在基金因子的选择上，由于基金的非业绩数据更新频率较低，因此我们主要从基金的历史业绩入手，根据基金历史业绩数据进行一定的处理，得到各个因子。我们主要将因子分为3个大类，包括收益类因子，风险类因子、风险调整收益类因子。如下所示。
# <center>
# 
# | 因子分类       | 因子解释      | 因子命名    |
# |:-------------: |:-------------:|:------:|
# | 收益类因子      | 历史60个交易日净值收益 |Return_60B|
# | 收益类因子      | 历史20个交易日净值收益 |Return_20B|
# | 风险类因子      | 历史60个交易日波动率 |Vol_60B|
# | 风险类因子      | 历史20个交易日波动率 |Vol_20B|
# | 风险类因子      | 历史60个交易日最大回撤 |Maxdrawdown_60B|
# | 风险类因子      | 历史20个交易日最大回撤 |Maxdrawdown_20B|
# | 风险调整收益类因子      | 历史60个交易日夏普比率 |Sharpe_60B|
# | 风险调整收益类因子      | 历史20个交易日夏普比率 |Sharpe_20B|

# In[ ]:

def get_date_list(headortail = 1):
    ## 可以得到20015年1月至2015年9月交易日里每个月的月初与月末日期list
    #获取20140301-20170531的是否交易/日历日期两列数据
    calendar_date = DataAPI.TradeCalGet(exchangeCD=u"XSHG",beginDate=u"20140301",endDate=u"20170531",
                                        field=u"",pandas="1")[['isOpen','calendarDate']]
    #取出交易日日期
    calendar_date = calendar_date[calendar_date['isOpen']==1]
    #解析交易日期‘to_datetime’
    calendar_date['calendarDate'] = pd.to_datetime(calendar_date['calendarDate'])
    calendar_date['year'] = calendar_date['calendarDate'].map(lambda x:x.year) 
    calendar_date['month'] = calendar_date['calendarDate'].map(lambda x:x.month)
    if headortail == 1:
        trade_date = calendar_date.groupby(['year','month']).tail(1)['calendarDate'].tolist()
    else:
        trade_date = calendar_date.groupby(['year','month']).head(1)['calendarDate'].tolist()
    return trade_date
print(get_date_list())


# In[ ]:

def get_stage_analysis_fund(select_summary,tail_date):
    tail_date = get_date_list(1) #得到月末日期list
    select_summary = select_summary[:].reset_index(1)
    map_dict = {}
    select_index = list(select_summary.index.drop_duplicates())
    for date in tail_date:
        for i_date in select_index:
            if date.strftime('%Y-%m-%d') >= i_date:
                map_dict[date] = i_date
    time_equity_fund_dict = {}
    for date in tail_date:
        time_equity_fund_dict[date] = select_summary.ix[map_dict[date]]['secID'].tolist()
    
    return pd.Series(time_equity_fund_dict)


# In[ ]:

tail_date = get_date_list()
print(tail_date)


# In[ ]:

#计算最大回撤
def compute_maxdrawdow(return_list):
    if len(return_list) < 18:
        return np.NAN
    else:
        return max([(return_list[i] - min(return_list[i+1:]))/return_list[i] for i in range(1,len(return_list)-1)])
#计算因子：计算历史60个交易日净值收益、历史20个交易日净值收益、60个交易日波动率、20个交易日波动率、60日最大回撤、20日最大回撤、60交易日夏普率、20交易日夏普率
def get_analysis_factor_summary(time_equity_fund_s):
    return_60B_dict = {}
    return_20B_dict = {}
    vol_60B_dict = {}
    vol_20B_dict = {}
    maxdrowdown_60B_dict = {}
    maxdrowdown_20B_dict = {}
    sharpe_60B_dict = {}
    sharpe_20B_dict = {}

    for index in time_equity_fund_s.index:
        before_time = cal.advanceDate(index,'-60B')
        fund_nav = DataAPI.FundNavGet(dataDate=u"",secID=time_equity_fund_s[index],ticker=u"",beginDate=before_time.strftime('%Y%m%d'),endDate=index.strftime('%Y%m%d'),
                                      field=u"secID,endDate,ADJUST_NAV",pandas="1")
        def get(x):
            try:
                return x['ADJUST_NAV'].iloc[-1] / x['ADJUST_NAV'].iloc[-21] - 1
            except:
                return np.NAN
        #60日净值收益
        return_60B_dict[index] = fund_nav.groupby(['secID']).apply(lambda x:x['ADJUST_NAV'].iloc[-1] / x['ADJUST_NAV'].iloc[0] - 1)
        #20日净值收益
        return_20B_dict[index] = fund_nav.groupby(['secID']).apply(lambda x:get(x))
        #60日波动率
        vol_60B_dict[index] = fund_nav.groupby(['secID']).apply(lambda x:x['ADJUST_NAV'].pct_change().std())
        #20日波动率
        vol_20B_dict[index] = fund_nav.groupby(['secID']).apply(lambda x:x['ADJUST_NAV'].pct_change().iloc[-21:-1].std())
        #60日最大回撤
        maxdrowdown_60B_dict[index] = fund_nav.groupby(['secID']).apply(lambda x:compute_maxdrawdow(x['ADJUST_NAV'].tolist()))
        #20日最大回撤
        maxdrowdown_20B_dict[index] = fund_nav.groupby(['secID']).apply(lambda x:compute_maxdrawdow(x['ADJUST_NAV'].iloc[-21:-1].tolist()))
        #60日夏普率
        sharpe_60B_dict[index] = fund_nav.groupby(['secID']).apply(lambda x:(x['ADJUST_NAV'].iloc[-1] / x['ADJUST_NAV'].iloc[0] - 1) / x['ADJUST_NAV'].pct_change().std())
        #20日夏普率
        sharpe_20B_dict[index] = fund_nav.groupby(['secID']).apply(lambda x:get(x) / x['ADJUST_NAV'].pct_change().iloc[-21:-1].std()) ##假设无风险收益率为0，计算夏普率
        
    return pd.DataFrame(return_60B_dict).T,pd.DataFrame(return_20B_dict).T,pd.DataFrame(vol_60B_dict).T,pd.DataFrame(vol_20B_dict).T,pd.DataFrame(maxdrowdown_60B_dict).T,pd.DataFrame(maxdrowdown_20B_dict).T,pd.DataFrame(sharpe_60B_dict).T,pd.DataFrame(sharpe_20B_dict).T


# In[ ]:

#计算
def get_analysis_return_summary(time_equity_fund_s):
    total_fund_list=[]
    for index in time_equity_fund_s.index:
        total_fund_list=total_fund_list+time_equity_fund_s[index]
    total_fund_list=list(np.unique(total_fund_list))
    quotation_summary=pd.DataFrame()
    for index in time_equity_fund_s.index:
        fund_tmp = DataAPI.FundNavGet(dataDate=index.strftime('%Y%m%d'),secID=total_fund_list,ticker=u"",beginDate=u"",endDate=u"",field=u"secID,endDate,ADJUST_NAV",pandas="1")
        quotation_summary=pd.concat([quotation_summary,fund_tmp],axis=0)
    quotation_summary['endDate']=quotation_summary['endDate'].map(lambda x:parse(x))
    return quotation_summary.set_index(['endDate','secID']).unstack()['ADJUST_NAV']


# In[ ]:

def get_analysis_summary(time_equity_fund_s):
    return_60B_df, return_20B_df, vol_60B_df, vol_20B_df, maxdrowdown_60B_df, maxdrowdown_20B_df, sharpe_60B_df, sharpe_20B_df = get_analysis_factor_summary(time_equity_fund_s)
    quotation_summary = get_analysis_return_summary(time_equity_fund_s)
    fund_return_summary = quotation_summary.pct_change().shift(-1)
    name_dict = pd.Series(['Return_60B','Return_20B','Vol_60B','Vol_20B','MaxDrowDown_60B','MaxDrowDown_20B','Sharpe_60B','Sharpe_20B','Return']).to_dict()
    analysis_summary = pd.concat([return_60B_df.stack(), return_20B_df.stack(), vol_60B_df.stack(), vol_20B_df.stack(), maxdrowdown_60B_df.stack(), maxdrowdown_20B_df.stack(), sharpe_60B_df.stack(), sharpe_20B_df.stack(),fund_return_summary.stack()],axis=1).rename(columns=name_dict)
    return analysis_summary


# In[ ]:

time_equity_fund_s = get_stage_analysis_fund(equity_select_summary,tail_date)
time_bond_fund_s = get_stage_analysis_fund(bond_select_summary,tail_date)
time_mix_fund_s = get_stage_analysis_fund(mix_select_summary,tail_date)
time_equity_fund_s.head()
#time_bond_fund_s.head()


# In[ ]:

eqfund_analysis_summary =  get_analysis_summary(time_equity_fund_s)


# In[ ]:

eqfund_analysis_summary.head()


# In[ ]:

bofund_analysis_summary =  get_analysis_summary(time_bond_fund_s)
bofund_analysis_summary.head()


# In[ ]:

mixfund_analysis_summary =  get_analysis_summary(time_mix_fund_s)
mixfund_analysis_summary.head()


# 

# 如上我们便得到了不同类基金的基金池对应的各期的因子值以及下一期的收益。

# ## 三、因子的有效性测试
# --------
# 在因子有效性的测试上，我们使用IC作为度量因子有效性的标准，并采用月度作为测算的频率。在每个月月底，我们对于每一个基金类别，计算其因子值与下期基金净值收益率之间的相关系数，并对每期数据取平均，得到每一个因子对于每一类基金的平均IC值。
# 

# In[ ]:

def get_ic_series(fator_return_df,factor_name):
    """
    返回IC序列
    """
    
    def omit_nan_spearmanr(x):
        x = x.dropna()
        if len(x) > 0:
            return st.spearmanr(x['Return'],x[factor_name])[0]
    ic_result = fator_return_df.groupby(level=0).apply(lambda x:omit_nan_spearmanr(x))
    ic_result = pd.DataFrame(ic_result.dropna()).rename(columns={0:factor_name})
    # ic_result['IC'] = ic_result[0].map(lambda x:x[0])
    # ic_result['IC_Pvalue'] = ic_result[0].map(lambda x:x[1])

    return ic_result


# In[ ]:

def get_ic_summary(analysis_summary):
    ic_list = []
    for col in analysis_summary.columns[:-1]:
        ic_list.append(get_ic_series(analysis_summary,col))
    return pd.concat(ic_list,axis=1)


# 对于股票型基金，其因子IC值汇总如下。

# In[ ]:

IC_eqfund = get_ic_summary(eqfund_analysis_summary)
IC_eqfund.describe()


# 从IC均值以及IC中位数来看，因子`Return_20B`对下一期收益的预测能力最强，平均来说这一期`Return_20B`越大，下一期收益也越高。风险收益类的信号大都是反向指标，这与我们的直觉比较吻合，但是从数值来看，预测能力太小。

# 对于债券型基金，其因子IC值汇总如下。

# In[ ]:

IC_bofund = get_ic_summary(bofund_analysis_summary)
IC_bofund.describe()


# 从IC均值来看，因子Return_60B对债券型基金下一期收益的预测能力最强，平均来说这一期Return_60B越大，下一期收益也越高。风险收益类的信号都是正向指标，而Sharpe值对下一期收益没有什么预测能力。这个可能的原因是，债券型基金本身分成利率债和信用债，信用债因为有信用风险，本身的利率就要高过利率债券，当然其受利率的影响波动也更大。也就是说，债券型基金里面过去收益较高都是一些信用债基金，其未来当然收益也不会低，并且这类基金通常会有更大的波动和最大回撤，但其收益在大多数情况下都会比利率债基高。

# 对于混合型基金，其因子IC值汇总如下。

# In[ ]:

IC_mixfund = get_ic_summary(mixfund_analysis_summary)
IC_mixfund.describe()


# 对于混合型基金，仍然是因子`Return_20B`对下一期收益的预测能力最强。

# ## 四、单因子选择基金效果
# ---------
# 通过上面的单因子测试，我们筛选出了各类基金中最有效的因子，接下来我们以因子选基来看下累计收益的效果。

# In[ ]:

def get_select_fund_return_plot(analysis_summary,factor_name,threhold):
    
    fund_return_summary = analysis_summary['Return'].unstack()
    avg_fund_return = analysis_summary.groupby(analysis_summary.index.get_level_values(0)).apply(lambda x:x['Return'].mean())
    
    def factor_select(x):
        return list(x.dropna()[x.dropna() > x.dropna().quantile(threhold)].index)
    
    select_fund = analysis_summary[factor_name].unstack().apply(lambda x:factor_select(x),axis=1)
    
    def select_return(x):
        return x.ix[select_fund.ix[x.name]].dropna().mean()
    
    select_fund_return = fund_return_summary.apply(lambda x:select_return(x),axis=1)
    
    fig, ax = plt.subplots(figsize=(8,4))
    (1 + avg_fund_return).cumprod().plot(ax=ax,color='orange',linewidth=3)
    (1+ select_fund_return).cumprod().plot(ax=ax,color='deepskyblue',linewidth=3)
    ax.legend(['benchmark_return','selected_return'],loc='upper left',bbox_to_anchor=(0.01,0.99),prop=font,frameon= False)


# 对于股票型基金，我们每一期选取`Return_20B`最大的20%作为我们的选择的基金。

# In[ ]:

get_select_fund_return_plot(eqfund_analysis_summary,'Return_20B',0.8)


# 可以看出选取历史20个交易日净值收益最好的20%的股票基金表现的要比全市场股票型基金平均值要好不少，尤其是在牛市的时候。

# 对于债券型基金，我们每一期选取`Return_60B`最大的20%作为我们的选择的基金。

# In[ ]:

get_select_fund_return_plot(bofund_analysis_summary,'Return_60B',0.8)


# 可以看出选取的债券基金要比全市场平均的水平好很多，但值得注意的是最大回撤和波动率也明显更大。原因也是我之前所说的，债券型基金里面，信用债可能比利率债要表现好，但是存在更大的风险。

# 对于混合型基金，我们每一期选取`Return_20B`最大的20%作为我们的选择的基金。

# In[ ]:

get_select_fund_return_plot(mixfund_analysis_summary,'Return_20B',0.8)


# 可以看出选取历史20个交易日净值收益最好的20%的混合型基金表现的要比全市场混合型基金平均值也要好不少。

# ## 参考文献
# ---------
# * 广发证券-FOF系列专题之四:基于基金精选的FOF组合策略.2017.04.14
