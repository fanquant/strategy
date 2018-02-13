# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 14:06:52 2018

@author: fanl
"""

import pyodbc
import pandas as pd
from pandas import DataFrame
import pandas.io.sql as sql
import xlrd
from datetime import datetime
import time
from dateutil.parser import parse
conn=pyodbc.connect('DRIVER={SQL Server Native Client 10.0};SERVER=txic_service,1433;DATABASE=txnfdb;UID=txtg;PWD=tianxiang')
#SQL查询全部基金交易数据，并读取数据库列名称为表头(较慢)
#一、基于天相数据库进行基金筛选以及分类_______________________________________________________________________________________________________________________

#fund_summary=sql.read_sql('''select F_FUND_CODE,F_SETUP_DATE,F_STYLE_CODE,F_STYLE_NAME from fund.T_FUND ''',conn)
#剔除缺失值
#fund_summary=fund_summary.dropna()
#fund_summary.to_csv('D:/范良/独立研究/python/策略研究/多因子/fund_summary.csv')
f=open('D:/范良/独立研究/python/策略研究/多因子/fund_summary.csv')
fund_summary=pd.read_csv(f)
fund_summary['F_SETUP_DATE']
#--------------------------------------------------------------------------------------------------------------------

####
#转换日期格式由int型转换为“%Y-%m-%d”型,未解决！！！

####
#--------------------------------------------------------------------------------------------------------------------

drop_fund_style_code=['A0201','A0202','C01','C02','C04','D','E','F','J01','J02']
#剔除指数型基金、货币基金、QDII、其他类型基金。
fund_summary=fund_summary[~fund_summary['F_STYLE_CODE'].isin(drop_fund_style_code)]
#按基金风格分组，剔除设立日期
group_fund = fund_summary.groupby(['F_STYLE_CODE']).apply(lambda x:x[['F_FUND_CODE','F_SETUP_DATE']].set_index(['F_FUND_CODE'])['F_SETUP_DATE'].to_dict())
group_fund
#分别提取股票型、混合型、债券型基金股票仓位及股票市值
fund_hold_summary=sql.read_sql('''select fund.T_TOP10_STOCK.F_FUND_CODE,fund.T_TOP10_STOCK.F_END_DATE,fund.T_TOP10_STOCK.F6 as F_FUND_ratioInNa,fund.T_TOP10_STOCK.F4 as marketValue,fund.T_FUND.F_STYLE_CODE,fund.T_FUND.F_STYLE_NAME
                               from fund.T_TOP10_STOCK INNER JOIN fund.T_FUND
                               on fund.T_TOP10_STOCK.F_FUND_CODE=fund.T_FUND.F_FUND_CODE''',conn)

#fund_hold_summary.to_csv('D:/范良/独立研究/python/策略研究/多因子/fund_hold_summary.csv')
h=open('D:/范良/独立研究/python/策略研究/多因子/fund_hold_summary.csv')
fund_hold_summary=pd.read_csv(h)
fund_hold_summary['F_END_DATE']
####
#转换报告期数据类型

####
#剔除指数型基金、货币基金、QDII、其他类型基金（商品期货等）

fund_hold_summary=fund_hold_summary[~fund_hold_summary['F_STYLE_CODE'].isin(drop_fund_style_code)]
#积极股票型基金持仓及股票市值
ActiveStock_hold_summary=fund_hold_summary[fund_hold_summary['F_STYLE_CODE']=='A0101']
#稳健股票型基金持仓及股票市值
RobustStock_hold_summary=fund_hold_summary[fund_hold_summary['F_STYLE_CODE']=='A0102']
#灵活配置型基金持仓及股票市值
FlexibleConf_hold_summary=fund_hold_summary[fund_hold_summary['F_STYLE_CODE']=='B01']
#积极配置型基金持仓及股票市值
ActiveConf_hold_summary=fund_hold_summary[fund_hold_summary['F_STYLE_CODE']=='B02']
#保守配置型基金持仓及股票市值
ConservativeConf_hold_summary=fund_hold_summary[fund_hold_summary['F_STYLE_CODE']=='B03']
#保本型基金持仓及股票市值
CapitalGuaran_hold_summary=fund_hold_summary[fund_hold_summary['F_STYLE_CODE']=='B04']
#特定策略型基金持仓及股票市值
SpecificStrategic_hold_summary=fund_hold_summary[fund_hold_summary['F_STYLE_CODE']=='B05']
#二级债基持仓及股票市值
SecondaryMarket_hold_summary=fund_hold_summary[fund_hold_summary['F_STYLE_CODE']=='C03']
#赋值转换
AS_hold_summary=ActiveStock_hold_summary
RS_hold_summary=RobustStock_hold_summary
FC_hold_summary=FlexibleConf_hold_summary
AC_hold_summary=ActiveConf_hold_summary
CC_hold_summary=ConservativeConf_hold_summary
CG_hold_summary=CapitalGuaran_hold_summary
SS_hold_summary=SpecificStrategic_hold_summary
SM_hold_summary=SecondaryMarket_hold_summary
###
#计算证券仓位水平及证券市值规模
def get_fund_describe_summary(fund_hold,group_fund,fund_type):
    #取出公告日期为季末基金数据
    fund_hold=fund_hold[fund_hold['F_END_DATE'].apply(lambda x:x[5:]).isin(['3/31','6/30','9/30','12/31'])]
    #计算证券仓位水平
    fund_hold_percent=fund_hold.groupby(['F_END_DATE','F_FUND_CODE']).apply(lambda x:x['F_FUND_ratioInNa'].sum())
    #计算证券市值规模
    fund_hold_marketValue=fund_hold.groupby(['F_END_DATE','F_FUND_CODE']).apply(lambda x:x['marketValue'].sum())
    #数据聚合
    fund_hold_summary=pd.concat([fund_hold_percent,fund_hold_marketValue],axis=1).rename(columns= {0:'证券仓位水平',1:'证券市值规模'})
    #get_level_values(1)
    fund_hold_summary['F_SETUP_DATE']=fund_hold_summary.index.get_level_values(1).map(lambda x:group_fund[fund_type][x])
    #设置成立日期为时间戳
    fund_hold_summary['F_SETUP_DATE']=fund_hold_summary['F_SETUP_DATE'].map(lambda x:parse(x))
    #计算成立天数
    fund_hold_summary['diff_days']=fund_hold_summary.groupby(fund_hold_summary.index.get_level_values(0),group_keys=False).apply(lambda x:(parse(x.name)-x['F_SETUP_DATE']).map(lambda x:x.days))
    #计算基金净资产
    fund_hold_summary['基金净资产']=fund_hold_summary['证券市值规模']/fund_hold_summary['证券仓位水平'] *100
    return fund_hold_summary
##(1)积极股票型基金信息汇总
#有一些基金中途发生了转型 数据库成立日期为转型后日期，因此diff_days为负数
#剔除资产低于1亿元，公告日距离基金成立日期短于365天的基金
AS_fund_summary=get_fund_describe_summary(AS_hold_summary,group_fund,'A0101')
AS_select_summary=AS_fund_summary[(AS_fund_summary['基金净资产'] >1e+08)&(AS_fund_summary['diff_days']>365)]
AS_select_summary
##同理，其他类型基金的信息处理
##(2)稳健股票型
RS_fund_summary=get_fund_describe_summary(RS_hold_summary,group_fund,'A0102')
RS_select_summary=RS_fund_summary[(RS_fund_summary['基金净资产']>1e+08)&(RS_fund_summary['diff_days']>365)]
##（3）灵活配置型
FC_fund_summary=get_fund_describe_summary(FC_hold_summary,group_fund,'B01')
FC_select_summary=FC_fund_summary[(FC_fund_summary['基金净资产']>1e+08)&(FC_fund_summary['diff_days']>365)]
##(4)积极配置型
AC_fund_summary=get_fund_describe_summary(AC_hold_summary,group_fund,'B02')
AC_select_summary=AC_fund_summary[(AC_fund_summary['基金净资产']>1e+08)&(AC_fund_summary['diff_days']>365)]
##(5)保守配置型
CC_fund_summary=get_fund_describe_summary(CC_hold_summary,group_fund,'B03')
CC_select_summary=CC_fund_summary[(CC_fund_summary['基金净资产']>1e+08)&(CC_fund_summary['diff_days']>365)]
##(6)保本型
CG_fund_summary=get_fund_describe_summary(CG_hold_summary,group_fund,'B04')
CG_select_summary=CG_fund_summary[(CG_fund_summary['基金净资产']>1e+08)&(CG_fund_summary['diff_days']>365)]
##(7)特定策略型
SS_fund_summary=get_fund_describe_summary(SS_hold_summary,group_fund,'B05')
SS_select_summary=SS_fund_summary[(SS_fund_summary['基金净资产']>1e+08)&(SS_fund_summary['diff_days']>365)]
##(8)二级债基
SM_fund_summary=get_fund_describe_summary(SM_hold_summary,group_fund,'C03')
SM_select_summary=SM_fund_summary[(SM_fund_summary['基金净资产']>1e+08)&(SM_fund_summary['diff_days']>365)]

###二、基金因子选取与计算___________________________________________________________________________________
#提取交易日
#calendar_date=sql.read_sql(''' SELECT F_DAY from stock.T_NATURE_TRADEDAY where F_SHANGHAI=1 ''',conn)
#calendar_date.to_csv('D:/范良/独立研究/python/策略研究/多因子/calendar_date.csv')
t=open('D:/范良/独立研究/python/策略研究/多因子/calendar_date.csv')
calendar_date=pd.read_csv(t)
calendar_date
##获取月末最后一个交易日时间戳
def get_date_list(headortail=1):
    ## 可以得到每个月的月初与月末日期list
    #解析交易日期‘to_datetime’
    calendar_date['F_DAY']=pd.to_datetime(calendar_date['F_DAY'])
    calendar_date['year']=calendar_date['F_DAY'].map(lambda x:x.year)
    calendar_date['month']=calendar_date['F_DAY'].map(lambda x:x.month)
    if headortail==1:
        trade_date=calendar_date.groupby(['year','month']).tail(1)['F_DAY'].tolist()
    else:
        trade_date=calendar_date.groupby(['year','month']).head(1)['F_DAY'].tolist()
    return trade_date
###获得每个月末交易日的基金代码名单
def get_stage_analysis_fund(select_summary,tail_date):
    tail_date=get_date_list(1) #得到月末日期list
    select_summary=select_summary[:].reset_index(1)
    map_dict={}
    select_index=list(select_summary.index.drop_duplicates())
    for date in tail_date:
        for i_date in select_index:
            if date.strftime('%Y-%m-%d')>=i_date:
                map_dict['date']=i_date
    time_equity_fund_dict={}
    for date in tail_date:
        time_equity_fund_dict[date]=select_summary.ix[map_dict['date']]['F_FUND_CODE'].tolist()
    return pd.Series(time_equity_fund_dict)
tail_date=get_date_list()

#计算最大回撤
def compute_maxdrawdow(return_list):
    if len(return_list) < 18:
        return np.NAN
    else:
        return max([(return_list[i] - min(return_list[i+1:]))/return_list[i] for i in range(1,len(return_list)-1)])
#计算因子：计算历史60个交易日净值收益、历史20个交易日净值收益、60个交易日波动率、20个交易日波动率、60日最大回撤、20日最大回撤、60交易日夏普率、20交易日夏普率
time_SM_fund_s=get_stage_analysis_fund(SM_select_summary,tail_date)
time_SM_fund_s.index
##提取基金净值
#fund_nav=sql.read_sql('''select '交易日期'=F_TRADE_DATE,'基金代码'=F_FUND_CODE,'基金净值'=F10,'基金累计净值'=F11 FROM fund.T_NAV_RATE''',conn)
#fund_nav.to_csv('D:/范良/独立研究/python/策略研究/多因子/fund_nav.csv')
n=open('D:/范良/独立研究/python/策略研究/多因子/fund_nav.csv')
fund_nav=pd.read_csv(n)
fund_nav['交易日期']
def get_analysis_factor_summary(time_equity_fund_s):
    return_60B_dict={}
    return_20B_dict={}
    vol_60B_dict={}
    vol_20B_dict={}
    maxdrowdown_60B_dict={}
    maxdrowdown_20B_dict={}
    sharpe_60B_dict={}
    sharpe_20B_dict={}








