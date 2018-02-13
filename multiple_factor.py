# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 19:13:36 2018

@author: fanl
"""
import numpy as np
import pandas as pd
# 导入金融数据分析模块 import CAL.PyCAL import *
import datetime
import scipy.stats as st
from dateutil.parser import parse
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
##不知道啥意思？？cal=Calendar('China.SEE')
#连接数据库，输出基金数据
import pyodbc
from pandas import DataFrame
import pandas.io.sql as sql
import math
conn=pyodbc.connect('DRIVER={SQL Server Native Client 10.0};SERVER=txic_service,1433;DATABASE=txnfdb;UID=txtg;PWD=tianxiang')
#SQL查询全部基金交易数据，并读取数据库列名称为表头(较慢)
fund_summary=sql.read_sql('''select '基金代码'=F_FUND_CODE,'基金简称'=F_FUND_NAME,'设立日期'=F_SETUP_DATE,
                          '基金风格'=F_STYLE_NAME,'基金状态'=F_STATUS,'基金终止日期'=F55,'数据状态'=S_DATA_STATE
                          from FUND.T_FUND ''',conn)
fund_summary['基金状态'].unique()
#
fund_summary[fund_summary['基金风格']==('增强指数型')]
fund_summary[fund_summary['基金风格']==('纯指数型')]
#剔除指数型基金
list=['纯指数型','增强指数型']
fund_summary_new=fund_summary[~fund_summary['基金风格'].isin(list) ]
fsn=fund_summary_new


