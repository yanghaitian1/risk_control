# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 16:52:55 2020

@author: Lenovo
"""

#!/usr/bin/env python
# coding: utf-8
# author: liguolong

# In[51]:
import pandas as pd
import numpy as np
import json
import re
import warnings
warnings.filterwarnings('ignore')


# In[9]:
def fillzero(x):
    if x == '':
        return 0
    else: 
        return float(x)

def fill_missing(x):
    if x=='':
        return -9999
    else:
        return float(x)
    
def add_two(x,y):
    if (x== -9999) & (y== -9999):
        return -9999
    elif (x==-9999) & (y>=0):
        return y
    elif (y==-9999) & (x>=0):
        return x
    else:
        return x+y

def myDivide(a, b):
    if b==0:
        return 0
    else: return a/b
    
def edu(x):
    if x=='91': return 0
    elif x=='60': return 1
    elif (x=='30')|(x=='40'): return 2
    elif (x=='10')|(x=='20'): return 3
    else: return -9999   

def monthList(n):
    month = []
    for i in range(n-11, n+1):
        s = 'PY01_MM' + str(i).rjust(2,'0')
        month.append(s)
    return month

def latest5years(x):
    monthArray = monthList(72)+monthList(60)+monthList(48)+monthList(36)+monthList(24)+monthList(12)
    return ''.join(x[monthArray].tolist())

def acountStatus(x):
    if (pd.isnull(x['PD01BD01']))|(x['PD01BD01']==''):
        return x['PD01CD01']
    else:
        return x['PD01BD01']
        
def get5class(x):
    if (pd.isnull(x['PD01BD03']))|(x['PD01BD03']==''):
        return x['PD01CD02']
    else:
        return x['PD01BD03']

def getBalance(x):
    if (pd.isnull(x['PD01BJ01']))|(x['PD01BJ01']==''):
        return x['PD01CJ01'].replace(',','')
    else:
        return x['PD01BJ01'].replace(',','')

def loanCreditAmt(x):
    if (x['PD01AD01'] == 'D1') | (x['PD01AD01'] == 'R4'):
        return x['PD01AJ01'].replace(',','')
    elif x['PD01AD01'] == 'R1':
        return x['PD01AJ02'].replace(',','')
    else:
        return np.nan

def specialStatus(x):
    if (x['PD01AD01'] == 'D1')&(x['账户状态'] not in ['1','2','3','5']): return 1
    elif (x['PD01AD01'] in ['R1','R4'])&(x['账户状态'] not in ['1','2','3']): return 1
    elif (x['PD01AD01'] in ['R2','R3'])&(x['账户状态'] not in ['1','4','6']): return 1
    else: return 0

def specialPay(x):
    pattern = r'[GDZ]?'
    return len([c for c in re.findall(pattern, x) if len(c)>0])

def latest24(x):
    pattern1 = r'\d?'
    pattern2 = r'\d*'
    n1 = [c for c in re.findall(pattern1, x) if c!='']
    n2 = [c for c in re.findall(pattern2, x) if c!='']
    return len(n1),max([len(i) for i in n2], default=0)

# In[95]:
def parsePboc(x):
    pboc = json.loads(x)
    var = {}
    
    # In[]
    # # PA01B基本信息
    if (isinstance(pboc['PA01BList'],list)) & (len(pboc['PA01BList'])>0):
        var['报告编号'] = pboc['PA01BList'][0]['PA01AI01']
        var['报告时间'] = pboc['PA01BList'][0]['PA01AR01']
        var['证件号码'] = pboc['PA01BList'][0]['PA01BI01']
        var['姓名'] = pboc['PA01BList'][0]['PA01BQ01']
        var['防欺诈警示标志'] = pboc['PA01BList'][0]['PA01DQ01']
        var['性别'] = pboc['PA01BList'][0]['PB01AD01'] # 0未知，1男，2女，9未说明
        var['出生日期'] = pboc['PA01BList'][0]['PB01AR01']
        var['学历'] = pboc['PA01BList'][0]['PB01AD02']
        var['是否本科及以上学历'] = 1 if (pboc['PA01BList'][0]['PB01AD02']=='10')|(pboc['PA01BList'][0]['PB01AD02']=='20') else 0  
        var['学历详细分类'] = edu(var['学历'])
        var['学位'] = pboc['PA01BList'][0]['PB01AD03']
        var['就业状况'] = pboc['PA01BList'][0]['PB01AD04']
        var['电子邮箱'] = pboc['PA01BList'][0]['PB01AQ01']
        var['通讯地址'] = pboc['PA01BList'][0]['PB01AQ02']
        var['国籍'] = pboc['PA01BList'][0]['PB01AD05']
        var['户籍地址'] = pboc['PA01BList'][0]['PB01AQ03']
        var['婚姻状态'] = pboc['PA01BList'][0]['PB020D01'] # 10未婚，20已婚，30丧偶，40离婚，91单身，99未知
    
    # In[]
    # # PB02B手机号码
    phone = pd.DataFrame()
    i = 0
    for j in pboc['PB02BList']:
        tmp = pd.DataFrame(j, index=[i])
        i+=1
        phone = pd.concat([phone, tmp])
        del tmp
    if phone.shape[0]>0:
        phone['PB01BR01'] = pd.to_datetime(phone['PB01BR01'])
        phone.sort_values(['PB01BR01'], ascending=False,inplace=True)
        var['手机号码'] = phone['PB01BQ01'].iloc[0]
    
    # In[]
    # # PC01数字解读
    if (isinstance(pboc['PC01List'],list))&(len(pboc['PC01List'])>0):
        var['数字解读分'] = pd.to_numeric(pboc['PC01List'][0]['PC010Q01'])
        var['相对位置'] = pboc['PC01List'][0]['PC010Q02']
    
    # In[]:
    # # PC02A
    PC02A = pd.DataFrame()
    j = 0
    for i in pboc['PC02AList']:
        tmp = pd.DataFrame(i, index=[j])
        j += 1
        PC02A = pd.concat([PC02A, tmp])
        del tmp
    
    # 被追偿信息汇总
    if PC02A.shape[0]>0:
        repay = PC02A[PC02A['PA02_01']=='PC02B']
        if repay.shape[0]>0:
            var['资产处置笔数'] = repay[repay['PC02AD01']=='1']['PC02AS03'].values[0]
            var['保证人代偿笔数'] = repay[repay['PC02AD01']=='2']['PC02AS03'].values[0]
        
        # 信贷交易提示信息段
        credit = PC02A[PC02A['PA02_01']=='PC02A'].replace('--','')
        if credit.shape[0] > 0:
            businessType = {'11':'个人住房贷款','12':'个人商用房（包括商住两用房）贷款','19':'其他类贷款',
                           '21':'贷记卡','22':'准贷记卡','99':'其他'}
            credit['业务类型'] = credit['PC02AD01'].map(businessType)
            var['个人住房贷款笔数'] = credit[credit['业务类型']=='个人住房贷款']['PC02AS03'].values[0]
            var['个人商用房（包括商住两用房）贷款笔数'] = credit[credit['业务类型']=='个人商用房（包括商住两用房）贷款']['PC02AS03'].values[0]
            var['其他类贷款笔数'] = credit[credit['业务类型']=='其他类贷款']['PC02AS03'].values[0]
            var['贷记卡账户数'] = credit[credit['业务类型']=='贷记卡']['PC02AS03'].values[0]
            var['准贷记卡账户数'] = credit[credit['业务类型']=='准贷记卡']['PC02AS03'].values[0]
        
        # 逾期（透支）汇总信息
        overdue = PC02A[PC02A['PA02_01']=='PC02D'].replace('--','')
        if overdue.shape[0] > 0:
            var['非循环贷账户_逾期记录账户数'] = overdue[overdue['PC02AD01']=='1']['PC02AS03'].values[0]
            var['循环额度下分账户_逾期记录账户数'] = overdue[overdue['PC02AD01']=='2']['PC02AS03'].values[0]
            var['循环贷账户_逾期记录账户数'] = overdue[overdue['PC02AD01']=='3']['PC02AS03'].values[0]
            var['贷记卡账户_逾期记录账户数'] = overdue[overdue['PC02AD01']=='4']['PC02AS03'].values[0]
            var['准贷记卡账户_逾期记录账户数'] = overdue[overdue['PC02AD01']=='5']['PC02AS03'].values[0]
        
            var['非循环贷账户_逾期月份数总和'] = overdue[overdue['PC02AD01']=='1']['PC02DS03'].values[0]
            var['循环额度下分账户_逾期月份数总和'] = overdue[overdue['PC02AD01']=='2']['PC02DS03'].values[0]
            var['循环贷账户_逾期月份数总和'] = overdue[overdue['PC02AD01']=='3']['PC02DS03'].values[0]
            var['贷记卡账户_逾期月份数总和'] = overdue[overdue['PC02AD01']=='4']['PC02DS03'].values[0]
            var['准贷记卡账户_逾期月份数总和'] = overdue[overdue['PC02AD01']=='5']['PC02DS03'].values[0]
        
            var['非循环贷账户_单月最高逾期总额'] = overdue[overdue['PC02AD01']=='1']['PC02BJ02'].values[0]
            var['循环额度下分账户_单月最高逾期总额'] = overdue[overdue['PC02AD01']=='2']['PC02BJ02'].values[0]
            var['循环贷账户_单月最高逾期总额'] = overdue[overdue['PC02AD01']=='3']['PC02BJ02'].values[0]
            var['贷记卡账户_单月最高逾期总额'] = overdue[overdue['PC02AD01']=='4']['PC02BJ02'].values[0]
            var['准贷记卡账户_单月最高逾期总额'] = overdue[overdue['PC02AD01']=='5']['PC02BJ02'].values[0]
        
            var['非循环贷账户_最长逾期月数'] = overdue[overdue['PC02AD01']=='1']['PC02DS04'].values[0]
            var['循环额度下分账户_最长逾期月数'] = overdue[overdue['PC02AD01']=='2']['PC02DS04'].values[0]
            var['循环贷账户_最长逾期月数'] = overdue[overdue['PC02AD01']=='3']['PC02DS04'].values[0]
            var['贷记卡账户_最长逾期月数'] = overdue[overdue['PC02AD01']=='4']['PC02DS04'].values[0]
            var['准贷记卡账户_最长逾期月数'] = overdue[overdue['PC02AD01']=='5']['PC02DS04'].values[0]
        
            # 贷款最长逾期月数
            var['贷款最长逾期月数'] = max([fillzero(var['非循环贷账户_最长逾期月数']),
                                      fillzero(var['循环额度下分账户_最长逾期月数']),
                                      fillzero(var['循环贷账户_最长逾期月数'])],default=0)
        
        # 公共信息概要
        publicSum = PC02A[PC02A['PA02_01']=='PC04'].replace('--','')
        if publicSum.shape[0] > 0:
            var['欠税信息'] = publicSum[publicSum['PC02AD01']=='1']['PC02AS03'].values[0]
            var['民事判决'] = publicSum[publicSum['PC02AD01']=='2']['PC02AS03'].values[0]
            var['强制执行'] = publicSum[publicSum['PC02AD01']=='3']['PC02AS03'].values[0]
            var['行政处罚'] = publicSum[publicSum['PC02AD01']=='4']['PC02AS03'].values[0]
    # In[]:
    # PC02
    if (isinstance(pboc['PC02List'],list)) & (len(pboc['PC02List'])>0):
        var['信贷交易提示信息_账户数合计'] = pboc['PC02List'][0]['PC02AS01']
        var['信贷交易提示信息_业务类型数量'] = pboc['PC02List'][0]['PC02AS02']
        # 被追偿信息汇总
        var['被追偿汇总信息_账户数合计'] = pboc['PC02List'][0]['PC02BS01']
        var['被追偿汇总信息_余额合计'] = pboc['PC02List'][0]['PC02BJ01']
        var['被追偿汇总信息_业务类型数量'] = pboc['PC02List'][0]['PC02BS02']
        # 呆账信息汇总
        var['呆账笔数'] = pboc['PC02List'][0]['PC02CS01']
        var['呆账汇总信息_余额'] = pboc['PC02List'][0]['PC02CJ01'].replace(',','')
        # 逾期透支信息汇总
        var['逾期透支汇总信息_账户类型数量'] = pboc['PC02List'][0]['PC02DS01']
        # 非循环账户信息汇总（已结清的贷款不在统计范围内）
        var['非循环账户信息汇总_管理机构数'] = pboc['PC02List'][0]['PC02ES01']
        var['非循环账户信息汇总_账户数'] = pboc['PC02List'][0]['PC02ES02']
        var['非循环账户信息汇总_授信总额'] = pboc['PC02List'][0]['PC02EJ01'].replace(',','')
        var['非循环账户信息汇总_余额'] = pboc['PC02List'][0]['PC02EJ02'].replace(',','')
        var['非循环贷最近6个月平均应还款'] = pboc['PC02List'][0]['PC02EJ03'].replace(',','')
        # 循环额度下分账户
        var['循环额度下分账户信息汇总_管理机构数'] = pboc['PC02List'][0]['PC02FS01']
        var['循环额度下分账户信息汇总_账户数'] = pboc['PC02List'][0]['PC02FS02']
        var['循环额度下分账户信息汇总_授信总额'] = pboc['PC02List'][0]['PC02FJ01'].replace(',','')
        var['循环额度下分账户信息汇总_余额'] = pboc['PC02List'][0]['PC02FJ02'].replace(',','')
        var['循环额度下分账户最近6个月平均应还款'] = pboc['PC02List'][0]['PC02FJ03'].replace(',','')
        # 循环贷账户信息汇总
        var['循环贷账户信息汇总_管理机构数'] = pboc['PC02List'][0]['PC02GS01']
        var['循环贷账户信息汇总_账户数'] = pboc['PC02List'][0]['PC02GS02']
        var['循环贷账户信息汇总_授信总额'] = pboc['PC02List'][0]['PC02GJ01'].replace(',','')
        var['循环贷账户信息汇总_余额'] = pboc['PC02List'][0]['PC02GJ02'].replace(',','')
        var['循环贷最近6个月平均应还款'] = pboc['PC02List'][0]['PC02GJ03'].replace(',','')
        # 贷记卡账户信息汇总
        var['贷记卡账户信息汇总_发卡机构数'] = pboc['PC02List'][0]['PC02HS01']
        var['贷记卡账户信息汇总_账户数'] = pboc['PC02List'][0]['PC02HS02']
        var['贷记卡账户信息汇总_授信总额'] = pboc['PC02List'][0]['PC02HJ01'].replace(',','')
        var['贷记卡账户信息汇总_单家行最高授信额'] = pboc['PC02List'][0]['PC02HJ02'].replace(',','')
        var['贷记卡账户信息汇总_单家行最低授信额'] = pboc['PC02List'][0]['PC02HJ03'].replace(',','')
        var['贷记卡账户信息汇总_已用额度'] = pboc['PC02List'][0]['PC02HJ04'].replace(',','')
        var['贷记卡账户信息汇总__最近6个月平均使用额度'] = pboc['PC02List'][0]['PC02HJ05'].replace(',','')
        # 准贷记卡账户信息汇总
        var['准贷记卡账户信息汇总_发卡机构数'] = pboc['PC02List'][0]['PC02IS01']
        var['准贷记卡账户信息汇总_账户数'] = pboc['PC02List'][0]['PC02IS02']
        var['准贷记卡账户信息汇总_授信总额'] = pboc['PC02List'][0]['PC02IJ01'].replace(',','')
        var['准贷记卡账户信息汇总_单家行最高授信额'] = pboc['PC02List'][0]['PC02IJ02'].replace(',','')
        var['准贷记卡账户信息汇总_单家行最低授信额'] = pboc['PC02List'][0]['PC02IJ03'].replace(',','')
        var['准贷记卡账户信息汇总_已用额度'] = pboc['PC02List'][0]['PC02IJ04'].replace(',','')
        var['准贷记卡账户信息汇总__最近6个月平均使用额度'] = pboc['PC02List'][0]['PC02IJ05'].replace(',','')
        # 上一次查询信息
        var['上一次查询日期'] = pboc['PC02List'][0]['PC05AR01']
        var['上一次查询机构类型'] = pboc['PC02List'][0]['PC05AD01']
        var['上一次查询机构代码'] = pboc['PC02List'][0]['PC05AI01']
        var['上一次查询原因'] = pboc['PC02List'][0]['PC05AQ01']
        # 查询记录信息汇总
        var['查询记录汇总_M1_贷款审批查询机构数'] = pboc['PC02List'][0]['PC05BS01']
        var['查询记录汇总_M1_信用卡审批查询机构数'] = pboc['PC02List'][0]['PC05BS02']
        var['查询记录汇总_M1_贷款审批查询次数'] = pboc['PC02List'][0]['PC05BS03']
        var['查询记录汇总_M1_信用卡审批查询次数'] = pboc['PC02List'][0]['PC05BS04']
        
        var['查询记录汇总_M1_本人查询次数'] = pboc['PC02List'][0]['PC05BS05']
        var['查询记录汇总_Y2_贷后管理查询次数'] = pboc['PC02List'][0]['PC05BS06']
        var['查询记录汇总_Y2_担保资格审查查询次数'] = pboc['PC02List'][0]['PC05BS07']
        var['查询记录汇总_Y2_特约商户实名审查查询次数'] = pboc['PC02List'][0]['PC05BS08']
    
        # 最近6个月贷款平均应还款
        var['最近6个月贷款平均应还款'] = fillzero(var['非循环贷最近6个月平均应还款']) + fillzero(var['循环额度下分账户最近6个月平均应还款']) + fillzero(var['循环贷最近6个月平均应还款'])
        var['未销户贷记卡近6个月平均应还'] = var['贷记卡账户信息汇总__最近6个月平均使用额度']
        var['最近1个月信用卡及贷款审批查询次数'] =  add_two(fill_missing(var['查询记录汇总_M1_贷款审批查询次数']),
                                                          fill_missing(var['查询记录汇总_M1_信用卡审批查询次数']))
        
    # In[]:
    # # PF05A 住房公积金
    PF05A = pd.DataFrame()
    i = 0
    for j in pboc['PF05AList']:
        tmp = pd.DataFrame(j, index=[i])
        i += 1
        PF05A = pd.concat([PF05A, tmp])
        del tmp
    if PF05A.shape[0]>0:
        PF05A = PF05A[(PF05A['PF05AR04'] != '') & (pd.notnull(PF05A['PF05AR04']))]
        PF05A['PF05AR04'] = pd.to_datetime(PF05A['PF05AR04'])

        tmp = PF05A.sort_values(['PF05AR04'], ascending=False)
        tmp = tmp[tmp['PF05AR04']==tmp['PF05AR04'].max()]
        tmp = tmp[tmp['PF05AD01']=='1']
        if tmp.shape[0] > 0:
            var['公积金月缴存额'] = fillzero(tmp['PF05AJ01'].iloc[0].replace(',',''))
            var['公积金个人缴存比例'] = fillzero(tmp['PF05AQ03'].iloc[0].replace(',',''))
            var['公积金单位缴存比例'] = fillzero(tmp['PF05AQ02'].iloc[0].replace(',',''))
            var['个人正常月收入'] = myDivide(var['公积金月缴存额'], var['公积金个人缴存比例'] + var['公积金单位缴存比例'])*100
        
#             最近6个月平均负债收入比
            var['最近6个月平均负债收入比'] = myDivide(var['最近6个月贷款平均应还款'], var['个人正常月收入'])
        del tmp
            
    # In[]:
    # # PH01 查询明细
    PH01 = pd.DataFrame()
    i = 0
    for j in pboc['PH01List']:
        tmp = pd.DataFrame(j, index=[i])
        i += 1
        PH01 = pd.concat([PH01, tmp])
        del tmp
    if PH01.shape[0]>0:
        PH01['查询时间距报告日期天数'] = (pd.to_datetime(var['报告时间']) - pd.to_datetime(PH01['PH010R01'])).dt.days
        reasonMap = {'01':'贷后管理','02':'贷款审批','03':'信用卡审批','08':'担保资格审查','09':'司法调查','16':'公积金提取复核查询',
                    '18':'股指期货开户','19':'特约商户实名审查','20':'保前审查','21':'保后管理','22':'法人代表负责人高管等资信审查',
                    '23':'客户准入资格审查','24':'融资审批','25':'资信审查','26':'额度审批'}
        PH01['查询原因'] = PH01['PH010Q03'].map(reasonMap)
        # 查询次数
        for m in [1,3,6,9,12,24]:
            days = m*30
            v = 'M' + str(m) + '_查询次数'
            var[v] = PH01[PH01['查询时间距报告日期天数']<=days]['PH010Q02'].count()
        # 查询机构数
        for m in [1,3,6,9,12,24]:
            days = m*30
            v = 'M' + str(m) + '_查询机构数'
            var[v] = PH01[PH01['查询时间距报告日期天数']<=days]['PH010Q02'].nunique()
        # 信用卡审批查询次数
        for m in [1,3,6,9,12,24]:
            days = m*30
            v = 'M' + str(m) + '_信用卡审批查询次数'
            var[v] = PH01[(PH01['查询时间距报告日期天数']<=days)&(PH01['查询原因']=='信用卡审批')]['PH010Q02'].count()
        # 信用卡审批查询机构数
        for m in [1,3,6,9,12,24]:
            days = m*30
            v = 'M' + str(m) + '_信用卡审批查询机构数'
            var[v] = PH01[(PH01['查询时间距报告日期天数']<=days)&(PH01['查询原因']=='信用卡审批')]['PH010Q02'].nunique()
        # 贷款审批查询次数
        for m in [1,3,6,9,12,24]:
            days = m*30
            v = 'M' + str(m) + '_贷款审批查询次数'
            var[v] = PH01[(PH01['查询时间距报告日期天数']<=days)&(PH01['查询原因']=='贷款审批')]['PH010Q02'].count()
        # 贷款审批查询机构数
        for m in [1,3,6,9,12,24]:
            days = m*30
            v = 'M' + str(m) + '_贷款审批查询机构数'
            var[v] = PH01[(PH01['查询时间距报告日期天数']<=days)&(PH01['查询原因']=='贷款审批')]['PH010Q02'].nunique()
        
        # 信用卡_贷款审批查询次数 / 机构数
        for m in [1,3,6,9,12,24]:
            v1 = 'M' + str(m) + '_信用卡_贷款查询次数'
            var[v1] = var['M' + str(m) + '_信用卡审批查询次数'] + var['M' + str(m) + '_贷款审批查询次数']
            
        for m in [1,3,6,9,12,24]:
            days = m*30
            v = 'M' + str(m) + '_信用卡_贷款审批查询机构数'
            var[v] = PH01[(PH01['查询时间距报告日期天数']<=days)&(PH01['查询原因'].isin(['贷款审批','信用卡审批']))]['PH010Q02'].nunique()
        var['最近1个月信用卡及贷款审批查询机构数'] = var['M1_信用卡_贷款审批查询机构数']
        
    # In[]
    # # PD01信贷交易明细
    PD01 = pd.DataFrame()
    i = 0
    for d in pboc['PD01List']:
        tmp = pd.DataFrame(d,index=[i])
        PD01 = pd.concat([PD01, tmp])
        del tmp
        i += 1
    
    # In[]:
    if PD01.shape[0]>0:
        #机构类型代码映射
        orgMap = {'11':'商业银行','12':'村镇银行','14':'住房储蓄银行','15':'外资银行','16':'财务公司',
                 '21':'信托公司','22':'融资租赁公司','23':'汽车金融公司','24':'消费金融公司','25':'贷款公司',
                 '26':'金融资产管理公司','31':'证券公司','41':'保险公司','51':'小额贷款公司','52':'公积金管理中心',
                 '53':'融资担保公司','99':'其他机构'}
        PD01['机构类型'] = PD01['PD01AD02'].map(orgMap)
        # 业务种类
        loanType = {'11':'个人住房商业贷款','12':'个人商用房（含商住两用）贷款','13':'个人住房公积金贷款','21':'个人汽车消费贷款',
                   '31':'个人助学贷款','32':'国家助学贷款','33':'商业助学贷款','41':'个人经营性贷款','51':'农户贷款',
                   '52':'经营性农户贷款','53':'消费性农户贷款','91':'其他个人消费贷款','99':'其他贷款',
                    '71':'准贷记卡','81':'贷记卡','82':'大额专项分期卡'}
        PD01['业务种类'] = PD01['PD01AD03'].map(loanType)
        # 担保方式
        guaranteeType = {'1':'质押','2':'抵押','3':'保证','4':'信用免担保',
                   '5':'组合(含保证)','6':'组合（不含保证）','7':'农户联保','9':'其他'}
        PD01['担保方式'] = PD01['PD01AD07'].map(guaranteeType)
        # 机构代码
        PD01['机构代码'] = PD01['PD01AI02']
        # 借款金额
        PD01['借款金额'] = PD01['PD01AJ01'].apply(lambda x: str(x).replace(',',''))
        PD01['借款金额'] = pd.to_numeric(PD01['借款金额'])
        # 授信金额
        PD01['授信金额'] = PD01['PD01AJ02'].apply(lambda x: str(x).replace(',',''))
        PD01['授信金额'] = pd.to_numeric(PD01['授信金额'])
        # 共享授信金额
        PD01['共享授信金额'] = PD01['PD01AJ03'].apply(lambda x: str(x).replace(',',''))
        PD01['共享授信金额'] = pd.to_numeric(PD01['共享授信金额'])
        # 开立日期
        PD01['开立日期'] = PD01['PD01AR01']
        # 还款期数
        PD01['还款期数'] = PD01['PD01AS01']
        # 本月应还款
        PD01['本月应还款'] = PD01['PD01CJ04'].apply(lambda x: str(x).replace(',',''))
        PD01['本月应还款'] = pd.to_numeric(PD01['本月应还款'])
        # 本月实还款
        PD01['本月实还款'] = PD01['PD01CJ05'].apply(lambda x: str(x).replace(',',''))
        PD01['本月实还款'] = pd.to_numeric(PD01['本月实还款'])
        # 当前逾期总额
        PD01['当前逾期总额'] = PD01['PD01CJ06'].apply(lambda x: str(x).replace(',',''))
        PD01['当前逾期总额'] = pd.to_numeric(PD01['当前逾期总额'])
        # 当前逾期期数
        PD01['当前逾期期数'] = PD01['PD01CS02'].apply(lambda x: np.nan if x=='' else float(x))
        # 逾期31-60天未还本金
        PD01['逾期31-60天未还本金'] = PD01['PD01CJ07'].apply(lambda x: str(x).replace(',',''))
        PD01['逾期31-60天未还本金'] = pd.to_numeric(PD01['逾期31-60天未还本金'])
        # 逾期61-90天未还本金
        PD01['逾期61-90天未还本金'] = PD01['PD01CJ08'].apply(lambda x: str(x).replace(',',''))
        PD01['逾期61-90天未还本金'] = pd.to_numeric(PD01['逾期61-90天未还本金'])
        # 逾期91-180天未还本金
        PD01['逾期91-180天未还本金'] = PD01['PD01CJ09'].apply(lambda x: str(x).replace(',',''))
        PD01['逾期91-180天未还本金'] = pd.to_numeric(PD01['逾期91-180天未还本金'])
        # 逾期180天以上未还本金
        PD01['逾期180天以上未还本金'] = PD01['PD01CJ10'].apply(lambda x: str(x).replace(',',''))
        PD01['逾期180天以上未还本金'] = pd.to_numeric(PD01['逾期180天以上未还本金'])
        # 最近6个月平均使用额度
        PD01['最近6个月平均使用额度'] = PD01['PD01CJ12'].apply(lambda x: str(x).replace(',',''))
        PD01['最近6个月平均使用额度'] = pd.to_numeric(PD01['最近6个月平均使用额度'])
        # 最大使用额度
        PD01['最大使用额度'] = PD01['PD01CJ14'].apply(lambda x: str(x).replace(',',''))
        PD01['最大使用额度'] = pd.to_numeric(PD01['最大使用额度'])
        # 剩余还款期数
        PD01['剩余还款期数'] = PD01['PD01CS01']
        # 应还款日
        PD01['应还款日'] = PD01['PD01CR02']
        # 已用额度
        PD01['已用额度'] = PD01['PD01CJ02'].apply(lambda x: str(x).replace(',',''))
        PD01['已用额度'] = pd.to_numeric(PD01['已用额度'])
        # 关闭日期
        PD01['关闭日期'] = PD01['PD01BR01']
        # 币种
        PD01['币种'] = PD01['PD01AD04']
        # 余额
        PD01['余额'] = PD01.apply(getBalance, axis=1)
        PD01['余额'] = pd.to_numeric(PD01['余额'])
    
        # 贷款合同金额 D1/R4借款金额 PD01AJ01， R1授信金额 PD01AJ02
        PD01['贷款合同金额'] = PD01.apply(loanCreditAmt, axis=1)
        PD01['贷款合同金额'] = pd.to_numeric(PD01['贷款合同金额'])
        PD01['账户状态'] = PD01.apply(acountStatus, axis=1)
        PD01['五级分类'] = PD01.apply(get5class, axis=1)
        var['贷记卡、准贷记卡验证为止付_账户数'] = PD01[(PD01['PD01AD01'].isin(['R2','R3']))&
                            (PD01['账户状态'].isin(['31','3']))]['PD01AI01'].count()
        var['贷款、贷记卡、准贷记卡当前逾期_账户数'] = PD01[PD01['当前逾期期数']>0]['PD01AI01'].count()
        PD01['特殊状态'] = PD01[(PD01['账户状态']!='')&(pd.notnull(PD01['账户状态']))].apply(specialStatus,axis=1)
        var['贷记卡、准贷记卡、贷款账户状态验证_账户数'] = PD01['特殊状态'].sum()
    
    # In[]:
        loan = PD01[PD01['PD01AD01'].isin(['D1','R1','R4'])] # D1:非循环贷 R1:循环贷 R4:循环额度下分账户
        creditCard = PD01[PD01['PD01AD01'].isin(['R2'])] # R2:贷记卡  R3:准贷记卡
    
    # In[]:
    # 贷款字段
        if loan.shape[0]>0:
            loan['开立日期距离报告日期天数'] = (pd.to_datetime(var['报告时间']) - pd.to_datetime(loan['PD01AR01'])).dt.days
            loan['是否本行贷款'] = loan['PD01AI02'].apply(lambda x: 1 if '华瑞' in x else 0)
    
            # 新增贷款
            for m in [1,3,6,9,12,24]:
                days = 30*m
                v = 'M' + str(m) + '_新增贷款笔数'
                var[v] = loan[loan['开立日期距离报告日期天数']<=days]['PD01AD01'].count()
            # 本行新增贷款
            for m in [1,3,6,9,12,24]:
                days = 30*m
                v = 'M' + str(m) + '_本行新增贷款笔数'
                var[v] = loan[(loan['开立日期距离报告日期天数']<=days)&(loan['是否本行贷款']==1)]['PD01AD01'].count()
            # 他行新增贷款
            for m in [1,3,6,9,12,24]:
                days = 30*m
                v = 'M' + str(m) + '_他行新增贷款笔数'
                var[v] = loan[(loan['开立日期距离报告日期天数']<=days)&(loan['是否本行贷款']==0)]['PD01AD01'].count()
    
            # 贷款笔数
            var['贷款总笔数'] = loan[loan['PD01AD01'].isin(['D1','R4'])]['PD01AJ01'].count() + loan[loan['PD01AD01'].isin(['R1','R4'])]['PD01AJ02'].count()
            # 结清贷款笔数
            var['结清贷款笔数'] = loan[loan['账户状态']=='3']['PD01AI01'].count()
            var['结清贷款笔数占所有贷款笔数比例'] = var['结清贷款笔数'] / loan['PD01AI01'].count()
            var['最早一笔贷款发放时间距今月份'] = float(loan['开立日期距离报告日期天数'].max() / 30)
            var['贷款总金额'] = fillzero(var['非循环账户信息汇总_授信总额']) + fillzero(var['循环额度下分账户信息汇总_授信总额']) + fillzero(var['循环贷账户信息汇总_授信总额'])                  # 已经结清的贷款不在统计范围内
            var['贷款总金额LOG'] = np.log(1+var['贷款总金额'])
            var['非本行贷款机构数'] = loan[loan['是否本行贷款']==0]['PD01AI02'].nunique()
            var['商业银行贷款笔数'] = loan[loan['PD01AD02'].isin(['11','12','14','15'])]['PD01AI01'].count()
            var['小额信贷公司贷款笔数'] = loan[loan['PD01AD02'].isin(['51'])]['PD01AI01'].count()
            
            # 行内当前逾期贷款_账户数
            var['行内当前逾期贷款_账户数'] = loan[(loan['是否本行贷款']==1)&(loan['当前逾期期数']>0)]['PD01AI01'].count()
            # 贷款五级分类_非正常_账户数
            var['贷款五级分类_非正常_账户数'] = loan[loan['五级分类'].isin(['2','3','4','5','9'])]['PD01AI01'].count()
            # 贷款最高本金余额
            var['贷款最高本金余额'] = loan[(loan['账户状态']!='3')&(loan['PD01AD04']=='CNY')]['余额'].max()
            # 贷款累计本金余额
            var['贷款累计本金余额'] = loan[(loan['账户状态']!='3')&(loan['PD01AD04']=='CNY')]['余额'].sum()
            # 单家最高发放贷款金额
            var['单家最高发放贷款金额'] = loan[loan['PD01AD04']=='CNY'].groupby(['PD01AI02'])['贷款合同金额'].sum().max()
            # 贷款当前逾期
            var['贷款当前逾期期数_最大值'] = loan['当前逾期期数'].max()
            var['贷款当前逾期期数_总和'] = loan['当前逾期期数'].sum()
            # 信用贷款金额
            var['信用贷款金额'] = loan[(loan['PD01AD04']=='CNY')&(loan['PD01AD07']=='4')]['贷款合同金额'].sum()
            # 结清贷款合同金额
            var['结清贷款合同金额'] = loan[(loan['PD01AD04']=='CNY')&(loan['账户状态']=='3')]['贷款合同金额'].sum()
            # 个人消费贷款机构数
            var['个人消费贷款机构数'] = loan[loan['PD01AD03']=='91']['PD01AI02'].nunique()
            # M12 新增个人消费贷款授信金额
            var['M12新增个人消费贷款授信金额'] = loan[(loan['PD01AD04']=='CNY')&(loan['开立日期距离报告日期天数']<=360)&(loan['PD01AD03']=='91')]['贷款合同金额'].sum()
            # 行内当前未结清贷款_余额总和
            var['行内当前未结清贷款_余额'] = loan[(loan['PD01AD04']=='CNY')&(loan['是否本行贷款']==1)&(loan['账户状态']!='3')]['余额'].sum()
            # 所有贷款总金额（包含已结清和未结清）
            var['所有贷款总金额'] = loan['贷款合同金额'].sum()
            var['所有贷款总金额LOG'] = np.log(1+var['所有贷款总金额'])
            
            # 未结清贷款字段
            unpayed_loan = loan[(loan['账户状态']!='3')&(loan['账户状态']!='')&(pd.notnull(loan['账户状态']))]
            if unpayed_loan.shape[0]>0:
                var['未结清贷款合同金额'] = unpayed_loan[unpayed_loan['PD01AD04']=='CNY']['贷款合同金额'].sum()
                var['未结清贷款平均合同金额'] = unpayed_loan[(unpayed_loan['PD01AD04']=='CNY')&(unpayed_loan['贷款合同金额']>0)]['贷款合同金额'].mean()
                # 未结清_行内贷款_非正常五级分类_账户数
                var['未结清_行内贷款_非正常五级分类_账户数'] = unpayed_loan[(unpayed_loan['是否本行贷款']==1)
                                        &(unpayed_loan['五级分类'].isin(['2','3','4','5','9']))]['PD01AI01'].count()
            
    # In[]:
    # 信用卡字段
        if creditCard.shape[0]>0:
            creditCard['开立日期距离报告日期天数'] = (pd.to_datetime(var['报告时间']) - pd.to_datetime(creditCard['PD01AR01'])).dt.days
            creditCard['开立日期距今月份'] = creditCard['开立日期距离报告日期天数'] / 30
            # 新开信用卡
            for m in [1,3,6,9,12,24]:
                days = 30*m
                v = 'M' + str(m) + '_新开信用卡张数'
                var[v] = creditCard[creditCard['开立日期距离报告日期天数']<=days]['PD01AD01'].count()
    
            var['最早信用卡开卡距今月份'] = creditCard[~creditCard['账户状态'].isin(['4','6'])]['开立日期距今月份'].max()
            var['最近信用卡开卡距今月份'] = creditCard[~creditCard['账户状态'].isin(['4','6'])]['开立日期距今月份'].min()
            var['信用卡当前逾期期数_最大值'] = creditCard['当前逾期期数'].max()
            var['信用卡当前逾期期数_总和'] = creditCard['当前逾期期数'].sum()
            var['活跃的信用卡账户数'] = creditCard[(creditCard['已用额度']>0)&(creditCard['账户状态'].isin(['4','6'])==False)]['PD01AI01'].count()
    
    #del loan,creditCard
    
    # In[]:
    # PY01近5年 / 近24个月还款记录
    PY01 = pd.DataFrame()
    i = 0
    for d in pboc['PY01List']:
        tmp = pd.DataFrame(d,index=[i])
        PY01 = pd.concat([PY01, tmp])
        del tmp
        i += 1
    
    if PY01.shape[0]>0:
        PY01['最近5年还款记录'] = PY01.apply(latest5years, axis=1)
    
        # 贷款(D1, R1, R4) and 贷记卡(R2)
        loan_creditcard = PD01.merge(PY01, on='PD01AI01', how='outer')
        loan_creditcard = loan_creditcard[loan_creditcard['PY01_TYPE']=='PD01E']
    else:
        loan_creditcard = pd.DataFrame()
    
    # In[]:
    if loan_creditcard.shape[0] > 0:
        # 最近24月状态起始年月
        loan_creditcard['最近24月状态起始年月'] = loan_creditcard['PD01DR01']
        # 最近24月状态截至年月
        loan_creditcard['最近24月状态截至年月'] = loan_creditcard['PD01DR02']
        # 最近5年状态起始年月
        loan_creditcard['最近5年状态起始年月'] = loan_creditcard['PD01ER01']
        # 最近5年状态截至年月
        loan_creditcard['最近5年状态截至年月'] = loan_creditcard['PD01ER02']
        # 最近5年还款记录有记录的月数
        loan_creditcard['最近5年还款记录有记录的月数'] = loan_creditcard['PD01ES01']
        # 最近5年还款记录有记录的月数
        loan_creditcard['最近5年还款记录or最近24月还款记录标志'] = loan_creditcard['PY01_TYPE']
        
        loan_creditcard['账户状态'] = loan_creditcard.apply(acountStatus, axis=1)
        loan_creditcard['五级分类'] = loan_creditcard.apply(get5class, axis=1)
        
        loan_creditcard['latest24MonStatus'] = loan_creditcard['最近5年还款记录'].apply(lambda x: str(x).rjust(24,'S'))

        # 特殊结清
        tmp = loan_creditcard[loan_creditcard['账户状态'].isin(['1','2'])]
        tmp['status'] = tmp['latest24MonStatus'].apply(lambda x: x[-24:])
        tmp['特殊结清次数'] = tmp['status'].apply(lambda x: specialPay(x))
        var['贷记卡、准贷记卡、贷款最近24期状态_账户数'] = tmp['特殊结清次数'].sum()
    
    # In[]:
    # 贷款 and 信用卡逾期
        loan = loan_creditcard[loan_creditcard['PD01AD01'].isin(['D1','R1','R4'])] # D1:非循环贷 R1:循环贷 R4:循环额度下分账户
        loan['是否本行贷款'] = loan['PD01AI02'].apply(lambda x: 1 if '华瑞' in x else 0)
        creditCard = loan_creditcard[loan_creditcard['PD01AD01'].isin(['R2'])] # R2:贷记卡  R3:准贷记卡
        
    # In[]:
    # 贷款逾期
        if loan.shape[0]>0:
            # 贷款逾期次数 / 最大连续逾期次数
            loan['latest24MonStatus'] = loan['最近5年还款记录'].apply(lambda x: str(x).rjust(24,'S'))
            tmp = loan[loan['账户状态'].isin(['1','2'])].copy()
            for m in [1,3,6,12,24]:
                v1 = '贷款最近' + str(m) + '个月发生逾期次数'
                v2 = '近' + str(m) + '月贷款最大连续逾期次数'
                tmp['status'] = tmp['latest24MonStatus'].apply(lambda x: x[-m:])
                tmp['出现数字次数'] = tmp['status'].apply(lambda x: latest24(x)[0])
                tmp['连续数字最大长度'] = tmp['status'].apply(lambda x: latest24(x)[1])
                var[v1] = tmp['出现数字次数'].sum()
                var[v2] = tmp['连续数字最大长度'].max()
            
            tmp = loan[(loan['账户状态'].isin(['1','2']))&(loan['是否本行贷款']==1)].copy()
            tmp['status'] = tmp['latest24MonStatus'].apply(lambda x: x[-24:])
            tmp['出现数字次数'] = tmp['status'].apply(lambda x: latest24(x)[0])
            tmp['连续数字最大长度'] = tmp['status'].apply(lambda x: latest24(x)[1])
            var['近2年内行内贷款_最大连续逾期期数'] = tmp['连续数字最大长度'].max()
            var['近2年内行内贷款_累计逾期次数'] = tmp['出现数字次数'].sum()
    
            # 个人住房贷款逾期
            personhouseloan = loan[loan['PD01AD03']=='11']
            personhouseloan['latest24MonStatus'] = personhouseloan['最近5年还款记录'].apply(lambda x: str(x).rjust(24,'S'))
            tmp = personhouseloan[personhouseloan['账户状态'].isin(['1','2'])].copy()
            tmp['status'] = tmp['latest24MonStatus'].apply(lambda x: x[-24:])
            tmp['出现数字次数'] = tmp['status'].apply(lambda x: latest24(x)[0])
            var['近24个月个人住房贷款逾期次数'] = tmp['出现数字次数'].sum()
            
    # In[]:
    # 信用卡逾期
        if creditCard.shape[0]>0:
            creditCard['latest24MonStatus'] = creditCard['最近5年还款记录'].apply(lambda x: str(x).rjust(24,'S'))
            tmp = creditCard[creditCard['账户状态'].isin(['1','2'])].copy()
            for m in [1,3,6,12,24]:
                v1 = '信用卡最近' + str(m) + '个月发生逾期次数'
                v2 = '近' + str(m) + '月信用卡最大连续逾期次数'
                tmp['status'] = tmp['latest24MonStatus'].apply(lambda x: x[-m:])
                tmp['出现数字次数'] = tmp['status'].apply(lambda x: latest24(x)[0])
                tmp['连续数字最大长度'] = tmp['status'].apply(lambda x: latest24(x)[1])
                var[v1] = tmp['出现数字次数'].sum()
                var[v2] = tmp['连续数字最大长度'].max()
            
        # In[]:
        # # 交叉变量
        for m in [1,3,12,24]:
            v = '近' + str(m) + '月最大连续逾期次数'
            if (loan.shape[0]>0) & (creditCard.shape[0]>0):
                var[v] = np.nanmax([var['近'+str(m)+'月贷款最大连续逾期次数'],var['近'+str(m)+'月信用卡最大连续逾期次数']])
            elif (loan.shape[0]==0) & (creditCard.shape[0]>0):
                var[v] = var['近'+str(m)+'月信用卡最大连续逾期次数']
            elif (creditCard.shape[0] == 0) & (loan.shape[0]>0):
                var[v] = var['近'+str(m)+'月贷款最大连续逾期次数']

    # In[]:
    # # PD03A 对外担保
    PD03A = pd.DataFrame()
    i = 0
    for j in pboc['PD03AList']:
        tmp = pd.DataFrame(j ,index=[i])
        i += 1
        PD03A = pd.concat([PD03A, tmp])
        del tmp
    if PD03A.shape[0]>0:
        PD03A['PD03AJ02'] = PD03A['PD03AJ02'].apply(lambda x: str(x).replace(',',''))
        PD03A['PD03AJ02'] = pd.to_numeric(PD03A['PD03AJ02'])
        
        tmp = PD03A[(PD03A['PD03AD03']=='2')&(PD03A['PD03AD05']!='1')]
        if tmp.shape[0]>0:
            var['对外贷款担保_非正常五级分类_余额'] = tmp['PD03AJ02'].sum()
        del tmp
        
        tmp = PD03A[(PD03A['PD03AD03']=='2')&(PD03A['PD03AD05']!='1')&(PD03A['PD03AD07']!='3')]
        if tmp.shape[0]>0:
            var['未结清_对外担保_非正常五级分类_账户数'] = tmp['PA01AI01'].count()
        del tmp
    return var

# In[]:
def pboc_result(x):
    NUM_VAR_LIST = [
    '个人住房贷款笔数',
    '最近1个月信用卡及贷款审批查询次数',
    '最早信用卡开卡距今月份',
    '结清贷款笔数占所有贷款笔数比例',
    '最早一笔贷款发放时间距今月份',
    '贷款总金额',
    '非本行贷款机构数',
    '商业银行贷款笔数',
    '小额信贷公司贷款笔数',
    '未结清贷款平均合同金额',
    '贷款最长逾期月数',
    '未结清_行内贷款_非正常五级分类_账户数',
    '近2年内行内贷款_最大连续逾期期数',
    '近2年内行内贷款_累计逾期次数',
    '行内当前逾期贷款_账户数',
    '数字解读分',
    '呆账笔数',
    '资产处置笔数',
    '保证人代偿笔数',
    '贷款五级分类_非正常_账户数',
    '对外贷款担保_非正常五级分类_余额',
    '贷记卡、准贷记卡、贷款账户状态验证_账户数',
    '贷记卡、准贷记卡验证为止付_账户数',
    '贷记卡、准贷记卡、贷款最近24期状态_账户数',
    '贷款、贷记卡、准贷记卡当前逾期_账户数',
    '欠税信息',
    '民事判决',
    '强制执行',
    '行政处罚',
    '贷款总金额LOG',
    '最近1个月信用卡及贷款审批查询机构数',
    '贷款总笔数',
    '贷款最高本金余额',
    '贷款累计本金余额',
    '单家最高发放贷款金额',
    '贷款当前逾期期数_最大值',
    '贷款当前逾期期数_总和',
    '信用卡当前逾期期数_最大值',
    '信用卡当前逾期期数_总和',
    '信用卡最近1个月发生逾期次数',
    '信用卡最近3个月发生逾期次数',
    '信用卡最近6个月发生逾期次数',
    '信用卡最近12个月发生逾期次数',
    '信用卡最近24个月发生逾期次数',
    '近1月最大连续逾期次数',
    '近3月最大连续逾期次数',
    '近12月最大连续逾期次数',
    '近24月最大连续逾期次数',
    'M3_信用卡_贷款查询次数',
    '公积金月缴存额',
    '公积金个人缴存比例',
    '公积金单位缴存比例',
    '个人正常月收入',
    '非循环贷最近6个月平均应还款',
    '循环贷最近6个月平均应还款',
    '循环额度下分账户最近6个月平均应还款',
    '最近6个月贷款平均应还款',
    '最近6个月平均负债收入比',
    '信用贷款金额',
    '结清贷款合同金额',
    '结清贷款笔数',
    '个人消费贷款机构数',
    'M12新增个人消费贷款授信金额',
    '是否本科及以上学历',
    '近1月信用卡最大连续逾期次数',
    '近3月信用卡最大连续逾期次数',
    '近6月信用卡最大连续逾期次数',
    '近12月信用卡最大连续逾期次数',
    '近24月信用卡最大连续逾期次数',
    '近1月贷款最大连续逾期次数',
    '近3月贷款最大连续逾期次数',
    '近12月贷款最大连续逾期次数',
    '近24月贷款最大连续逾期次数',
    '贷款最近1个月发生逾期次数',
    '贷款最近3个月发生逾期次数',
    '贷款最近6个月发生逾期次数',
    '贷款最近12个月发生逾期次数',
    '贷款最近24个月发生逾期次数',
    '未结清_对外担保_非正常五级分类_账户数',
    '行内当前未结清贷款_余额',
    '所有贷款总金额',
    '学历详细分类',
    '未销户贷记卡近6个月平均应还',
    '最近信用卡开卡距今月份',
    '活跃的信用卡账户数',
    '近24个月个人住房贷款逾期次数',
    '所有贷款总金额LOG']
    
    CHAR_VAR_LIST = ['婚姻状态','出生日期','性别','学历','学位','手机号码']
    # 字段初始化
    var = {}
    for i in NUM_VAR_LIST:
        var[i] = -9999
    for i in CHAR_VAR_LIST:
        var[i] = 'null'
    # 生成字段
    var1 = parsePboc(x)
    var.update(var1)
    for i in NUM_VAR_LIST:
        if (var[i] == '') | pd.isnull(var[i]):
            var[i] = -9999
        var[i] = float(var[i])
        
    for i in CHAR_VAR_LIST:
        if (var[i] == '') | pd.isnull(var[i]):
            var[i] = 'null'
        var[i] = str(var[i])
    return var
