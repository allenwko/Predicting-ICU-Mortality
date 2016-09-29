import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle
import time
import re
import glob

from xgboost import XGBClassifier
from sklearn.cross_validation import cross_val_score, train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_curve, roc_auc_score, confusion_matrix
from sklearn.grid_search import GridSearchCV

def pr_curve(truthvec, scorevec, name, digit_prec=2):
    threshvec = np.unique(np.round(scorevec,digit_prec))
    numthresh = len(threshvec)
    tpvec = np.zeros(numthresh)
    fpvec = np.zeros(numthresh)
    fnvec = np.zeros(numthresh)

    for i in range(numthresh):
        thresh = threshvec[i]
        tpvec[i] = sum(truthvec[scorevec>=thresh])
        fpvec[i] = sum(1-truthvec[scorevec>=thresh])
        fnvec[i] = sum(truthvec[scorevec<thresh])
    recallvec = tpvec/(tpvec + fnvec)
    precisionvec = tpvec/(tpvec + fpvec)
    plt.subplot(121)
    plt.plot(precisionvec,recallvec, label = name)
    plt.axis([0, 1, 0, 1])
#    plt.legend(loc=3)
    plt.title('PR Curves')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    return (recallvec, precisionvec, threshvec)

def build_feat(hyp_df, items):
    '''Takes dataframe with icustay_id, patient intime, and takes a list of itemids
        Outputs a dataframe with most recent measurements at 24 hours after admission'''
    intime_df = hyp_df[['icustay_id', 'intime']]
    shapelist = []

    count = 0
    start_time = time.time()
    total = len(items)
    for item in items:
        shapedict = {}
        item = str(item)
        # Print status
        if count != 0:
            rem = ((float(total) - count)/ count * (time.time()-start_time))/60
            timeper = (time.time() - start_time)/count
            print '\r%d/%d, %.1f min est. time remaining, %.1f s per item.' % (count, total, rem, timeper),

        # Get the itemid table
        fname = 'items/itemid' + item + '.p'
        if not glob.glob(fname):
            print 'fetching item %d from sql server' % int(item)
            que = '''select icustay_id, itemid, value, valuenum, charttime from chartevents where itemid=%d''' % int(item)
            buffer_df = pd.read_sql(que, db)
            buffer_df.to_pickle(fname)
        else:
            buffer_df = pd.read_pickle(fname)
        buffer_df.drop(['itemid'], axis=1, inplace=True)

        #check to get value or valuenum
        if float_check(buffer_df.sample(30).value, 0.2):
            buffer_df.rename(columns = {'valuenum':item}, inplace=True)
        else:
            buffer_df.rename(columns = {'value':item}, inplace=True)
        buffer_df.dropna(subset=['icustay_id'], inplace=True)

        #group by icustay_id, get the most recent measurement at 24 hours after admit
        buffer_df = buffer_df.merge(intime_df, on='icustay_id', how = 'left')
        buffer_df.dropna(subset=['intime'], inplace=True)
        if not buffer_df.empty:
            buffer_df['meas_time'] = buffer_df.charttime - buffer_df.intime
            if not buffer_df[buffer_df.meas_time < pd.Timedelta('24 hours')].empty:
                time_group = buffer_df[buffer_df.meas_time < pd.Timedelta('24 hours')].groupby('icustay_id')
                buffer_group = None
                meas_df = time_group.apply(lambda x: x.sort_values(by='charttime', ascending=False).iloc[0])
                time_group = None
                meas_df.drop(['icustay_id'], axis=1, inplace=True)
                meas_df.reset_index(inplace=True)

                #attach values to hyp_df
                hyp_df = hyp_df.merge(meas_df[['icustay_id', item]], on='icustay_id', how='left')
                meas_df = None
                
        shapedict['count'] = count
        shapedict['shape'] = hyp_df.shape
        shapedict['item'] = item
        shapelist.append(shapedict)

        count += 1
    
    print 'done!'
    return hyp_df

def build_feat_ts(hyp_df, items):
    '''Takes dataframe with icustay_id, patient intime, and takes a list of itemids
        Outputs a dataframe with most recent measurements at 24 hours after admission
        Operates on time series, and returns median, std, and range.
        The intent is to merge with the results from function build_feat'''
    intime_df = hyp_df[['icustay_id', 'intime']]
    final_df = hyp_df[['icustay_id']]
#     shapelist = [] # for debugging

    count = 0
    start_time = time.time()
    total = len(items)
    for item in items:
        shapedict = {}
        item = str(item)
        # Print status
        if count != 0:
            rem = ((float(total) - count)/ count * (time.time()-start_time))/60
            timeper = (time.time() - start_time)/count
            print '\r%d/%d, %.1f min est. time remaining, %.1f s per item.' % (count, total, rem, timeper),

        # Get the itemid table
        fname = 'items/itemid' + item + '.p'
        if not glob.glob(fname):
            print 'fetching item %d from sql server' % item
            que = '''select icustay_id, itemid, value, valuenum, charttime from chartevents where itemid=%d''' % item
            buffer_df = pd.read_sql(que, db)
            buffer_df.to_pickle(fname)
        else:
            buffer_df = pd.read_pickle(fname)
        buffer_df.drop(['itemid'], axis=1, inplace=True)

        #check to get value or valuenum
        if float_check(buffer_df.sample(30).value, 0.2):
            buffer_df.rename(columns = {'valuenum':item}, inplace=True)
        else:
            buffer_df.rename(columns = {'value':item}, inplace=True)
        buffer_df.dropna(subset=['icustay_id'], inplace=True)

        #group by icustay_id, get the most recent measurement at 24 hours after admit
        buffer_df = buffer_df.merge(intime_df, on='icustay_id', how = 'left')
        buffer_df.dropna(subset=['intime'], inplace=True)
        if not buffer_df.empty:
            buffer_df['meas_time'] = buffer_df.charttime - buffer_df.intime
#            if not buffer_df[buffer_df.meas_time < pd.Timedelta('24 hours')].empty:
            time_group = buffer_df[buffer_df.meas_time < pd.Timedelta('24 hours')].groupby('icustay_id')
            buffer_df = None
            meas_df = time_group.agg({item: [np.mean, np.std, np.ptp]})
            time_group = None
            meas_df.reset_index()
            meas_df.columns= meas_df.columns.droplevel()
            col_names = {}
            for col in meas_df.columns:
                col_names[col] = str(item)+'_'+col
            meas_df.rename(columns = col_names, inplace=True)
            meas_df.reset_index(inplace=True)

            #attach values to hyp_df
            final_df = final_df.merge(meas_df, on='icustay_id', how='left')
            meas_df = None
#         shapedict['count'] = count
#         shapedict['shape'] = hyp_df.shape
#         shapedict['item'] = item
#         shapelist.append(shapedict)
        count += 1

    print 'done!'
    return final_df
