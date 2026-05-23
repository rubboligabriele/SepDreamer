import pandas as pd
import numpy as np
from preprocessing.utils.columns import *

# fill-in missing ICUSTAY IDs in bacterio and abx. We will look at their subject ID
# and find a matching ICU stay ID such that the event takes place
# within the admission
def impute_icustay_ids(demog, target, window=48 * 3600):
    """
    Finds an ICU stay ID from the demog table such that the 
    subject ID matches, and the admission in and out times wrap
    around the given chart time. If subject_id_col is None,
    uses hadm ID only.
    """
    filtered_demog = demog[[C_SUBJECT_ID, C_HADM_ID, C_ICUSTAYID, C_INTIME, C_OUTTIME]]
    
    if C_SUBJECT_ID in target.columns:
        filtered_demog = filtered_demog[filtered_demog[C_SUBJECT_ID].isin(target[C_SUBJECT_ID])]
        subject_id_groups = filtered_demog.groupby(C_SUBJECT_ID).groups
        hadm_groups = filtered_demog.groupby(C_HADM_ID).groups

        def impute(row):
            same_subj = subject_id_groups.get(int(row[C_SUBJECT_ID]), [])
            if len(same_subj) >= 1:
                matching_rows = demog.iloc[same_subj]
                matching_rows = matching_rows[(matching_rows[C_INTIME] <= row[C_CHARTTIME] + window) &
                                              (matching_rows[C_OUTTIME] >= row[C_CHARTTIME] - window)]
                if len(matching_rows) > 0:
                    return matching_rows.iloc[0][C_ICUSTAYID]
            
            # Now check hadm ID and just grab the first one
            if not pd.isna(row[C_HADM_ID]):
                same_hadm = hadm_groups.get(int(row[C_HADM_ID]), [])
                if len(same_hadm) == 1:
                    return demog.iloc[same_hadm[0]][C_ICUSTAYID]
            return None
        return target.progress_apply(impute, axis=1)
    else:
        filtered_demog = filtered_demog[filtered_demog[C_HADM_ID].isin(target[C_HADM_ID])]
        hadm_groups = filtered_demog.groupby(C_HADM_ID).groups

        def impute(row):
            # Just check hadm ID
            if not pd.isna(row[C_HADM_ID]):
                same_hadm = hadm_groups.get(int(row[C_HADM_ID]), [])
                if len(same_hadm) == 1:
                    return demog.iloc[same_hadm[0]][C_ICUSTAYID]
            return None
        return target.progress_apply(impute, axis=1)

LOG_OUTLIERS = True

def is_outlier(col, lower=None, upper=None):
    if lower is not None and upper is not None:
        result = (col < lower) | (col > upper)
    elif lower is not None:
        result = col < lower
    elif upper is not None:
        result = col > upper
    else:
        result = np.zeros(len(col))
    if LOG_OUTLIERS: print('(' + str(result.sum()) + ' outliers) ', end='')
    return result

def fill_outliers(df, spec, provenance=None):
    """
    Remove outliers according to a specification. Each key in the
    spec dictionary should correspond to a column in df, and the value
    should be a tuple (lower, upper) indicating the lower and upper limits
    for allowed values in that column. Other values will be set to pd.NA.
    
    A modified copy of the dataframe is returned.
    """
    copy_df = df.copy()
    for col, (min_val, max_val) in spec.items():
        print('filtering', col, end=' ')
        outliers = is_outlier(copy_df[col], min_val, max_val)
        if provenance:
            provenance.record("outlier", row=copy_df.loc[outliers, C_ICUSTAYID], col=col)
        copy_df.loc[outliers, col] = pd.NA
        print('')
    return copy_df
