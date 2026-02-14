import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm

from src.preprocessing.provenance import ProvenanceWriter
from src.preprocessing.columns import *
from src.preprocessing.utils import load_csv
from src.preprocessing.imputation import fill_outliers, fill_stepwise, sample_and_hold

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

def remove_outliers(df, provenance=None):
    # Transfer temperatures tagged as celsius but obviously fahrenheit
    wrong_unit_temps = (df[C_TEMP_C] > 90) & pd.isna(df[C_TEMP_F])
    if provenance:
        provenance.record("temp_F logged as temp_C", row=df.loc[wrong_unit_temps].index, col=C_TEMP_F, reference_col=C_TEMP_C)
    df.loc[wrong_unit_temps, C_TEMP_F] = df.loc[wrong_unit_temps, C_TEMP_C]

    # Multiply FiO2 to be in percentage instead of fraction
    if provenance:
        provenance.record("FiO2_100 < 1", row=df.loc[df[C_FIO2_100] < 1].index, col=C_FIO2_100)
    df.loc[df[C_FIO2_100] < 1, C_FIO2_100] = df.loc[df[C_FIO2_100] < 1, C_FIO2_100] * 100

    df = fill_outliers(df, {
        # Vitals
        C_WEIGHT: (None, 300),      # weight
        C_HR: (None, 250),      # HR
        C_SYSBP: (0, 300),         # BP
        C_MEANBP: (0, 200),
        C_DIABP: (0, 200),
        C_RR: (None, 80),      # RR
        C_SPO2: (None, 150),     # SpO2
        C_TEMP_C: (None, 90),      # temp
        C_FIO2_100: (20, None),      # FiO2
        C_FIO2_1: (None, 1.5),
        C_O2FLOW: (None, 70),      # O2 flow
        C_PEEP: (0, 40),         # PEEP
        C_TIDALVOLUME: (None, 1800),    # Tidal volume
        C_MINUTEVENTIL: (None, 50),      # Minute volume
        
        # Labs
        C_POTASSIUM: (1, 15),         # K+
        C_SODIUM: (95, 178),       # Na+
        C_CHLORIDE: (70, 150),       # Cl
        C_GLUCOSE: (1, 1000),       # Glc
        C_CREATININE: (None, 150),     # creatinine
        C_MAGNESIUM: (None, 10),      # Mg
        C_CALCIUM: (None, 20),      # Ca
        C_IONISED_CA: (None, 5),       # ionized Ca
        C_CO2_MEQL: (None, 120),     # CO2
        C_SGPT: (None, 10000),   # SGPT
        C_SGOT: (None, 10000),   # SGOT
        C_HB: (None, 20),      # hemoglobin
        C_HT: (None, 65),      # hematocrit
        C_WBC_COUNT: (None, 500),     # WBC
        C_PLATELETS_COUNT: (None, 2000),    # platelet
        C_INR: (None, 20),      # INR
        C_ARTERIAL_PH: (6.7, 8),        # pH
        C_PAO2: (None, 700),     # PO2
        C_PACO2: (None, 200),     # PCO2
        C_ARTERIAL_BE: (-50, None),     # base excess
        C_ARTERIAL_LACTATE: (None, 30),      # lactate
    }, provenance=provenance)

    # Clamp SpO2 to 100 after removing outliers
    if provenance:
        provenance.record("Clamp SpO2 > 100", row=df.loc[df[C_SPO2] > 100].index, col=C_SPO2)
    df.loc[df[C_SPO2] > 100, C_SPO2] = 100
    
    return df

def convert_fio2_units(df, provenance=None, metadata=None):
    """Converts FiO2 Set (a.k.a. FiO2_1) to FiOS (a.k.a. FiO2_100)."""
    # approximate that FiO2 Set (col 24) ~= FiO2 (col 23) / 100 
    missing_fio2_set = pd.isna(df[C_FIO2_1]) & ~pd.isna(df[C_FIO2_100])
    if provenance:
        provenance.record("Transfer FiO2_1 from FiO2_100", row=df.loc[missing_fio2_set].index, col=C_FIO2_1, reference_col=C_FIO2_100, metadata=metadata)
    df.loc[missing_fio2_set, C_FIO2_1] = df.loc[missing_fio2_set, C_FIO2_100] / 100
    missing_fio2 = ~pd.isna(df[C_FIO2_1]) & pd.isna(df[C_FIO2_100])
    if provenance:
        provenance.record("Transfer FiO2_100 from FiO2_1", row=df.loc[missing_fio2].index, col=C_FIO2_100, reference_col=C_FIO2_1, metadata=metadata)
    df.loc[missing_fio2, C_FIO2_100] = df.loc[missing_fio2, C_FIO2_1] * 100
    return df
    
def estimate_fio2(df, provenance=None):
    # First fill in missing values
    df = convert_fio2_units(df, provenance=provenance, metadata="R1")
    
    sah_fio2 = {}
    for col in [C_INTERFACE, C_FIO2_100, C_O2FLOW]:    
        print("SAH on " + col)
        sah_fio2[col] = sample_and_hold(df[C_ICUSTAYID], df[C_TIMESTEP], df[col], SAH_HOLD_DURATION[col])
        print("Eliminated {:.1f}% of NA values".format((1 - pd.isna(sah_fio2[col]).sum() / pd.isna(df[col]).sum()) * 100))
    
    # NO FiO2, YES O2 flow, no interface OR cannula
    print('NO FiO2, YES O2 flow, no interface OR cannula ', end='')
    mask = (pd.isna(sah_fio2[C_FIO2_100]) & ~pd.isna(sah_fio2[C_O2FLOW]) & 
            ((sah_fio2[C_INTERFACE] == 0) | (sah_fio2[C_INTERFACE] == 2)))
    print('({} rows) '.format(mask.sum()), end='')
    if mask.any():
        if provenance:
            provenance.record("FiO2 estimation", row=df.loc[mask].index, col=C_FIO2_100, metadata="C1")
        df.loc[mask, C_FIO2_100] = fill_stepwise(sah_fio2[C_O2FLOW].loc[mask], zip(*(
            [15, 12, 10, 8, 6, 5, 4, 3, 2, 1],
            [70, 62, 55, 50, 44, 40, 36, 32, 28, 24]
        )))
    print('[DONE]')

    # NO FiO2, NO O2 flow, no interface OR cannula
    print('NO FiO2, NO O2 flow, no interface OR cannula ', end='')
    mask = (pd.isna(sah_fio2[C_FIO2_100]) & pd.isna(sah_fio2[C_O2FLOW]) & 
            ((sah_fio2[C_INTERFACE] == 0) | (sah_fio2[C_INTERFACE] == 2)))
    print('({} rows) '.format(mask.sum()), end='')
    if mask.any():
        if provenance:
            provenance.record("FiO2 estimation", row=df.loc[mask].index, col=C_FIO2_100, metadata="C2")
        df.loc[mask, C_FIO2_100] = 21
    print('[DONE]')

    # NO FiO2, YES O2 flow, face mask OR.... OR ventilator (assume it's face mask)
    print('NO FiO2, YES O2 flow, face mask OR.... OR ventilator (assume it\'s face mask) ', end='')
    mask = (pd.isna(sah_fio2[C_FIO2_100]) & ~pd.isna(sah_fio2[C_O2FLOW]) & 
            (pd.isna(sah_fio2[C_INTERFACE]) | sah_fio2[C_INTERFACE].isin((1, 3, 4, 5, 6, 9, 10))))
    print('({} rows) '.format(mask.sum()), end='')
    if mask.any():
        if provenance:
            provenance.record("FiO2 estimation", row=df.loc[mask].index, col=C_FIO2_100, metadata="C3")
        df.loc[mask, C_FIO2_100] = fill_stepwise(sah_fio2[C_O2FLOW].loc[mask], zip(*(
            [15, 12, 10, 8, 6, 4],
            [75, 69, 66, 58, 40, 36]
        )))
    print('[DONE]')

    # NO FiO2, NO O2 flow, face mask OR ....OR ventilator
    print('NO FiO2, NO O2 flow, face mask OR ....OR ventilator ', end='')
    mask = (pd.isna(sah_fio2[C_FIO2_100]) & pd.isna(sah_fio2[C_O2FLOW]) & 
            (pd.isna(sah_fio2[C_INTERFACE]) | sah_fio2[C_INTERFACE].isin((1, 3, 4, 5, 6, 9, 10))))
    print('({} rows) '.format(mask.sum()), end='')
    if mask.any():
        if provenance:
            provenance.record("FiO2 estimation", row=df.loc[mask].index, col=C_FIO2_100, metadata="C4")
        df.loc[mask, C_FIO2_100] = pd.NA
    print('[DONE]')

    # NO FiO2, YES O2 flow, Non rebreather mask
    print('NO FiO2, YES O2 flow, Non rebreather mask ', end='')
    mask = (pd.isna(sah_fio2[C_FIO2_100]) & ~pd.isna(sah_fio2[C_O2FLOW]) & sah_fio2[C_INTERFACE] == 7)
    print('({} rows) '.format(mask.sum()), end='')
    if mask.any():
        if provenance:
            provenance.record("FiO2 estimation", row=df.loc[mask].index, col=C_FIO2_100, metadata="C5")
        df.loc[mask, C_FIO2_100] = fill_stepwise(sah_fio2[C_O2FLOW].loc[mask], zip(*(
            [9.99, 8, 6],
            [80, 70, 60]
        )), zip(*(
            [10, 15],
            [90, 100]
        )))
    print('[DONE]')

    # NO FiO2, NO O2 flow, NRM
    print('NO FiO2, NO O2 flow, NRM ', end='')
    mask = (pd.isna(sah_fio2[C_FIO2_100]) & pd.isna(sah_fio2[C_O2FLOW]) & sah_fio2[C_INTERFACE] == 7)
    print('({} rows) '.format(mask.sum()), end='')
    if mask.any():
        if provenance:
            provenance.record("FiO2 estimation", row=df.loc[mask].index, col=C_FIO2_100, metadata="C6")
        df.loc[mask, C_FIO2_100] = pd.NA
    print('[DONE]')

    # update again FiO2 columns
    df = convert_fio2_units(df, provenance=provenance, metadata="R2")
    print('[DONE]')
    return df

def estimate_gcs(rass):
    """
    Estimates the Glasgow Coma Scale value from the value of
    the Richmond Agitation Sedation Scale.
    """
    if rass >= 0: return 15
    elif rass == -1: return 14
    elif rass == -2: return 12
    elif rass == -3: return 11
    elif rass == -4: return 6
    elif rass == -5: return 3
    return pd.NA

def estimate_vitals(df, provenance=None):
    # BP - if we have two values, we can impute the others using the definition of mean BP
    print('BP ', end='')
    ii = ~pd.isna(df.loc[:, C_SYSBP]) & ~pd.isna(df.loc[:, C_MEANBP]) & pd.isna(df.loc[:, C_DIABP])
    if provenance:
        provenance.record("BP estimation", row=df.loc[ii].index, col=C_DIABP)
    df.loc[ii, C_DIABP] = (3 * df.loc[ii, C_MEANBP] - df.loc[ii, C_SYSBP]) / 2
    ii = ~pd.isna(df.loc[:, C_SYSBP]) & ~pd.isna(df.loc[:, C_DIABP]) & pd.isna(df.loc[:, C_MEANBP])
    if provenance:
        provenance.record("BP estimation", row=df.loc[ii].index, col=C_MEANBP)
    df.loc[ii, C_MEANBP] = (df.loc[ii, C_SYSBP] + 2 * df.loc[ii, C_DIABP]) / 3
    ii = ~pd.isna(df.loc[:, C_MEANBP]) & ~pd.isna(df.loc[:, C_DIABP]) & pd.isna(df.loc[:, C_SYSBP])
    if provenance:
        provenance.record("BP estimation", row=df.loc[ii].index, col=C_SYSBP)
    df.loc[ii, C_SYSBP] = 3 * df.loc[ii, C_MEANBP] - 2 * df.loc[ii, C_DIABP]
    print('[DONE]')

    # TEMP
    # some values recorded in the wrong column
    print('TEMP ', end='')
    ii = (df.loc[:, C_TEMP_F] > 25) & (df.loc[:, C_TEMP_F] < 45)  # tempF close to 37deg??!
    if provenance:
        provenance.record("Temp_C from Temp_F", row=df.loc[ii].index, col=C_TEMP_C, reference_col=C_TEMP_F)
    df.loc[ii, C_TEMP_C] = df.loc[ii, C_TEMP_F]
    df.loc[ii, C_TEMP_F] = np.nan
    ii = (df.loc[:, C_TEMP_C] > 70)  # tempC > 70?!!! probably degF
    if provenance:
        provenance.record("Temp_F from Temp_C", row=df.loc[ii].index, col=C_TEMP_F, reference_col=C_TEMP_C)
    df.loc[ii, C_TEMP_F] = df.loc[ii, C_TEMP_C]
    df.loc[ii, C_TEMP_C] = np.nan
    
    ii = ~pd.isna(df.loc[:, C_TEMP_C]) & pd.isna(df.loc[:, C_TEMP_F])
    if provenance:
        provenance.record("Calculate Temp_F from Temp_C", row=df.loc[ii].index, col=C_TEMP_F, reference_col=C_TEMP_C)
    df.loc[ii, C_TEMP_F] = df.loc[ii, C_TEMP_C] * 1.8 + 32
    ii = ~pd.isna(df.loc[:, C_TEMP_F]) & pd.isna(df.loc[:, C_TEMP_C])
    if provenance:
        provenance.record("Calculate Temp_C from Temp_F", row=df.loc[ii].index, col=C_TEMP_C, reference_col=C_TEMP_F)
    df.loc[ii, C_TEMP_C] = (df.loc[ii, C_TEMP_F] - 32) / 1.8
    print('[DONE]')

    # Hb/Ht
    print('Hb/Ht ', end='')
    ii = ~pd.isna(df.loc[:, C_HB]) & pd.isna(df.loc[:, C_HT])
    if provenance:
        provenance.record("Calculate Ht from Hb", row=df.loc[ii].index, col=C_HT, reference_col=C_HB)
    df.loc[ii, C_HT] = (df.loc[ii, C_HB] * 2.862) + 1.216
    ii = ~pd.isna(df.loc[:, C_HT]) & pd.isna(df.loc[:, C_HB])
    if provenance:
        provenance.record("Calculate Hb from Ht", row=df.loc[ii].index, col=C_HB, reference_col=C_HT)
    df.loc[ii, C_HB] = (df.loc[ii, C_HT] - 1.216) / 2.862
    print('[DONE]')

    # BILI
    print('BILI ', end='')
    ii = ~pd.isna(df.loc[:, C_TOTAL_BILI]) & pd.isna(df.loc[:, C_DIRECT_BILI])
    if provenance:
        provenance.record("Calculate direct bili from total bili", row=df.loc[ii].index, col=C_DIRECT_BILI, reference_col=C_TOTAL_BILI)
    df.loc[ii, C_DIRECT_BILI] = (df.loc[ii, C_TOTAL_BILI] * 0.6934) - 0.1752
    ii = ~pd.isna(df.loc[:, C_DIRECT_BILI]) & pd.isna(df.loc[:, C_TOTAL_BILI])
    if provenance:
        provenance.record("Calculate total bili from direct bili", row=df.loc[ii].index, col=C_TOTAL_BILI, reference_col=C_DIRECT_BILI)
    df.loc[ii, C_TOTAL_BILI] = (df.loc[ii, C_DIRECT_BILI] + 0.1752) / 0.6934
    print('[DONE]')
    
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Generates a preliminary CSV '
        'containing patient state information at each timestep for which data '
        'is available (no binning). Also generates a dataframe of relevant '
        'timestamps for each patient (qstime).'))
    parser.add_argument('input', type=str,
                        help='Path to patient states CSV file')
    parser.add_argument('output', type=str,
                        help='Path at which to write output')
    parser.add_argument('--data', dest='data_dir', type=str, default='/data/',
                        help='Directory in which raw and preprocessed data is stored (default is ../data/ directory)')
    parser.add_argument('--no-outliers', dest='outliers', default=True, action='store_false',
                        help="Don't replace outliers with NaNs")
    parser.add_argument('--no-fio2', dest='fio2', default=True, action='store_false',
                        help="Don't estimate FiO2")
    parser.add_argument('--no-gcs', dest='gcs', default=True, action='store_false',
                        help="Don't estimate GCS from RASS")
    parser.add_argument('--no-vitals', dest='vitals', default=True, action='store_false',
                        help="Don't estimate vitals (e.g. temp, BP, Hb/Ht)")
    parser.add_argument('--no-sample-and-hold', dest='sample_and_hold', default=True, action='store_false',
                        help="Don't fill in missing values with sample-and-hold")
    parser.add_argument('--mask-file', dest='mask_file', default=None, type=str,
                        help="Path to write a mask file indicating where values were changed (+1 if a value was added or changed, or -1 if a value was removed)")
    parser.add_argument('--provenance-dir', dest='provenance_dir', default=None, type=str,
                        help="Path to directory in which to write provenance files (indicating sources and reasons for all changes)")
    parser.add_argument('--mask-out', dest='mask_out', default=None, type=str,
                        help="Path to write MedDreamer mask (1=observed,0=missing)")
    parser.add_argument('--delta-out', dest='delta_out', default=None, type=str,
                        help="Path to write MedDreamer delta (time since last observed per feature)")
    
    args = parser.parse_args()
    data_dir = args.data_dir or os.path.join(PARENT_DIR, 'data')

    df = load_csv(args.input)
    old_df = df.copy() if args.mask_file else None
    
    provenance = ProvenanceWriter(args.provenance_dir, verbose=True) if args.provenance_dir else None
    
    if args.outliers:
        print("Remove outliers")
        df = remove_outliers(df, provenance=provenance)

    if args.fio2:
        print("Estimate FiO2")
        df = estimate_fio2(df, provenance=provenance)

    if args.gcs:
        print("Estimate GCS from RASS")
        df[C_GCS] = df[C_GCS].astype("Float64")
        ii = pd.isna(df[C_GCS])
        if provenance:
            provenance.record("Estimate GCS from RASS", row=df.loc[ii].index, col=C_GCS, reference_col=C_RASS)
        df.loc[ii, C_GCS] = df.loc[ii, C_RASS].apply(estimate_gcs)

    if args.vitals:    
        print("Estimate vitals")
        df = estimate_vitals(df, provenance=provenance)
    
    if args.sample_and_hold:
        print("Sample and hold")
        sah_series = {
            C_BLOC: df[C_BLOC],
            C_ICUSTAYID: df[C_ICUSTAYID],
            C_TIMESTEP: df[C_TIMESTEP]
        }
        for col in SAH_FIELD_NAMES:
            print("SAH on " + col)
            sah_series[col] = sample_and_hold(df[C_ICUSTAYID],
                                              df[C_TIMESTEP],
                                              df[col],
                                              SAH_HOLD_DURATION[col],
                                              provenance=provenance,
                                              col_name=col)
            print("Eliminated {:.1f}% of NA values".format((1 - pd.isna(sah_series[col]).sum() / pd.isna(df[col]).sum()) * 100))

        df = pd.DataFrame(sah_series)
        del sah_series

    ID_COLS = [C_BLOC, C_ICUSTAYID, C_TIMESTEP]

    feature_cols = [c for c in (CHART_FIELD_NAMES + LAB_FIELD_NAMES + [C_MECHVENT, C_EXTUBATED]) if c in df.columns]

    df = df.sort_values([C_ICUSTAYID, C_TIMESTEP]).reset_index(drop=True)

    # Mask: 1 if value is present in the original dataframe, 0 if value is missing in the original dataframe
    mask_df = df[ID_COLS].copy()
    mask_df[feature_cols] = (~df[feature_cols].isna()).astype(np.float32)

    # Delta: time since last observation for each feature
    delta_df = df[ID_COLS].copy()
    delta_vals = np.zeros((len(df), len(feature_cols)), dtype=np.float32)

    for stay_id, g in df.groupby(C_ICUSTAYID, sort=False):
        idx = g.index.to_numpy()
        t = g[C_TIMESTEP].to_numpy().astype(np.float64)

        dt = np.zeros_like(t)
        dt[1:] = t[1:] - t[:-1]
        dt = np.clip(dt, a_min=0, a_max=None)

        m = (~g[feature_cols].isna()).to_numpy().astype(np.float32)

        acc = np.zeros((len(feature_cols),), dtype=np.float64)

        for k in range(len(idx)):
            if k == 0:
                acc[:] = 0.0
            else:
                acc += dt[k]
            acc[m[k] == 1.0] = 0.0
            delta_vals[idx[k], :] = acc.astype(np.float32)

    delta_df[feature_cols] = delta_vals
    
    print("Write")
    df.to_csv(args.output, index=False, float_format='%g')

    if args.mask_out:
        mask_df.to_csv(args.mask_out, index=False, float_format='%g')

    if args.delta_out:
        delta_df.to_csv(args.delta_out, index=False, float_format='%g')
    if provenance:
        provenance.close()
    
    if args.mask_file:
        print("Write mask file")
        
        # Compare the old dataframe to the new one and see where values have cropped up
        mask_series = {
            C_BLOC: df[C_BLOC],
            C_ICUSTAYID: df[C_ICUSTAYID],
            C_TIMESTEP: df[C_TIMESTEP]
        }
        
        for col in set(old_df.columns) & set(df.columns):
            if col in mask_series: continue
            old = old_df[col]
            new = df[col]
            mask_series[col] = np.where((pd.isna(old) & ~pd.isna(new)) | (~pd.isna(old) & (old != new)),
                                        1, np.where(~pd.isna(old) & pd.isna(new), -1, 0))
            
        pd.DataFrame(mask_series).to_csv(args.mask_file, index=False)
            
            