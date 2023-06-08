# # Semantic Segmentation (regimes) with arc curves # 
# “arc curve” annotates the raw time series with information about the likelihood of a regime change.
# 
# https://stumpy.readthedocs.io/en/latest/Tutorial_Semantic_Segmentation.html
# 
# https://sites.google.com/site/onlinesemanticsegmentation/


# SETUP
import stumpy
import helper.testdata as testdata
import helper.visualize as visualize
import helper.results as results

# SETUP
use_case = "segmentation"
data_names = ['WalkJogRun2_80_3800_6800', 'PigInternalBleedingDatasetCVP_100_7501', 'Cane_100_2345', 'PulsusParadoxusSP02_30_10000', 'InsectEPG1_50_3802', 'RoboticDogActivityX_60_8699', 'Fetal2013_70_6000_12000', 'WalkJogRun1_80_3800_6800', 'RoboticDogActivityY_60_10699', 'PigInternalBleedingDatasetAirwayPressure_400_7501', 'InsectEPG2_50_1800', 'SuddenCardiacDeath2_25_3250', 'GrandMalSeizures_10_8200', 'NogunGun_150_3000', 'EEGRat_10_1000', 'PulsusParadoxusECG1_30_10000', 'InsectEPG4_50_3160', 'RoboticDogActivityY_64_4000', 'Powerdemand_12_4500', 'PulsusParadoxusECG2_30_10000', 'PigInternalBleedingDatasetArtPressureFluidFilled_100_7501', 'TiltABP_210_25000', 'InsectEPG3_50_1710', 'SuddenCardiacDeath3_25_3250', 'GrandMalSeizures2_10_4550', 'SuddenCardiacDeath1_25_6200_7600', 'EEGRat2_10_1000', 'GreatBarbet1_50_1900_3700', 'SimpleSynthetic_125_3000_5000', 'DutchFactory_24_2184', 'GreatBarbet2_50_1900_3700', 'TiltECG_200_25000']
# <nemonic_name> _<recommended subsequence length>_<location of 1st change point>_...<location of ith change point>.txt

for data_name in data_names:
    print(data_name, ":")
    T = testdata.load_from_txt("../data/" + use_case + "/" + data_name + ".txt")
    # TODO ground_truth = extract gropund truths
    print("Ground Truth: ", ground_truth)

    # SETUP Hyperparams
    target_w = int(data_name.split('_')[1].split('_')[-1])
    ds = [1,2,3,4,5,6,7,8]
    L = m
    n_regimes = data_name.count('_')
    excl_factor = 1

    # calculate:
    for d in ds:
        m = round((target_w-1)/d) + 1
        file_name = data_name + "_d" + str(d) + "_m" + str(m) + "_L" + str(L) + "_nregimes" + str(n_regimes) + "_exclfactor" + str(excl_factor)
        file_path = "../results/" + use_case + "/" + data_name + "/" + file_name

        if d == 1:
            mp = stumpy.stump(T, m=m)
        else:
            mp = stumpy.stump_dil(T, m=m, d=d)
        cac, regime_locations = stumpy.fluss(mp[:, 1], L=L, n_regimes=n_regimes, excl_factor=excl_factor)

        # TODO Abweichung vom ground truth berechnen und in csv speichern

        results.save([T, m, d, L, n_regimes, excl_factor, mp, cac, regime_locations], file_path + ".npy")

    # visualize:
    for d in ds:
        m = round((target_w-1)/d) + 1
        file_name = data_name + "_d" + str(d) + "_m" + str(m) + "_L" + str(L) + "_nregimes" + str(n_regimes) + "_exclfactor" + str(excl_factor)
        file_path = "../results/" + use_case + "/" + data_name + "/" + file_name

        T, m, d, L, n_regimes, excl_factor, mp, cac, regime_locations = results.load(file_path + ".npy")

        print(regime_locations)
        plot = visualize._segmentation_regimecac(T, m, d, L, n_regimes, excl_factor, mp, cac, regime_locations)
        plot.savefig(file_path + "_regimecac")
        # TODO Visualisierung an Original Paper orientieren: Arc Curve, Time Series, Matrix Profile


