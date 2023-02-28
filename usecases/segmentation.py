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
data_names = ['WalkJogRun2_80_3800_6800.txt', 'PigInternalBleedingDatasetCVP_100_7501.txt', 'Cane_100_2345.txt', 'PulsusParadoxusSP02_30_10000.txt', 'InsectEPG1_50_3802.txt', 'RoboticDogActivityX_60_8699.txt', 'Fetal2013_70_6000_12000.txt', 'WalkJogRun1_80_3800_6800.txt', 'RoboticDogActivityY_60_10699.txt', 'PigInternalBleedingDatasetAirwayPressure_400_7501.txt', 'InsectEPG2_50_1800.txt', 'SuddenCardiacDeath2_25_3250.txt', 'GrandMalSeizures_10_8200.txt', 'NogunGun_150_3000.txt', 'UCR_semantic segmentation_2017.ppt', 'EEGRat_10_1000.txt', 'PulsusParadoxusECG1_30_10000.txt', 'temp.py', 'InsectEPG4_50_3160.txt', 'RoboticDogActivityY_64_4000.txt', 'Powerdemand_12_4500.txt', 'PulsusParadoxusECG2_30_10000.txt', 'PigInternalBleedingDatasetArtPressureFluidFilled_100_7501.txt', 'TiltABP_210_25000.txt', 'InsectEPG3_50_1710.txt', 'SuddenCardiacDeath3_25_3250.txt', 'GrandMalSeizures2_10_4550.txt', 'SuddenCardiacDeath1_25_6200_7600.txt', 'EEGRat2_10_1000.txt', 'GreatBarbet1_50_1900_3700.txt', 'SimpleSynthetic_125_3000_5000.txt', 'DutchFactory_24_2184.txt', 'GreatBarbet2_50_1900_3700.txt', 'TiltECG_200_25000.txt']

for data_name in data_names:
    T = testdata.load_from_txt("../data/" + use_case + "/" + data_name + ".txt")
    # TODO ground_truth = extract gropund truths

    # ### stumpy without dilation ###

    # SETUP for stumpy without dilation
    m = data_name.split('_')[1].split('_')[-1]
    d = 1
    L = m
    n_regimes = data_name.count('_')
    excl_factor = 1

    file_name = data_name + "_d" + str(d) + "_m" + str(m) + "_L" + str(L) + "_nregimes" + str(n_regimes) + "_exclfactor" + str(excl_factor)
    file_path = "../results/" + use_case + "/" + data_name + "/" + file_name


    # calculate:
    mp = stumpy.stump(T, m=m)
    cac, regime_locations = stumpy.fluss(mp[:, 1], L=L, n_regimes=n_regimes, excl_factor=excl_factor)
    results.save([T, m, d, L, n_regimes, excl_factor, mp, cac, regime_locations], file_path + ".npy")

    # visualize:
    T, m, d, L, n_regimes, excl_factor, mp, cac, regime_locations = results.load(file_path + ".npy")
    print(regime_locations)
    plot = visualize.segmentation_regimecac(T, m, d, L, n_regimes, excl_factor, mp, cac, regime_locations)
    plot.savefig(file_path + "_regimecac")
    # TODO Visualisierung an Original Paper orientieren: Arc Curve, Time Series, Matrix Profile


    # ### stumpy with dilation ###

    # SETUP for stumpy with dilation
    target_w = m
    ds = [2,3,4,5,6,7,8]

    # calculate:
    for d in ds:
        m = round((target_w-1)/d) + 1
        file_name = data_name + "_d" + str(d) + "_m" + str(m) + "_L" + str(L) + "_nregimes" + str(n_regimes) + "_exclfactor" + str(excl_factor)
        file_path = "../results/" + use_case + "/" + data_name + "/" + file_name

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
        plot = visualize.segmentation_regimecac(T, m, d, L, n_regimes, excl_factor, mp, cac, regime_locations)
        plot.savefig(file_path + "_regimecac")
        # TODO Visualisierung an Original Paper orientieren: Arc Curve, Time Series, Matrix Profile


