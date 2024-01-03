import pytest
import numpy as np

from triqler import pgm
from triqler.hyperparameters import funcLogHypsec, funcGamma, funcHypsec
from triqler.parsers import PeptideQuantRow


def test_getPosteriors(quant_rows, params):
    (
        bayesQuantRow,
        muGroupDiffs,
        probsBelowFoldChange,
        posteriorDists,
    ) = pgm.getPosteriors(quant_rows, params)

    print(bayesQuantRow, muGroupDiffs, probsBelowFoldChange, posteriorDists)

    np.testing.assert_almost_equal(
        bayesQuantRow,
        [
            8.24490284e00,
            1.24918641e01,
            8.05665565e00,
            5.89933126e-03,
            7.10587370e-03,
            5.87790167e-03,
            1.83692123e01,
            1.83278372e01,
            1.45274058e01,
        ],
    )
    assert muGroupDiffs == {
        (0, 1): 10.489210997634393,
        (0, 2): -0.8449359955024008,
        (1, 2): -11.334146993137496,
    }
    assert probsBelowFoldChange == {
        (0, 1): 1.5005959053077542e-11,
        (0, 2): 0.9979268519372426,
        (1, 2): 1.783188069683151e-12,
    }
    assert posteriorDists is None


@pytest.fixture
def params():
    params = dict()
    params["muDetect"], params["sigmaDetect"] = 1.056334, 0.372395
    params["muXIC"], params["sigmaXIC"] = 3.276315, 0.953023
    params["muProtein"], params["sigmaProtein"] = 0.066437, 0.239524
    params["muFeatureDiff"], params["sigmaFeatureDiff"] = -0.013907, 0.149265
    params["shapeInGroupStdevs"], params["scaleInGroupStdevs"] = 1.027176, 0.089433

    params["proteinQuantCandidates"] = np.arange(-5.0, 5.0 + 1e-10, 0.01)
    params["proteinPrior"] = funcLogHypsec(
        params["proteinQuantCandidates"], params["muProtein"], params["sigmaProtein"]
    )

    params["groups"] = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    sigmaCandidates = np.arange(0.001, 3.0, 0.001)
    gammaCandidates = funcGamma(
        sigmaCandidates, params["shapeInGroupStdevs"], params["scaleInGroupStdevs"]
    )
    support = np.where(gammaCandidates > max(gammaCandidates) * 0.01)
    params["sigmaCandidates"] = np.linspace(
        sigmaCandidates[support[0][0]], sigmaCandidates[support[0][-1]], 20
    )

    qc = params["proteinQuantCandidates"]
    params["proteinDiffCandidates"] = np.linspace(
        2 * qc[0], 2 * qc[-1], len(qc) * 2 - 1
    )

    params["inGroupDiffPrior"] = funcHypsec(
        params["proteinDiffCandidates"], 0, params["sigmaCandidates"][:, np.newaxis]
    )

    params["foldChangeEval"] = 1.62

    params["returnPosteriors"] = False
    return params


@pytest.fixture
def quant_rows():
    return [
        PeptideQuantRow(
            combinedPEP=2.7104394704636187e-07,
            charge=3,
            featureGroup=6652,
            spectrum=4463600,
            linkPEP=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            quant=[
                1575.5194217806763,
                1762.3673201392012,
                1820.024880133668,
                0.0,
                118.88882650834157,
                0.0,
                3695.803309600916,
                3401.718214956748,
                3505.93517181495,
            ],
            identificationPEP=[
                4.6886914550903214e-07,
                2.8216462943930765e-07,
                2.845664817119342e-07,
                0.9996593387961632,
                0.9996593387961632,
                0.9996593387961632,
                2.714661175851063e-07,
                3.261983406721569e-07,
                2.8223038206487416e-07,
            ],
            peptide="R.SRVTDPVGDIVSFMHSFEEK.Y",
            protein=["HPRR1950515_poolA"],
        ),
        PeptideQuantRow(
            combinedPEP=6.482911522038437e-07,
            charge=4,
            featureGroup=11166,
            spectrum=7475400,
            linkPEP=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            quant=[
                1704.0042305855984,
                2204.037667047334,
                1829.8065815771586,
                0.0,
                0.0,
                28.57000951657358,
                3119.0550454316717,
                2847.289188099933,
                2757.6343237502724,
            ],
            identificationPEP=[
                6.500063938208456e-07,
                6.526943225892978e-07,
                6.496955121670922e-07,
                0.9996593389246767,
                0.9996593389246767,
                0.9996593389246767,
                6.666674944311168e-07,
                7.859887142247501e-07,
                6.763732579573301e-07,
            ],
            peptide="R.SRVTDPVGDIVSFMHSFEEKYGR.A",
            protein=["HPRR1950515_poolA"],
        ),
        PeptideQuantRow(
            combinedPEP=2.7804280663893726e-06,
            charge=4,
            featureGroup=12112,
            spectrum=8109900,
            linkPEP=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            quant=[
                689.6275053023554,
                3230.0901501086432,
                567.4516681905867,
                0.0,
                0.0,
                0.0,
                2282.9223205622375,
                2190.696796520716,
                1543.0469410104322,
            ],
            identificationPEP=[
                4.724230953989661e-06,
                5.3609765079509764e-05,
                4.055672489178264e-06,
                5.3609765079509764e-05,
                5.3609765079509764e-05,
                5.3609765079509764e-05,
                2.8158679113987617e-06,
                2.790875551328398e-06,
                1.3412374050281173e-05,
            ],
            peptide="R.FLLVYLHGDDHQDSDEFCRNTLCAPEVISLINTR.M",
            protein=["HPRR1950515_poolA"],
        ),
        PeptideQuantRow(
            combinedPEP=3.117471149685594e-06,
            charge=3,
            featureGroup=8940,
            spectrum=5994900,
            linkPEP=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            quant=[
                3480.080511792603,
                3191.43767663111,
                2820.953600530974,
                0.0,
                0.0,
                0.0,
                3986.639126880911,
                4902.698544709854,
                3632.927656538313,
            ],
            identificationPEP=[
                0.00171353800246965,
                0.0005945171245153036,
                7.726352475811993e-06,
                0.00171353800246965,
                0.00171353800246965,
                0.00171353800246965,
                3.1390749680726415e-06,
                3.1952417308689363e-06,
                3.1297031838350975e-06,
            ],
            peptide="R.FLLVYLHGDDHQDSDEFCR.N",
            protein=["HPRR1950515_poolA"],
        ),
        PeptideQuantRow(
            combinedPEP=5.288865901400665e-06,
            charge=4,
            featureGroup=1196,
            spectrum=803500,
            linkPEP=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            quant=[
                592.1957757348129,
                931.7087222720427,
                406.3962437825501,
                0.0,
                211.83074580519755,
                0.0,
                871.5945498038101,
                864.7196852894161,
                816.9075614071153,
            ],
            identificationPEP=[
                5.316393601018099e-06,
                5.314210915097206e-06,
                5.391593191683697e-06,
                0.9964918237502542,
                0.9964918237502542,
                0.9964918237502542,
                6.570453672138932e-06,
                5.4377498021063175e-06,
                6.341512817753525e-06,
            ],
            peptide="R.VTDPVGDIVSFMHSFEEKYGR.A",
            protein=["HPRR1950515_poolA"],
        ),
        PeptideQuantRow(
            combinedPEP=7.960388499455959e-06,
            charge=3,
            featureGroup=1668,
            spectrum=1113700,
            linkPEP=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            quant=[
                47103.46186372248,
                45099.422895148695,
                43573.702862780054,
                0.0,
                0.0,
                0.0,
                80017.64057315495,
                75609.92233413903,
                75695.56942717802,
            ],
            identificationPEP=[
                3.628617069073581e-05,
                5.249632036397056e-05,
                4.9480441238092254e-05,
                9.176174859037989e-05,
                9.176174859037989e-05,
                9.176174859037989e-05,
                8.004916422499342e-06,
                9.176174859037989e-05,
                8.039423649353061e-06,
            ],
            peptide="R.AHPVFYQGTYSQALNDAKR.E",
            protein=["HPRR1950515_poolA"],
        ),
        PeptideQuantRow(
            combinedPEP=4.518750755187076e-05,
            charge=3,
            featureGroup=4858,
            spectrum=3250000,
            linkPEP=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            quant=[
                1615.1723519661937,
                1658.8081355948389,
                1408.1748284893545,
                0.0,
                1005.5097145557918,
                0.0,
                2459.460480234012,
                2139.3943171175247,
                2067.9884396478838,
            ],
            identificationPEP=[
                4.567497861640568e-05,
                5.854121129411638e-05,
                4.9428138152229906e-05,
                0.9904595976684796,
                0.9904595976684796,
                0.9904595976684796,
                4.7788451285279976e-05,
                5.197240955467919e-05,
                4.9008010940321256e-05,
            ],
            peptide="R.VTDPVGDIVSFMHSFEEK.Y",
            protein=["HPRR1950515_poolA"],
        ),
        PeptideQuantRow(
            combinedPEP=5.6142304936461626e-05,
            charge=4,
            featureGroup=11447,
            spectrum=7667600,
            linkPEP=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            quant=[
                0.0,
                1296.9810464401971,
                739.210812761407,
                0.0,
                0.0,
                0.0,
                1370.0982986484503,
                1088.5154402642777,
                405.97989191356316,
            ],
            identificationPEP=[
                0.9996593578293395,
                5.679977726802399e-05,
                5.7105367625820413e-05,
                0.9996593578293395,
                0.9996593578293395,
                0.9996593578293395,
                8.518983803718072e-05,
                0.00014064453043627356,
                0.9996593578293395,
            ],
            peptide="R.AHPVFYQGTYSQALNDAKRELR.F",
            protein=["HPRR1950515_poolA"],
        ),
        PeptideQuantRow(
            combinedPEP=0.00023380017175003343,
            charge=2,
            featureGroup=11283,
            spectrum=7554900,
            linkPEP=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            quant=[
                22329.91204404644,
                21928.6988208084,
                23456.758277049743,
                1113.5724856505278,
                1154.4542641029439,
                1217.199590451562,
                43616.165658578684,
                40958.59719565662,
                41280.42246329136,
            ],
            identificationPEP=[
                0.8527902481471714,
                0.00027693230846226324,
                0.00028446820215899393,
                0.8129022476634267,
                0.684950975223747,
                0.915874081463779,
                0.0002447306687325401,
                0.00024332989821584938,
                0.00023849639564776925,
            ],
            peptide="R.NTLCAPEVISLINTR.M",
            protein=["HPRR1950515_poolA"],
        ),
        PeptideQuantRow(
            combinedPEP=0.0012005487362288742,
            charge=3,
            featureGroup=11877,
            spectrum=7957000,
            linkPEP=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            quant=[
                2732.1087453314817,
                3428.5861012386163,
                2037.141690018857,
                0.0,
                0.0,
                0.0,
                6666.665635240884,
                6697.895467806992,
                7260.385673031901,
            ],
            identificationPEP=[
                0.041975882598652925,
                0.010927776221381413,
                0.9852413004750691,
                0.9852413004750691,
                0.9852413004750691,
                0.9852413004750691,
                0.0014019521279400315,
                0.0012453119133044588,
                0.0012693094012189432,
            ],
            peptide="R.MLFWACSTNKPEGYR.V",
            protein=["HPRR1950515_poolA"],
        ),
        PeptideQuantRow(
            combinedPEP=0.0023379880194026436,
            charge=2,
            featureGroup=686,
            spectrum=457700,
            linkPEP=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            quant=[
                0.0,
                2992.1243619413535,
                2036.2265796967074,
                0.0,
                0.0,
                0.0,
                7297.233048209398,
                7607.180520063822,
                6760.828481153428,
            ],
            identificationPEP=[
                0.0034750434231952676,
                0.0034750434231952676,
                0.0024569022364353543,
                0.0034750434231952676,
                0.0034750434231952676,
                0.0034750434231952676,
                0.00245013008758721,
                0.0024776333280206453,
                0.0024679557273312724,
            ],
            peptide="R.LEGLIQPDDLINQL.-",
            protein=["HPRR1950515_poolA"],
        ),
        PeptideQuantRow(
            combinedPEP=0.022755016993493644,
            charge=3,
            featureGroup=6770,
            spectrum=4542600,
            linkPEP=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            quant=[
                499.59655134262226,
                784.2210065544632,
                621.4144544776436,
                0.0,
                0.0,
                0.0,
                502.894423922525,
                516.1874085240363,
                327.186292247236,
            ],
            identificationPEP=[
                0.10364150662400651,
                0.02533056441530923,
                0.02848723545621379,
                0.1055670446292355,
                0.1055670446292355,
                0.1055670446292355,
                0.09857928551145678,
                0.1055670446292355,
                0.08228807061332011,
            ],
            peptide="R.RMTVVGRLEGLIQPDDLINQL.-",
            protein=["HPRR1950515_poolA"],
        ),
        PeptideQuantRow(
            combinedPEP=0.4316295723715946,
            charge=2,
            featureGroup=10223,
            spectrum=6845100,
            linkPEP=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            quant=[
                1387.3228350925351,
                1619.1363019285855,
                0.0,
                0.0,
                0.0,
                0.0,
                224.43794147476063,
                4593.984229459955,
                1677.2924661241034,
            ],
            identificationPEP=[
                0.5467287392713549,
                0.6505089356445992,
                0.9916015088418297,
                0.9916015088418297,
                0.9916015088418297,
                0.9916015088418297,
                0.9916015088418297,
                0.607344035772402,
                0.7429809696075873,
            ],
            peptide="R.FIRPDPR.S",
            protein=["HPRR1950515_poolA"],
        ),
    ]
