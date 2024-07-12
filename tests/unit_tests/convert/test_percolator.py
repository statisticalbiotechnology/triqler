from unittest.mock import patch

from triqler.convert.percolator import PercolatorPoutPsms, parsePsmsPout


# Mock parsers.getTsvReader
class MockTsvReader:
    def __init__(self, tsv_string):
        self.lines = tsv_string.strip().split('\n')
        self.data = [line.strip().split('\t') for line in self.lines]
        self.index = -1

    def __iter__(self):
        return self

    def __next__(self):
        self.index += 1
        if self.index < len(self.data):
            return self.data[self.index]
        else:
            raise StopIteration


def test_parsePsmsPout():
    tsv_data = """PSMId\tfilename\tscore\tq-value\tposterior_error_prob\tpeptide\tproteinIds
target_0_1681601_4_1\t./Quandenser_output/consensus_spectra/MaRaCluster.consensus.part1.ms2\t3.36331\t0.000339213\t6.11202e-11\tK.RGWGQLTSNLLLIGMEGNVTPAHYDEQQNFFAQIK.G\tHPRR4280170_poolB
target_0_1609101_3_1\t./Quandenser_output/consensus_spectra/MaRaCluster.consensus.part1.ms2\t3.17684\t0.000339213\t2.04056e-10\tK.KADCPIAWANLMLFDYKDQLK.T\tHPRR4040265_poolB
target_0_1764801_3_1\t./Quandenser_output/consensus_spectra/MaRaCluster.consensus.part1.ms2\t3.02753\t0.000339213\t5.35748e-10\tR.FPGNLLLNPFGISITSQSLNPGPFRTPK.A\tHPRR4030528_poolB
target_0_1726301_3_1\t./Quandenser_output/consensus_spectra/MaRaCluster.consensus.part1.ms2\t2.99247\t0.000339213\t6.72044e-10\tR.ATTNIIQPLLHAQWVLGDWSECSSTCGAGWQR.R\tHPRR2700128_poolA"""

    mock_reader = MockTsvReader(tsv_data)

    expected_output = [
        PercolatorPoutPsms(
            "target_0_1681601_4_1",
            "MaRaCluster.consensus.part1",
            1681601,
            4,
            3.36331,
            0.000339213,
            6.11202e-11,
            "K.RGWGQLTSNLLLIGMEGNVTPAHYDEQQNFFAQIK.G",
            ["HPRR4280170_poolB"],
        ),
        PercolatorPoutPsms(
            "target_0_1609101_3_1",
            "MaRaCluster.consensus.part1",
            1609101,
            3,
            3.17684,
            0.000339213,
            2.04056e-10,
            "K.KADCPIAWANLMLFDYKDQLK.T",
            ["HPRR4040265_poolB"],
        ),
        PercolatorPoutPsms(
            "target_0_1764801_3_1",
            "MaRaCluster.consensus.part1",
            1764801,
            3,
            3.02753,
            0.000339213,
            5.35748e-10,
            "R.FPGNLLLNPFGISITSQSLNPGPFRTPK.A",
            ["HPRR4030528_poolB"],
        ),
        PercolatorPoutPsms(
            "target_0_1726301_3_1",
            "MaRaCluster.consensus.part1",
            1726301,
            3,
            2.99247,
            0.000339213,
            6.72044e-10,
            "R.ATTNIIQPLLHAQWVLGDWSECSSTCGAGWQR.R",
            ["HPRR2700128_poolA"],
        ),
    ]

    with patch("triqler.parsers.getTsvReader", return_value=mock_reader):
        result = list(parsePsmsPout("mock_pout_file"))

        assert result == expected_output
