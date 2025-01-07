import sys

import pandas as pd

def read_triqler_result_file(filename):
    records = []
    first = True
    with open(filename, 'r') as file:
        for line in file:
            record = line[:-1].split('\t')
            if first:
                first = False
                headers = record
                num_fields = len(record)
            else:
                records.append(record[:num_fields-1] + [";".join(record[num_fields-1:])])
    df = pd.DataFrame(records, columns=headers)
    return df.apply(pd.to_numeric, errors='coerce')

df1 = read_triqler_result_file(sys.argv[1])
df2 = read_triqler_result_file(sys.argv[2])
return_code = pd.testing.assert_frame_equal(df1, df2, check_exact=False, rtol=1e-3)

sys.exit(return_code)