import random
import string


def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


def generate_csv_file(path, sequence_length=5000):
    '''
    Generate a random data set that fit the feature_config_test_e2e.yaml feature config format.
    This is to train dummy ranking model for validation purpose.

    @param sequence_length: number of sequence to generate
    '''
    with open(path, mode='w') as input:
        # feature1 is an int64, feature2 a float and feature3 a string
        # NOTE: discard feature3 from training from now, as it fails to be parsed
        input.write('"query_id","query_str","rank","feature1","feature2","feature3","label"\n')

        for query in range(sequence_length):
            rank = 0
            query_string = get_random_string(8)
            for i in range(random.randint(1, 6)):
                rank += 1
                input.write('"{}","{}",{},{},{},"{}",{}\n'.
                            format("q" + str(query).zfill(2), query_string, rank, random.randint(1,999),
                                   random.random(), get_random_string(random.randint(4,20)),
                                   1 if rank == 1 else 0))


if __name__ == "__main__":
    generate_csv_file("test.csv")