import os

import dataset_generator
from ml4ir.applications.ranking.config.parse_args import RankingArgParser
from ml4ir.applications.ranking.pipeline import RankingPipeline
from ml4ir.base.model.relevance_model import RelevanceModel

TEST_DIRECTORY = "TEST_DIR"
TEST_FEATURE_CONFIG = "feature_config_test_e2e.yaml"


def generate_ranking_model(data_path, feature_config):
    rp = RankingPipeline(RankingArgParser().parse_args([
        "--data_dir", data_path,
        "--feature_config", feature_config,
        "--run_id", "e2e_run",
        "--data_format", "csv",
         "--execution_mode", "train_only",
        "--batch_size", "32",
    ]))
    rp.run()
    return rp.get_relevance_model(), rp.get_relevance_dataset().validation


def do_predict(relevance_model: RelevanceModel, dataset) -> object:

    os.makedirs("e2e_python_predict", exist_ok=True)
    relevance_model.predict(dataset, logs_dir= "e2e_python_predict")


if __name__ == "__main__":
    """
    Script to :
        - create a random dataset to train in a Ranking Pipeline.
        - execute the default Ranking Pipeline training on it
        - do and log prediction file on a validation set to be able to compare with Java execution of the model
    """
    if not os.path.exists(TEST_DIRECTORY):
        # TODO? : create 3 dataset {test, train, validation} to avoid execution errors (only need train)
        os.makedirs(TEST_DIRECTORY + "/test")
        os.makedirs(TEST_DIRECTORY + "/train")
        os.makedirs(TEST_DIRECTORY + "/validation")
        dataset_generator.generate_csv_file(TEST_DIRECTORY + "/test/test.csv")
        dataset_generator.generate_csv_file(TEST_DIRECTORY + "/train/train.csv")
        # TODO? : limit the validation data to 5ish. From my experience, if a minimal amount of data is not there
        #         the predictions_file is not generated (even when reducing the log_frequency)
        dataset_generator.generate_csv_file(TEST_DIRECTORY + "/validation/validation.csv")

    model, validation_dataset = generate_ranking_model(TEST_DIRECTORY, TEST_FEATURE_CONFIG)

    do_predict(model, validation_dataset)
