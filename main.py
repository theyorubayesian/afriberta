import os
import shutil

from absl import app
from absl import flags
from ner_scripts.train_ner import train

from src.trainer import TrainingManager
from src.utils import load_config

EXPERIMENT_PATH = "experiments"
EXPERIMENT_CONFIG_NAME = "config.yml"

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "experiment_name",
    "",
    "Experiment name: experiment outputs will be saved in a created experiment name directory",
)
flags.DEFINE_string("config_path", "config.yml", "Config file path")
flags.DEFINE_bool("do_eval_alone", False, "Evaluate a trained model only")
flags.DEFINE_string("language", "", "Language to train monolingual model on. If empty, multilingual model will be trained")
flags.DEFINE_string("tokenizer", "", "Tokenizer to use when training model")

def main(argv):
    config = load_config(FLAGS.config_path)
    if FLAGS.do_eval_alone:
        config["training"]["resume_training"] = True
    
    if FLAGS.language:
        config["data"]["train"] =  config["data"]["train"] + FLAGS.language
    
    if FLAGS.tokenizer:
        config["model"]["tokenizer_path"] = FLAGS.tokenizer

    experiment_path = os.path.join(EXPERIMENT_PATH, FLAGS.experiment_name)
    os.makedirs(experiment_path, exist_ok=True)

    experiment_config_path = os.path.join(experiment_path, EXPERIMENT_CONFIG_NAME)
    shutil.copy2(FLAGS.config_path, experiment_config_path)

    trainer = TrainingManager(config, experiment_path)
    
    if bool(FLAGS.do_eval_alone):
        trainer.evaluate()
    else:
        trainer.train()


if __name__ == "__main__":
    flags.mark_flag_as_required("experiment_name")
    app.run(main)
