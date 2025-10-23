"""Training and evaluation"""

import run_lib
from absl import app, flags
from ml_collections.config_flags import config_flags
import logging
import os
from rdkit import RDLogger


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config', None, 'Training configuration.', lock_config=True
)
flags.DEFINE_string('workdir', None, 'Work directory.')
flags.DEFINE_enum('mode', None, ['train', 'eval'],
                  'Running mode: train or eval')
flags.DEFINE_string('eval_folder', 'eval', 'The folder name for storing evaluation results')
flags.DEFINE_boolean('deterministic', True, 'Set random seed for reproducibility.')
# define flag for lr
flags.DEFINE_float('lr', 2e-4, 'Learning rate for the optimizer.')
flags.DEFINE_enum('coeff_mode', 'exp', ['constant', 'inv_dist', 'inv_dist2', 'exp'], 
                  'Mode for computing energy coefficients.')
flags.mark_flags_as_required(['workdir', 'config', 'mode'])


def main(argv):
    # Set random seed
    if FLAGS.deterministic:
        run_lib.set_random_seed(FLAGS.config)

    # Ignore info output by RDKit
    RDLogger.DisableLog('rdApp.error')
    RDLogger.DisableLog('rdApp.warning')

    if FLAGS.mode == 'train':
        # Create the working directory
        if not os.path.exists(FLAGS.workdir):
            os.makedirs(FLAGS.workdir)
        # Set logger so that it outputs to both console and file
        gfile_stream = open(os.path.join(FLAGS.workdir, 'stdout.txt'), 'a')
        handler = logging.StreamHandler(gfile_stream)
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel('INFO')
        # Run the training pipeline
        if FLAGS.lr != 2e-4:
            FLAGS.config.optim.lr = FLAGS.lr
            logger.info(f"Overriding learning rate to {FLAGS.lr}")
        else:
            logger.info("Using default learning rate of 2e-4")

        if FLAGS.coeff_mode != 'exp':
            FLAGS.config.training.coeff_mode = FLAGS.coeff_mode
            logger.info(f"Overriding coeff mode to {FLAGS.coeff_mode}")
        else:
            logger.info("Using default coeff mode of 'exp'")
            
        run_lib.train(FLAGS.config, FLAGS.workdir)
    elif FLAGS.mode == 'eval':
        # Run the evaluation pipeline
        gfile_stream = open(os.path.join(FLAGS.workdir, 'eval_stdout.txt'), 'a')
        handler = logging.StreamHandler(gfile_stream)
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel('INFO')
        run_lib.evaluate(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
    else:
        raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == '__main__':
    app.run(main)
