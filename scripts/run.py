import logging
import os
import sys
from typing import Tuple

sys.path.append('..')

import hydra
import numpy as np
import polars as pl
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from source.dataset import SequentialDataModule, load_data
from source.embedding_manager import EmbeddingManager
from source.logger import Task
from source.optimizer import ConstrainedNormAdam
from source.recommender import SASRecModel

from source.winter.evaluation import ColdStartEvaluationPipeline
from source.winter.recommender import ColdStartSequentialRecommender, SASRecModelWithTrainableDelta


def get_task_name(config: DictConfig) -> str:
    task_name = 'init({})'.format('text' if config.use_pretrained_item_embeddings else 'rand')

    if config.train_delta:
        task_name += f'-delta-{config.max_delta_norm}'

    return task_name


def get_task(config: DictConfig) -> Task:
    task = Task.init(
        project_name=config.project_name,
        task_name=get_task_name(config),
        reuse_last_task_id=False,
    )
    task.connect(OmegaConf.to_container(config))
    return task


def get_datamodule(config: DictConfig) -> SequentialDataModule:
    return SequentialDataModule(
        train_filepath=config.dataset.train_filepath,
        val_filepath=config.dataset.val_filepath,
        max_length=config.dataset.max_length,
    )


def get_model(config: DictConfig) -> SASRecModel:
    model_params = dict(
        num_items=config.model.num_items,
        embedding_dim=config.model.embedding_dim,
        num_blocks=config.model.num_blocks,
        num_heads=config.model.num_heads,
        intermediate_dim=config.model.embedding_dim,
        p=config.model.p,
        max_length=config.model.max_length,
    )

    if config.train_delta:
        return SASRecModelWithTrainableDelta(max_delta_norm=config.max_delta_norm, **model_params)
    else:
        return SASRecModel(**model_params)


def get_trainer(config: DictConfig) -> Trainer:
    early_stopping = EarlyStopping(
        monitor=config.early_stopping.monitor,
        patience=config.early_stopping.patience,
        mode=config.early_stopping.mode,
    )
    model_checkpoint = ModelCheckpoint(
        dirpath=os.path.join(config.checkpoint_dir, Task.current_task().id, 'recommender'),
        monitor=config.model_checkpoint.monitor,
        mode=config.model_checkpoint.mode,
    )
    trainer = Trainer(
        devices=config.trainer.devices,
        callbacks=[early_stopping, model_checkpoint],
        max_epochs=config.trainer.max_epochs,
        val_check_interval=config.trainer.val_check_interval,
    )
    return trainer


def get_item_embeddings(config: DictConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    warm_item_embeddings = np.load(config.dataset.item_embeddings.warm)
    cold_item_embeddings = np.load(config.dataset.item_embeddings.cold)

    embedding_manager = EmbeddingManager(
        config.model.embedding_dim,
        reduce=config.model.embedding_dim != warm_item_embeddings.shape[1],
        normalize=True,
    )
    logging.info(f'EmbeddingManager: reduce = {embedding_manager.reduce}.')

    warm_item_embeddings = embedding_manager.fit_transform(warm_item_embeddings)
    cold_item_embeddings = embedding_manager.transform(cold_item_embeddings)

    embedding_manager.save(
        os.path.join(config.checkpoint_dir, Task.current_task().id, 'embedding_manager.pkl')
    )

    return torch.tensor(warm_item_embeddings).float(), torch.tensor(cold_item_embeddings).float()


def add_cold_item_embeddings(model: SASRecModel, cold_item_embeddings: torch.Tensor) -> None:
    item_embeddings = model.item_embedding.weight[: model.num_items + 1]

    if isinstance(model, SASRecModelWithTrainableDelta):
        delta_embeddings = model.delta_embedding.weight[: model.num_items + 1]
        model.set_pretrained_item_embeddings(
            item_embeddings=torch.vstack(
                [item_embeddings, cold_item_embeddings.to(item_embeddings.device)]
            ),
            delta_embeddings=torch.vstack(
                [
                    delta_embeddings,
                    torch.zeros_like(cold_item_embeddings).to(delta_embeddings.device),
                ]
            ),
            add_padding_embedding=False,
            freeze=True,
        )
    else:
        model.set_pretrained_item_embeddings(
            item_embeddings=torch.vstack(
                [item_embeddings, cold_item_embeddings.to(item_embeddings.device)]
            ),
            add_padding_embedding=False,
            freeze=True,
        )


def report_results_to_clearm(results: pl.DataFrame) -> None:
    task = Task.current_task()
    logger = task.get_logger()

    results = results.to_pandas().round(4)

    task.register_artifact(name='cold-start-evaluation', artifact=results)
    logger.report_table(
        title='cold-start-evaluation',
        series='',
        iteration=0,
        table_plot=results,
    )

    results = results.set_index(['recommend-cold-items', 'filter-cold-items'])

    for (recommend_cold_items, filter_cold_items), row in results.iterrows():
        for metric_name, metric_value in row.items():
            logger.report_single_value(
                '/'.join(
                    [
                        f'recommend-cold-items={recommend_cold_items}',
                        f'filter-cold-items={filter_cold_items}',
                        metric_name,
                    ]
                ),
                metric_value,
            )


@hydra.main(config_path='../configs', config_name='main', version_base=None)
def main(config: DictConfig) -> None:
    task = get_task(config)

    seed_everything(config.seed)

    if not os.path.exists(os.path.join(config.checkpoint_dir, task.id)):
        os.makedirs(os.path.join(config.checkpoint_dir, task.id))

    datamodule = get_datamodule(config)
    model = get_model(config)

    # Set pre-trained item embeddings if necessary
    if config.use_pretrained_item_embeddings:
        warm_item_embeddings, cold_item_embeddings = get_item_embeddings(config)
        model.set_pretrained_item_embeddings(
            warm_item_embeddings.clone(),
            add_padding_embedding=True,
            freeze=False,
        )
        logging.info('Set pre-trained item embeddings.')

    recommender = ColdStartSequentialRecommender(
        model,
        learning_rate=config.recommender.learning_rate,
        remove_seen=config.recommender.remove_seen,
        metrics=config.recommender.metrics,
        topk=config.recommender.topk,
    )

    # Change optimizer if necessary
    if isinstance(model, SASRecModelWithTrainableDelta):
        recommender.configure_optimizers = lambda: ConstrainedNormAdam(
            model.parameters(),
            constrained_params=model.delta_embedding.parameters(),
            max_norm=config.max_delta_norm,
            lr=config.recommender.learning_rate,
        )
        logging.info('Switched to ConstrainedNormAdam.')

    # Train recommender
    trainer = get_trainer(config)
    trainer.fit(recommender, datamodule=datamodule)

    # Load checkpoint
    recommender = ColdStartSequentialRecommender.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
        model=model,
        remove_seen=config.recommender.remove_seen,
        metrics=config.recommender.metrics,
        topk=config.recommender.topk,
    )
    logging.info(f'Loaded checkpoint from {trainer.checkpoint_callback.best_model_path}.')

    # Set cold item embeddings
    if config.use_pretrained_item_embeddings:
        add_cold_item_embeddings(recommender.model, cold_item_embeddings)

    # Run evaluation
    test_interactions = load_data(config.dataset.test_filepath)
    ground_truth = load_data(config.dataset.gt_filepath)

    results = ColdStartEvaluationPipeline(
        recommender,
        trainer,
        test_interactions,
        ground_truth,
    ).run()
    report_results_to_clearm(results)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
