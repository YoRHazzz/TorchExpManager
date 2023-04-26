import os.path
import shutil
import sys
import time
import timeit
from typing import Dict, Set

import pandas as pd
import torch
from tabulate import tabulate
from torch.optim import Optimizer
from tqdm import tqdm

from .DataLoaderWrapper import BaseDataLoaderWrapper
from .ModelWrapper import BaseModelWrapper
from .TorchExpTimeEstimator import ExpTimeEstimator
from .utils import Config, save_model, load_model


class TorchExpManager:
    def __init__(self, model_wrapper: BaseModelWrapper, train_dataloader: BaseDataLoaderWrapper,
                 valid_dataloader: BaseDataLoaderWrapper, test_dataloader: BaseDataLoaderWrapper, config: Config,
                 optimizer: Optimizer, train_metrics: Set[str] = None, valid_metrics: Set[str] = None,
                 test_metrics: Set[str] = None, save_metric: str = None, save_check_op: str = None,
                 verbose: bool = True,
                 iter_verbose: bool = False, log_dir: str = None,
                 exp_name: str = None, only_test: bool = False, early_stop_threshold: int = 5):
        self.model_wrapper = model_wrapper
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.config = config
        self.optimizer = optimizer
        self.train_metrics = set(train_metrics or {}).union({'loss', 'num_samples'})
        self.valid_metrics = set(valid_metrics or {}).union({'num_samples'})
        self.test_metrics = set(test_metrics or {}).union({'num_samples'})
        self.save_metric = save_metric or 'epoch'
        self.p_bar_postfix = {'iter', 'loss', 'loss_mean', 'accuracy'}
        if self.save_metric != 'epoch' and self.save_metric not in self.valid_metrics:
            raise ValueError(
                f"The save metric '{self.save_metric}' is not in the set of valid metrics '{self.valid_metrics}'.")
        self.save_check_op = save_check_op or '>'
        self.ops = {'<': lambda x, y: x < y, '>': lambda x, y: x > y}

        self.implemented_metric_set = set(self.model_wrapper.metric2func.keys())
        self._check_metric(self.train_metrics)
        self._check_metric(self.valid_metrics)
        self._check_metric(self.test_metrics)

        # Set instance variables related to logging and output
        self.verbose = verbose
        self.iter_verbose = iter_verbose
        self.default_columns_order = ['epoch', 'iter', 'stage', 'loss', 'loss_mean', 'loss_std',
                                      'total_samples', 'total_correct',
                                      'num_samples', 'num_correct', 'accuracy']
        self.order_cache = {}
        self.log_dir = log_dir or "logs"
        self.tmp_dir = os.path.join(self.log_dir, "tmp")
        if os.path.isdir(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
        os.makedirs(os.path.join(self.tmp_dir, "epoch_details"), exist_ok=True)

        self.exp_name = exp_name or self.get_exp_name()
        self.exp_dir = os.path.join(self.log_dir, self.exp_name)
        self.early_stop_threshold = early_stop_threshold

        # Set instance variables related to testing
        self.only_test = self.config['test'] if 'test' in self.config else only_test
        self.saved_model_dir = os.path.join(self.tmp_dir, 'saved_model')
        os.makedirs(self.saved_model_dir, exist_ok=True)

        self.stage2Categorical: Dict[str, pd.Categorical] = {
            'train': pd.Categorical(['train'], categories=['train', 'valid', 'test', '']),
            'valid': pd.Categorical(['valid'], categories=['train', 'valid', 'test', '']),
            'test': pd.Categorical(['test'], categories=['train', 'valid', 'test', ''])
        }

        # Set instance variables related to training
        self.epoch = self.best_epoch = 0

        # Sanity Check
        if self.verbose:
            self.verbose_sanity_check()

    def verbose_sanity_check(self):
        print(f"Sanity Check: config")
        self.config.show_status()
        print(f"Sanity Check: train metrics = {self.train_metrics}")
        print(f"Sanity Check: valid metrics = {self.valid_metrics}")
        print(f"Sanity Check: test metrics = {self.test_metrics}")
        print(f"Sanity Check: save metric = '{self.save_metric}'")
        print(f"Sanity Check: save check op = '{self.save_check_op}'"
              f" [current {self.save_metric} {self.save_check_op} best {self.save_metric}]")
        print(f"Sanity Check: early stop threshold = {self.early_stop_threshold}")
        print(f"Sanity Check: log dir = {self.log_dir}")
        print(f"Sanity Check: only test = {self.only_test}")
        print(f"Sanity Check: experiment name = {self.exp_name}")

    def _check_metric(self, metric_set):
        if not metric_set.issubset(self.implemented_metric_set):
            raise ValueError(f"Metrics {metric_set - self.implemented_metric_set} don't exist!")

    def get_exp_name(self):
        used_nums = {int(name[3:]) for name in os.listdir(self.log_dir) if name.startswith('exp')}
        new_num = max(used_nums, default=0) + 1
        return f'exp{new_num}'

    @staticmethod
    def build_pbar_kwargs(epoch, num_epochs, stage, total: int):
        return dict(
            total=total,
            colour='blue',
            unit='samples',
            desc=f'Epoch {epoch:>0{len(str(num_epochs))}d}/{num_epochs} {stage}',
            file=sys.stdout,
            dynamic_ncols=True,
            ascii=True
        )

    def run_one_epoch(self, epoch: int, num_epoch: int, stage: str, data_loader: BaseDataLoaderWrapper, metrics: Set,
                      backward_propagate: bool,
                      timeit: bool = True) \
            -> (pd.DataFrame, pd.DataFrame):
        iter_details = []
        metric_result = {}
        start_time = time.time()
        p_bar = None
        if self.verbose:
            p_bar = tqdm(**self.build_pbar_kwargs(epoch, num_epoch, stage, data_loader.num_samples))
        for idx, data in enumerate(data_loader):
            metric_result = {'iter': idx}
            with torch.set_grad_enabled(backward_propagate):
                out = self.model_wrapper(data['x'])
            for metric_name in metrics:
                if metric_name not in metric_result:
                    self.model_wrapper.metric2func[metric_name](out, data['y'], metric_result)
            if backward_propagate:
                self.optimizer.zero_grad()
                metric_result['loss'].backward()
                self.optimizer.step()
            for key, value in metric_result.items():
                if isinstance(value, torch.Tensor):
                    metric_result[key] = value.item()
            if timeit:
                end_time = time.time()
                metric_result['iter_time'] = end_time - start_time
                start_time = end_time
            if self.verbose:
                p_bar.set_postfix({key: metric_result[key] for key in self.p_bar_postfix if key in metric_result})
                p_bar.update(metric_result['num_samples'])
            iter_details.append(metric_result.copy())
        iter_details = pd.DataFrame.from_records(iter_details)

        iter_summary = self.model_wrapper.base_collect(iter_details)
        if self.verbose:
            p_bar.set_postfix({key: iter_summary[key] for key in self.p_bar_postfix if key in iter_summary})
            p_bar.close()
        iter_summary = pd.DataFrame.from_records([iter_summary])
        iter_summary['epoch'] = iter_details['epoch'] = epoch
        iter_summary['stage'] = self.stage2Categorical[stage]
        iter_details['stage'] = self.stage2Categorical[stage].repeat(len(iter_details))
        iter_details = self.reindex(stage + 'iter_details', iter_details)
        iter_summary = self.reindex(stage + 'iter_summary', iter_summary)

        return iter_summary, iter_details

    def reindex(self, name: str, df: pd.DataFrame):
        need_cache = (name not in self.order_cache.keys())
        if need_cache:
            self.order_cache[name] = self.get_columns_order(df)
        col_order = self.order_cache[name]
        return df.reindex(columns=col_order)

    def build_run_kwargs(self, epoch: int, num_epoch, stage: str):
        return dict(
            epoch=epoch,
            num_epoch=num_epoch,
            stage=stage,
            data_loader=getattr(self, f"{stage}_dataloader"),
            metrics=getattr(self, f"{stage}_metrics"),
            backward_propagate=True if stage == 'train' else False,
            timeit=True,
        )

    def get_columns_order(self, df: pd.DataFrame):
        return [*[col for col in self.default_columns_order if col in df.columns],
                *[col for col in df.columns if col not in self.default_columns_order]]

    def update_summary_table(self, summary_table: pd.DataFrame, iter_summary: pd.DataFrame):
        if summary_table.empty:
            col_order = self.get_columns_order(iter_summary)
            iter_summary = iter_summary.reindex(columns=col_order)
        return pd.concat([summary_table, iter_summary])

    def get_summary_from_one_epoch(self, epoch, num_epochs, stage, summary_table):
        iter_summary, iter_details = self.run_one_epoch(
            **self.build_run_kwargs(epoch, num_epochs, stage))
        summary_table = self.update_summary_table(summary_table, iter_summary)
        return summary_table, iter_summary, iter_details

    def train(self, epoch, num_epochs, train_summary_table):
        self.model_wrapper.train()
        return self.get_summary_from_one_epoch(epoch, num_epochs, 'train', train_summary_table)

    def eval(self, epoch, num_epochs, eval_summary_table, stage):
        self.model_wrapper.eval()
        # torch.cuda.empty_cache()
        return self.get_summary_from_one_epoch(epoch, num_epochs, stage, eval_summary_table)

    def run(self):
        if 'num_epochs' not in self.config:
            self.config['num_epochs'] = 100
        if 'eval_interval' not in self.config:
            self.config['eval_interval'] = 2
        num_epochs = self.config['num_epochs']
        time_estimator = ExpTimeEstimator(num_epochs, num_epochs // self.config['eval_interval'])
        train_summary_table = valid_summary_table = test_summary_table = pd.DataFrame()
        if not self.only_test:
            best_metric, early_stop_count = None, 0
            for self.epoch in range(1, num_epochs + 1):
                # train and save to csv
                train_summary_table, train_iter_summary, train_iter_details = \
                    self.train(self.epoch, num_epochs, train_summary_table)
                time_estimator.update_train(train_iter_summary.iloc[0]['epoch_time'])
                train_iter_details.to_csv(
                    os.path.join(self.tmp_dir, 'epoch_details',
                                 f'epoch{self.epoch:>0{len(str(num_epochs))}d}_train.csv'),
                    index=False)

                # valid and save to csv
                valid_iter_summary = None
                need_eval = ('eval_interval' in self.config and self.epoch % self.config['eval_interval'] == 0)
                if need_eval:
                    valid_summary_table, valid_iter_summary, valid_iter_details = \
                        self.eval(self.epoch, num_epochs, valid_summary_table, 'valid')
                    time_estimator.update_valid(valid_iter_summary.iloc[0]['epoch_time'])
                    valid_iter_details.to_csv(
                        os.path.join(self.tmp_dir, 'epoch_details',
                                     f'epoch{self.epoch:>0{len(str(num_epochs))}d}_valid.csv'), index=False)

                    # check is best
                    current_save_metric = valid_iter_summary.iloc[0][self.save_metric]
                    is_best = (best_metric is None) or self.ops[self.save_check_op](current_save_metric, best_metric)
                    if is_best:
                        best_metric, self.best_epoch = current_save_metric, self.epoch
                        save_model(self.model_wrapper.model, self.saved_model_dir, "model_best.pt",
                                   epoch=self.best_epoch, metric_name=self.save_metric, metric=best_metric)
                        early_stop_count = 0
                    else:
                        early_stop_count += 1
                table = pd.concat([train_iter_summary, valid_iter_summary]).fillna("")
                print(tabulate(table, headers='keys', tablefmt='psql', numalign='center', stralign='center',
                               showindex=False))

                time_estimator.print_status(
                    remaining_train=(self.early_stop_threshold - early_stop_count) * self.config['eval_interval']
                    if early_stop_count else None,
                    remaining_valid=(self.early_stop_threshold - early_stop_count) if early_stop_count else None,
                )
                print(f"Best {self.save_metric}: {best_metric} (epoch {self.best_epoch}) | "
                      f"Early stop count: {early_stop_count}/{self.early_stop_threshold}")
                need_early_stop = self.early_stop_threshold and early_stop_count >= self.early_stop_threshold
                if need_early_stop:
                    print("Early Stop!")
                    break

        ckpt = load_model(self.model_wrapper.model, self.saved_model_dir, "model_best.pt",
                          self.model_wrapper.device)
        if 'epoch' in ckpt:
            self.best_epoch = ckpt['epoch']
        test_summary_table, test_iter_summary, test_iter_details = \
            self.eval(self.best_epoch, num_epochs, test_summary_table, 'test')
        test_iter_details.to_csv(
            os.path.join(self.tmp_dir, 'epoch_details', f'epoch{num_epochs:>0{len(str(num_epochs))}d}_test.csv'),
            index=False)
        summary_table = pd.concat([train_summary_table, valid_summary_table, test_summary_table])
        summary_table = summary_table.sort_values(by=['epoch', 'stage'], ascending=True)

        train_summary_table.to_csv(os.path.join(self.tmp_dir, 'train_summary_table.csv'), index=False)
        valid_summary_table.to_csv(os.path.join(self.tmp_dir, 'valid_summary_table.csv'), index=False)
        test_summary_table.to_csv(os.path.join(self.tmp_dir, 'test_summary_table.csv'), index=False)
        summary_table.to_csv(os.path.join(self.tmp_dir, 'summary_table.csv'), index=False)

        print(f"move tmp_dir: {self.tmp_dir} -> {self.exp_dir}")
        os.rename(self.tmp_dir, self.exp_dir)
        pass

    @staticmethod
    def test_dataloader_time(dataloader):
        def iterate_dataloader():
            for _ in dataloader:
                pass

        total_time = timeit.timeit(iterate_dataloader, number=1)
        return total_time

    def dataloader_time_test(self):
        print(f"train loader: {self.test_dataloader_time(self.train_dataloader)}/epoch")
        print(f"valid loader: {self.test_dataloader_time(self.valid_dataloader)}/epoch")
        print(f"test loader: {self.test_dataloader_time(self.test_dataloader)}/epoch")
