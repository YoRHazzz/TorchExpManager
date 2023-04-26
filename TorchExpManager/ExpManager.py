import datetime
import os.path
import sys
import time
from typing import Any, Dict, Set
import shutil

import pandas as pd
import torch
from torch.optim import Optimizer
from tqdm import tqdm
from tabulate import tabulate

from .DataLoaderWrapper import BaseDataLoaderWrapper
from .ModelWrapper import BaseModelWrapper
from .utils import detach_if_requires_grad, EMATimeEstimator, format_time, current_time_str, Config


class TorchExpManager:
    def __init__(self, model_wrapper: BaseModelWrapper, train_dataloader: BaseDataLoaderWrapper,
                 valid_dataloader: BaseDataLoaderWrapper, test_dataloader: BaseDataLoaderWrapper, config: Config,
                 optimizer: Optimizer, train_metrics: Set[str] = None, valid_metrics: Set[str] = None,
                 test_metrics: Set[str] = None, save_metric: str = None, save_check_op: str = None,
                 verbose: bool = True,
                 iter_verbose: bool = False, log_dir: str = None,
                 exp_name: str = None, only_test: bool = False, early_stop: bool = False):
        self.model_wrapper = model_wrapper
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.config = config
        self.optimizer = optimizer
        self.train_metrics = set(train_metrics or {})
        self.train_metrics.update({'loss', 'num_samples'})
        self.valid_metrics = set(valid_metrics or {})
        self.valid_metrics.update({'num_samples'})
        self.test_metrics = set(test_metrics or {})
        self.test_metrics.update({'num_samples'})
        self.save_metric = save_metric or 'epoch'
        self.save_check_op = save_check_op or '>'
        self.ops = {'<': lambda x, y: x < y, '>': lambda x, y: x > y}

        self._check_metric(self.train_metrics)
        self._check_metric(self.valid_metrics)
        self._check_metric(self.test_metrics)
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

        self.only_test = self.config['test'] if 'test' in self.config else only_test
        self.saved_model_dir = os.path.join(self.tmp_dir, 'saved_model')
        os.makedirs(self.saved_model_dir, exist_ok=True)

        self.stage2Categorical: Dict[str, pd.Categorical] = {
            'train': pd.Categorical(['train'], categories=['train', 'valid', 'test', '']),
            'valid': pd.Categorical(['valid'], categories=['train', 'valid', 'test', '']),
            'test': pd.Categorical(['test'], categories=['train', 'valid', 'test', ''])
        }

        self.exp_name = exp_name or current_time_str()
        self.exp_dir = os.path.join(self.log_dir, self.exp_name)
        self.early_stop = early_stop

        if self.verbose:
            print(f"Sanity Check: config")
            self.config.show_status()
            print(f"Sanity Check: train_metrics = {self.train_metrics}")
            print(f"Sanity Check: valid_metrics = {self.valid_metrics}")
            print(f"Sanity Check: test_metrics = {self.test_metrics}")
            print(f"Sanity Check: save_metric = '{self.save_metric}'")
            print(f"Sanity Check: save_check_op = "
                  f"current {self.save_metric} '{self.save_check_op}' best {self.save_metric}")
            print(f"Sanity Check: log_dir = {self.log_dir}")
            print(f"Sanity Check: only_test = {self.only_test}")
            print(f"Sanity Check: experiment_name = {self.exp_name}")

    def _check_metric(self, metric_list):
        for metric in metric_list:
            if metric not in self.model_wrapper.metric2func:
                raise NotImplementedError(f"{metric}_func don't exist!")

    @staticmethod
    def metric_result_detach(metric_result: Dict[str, Any]):
        for key, value in metric_result.items():
            if isinstance(value, torch.Tensor):
                metric_result[key] = detach_if_requires_grad(value).item()

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
            metric_result['iter'] = idx
            with torch.set_grad_enabled(backward_propagate):
                out = self.model_wrapper(data['x'])
            for metric_name in metrics:
                if metric_name not in metric_result:
                    self.model_wrapper.metric2func[metric_name](out, data['y'], metric_result)
            if backward_propagate:
                self.optimizer.zero_grad()
                metric_result['loss'].backward()
                self.optimizer.step()
            self.metric_result_detach(metric_result)
            if timeit:
                end_time = time.time()
                metric_result['time_cost'] = end_time - start_time
                start_time = end_time
            if self.verbose:
                p_bar.set_postfix(metric_result)
                p_bar.update(metric_result['num_samples'])
            iter_details.append(metric_result.copy())
            metric_result.clear()
        iter_details = pd.DataFrame(iter_details)

        iter_summary = self.model_wrapper.collect(iter_details)
        if self.verbose:
            p_bar.set_postfix(iter_summary)
            p_bar.close()
        iter_summary = pd.DataFrame([iter_summary])
        iter_summary['epoch'] = iter_details['epoch'] = epoch
        iter_summary['stage'] = self.stage2Categorical[stage]
        iter_details['stage'] = self.stage2Categorical[stage].repeat(len(iter_details))

        iter_details = self.reindex(stage + 'iter_details', iter_details)
        iter_summary = self.reindex(stage + 'iter_summary', iter_summary)

        return iter_summary, iter_details

    def reindex(self, name: str, df: pd.DataFrame):
        need_cache = (name not in self.order_cache)
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
        torch.cuda.empty_cache()
        return self.get_summary_from_one_epoch(epoch, num_epochs, stage, eval_summary_table)

    def save_model(self, best_metric, best_epoch):
        model_path = os.path.join(self.saved_model_dir, 'best_model.pt')
        check_point = {
            'epoch': best_epoch,
            'metric_name': self.save_metric,
            'metric': best_metric,
            'state_dict': self.model_wrapper.model.state_dict(),
        }
        torch.save(check_point, model_path)
        print(f"{current_time_str()} model saved to {model_path}")
        pass

    def run(self):
        num_epochs = self.config['num_epochs']
        num_train = num_epochs
        num_valid = num_epochs // self.config['eval_interval'] if 'eval_interval' in self.config else 0
        time_estimator = ExpTimeEstimator(num_train, num_valid)
        train_summary_table = valid_summary_table = test_summary_table = pd.DataFrame()
        if not self.only_test:
            best_metric, early_stop = None, False
            for epoch in range(1, num_epochs + 1):
                train_iter_summary = valid_iter_summary = None
                # train and save to csv
                train_summary_table, train_iter_summary, train_iter_details = \
                    self.train(epoch, num_epochs, train_summary_table)
                time_estimator.update_train(train_iter_summary.iloc[0]['epoch_time'])
                train_iter_details.to_csv(
                    os.path.join(self.tmp_dir, 'epoch_details', f'epoch{epoch:>0{len(str(num_epochs))}d}_train.csv'),
                    index=False)

                # valid and save to csv
                need_eval = ('eval_interval' in self.config and epoch % self.config['eval_interval'] == 0)
                if need_eval:
                    valid_summary_table, valid_iter_summary, valid_iter_details = \
                        self.eval(epoch, num_epochs, valid_summary_table, 'valid')
                    time_estimator.update_valid(valid_iter_summary.iloc[0]['epoch_time'])
                    valid_iter_details.to_csv(
                        os.path.join(self.tmp_dir, 'epoch_details',
                                     f'epoch{epoch:>0{len(str(num_epochs))}d}_valid.csv'), index=False)

                    # check is best
                    current_save_metric = valid_iter_summary.iloc[0][self.save_metric]
                    is_best = (best_metric is None) or self.ops[self.save_check_op](current_save_metric, best_metric)
                    if is_best:
                        best_metric, best_epoch = current_save_metric, epoch
                        self.save_model(best_metric, best_epoch)
                        early_stop = True
                table = pd.concat([train_iter_summary, valid_iter_summary]).fillna("")
                print(tabulate(table, headers='keys', tablefmt='psql', numalign='center', stralign='center',
                               showindex=False))
                time_estimator.print_status()
                if self.early_stop and early_stop:
                    break
        test_summary_table, test_iter_summary, test_iter_details = \
            self.eval(num_epochs, num_epochs, test_summary_table, 'test')
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


from time import strftime, gmtime


class ExpTimeEstimator:
    def __init__(self, num_train, num_valid):
        self.train_time_estimator = EMATimeEstimator(num_train)
        self.valid_time_estimator = EMATimeEstimator(num_valid)

    def update_train(self, x):
        self.train_time_estimator(x)

    def update_valid(self, x):
        self.valid_time_estimator(x)

    @property
    def elapsed_time(self):
        return self.train_time_estimator.elapsed_time + self.valid_time_estimator.elapsed_time

    @property
    def x(self):
        return self.train_time_estimator.x + self.valid_time_estimator.x

    @property
    def remaining_time(self):
        return self.train_time_estimator.remaining_time + self.valid_time_estimator.remaining_time

    @property
    def total_time(self):
        return self.train_time_estimator.total_time + self.valid_time_estimator.total_time

    def print_status(self):
        expected_end_time = datetime.datetime.now() + datetime.timedelta(seconds=self.remaining_time)
        print(
            f"Time: {format_time(self.x)} -> {format_time(self.elapsed_time)}/{format_time(self.total_time)}"
            f"{'+?' if self.valid_time_estimator.elapsed_calls == 0 else ''}"
            f" | Expected end Time: "
            f"{expected_end_time.strftime('%Y-%m-%d %H:%M:%S')}"
            f"{'+?' if self.valid_time_estimator.elapsed_calls == 0 else ''}")
