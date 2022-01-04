import math
import os, time
import shutil
import torch
from metrics import Result


class logger:
    def __init__(self, args, prepare=True):
        self.args = args
        self.best_result = Result()
        self.best_result.set_to_worst()


    def conditional_print(self, split, i, epoch, lr, n_set, blk_avg_meter,
                          avg_meter):
        if (i + 1) % self.args.print_freq == 0:
            avg = avg_meter.average()
            blk_avg = blk_avg_meter.average()
            print(
                '{split} Epoch: {0} [{1}/{2}]\tlr={lr} '
                't_Data={blk_avg.data_time:.3f}({average.data_time:.3f}) '
                't_GPU={blk_avg.gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                'RMSE={blk_avg.rmse:.2f}({average.rmse:.2f}) '
                'MAE={blk_avg.mae:.2f}({average.mae:.2f}) '
                'iRMSE={blk_avg.irmse:.2f}({average.irmse:.2f}) '
                'iMAE={blk_avg.imae:.2f}({average.imae:.2f})\n\t'
                'silog={blk_avg.silog:.2f}({average.silog:.2f}) '
                'squared_rel={blk_avg.squared_rel:.2f}({average.squared_rel:.2f}) '
                'Delta1={blk_avg.delta1:.3f}({average.delta1:.3f}) '
                'REL={blk_avg.absrel:.3f}({average.absrel:.3f})\n\t'
                'Lg10={blk_avg.lg10:.3f}({average.lg10:.3f}) '
                'Photometric={blk_avg.photometric:.3f}({average.photometric:.3f}) '
                .format(epoch,
                        i + 1,
                        n_set,
                        lr=lr,
                        blk_avg=blk_avg,
                        average=avg,
                        split=split.capitalize()))
            blk_avg_meter.reset(False)



    def get_ranking_error(self, result):
        return getattr(result, self.args.rank_metric)

    def get_ranking_error(self, result):
        return getattr(result, self.args.rank_metric)

    def rank_conditional_save_best(self, mode, result, epoch):
        error = self.get_ranking_error(result)
        best_error = self.get_ranking_error(self.best_result)
        is_best = error < best_error
        if is_best and mode == "val":
            self.old_best_result = self.best_result
            self.best_result = result

        return is_best


    def conditional_summarize(self, mode, avg, is_best):
        print("\n*\nSummary of ", mode, "round")
        print(''
              'RMSE={average.rmse:.3f}\n'
              'MAE={average.mae:.3f}\n'
              'Photo={average.photometric:.3f}\n'
              'iRMSE={average.irmse:.3f}\n'
              'iMAE={average.imae:.3f}\n'
              'squared_rel={average.squared_rel}\n'
              'silog={average.silog}\n'
              'Delta1={average.delta1:.3f}\n'
              'REL={average.absrel:.3f}\n'
              'Lg10={average.lg10:.3f}\n'
              't_GPU={time:.3f}'.format(average=avg, time=avg.gpu_time))
        if is_best and mode == "val":
            print("New best model by %s (was %.3f)" %
                  (self.args.rank_metric,
                   self.get_ranking_error(self.old_best_result)))
        elif mode == "val":
            print("(best %s is %.3f)" %
                  (self.args.rank_metric,
                   self.get_ranking_error(self.best_result)))
        print("*\n")






