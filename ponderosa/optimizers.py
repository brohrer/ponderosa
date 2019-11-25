import datetime as dt
import os
import random as rnd
import toolbox as tb


class HPOptimizer(object):
    def __init__(
        self,
        report_dir=os.path.join(
            "reports",
            "hpo_" + dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        ),
        report_filename="hpo_results.csv",
        report_plot_filename="hpo_results.png",
    ):
        self.report_dir = report_dir
        self.report_filename = os.path.join(
            self.report_dir, report_filename)
        self.report_plot_filename = os.path.join(
            self.report_dir, report_plot_filename)

        # Ensure that report directory exists
        try:
            os.mkdir(self.report_dir)
        except Exception:
            pass

    def optimize(self, initialize, conditions):
        print(
            "\nThis is going to take a while.\n"
            + "    You can check on the best-so-far solution at any time in\n"
            + "    " + self.report_plot_filename + "\n"
            + "    The full results log is maintained in\n"
            + "    " + self.report_filename + "\n\n"
        )

        best_error = 1e10
        best_condition = None
        condition_history = []
        for condition in self.condition_generator(conditions):
            print("    Evaluating condition", condition)
            autoencoder, training_set, tuning_set = initialize(**condition)
            error, info = autoencoder.evaluate_hyperparameters(
                training_set, tuning_set)
            condition["error"] = error
            condition["info"] = info
            condition_history.append(condition)
            tb.results_dict_list_to_csv(
                condition_history, self.report_filename)

            if error < best_error:
                best_error = error
                best_condition = condition

            results_so_far = tb.results_csv_to_dict_list(self.report_filename)
            tb.progress_report(results_so_far, self.report_plot_filename)
        return best_error, best_condition

    def condition_generator(self, conditions):
        pass


class Random(HPOptimizer):
    def __init__(self, n_iter=1e10):
        super().__init__()
        print(self.report_filename)
        self.n_iter = n_iter

    def condition_generator(self, unexpanded_conditions):
        conditions = tb.grid_expand(unexpanded_conditions)
        rnd.shuffle(conditions)
        if self.n_iter < len(conditions):
            conditions = conditions[:self.n_iter]

        for condition in conditions:
            self.n_iter += 1
            yield condition


Grid = Random
