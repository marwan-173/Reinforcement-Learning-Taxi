from experiments.reward_comparison import main as reward_main
from experiments.parameter_sweeps import main as sweep_main
from experiments.ablation_study import main as ablation_main
from experiments.plot_results import main as plot_main
from experiments.report_plots import main as report_plot_main
import time

def main():
    start_time = time.time()
    print("Running reward comparison...")
    reward_main()

    print("\nRunning parameter sweeps...")
    sweep_main()

    print("\nRunning ablation study...")
    ablation_main()

    print("\nGenerating plots...")
    plot_main()

    print("\nGenerating report plots...")
    report_plot_main()

    elapsed = time.time() - start_time

    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    print(f"\nAll experiments completed in: {hours:02}:{minutes:02}:{seconds:02}")

if __name__ == "__main__":
    main()