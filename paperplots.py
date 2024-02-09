import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, default='plant', help='dataset type')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='validation ratio')
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--length', type=int, default=8.27)
    parser.add_argument('--width', type=int, default=3.7)
    parser.add_argument('--rmse_lim', type=float, default=5e-3)
    parser.add_argument('--font_size', type=int, default=8)
    parser.add_argument('--linewidth', type=int, default=2)
    parser.add_argument('--barwidth', type=float, default=0.02)
    parser.add_argument('--job', type=str, default='learning_curves', help='job type')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    print(args)

    #Always use the LaTeX extension to render figures by including plt.rcParams['text.usetex'] = True' at the top of your script
    # plt.rcParams['text.usetex'] = True

    # if args.job == 'learning_curves':
    #     print(f'Plotting learning curves for {args.dataset_type} dataset')
    #     for run in range(10):
    #         fig, axs = plt.subplots(1, 2, dpi=args.dpi)
    #         plt.rcParams['text.usetex'] = True
    #
    #         for model in ['NN', 'PINN', 'NNOPT']:
    #             for type in ['train', 'val']:
    #                 losses = np.load(f'results/{args.dataset_type}_{args.val_ratio}_{model}_{type}_losses_run{run}.npy')
    #                 violations = np.load(f'results/{args.dataset_type}_{args.val_ratio}_{model}_{type}_violations_run{run}.npy')
    #
    #                 label_name = f'{"KKT-hPINN" if model == "NNOPT" else model} {type}'
    #                 axs[0].plot(np.sqrt(losses), label=label_name, linewidth=args.linewidth)
    #                 axs[1].plot(violations, label=label_name, linewidth=args.linewidth)
    #
    #                 axs[0].grid(True, linestyle='-', axis='y')
    #                 axs[0].set_xlabel('Epochs', fontdict={'size': args.font_size})
    #                 axs[0].set_ylabel('RMSE', fontdict={'size': args.font_size})
    #                 axs[0].set_xlim(0, 1000)
    #                 axs[0].set_ylim(0, args.rmse_lim)
    #                 axs[0].legend(prop={'size': args.font_size})
    #                 axs[0].tick_params(axis="x", labelsize=args.font_size)
    #                 axs[0].tick_params(axis="y", labelsize=args.font_size)
    #
    #                 axs[1].set_yscale('log')
    #                 axs[1].grid(True, linestyle='-', axis='y')
    #                 axs[1].set_xlabel('Epochs', fontdict={'size': args.font_size})
    #                 axs[1].set_ylabel('Violation', fontdict={'size': args.font_size})
    #                 axs[1].set_xlim(0, 2000)
    #                 axs[1].set_ylim(1e-8, 1e-1)
    #                 axs[1].legend(prop={'size': args.font_size})
    #                 axs[1].tick_params(axis="x", labelsize=args.font_size)
    #                 axs[1].tick_params(axis="y", labelsize=args.font_size)
    #
    #         plt.tight_layout()
    #         plt.savefig(f'{args.dataset_type}_figs/{args.dataset_type}_{args.val_ratio}_learning_curves_run{run}.pdf',
    #                     format='pdf', dpi=args.dpi)
    #         plt.close()

    if args.job == 'learning_curves':
        colors = {'NN train': 'orange',
                  'NN val': 'darkgreen',
                  'PINN train': 'blue',
                  'PINN val': 'darkviolet',
                  'KKT-hPINN train': 'red',
                  'KKT-hPINN val': 'brown'}

        fig, axs = plt.subplots(1, 2, figsize=(7, 3), dpi=args.dpi)
        plt.rcParams['text.usetex'] = True
        for model in ['NN', 'PINN', 'NNOPT']:
            for type in ['train', 'val']:
                all_losses = []
                all_violations = []
                for run in range(10):
                    losses = np.load(f'results/{args.dataset_type}_{args.val_ratio}_{model}_{type}_losses_run{run}.npy')
                    violations = np.load(
                        f'results/{args.dataset_type}_{args.val_ratio}_{model}_{type}_violations_run{run}.npy')
                    all_losses.append(np.sqrt(losses))  #######
                    all_violations.append(violations)

                iterations = np.arange(len(all_losses[0]))
                mean_losses = np.mean(all_losses, axis=0)
                std_losses = np.std(all_losses, axis=0)

                mean_violations = np.mean(all_violations, axis=0)
                std_violations = np.std(all_violations, axis=0)

                label_name = f'{"KKT-hPINN" if model == "NNOPT" else model} {type}'

                axs[0].plot(iterations, mean_losses, label=label_name, color=colors[label_name])
                axs[1].plot(mean_violations, label=label_name, color=colors[label_name])
                # if type == 'val':
                #     axs[0].fill_between(iterations, mean_losses - std_losses, mean_losses + std_losses, alpha=0.4,
                #                         linewidth=0, color=colors[label_name])
                #     axs[1].fill_between(iterations, mean_violations - std_violations, mean_violations + std_violations,
                #                         alpha=0.4, linewidth=0, color=colors[label_name])

        axs[0].grid(True, alpha=0.4, linestyle='-', axis='y')
        axs[0].set_xlabel('Epochs', fontdict={'size': args.font_size})
        axs[0].set_ylabel('RMSE', fontdict={'size': args.font_size})
        axs[0].set_ylim(0, args.rmse_lim)
        axs[0].legend(frameon=False, prop={'size': args.font_size})
        axs[0].tick_params(axis="x", labelsize=args.font_size)
        axs[0].tick_params(axis="y", labelsize=args.font_size)

        axs[1].set_yscale('log')
        axs[1].grid(True, alpha=0.4, linestyle='-', axis='y')
        axs[1].set_xlabel('Epochs', fontdict={'size': args.font_size})
        axs[1].set_ylabel('Violation', fontdict={'size': args.font_size})
        axs[1].set_ylim(1e-8, 1e-1)
        # axs[1].legend(frameon=False, prop={'size': args.font_size})
        axs[1].tick_params(axis="x", labelsize=args.font_size)
        axs[1].tick_params(axis="y", labelsize=args.font_size)

        plt.tight_layout()
        plt.savefig(f'{args.dataset_type}_figs/{args.dataset_type}_{args.val_ratio}_BETTER_learning_curves.pdf',
                    format='pdf', dpi=args.dpi)
        plt.close()

    elif args.job == 'table':
        mean_df = pd.DataFrame(index=['NN', 'NNPost', 'PINN', 'NNOPT'],
                               columns=['test', 'constrained', 'unconstrained'])
        std_df = pd.DataFrame(index=['NN', 'NNPost', 'PINN', 'NNOPT'],
                              columns=['test', 'constrained', 'unconstrained'])

        for model in ['NN', 'NNPost', 'PINN', 'NNOPT']:
            for type in ['test', 'constrained', 'unconstrained']:
                losses = []
                for run in range(10):
                    loss = np.load(f'results/{args.dataset_type}_{args.val_ratio}_{model}_{type}_loss_run{run}.npy')
                    losses.append(loss)
                losses = np.sqrt(losses)
                mean_loss = np.mean(losses)
                std_loss = np.std(losses)
                mean_df.loc[model, type] = mean_loss
                std_df.loc[model, type] = std_loss

        mean_df.to_csv(f'{args.dataset_type}_figs/{args.dataset_type}_{args.val_ratio}_mean_losses.csv')
        std_df.to_csv(f'{args.dataset_type}_figs/{args.dataset_type}_{args.val_ratio}_std_losses.csv')

    elif args.job == 'bar':
        ratios = [0.2, 0.3, 0.4]  ### choose ratios
        colors = {'NN': '#dee0e2',
                  'NNPost': '#b2b8bc',
                  'KKT-hPINN': '#313638'}
        fig, ax = plt.subplots(1, 1, figsize=(7, 3), dpi=args.dpi)
        plt.rcParams['text.usetex'] = True
        for ratio in ratios:
            for i, model in enumerate(['NN', 'NNPost', 'NNOPT']):
                label_name = f'{"KKT-hPINN" if model == "NNOPT" else model}'
                mean_df = pd.read_csv(f'{args.dataset_type}_figs/{args.dataset_type}_{ratio}_mean_losses.csv')
                std_df = pd.read_csv(f'{args.dataset_type}_figs/{args.dataset_type}_{ratio}_std_losses.csv')
                rmse = mean_df.loc[i if i != 2 else i+1, 'test']

                if model == 'NN':
                    nn_rmse = rmse
                else:
                    improvement = (nn_rmse - rmse) / nn_rmse * 100
                    text_offset = 0.00004
                    ax.text(ratio + i * args.barwidth, rmse + text_offset, f'{improvement:.1f}\%', color='black',
                            fontdict={'size': args.font_size}, ha='center')
                ax.bar(ratio + i * args.barwidth,
                       rmse,
                       # yerr=std_df.loc[i+1, 'test'],
                       width=args.barwidth,
                       label=label_name,
                       color=colors[label_name])
        ax.set_xticks(ratios)
        ax.set_xticklabels(ratios)

        ax.set_ylim(0, 0.005)

        ax.set_xlabel('Unused training sample ratio', fontdict={'size': args.font_size})
        ax.set_ylabel('RMSE', fontdict={'size': args.font_size})

        ax.tick_params(axis="x", labelsize=args.font_size)
        ax.tick_params(axis="y", labelsize=args.font_size)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in ['NN', 'NNPost', 'KKT-hPINN']]
        ax.legend(handles, ['NN', 'NNPost', 'KKT-hPINN'], frameon=False, prop={'size': args.font_size})

        plt.savefig(f'{args.dataset_type}_figs/{args.dataset_type}_BETTER_barchart.pdf',
                    format='pdf', bbox_inches='tight', dpi=args.dpi)
        plt.close()

    else:
        raise NotImplementedError


