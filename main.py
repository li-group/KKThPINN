import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils import data

from models import NN, NNOPT
from utils import load_data, get_scaledABb, Data_cstr, PINNLoss, Data_plant, Data_distillation
import argparse
import time
import copy

device = 'cpu'


def parse_arguments():
    parser = argparse.ArgumentParser(description='Parse global variables for the script.')

    # 3 for cstr, 4 for plant, 5 for distillation
    parser.add_argument('--input_dim', type=int, default=3, help='Input dimension')
    parser.add_argument('--hidden_dim', type=int, default=32, help='Hidden dimension')
    parser.add_argument('--hidden_num', type=int, default=1, help='Number of hidden layers')
    # 3 for cstr, 5 for plant, 10 for distillation
    parser.add_argument('--z0_dim', type=int, default=3, help='Dimension of z0')

    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--eta', type=float, default=1, help='Hyperparameter for soft constraints')

    parser.add_argument('--dataset_type', type=str, help='choose from cstr, plant, distillation')
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='validation ratio')
    parser.add_argument('--job', type=str, help='choose from train, repeat')

    args = parser.parse_args()
    return args


def test(model, test_loader, loss_func):
    model.eval()
    test_loss = 0
    test_violation = 0
    expanded_b = b.unsqueeze(1)

    with torch.no_grad():
        for batch_idx, (X, Y) in enumerate(test_loader):
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            pred_diff = torch.mm(A, X.T.float()) + \
                        torch.mm(B, pred.T.float()) - \
                        expanded_b.repeat(1, X.T.shape[1])
            if isinstance(loss_func, PINNLoss):
                test_loss += nn.MSELoss()(pred, Y).item()  # return and record the MSE loss for comparison
            elif isinstance(loss_func, nn.MSELoss):
                test_loss += loss_func(pred, Y).item()
            else:
                raise ValueError('Loss function not supported!')
            test_violation += torch.abs(pred_diff.view(-1)).mean()
    test_loss /= len(test_loader.dataset)  # Test set Average loss
    test_violation /= len(test_loader.dataset)  # Test set Average violation
    return test_loss, test_violation


def test_spec(model, test_loader, loss_func, constrained_indexes, unconstrained_indexes):
    model.eval()
    constrained_loss = 0
    unconstrained_loss = 0

    with torch.no_grad():
        for batch_idx, (X, Y) in enumerate(test_loader):
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            for constrained_index in constrained_indexes:
                if isinstance(loss_func, PINNLoss):
                    constrained_loss += nn.MSELoss()(pred[:, constrained_index], Y[:, constrained_index]).item()  # return and record the MSE loss for comparison
                elif isinstance(loss_func, nn.MSELoss):
                    constrained_loss += loss_func(pred[:, constrained_index], Y[:, constrained_index]).item()
                else:
                    raise ValueError('Loss function not supported!')

            for unconstrained_index in unconstrained_indexes:
                if isinstance(loss_func, PINNLoss):
                    unconstrained_loss += nn.MSELoss()(pred[:, unconstrained_index],
                                                       Y[:, unconstrained_index]).item()
                elif isinstance(loss_func, nn.MSELoss):
                    unconstrained_loss += loss_func(pred[:, unconstrained_index], Y[:, unconstrained_index]).item()
                else:
                    raise ValueError('Loss function not supported!')

    constrained_loss /= len(test_loader.dataset) * len(constrained_indexes)
    if len(unconstrained_indexes) == 0:
        return constrained_loss, 0
    else:
        unconstrained_loss /= len(test_loader.dataset) * len(unconstrained_indexes)
    return constrained_loss, unconstrained_loss


def test_post(model, test_loader, loss_func, A, B, b, constrained_indexes, unconstrained_indexes):
    model.eval()
    test_loss = 0
    test_violation = 0
    constrained_loss = 0
    unconstrained_loss = 0
    expanded_b = b.unsqueeze(1)

    BBB_ = torch.mm(B.t(),
                    torch.inverse(
                        torch.mm(B, B.t())
                    ))
    Astar = - torch.mm(BBB_, A)
    Bstar = torch.eye(B.shape[1]).to(device) - torch.mm(BBB_, B)
    bstar = torch.matmul(BBB_, b)

    with torch.no_grad():
        for batch_idx, (X, Y) in enumerate(test_loader):
            X, Y = X.to(device), Y.to(device)
            pred = model(X)

            e = torch.ones((X.shape[0], 1)).to(device)
            # bstar is 1d tensor (q, ), need to transform it into 2d tensor (q, 1) in the following expression
            pred = torch.mm(X, Astar.T) + torch.mm(pred, Bstar.T) + torch.mm(e, bstar.unsqueeze(1).T)

            pred_diff = torch.mm(A, X.T.float()) + \
                        torch.mm(B, pred.T.float()) - \
                        expanded_b.repeat(1, X.T.shape[1])

            for constrained_index in constrained_indexes:
                if isinstance(loss_func, PINNLoss):
                    constrained_loss += nn.MSELoss()(pred[:, constrained_index], Y[:, constrained_index]).item()  # return and record the MSE loss for comparison
                elif isinstance(loss_func, nn.MSELoss):
                    constrained_loss += loss_func(pred[:, constrained_index], Y[:, constrained_index]).item()
                else:
                    raise ValueError('Loss function not supported!')
            for unconstrained_index in unconstrained_indexes:
                if isinstance(loss_func, PINNLoss):
                    unconstrained_loss += nn.MSELoss()(pred[:, unconstrained_index], Y[:, unconstrained_index]).item()
                elif isinstance(loss_func, nn.MSELoss):
                    unconstrained_loss += loss_func(pred[:, unconstrained_index], Y[:, unconstrained_index]).item()
                else:
                    raise ValueError('Loss function not supported!')

            if isinstance(loss_func, PINNLoss):
                test_loss += nn.MSELoss()(pred, Y).item()  # return and record the MSE loss for comparison
            elif isinstance(loss_func, nn.MSELoss):
                test_loss += loss_func(pred, Y).item()
            else:
                raise ValueError('Loss function not supported!')
            test_violation += torch.abs(pred_diff.view(-1)).mean()
    test_loss /= len(test_loader.dataset)  # Test set Average loss
    test_violation /= len(test_loader.dataset)  # Test set Average violation
    constrained_loss /= len(test_loader.dataset) * len(constrained_indexes)
    if len(unconstrained_indexes) == 0:
        return test_loss, test_violation, constrained_loss, 0
    else:
        unconstrained_loss /= len(test_loader.dataset) * len(unconstrained_indexes)
    return test_loss, test_violation, constrained_loss, unconstrained_loss


def train(model, train_loader, val_loader, loss_func):
    expanded_b = b.unsqueeze(1)
    t_total = time.time()
    print('Start Training...')
    min_loss = 123456789
    train_losses = []
    val_losses = []
    train_violations = []
    val_violations = []
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        t = time.time()
        print('-------- Epoch ' + str(epoch + 1) + ' --------')

        model.train()
        train_loss = 0
        train_violation = 0
        for batch_idx, (X, Y) in enumerate(train_loader):
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            pred_diff = torch.mm(A, X.T.float()) + \
                        torch.mm(B, pred.T.float()) - \
                        expanded_b.repeat(1, X.T.shape[1])

            if isinstance(loss_func, PINNLoss):
                loss = loss_func(X, pred, Y)
                loss2record = nn.MSELoss()(pred, Y)
                train_loss += loss2record.item()
            elif isinstance(loss_func, nn.MSELoss):
                loss = loss_func(pred, Y)
                train_loss += loss.item()
            else:
                raise ValueError('Loss function not supported!')

            train_violation += torch.abs(pred_diff.view(-1)).mean()

            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader.dataset)
        train_violation /= len(train_loader.dataset)
        val_loss, val_violation = test(model, val_loader, loss_func)
        train_losses.append(train_loss), train_violations.append(train_violation.detach().item())
        val_losses.append(val_loss), val_violations.append(val_violation.detach().item())

        if np.mean(val_loss) < min_loss:
            model_max = copy.deepcopy(model)
            min_loss = np.mean(val_losses[-1])
            # torch.save(model, path)
        print('epoch: {:05d}'.format(epoch + 1),
              'loss_train: {:.5f}'.format(train_loss),
              'loss_val: {:.5f}'.format(val_loss),
              'violation_train: {:.5f}'.format(train_violation),
              'violation_val: {:.5f}'.format(val_violation),
              'time: {:.4f}s'.format(time.time() - t))
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    return train_losses, val_losses, train_violations, val_violations, model_max


if __name__ == "__main__":
    args = parse_arguments()
    print(args)

    if args.dataset_type == 'cstr':
        dataset_arr, scaler = load_data(args.dataset_path)
        Data_class = Data_cstr
    elif args.dataset_type == 'plant':
        dataset_arr, scaler = load_data(args.dataset_path)
        Data_class = Data_plant
    elif args.dataset_type == 'distillation':
        dataset_arr, scaler = load_data(args.dataset_path)
        Data_class = Data_distillation
    else:
        raise ValueError('Dataset not supported!')

    params = {'batch_size': args.batch_size,
              'shuffle': True}

    dataset = Data_class(dataset_arr)
    dataset.resplit_data(args.val_ratio)

    A, B, b = get_scaledABb(dataset.A, dataset.B, dataset.b, scaler)
    print(A.dtype, B.dtype, b.dtype)
    print((torch.mm(A, dataset.X.T.float()) + torch.mm(B, dataset.Y.T.float()))[:, :5])
    print(b)

    train_loader = data.DataLoader(dataset.train_set, **params)
    val_loader = data.DataLoader(dataset.val_set, **params)
    test_loader = data.DataLoader(dataset.test_set, **params)

    if args.job == 'train':
        NN_model = NN(args.input_dim, args.hidden_dim, args.hidden_num, args.z0_dim).to(device)
        PINN_model = NN(args.input_dim, args.hidden_dim, args.hidden_num, args.z0_dim).to(device)
        NNOPT_model = NNOPT(args.input_dim, args.hidden_dim, args.hidden_num, args.z0_dim, A, B, b).to(device)

        NN_train_losses, NN_val_losses, \
            NN_train_violations, NN_val_violations, \
            NN_model_max = train(NN_model, train_loader, val_loader, nn.MSELoss())

        PINN_train_losses, PINN_val_losses, \
            PINN_train_violations, PINN_val_violations, \
            PINN_model_max = train(PINN_model, train_loader, val_loader, PINNLoss(A, B, b, args.eta))

        NNOPT_train_losses, NNOPT_val_losses, \
            NNOPT_train_violations, NNOPT_val_violations, \
            NNOPT_model_max = train(NNOPT_model, train_loader, val_loader, nn.MSELoss())

        # SAVE RESULTS
        np.save(f'results/temp_{args.dataset_type}_NN_train_losses.npy', NN_train_losses)
        np.save(f'results/temp_{args.dataset_type}_NN_val_losses.npy', NN_val_losses)
        np.save(f'results/temp_{args.dataset_type}_NN_train_violations.npy', NN_train_violations)
        np.save(f'results/temp_{args.dataset_type}_NN_val_violations.npy', NN_val_violations)

        np.save(f'results/temp_{args.dataset_type}_PINN_train_losses.npy', PINN_train_losses)
        np.save(f'results/temp_{args.dataset_type}_PINN_val_losses.npy', PINN_val_losses)
        np.save(f'results/temp_{args.dataset_type}_PINN_train_violations.npy', PINN_train_violations)
        np.save(f'results/temp_{args.dataset_type}_PINN_val_violations.npy', PINN_val_violations)

        np.save(f'results/temp_{args.dataset_type}_NNOPT_train_losses.npy', NNOPT_train_losses)
        np.save(f'results/temp_{args.dataset_type}_NNOPT_val_losses.npy', NNOPT_val_losses)
        np.save(f'results/temp_{args.dataset_type}_NNOPT_train_violations.npy', NNOPT_train_violations)
        np.save(f'results/temp_{args.dataset_type}_NNOPT_val_violations.npy', NNOPT_val_violations)

    elif args.job == 'repeat':
        NN_model = NN(args.input_dim, args.hidden_dim, args.hidden_num, args.z0_dim).to(device)
        PINN_model = NN(args.input_dim, args.hidden_dim, args.hidden_num, args.z0_dim).to(device)
        NNOPT_model = NNOPT(args.input_dim, args.hidden_dim, args.hidden_num, args.z0_dim, A, B, b).to(device)
        for run in range(10):
            print('-------- Run ' + str(run + 1) + ' --------')
            NN_model.reset_parameters()
            PINN_model.reset_parameters()
            NNOPT_model.reset_parameters()

            NN_train_losses, NN_val_losses, \
                NN_train_violations, NN_val_violations, \
                NN_model_max = train(NN_model, train_loader, val_loader, nn.MSELoss())

            PINN_train_losses, PINN_val_losses, \
                PINN_train_violations, PINN_val_violations, \
                PINN_model_max = train(PINN_model, train_loader, val_loader, PINNLoss(A, B, b, args.eta))

            NNOPT_train_losses, NNOPT_val_losses, \
                NNOPT_train_violations, NNOPT_val_violations, \
                NNOPT_model_max = train(NNOPT_model, train_loader, val_loader, nn.MSELoss())

            # SAVE RESULTS
            np.save(f'results/{args.dataset_type}_{args.val_ratio}_NN_train_losses_run{run}.npy', NN_train_losses)
            np.save(f'results/{args.dataset_type}_{args.val_ratio}_NN_val_losses_run{run}.npy', NN_val_losses)
            np.save(f'results/{args.dataset_type}_{args.val_ratio}_NN_train_violations_run{run}.npy', NN_train_violations)
            np.save(f'results/{args.dataset_type}_{args.val_ratio}_NN_val_violations_run{run}.npy', NN_val_violations)

            np.save(f'results/{args.dataset_type}_{args.val_ratio}_PINN_train_losses_run{run}.npy', PINN_train_losses)
            np.save(f'results/{args.dataset_type}_{args.val_ratio}_PINN_val_losses_run{run}.npy', PINN_val_losses)
            np.save(f'results/{args.dataset_type}_{args.val_ratio}_PINN_train_violations_run{run}.npy', PINN_train_violations)
            np.save(f'results/{args.dataset_type}_{args.val_ratio}_PINN_val_violations_run{run}.npy', PINN_val_violations)

            np.save(f'results/{args.dataset_type}_{args.val_ratio}_NNOPT_train_losses_run{run}.npy', NNOPT_train_losses)
            np.save(f'results/{args.dataset_type}_{args.val_ratio}_NNOPT_val_losses_run{run}.npy', NNOPT_val_losses)
            np.save(f'results/{args.dataset_type}_{args.val_ratio}_NNOPT_train_violations_run{run}.npy', NNOPT_train_violations)
            np.save(f'results/{args.dataset_type}_{args.val_ratio}_NNOPT_val_violations_run{run}.npy', NNOPT_val_violations)

            # apply the best model to test set
            NN_test_loss, NN_test_violation = test(NN_model_max, test_loader, nn.MSELoss())
            PINN_test_loss, PINN_test_violation = test(PINN_model_max, test_loader, PINNLoss(A, B, b, args.eta))
            NNOPT_test_loss, NNOPT_test_violation = test(NNOPT_model_max, test_loader, nn.MSELoss())
            # also store the constrained loss and unconstrained loss
            NN_constrained_loss, NN_unconstrained_loss = test_spec(NN_model_max, test_loader, nn.MSELoss(),
                                                                   dataset.constrained_indexes, dataset.unconstrained_indexes)
            PINN_constrained_loss, PINN_unconstrained_loss = test_spec(PINN_model_max, test_loader, PINNLoss(A, B, b, args.eta),
                                                                       dataset.constrained_indexes, dataset.unconstrained_indexes)
            NNOPT_constrained_loss, NNOPT_unconstrained_loss = test_spec(NNOPT_model_max, test_loader, nn.MSELoss(),
                                                                         dataset.constrained_indexes, dataset.unconstrained_indexes)
            # also store the post-projected loss
            NNPost_test_loss, NNPost_test_violation, NNPost_constrained_loss, NNPost_unconstrained_loss = \
                test_post(NN_model_max, test_loader, nn.MSELoss(), A, B, b, dataset.constrained_indexes, dataset.unconstrained_indexes)

            # SAVE RESULTS
            np.save(f'results/{args.dataset_type}_{args.val_ratio}_NN_test_loss_run{run}.npy', NN_test_loss)
            np.save(f'results/{args.dataset_type}_{args.val_ratio}_NN_test_violation_run{run}.npy', NN_test_violation)
            np.save(f'results/{args.dataset_type}_{args.val_ratio}_NN_constrained_loss_run{run}.npy', NN_constrained_loss)
            np.save(f'results/{args.dataset_type}_{args.val_ratio}_NN_unconstrained_loss_run{run}.npy', NN_unconstrained_loss)

            np.save(f'results/{args.dataset_type}_{args.val_ratio}_PINN_test_loss_run{run}.npy', PINN_test_loss)
            np.save(f'results/{args.dataset_type}_{args.val_ratio}_PINN_test_violation_run{run}.npy', PINN_test_violation)
            np.save(f'results/{args.dataset_type}_{args.val_ratio}_PINN_constrained_loss_run{run}.npy', PINN_constrained_loss)
            np.save(f'results/{args.dataset_type}_{args.val_ratio}_PINN_unconstrained_loss_run{run}.npy', PINN_unconstrained_loss)

            np.save(f'results/{args.dataset_type}_{args.val_ratio}_NNOPT_test_loss_run{run}.npy', NNOPT_test_loss)
            np.save(f'results/{args.dataset_type}_{args.val_ratio}_NNOPT_test_violation_run{run}.npy', NNOPT_test_violation)
            np.save(f'results/{args.dataset_type}_{args.val_ratio}_NNOPT_constrained_loss_run{run}.npy', NNOPT_constrained_loss)
            np.save(f'results/{args.dataset_type}_{args.val_ratio}_NNOPT_unconstrained_loss_run{run}.npy', NNOPT_unconstrained_loss)

            np.save(f'results/{args.dataset_type}_{args.val_ratio}_NNPost_test_loss_run{run}.npy', NNPost_test_loss)
            np.save(f'results/{args.dataset_type}_{args.val_ratio}_NNPost_test_violation_run{run}.npy', NNPost_test_violation)
            np.save(f'results/{args.dataset_type}_{args.val_ratio}_NNPost_constrained_loss_run{run}.npy', NNPost_constrained_loss)
            np.save(f'results/{args.dataset_type}_{args.val_ratio}_NNPost_unconstrained_loss_run{run}.npy', NNPost_unconstrained_loss)

    else:
        raise ValueError('Job not supported!')







